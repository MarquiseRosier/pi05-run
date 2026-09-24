"""On-disk store for the full per-feature deltas of a counterfactual probe run.

The probe reduces each transcoder code to one scalar per feature (max over
batch and action tokens) for every layer and denoise step. Keeping only the
top-K of those per cell, as the CSV does, leaves three holes the write-up then
has to talk around: a feature absent from the placebo's top-K has a *bounded*
rather than known response, a traced circuit node the probe never recorded has
an *unknown* rather than zero response, and the only available null pool is
"features that made some top-K".

At 16384 features x 18 layers x 10 steps the full reduced code is 11.8 MB per
forward pass in float32, and it is mostly zeros, so storing all of it
compressed costs tens of megabytes per run. This module does that.

Layout: one ``.npz`` per (state, noise draw, prompt) block holding the baseline
code ``baseline`` and one ``delta__<condition>__dose<d>`` array per measured
condition, each shaped ``[layers, steps, features]``; ``index.json`` lists the
blocks and the layer order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

INDEX_NAME = "index.json"


def _dose_key(dose: float) -> str:
    return f"{float(dose):g}"


def delta_array_key(condition: str, dose: float) -> str:
    return f"delta__{condition}__dose{_dose_key(dose)}"


@dataclass
class Block:
    state: int
    noise: int
    prompt: str
    file: str
    conditions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DeltaStore:
    root: Path
    layer_names: list[str]
    num_steps: int
    num_features: int
    blocks: list[Block] = field(default_factory=list)

    # ------------------------------------------------------------------ writing

    @classmethod
    def create(cls, root: Path, *, layer_names: list[str], num_steps: int, num_features: int) -> "DeltaStore":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        store = cls(root=root, layer_names=list(layer_names), num_steps=int(num_steps), num_features=int(num_features))
        store._write_index()
        return store

    def _stack(self, by_key: dict[tuple[str, int], np.ndarray]) -> np.ndarray:
        out = np.zeros((len(self.layer_names), self.num_steps, self.num_features), dtype=np.float32)
        seen = 0
        for (name, step), vector in by_key.items():
            if name not in self.layer_names:
                raise KeyError(f"Unknown layer {name!r}; the store was created with {self.layer_names}")
            if not 0 <= int(step) < self.num_steps:
                raise IndexError(f"Denoise step {step} outside [0, {self.num_steps})")
            vector = np.asarray(vector, dtype=np.float32).reshape(-1)
            if vector.shape[0] != self.num_features:
                raise ValueError(f"Expected {self.num_features} features, got {vector.shape[0]} for {name!r}")
            out[self.layer_names.index(name), int(step)] = vector
            seen += 1
        expected = len(self.layer_names) * self.num_steps
        if seen != expected:
            raise ValueError(f"Block has {seen} (layer, step) cells; expected {expected}")
        return out

    def write_block(
        self,
        *,
        state: int,
        noise: int,
        prompt: str,
        baseline_max: dict[tuple[str, int], np.ndarray],
        deltas: dict[tuple[str, float], dict[tuple[str, int], np.ndarray]],
    ) -> Path:
        """Persist one (state, noise draw, prompt) block: its baseline code and every delta."""
        arrays: dict[str, np.ndarray] = {"baseline": self._stack(baseline_max)}
        conditions = []
        for (condition, dose), by_key in deltas.items():
            key = delta_array_key(condition, dose)
            arrays[key] = self._stack(by_key)
            conditions.append({"condition": condition, "dose": float(dose), "key": key})
        file = f"s{int(state)}_n{int(noise)}_{prompt}.npz"
        np.savez_compressed(self.root / file, **arrays)
        self.blocks = [b for b in self.blocks if not (b.state == state and b.noise == noise and b.prompt == prompt)]
        self.blocks.append(Block(state=int(state), noise=int(noise), prompt=str(prompt), file=file, conditions=conditions))
        self._write_index()
        return self.root / file

    def _write_index(self) -> None:
        payload = {
            "format_version": 1,
            "layer_names": self.layer_names,
            "num_steps": self.num_steps,
            "num_features": self.num_features,
            "blocks": [
                {"state": b.state, "noise": b.noise, "prompt": b.prompt, "file": b.file, "conditions": b.conditions}
                for b in sorted(self.blocks, key=lambda b: (b.state, b.noise, b.prompt))
            ],
        }
        (self.root / INDEX_NAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ reading

    @classmethod
    def open(cls, root: Path) -> "DeltaStore":
        root = Path(root)
        index = json.loads((root / INDEX_NAME).read_text())
        store = cls(
            root=root,
            layer_names=list(index["layer_names"]),
            num_steps=int(index["num_steps"]),
            num_features=int(index["num_features"]),
        )
        store.blocks = [Block(**b) for b in index.get("blocks", [])]
        return store

    @staticmethod
    def exists(root: Path) -> bool:
        return (Path(root) / INDEX_NAME).exists()

    def layer_position(self, layer_index: int) -> int | None:
        """Position in the store of the action-expert layer with this numeric index."""
        for position, name in enumerate(self.layer_names):
            if _numeric_layer_index(name) == int(layer_index):
                return position
        return None

    def layer_indices(self) -> list[int]:
        return [_numeric_layer_index(name) for name in self.layer_names]

    def step_for_tau(self, tau: float) -> int:
        """Denoise step whose flow time is ``tau`` (``tau = 1 - step / num_steps``)."""
        step = int(round((1.0 - float(tau)) * self.num_steps))
        return min(max(step, 0), self.num_steps - 1)

    def tau_for_step(self, step: int) -> float:
        return 1.0 - int(step) / self.num_steps

    def select_blocks(self, *, prompt: str | None = None) -> list[Block]:
        return [b for b in self.blocks if prompt is None or b.prompt == prompt]

    def load(self, block: Block) -> dict[str, np.ndarray]:
        with np.load(self.root / block.file) as handle:
            return {key: handle[key] for key in handle.files}

    def iter_deltas(self, *, condition: str, prompt: str | None = None) -> Iterator[tuple[Block, float, np.ndarray]]:
        for block in self.select_blocks(prompt=prompt):
            arrays = self.load(block)
            for entry in block.conditions:
                if entry["condition"] != condition:
                    continue
                yield block, float(entry["dose"]), arrays[entry["key"]]

    def iter_baselines(self, *, prompt: str | None = None) -> Iterator[tuple[Block, np.ndarray]]:
        for block in self.select_blocks(prompt=prompt):
            yield block, self.load(block)["baseline"]

    def mean_abs_delta(self, *, condition: str, prompt: str | None = None) -> tuple[np.ndarray, int]:
        """Mean |delta| over every (block, dose) cell, shaped [layers, steps, features]."""
        total = np.zeros((len(self.layer_names), self.num_steps, self.num_features), dtype=np.float64)
        count = 0
        for _block, _dose, delta in self.iter_deltas(condition=condition, prompt=prompt):
            total += np.abs(delta, dtype=np.float64)
            count += 1
        if count == 0:
            return total.astype(np.float32), 0
        return (total / count).astype(np.float32), count

    def exercised_mask(self, *, condition: str, prompt: str | None = None) -> np.ndarray:
        """Features the scene exercises at each (layer, step).

        A feature is exercised if it is active in some baseline or moves under
        the perturbation in some cell. A feature that is neither has a genuine
        zero response on this scene, which is different from an unknown one,
        and different again from a feature that was active and did not move.
        """
        mask = np.zeros((len(self.layer_names), self.num_steps, self.num_features), dtype=bool)
        for block in self.select_blocks(prompt=prompt):
            arrays = self.load(block)
            mask |= arrays["baseline"] > 0
            for entry in block.conditions:
                if entry["condition"] == condition:
                    mask |= arrays[entry["key"]] != 0
        return mask

    def conditions(self) -> list[str]:
        return sorted({entry["condition"] for block in self.blocks for entry in block.conditions})

    def prompts(self) -> list[str]:
        return sorted({block.prompt for block in self.blocks})


def _numeric_layer_index(name: str) -> int:
    digits = [part for part in str(name).split(".") if part.isdigit()]
    return int(digits[-1]) if digits else -1


def sort_layer_names(names: set[str] | list[str]) -> list[str]:
    return sorted(set(names), key=lambda n: (_numeric_layer_index(n), n))
