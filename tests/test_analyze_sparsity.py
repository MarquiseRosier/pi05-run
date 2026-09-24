from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_analyze_sparsity_writes_overview(tmp_path: Path) -> None:
    capture = tmp_path / "transcoder_capture"
    capture.mkdir()
    events = capture / "events.jsonl"
    rows = []
    for layer in range(3):
        for t in (1.0, 0.5, 0.1):
            token_l0 = [180 + layer * 10 + int(t * 20) + i % 7 for i in range(8)]
            rows.append(
                {
                    "type": "transcoder_latent",
                    "layer": layer,
                    "chunk": 0,
                    "timestep": [t],
                    "shape": [1, 8, 16384],
                    "l0_mean": sum(token_l0) / len(token_l0),
                    "token_l0": token_l0,
                }
            )
    events.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    script = Path(__file__).resolve().parents[1] / "run" / "pi0.5" / "analyze_sparsity.py"
    result = subprocess.run([sys.executable, str(script), "--run", str(tmp_path)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads((capture / "sparsity_summary.json").read_text(encoding="utf-8"))
    assert summary["dictionary_size"] == 16384
    assert 100.0 < summary["mean_l0"] < 400.0
    assert summary["mean_l0_pct"] < 5.0
    assert (capture / "sparsity_overview.png").exists()
    assert (capture / "l0_percent_vs_t_by_layer.png").exists()
