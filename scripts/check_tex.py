#!/usr/bin/env python
"""Compile the paper section and fail on anything a reviewer's PDF would show.

Regex checks on LaTeX source catch brace and environment imbalance and little
else. Only the engine knows whether a macro is defined, whether a tabular row
has the right number of cells, whether every \\ref resolves, and whether a
table overruns the measure. This runs Tectonic on the standalone wrapper at the
ICLR text width and fails on:

  * any TeX error (the engine's own exit status);
  * unresolved references in the final pass (``There were undefined
    references``, or ``??`` in the extracted text);
  * an ``Overfull \\hbox`` wider than --max-overfull points, which is a table or
    line running into the margin.

Fonts are not checked: the wrapper deliberately omits the template's Times
package, which is not available to the XeTeX-based engine.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1] / "docs" / "paper"
WRAPPER = "counterfactual_probe_standalone.tex"


def compile_tex(paper_dir: Path, wrapper: str, out_dir: Path) -> tuple[int, str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["tectonic", "-o", str(out_dir), "--keep-logs", wrapper],
        cwd=paper_dir, capture_output=True, text=True,
    )
    log_path = out_dir / (Path(wrapper).stem + ".log")
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    return proc.returncode, proc.stdout + proc.stderr + "\n" + log, out_dir / (Path(wrapper).stem + ".pdf")


def overfull_boxes(log: str) -> list[tuple[float, str]]:
    seen: dict[str, float] = {}
    for m in re.finditer(r"Overfull \\hbox \(([\d.]+)pt too wide\) in paragraph at lines (\d+--\d+)", log):
        seen[m.group(2)] = max(seen.get(m.group(2), 0.0), float(m.group(1)))
    return sorted(((pts, where) for where, pts in seen.items()), reverse=True)


def pdf_text(pdf: Path) -> str | None:
    if shutil.which("pdftotext") is None or not pdf.exists():
        return None
    return subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout


def check(paper_dir: Path = PAPER_DIR, wrapper: str = WRAPPER, *, max_overfull: float = 2.0,
          out_dir: Path | None = None) -> list[str]:
    """Return a list of failures; empty means the section renders cleanly."""
    if shutil.which("tectonic") is None:
        return ["tectonic is not installed (brew install tectonic)"]
    out_dir = out_dir or (paper_dir / "build" / "check")
    rc, log, pdf = compile_tex(paper_dir, wrapper, out_dir)
    failures: list[str] = []
    if rc != 0:
        errors = [l for l in log.splitlines() if l.startswith("!") or "error:" in l.lower()]
        failures.append("compile failed: " + (errors[0] if errors else f"exit {rc}"))
        return failures
    if "There were undefined references" in log:
        refs = sorted(set(re.findall(r"Reference `([^']+)' on page", log)))
        failures.append(f"unresolved references: {refs}")
    text = pdf_text(pdf)
    if text is not None and re.search(r"\?\?", text):
        failures.append("'??' appears in the rendered text (an unresolved reference)")
    for pts, where in overfull_boxes(log):
        if pts > max_overfull:
            failures.append(f"overfull hbox {pts:.1f}pt at lines {where}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper-dir", type=Path, default=PAPER_DIR)
    parser.add_argument("--wrapper", default=WRAPPER)
    parser.add_argument("--max-overfull", type=float, default=2.0, help="Tolerated overfull width in points.")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    failures = check(args.paper_dir, args.wrapper, max_overfull=args.max_overfull, out_dir=args.out_dir)
    out = args.out_dir or (args.paper_dir / "build" / "check")
    if failures:
        print("TEX CHECK FAILED")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    text = pdf_text(out / (Path(args.wrapper).stem + ".pdf")) or ""
    pages = text.count("\f") + 1 if text else "?"
    print(f"TEX CHECK OK: compiled, references resolved, no overfull box above {args.max_overfull}pt ({pages} pages)")


if __name__ == "__main__":
    main()
