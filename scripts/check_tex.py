#!/usr/bin/env python
"""Compile the paper section and fail on anything a reviewer's PDF would show.

Regex checks on LaTeX source catch brace and environment imbalance and little
else. Only the engine knows whether a macro is defined, whether a tabular row
has the right number of cells, whether every \\ref resolves, and whether a
table overruns the measure. This runs Tectonic on the standalone wrapper at the
ICLR text width and fails on:

  * any TeX error (the engine's own exit status);
  * unresolved references in the final pass, except those pointing at a label
    that another section in the same directory defines -- a standalone build of
    one section cannot resolve a cross-section reference, but the assembled
    paper can, so those are reported and not failed;
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
WRAPPER = "counterfactual_probe_section_standalone.tex"


def discover_wrappers(paper_dir: Path = PAPER_DIR) -> list[str]:
    """Every per-experiment standalone build, in a stable order."""
    return sorted(p.name for p in paper_dir.glob("*_standalone.tex"))


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


def labels_defined_in(paper_dir: Path) -> set[str]:
    """Every label defined by any section in the directory.

    A reference to one of these from a standalone build is a cross-section
    reference, which the assembled paper resolves. A reference to anything else
    is a typo and must fail.
    """
    labels: set[str] = set()
    for path in paper_dir.glob("*.tex"):
        if path.name.endswith("_standalone.tex") or path.name.startswith("_"):
            continue
        labels.update(re.findall(r"\\label\{([^}]+)\}", path.read_text(errors="replace")))
    return labels


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
    external = labels_defined_in(paper_dir)
    unresolved = sorted(set(re.findall(r"Reference `([^']+)' on page", log)))
    unknown = [r for r in unresolved if r not in external]
    cross = [r for r in unresolved if r in external]
    if unknown:
        failures.append(f"unresolved references defined nowhere: {unknown}")
    if cross:
        print(f"     cross-section references (resolve in the assembled paper): {cross}")
    text = pdf_text(pdf)
    if text is not None and re.search(r"\?\?", text) and not cross:
        failures.append("'??' appears in the rendered text (an unresolved reference)")
    for pts, where in overfull_boxes(log):
        if pts > max_overfull:
            failures.append(f"overfull hbox {pts:.1f}pt at lines {where}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper-dir", type=Path, default=PAPER_DIR)
    parser.add_argument("--wrapper", default=None, help="One wrapper file. Default: every section.")
    parser.add_argument("--section", default=None, help="Section stem, e.g. e4_time_conditioning.")
    parser.add_argument("--max-overfull", type=float, default=2.0, help="Tolerated overfull width in points.")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.section:
        wrappers = [f"{args.section}_standalone.tex"]
    elif args.wrapper:
        wrappers = [args.wrapper]
    else:
        wrappers = discover_wrappers(args.paper_dir)
    if not wrappers:
        print("TEX CHECK: no standalone wrappers found")
        sys.exit(1)

    any_failed = False
    for wrapper in wrappers:
        stem = Path(wrapper).stem
        out = args.out_dir or (args.paper_dir / "build" / stem)
        if not (args.paper_dir / wrapper).exists():
            print(f"FAIL {stem}: no such wrapper")
            any_failed = True
            continue
        failures = check(args.paper_dir, wrapper, max_overfull=args.max_overfull, out_dir=out)
        if failures:
            any_failed = True
            print(f"FAIL {stem}")
            for f in failures:
                print("  -", f)
            continue
        text = pdf_text(out / (stem + ".pdf")) or ""
        pages = text.count("\f") + 1 if text else "?"
        print(f"ok   {stem}  ({pages} pages)")
    if any_failed:
        print("\nTEX CHECK FAILED")
        sys.exit(1)
    print(f"\nTEX CHECK OK: {len(wrappers)} section(s) compiled, references resolved, "
          f"no overfull box above {args.max_overfull}pt")


if __name__ == "__main__":
    main()
