#!/usr/bin/env python
"""Prove the LaTeX check is effective: it must pass on the real section and fail on planted faults.

Each fault is one a source-level regex check either cannot see or sees only by
accident: a macro that does not exist, a tabular row with one cell too many,
a reference to a label that is not defined, and a table too wide for the
measure. Skips cleanly when the engine is not installed.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_tex import PAPER_DIR, WRAPPER, check  # noqa: E402

SECTION = "counterfactual_probe_section.tex"


def _skip() -> bool:
    if shutil.which("tectonic") is None:
        print("  (skipped: tectonic not installed)")
        return True
    return False


def _sandbox(mutate) -> Path:
    """Copy the wrapper, the section and everything the wrapper inputs."""
    tmp = Path(tempfile.mkdtemp())
    for name in (WRAPPER, SECTION, "_standalone_preamble.tex"):
        (tmp / name).write_text((PAPER_DIR / name).read_text())
    src = (tmp / SECTION).read_text()
    (tmp / SECTION).write_text(mutate(src))
    return tmp


def test_the_real_section_renders_cleanly() -> None:
    if _skip():
        return
    failures = check(PAPER_DIR, WRAPPER)
    assert not failures, failures


def test_an_undefined_macro_fails_the_check() -> None:
    if _skip():
        return
    tmp = _sandbox(lambda s: s.replace("\\subsection{Hypotheses}", "\\subsection{Hypotheses}\\notamacro{x}", 1))
    failures = check(tmp, WRAPPER)
    assert failures and failures[0].startswith("compile failed"), failures


def test_a_row_with_too_many_cells_fails_the_check() -> None:
    if _skip():
        return
    tmp = _sandbox(lambda s: s.replace("0 & 0 & 0.5 & $4.67$", "0 & 0 & 0.5 & extra & $4.67$", 1))
    failures = check(tmp, WRAPPER)
    assert failures and failures[0].startswith("compile failed"), failures


def test_a_dangling_reference_fails_the_check() -> None:
    if _skip():
        return
    tmp = _sandbox(lambda s: s.replace("Eq.~\\ref{eq:reduce}", "Eq.~\\ref{eq:does-not-exist}", 1)
                   if "Eq.~\\ref{eq:reduce}" in s else s.replace("\\ref{eq:delta}", "\\ref{eq:does-not-exist}", 1))
    failures = check(tmp, WRAPPER)
    assert any("unresolved" in f or "??" in f for f in failures), failures


def test_a_table_wider_than_the_measure_fails_the_check() -> None:
    if _skip():
        return
    tmp = _sandbox(lambda s: s.replace("\\begin{tabular}{p{0.9in}p{1.55in}p{1.6in}p{0.7in}}",
                                       "\\begin{tabular}{p{1.9in}p{1.9in}p{1.9in}p{0.9in}}", 1))
    failures = check(tmp, WRAPPER)
    assert any(f.startswith("overfull") for f in failures), failures


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()
