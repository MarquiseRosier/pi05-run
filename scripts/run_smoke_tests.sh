#!/usr/bin/env bash
# Run every smoke test, and exit non-zero if any of them fails.
#
# Six of these import pi05_mi, so they need src on the path; run bare, they
# fail with a module error that looks like a regression and is not one. This
# also gates on the exit code, which a shell chain of && does not do reliably
# once a test writes to stderr.
set -uo pipefail

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-.venv/bin/python}
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

failed=()
for test in scripts/smoke_test_*.py; do
    name=$(basename "$test")
    if output=$("$PYTHON" "$test" 2>&1); then
        printf 'PASS  %-42s %s\n' "$name" "$(printf '%s' "$output" | tail -1)"
    else
        printf 'FAIL  %s\n' "$name"
        printf '%s\n' "$output" | tail -20 | sed 's/^/      /'
        failed+=("$name")
    fi
done

echo
if ((${#failed[@]})); then
    echo "${#failed[@]} suite(s) failed: ${failed[*]}"
    exit 1
fi
echo "all smoke tests passed"
