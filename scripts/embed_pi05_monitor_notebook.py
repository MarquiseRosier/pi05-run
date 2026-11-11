#!/usr/bin/env python
"""Refresh editable pilot cells from CLI sources; never used at notebook runtime."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/pi05_occlusion_monitor_pilot.ipynb'
SOURCES = (
    ('pilot-core', 'Pilot Definitions — Configuration, Occlusion, And Persistence',
     'src/pi05_mi/occlusion_pilot.py', {}),
    ('pilot-collector', 'Pilot Definitions — Batched Collection',
     'scripts/run_pi05_occlusion_pilot.py', {'main': 'collect_occlusion_pilot', 'arguments': 'pilot_arguments'}),
    ('pilot-fitting', 'Pilot Definitions — Monitor Fitting',
     'scripts/fit_pi05_occlusion_monitor.py', {'main': 'fit_occlusion_monitor'}),
    ('pilot-runner', 'Pilot Definitions — Budget And Live Reporting',
     'scripts/run_pi05_monitor_budget.py', {'main': 'run_monitor'}),
)

# Read support-library provenance without starting git or a Python worker.
GIT_READER = '''
def notebook_support_commit(repo):
    git = Path(repo) / '.git'
    if git.is_file():
        git = (Path(repo) / git.read_text().strip().removeprefix('gitdir: ')).resolve()
    head = (git / 'HEAD').read_text().strip()
    if not head.startswith('ref: '):
        return head
    ref = head[5:]
    if (git / ref).exists():
        return (git / ref).read_text().strip()
    packed = git / 'packed-refs'
    if packed.exists():
        for line in packed.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    raise RuntimeError('Cannot determine support-library revision; rerun Clone Or Update Repo.')
'''


def render_source(path, renames):
    source = path.read_text()
    tree = ast.parse(source)
    doc = ast.get_docstring(tree) or ''
    lines = source.splitlines(keepends=True)
    edits = []
    for node in tree.body:
        remove = (
            isinstance(node, ast.ImportFrom) and node.module == 'pi05_mi.occlusion_pilot'
            or isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == 'sys.path.insert'
            or isinstance(node, ast.If) and '__name__' in ast.unparse(node.test)
        )
        if remove:
            edits.append((node.lineno - 1, node.end_lineno, ''))
    for start, end, replacement in reversed(edits):
        lines[start:end] = [replacement]
    source = ''.join(lines)
    for old, new in renames.items():
        source = source.replace(f'def {old}(', f'def {new}(')
    source = source.replace('args = arguments(argv)', 'args = pilot_arguments(argv)')
    source = source.replace('description=__doc__', f'description={doc!r}')
    if path.name == 'run_pi05_occlusion_pilot.py':
        source = source.replace('import subprocess\n', '')
        source = source.replace('subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()',
                                'notebook_support_commit(LOCAL_REPO), notebook_pilot_version=NOTEBOOK_PILOT_VERSION')
    if path.name == 'run_pi05_monitor_budget.py':
        source = source.replace('        from run_pi05_occlusion_pilot import main as collect_main\n', '')
        source = source.replace('        collect_main(collection_args)', '        collect_occlusion_pilot(collection_args)')
        source = source.replace('            from fit_pi05_occlusion_monitor import main as fit_main\n', '')
        source = source.replace('            fit_main(', '            fit_occlusion_monitor(')
    return source.rstrip() + '\n'


def embedded_cells():
    cells = []
    for cell_id, title, path, renames in SOURCES:
        source = '# @title ' + title + '\n' + render_source(ROOT / path, renames)
        if cell_id == 'pilot-core':
            source += GIT_READER
        cells.append(dict(cell_type='code', id=cell_id, metadata={}, execution_count=None,
                          outputs=[], source=source.splitlines(keepends=True)))
    digest = hashlib.sha256(''.join(''.join(c['source']) for c in cells).encode()).hexdigest()
    cells[-1]['source'].append(f'\nNOTEBOOK_PILOT_VERSION = {digest!r}\n')
    return cells


def main():
    notebook = json.loads(NOTEBOOK.read_text())
    ids = {entry[0] for entry in SOURCES}
    notebook['cells'] = [c for c in notebook['cells'] if c.get('id') not in ids]
    index = next(i for i, c in enumerate(notebook['cells'])
                 if ''.join(c.get('source', [])).startswith('# @title Run Budgeted'))
    notebook['cells'][index:index] = embedded_cells()
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
