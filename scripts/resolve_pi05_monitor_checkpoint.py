"""Resolve the shared checkpoint used by the feature and sanity-run notebooks."""
from pathlib import Path

SHARED_CHECKPOINT = Path('transcoders/pi05_libero/allframes_80-10-10_epoch1_b8_exp16_latest_lambda1e-4/step_027233.pt')


def resolve_checkpoint(value, drive_root, repo_root):
    drive_root, repo_root = Path(drive_root), Path(repo_root)
    if str(value).strip():
        candidate = Path(str(value).strip()).expanduser()
        candidates = [candidate] if candidate.is_absolute() else [drive_root / candidate, repo_root / candidate]
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError('Explicit CHECKPOINT not found: ' + ', '.join(map(str, candidates)))

    # The canonical default is already used in Ahmed/Jayden/Marquise notebooks.
    canonical = drive_root / SHARED_CHECKPOINT
    if canonical.is_file():
        return canonical
    roots = [drive_root / 'transcoders', drive_root / 'outputs/transcoders',
             drive_root / 'checkpoints', repo_root / 'checkpoints', repo_root / 'outputs/transcoders']
    matches = sorted({path.resolve() for root in roots if root.is_dir()
                      for path in root.rglob(SHARED_CHECKPOINT.name) if path.is_file()})
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError('Multiple copies of step_027233.pt found; set CHECKPOINT explicitly:\n' +
                         '\n'.join(map(str, matches)))
    raise FileNotFoundError(
        f'Expected the shared trained transcoder at {canonical}.\n'
        'Also searched for step_027233.pt under:\n' + '\n'.join(map(str, roots)) +
        '\nSet CHECKPOINT to its actual path if it has another filename. '
        'Weights live in Drive, not Git; no replacement checkpoint was downloaded or trained.'
    )
