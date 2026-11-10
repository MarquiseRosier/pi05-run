#!/usr/bin/env python
"""Repair/check the monitor's versioned LeRobot runtime without touching data caches."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

LEROBOT_VERSION = '0.4.4'
# The published 0.4.4 wheel omits the pi extra present in the GitHub source.
# Its libero extra requests stock Transformers, so explicitly override that
# dependency with the fork required by the wheel's own PI05 implementation.
REQUIREMENT = f'lerobot[libero]=={LEROBOT_VERSION}'
TRANSFORMERS_OVERRIDES = Path(__file__).with_name('pi05_transformers_overrides.txt')


def repair_command():
    return shlex.join([sys.executable, str(Path(__file__).resolve()), '--repair',
                       '--uv', '/content/uv-bin/uv', '--python', sys.executable, '--require-cuda'])


def validate_pi_transformers():
    """Run PI05Pytorch's compatibility gate without allocating model weights."""
    message = (
        'Pi0.5 requires the pinned fix/lerobot_openpi Transformers fork; '
        'PyPI LeRobot 0.4.4 does not provide a pi extra. '
        f'Run {repair_command()}. Restart the notebook session after replacing loaded packages.'
    )
    try:
        from transformers.models.siglip import check
        compatible = check.check_whether_transformers_replace_is_installed_correctly()
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(message) from exc
    if not compatible:
        raise RuntimeError(message)


def validate_runtime(require_cuda=False):
    from importlib.metadata import version
    installed = version('lerobot')
    if installed != LEROBOT_VERSION:
        raise RuntimeError(
            f'This collector requires lerobot=={LEROBOT_VERSION}; found {installed}. '
            f'Run {repair_command()}. '
            'A class-name alias alone does not restore the simulator observation API.'
        )
    validate_pi_transformers()
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.scripts.lerobot_eval import preprocess_observation
    import torch
    cfg = LiberoEnv(task='libero_spatial', task_ids=[0], fps=20)
    assert cfg.init_states and cfg.task_ids == [0]
    assert all(callable(f) for f in (make_env, make_env_pre_post_processors, make_policy,
                                    make_pre_post_processors, preprocess_observation,
                                    PI05Policy.predict_action_chunk))
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError('Pinned packages installed, but CUDA is unavailable in the target Python.')
    packages = {}
    for name in ('lerobot', 'torch', 'torchvision', 'transformers', 'hf-libero', 'mujoco', 'robosuite'):
        from importlib.metadata import PackageNotFoundError
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    if require_cuda:
        missing = [name for name in ('hf-libero', 'mujoco', 'robosuite') if packages[name] is None]
        if missing:
            raise RuntimeError('Missing simulator dependencies: ' + ', '.join(missing) + '. Run --repair.')
    return dict(status='passed', packages=packages, cuda_available=torch.cuda.is_available(),
                pi_transformers_compatible=True,
                scope='Imports/config and Pi0.5 Transformers compatibility only; '
                      'no weights, assets, simulator rollout or GPU inference tested')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--repair', action='store_true', help='Reinstall the compatible Transformers build.')
    mode.add_argument('--ensure', action='store_true', help='Check first; repair only if validation fails.')
    parser.add_argument('--uv', default='/content/uv-bin/uv')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--check-here', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_here:
        print(json.dumps(validate_runtime(args.require_cuda), indent=2))
        return
    command = [args.python, str(Path(__file__).resolve()), '--check-here']
    if args.require_cuda:
        command.append('--require-cuda')
    env = {**os.environ, 'MPLBACKEND': 'Agg'}
    if args.ensure:
        # Probe in a child: a failed import must not leave stale modules loaded
        # when checking the repaired installation in the target environment.
        print('Checking Pi0.5 runtime before starting collection.', flush=True)
        if subprocess.run(command, env=env, check=False).returncode == 0:
            return
    if args.repair or args.ensure:
        command = [args.uv, 'pip', 'install', '--python', args.python, '--torch-backend', 'cu128',
                   '--overrides', str(TRANSFORMERS_OVERRIDES),
                   '--reinstall-package', 'transformers',
                   REQUIREMENT, 'hf-transfer', 'opencv-python', 'numpy', 'langfuse',
                   'scikit-learn', 'joblib', 'pillow']
        print('Repairing packages in place; preserving model caches and run outputs.', flush=True)
        subprocess.run(command, check=True)
    command = [args.python, str(Path(__file__).resolve()), '--check-here']
    if args.require_cuda:
        command.append('--require-cuda')
    subprocess.run(command, check=True, env=env)


if __name__ == '__main__':
    main()
