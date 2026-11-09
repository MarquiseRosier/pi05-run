#!/usr/bin/env python
"""Repair/check the monitor's versioned LeRobot runtime without touching data caches."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

LEROBOT_VERSION = '0.4.4'
REQUIREMENT = f'lerobot[libero]=={LEROBOT_VERSION}'


def validate_runtime(require_cuda=False):
    from importlib.metadata import version
    installed = version('lerobot')
    if installed != LEROBOT_VERSION:
        raise RuntimeError(
            f'This collector requires lerobot=={LEROBOT_VERSION}; found {installed}. '
            'Run scripts/pi05_monitor_runtime.py --repair --uv /content/uv-bin/uv '
            '--python /content/lerobot-venv/bin/python --require-cuda. '
            'A class-name alias alone does not restore the simulator observation API.'
        )
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
                scope='Imports/config only; no weights, assets, simulator rollout or GPU inference tested')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair', action='store_true')
    parser.add_argument('--uv', default='/content/uv-bin/uv')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--check-here', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_here:
        print(json.dumps(validate_runtime(args.require_cuda), indent=2))
        return
    if args.repair:
        command = [args.uv, 'pip', 'install', '--python', args.python, '--torch-backend', 'cu128',
                   REQUIREMENT, 'hf-transfer', 'opencv-python', 'numpy', 'langfuse',
                   'scikit-learn', 'joblib', 'pillow']
        print('Repairing packages in place; preserving model caches and run outputs.', flush=True)
        subprocess.run(command, check=True)
    command = [args.python, str(Path(__file__).resolve()), '--check-here']
    if args.require_cuda:
        command.append('--require-cuda')
    subprocess.run(command, check=True, env={**os.environ, 'MPLBACKEND': 'Agg'})


if __name__ == '__main__':
    main()
