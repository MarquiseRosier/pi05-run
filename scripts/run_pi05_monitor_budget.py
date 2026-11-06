#!/usr/bin/env python
"""Run resumable batched collection and fitting under a wall-clock budget.

The budget includes model loading. No promise about sample size or accuracy is made.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pi05_mi.occlusion_pilot import RunJournal, atomic_json


def run_phase(command, log, seconds):
    if seconds <= 0:
        return {'status': 'budget_exhausted'}
    with log.open('ab', buffering=0) as output:
        proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        limit = time.monotonic() + seconds
        next_update = 0
        try:
            while proc.poll() is None:
                remaining = limit - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, seconds)
                if time.monotonic() >= next_update:
                    state_file = log.parent / 'run_state.json'
                    if state_file.exists():
                        state = json.loads(state_file.read_text())
                        print(json.dumps({k: state.get(k) for k in ('phase', 'status', 'completed_episodes', 'last_control_steps')}), flush=True)
                    next_update = time.monotonic() + 15
                try:
                    proc.wait(timeout=min(1, remaining))
                except subprocess.TimeoutExpired:
                    pass
            return {'status': 'complete' if proc.returncode == 0 else 'failed', 'exit_code': proc.returncode}
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            return {'status': 'budget_limited', 'exit_code': proc.returncode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--minutes', type=float, default=15)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--preset', choices=['smoke', 'pilot', 'a100'], default='a100')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--comparison-width', type=int, default=256)
    args = parser.parse_args()
    if args.minutes < 3:
        raise ValueError('Reserve at least three minutes including loading and reporting')
    started = time.monotonic()
    deadline = started + args.minutes * 60
    root = args.output_dir.resolve(); root.mkdir(parents=True, exist_ok=True)
    journal = RunJournal(root)
    journal.update('planning', budget_seconds=args.minutes * 60)
    scripts = Path(__file__).resolve().parent
    collection_seconds = max(1, args.minutes * 60 - 125)
    command = [sys.executable, str(scripts / 'run_pi05_occlusion_pilot.py'), '--output-dir', str(root),
               '--preset', args.preset, '--batch-size', str(args.batch_size), '--wall-seconds', str(collection_seconds)]
    if args.checkpoint:
        command += ['--checkpoint', str(args.checkpoint), '--comparison-width', str(args.comparison_width)]
    print(f'Collecting in GPU batches of {args.batch_size}; log: {root / "collection.log"}', flush=True)
    collection = run_phase(command, root / 'collection.log', collection_seconds)
    # Stop hooks have at most 5 seconds to flush. Keep the remaining budget for fitting/reporting.
    journal.update('monitor_fitting', collection_result=collection)
    print('Fitting held-out risk monitors; log:', root / 'analysis.log', flush=True)
    analysis = run_phase([sys.executable, str(scripts / 'fit_pi05_occlusion_monitor.py'),
                          '--run-dir', str(root), '--allow-partial'], root / 'analysis.log',
                         max(0, deadline - time.monotonic() - 6))
    episodes = []
    for file in sorted((root / 'episodes').glob('*/episode.json')):
        ep = json.loads(file.read_text())
        episodes.append({k: ep[k] for k in ('episode_id', 'group_id', 'split', 'condition', 'success', 'first_harm_step', 'steps')})
    counts = {s: {'episodes': sum(e['split'] == s for e in episodes),
                  'harmful': sum(e['split'] == s and e['first_harm_step'] is not None for e in episodes),
                  'successful': sum(e['split'] == s and e['success'] for e in episodes)}
              for s in ('train', 'calibration', 'test')}
    summary = {'elapsed_seconds': time.monotonic() - started, 'collection': collection, 'analysis': analysis,
               'completed_episodes': len(episodes), 'unique_starts': len({e['group_id'] for e in episodes}),
               'counts': counts, 'episodes': episodes,
               'claim_scope': 'Exploratory feasibility only. Completion under a wall-clock budget is not random sampling. No harm prevention has been tested.'}
    atomic_json(root / 'budget_summary.json', summary)
    lines = ['# Budgeted occlusion pilot', '', summary['claim_scope'], '',
             f"Completed {len(episodes)} episodes across {summary['unique_starts']} starting scenes.", '',
             '| Split | Completed episodes | Harmful | Successful |', '|---|---:|---:|---:|']
    for split, c in counts.items():
        lines.append(f"| {split} | {c['episodes']} | {c['harmful']} | {c['successful']} |")
    pointer = root / 'latest_future_harm.json'
    if pointer.exists():
        report = Path(json.loads(pointer.read_text())['report'])
        lines.extend(['', report.read_text()])
    else:
        lines.extend(['', '**No validated monitor result is available from this attempt.**',
                      'Check analysis.log: there may be too few complete split groups, no outcome variation, or an integration error.'])
    (root / 'budget_summary.md').write_text('\n'.join(lines) + '\n')
    journal.update('report', 'complete' if analysis['status'] == 'complete' else 'partial',
                   summary=str(root / 'budget_summary.md'), completed_episodes=len(episodes),
                   elapsed_seconds=summary['elapsed_seconds'], collection_result=collection, analysis_result=analysis)
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
