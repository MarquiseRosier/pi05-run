#!/usr/bin/env python
"""Run resumable batched collection and fitting under a wall-clock budget.

Collection and fitting run in the calling thread with live output. The budget
includes model loading and is cooperative, not a forced process timeout.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pi05_mi.occlusion_pilot import RunJournal, atomic_json, episode_control_valid


def analysis_readiness(root):
    """Only complete paired groups in all splits can reach the fitter."""
    manifest_path = root / 'manifest.json'
    if not manifest_path.exists():
        return False, 'Collection did not produce a manifest.'
    planned = json.loads(manifest_path.read_text())['episodes']
    groups = {}
    for job in planned:
        groups.setdefault(job['group_id'], []).append(job)
    complete = [group for group in groups.values() if all(
        (root / 'episodes' / job['episode_id'] / 'episode.json').exists()
        and (root / 'episodes' / job['episode_id'] / 'features.npz').exists() for job in group)]
    excluded = [group[0]['group_id'] for group in complete if any(not episode_control_valid(
        json.loads((root / 'episodes' / job['episode_id'] / 'episode.json').read_text())) for job in group)]
    atomic_json(root / 'control_exclusions.json', {'group_ids': sorted(excluded),
                'reason': 'At least one camera frame had no feasible same-size miss control'})
    complete = [group for group in complete if group[0]['group_id'] not in excluded]
    splits = {group[0]['split'] for group in complete}
    missing = sorted({'train', 'calibration', 'test'} - splits)
    if missing:
        return False, ('No complete paired groups for: ' + ', '.join(missing)
                       + f'. Excluded {len(excluded)} groups with infeasible miss controls. See collection.log.')
    return True, None


class Tee:
    """Mirror Python output to the notebook/terminal and the current phase log."""
    def __init__(self, live, log):
        self.live, self.log = live, log

    def write(self, text):
        self.live.write(text)
        # Libraries may retain a logging handler bound to this stream after
        # the phase ends. Such messages should remain visible on later runs.
        if not self.log.closed:
            self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.live.flush()
        if not self.log.closed:
            self.log.flush()

    def __getattr__(self, name):
        return getattr(self.live, name)


def run_phase(function, log, seconds):
    if seconds <= 0:
        return {'status': 'budget_exhausted'}
    if log.exists():
        log.rename(log.with_name(f'{log.stem}.{time.time_ns()}.log'))
    started = time.monotonic()
    with log.open('w', buffering=1) as output:
        with redirect_stdout(Tee(sys.stdout, output)), redirect_stderr(Tee(sys.stderr, output)):
            try:
                function()
            except BaseException as exc:
                traceback.print_exc(file=output)
                RunJournal(log.parent).update(log.stem, 'failed', error=f'{type(exc).__name__}: {exc}')
                raise  # Preserve the original notebook traceback and exception type.
    return {'status': 'complete', 'exit_code': 0,
            'elapsed_seconds': time.monotonic() - started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--minutes', type=float, default=15)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--preset', choices=['smoke', 'pilot', 'a100'], default='a100')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--comparison-width', type=int, default=256)
    args = parser.parse_args(argv)
    if args.minutes < 3:
        raise ValueError('Reserve at least three minutes including loading and reporting')
    started = time.monotonic()
    deadline = started + args.minutes * 60
    root = args.output_dir.resolve(); root.mkdir(parents=True, exist_ok=True)
    journal = RunJournal(root)
    journal.update('planning', budget_seconds=args.minutes * 60, error=None)
    collection_seconds = max(1, args.minutes * 60 - 125)
    collection_args = ['--output-dir', str(root), '--preset', args.preset,
                       '--batch-size', str(args.batch_size), '--wall-seconds', str(collection_seconds)]
    if args.checkpoint:
        collection_args += ['--checkpoint', str(args.checkpoint), '--comparison-width', str(args.comparison_width)]

    def collect():
        from run_pi05_occlusion_pilot import main as collect_main
        collect_main(collection_args)

    print(f'Collecting in this thread in GPU batches of {args.batch_size}; log: {root / "collection.log"}', flush=True)
    collection = run_phase(collect, root / 'collection.log', collection_seconds)
    if json.loads(journal.path.read_text()).get('status') == 'budget_limited':
        collection['status'] = 'budget_limited'
    ready, reason = analysis_readiness(root)
    if ready:
        journal.update('monitor_fitting', collection_result=collection)
        print('Fitting held-out risk monitors; log:', root / 'analysis.log', flush=True)
        def fit():
            from fit_pi05_occlusion_monitor import main as fit_main
            fit_main(['--run-dir', str(root), '--allow-partial'])
        analysis = run_phase(fit, root / 'analysis.log', max(0, deadline - time.monotonic() - 6))
    else:
        analysis = dict(status='skipped', reason=reason)
        journal.update('monitor_fitting', 'skipped', collection_result=collection, skip_reason=reason)
        with (root / 'analysis.log').open('a') as handle:
            handle.write('Analysis skipped: ' + reason + '\n')
        print('Analysis skipped:', reason, flush=True)
    episodes = []
    for file in sorted((root / 'episodes').glob('*/episode.json')):
        ep = json.loads(file.read_text())
        episodes.append({**{k: ep[k] for k in ('episode_id', 'group_id', 'split', 'condition', 'success', 'first_harm_step', 'steps')},
                         'matched_control_valid': bool(episode_control_valid(ep))})
    excluded = sorted({e['group_id'] for e in episodes if not e['matched_control_valid']})
    counts = {s: {'episodes': sum(e['split'] == s for e in episodes),
                  'harmful': sum(e['split'] == s and e['first_harm_step'] is not None for e in episodes),
                  'successful': sum(e['split'] == s and e['success'] for e in episodes)}
              for s in ('train', 'calibration', 'test')}
    summary = {'elapsed_seconds': time.monotonic() - started, 'collection': collection, 'analysis': analysis,
               'completed_episodes': len(episodes), 'unique_starts': len({e['group_id'] for e in episodes}),
               'counts': counts, 'episodes': episodes, 'control_excluded_groups': excluded,
               'claim_scope': 'Exploratory feasibility only. Completion under a wall-clock budget is not random sampling. No harm prevention has been tested.'}
    atomic_json(root / 'budget_summary.json', summary)
    lines = ['# Budgeted occlusion pilot', '', summary['claim_scope'], '',
             f"Completed {len(episodes)} episodes across {summary['unique_starts']} starting scenes.", '',
             '| Split | Completed episodes | Harmful | Successful |', '|---|---:|---:|---:|']
    for split, c in counts.items():
        lines.append(f"| {split} | {c['episodes']} | {c['harmful']} | {c['successful']} |")
    if excluded:
        lines.extend(['', f'{len(excluded)} paired groups excluded from fitting because a same-size miss control '
                      'could not fit. Counts above include those rollouts; fitted results are conditional on control feasibility.'])
    pointer = root / 'latest_future_harm.json'
    if analysis['status'] == 'complete' and pointer.exists():
        report = Path(json.loads(pointer.read_text())['report'])
        lines.extend(['', report.read_text()])
    else:
        lines.extend(['', '**No validated monitor result is available from this attempt.**',
                      analysis.get('reason', 'Check collection.log and analysis.log for the failing phase.')])
    state = json.loads(journal.path.read_text())
    if collection['status'] == 'failed':
        lines.extend(['', '**Collection failed.**', str(state.get('error') or 'Inspect collection.log.')])
    (root / 'budget_summary.md').write_text('\n'.join(lines) + '\n')
    journal.update('report', 'complete' if analysis['status'] == 'complete' and collection['status'] != 'failed' else 'partial',
                   summary=str(root / 'budget_summary.md'), completed_episodes=len(episodes),
                   elapsed_seconds=summary['elapsed_seconds'], collection_result=collection, analysis_result=analysis)
    print('\n'.join(lines), flush=True)
    if collection['status'] == 'failed' or analysis['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
