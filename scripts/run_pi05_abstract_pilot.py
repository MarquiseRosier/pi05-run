"""Minimal two-hour monitor pilot orchestration; all work stays in the calling thread."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

from pi05_mi.occlusion_pilot import atomic_json, RunJournal, manifest_changes, PilotConfig, manifest
from pi05_monitor_evidence import export_evidence, atomic_text, archive_evidence
from run_pi05_monitor_budget import analysis_readiness, run_phase


def freeze_abstract_protocol(root, *, checkpoint, batch_size, started_unix, source_digest,
                             cohort='core', allow_partial=True, support_hashes=None):
    if cohort not in ('core', 'feasibility'):
        raise ValueError('Choose core or feasibility before collecting; no outcome-based cohort changes')
    if batch_size not in (1, 4):
        raise ValueError('This minimal protocol supports verified L4 batch 1 or A100 batch 4')
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    count = 4 if cohort == 'core' else 1
    protocol = dict(protocol_version=1, experiment='frozen projected matched monitor, not steering',
                    started_unix=float(started_unix), deadline_unix=float(started_unix)+7200,
                    start_utc=datetime.fromtimestamp(started_unix, timezone.utc).isoformat(),
                    deadline_utc=datetime.fromtimestamp(started_unix+7200, timezone.utc).isoformat(),
                    setup_deadline_unix=started_unix+1200, collection_deadline_unix=started_unix+5400,
                    analysis_deadline_unix=started_unix+6600,
                    notebook_source_sha256=source_digest, support_source_sha256=support_hashes or {},
                    checkpoint=str(Path(checkpoint)), cohort=cohort, tasks=[0,1,2],
                    starts_per_task_per_split=count, planned_episodes=39*count,
                    planned_groups=9*count, batch_size=batch_size,
                    policy='lerobot/pi05_libero_finetuned', layers=[8], comparison_width=256,
                    mode='probe', horizon=280, action_steps=50, inference_steps=10,
                    monitor_stride=8, image_stride=8, seed=1000,
                    force_threshold_n=10, wrong_lift_threshold_m=.05,
                    future_steps=40, logistic_C=.1, calibration_false_stop_target=.1,
                    primary='expert_transcoder_l08_minus_expert_output_l08', primary_domain='all_test',
                    diagnostic='expert_transcoder_l08_minus_expert_input_l08',
                    delays_seconds=[0,.1,.25,.5,1,2,5], force_sensitivity_n=[5,10,20],
                    sensitivity_mode='fixed 10 N fitted monitor, changed evaluation endpoint only',
                    bootstrap='500 paired start-group draws within each fixed task; no frame bootstrap',
                    allow_partial=bool(allow_partial),
                    partial_scope='time-budget-selected exploratory pilot; complete valid groups only',
                    cohort_selection='Core is the minimal reporting target; feasibility is infrastructure only. No outcome-based resizing.',
                    image_encoder='not_validated; omitted; lossless policy images retained for future extraction',
                    reset_protocol='fresh_seeded_simulator_paired_state_v2_no_terminal_autoreset',
                    stop_rule='Setup by minute 20; cooperative collection stops by minute 90; fit before minute 110; export by minute 120. A blocking native call can overrun; record actual time.',
                    no_prevention_claim=True)
    path = root / 'sprint_protocol.json'
    if path.exists():
        changes = manifest_changes(json.loads(path.read_text()), protocol)
        if changes:
            raise ValueError('Frozen sprint changed: ' + '; '.join(changes[:8]) + '. Use a new RUN_NAME; old evidence preserved.')
    else:
        atomic_json(path, protocol)
    if not (root/'planned_manifest.json').exists():
        atomic_json(root/'planned_manifest.json', manifest(PilotConfig(tasks=(0,1,2),
            train_starts=count, calibration_starts=count, test_starts=count,
            batch_size=batch_size, layers=(8,))))
    return protocol


def run_abstract_pilot(root, protocol, *, collect_function, fit_function):
    root = Path(root); journal = RunJournal(root)
    count = protocol['starts_per_task_per_split']
    collection_args = ['--output-dir', str(root), '--tasks','0,1,2',
                       '--train-starts',str(count),'--calibration-starts',str(count),'--test-starts',str(count),
                       '--batch-size',str(protocol['batch_size']), '--layers','8',
                       '--checkpoint',protocol['checkpoint'], '--comparison-width','256',
                       '--action-steps','50', '--monitor-stride','8', '--image-stride','8', '--smoke-gate',
                       '--setup-deadline-unix',str(protocol['setup_deadline_unix'])]
    journal.update('sprint', started_unix=protocol['started_unix'], deadline_unix=protocol['deadline_unix'])
    print('Absolute deadline:', protocol['deadline_utc'], flush=True)
    try:
        if time.time() < protocol['setup_deadline_unix']:
            remaining = max(0, protocol['collection_deadline_unix'] - time.time() - 30)
            if remaining > 0:
                collection_args += ['--wall-seconds', str(remaining)]
                run_phase(lambda: collect_function(collection_args), root/'collection.log', remaining)
        else:
            journal.update('collection','skipped', reason='Minute-20 setup gate expired; preserve/analyze existing captures')
            print('Setup gate expired: no new GPU collection. Exporting available evidence.', flush=True)
        ready, reason = analysis_readiness(root)
        if ready and time.time() < protocol['analysis_deadline_unix']:
            fit_args = ['--run-dir',str(root),'--future-steps','40','--C','0.1','--false-stop-rate','0.1']
            fit_args += ['--families','state,expert_input_l08,expert_output_l08,expert_transcoder_l08']
            if protocol['allow_partial']:
                fit_args.append('--allow-partial')
            run_phase(lambda: fit_function(fit_args), root/'analysis.log', protocol['analysis_deadline_unix']-time.time())
        else:
            journal.update('monitor_fitting','skipped', reason=reason or 'Analysis budget expired')
            print('Comparison not fitted:', reason or 'Analysis budget expired', flush=True)
    except BaseException as exc:
        journal.update('sprint','failed', error=f'{type(exc).__name__}: {exc}')
        # Write the partial packet without replacing the native failure traceback.
        try:
            export_evidence(root)
        except Exception as export_error:
            print('Evidence export also failed:', export_error, flush=True)
        raise
    output = export_evidence(root)
    journal.update('sprint','finished', elapsed_seconds=time.time()-protocol['started_unix'],
                   deadline_overrun_seconds=max(0, time.time()-protocol['deadline_unix']))
    atomic_text(Path(output)/'run_state.json', journal.path.read_text())
    archive_evidence(output)
    return output
