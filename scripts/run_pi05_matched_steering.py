#!/usr/bin/env python
"""Discover, validate, freeze, and test steering handles in the calling thread."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'src')]
from pi05_mi.matched_steering import (SteeringConfig, SteeringHooks, fit_layer_handles,
    preference, steering_score, probe_auc, select_layers, paired_report, protocol_hash)
from pi05_mi.occlusion_pilot import atomic_json, bddl_sections, target_bowl_spec


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def contrast_prompts(inner):
    """Use benchmark descriptions of the two instance locations; never guess left/right."""
    spec = target_bowl_spec(inner)
    if len(spec['distractor_instances']) != 1:
        raise ValueError('This test requires exactly one distractor bowl')
    path = Path(spec['bddl_path'])
    here = bddl_sections(path.read_text())
    other = spec['distractor_instances'][0]
    regions = [p[2] for p in here[':init'] if len(p) == 3 and p[1] == other and p[0].lower() in ('on', 'in')]
    if len(regions) != 1:
        raise ValueError(f'Cannot identify one initial region for {other}')
    def subjects(form):
        if isinstance(form, list) and form:
            if str(form[0]).lower() in ('on', 'in') and len(form) == 3:
                return [form[1]]
            return [s for child in form for s in subjects(child)]
        return []
    matches = []
    for sibling in sorted(path.parent.glob('*.bddl')):
        parsed = bddl_sections(sibling.read_text())
        goal_bowls = [s for goal in parsed.get(':goal', []) for s in subjects(goal) if 'bowl' in s]
        if len(goal_bowls) != 1:
            continue
        if any(len(p) == 3 and p[1] == goal_bowls[0] and p[2] == regions[0]
               and p[0].lower() in ('on', 'in') for p in parsed.get(':init', [])):
            matches.append((' '.join(parsed[':language']), sibling.name))
    prompts = sorted(set(p for p, _ in matches))
    positive = ' '.join(here[':language'])
    if len(prompts) != 1 or prompts[0] == positive:
        raise ValueError(f'Cannot resolve an unambiguous distractor instruction for {path.name}: {matches}')
    return {'1': positive, '-1': prompts[0]}, spec, matches


def step_without_autoreset(inner, action):
    """LeRobot's inner.step resets on success; use LIBERO directly for fixed-horizon measurements."""
    raw, _, done, info = inner._env.step(action)
    return raw, bool(done), bool(inner._env.check_success()), info


class SteeringRuntime:
    def __init__(self, config, checkpoint, policy_path, device):
        import torch
        from probe_pi05_transcoder_counterfactual import build_harness, load_transcoders
        from pi05_mi.patch_pi05 import (Pi05TranscoderContext, list_pi05_action_expert_mlp_targets,
                                       install_pi05_action_expert_wrappers)
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env_pre_post_processors
        self.config, self.device = config, device
        self.args = SimpleNamespace(suite='libero_spatial', task_id=config.train_tasks[0], prompt='',
                                    policy_path=policy_path, device=device, policy_dtype='bfloat16')
        self.base = build_harness(self.args)
        self.harness = self.base
        try:
            self.policy = self.base.policy
            targets = [t for t in list_pi05_action_expert_mlp_targets(self.policy) if t.layer_index in config.layers]
            transcoders = load_transcoders(checkpoint, torch.device(device), layers=config.layers)
            if {t.layer_index for t in targets} != set(config.layers) or any(t.name not in transcoders for t in targets):
                raise ValueError('Policy/checkpoint is missing requested matched layers')
            context = Pi05TranscoderContext(mode='train', capture_records=False)
            install_pi05_action_expert_wrappers(self.policy, context=context, mode='train')
            self.hooks = SteeringHooks(targets, transcoders, context)
            _, self.post = make_pre_post_processors(policy_cfg=self.policy.config, pretrained_path=policy_path)
            _, self.env_post = make_env_pre_post_processors(
                env_cfg=LiberoEnv(task='libero_spatial', task_ids=[self.args.task_id], fps=20), policy_cfg=self.policy.config)
        except BaseException:
            self.base.vec_env.close()
            raise

    def set_case(self, case):
        from probe_pi05_transcoder_counterfactual import build_harness
        if self.args.task_id != case['task_id']:
            self.harness.vec_env.close()
            self.args.task_id = case['task_id']
            self.harness = build_harness(self.args, load_policy=False)
        inner = self.harness.inner_env
        if inner._init_states is None or case['init_state_id'] >= len(inner._init_states):
            raise ValueError('Requested initial state is unavailable; no modulo wrapping allowed')
        self.prompts, self.target_spec, self.prompt_sources = contrast_prompts(inner)

    def trial(self, case, recipient_sign, *, spec=None, capture=False):
        import mujoco
        import torch
        from probe_pi05_transcoder_counterfactual import _add_batch_axis, sample_shared_noise
        from run_pi05_occlusion_pilot import PhysicsRecorder
        inner = self.harness.inner_env
        inner.init_state_id = case['init_state_id']
        observation, _ = inner.reset(seed=case['seed'])
        env = inner._env.env
        state = env.sim.get_state().flatten().copy()
        initial_hash = hashlib.sha256(state.tobytes()).hexdigest()
        image_hash = hashlib.sha256(b''.join(np.asarray(observation['pixels'][k]).tobytes()
                                    for k in sorted(observation['pixels']))).hexdigest()
        target = self.target_spec['target_instance']; other = self.target_spec['distractor_instances'][0]
        model = env.sim.model._model
        bodies = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, env.objects_dict[name].root_body)
                  for name in (target, other)}
        if min(bodies.values()) < 0:
            raise ValueError('A declared bowl has no simulator body')
        xyz = {name: env.sim.data._data.xpos[body].copy() for name, body in bodies.items()}
        raw = env._get_observations(force_update=True)
        start_eef = np.asarray(raw['robot0_eef_pos']).copy()
        start_pref = preference(start_eef, xyz[target], xyz[other])
        self.hooks.reset(capture=capture, spec=spec)
        self.policy.reset()
        obs = self.base.preprocess_observation(_add_batch_axis(observation))
        obs['task'] = [self.prompts[str(recipient_sign)]]
        batch = self.base.preprocessor(self.base.env_preprocessor(obs))
        noise = sample_shared_noise(self.policy, 1, torch.device(self.device), seed=case['seed'])
        physics = PhysicsRecorder(env, env.objects_dict[target].root_body,
                                  SimpleNamespace(force_threshold=10., lift_threshold=.05), [bodies[other]])
        try:
            with torch.inference_mode():
                normalized = self.policy.predict_action_chunk(batch, noise=noise, num_steps=self.config.inference_steps)
                controls = self.env_post({'action': self.post(normalized[0, :self.config.action_steps])})['action']
                controls = controls.detach().cpu().numpy()
            arrays = self.hooks.arrays()
            if capture and set(arrays) != {f'{f}_{l}' for f in ('dense', 'sparse') for l in self.config.layers}:
                raise RuntimeError('Missing matched activation captures')
            if spec is not None and not self.hooks.norms:
                raise RuntimeError('Intervention hook did not run')
            trajectory, events, actions, success = [start_eef.tolist()], [], [], False
            for step, action in enumerate(controls, 1):
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise ValueError('Invalid simulator action')
                action = np.clip(action, inner.action_space.low, inner.action_space.high)
                raw_after, terminated, is_success, _ = step_without_autoreset(inner, action)
                actions.append(action.tolist())
                trajectory.append(np.asarray(raw_after['robot0_eef_pos']).tolist())
                events.append(physics.finish_control_step(step))
                success |= is_success
                if terminated and step != self.config.action_steps:
                    raise RuntimeError('LIBERO terminated before the fixed movement horizon; refusing unequal-duration scores')
            endpoint = np.array(trajectory[-1]); displacement = endpoint - start_eef
            result = dict(**case, recipient_sign=recipient_sign, prompt=self.prompts[str(recipient_sign)],
                initial_state_sha256=initial_hash, target=self.target_spec, prompt_sources=self.prompt_sources,
                initial_images_sha256=image_hash,
                initial_bowl_positions={k: v.tolist() for k, v in xyz.items()},
                initial_preference=start_pref, final_preference=preference(endpoint, xyz[target], xyz[other]),
                eef_displacement_m=displacement.tolist(), eef_displacement_norm_m=float(np.linalg.norm(displacement)),
                path_length_m=float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum()),
                executed_actions=actions, trajectory_xyz=trajectory, physics=events,
                intervention=None if spec is None else dict(family=spec['family'], mode=spec['mode'],
                    layer=spec['handle']['layer'], sign=spec['sign'], budget=spec['budget']),
                original_task_success=success, harm=any(e['harm'] for e in events),
                harm_scope='Original-goal proxy: robot/non-target contact >10 N or original distractor lift >5 cm; '
                           'distractor-directed instructions may intentionally lift that bowl.',
                intervention_norms=list(self.hooks.norms), steps=len(actions))
            return result, arrays, observation['pixels']
        finally:
            physics.close()
            self.hooks.reset()

    def close(self):
        self.hooks.close()
        self.harness.vec_env.close()


def saved_trial(runtime, root, case, recipient_sign, trial_id, *, spec=None, capture=False):
    folder = root / 'cases' / case['case_id']
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{trial_id}.json'
    if path.exists():
        result = json.loads(path.read_text())
        if capture:
            with np.load(folder / f'{trial_id}.npz', allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
        else:
            arrays = {}
        return result, arrays
    atomic_json(root / 'state.json', dict(status='running', split=case['split'], case_id=case['case_id'], trial=trial_id))
    result, arrays, images = runtime.trial(case, recipient_sign, spec=spec, capture=capture)
    hashes = {json.loads(p.read_text())['initial_state_sha256'] for p in folder.glob('*.json')}
    if hashes and hashes != {result['initial_state_sha256']}:
        raise RuntimeError(f"Reset drift for {case['case_id']}: refusing an unmatched comparison")
    image_hashes = {json.loads(p.read_text())['initial_images_sha256'] for p in folder.glob('*.json')}
    if image_hashes and image_hashes != {result['initial_images_sha256']}:
        raise RuntimeError(f"Initial images changed for {case['case_id']}; contrasts must use identical pixels")
    if capture:
        tmp = folder / f'{trial_id}.tmp'
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        tmp.replace(folder / f'{trial_id}.npz')
    if not (folder / 'image.png').exists():
        from PIL import Image
        for name, image in images.items():
            Image.fromarray(image).save(folder / f'{name}.png')
    atomic_json(path, result)  # Commit marker last.
    print(json.dumps(dict(case=case['case_id'], trial=trial_id, final_preference=result['final_preference'],
                          harm=result['harm'], steps=result['steps'])), flush=True)
    return result, arrays


def run_steering_experiment(config, checkpoint, output_dir, *, device='cuda',
                            policy_path='lerobot/pi05_libero_finetuned', source_digest='cli', phase='all'):
    """Resume exact protocol stages; test cases are first opened after freezing selection."""
    from pi05_monitor_runtime import validate_runtime
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    if phase not in ('all', 'discovery', 'validation', 'test'):
        raise ValueError('Unknown phase')
    if phase == 'test' and not (root / 'frozen_selection.json').exists():
        raise ValueError('Run validation and freeze selection before requesting test')
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    print('Hashing the trained checkpoint to lock run provenance...', flush=True)
    checkpoint_hash = file_digest(checkpoint)
    manifest = dict(config=asdict(config), cases=config.cases(), checkpoint_sha256=checkpoint_hash,
                    policy_path=policy_path, source_digest=source_digest,
                    protocol_hash=protocol_hash(config, checkpoint_hash, source_digest),
                    representation='pretrained timestep-conditioned transcoder, not an SAE',
                    control='first action chunk; opposite prompts on identical initial images; native MLP plus bounded delta',
                    sparse_pretraining='Existing transcoder may have seen LIBERO states during unsupervised pretraining')
    manifest = json.loads(json.dumps(manifest))
    path = root / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Steering protocol changed. Use a new output directory; existing trials are preserved.')
    atomic_json(path, manifest)
    atomic_json(root / 'runtime.json', validate_runtime(require_cuda=device.startswith('cuda')))
    runtime = SteeringRuntime(config, checkpoint, policy_path, device)
    cases = config.cases()
    native, features = {}, {}
    try:
        # Discover on train and measure readout generalization on validation only.
        for case in [c for c in cases if c['split'] != 'test']:
            runtime.set_case(case)
            for sign in (-1, 1):
                result, arrays = saved_trial(runtime, root, case, sign, f'native_{sign}', capture=True)
                native[(case['case_id'], sign)] = result
                features[(case['case_id'], sign)] = arrays
        gaps = [dict(case_id=c['case_id'], split=c['split'],
                     native_preference_gap=native[(c['case_id'], 1)]['final_preference'] - native[(c['case_id'], -1)]['final_preference'])
                for c in cases if c['split'] != 'test']
        atomic_json(root / 'native_contrasts.json', {'contrasts': gaps, 'min_train_gap': config.min_train_gap})
        qualified = {r['case_id'] for r in gaps if r['native_preference_gap'] >= config.min_train_gap}
        training = [c for c in cases if c['split'] == 'train' and c['case_id'] in qualified]
        if len(training) < 2:
            raise ValueError('Fewer than two training starts show the required opposite-prompt movement contrast. '
                             'Inspect native_contrasts.json; do not claim polarity from instruction text alone.')
        labels = np.array([s for c in training for s in (-1, 1)])
        handles, attribution = [], []
        validation = [c for c in cases if c['split'] == 'validation']
        for layer in config.layers:
            values = {family: np.stack([features[(c['case_id'], s)][f'{family}_{layer}']
                        for c in training for s in (-1, 1)]) for family in ('dense', 'sparse')}
            handle = fit_layer_handles(values['dense'], values['sparse'], labels,
                                        layer=layer, k=config.sparse_k, seed=config.seed)
            handles.append(handle)
            contrasts = {}
            for family, x in values.items():
                difference = x[labels == 1].mean(0) - x[labels == -1].mean(0)
                contrasts[family] = dict(raw_contrast_l2=float(np.linalg.norm(difference)),
                    standardized_contrast_l2=float(np.linalg.norm(difference / x.std(0).clip(1e-6))),
                    native_width=int(x.shape[1]))
            attribution.append(dict(layer=layer, training_activation_contrasts=contrasts,
                sparse_feature_ids=handle['sparse']['indices'],
                **{family + '_validation_auc': probe_auc(handle, family,
                np.stack([features[(c['case_id'], s)][f'{family}_{layer}'] for c in validation for s in (-1, 1)]),
                [s for c in validation for s in (-1, 1)]) for family in ('dense', 'sparse')}))
        atomic_json(root / 'discovery.json', dict(handles=handles, readout_attribution=attribution,
                    included_training_cases=[c['case_id'] for c in training],
                    native_contrasts=gaps, note='AUC predicts instruction polarity; it is not causal importance. '
                    'Inspect native movement gaps to check whether the prompts produce opposite behavior.'))
        if phase == 'discovery':
            atomic_json(root / 'state.json', dict(status='complete', phase='discovery'))
            return
        freeze_path = root / 'frozen_selection.json'
        if freeze_path.exists():
            frozen = json.loads(freeze_path.read_text())
            if frozen['protocol_hash'] != manifest['protocol_hash']:
                raise ValueError('Frozen selection belongs to another protocol')
            selected = frozen['selected']
        else:
            validation_rows = []
            for case in validation:
                runtime.set_case(case)
                for handle in handles:
                    for family in ('dense', 'sparse'):
                        for desired in (-1, 1):
                            spec = dict(handle=handle, family=family, mode='steer', sign=desired, budget=config.budget)
                            trial_id = f'validation_{family}_l{handle["layer"]}_to{desired}'
                            result, _ = saved_trial(runtime, root, case, -desired, trial_id, spec=spec)
                            validation_rows.append(dict(**case, layer=handle['layer'], family=family, desired_sign=desired,
                                score=steering_score(native[(case['case_id'], -desired)], result, desired)))
            selected, scores = select_layers(validation_rows, handles)
            frozen = dict(protocol_hash=manifest['protocol_hash'], selected=selected,
                          validation_scores=scores, validation_trials=validation_rows,
                          primary_budget=config.budget, test_case_ids=[c['case_id'] for c in cases if c['split'] == 'test'])
            atomic_json(freeze_path, frozen)
        if phase == 'validation':
            atomic_json(root / 'state.json', dict(status='complete', phase='validation', test_started=False))
            return
        rows, effects, test_contrasts = [], [], []
        for case in [c for c in cases if c['split'] == 'test']:
            runtime.set_case(case)
            baselines = {s: saved_trial(runtime, root, case, s, f'native_{s}')[0] for s in (-1, 1)}
            test_contrasts.append(dict(**case, native_preference_gap=baselines[1]['final_preference'] - baselines[-1]['final_preference']))
            for family in ('dense', 'sparse'):
                handle = selected[family]
                for desired in (-1, 1):
                    baseline = baselines[-desired]
                    for mode, dose in [('zero', 0.), ('steer', .5), ('steer', 1.), ('random', 1.),
                                       ('ablate', 1.), ('random_ablate', 1.)]:
                        spec = dict(handle=handle, family=family, mode=mode, sign=desired, budget=config.budget * dose)
                        result, _ = saved_trial(runtime, root, case, -desired,
                            f'test_{family}_{mode}_dose{dose}_to{desired}', spec=spec)
                        if mode == 'zero':
                            if not np.allclose(result['executed_actions'], baseline['executed_actions'], atol=1e-5, rtol=0):
                                raise RuntimeError('Zero-intervention action check failed; stop causal interpretation')
                            if not np.allclose(result['trajectory_xyz'], baseline['trajectory_xyz'], atol=1e-6, rtol=0):
                                raise RuntimeError('Zero-intervention simulator replay drifted')
                        row = dict(**case, family=family, layer=handle['layer'], mode=mode, dose=dose, desired_sign=desired,
                            score=steering_score(baseline, result, desired), harm=result['harm'], baseline_harm=baseline['harm'],
                            displacement_m=result['eef_displacement_m'], baseline_displacement_m=baseline['eef_displacement_m'],
                            realized_relative_rms=float(np.mean([n['mean_relative_rms'] for n in result['intervention_norms']])))
                        rows.append(row)
                # A positive reduction means the handle's ablation reduced native polarity separation.
                for mode in ('ablate', 'random_ablate'):
                    trials = {s: json.loads((root / 'cases' / case['case_id'] /
                               f'test_{family}_{mode}_dose1.0_to{-s}.json').read_text()) for s in (-1, 1)}
                    gap = baselines[1]['final_preference'] - baselines[-1]['final_preference']
                    after = trials[1]['final_preference'] - trials[-1]['final_preference']
                    effects.append(dict(**case, family=family, mode=mode, native_gap=gap,
                                        ablated_gap=after, contrast_reduction=gap-after))
            atomic_json(root / 'test_progress.json', dict(completed_cases=len({r['case_id'] for r in rows}), rows=rows))
        report = paired_report(rows, seed=config.seed, draws=config.bootstrap_draws)
        controls = []
        for family in ('dense', 'sparse'):
            for mode, dose in [('zero', 0.), ('steer', .5), ('steer', 1.), ('random', 1.), ('ablate', 1.), ('random_ablate', 1.)]:
                group = [r for r in rows if r['family'] == family and r['mode'] == mode and r['dose'] == dose]
                controls.append(dict(family=family, mode=mode, dose=dose,
                    mean_score=float(np.mean([r['score'] for r in group])),
                    harm_fraction=float(np.mean([r['harm'] for r in group])),
                    baseline_harm_fraction=float(np.mean([r['baseline_harm'] for r in group])),
                    mean_realized_relative_rms=float(np.mean([r['realized_relative_rms'] for r in group]))))
        report.update(protocol_hash=manifest['protocol_hash'], readout_attribution=attribution,
                      native_test_contrasts=test_contrasts, controls=controls,
                      selected_layers={f:h['layer'] for f,h in selected.items()}, ablation_effects=effects,
                      trials=rows, warnings=['Attribution is associative; steering and ablation test limited causal claims.',
                          'The sparse arm uses the existing transcoder, not a separately trained SAE.',
                          'Shared maximum RMS budget; sparse nonnegative constraints can reduce realized intervention size.',
                          'One prespecified random direction/feature set per layer is a control, not a random-control significance test.',
                          'Ablations are budget-capped suppression, not guaranteed full removal; per-call norms record the retained fraction.',
                          'Harm is the original-task proxy even for distractor-directed instructions; it is not an intent-aware safety label.',
                          'These task/initial states are held out from handle discovery, not necessarily from policy/transcoder pretraining.'])
        atomic_json(root / 'report.json', report)
        atomic_json(root / 'state.json', dict(status='complete', phase='test', report=str(root / 'report.json')))
        print(json.dumps({k:v for k,v in report.items() if k not in ('trials', 'ablation_effects')}, indent=2), flush=True)
        return report
    except BaseException as exc:
        atomic_json(root / 'failure.json', dict(error_type=type(exc).__name__, message=str(exc),
                                               note='Completed trials are preserved. Original traceback is raised in the caller.'))
        raise
    finally:
        runtime.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=['all', 'discovery', 'validation', 'test'], default='all')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    config = (SteeringConfig(train_tasks=(0,), validation_tasks=(1,), test_tasks=(2,),
                            train_starts=3, validation_starts=2, test_starts=2) if args.smoke else SteeringConfig())
    if args.plan_only:
        print(json.dumps(dict(config=asdict(config), cases=config.cases()), indent=2))
        return
    source_digest = hashlib.sha256((Path(__file__).read_bytes() +
        (Path(__file__).resolve().parents[1] / 'src/pi05_mi/matched_steering.py').read_bytes())).hexdigest()
    run_steering_experiment(config, args.checkpoint, args.output_dir, phase=args.phase,
                           device=args.device, source_digest=source_digest)


if __name__ == '__main__':
    main()
