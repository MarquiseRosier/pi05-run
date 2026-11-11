"""Real task definitions and small MuJoCo scenes exercise instance isolation."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from pi05_mi.occlusion_pilot import target_bowl_spec, task_bddl_path, ImageOccluder
from run_pi05_occlusion_pilot import resolve_bowl_instances, bowl_masks, write_images, Slot, PhysicsRecorder

FIXTURES = Path(__file__).parent / 'fixtures/libero_spatial'


def inner(path):
    # Exact LeRobot 0.4.4 -> OffScreenRenderEnv -> BDDLBaseDomain nesting.
    return NS(_env=NS(env=NS(bddl_file_name=str(path))))


@pytest.mark.parametrize('path', sorted(FIXTURES.glob('*.bddl')), ids=lambda p: p.stem)
def test_all_packaged_spatial_tasks_resolve_the_goal_bowl(path):
    spec = target_bowl_spec(inner(path))
    assert spec['target_instance'] == 'akita_black_bowl_1'
    assert spec['distractor_instances'] == ['akita_black_bowl_2']
    assert spec['goal'] == [['And', ['On', 'akita_black_bowl_1', 'plate_1']]]
    assert len(spec['bddl_sha256']) == 64


def task_text(goal='(On bowl_2 plate_1)', interest='bowl_1 bowl_2 plate_1'):
    return f'''(define (problem test)
    (:objects bowl_1 bowl_2 - black_bowl plate_1 - plate)
    (:obj_of_interest {interest})
    ; Deliberately list the other bowl first; only the goal determines the target.
    (:goal (And {goal})))'''


def test_goal_subject_wins_over_object_order_and_stale_wrapper_path(tmp_path):
    path = tmp_path / 'task.bddl'; path.write_text(task_text())
    env = inner(path); env._task_bddl_file = '/stale/python3.12/task.bddl'
    spec = target_bowl_spec(env)
    assert spec['target_instance'] == 'bowl_2'
    assert spec['distractor_instances'] == ['bowl_1']


@pytest.mark.parametrize('text, message', [
    (task_text('(On bowl_1 plate_1) (On bowl_2 plate_1)'), 'one goal bowl'),
    (task_text('(On plate_1 bowl_1)'), 'one goal bowl'),
    (task_text(interest='bowl_1'), 'one goal bowl'),
    (task_text('(Or (On bowl_1 plate_1) (On bowl_2 plate_1))'), 'Unsupported'),
    (task_text()[:-1], 'balanced'),
])
def test_ambiguity_missing_interest_and_unsupported_goals_fail(tmp_path, text, message):
    path = tmp_path / 'task.bddl'; path.write_text(text)
    with pytest.raises((RuntimeError, ValueError), match=message):
        target_bowl_spec(inner(path))


def test_missing_task_path_is_reported_not_an_empty_candidate_list(tmp_path):
    with pytest.raises(FileNotFoundError, match='Active LIBERO task BDDL is missing'):
        task_bddl_path(inner(tmp_path / 'missing.bddl'))
    with pytest.raises(RuntimeError, match='exposes no task BDDL path'):
        task_bddl_path(NS())


def scene():
    import mujoco
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="target_root"><freejoint/><geom size=".02"/>
      <body name="target_child"><geom size=".01" pos="0 0 .02"/></body></body>
    <body name="other_root" pos=".2 0 0"><freejoint/><geom size=".02"/></body>
    </worldbody></mujoco>''')
    env = NS(sim=NS(model=NS(_model=model)), objects_dict={
        'bowl_2': NS(root_body='target_root'), 'bowl_1': NS(root_body='other_root')})
    spec = dict(target_instance='bowl_2', distractor_instances=['bowl_1'])
    return env, spec


def test_registry_resolves_exact_instances_and_descendant_geoms():
    env, spec = scene()
    instances = resolve_bowl_instances(env, spec)
    assert len(instances['bowl_2']['geom_ids']) == 2
    assert len(instances['bowl_1']['geom_ids']) == 1
    env.objects_dict['bowl_1'].root_body = 'target_root'
    with pytest.raises(RuntimeError, match='disjoint geometry'):
        resolve_bowl_instances(env, spec)
    env.objects_dict.pop('bowl_1')
    with pytest.raises(RuntimeError, match='no registered simulator body'):
        resolve_bowl_instances(env, spec)


def test_observation_masks_follow_rgb_orientation_and_initial_visibility(monkeypatch):
    import mujoco
    from types import ModuleType
    for name in ('robosuite', 'robosuite.utils', 'robosuite.utils.mjcf_utils',
                 'probe_pi05_transcoder_counterfactual'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules['robosuite'].macros = NS(IMAGE_CONVENTION='opencv')
    sys.modules['robosuite.utils.mjcf_utils'].IMAGE_CONVENTION_MAPPING = {'opencv': -1}
    sys.modules['probe_pi05_transcoder_counterfactual']._add_batch_axis = lambda x: x
    env, spec = scene()
    instances = resolve_bowl_instances(env, spec)
    seg = np.full((40, 40, 2), -1, np.int32)
    seg[2:5, 3:7] = [int(mujoco.mjtObj.mjOBJ_GEOM), instances['bowl_2']['geom_ids'][0]]
    rgb = np.full((40, 40, 3), 80, np.uint8)
    rgb[2:5, 3:7] = [255, 0, 0]
    raw = {name: np.zeros(3) for name in ('robot0_eef_pos', 'robot0_eef_quat',
                                        'robot0_gripper_qpos', 'robot0_joint_vel')}
    env._get_observations = lambda **kwargs: raw
    env.sim.render = lambda **kwargs: seg[::-1]
    env.sim.get_state = lambda: np.zeros(4)
    slot = object.__new__(Slot)
    slot.env, slot.instances, slot.target_instance = env, instances, 'bowl_2'
    slot.inner = NS(camera_name_mapping={'agentview_image': 'image'},
                    _format_raw_obs=lambda raw: {'pixels': {'image': rgb}})
    slot.physics = NS(mj=mujoco, target_geoms=set(instances['bowl_2']['geom_ids']))
    slot.occluder, slot.steps, slot.last_action = ImageOccluder('box'), 0, np.zeros(7)
    _, _, visibility, _, audits = slot.observe()
    masks, cutout = audits['image']
    np.testing.assert_array_equal(cutout[:, :, 3] > 0, masks['bowl_2__before'])
    assert (cutout[2:5, 3:7, :3] == [255, 0, 0]).all()
    assert visibility['image']['target_pixels_before'] == 12
    assert visibility['image']['target_pixels_after'] == 0
    seg[:] = -1
    with pytest.raises(RuntimeError, match='no visible pixels in any initial camera'):
        slot.observe()
    slot.steps = 1
    assert slot.observe()[2]['image']['fraction_remaining'] is None
    # A failed control at a replanning step must persist even between monitor ticks.
    seg[:] = [int(mujoco.mjtObj.mjOBJ_GEOM), instances['bowl_2']['geom_ids'][0]]
    slot.occluder, slot.steps = ImageOccluder('miss_box'), 50
    slot.job, slot.occlusion_failures = {'episode_id': 'test_miss_box'}, []
    _, images, visibility, _, _ = slot.observe()
    np.testing.assert_array_equal(images['image'], rgb)
    assert visibility['image']['placement_status'] == 'no_same_size_miss'
    assert slot.occlusion_failures == [{'step': 50, 'camera': 'image', 'reason': 'no_same_size_miss'}]


def test_physics_labels_use_registered_distractor_even_without_bowl_in_body_name():
    import mujoco
    from pi05_mi.occlusion_pilot import PilotConfig
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="robot0_hand"><geom size=".01"/></body>
    <body name="target"><freejoint/><geom size=".02"/></body>
    <body name="other" pos=".2 0 0"><freejoint/><geom size=".02"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    sim = NS(model=NS(_model=model), data=NS(_data=data), step=lambda: None)
    other = model.body('other').id
    recorder = PhysicsRecorder(NS(sim=sim), 'target', PilotConfig(), [other])
    assert recorder.wrong_bowls == [other]
    assert not recorder.movable.intersection(recorder.target_geoms)
    data.xpos[other, 2] += .06
    assert recorder.finish_control_step(1)['harm']
    recorder.close()


def test_masks_cutouts_and_saved_annotations_match_overlay(tmp_path):
    import mujoco
    from PIL import Image
    env, spec = scene()
    instances = resolve_bowl_instances(env, spec)
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    seg = np.full((40, 40, 2), -1, dtype=np.int32)
    seg[10:15, 10:15] = [geom_type, instances['bowl_2']['geom_ids'][0]]
    seg[12:17, 18:23] = [geom_type, instances['bowl_1']['geom_ids'][0]]
    # A non-geom with the same integer ID must not be included.
    seg[0, 0] = [int(mujoco.mjtObj.mjOBJ_SITE), instances['bowl_2']['geom_ids'][0]]
    masks, _ = bowl_masks(seg, instances, 'bowl_2', None, geom_type)
    target = masks['bowl_2__before']
    rgb = np.full((40, 40, 3), 80, dtype=np.uint8)
    overlay, info = ImageOccluder('box').apply('image', rgb, target)
    masks, counts = bowl_masks(seg, instances, 'bowl_2', info['rectangle'], geom_type)
    assert masks['bowl_2__before'].sum() == 25
    assert masks['bowl_2__after'].sum() == 0
    assert not (masks['bowl_2__before'] & masks['bowl_1__before']).any()
    assert counts['bowl_1']['overlay_overlap_pixels'] > 0  # Explicitly reveal collateral hiding.
    rgba = np.zeros((40, 40, 4), np.uint8)
    rgba[target, :3] = rgb[target]; rgba[target, 3] = 255
    write_images(tmp_path, {'image': overlay}, {'image': (masks, rgba)})
    with np.load(tmp_path / 'image_bowl_masks.npz') as saved:
        for name, mask in masks.items():
            np.testing.assert_array_equal(saved[name], mask)
    cutout = np.asarray(Image.open(tmp_path / 'image_target.png'))
    np.testing.assert_array_equal(cutout[:, :, 3] > 0, target)
    assert (cutout[~target] == 0).all()
