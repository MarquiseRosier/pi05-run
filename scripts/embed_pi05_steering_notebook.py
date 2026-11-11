"""Maintain literal notebook experiment cells. This generator is never run in Colab."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'notebooks/pi05_matched_steering.ipynb'


def code(source, cell_id):
    return dict(cell_type='code', id=cell_id, metadata={}, execution_count=None, outputs=[],
                source=source.splitlines(keepends=True))


def inline_source(path):
    source = path.read_text(); lines = source.splitlines(keepends=True)
    remove = []
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.ImportFrom) and node.module in
                ('pi05_mi.matched_steering', 'pi05_mi.occlusion_pilot', 'run_pi05_occlusion_pilot')
            or isinstance(node, ast.FunctionDef) and node.name == 'main'
            or isinstance(node, ast.If) and '__name__' in ast.unparse(node.test)
            or isinstance(node, ast.Assign) and any(ast.unparse(t).startswith('sys.path') for t in node.targets)):
            remove.append((node.lineno - 1, node.end_lineno))
    for first, last in sorted(remove, reverse=True):
        lines[first:last] = []
    return ''.join(lines)


def definition_cells():
    helpers = 'import re\nimport os\nimport json\nimport hashlib\nfrom pathlib import Path\nimport numpy as np\n\n'
    for file, names in [
        ('src/pi05_mi/occlusion_pilot.py', {'task_bddl_path','bddl_sections','target_bowl_spec','atomic_json'}),
        ('scripts/run_pi05_occlusion_pilot.py', {'descend_geoms','PhysicsRecorder'})]:
        text = (ROOT / file).read_text()
        for node in ast.parse(text).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                helpers += ast.get_source_segment(text, node) + '\n\n'
    cells = [code('# @title Definitions: Task Identity And Contact Measurements\n' + helpers, 'steering-helpers'),
             code('# @title Definitions: Matched Handles And Scoring\n' + inline_source(ROOT / 'src/pi05_mi/matched_steering.py'), 'steering-core'),
             code('# @title Definitions: Discovery, Validation, And Test Runner\n' + inline_source(ROOT / 'scripts/run_pi05_matched_steering.py'), 'steering-runner')]
    digest = hashlib.sha256(''.join(''.join(c['source']) for c in cells).encode()).hexdigest()
    cells[-1]['source'].append(f'\nSTEERING_NOTEBOOK_VERSION = {digest!r}\n')
    return cells


PROTOCOL = '''# Matched movement steering: linear probe versus sparse transcoder

This experiment asks whether handles discovered from contrasting activations can **cause a requested change in actual robot movement**. The available checkpoint is a timestep-conditioned transcoder, not a sparse autoencoder. We compare a dense linear-probe direction with a small set of native sparse features at the same action-expert MLP output.

1. **Opposite examples:** on identical images and initial simulator states, instruct Pi0.5 to pick the goal bowl or the distractor bowl. Both instructions come from LIBERO descriptions of the actual bowl locations. Record actual gripper movement and matched dense/sparse activations. Keep training pairs only if their native movement contrast exceeds the prespecified threshold; never filter validation/test by their outcomes.
2. **Discovery:** fit a standardized dense linear probe; rank sparse features by training-only standardized activation contrast and fit a sparse readout on the selected units. Pool action tokens and denoising calls identically. Validation AUC measures prediction, not causal importance.
3. **Validation:** intervene in both directions at candidate layers. Choose ONE shared layer by the mean movement effect of both arms. Freeze layer, features, directions, dose and test IDs before opening test cases.
4. **Held-out test:** execute the first 50 predicted actions in the simulator, with zero, half/full dose, random, budget-capped ablation and random-ablation interventions. Every condition constructs a fresh simulator with its seed set before construction, bypasses LeRobot’s success-triggered auto-reset for a fixed measurement horizon, uses identical starting pixels and noise, and preserves the native MLP plus an intervention delta. Sparse latent edits remain nonnegative. Both methods have the same per-token maximum output-RMS budget; report realized norms because sparse constraints may limit spending.
5. **Primary score:** let `P = (distance_to_distractor - distance_to_target) / bowl_separation`, using fixed initial bowl positions and the final measured gripper position. Score = `50 × desired_sign × (P_intervened − P_baseline)`. Report **sparse score − dense score**, averaging both directions, weighting tasks equally, with a paired task-bootstrap interval when at least three test tasks exist. Also report measured XYZ displacement, force/lift proxies relative to the original task, and whether capped ablation reduces the native prompt contrast. A distractor-directed instruction may intentionally lift that bowl, so the original-task harm proxy is not an intent-aware safety label.

Full split: tasks 0–4 train (10 starts each), 5–6 validation (3 each), 7–9 test (5 each). Both prompt variants and every intervention stay in the same task split. Candidate layers default to 0, 8, 17. Full defaults require about **574 single-chunk simulator trials**; smoke requires 86 and is only an infrastructure check. There is no 15-minute cutoff that prevents reaching the test stage. Each completed trial is resumable. Initial state arrays and camera pixels are checked before inference and saved per case; any mismatch writes reset diagnostics and stops before policy work. The reset-v2 protocol requires a new output folder to keep older reused-simulator trials separate. Interrupting the cell keeps completed trials; rerun with identical controls to resume.

These tests establish limited causal evidence for a handle, not proof of a uniquely represented concept, long-horizon task control, or safety. One fixed random control is not a random-control significance test. The existing policy/transcoder may have seen test tasks during pretraining; the split holds out handle discovery and selection. Do not tune settings after inspecting test scores and reuse the same test set as if untouched.

Setup commands install packages and prepare caches. **All experiment definitions are in the notebook; discovery, fitting and simulation execute directly in this kernel with native tracebacks.** No Python worker runs the experiment. Use a new run name after changing the protocol. Keep the existing occlusion experiment on its original branch.
'''


LAUNCH = '''# @title Run Or Resume Matched Steering In This Kernel
from IPython.display import display, Markdown
sys.path[:0] = [str(LOCAL_REPO / "scripts"), str(LOCAL_REPO / "src")]
from resolve_pi05_monitor_checkpoint import resolve_checkpoint
if "STEERING_NOTEBOOK_VERSION" not in globals():
    raise RuntimeError("Run the three experiment definition cells above first.")
if Path(RUN_NAME).name != RUN_NAME:
    raise ValueError("RUN_NAME must be one folder name")
CHECKPOINT = str(resolve_checkpoint(CHECKPOINT, DRIVE_ROOT, LOCAL_REPO))
layers = tuple(map(int, CANDIDATE_LAYERS.split(',')))
kwargs = dict(layers=layers, sparse_k=int(SPARSE_FEATURES), budget=float(RMS_BUDGET))
if STUDY_SIZE == "smoke":
    kwargs.update(train_tasks=(0,), validation_tasks=(1,), test_tasks=(2,),
                  train_starts=3, validation_starts=2, test_starts=2)
config = SteeringConfig(**kwargs)
RUN_DIR = DRIVE_OUTPUTS / "matched-steering" / RUN_NAME
print("Checkpoint:", CHECKPOINT)
print("Notebook experiment:", STEERING_NOTEBOOK_VERSION[:12])
print("Persistent output:", RUN_DIR)
print("Cases:", {s: sum(c['split'] == s for c in config.cases()) for s in ('train','validation','test')})
report = run_steering_experiment(config, CHECKPOINT, RUN_DIR,
    source_digest=STEERING_NOTEBOOK_VERSION, phase=PHASE)
if report:
    display(Markdown(f"Linear probe: **{report['dense_score']:.3f}** points. "
                     f"Sparse transcoder: **{report['sparse_score']:.3f}** points. "
                     f"Paired difference (sparse − dense): **{report['sparse_minus_dense']:.3f}**. "
                     f"95% task-bootstrap interval: {report['task_bootstrap_95ci']}."))
'''


def notebook():
    base = json.loads((ROOT / 'notebooks/pi05_occlusion_monitor_pilot.ipynb').read_text())
    controls = ''.join(base['cells'][1]['source']).split('# Use a fresh name')[0]
    controls = controls.replace('feature/marquise-occlusion-monitor-pilot', 'feature/marquise-matched-steering')
    controls += '''INFERENCE_BATCH_SIZE = 1
RUN_NAME = "matched-steering-reset-002" # @param {type:"string"}
STUDY_SIZE = "full" # @param ["smoke", "full"]
PHASE = "all" # @param ["discovery", "validation", "test", "all"]
CANDIDATE_LAYERS = "0,8,17" # @param {type:"string"}
SPARSE_FEATURES = 16 # @param {type:"integer"}
RMS_BUDGET = 0.1 # @param {type:"number"}
CHECKPOINT = "" # @param {type:"string"}
'''
    setup = copy.deepcopy(base['cells'][2:7])
    for cell in setup:
        cell['execution_count'], cell['outputs'] = None, []
    cells = [dict(cell_type='markdown', metadata={}, source=PROTOCOL.splitlines(keepends=True)),
             code(controls, 'steering-controls'), *setup, *definition_cells(), code(LAUNCH, 'steering-launch'),
             code('''# @title Inspect Saved Results Without Rerunning
for name in ("native_contrasts.json", "discovery.json", "frozen_selection.json", "test_progress.json", "report.json"):
    path = RUN_DIR / name
    if path.exists():
        data = json.loads(path.read_text())
        print(name)
        print(json.dumps({k:v for k,v in data.items() if k not in ("handles", "selected", "rows", "trials", "validation_trials")}, indent=2))
''', 'steering-inspect')]
    return dict(nbformat=4, nbformat_minor=5, metadata=base['metadata'], cells=cells)


if __name__ == '__main__':
    DEST.write_text(json.dumps(notebook(), indent=1, ensure_ascii=False) + '\n')
