"""Build the literal-cell Colab for deciding whether/how to retrain a transcoder."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from embed_pi05_transcoder_audit_notebook import code, markdown, definitions as audit_definitions, notebook as audit_notebook, MOUNT, CACHE

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'notebooks/pi05_transcoder_methodology_controls.ipynb'


def inline_method(path):
    lines = (ROOT/path).read_text().splitlines(keepends=True)
    remove = []
    for node in ast.walk(ast.parse(''.join(lines))):
        if isinstance(node, ast.ImportFrom) and (node.module == '__future__' or
            (node.module or '').startswith('pi05_mi') or
            node.module in ('run_pi05_transcoder_audit','probe_pi05_transcoder_counterfactual')):
            remove.append((node.lineno-1,node.end_lineno))
    for start, stop in sorted(remove,reverse=True): del lines[start:stop]
    return ''.join(lines)


def definitions():
    cells = audit_definitions()+[
        code('# @title Definitions 4 — Matched Fits And Decision Report\n'+inline_method('src/pi05_mi/transcoder_methodology.py'),'method-fit-definitions'),
        code('# @title Definitions 5 — Shared Teacher Collection And Evaluation\n'+inline_method('scripts/run_pi05_transcoder_methodology.py'),'method-run-definitions')]
    digest = hashlib.sha256(''.join(''.join(c['source']) for c in cells).encode()).hexdigest()
    cells[-1]['source'].append(f'\nMETHOD_NOTEBOOK_VERSION = {digest!r}\n')
    return cells


INTRO = '''# Should we retrain, and which training method should change?

This notebook makes **12 short layer-8 fits: four conditions × three seeds**, then compares them on the same held-out inputs. The existing checkpoint is an additional reference. It does not rerun bowl steering, fit instruction probes, or execute robot actions.

| Arm | Input to the policy denoiser | Time | Dictionary/training recipe | Main comparison |
|---|---|---|---|---|
| A | Pure noise | Training sampler | Original architecture/init/Adam | Reference collector distribution |
| B | Pure noise | Inference grid | Same as A | B−A: timestep sampling |
| C | Evolving denoising trajectory | Same grid | Same as A | C−B: action-input distribution |
| D | Same cached records as C | Same grid | Paper-informed adaptation below | D−C: recipe bundle |

A/B reuse each case's initial Gaussian draw at all requested times. They have identical observations/noise; only times differ. B/C have identical initial noise and grid. All arms receive equal record counts and optimizer updates. A/B/C start from byte-identical parameters for each paired seed. A is a controlled reconstruction of the collector distribution, **not a replay of historical training**: all three use a shared fixed train-only normalizer, uniform cached-record sampling, and float32 storage (the old trainer default was float16).

Default cohort: train tasks **0–4 × 4 starts × 2 noise draws = 40 cases**; validation tasks **5–6 × 3 starts × 2 = 12**; test tasks **7–9 × 3 starts × 2 = 18**. At 50 action tokens and 10 denoising steps, each training arm has **20,000 records**. Each fit receives **1,000 updates × 512 = 512,000 sampled records**, with replacement. Validation curves are diagnostic; all comparisons use the fixed final update, with **no test-based model selection**.

All reported nMSE values share a denominator computed from training trajectory outputs. Compare arms within this notebook; the old audit used a different reference variance. The initial-only corpus is intentionally small: repeated sampling does not create new scenes or establish convergence. Expect several GiB of saved records, model weights and resumable optimizer state on Drive.

All definitions are inside this notebook; experiments run in the current Python thread. Package installation uses system setup commands with live output. Use a high-RAM L4/A100. A changed runtime may require one restart. Budget checks happen between safe operations; a forward pass, checkpoint write or initialization can finish after the requested time. Resume with the same controls/run name. Partial runs do not release an unmatched arm comparison.
'''

RECIPE = '''## What this adopts from the paper—and what it does not

Reference: [DifFRACT, Appendix E](https://arxiv.org/html/2606.15796v1#A5), with separate attribution validation in Appendix K.3.

| Setting | A/B/C | D: paper-informed adaptation |
|---|---|---|
| Expansion/ReLU | Existing checkpoint dimensions | Same |
| Time network | Existing architecture | 256-wide, two SiLU stages, separate zero-initialized modulation projection |
| Initialization | Independent encoder/decoder; normalize after update | Encoder copied from decoder transpose; decoder normalized before training |
| AdamW | lr 1e−4, weight decay .01, constant rate | lr 2e−4, weight decay 0, cosine decay |
| Loss variance denominator | Shared frozen training-trajectory variance | Current minibatch population variance + 1e−6 |
| Decoder normalization | Every update | Every update |
| L1 coefficient | 1e−4 | 1e−4; VLA adaptation, not a paper stream-specific choice |
| Sampling | Uniform cached records, with replacement | Same |

**This is not a full paper reproduction.** It changes host model, dataset, sinusoidal embedding convention, sparsity coefficient, gradient clipping, minibatch size, buffer refresh strategy and training scale. D is a bundled comparison: it cannot isolate architecture from initialization or optimizer effects. A/B/C isolate the collector changes under the shared experimental recipe. These short runs cannot establish convergence or that further training is unnecessary.

## Read the evidence in this order

1. Check native-repeat identity, teacher/config/weight hashes, completed denominators, and learning curves.
2. Read C−B for trajectory collection, B−A for timestep sampling, and D−C for the adapted recipe. Negative reconstruction/action-error differences favor the left arm. Inspect all three fit seeds, task uncertainty and feature counts together.
3. Compare replacement to **zero and train-mean MLP-output controls**. If even these controls barely change actions, action error is an insensitive fidelity test here.
4. Improvement supports further training at this layer and budget. Flat or worsening results remain ambiguous if learning curves are improving or the new fits underperform the much longer-trained existing checkpoint.

Saved evidence includes per-time reconstruction, per-fit validation curves, intervention action arrays, controls, hashes and matched task-bootstrap differences. Three test tasks give imprecise uncertainty; seed results are reported separately. **Attribution, feature semantics, steering and safety remain untested.** A subsequent original-model feature-perturbation test is required before claiming circuit faithfulness.
'''

CONTROLS = '''# @title Controls
from pathlib import Path
DRIVE_ROOT = '/content/drive/MyDrive/groot-run-shared-programmer908' # @param {type:"string"}
CHECKPOINT = '' # @param {type:"string"}
RUN_NAME = 'transcoder-methodology-controls-001' # @param {type:"string"}
PHASE = 'all' # @param ["all", "collect", "train", "evaluate"]
LAYER = 8 # @param {type:"integer"}
FIT_SEEDS = '41,42,43' # @param {type:"string"}
UPDATES = 1000 # @param {type:"integer"}
BATCH_SIZE = 512 # @param {type:"integer"}
SESSION_MINUTES = 120 # @param {type:"integer"}
HF_OFFLINE = True # @param {type:"boolean"}
ALLOW_AUTH_REFRESH = True # @param {type:"boolean"}
POLICY_REPO = 'lerobot/pi05_libero_finetuned'
POLICY_REVISION = 'main' # resolves to snapshot; actual weight/config hashes frozen before collection
if not RUN_NAME or Path(RUN_NAME).name != RUN_NAME: raise ValueError('Use a single run-folder name')
'''

LAUNCH = '''# @title Run Or Resume In This Notebook Thread
from importlib.metadata import version
from IPython.display import display, Markdown
if 'METHOD_NOTEBOOK_VERSION' not in globals(): raise RuntimeError('Run all five Definitions cells first')
config = MethodConfig(layer=LAYER, fit_seeds=tuple(int(x) for x in FIT_SEEDS.split(',')),
                      updates=UPDATES, batch_size=BATCH_SIZE)
runtime_report = dict(packages={p:version(p) for p in ('torch','transformers','lerobot','hf-libero','mujoco','numpy')},
                      snapshots=snapshots, gpu=torch.cuda.get_device_name(0))
print('Definitions:',METHOD_NOTEBOOK_VERSION[:12])
print('Cases:',{s:sum(c['split']==s for c in config.cases()) for s in ('train','validation','test')})
print('Fits:',len(ARMS)*len(config.fit_seeds),'; records presented per fit:',UPDATES*BATCH_SIZE)
print('Persistent evidence:',RUN_DIR)
report = run_methodology(config,CHECKPOINT_PATH,RUN_DIR,source_digest=METHOD_NOTEBOOK_VERSION,
    runtime_report=runtime_report,policy_path=snapshots[POLICY_REPO],minutes=SESSION_MINUTES,phase=PHASE)
display(Markdown((RUN_DIR/'summary.md').read_text()))
'''

EXPORT = '''# @title Export Small Evidence Packet (No Multi-GB Training Files)
import zipfile
from IPython.display import display, FileLink
packet = Path('/content')/(RUN_NAME+'-evidence.zip')
with zipfile.ZipFile(packet,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(RUN_DIR.rglob('*')):
        if path.is_file() and (path.suffix in ('.json','.md') or path.name.endswith('_actions.npz')):
            archive.write(path,path.relative_to(RUN_DIR))
print('Raw activation records and trained weights remain on Drive:',RUN_DIR)
print('This packet contains denominators, per-time errors, learning curves, controls, and predictions.')
display(FileLink(str(packet)))
'''


def notebook():
    old = audit_notebook()
    installer = ''.join(next(c['source'] for c in old['cells'] if c['id']=='audit-install'))
    installer = installer.replace('if PHASE == "all":','if True:')
    mount = MOUNT.replace("'outputs/transcoder-audit'","'outputs/transcoder-methodology'")
    cache = CACHE.replace("if PHASE == 'all':",'if True:')
    cells = [markdown(INTRO,'method-intro'),markdown(RECIPE,'method-recipe'),code(CONTROLS,'method-controls'),
        code(mount,'method-mount'),code(installer,'method-install'),code(cache,'method-cache'),
        *definitions(),code(LAUNCH,'method-launch'),code(EXPORT,'method-export')]
    return dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),colab=dict(name=DEST.name),accelerator='GPU'),nbformat=4,nbformat_minor=5)


if __name__ == '__main__':
    DEST.write_text(json.dumps(notebook(),indent=1)+'\n')
    print(DEST)
