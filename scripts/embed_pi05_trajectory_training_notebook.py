"""Generate the live, main-thread trajectory training notebook."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path

from embed_pi05_transcoder_audit_notebook import code,markdown,CACHE
from embed_pi05_methodology_notebook import notebook as method_notebook,definitions as method_definitions

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'notebooks/pi05_transcoder_trajectory_training.ipynb'


def inline_training(path):
    lines=(ROOT/path).read_text().splitlines(keepends=True);remove=[]
    for node in ast.walk(ast.parse(''.join(lines))):
        if isinstance(node,ast.ImportFrom) and (node.module=='__future__' or
            (node.module or '').startswith('pi05_mi') or
            node.module in ('run_pi05_transcoder_methodology','probe_pi05_transcoder_counterfactual')):
            remove.append((node.lineno-1,node.end_lineno))
    for start,stop in sorted(remove,reverse=True): del lines[start:stop]
    return ''.join(lines)


def definitions():
    cells=method_definitions()+[
        code('# @title Definitions 6 — Training Recipe, Dashboard, Durable Checkpoints\n'+inline_training('src/pi05_mi/trajectory_training.py'),'trajectory-core'),
        code('# @title Definitions 7 — Dataset Trajectories And Training Loop\n'+inline_training('scripts/run_pi05_trajectory_training.py'),'trajectory-runner')]
    digest=hashlib.sha256(''.join(''.join(c['source']) for c in cells).encode()).hexdigest()
    cells[-1]['source'].append(f'\nTRAJECTORY_TRAINING_VERSION = {digest!r}\n')
    return cells


INTRO='''# Pi0.5 transcoder training — real inference trajectories, live metrics

Train a **fresh set of 18 action-expert transcoders** using the paper-informed recipe that improved the controlled pilot. Each teacher call captures all requested layers; dictionaries are optimized one at a time to limit GPU memory. This uses recorded LIBERO observations throughout episodes, with fresh Gaussian seeds and real 10-step denoising trajectories. No robot actions are executed.

**Run all cells on a high-RAM L4/A100.** Set `LAYER_IDS="8"` for a smaller single-layer run. The existing 7.39-GiB checkpoint is neither required nor overwritten. All experiment code is embedded below and executes in this notebook thread; external commands are used only for dependency installation.

| Default | Value |
|---|---:|
| Layers | 0–17 |
| Fresh activation-buffer cycles | 256 |
| Observation frames per cycle | 64 |
| Denoising calls / frame | 10 |
| Activation records / frame / layer | 500 for the current 50-token policy |
| Records per fresh buffer / layer | 32,000 |
| Optimizer batch | 1,024 records, sampled with replacement |
| Updates per cycle / layer | 32 |
| Total sampled records / layer | 8,388,608 |
| Train / validation / test split | 80 / 10 / 10% by episode |
| Validation frames / final test frames | 24 / 16, fixed before training |

This is a larger training run, **not a two-hour completion promise**. The dashboard estimates remaining time from measured cycle throughput. Session budgets pause work at safe boundaries and save resumable state; checkpoint upload and an in-flight operation may exceed the timer. Setup, model loading and dataset hashing occur before the training-session timer. Use the same run name and controls to resume. Runtime or protocol changes require a new run name.

## Recipe and evaluation labels

Reference: [DifFRACT Appendix E](https://arxiv.org/html/2606.15796v1#A5). We use trajectory harvesting, 16× ReLU dictionaries, the adapted 256-wide time network, decoder-based encoder initialization, unit decoder columns, AdamW with zero weight decay, and cosine decay. Loss: minibatch-variance-normalized reconstruction plus λ times L1 feature activity. The VLA-specific embedding, λ=1e−4, clipping, smaller buffers/batches and total training budget remain adaptations—not an exact paper reproduction.

**Fidelity is not attribution.** Reconstruction error, sparsity and normalized-action error are labelled separately. No metric here establishes semantic features, correct circuit tracing, successful steering, task success or collision prevention. Episode holdout does not imply unseen tasks or exclusion from policy pretraining.

Best measured **validation nMSE per layer** selects exported weights. Final test frames never affect optimization or selection. The final report measures each layer separately; it does not certify simultaneous replacement of all layers.
'''

CONTROLS='''# @title Controls — Freeze These For A Resumable Run
from pathlib import Path
DRIVE_ROOT = '/content/drive/MyDrive/groot-run-shared-programmer908' # @param {type:"string"}
RUN_NAME = 'pi05-trajectory-transcoders-001' # @param {type:"string"}
LAYER_IDS = ','.join(str(i) for i in range(18)) # @param {type:"string"}
CYCLES = 256 # @param {type:"integer"}
FRAMES_PER_CYCLE = 64 # @param {type:"integer"}
TRAIN_BATCH_SIZE = 1024 # @param {type:"integer"}
LAMBDA_L1 = 0.0001 # @param {type:"number"}
VALIDATE_EVERY_CYCLES = 8 # @param {type:"integer"}
SESSION_MINUTES = 120 # @param {type:"integer"}
CHECKPOINT_EVERY_MINUTES = 30 # @param {type:"integer"}
DATASET_REPO = 'HuggingFaceVLA/libero' # @param {type:"string"}
DATASET_ROOT = '' # @param {type:"string"}
ALLOW_DATASET_DOWNLOAD = True # @param {type:"boolean"}
HF_OFFLINE = True # @param {type:"boolean"}
ALLOW_AUTH_REFRESH = True # @param {type:"boolean"}
POLICY_REPO = 'lerobot/pi05_libero_finetuned'
POLICY_REVISION = 'main'
if not RUN_NAME or Path(RUN_NAME).name!=RUN_NAME: raise ValueError('Use a single run-folder name')
'''

MOUNT='''# @title Mount Drive — New Output Folder, Existing Checkpoints Preserved
from google.colab import drive
if not Path('/content/drive/MyDrive').exists(): drive.mount('/content/drive')
DRIVE_ROOT = Path(DRIVE_ROOT)
if not DRIVE_ROOT.is_dir(): raise FileNotFoundError(DRIVE_ROOT)
RUN_DIR = DRIVE_ROOT/'outputs/trajectory-training'/RUN_NAME
RUN_DIR.mkdir(parents=True,exist_ok=True)
print('Persistent training output:',RUN_DIR)
print('Resume pointer:',RUN_DIR/'latest.json')
'''

EXTRA='''# @title Ensure Charts And Video Decoder Dependencies
import importlib.util, subprocess, sys
missing=[p for p in ('matplotlib','av') if importlib.util.find_spec(p) is None]
if missing:
    subprocess.run([sys.executable,'-m','pip','install',*missing],check=True)
import matplotlib
matplotlib.use('Agg')
print('Chart backend:',matplotlib.get_backend(),'; video decoder: PyAV')
'''

DATASET='''# @title Open The Full Observation Dataset (No DataLoader Worker Processes)
import os,shutil
from huggingface_hub import constants as hf_constants
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
if DATASET_ROOT.strip():
    dataset_root=Path(DATASET_ROOT).expanduser()
else:
    dataset_root=Path('/content/lerobot-data')/DATASET_REPO
    candidates=[Path(HF_LEROBOT_HOME)/DATASET_REPO,Path('/content/hf_home/lerobot')/DATASET_REPO,
                DRIVE_ROOT/'hf_home/lerobot'/DATASET_REPO,DRIVE_ROOT/'libero_cache'/DATASET_REPO]
    if not (dataset_root/'meta/info.json').exists():
        cached=next((p for p in candidates if (p/'meta/info.json').is_file()),None)
        if cached is not None and cached.resolve()!=dataset_root.resolve():
            print('Staging cached dataset on local disk:',cached,flush=True)
            dataset_root.mkdir(parents=True,exist_ok=True)
            for name in ('meta','data','videos','images'):
                if (cached/name).is_dir(): shutil.copytree(cached/name,dataset_root/name,dirs_exist_ok=True)
print('Dataset:',DATASET_REPO,'root:',dataset_root,flush=True)
old_offline=hf_constants.HF_HUB_OFFLINE
old_env={k:os.environ.get(k) for k in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE')}
try:
    if ALLOW_DATASET_DOWNLOAD:
        hf_constants.HF_HUB_OFFLINE=False
        for key in old_env: os.environ.pop(key,None)
    dataset=LeRobotDataset(DATASET_REPO,root=dataset_root,video_backend='pyav',delta_timestamps=None)
finally:
    hf_constants.HF_HUB_OFFLINE=old_offline
    for key,value in old_env.items():
        if value is None: os.environ.pop(key,None)
        else: os.environ[key]=value
print('Dataset frames:',len(dataset),'; episodes:',dataset.meta.total_episodes)
print('Camera keys:',dataset.meta.camera_keys)
print('No simulator rollout, occlusion masks, or bowl inference is used for training.')
'''

LAUNCH='''# @title Train / Resume — Live Dashboard Below, Native Tracebacks Preserved
from importlib.metadata import version
from IPython.display import display,Markdown
import gc
if 'TRAJECTORY_TRAINING_VERSION' not in globals(): raise RuntimeError('Run all seven Definitions cells first')
if not isinstance(globals().get('snapshots'),dict) or any(
    not globals()['snapshots'].get(repo) or not Path(globals()['snapshots'][repo]).is_dir()
    for repo in (POLICY_REPO,'google/paligemma-3b-pt-224','lerobot/libero-assets')
):
    raise RuntimeError('Model cache setup is missing or incomplete in this session. '
        'Run "Prepare Model Cache And LIBERO Assets", then rerun this Train / Resume cell. '
        'After a runtime restart, rerun setup cells even when the files remain cached. '
        'No training has started and existing checkpoints are unchanged.')
if 'dataset' not in globals():
    raise RuntimeError('Run "Open The Full Observation Dataset" before this Train / Resume cell.')
config=TrajectoryTrainConfig(layers=tuple(int(x.strip()) for x in LAYER_IDS.split(',') if x.strip()),
    cycles=CYCLES,frames_per_cycle=FRAMES_PER_CYCLE,batch_size=TRAIN_BATCH_SIZE,
    lambda_l1=LAMBDA_L1,validate_every=VALIDATE_EVERY_CYCLES)
config.validate()
print('Notebook definitions:',TRAJECTORY_TRAINING_VERSION[:12])
print('Hashing teacher and dataset bytes to freeze training provenance...',flush=True)
provenance=dict(source_digest=TRAJECTORY_TRAINING_VERSION,
    teacher=snapshot_evidence(snapshots[POLICY_REPO]),dataset=fingerprint_dataset(dataset),
    packages={p:version(p) for p in ('torch','transformers','lerobot','hf-libero','numpy','matplotlib','av')},
    snapshots=snapshots,gpu=torch.cuda.get_device_name(0))
source=None
try:
    source=DatasetTrajectorySource(dataset,snapshots[POLICY_REPO],config)
    audit_json(RUN_DIR/'teacher_load.json',source.teacher_load)
    result=run_trajectory_training(config,source,dataset_episodes(dataset),RUN_DIR,
        '/content/pi05-trajectory-training',provenance,minutes=SESSION_MINUTES,
        checkpoint_minutes=CHECKPOINT_EVERY_MINUTES)
    print('Session result:',result['status'])
    print('Metrics:',RUN_DIR/'metrics.jsonl')
    print('Saved dashboard:',RUN_DIR/'live_training.png')
finally:
    if source is not None: source.close()
    del source;gc.collect();torch.cuda.empty_cache()
'''

EXPORT='''# @title Inspect Checkpoint Status And Download Small Training Evidence
import json,zipfile
from IPython.display import display,FileLink,Image
for name in ('live_status.json','latest.json','training_summary.json','failure.json'):
    path=RUN_DIR/name
    if path.exists():
        value=json.loads(path.read_text())
        print(name,':',value if name not in ('training_summary.json',) else {k:v for k,v in value.items() if k!='test'})
if (RUN_DIR/'live_training.png').exists(): display(Image(filename=str(RUN_DIR/'live_training.png'),width=1200))
packet=Path('/content')/(RUN_NAME+'-evidence.zip')
with zipfile.ZipFile(packet,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in RUN_DIR.rglob('*'):
        if 'checkpoints' not in path.relative_to(RUN_DIR).parts and path.is_file() and path.suffix in ('.json','.jsonl','.png'):
            archive.write(path,path.relative_to(RUN_DIR))
display(FileLink(str(packet)))
print('Weights and optimizer shards remain on Drive; only small evidence is zipped.')
print('Complete exports: trained_transcoders.pt — use load_trajectory_bundle(path), which handles the new time-network architecture.')
'''


def notebook():
    old=method_notebook()
    installer=''.join(next(c['source'] for c in old['cells'] if c['id']=='method-install'))
    installer=installer.replace('Select a high-RAM L4/A100 GPU runtime, or use PHASE=metadata.',
                                'Select a high-RAM L4/A100 GPU runtime for training.')
    # Installer came from the proven pinned Colab runtime; the experiment itself never starts a worker.
    cache=CACHE.replace("if PHASE == 'all':",'if True:')
    cache=cache.replace('Prepare Model Cache And LIBERO Assets (Skipped For Metadata Only)',
                        'Prepare Model Cache And LIBERO Assets — Run After Every Runtime Restart')
    cache=cache.replace('Select a GPU runtime for PHASE=all','Select a GPU runtime for training')
    cells=[markdown(INTRO,'trajectory-intro'),code(CONTROLS,'trajectory-controls'),code(MOUNT,'trajectory-mount'),
        code(installer,'trajectory-install'),code(EXTRA,'trajectory-extra-install'),code(cache,'trajectory-cache'),
        code(DATASET,'trajectory-dataset'),*definitions(),code(LAUNCH,'trajectory-launch'),code(EXPORT,'trajectory-export')]
    return dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),colab=dict(name=DEST.name),accelerator='GPU'),nbformat=4,nbformat_minor=5)


if __name__=='__main__':
    DEST.write_text(json.dumps(notebook(),indent=1)+'\n');print(DEST)
