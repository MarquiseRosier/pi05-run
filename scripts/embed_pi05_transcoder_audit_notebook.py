"""Generate a self-contained Colab audit; never runs the experiment in a worker."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'notebooks/pi05_transcoder_training_audit.ipynb'


def code(source,cell_id):
    return dict(cell_type='code',id=cell_id,metadata={},execution_count=None,outputs=[],source=source.splitlines(keepends=True))


def markdown(source,cell_id):
    return dict(cell_type='markdown',id=cell_id,metadata={},source=source.splitlines(keepends=True))


def inline(path):
    source=(ROOT/path).read_text(); lines=source.splitlines(keepends=True); remove=[]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.ImportFrom) and (node.level or node.module=='__future__' or
            (node.module or '').startswith('pi05_mi') or node.module=='probe_pi05_transcoder_counterfactual'):
            remove.append((node.lineno-1,node.end_lineno))
    for start,stop in sorted(remove,reverse=True): lines[start:stop]=[]
    return ''.join(lines)


def definitions():
    helpers='import argparse\nimport logging\nimport os\nfrom dataclasses import dataclass\nfrom pathlib import Path\nfrom typing import Any, Callable\nimport numpy as np\nimport torch\n\n'
    source=(ROOT/'scripts/probe_pi05_transcoder_counterfactual.py').read_text()
    lines=source.splitlines(keepends=True)
    for node in ast.parse(source).body:
        if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in {
            'Harness','ensure_libero_config','build_harness','_add_batch_axis','_infer_prompt'}:
            start=min([node.lineno]+[d.lineno for d in node.decorator_list])-1
            helpers+=''.join(lines[start:node.end_lineno])+'\n\n'
    cells=[
        code('# @title Definitions 1 — Exact Dictionary And Policy Guards\n'+inline('src/pi05_mi/transcoders.py')+'\n'+inline('src/pi05_mi/pi05_weights.py')+'\n'+inline('src/pi05_mi/patch_pi05.py'),'audit-model-definitions'),
        code('# @title Definitions 2 — Streaming Statistics And Checkpoint Evidence\n'+inline('src/pi05_mi/transcoder_audit.py'),'audit-statistics'),
        code('# @title Definitions 3 — Simulator Snapshots And Direct Audit Runner\n'+helpers+inline('scripts/run_pi05_transcoder_audit.py'),'audit-runner')]
    digest=hashlib.sha256(''.join(''.join(c['source']) for c in cells).encode()).hexdigest()
    cells[-1]['source'].append(f'\nAUDIT_NOTEBOOK_VERSION = {digest!r}\n')
    return cells


INTRO='''# Did the transcoder learn the right computation?

This notebook tests the suspected failure mechanisms using the **existing checkpoint**. It does not retrain, select steering features, or execute modified actions. All experiment definitions are literal notebook cells; diagnostics run directly in this kernel with native tracebacks. No repository checkout or Python experiment subprocess is used. Package installation uses external setup tools with live output.

| Test | Matched comparison / evidence |
|---|---|
| Did updates occur? | Saved optimizer step counters, training metrics, finite dictionary weights, decoder norms |
| Wrong collection distribution? | Native inference vs fresh noise at identical scheduled times vs fresh noise at training-distribution sampled times |
| Is it actually sparse? | Active features/token and observed never-fired fractions; no unsupported dead-feature claim |
| Does replacement preserve actions? | Original vs individual layers 0/8/17 vs all checkpoint layers; identical starting noise |
| Is the measurement trustworthy? | Original/original repeat, first-step noise-grid identity, exact record counts, streaming statistics |

Default cohort: **tasks 0, 7, 8, 9 × initial states 0, 1, 2 × 1 noise draw = 12 cases**. Each case uses its original task instruction and initial images. All checkpoint layers are measured over 10 denoising calls. The two fresh-noise conditions are diagnostic computations, not robot policies. Only normal-inference and replacement action predictions are compared. No action is executed in the simulator.

Use `PHASE="metadata"` for a CPU checkpoint inspection first. Use `PHASE="all"` on a high-RAM L4/A100 for the complete diagnostic. A fresh runtime may require **one session restart after installation**; rerun from Controls afterward. Keep the same run name to resume complete cases. Change the run name if the checkpoint, cohort, definitions, policy snapshot, or runtime changes.

The collection timer begins after loading. It stops between complete cases; it is not a hard wall-clock guarantee. Saved results are conditional on completed cases when partial. Report reconstruction by layer/time and action error against the repeat floor. No universal pass threshold is assumed. These scenes are **not established as held out from pretraining**, and this is not attribution validation, task success, steerability, or collision prevention.
'''

CONTROLS='''# @title Controls
from pathlib import Path
DRIVE_ROOT = '/content/drive/MyDrive/groot-run-shared-programmer908' # @param {type:"string"}
CHECKPOINT = '' # @param {type:"string"}
RUN_NAME = 'transcoder-training-audit-001' # @param {type:"string"}
PHASE = 'all' # @param ["metadata", "all"]
TASK_IDS = '0,7,8,9' # @param {type:"string"}
INIT_STATE_IDS = '0,1,2' # @param {type:"string"}
NOISE_DRAWS = 1 # @param {type:"integer"}
INFERENCE_STEPS = 10 # @param {type:"integer"}
REPLACEMENT_LAYERS = '0,8,17' # @param {type:"string"}
REPLACE_ALL = True # @param {type:"boolean"}
COLLECTION_MINUTES = 30 # @param {type:"integer"}
HF_OFFLINE = True # @param {type:"boolean"}
ALLOW_AUTH_REFRESH = True # @param {type:"boolean"}
POLICY_REPO = 'lerobot/pi05_libero_finetuned'
POLICY_REVISION = 'main' # resolved snapshot commit is persisted in the manifest
if PHASE not in ('metadata','all') or not RUN_NAME or Path(RUN_NAME).name!=RUN_NAME:
    raise ValueError('Use phase metadata/all and a single nonempty run-folder name')
'''

MOUNT='''# @title Mount Existing Drive Folder
from google.colab import drive
from pathlib import Path
if not Path('/content/drive/MyDrive').exists():
    drive.mount('/content/drive')
DRIVE_ROOT = Path(DRIVE_ROOT)
if not DRIVE_ROOT.is_dir():
    raise FileNotFoundError(f'{DRIVE_ROOT} is not visible. Use the same Drive shortcut/path as the working steering notebook.')
RUN_DIR = DRIVE_ROOT/'outputs/transcoder-audit'/RUN_NAME
CHECKPOINT_PATH = Path(CHECKPOINT) if CHECKPOINT.strip() else DRIVE_ROOT/'transcoders/pi05_libero/allframes_80-10-10_epoch1_b8_exp16_latest_lambda1e-4/step_027233.pt'
if not CHECKPOINT_PATH.is_absolute(): CHECKPOINT_PATH=DRIVE_ROOT/CHECKPOINT_PATH
if not CHECKPOINT_PATH.is_file(): raise FileNotFoundError(CHECKPOINT_PATH)
print('Checkpoint:', CHECKPOINT_PATH)
print('Size (GiB):', round(CHECKPOINT_PATH.stat().st_size/2**30,2))
print('Saved evidence:', RUN_DIR)
'''

CACHE='''# @title Prepare Model Cache And LIBERO Assets (Skipped For Metadata Only)
import os, shutil, site, tarfile
from pathlib import Path
snapshots = {}
if PHASE == 'all':
    cache=Path('/content/hf_home'); cache.mkdir(parents=True,exist_ok=True)
    archive=DRIVE_ROOT/'archives/hf_home.tar'
    policy_cache=cache/'hub'/('models--'+POLICY_REPO.replace('/','--'))/'snapshots'
    policy_cached=policy_cache.is_dir() and any((p/'config.json').is_file() and
        ((p/'model.safetensors').is_file() or (p/'model.safetensors.index.json').is_file()) for p in policy_cache.iterdir())
    if not policy_cached:
        if archive.is_file():
            print('Restoring model cache archive',flush=True)
            with tarfile.open(archive) as handle: handle.extractall(cache,filter='data')
        elif (DRIVE_ROOT/'hf_home').is_dir():
            print('Copying existing Drive model cache',flush=True)
            shutil.copytree(DRIVE_ROOT/'hf_home',cache,dirs_exist_ok=True)
    from huggingface_hub import snapshot_download, constants as hf_constants
    for repo,kind in [(POLICY_REPO,'model'),('google/paligemma-3b-pt-224','model'),('lerobot/libero-assets','dataset')]:
        revision=POLICY_REVISION if repo==POLICY_REPO else 'main'
        try:
            path=snapshot_download(repo,repo_type=kind,revision=revision,local_files_only=True)
        except Exception:
            if not ALLOW_AUTH_REFRESH: raise
            token=os.environ.get('HF_TOKEN')
            if not token:
                from google.colab import userdata
                try: token=userdata.get('HF_TOKEN')
                except Exception: token=None
            old=hf_constants.HF_HUB_OFFLINE; hf_constants.HF_HUB_OFFLINE=False
            try: path=snapshot_download(repo,repo_type=kind,revision=revision,token=token,local_files_only=False)
            finally: hf_constants.HF_HUB_OFFLINE=old
        snapshots[repo]=str(Path(path).resolve())
        print(repo,snapshots[repo],flush=True)
    roots=[Path(p)/'libero/libero' for p in site.getsitepackages()]
    libero_root=next((p for p in roots if p.is_dir()),None)
    if libero_root is None: raise RuntimeError('hf-libero is missing; rerun runtime setup')
    assets=libero_root/'assets'
    if not (assets/'scenes/libero_tabletop_base_style.xml').exists():
        shutil.copytree(snapshots['lerobot/libero-assets'],assets,dirs_exist_ok=True)
    if not (assets/'scenes/libero_tabletop_base_style.xml').exists(): raise RuntimeError('LIBERO assets are incomplete')
    Path('/content/pi05-audit-support').mkdir(exist_ok=True)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('Select a GPU runtime for PHASE=all')
    ram_gib=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:')))/2**20
    gpu_gib=torch.cuda.get_device_properties(0).total_memory/2**30
    print(f'GPU memory: {gpu_gib:.1f} GiB; system memory: {ram_gib:.1f} GiB')
    if gpu_gib<21 or ram_gib<23: raise RuntimeError('Use a high-RAM L4/A100: at least 22 GiB GPU and 24 GiB system RAM recommended')
else:
    print('Metadata-only: no simulator, policy, assets or GPU required.')
'''

LAUNCH='''# @title Run Or Resume Diagnostics In This Notebook Thread
from importlib.metadata import version, PackageNotFoundError
from IPython.display import display, Markdown
if 'AUDIT_NOTEBOOK_VERSION' not in globals(): raise RuntimeError('Run the three Definitions cells first')
packages={}
for name in ('torch','transformers','lerobot','hf-libero','mujoco','numpy'):
    try: packages[name]=version(name)
    except PackageNotFoundError: packages[name]=None
parse_ids=lambda value:tuple(int(v.strip()) for v in value.split(',') if v.strip())
config=AuditConfig(tasks=parse_ids(TASK_IDS),starts=parse_ids(INIT_STATE_IDS),seeds_per_start=NOISE_DRAWS,
    inference_steps=INFERENCE_STEPS,replacement_layers=parse_ids(REPLACEMENT_LAYERS),replace_all=REPLACE_ALL)
print('Notebook definitions:',AUDIT_NOTEBOOK_VERSION[:12])
print('Planned cases:',len(config.cases()))
runtime_report=dict(packages=packages,snapshots=snapshots,
                    gpu=torch.cuda.get_device_name(0) if PHASE=='all' else None)
report=run_transcoder_audit(config,CHECKPOINT_PATH,RUN_DIR,source_digest=AUDIT_NOTEBOOK_VERSION,
    runtime_report=runtime_report,policy_path=snapshots.get(POLICY_REPO,POLICY_REPO),
    minutes=COLLECTION_MINUTES,phase=PHASE)
if PHASE=='all' and (RUN_DIR/'summary.md').exists(): display(Markdown((RUN_DIR/'summary.md').read_text()))
else:
    lines=['| Layer | Width | Features | Finite weights | Mean decoder norm |', '|---|---:|---:|---|---:|']
    for row in report['layers']:
        cfg=row['config']; features=cfg.get('d_features') or cfg['d_model']*cfg.get('expansion_factor',16)
        lines.append(f"| {row['layer']} | {cfg['d_model']} | {features} | {row['finite']} | {row['decoder_norm_mean']} |")
    display(Markdown('\\n'.join(lines)))
    print('Full metadata and optimizer counters:',RUN_DIR/'checkpoint_audit.json')
'''

EXPORT='''# @title Read Saved Results And Export A Small Evidence Packet
import zipfile
from IPython.display import display, Markdown, FileLink
summary=RUN_DIR/'summary.md'
if summary.exists(): display(Markdown(summary.read_text()))
for name in ('checkpoint_audit.json','report.json','state.json','failure.json'):
    if (RUN_DIR/name).exists(): print(name,'→',RUN_DIR/name)
packet=Path('/content')/(RUN_NAME+'-evidence.zip')
with zipfile.ZipFile(packet,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(RUN_DIR.rglob('*')):
        if path.is_file() and (path.suffix in ('.json','.md') or path.name=='actions.npz'):
            archive.write(path,path.relative_to(RUN_DIR))
print('Full streaming statistics remain on Drive in cases/*/fit_stats.npz')
print('Small evidence packet:',packet)
display(FileLink(str(packet)))
# To download explicitly: from google.colab import files; files.download(str(packet))
'''


def notebook():
    template=json.loads((ROOT/'notebooks/pi05_abstract_a_collection.ipynb').read_text())
    installer=''.join(next(c['source'] for c in template['cells'] if ''.join(c['source']).startswith('# @title Install Pinned')))
    installer=installer.replace('pi05-abstract-install-receipt','pi05-audit-install-receipt').replace('/content/pi05-monitor-support','/content/pi05-audit-support')
    installer='# @title Reuse Or Install The Pinned Runtime\nif PHASE == "all":\n    if not Path("/dev/nvidia0").exists():\n        raise RuntimeError("Select a high-RAM L4/A100 GPU runtime, or use PHASE=metadata.")\n'+textwrap.indent(installer,'    ')+'\nelse:\n    import torch, numpy\n    print("Using current torch for metadata inspection only; no package replacement.")\n'
    cells=[markdown(INTRO,'audit-intro'),code(CONTROLS,'audit-controls'),code(MOUNT,'audit-mount'),
           code(installer,'audit-install'),code(CACHE,'audit-cache'),*definitions(),
           code(LAUNCH,'audit-launch'),code(EXPORT,'audit-export')]
    return dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),colab=dict(name=DEST.name),accelerator='GPU'),nbformat=4,nbformat_minor=5)


if __name__=='__main__':
    DEST.write_text(json.dumps(notebook(),indent=1)+'\n')
    print(DEST)
