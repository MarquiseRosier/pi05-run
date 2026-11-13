"""Dataset setup helpers embedded into the main-thread training notebook."""
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
import os
from pathlib import Path
import re
import shutil
import time


def stage_training_dataset(root, candidates):
    """Recover a metadata-only destination from an existing compatible data cache."""
    root=Path(root)
    if any((root/'data').rglob('*.parquet')): return
    info=root/'meta/info.json'
    for candidate in map(Path,candidates):
        other=candidate/'meta/info.json'
        if candidate.resolve()==root.resolve() or not other.is_file(): continue
        if not any((candidate/'data').rglob('*.parquet')): continue
        if info.exists() and info.read_bytes()!=other.read_bytes():
            print('Skipping cache with different dataset metadata:',candidate,flush=True)
            continue
        print('Staging existing observation data:',candidate,flush=True)
        for name in ('meta','data','videos','images'):
            if (candidate/name).is_dir():
                shutil.copytree(candidate/name,root/name,dirs_exist_ok=True)
        return


def dataset_rate_limit_retry(operation, *, attempts=4, sleep=time.sleep):
    """Retry only explicit 429s, retaining Hub partial downloads and honoring Retry-After."""
    for attempt in range(attempts):
        try: return operation()
        except Exception as exc:
            response=getattr(exc,'response',None)
            limited=getattr(response,'status_code',None)==429 or re.search(r'\b429\b',str(exc))
            if not limited: raise
            if attempt+1==attempts:
                raise RuntimeError('Hugging Face is still rate limiting this dataset download. '
                    'Completed and partial files have been kept. Wait before rerunning only the dataset cell; '
                    'do not delete its cache or restart the runtime.') from exc
            delay=60*(2**attempt)
            retry_after=getattr(response,'headers',{}).get('Retry-After')
            if retry_after:
                try: delay=max(delay,float(retry_after))
                except ValueError:
                    try: delay=max(delay,parsedate_to_datetime(retry_after).timestamp()-time.time())
                    except (ValueError,TypeError,OverflowError): pass
            if delay>600:
                raise RuntimeError(f'Hub requested a wait of {delay:.0f}s. Download paused; cache preserved. '
                    'Wait that long before rerunning the dataset cell.') from exc
            print(f'Hub rate limit: waiting {delay:.0f}s before retry {attempt+2}/{attempts}; '
                  'existing downloads are preserved.',flush=True)
            while delay>0:
                interval=min(30,delay);sleep(interval);delay-=interval
                if delay>0: print(f'Download retry in {delay:.0f}s...',flush=True)


@contextmanager
def training_dataset_downloads(dataset_module, hf_constants, *, allow_download, token=None, sleep=time.sleep):
    """Scope transport settings to this loader, including an already-imported Hub runtime."""
    keys=('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_XET','HF_HUB_ENABLE_HF_TRANSFER')
    environment={k:os.environ.get(k) for k in keys}
    settings={k:getattr(hf_constants,k) for k in
              ('HF_HUB_OFFLINE','HF_HUB_DISABLE_XET','HF_HUB_ENABLE_HF_TRANSFER') if hasattr(hf_constants,k)}
    original=dataset_module.snapshot_download
    def download(*args,**kwargs):
        if not allow_download:
            raise RuntimeError('Local dataset is incomplete and ALLOW_DATASET_DOWNLOAD=False. '
                'Set DATASET_ROOT to a complete cache or enable downloads.')
        kwargs.update(max_workers=1)
        if token: kwargs['token']=token
        return dataset_rate_limit_retry(lambda:original(*args,**kwargs),sleep=sleep)
    try:
        # This avoids the failing Xet client; ordinary HTTP can still be rate limited.
        os.environ['HF_HUB_DISABLE_XET']='1';hf_constants.HF_HUB_DISABLE_XET=True
        os.environ['HF_HUB_ENABLE_HF_TRANSFER']='0'
        if hasattr(hf_constants,'HF_HUB_ENABLE_HF_TRANSFER'): hf_constants.HF_HUB_ENABLE_HF_TRANSFER=False
        hf_constants.HF_HUB_OFFLINE=not allow_download
        os.environ['HF_HUB_OFFLINE']='0' if allow_download else '1'
        os.environ['HF_DATASETS_OFFLINE']='0' if allow_download else '1'
        dataset_module.snapshot_download=download
        yield
    finally:
        dataset_module.snapshot_download=original
        for k,v in settings.items(): setattr(hf_constants,k,v)
        if 'HF_HUB_DISABLE_XET' not in settings: delattr(hf_constants,'HF_HUB_DISABLE_XET')
        for k,v in environment.items():
            if v is None: os.environ.pop(k,None)
            else: os.environ[k]=v
