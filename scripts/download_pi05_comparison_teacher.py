"""Download the pinned comparison teacher with cache checks and visible progress."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import threading
import time

REPO = 'lerobot/pi05_libero_finetuned'
REVISION = '8e174154ef5f6c60a8da12ae99c303d8963138c1'
# Sizes from the Hub file manifest at the immutable revision above.
FILES = {
    'config.json': 2170,
    'train_config.json': 5621,
    'policy_preprocessor.json': 1903,
    'policy_postprocessor.json': 660,
    'policy_preprocessor_step_2_normalizer_processor.safetensors': 7668,
    'policy_postprocessor_step_0_unnormalizer_processor.safetensors': 7668,
    'model.safetensors': 7473096344,
}
WEIGHTS_SHA256 = '877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae'


def _progress(stop, filename, cache_root, interval=15):
    started = time.monotonic()
    partial = cache_root / 'models--lerobot--pi05_libero_finetuned' / 'blobs' / (WEIGHTS_SHA256 + '.incomplete')
    while not stop.wait(interval):
        detail = ''
        if filename == 'model.safetensors':
            try:
                size = partial.stat().st_size
                detail = f'; partial file {size / 1e9:.2f}/{FILES[filename] / 1e9:.2f} GB'
            except FileNotFoundError:
                detail = '; waiting for transfer or cache lock'
        print(f'Download active: {filename}; elapsed {(time.monotonic()-started)/60:.1f} min{detail}', flush=True)


def download_teacher(result_path):
    # Set before importing the Hub: standard HTTP retains resumable partial files.
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
    from huggingface_hub import hf_hub_download
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.errors import LocalEntryNotFoundError

    result_path = Path(result_path)
    result_path.unlink(missing_ok=True)  # A failed attempt must not leave a success marker.
    print('Teacher weights: 7.47 GB. Checking pinned cache; completed files are reused.', flush=True)
    snapshot = None
    for filename, expected_size in FILES.items():
        cached = None
        try:
            cached = Path(hf_hub_download(REPO, filename, revision=REVISION, local_files_only=True))
        except LocalEntryNotFoundError:
            pass
        valid = cached is not None and cached.is_file() and cached.stat().st_size == expected_size
        if valid:
            path = cached
            print(f'Cached: {filename} ({expected_size / 1e6:.2f} MB)', flush=True)
        else:
            print(f'Downloading/resuming: {filename} ({expected_size / 1e6:.2f} MB)', flush=True)
            stop = threading.Event()
            monitor = threading.Thread(target=_progress, args=(stop, filename, Path(HF_HUB_CACHE)), daemon=True)
            monitor.start()
            try:
                path = Path(hf_hub_download(
                    REPO, filename, revision=REVISION,
                    force_download=cached is not None,
                ))
            finally:
                stop.set()
                monitor.join(timeout=1)
            if path.stat().st_size != expected_size:
                raise IOError(f'Incomplete teacher file: {filename}; rerun this cell to repair it')
            print(f'Complete: {filename}', flush=True)
        snapshot = path.parent
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(str(snapshot))
    print('Teacher ready:', snapshot, flush=True)
    return snapshot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-path', type=Path, required=True)
    args = parser.parse_args()
    download_teacher(args.result_path)
