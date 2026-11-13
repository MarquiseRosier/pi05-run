import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pi05_training_data import stage_training_dataset,dataset_rate_limit_retry,training_dataset_downloads


def test_metadata_only_destination_recovers_existing_data(tmp_path):
    root=tmp_path/'partial';cached=tmp_path/'cached'
    for folder in (root,cached):
        (folder/'meta').mkdir(parents=True)
        (folder/'meta/info.json').write_text('{"version":"v3.0"}')
    (cached/'data').mkdir();(cached/'data/observations.parquet').write_bytes(b'existing')
    stage_training_dataset(root,[tmp_path/'missing',cached])
    assert (root/'data/observations.parquet').read_bytes()==b'existing'
    (root/'data/observations.parquet').write_bytes(b'keep partial cache')
    stage_training_dataset(root,[cached])
    assert (root/'data/observations.parquet').read_bytes()==b'keep partial cache'


def test_incompatible_cache_is_not_merged(tmp_path):
    root,cached=tmp_path/'root',tmp_path/'cached'
    for folder in (root,cached): (folder/'meta').mkdir(parents=True)
    (root/'meta/info.json').write_text('old');(cached/'meta/info.json').write_text('other')
    (cached/'data').mkdir();(cached/'data/data.parquet').write_bytes(b'data')
    stage_training_dataset(root,[cached])
    assert not (root/'data').exists()


def test_transport_retries_429_and_restores_settings(monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE','1')
    calls=[];sleeps=[]
    def download(*args,**kwargs):
        calls.append(kwargs.copy())
        assert constants.HF_HUB_DISABLE_XET and not constants.HF_HUB_OFFLINE
        assert not constants.HF_HUB_ENABLE_HF_TRANSFER
        if len(calls)==1: raise ConnectionError('HTTP status 429 Too Many Requests')
        return 'cached result'
    module=NS(snapshot_download=download)
    constants=NS(HF_HUB_OFFLINE=True,HF_HUB_DISABLE_XET=False,HF_HUB_ENABLE_HF_TRANSFER=True)
    with training_dataset_downloads(module,constants,allow_download=True,token='test-token',sleep=sleeps.append):
        result=module.snapshot_download('repo',local_dir='keep-this-cache',revision='v3.0')
    assert result=='cached result' and sum(sleeps)==60
    assert len(calls)==2 and all(c['max_workers']==1 and c['local_dir']=='keep-this-cache' for c in calls)
    assert all(c['token']=='test-token' and not c.get('force_download',False) for c in calls)
    assert module.snapshot_download is download and constants.HF_HUB_OFFLINE
    assert not constants.HF_HUB_DISABLE_XET and constants.HF_HUB_ENABLE_HF_TRANSFER
    assert os.environ['HF_HUB_OFFLINE']=='1'


def test_retry_after_and_bounded_retries():
    sleeps=[];calls=[]
    def fail():
        calls.append(1)
        exc=ConnectionError('rate limited')
        exc.response=NS(status_code=429,headers={'Retry-After':'90'})
        raise exc
    with pytest.raises(RuntimeError,match='still rate limiting'):
        dataset_rate_limit_retry(fail,attempts=2,sleep=sleeps.append)
    assert len(calls)==2 and sum(sleeps)==90
    def unauthorized(): raise PermissionError('401 Unauthorized')
    with pytest.raises(PermissionError): dataset_rate_limit_retry(unauthorized,sleep=lambda _:pytest.fail('must not retry'))


def test_offline_refuses_download_and_restores_on_failure():
    def forbidden(*a,**k): pytest.fail('network attempted')
    module=NS(snapshot_download=forbidden)
    constants=NS(HF_HUB_OFFLINE=False,HF_HUB_DISABLE_XET=False)
    with pytest.raises(RuntimeError,match='ALLOW_DATASET_DOWNLOAD=False'):
        with training_dataset_downloads(module,constants,allow_download=False):
            module.snapshot_download('repo')
    assert module.snapshot_download is forbidden and not constants.HF_HUB_OFFLINE
