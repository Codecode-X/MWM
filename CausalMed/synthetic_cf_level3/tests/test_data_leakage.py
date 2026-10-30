import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from synthetic_cf_level3.dataset_adapter import PublicDataset


def test_public_only_mount_and_cutoff(small_dataset, tmp_path):
    root, cfg = small_dataset
    mount = tmp_path / 'isolated_public'
    shutil.copytree(root / 'public', mount)
    data = PublicDataset(mount, 'test')
    queries = data.queries()
    for s in data:
        fork = queries[queries.patient_id == s['patient_id']].fork_time.iloc[0]
        assert s['times'][-1] == fork
        assert all(a['start'] < fork and a['end'] <= fork for a in s['treatment_intervals'])
        assert set(s) == {'patient_id', 'c', 'times', 'observations', 'observation_mask', 'treatment_intervals'}
    assert len(list(data.shared_matched_sequences())) == cfg['splits']['test']
    assert not (mount / 'oracle_private').exists()
    with pytest.raises(ValueError):
        PublicDataset(root / 'oracle_private')


def test_patient_split_isolation_and_schema(small_dataset):
    root, cfg = small_dataset
    ids = [set(PublicDataset(root / 'public', split).ids) for split in ['train', 'val', 'test']]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    forbidden = {'D_true', 'L_true', 'E_true', 'alpha_true', 'beta_true', 'Y_clean', 'propensity', 'Y_cf_true'}
    for p in (root / 'public').glob('*.parquet'):
        assert not (set(pd.read_parquet(p).columns) & forbidden)
    with pytest.raises(FileNotFoundError):
        pd.read_parquet(root / 'public/test_visits.parquet')


def test_reject_injected_hidden_column_or_path_escape(small_dataset, tmp_path):
    root, _ = small_dataset
    mount = tmp_path / 'public'
    shutil.copytree(root / 'public', mount)
    p = mount / 'train_visits.parquet'
    v = pd.read_parquet(p)
    v['D_true'] = 0
    v.to_parquet(p, index=False)
    with pytest.raises(ValueError, match='schema violation'):
        PublicDataset(mount, 'train')
    p.unlink()
    p.symlink_to(root / 'oracle_private/factual_visits.parquet')
    with pytest.raises(ValueError, match='outside'):
        PublicDataset(mount, 'train')
