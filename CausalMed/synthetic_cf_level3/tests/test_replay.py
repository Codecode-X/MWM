from copy import deepcopy

import numpy as np

from synthetic_cf_level3.common import sha256
from synthetic_cf_level3.generate import factual, generate
from synthetic_cf_level3.observation import clean_observations
from synthetic_cf_level3.simulator import PiecewiseTrajectory


def test_factual_replay(record, cfg):
    p = record['patient']
    trajectory = PiecewiseTrajectory(p, p.initial_state, 0., record['intervals'], cfg)
    Y = clean_observations(trajectory.state(p.times), p.c, cfg)[0]
    expected = clean_observations(record['states'].T, p.c, cfg)[0]
    assert np.max(np.abs(Y - expected)) < 1e-6


def test_rng_order_and_purpose_independence(cfg):
    first = factual(cfg, 'train', 1)
    factual(cfg, 'test', 17)
    again = factual(cfg, 'train', 1)
    assert np.array_equal(first['states'], again['states'])
    assert np.array_equal(first['observations'], again['observations'])
    ab = deepcopy(cfg)
    ab['observation']['Y_sd'] = 2
    changed = factual(ab, 'train', 1)
    assert np.array_equal(first['patient'].times, changed['patient'].times)
    assert np.array_equal(first['patient'].initial_state, changed['patient'].initial_state)
    assert not np.array_equal(first['patient'].initial_state, factual(cfg, 'val', 1)['patient'].initial_state)


def test_repeat_generation_and_worker_invariance(small_dataset, tmp_path):
    root, cfg = small_dataset
    parallel = deepcopy(cfg)
    parallel['workers'] = 2
    out = generate(parallel, tmp_path / 'replica', validate=False)
    for directory in ['public', 'oracle_private']:
        for p in (root / directory).iterdir():
            assert sha256(p) == sha256(out / directory / p.name), p.name
