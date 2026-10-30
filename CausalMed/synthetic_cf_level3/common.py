"""Configuration and deterministic, purpose-separated random streams."""
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).parent / 'configs/level3_v1.yaml'


def read_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    s = cfg['simulation']
    lo, hi = s['visit_gap']
    if not 0 < lo <= hi or s['fork_after'] + hi + max(s['horizons']) > s['years']:
        raise ValueError('invalid visit gaps or insufficient follow-up')
    if set(cfg['splits']) != {'train', 'val', 'test'} or min(cfg['splits'].values()) < 1:
        raise ValueError('all patient splits must be nonempty')
    if len(set(cfg['split_streams'].values())) != 3:
        raise ValueError('split streams must be independent')
    p0, p1 = cfg['behavior']['probability_bounds']
    if not 0 < p0 < p1 < 1:
        raise ValueError('positivity bounds must lie inside (0,1)')
    if cfg['behavior']['gamma'] < 0 or cfg['policies']['ids'] != [0, 1, 2, 3, 4]:
        raise ValueError('V1 requires nonnegative gamma and five fixed policy definitions')
    if s['samples_per_year'] < 1 or any(t <= 0 for t in s['horizons']):
        raise ValueError('positive prediction times and sampling rate required')
    return cfg


def rng_for(cfg, split, index, purpose):
    # No spawn() counters, global RNG or dependence on worker / branch order.
    return np.random.default_rng(np.random.SeedSequence(
        [cfg['seed'], cfg['split_streams'][split], index, purpose]))


def dump_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prediction_grid(cfg, end=None):
    s = cfg['simulation']
    end = max(s['horizons']) if end is None else end
    return np.unique(np.r_[np.arange(int(end * s['samples_per_year']) + 1) /
                              s['samples_per_year'], s['horizons'], end])
