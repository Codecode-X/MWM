from copy import deepcopy

import pytest

from synthetic_cf_level3.common import read_config
from synthetic_cf_level3.generate import factual, generate


@pytest.fixture
def cfg():
    return read_config()


@pytest.fixture
def record(cfg):
    return factual(cfg, 'train', 0)


@pytest.fixture(scope='session')
def small_dataset(tmp_path_factory):
    cfg = read_config()
    cfg['splits'] = {'train': 12, 'val': 4, 'test': 4}
    cfg['workers'] = 1
    root = tmp_path_factory.mktemp('level3') / 'dataset'
    generate(deepcopy(cfg), root, validate=False)
    return root, cfg
