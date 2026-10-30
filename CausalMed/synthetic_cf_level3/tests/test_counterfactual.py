from copy import deepcopy

import numpy as np
import pandas as pd

from synthetic_cf_level3.counterfactual_policies import decide
from synthetic_cf_level3.generate import branch, factual


def test_fork_pairing_dynamic_local_state_and_no_future_fact(record, cfg):
    specs = {'thresholds': {'3': 1.2, '4': 1.6}, 'delay_years': 1.}
    branches = [branch(record, p, specs, cfg) for p in range(5)]
    for b in branches:
        assert np.array_equal(b['hidden'][:, 0], branches[0]['hidden'][:, 0])
        assert b['clean'][0, 0] == branches[0]['clean'][0, 0]
    assert np.max(np.abs(branches[0]['hidden'][1] - branches[1]['hidden'][1])) > .1
    for p in [3, 4]:
        b = branches[p]
        expected = [decide(p, t, b['fork'], x[1], specs) for t, x in zip(b['visit_times'], b['visit_states'])]
        assert expected == b['visit_actions']
    modified = deepcopy(record)
    j = np.searchsorted(record['patient'].times, cfg['simulation']['fork_after'])
    modified['observations'][j + 1:] = -99999
    modified['states'][j + 1:] = -99999
    for p in [3, 4]:
        assert np.array_equal(branch(modified, p, specs, cfg)['hidden'], branches[p]['hidden'])
    # Branch traversal order cannot affect any deterministic counterfactual.
    for p in reversed(range(5)):
        assert np.array_equal(branch(record, p, specs, cfg)['hidden'], branches[p]['hidden'])


def test_dynamic_decisions_can_differ_between_worlds(record, cfg):
    low_specs = {'thresholds': {'3': 1.2, '4': 1.6}, 'delay_years': 1.}
    a, b = [branch(record, p, low_specs, cfg) for p in (3, 4)]
    assert a['visit_actions'] != b['visit_actions']
    assert not np.array_equal(a['visit_states'][1:], b['visit_states'][1:])


def test_schedules_irregular_independent_and_no_terminal_visit(cfg):
    a, b = [factual(cfg, 'test', i)['patient'].times for i in (0, 1)]
    assert a[0] == b[0] == 0 and a[-1] < 8 and b[-1] < 8
    assert np.all((np.diff(a) >= .25) & (np.diff(a) <= .75))
    assert np.var(np.diff(a)) > 0
    assert not np.array_equal(a, b)


def test_exported_horizons_and_train_thresholds(small_dataset):
    root, cfg = small_dataset
    import json
    specs = json.loads((root / 'public/policy_specs.json').read_text())
    train = pd.read_parquet(root / 'public/train_visits.parquet')
    assert specs['thresholds']['3'] == train.X2_obs.quantile(.5)
    assert specs['thresholds']['4'] == train.X2_obs.quantile(.8)
    outcomes = pd.read_parquet(root / 'oracle_private/cf_outcomes.parquet')
    assert set(outcomes.relative_time) == {1., 2., 3.}
    assert len(outcomes) == cfg['splits']['test'] * 5 * 3
