from copy import deepcopy

import numpy as np
import pytest

from synthetic_cf_level3.behavior_policy import ObservedHistory, propensity
from synthetic_cf_level3.counterfactual_policies import decide, treatment_duration


def test_positivity_and_randomized(cfg):
    for gamma in [0., .5, 1., 2.]:
        cfg['behavior']['gamma'] = gamma
        for Y in [-100., 75., 1000.]:
            for L in [-1000., 0., 1000.]:
                p = propensity(ObservedHistory(1., Y, L, 0., 76., 1), cfg)
                assert .15 <= p <= .85
                if gamma == 0:
                    assert p == .5


def test_feedback_decision_uses_observable_history(cfg):
    low = ObservedHistory(1., 75., .4, 0., 76., 0)
    high = ObservedHistory(1., 75., 1.4, 0., 76., 0)
    assert propensity(high, cfg) > propensity(low, cfg)
    worse = ObservedHistory(1., 65., .4, 0., 76., 0)
    assert propensity(worse, cfg) > propensity(low, cfg)
    persistence = ObservedHistory(1., 75., .4, 0., 76., 1)
    assert propensity(persistence, cfg) > propensity(low, cfg)
    assert not any(k in low.__dataclass_fields__ for k in ['D', 'E', 'alpha', 'beta'])
    ab = deepcopy(cfg)
    ab['behavior']['risk_weight'] = 0
    assert propensity(high, ab) == propensity(low, ab)


def test_intervention_definitions_and_delayed_visit():
    specs = {'delay_years': 1., 'thresholds': {'3': 1., '4': 2.}}
    assert decide(0, 9, 3., 100, specs) == 0
    assert decide(1, 3., 3., -100, specs) == 1
    times = [3.2, 3.7, 4.0, 4.4, 4.9]
    assert [decide(2, t, 3.2, 0, specs) for t in times] == [0, 0, 0, 1, 1]
    assert decide(3, 4, 3, 1., specs) == 0  # strict inequality
    assert decide(3, 4, 3, 1.1, specs) == 1
    assert decide(4, 4, 3, 1.1, specs) == 0
    with pytest.raises(ValueError):
        decide(5, 0, 0, 0, specs)
    assert abs(treatment_duration([(3., 4., 0), (4., 5., 1)], 3., 4.3) - .3) < 1e-12
