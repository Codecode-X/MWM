from copy import deepcopy

import numpy as np
import pytest

from synthetic_cf_level3.observation import clean_observations
from synthetic_cf_level3.simulator import PiecewiseTrajectory, check_state, integrate, rhs


@pytest.mark.parametrize('treatment', [0, 1])
@pytest.mark.parametrize('E', [0., 1.])
def test_boundary_invariance(record, cfg, treatment, E):
    p = record['patient']
    x = np.array([0., .4, E])
    derivative = rhs(0., x, treatment, p, cfg)
    assert derivative[0] > 0
    assert derivative[2] >= 0 if E == 0 else derivative[2] <= 0
    sol = integrate(p, x, 0., 8., treatment, cfg)
    states = sol.sol(np.linspace(0, 8, 193))
    assert np.isfinite(states).all()
    assert states[0].min() >= 0 and states[2].min() >= -1e-9 and states[2].max() <= 1 + 1e-8


def test_solver_tolerance_and_exact_switches(record, cfg):
    p = record['patient']
    intervals = [(0., .313, 0), (.313, 1.234, 1), (1.234, 2., 0)]
    traj = PiecewiseTrajectory(p, p.initial_state, 0., intervals, cfg)
    tight = deepcopy(cfg)
    tight['simulation']['solver'].update(rtol=1e-11, atol=1e-13)
    reference = PiecewiseTrajectory(p, p.initial_state, 0., intervals, tight)
    times = np.r_[np.linspace(0., 2., 101), .313, 1.234]
    assert np.max(np.abs(traj.state(times) - reference.state(times))) < 1e-7
    assert traj.treatment(.313) == 1 and traj.treatment(1.234) == 0
    E_exact = (1 - np.exp(-1.6 * (1.234 - .313))) * np.exp(-1.6 * (2 - 1.234))
    assert abs(traj.state(2.)[2] - E_exact) < 1e-8
    with pytest.raises(ValueError):
        traj.state(2.01)


def test_causal_effect_and_risk_ablations(record, cfg):
    p = record['patient']
    x = record['states'][7]
    no = integrate(p, x, 0, 3, 0, cfg).y[:, -1]
    yes = integrate(p, x, 0, 3, 1, cfg).y[:, -1]
    assert yes[0] < no[0] and yes[1] < no[1]
    assert clean_observations(yes, p.c, cfg)[0] > clean_observations(no, p.c, cfg)[0]
    higher = x.copy()
    higher[1] += .3
    assert integrate(p, higher, 0, 1, 0, cfg).y[0, -1] > integrate(p, x, 0, 1, 0, cfg).y[0, -1]
    ab = deepcopy(cfg)
    ab['dynamics']['risk_weight'] = 0
    assert abs(integrate(p, higher, 0, 1, 0, ab).y[0, -1] - integrate(p, x, 0, 1, 0, ab).y[0, -1]) < 1e-8


def test_fail_explicitly(record, cfg):
    p = record['patient']
    for x in ([float('nan'), 1, 0], [-1, 1, 0], [1, 1, 1.1]):
        with pytest.raises(FloatingPointError):
            check_state(x, cfg)
    with pytest.raises(ValueError):
        integrate(p, p.initial_state, 0, 1, 2, cfg)
    with pytest.raises(ValueError):
        PiecewiseTrajectory(p, p.initial_state, 0., [(0., 1., 0), (1.1, 2., 1)], cfg)
