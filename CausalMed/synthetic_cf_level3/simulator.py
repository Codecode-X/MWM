"""Private fixed ODE simulator. No learned vector field and no state clipping."""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit


@dataclass(frozen=True)
class Patient:
    patient_id: str
    c: float
    alpha: float
    beta: float
    initial_state: np.ndarray
    times: np.ndarray


def rhs(t, state, treatment, patient, cfg):
    D, L, E = state
    d = cfg['dynamics']
    return np.array([
        patient.alpha * (d['progress_base'] + d['progress_scale'] *
                         expit(d['D_sigmoid_scale'] * (D - d['D_sigmoid_center'])))
        + d['risk_weight'] * expit(d['L_sigmoid_scale'] * L)
        - patient.beta * E / (d['exposure_half'] + E) * D / (d['disease_half'] + D),
        d['tracking'] * (D - L) - d['exposure_risk'] * E +
        d['oscillation'] * np.sin(d['oscillation_D'] * D + patient.c),
        d['exposure_rate'] * (treatment - E),
    ])


def check_state(state, cfg):
    x = np.asarray(state)
    tol = 1e-7  # solver error check only; never clip state values
    if not np.isfinite(x).all() or np.max(np.abs(x)) > cfg['simulation']['solver']['state_limit']:
        raise FloatingPointError('nonfinite / out-of-range hidden state')
    if np.min(x[0]) < -tol or np.min(x[2]) < -tol or np.max(x[2]) > 1 + tol:
        raise FloatingPointError('D >= 0 and 0 <= E <= 1 invariant violated')


def integrate(patient, state, start, end, treatment, cfg):
    if treatment not in (0, 1) or not np.isfinite([start, end]).all() or end <= start:
        raise ValueError('invalid constant-treatment interval')
    check_state(state, cfg)
    opts = cfg['simulation']['solver']
    sol = solve_ivp(lambda t, x: rhs(t, x, treatment, patient, cfg), (start, end),
                    np.asarray(state, dtype=float), method=opts['method'],
                    rtol=opts['rtol'], atol=opts['atol'], dense_output=True)
    if not sol.success:
        raise RuntimeError(f'integration failed for {patient.patient_id}: {sol.message}')
    check_state(sol.y, cfg)
    check_state(sol.sol(np.linspace(start, end, 17)), cfg)
    return sol


class PiecewiseTrajectory:
    def __init__(self, patient, state, start, intervals, cfg):
        self.patient, self.cfg = patient, cfg
        self.start, self.initial = start, np.array(state, copy=True)
        self.intervals, self.solutions = list(intervals), []
        current = start
        for a, b, treatment in self.intervals:
            if a != current:
                raise ValueError('treatment intervals must be contiguous')
            sol = integrate(patient, state, a, b, treatment, cfg)
            self.solutions.append(sol)
            state, current = sol.y[:, -1], b
        self.end = current

    def state(self, times):
        q = np.atleast_1d(np.asarray(times, dtype=float))
        if not np.isfinite(q).all() or np.any(q < self.start) or np.any(q > self.end):
            raise ValueError('queries outside simulated time interval')
        ends = np.array([b for a, b, u in self.intervals])
        indices = np.minimum(np.searchsorted(ends, q, side='right'), len(ends) - 1)
        result = np.empty((3, len(q)))
        for j in np.unique(indices):
            selected = indices == j
            result[:, selected] = self.solutions[j].sol(q[selected])
        check_state(result, self.cfg)
        return result[:, 0] if np.ndim(times) == 0 else result

    def treatment(self, times):
        q = np.asarray(times)
        if np.any(q < self.start) or np.any(q > self.end):
            raise ValueError('treatment query outside trajectory')
        idx = np.minimum(np.searchsorted([b for a, b, u in self.intervals], q,
                                         side='right'), len(self.intervals) - 1)
        return np.array([u for a, b, u in self.intervals])[idx]
