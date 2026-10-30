"""Sequential randomization using only recorded observable history."""
from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class ObservedHistory:
    time: float
    Y: float
    X2: float
    previous_time: float | None = None
    previous_Y: float | None = None
    previous_treatment: int = 0


def propensity(history, cfg):
    b = cfg['behavior']
    R = 0.0
    if history.previous_time is not None:
        dt = history.time - history.previous_time
        if dt <= 0:
            raise ValueError('history must be chronological')
        R = np.clip((history.previous_Y - history.Y) / dt, 0, b['decline_clip']) / b['decline_clip']
    q = (b['outcome_weight'] * (b['outcome_center'] - history.Y) / b['outcome_scale']
         + b['risk_weight'] * history.X2 + b['decline_weight'] * R
         + b['previous_treatment_weight'] * history.previous_treatment)
    lower, upper = b['probability_bounds']
    p = lower + (upper - lower) * expit(b['gamma'] * (q - b['score_center']))
    if not np.isfinite(p) or not lower <= p <= upper:
        raise FloatingPointError('invalid treatment propensity')
    return float(p)
