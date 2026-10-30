"""Clinical observations are generated BEFORE the visit's treatment decision."""
import numpy as np


def clean_observations(state, c, cfg):
    D, L, E = np.asarray(state)
    p = cfg['observation']
    y = p['intercept'] + p['D_weight'] * D + p['L_weight'] * L + p['c_weight'] * c
    obs = np.array([y, D, L])
    if not np.isfinite(obs).all():
        raise FloatingPointError('nonfinite clinical observations')
    return obs


def observe(state, c, noise_rng, cfg):
    clean = clean_observations(state, c, cfg)
    obs = clean + noise_rng.normal(size=3) * np.array(
        [cfg['observation']['Y_sd'], cfg['observation']['X1_sd'], 0.0])
    if not np.isfinite(obs).all():
        raise FloatingPointError('nonfinite noisy observations')
    return obs
