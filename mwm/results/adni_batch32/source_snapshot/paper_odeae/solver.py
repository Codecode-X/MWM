"""CUDA Tsitouras 5(4), endpoint integration with adaptive embedded error control.

Coefficients: Diffrax v0.7.0, _solver/tsit5.py (Apache-2.0), pinned in reference/.
The controller is an explicitly documented reconstruction, not Diffrax's PID.
"""
import json
from pathlib import Path
import torch
from contextlib import contextmanager
from contextvars import ContextVar

_statistics=ContextVar('ode_statistics',default=None)


@contextmanager
def solver_statistics(trace=False):
    """Opt-in diagnostics; leaves numerical operations and tolerances unchanged."""
    result={'phase':'unspecified','calls':[],'trace':trace}
    token=_statistics.set(result)
    try:yield result
    finally:_statistics.reset(token)

TABLEAU = json.loads((Path(__file__).parent/'reference/tsit5_tableau.json').read_text())


def integrate(func, state, delta_years, rtol=1e-4, atol=1e-6, max_steps=512):
    collection=_statistics.get()
    record=None
    if collection is not None:
        record={'phase':collection['phase'],'batch_size':state.shape[-2],
                'active_patients':int(delta_years.count_nonzero()),'accepted':0,'rejected':0,'nfe':0}
        if collection['trace']:
            record.update(delta=delta_years.detach().cpu().tolist(),attempts=[])
        collection['calls'].append(record)
    if state.device.type != 'cuda' or delta_years.device != state.device:
        raise RuntimeError('ODE model execution requires CUDA tensors on one device')
    if not torch.isfinite(state).all() or not torch.isfinite(delta_years).all():
        raise FloatingPointError('nonfinite ODE input')
    if not delta_years.count_nonzero():
        return state
    factor = delta_years.reshape((1,)*(state.ndim-2)+(len(delta_years),1))
    coefficients = TABLEAU['a_lower']
    y, u, step = state, 0., .25
    for _ in range(max_steps):
        if record is not None:record['nfe']+=1+len(coefficients)
        step = min(step, 1.-u)
        stages = [func(y)*factor]
        for row in coefficients:
            trial = y + step*sum(a*k for a,k in zip(row,stages))
            stages.append(func(trial)*factor)
        candidate = y+step*sum(b*k for b,k in zip(TABLEAU['b_sol'],stages))
        error = step*sum(b*k for b,k in zip(TABLEAU['b_error'],stages))
        scale = atol+rtol*torch.maximum(y.abs(),candidate.abs())
        ratio = float((error/scale).square().mean().sqrt().detach())
        if record is not None and collection['trace']:
            record['attempts'].append({'u_start':u,'step':step,'u_proposed':u+step,
                                       'error_ratio':ratio,'accepted':ratio<=1.})
        if not torch.isfinite(candidate).all() or not torch.isfinite(error).all():
            raise FloatingPointError('nonfinite Tsit5 step')
        if ratio <= 1.:
            if record is not None:record['accepted']+=1
            y, u = candidate, u+step
            if u >= 1.-1e-12:
                return y
        elif record is not None:record['rejected']+=1
        multiplier = 5. if ratio == 0 else min(5.,max(.2,.9*ratio**(-.2)))
        step *= multiplier
        if step < 1e-8:
            raise RuntimeError('Tsit5 step underflow; no solver/CPU fallback is permitted')
    raise RuntimeError(f'Tsit5 exceeded {max_steps} steps')
