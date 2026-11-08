"""CUDA tangent flow for the publication model's two-layer Tanh field.

Kept independent of historical experiment launchers and model factories.
"""
import torch
from paper_odeae.solver import integrate, solver_statistics


def field_jacobian(field, state):
    hidden = field[0](state).tanh()
    return torch.einsum('dh,bh,he->bde', field[2].weight,
                        1-hidden.square(), field[0].weight)


def tangent(field, state, duration, rtol=1e-8, atol=1e-10):
    if state.device.type != 'cuda':
        raise RuntimeError('Tangent-flow diagnostics require CUDA')
    width = state.shape[-1]
    identity = torch.eye(width, device=state.device, dtype=state.dtype)
    augmented = torch.cat([state, identity.expand(len(state), width, width).reshape(len(state), -1)], -1)

    def vector(value):
        z = value[:, :width]
        jac = value[:, width:].reshape(-1, width, width)
        return torch.cat([field(z), torch.bmm(field_jacobian(field, z), jac).reshape(len(z), -1)], -1)

    with solver_statistics() as statistics:
        result = integrate(vector, augmented, state.new_full((len(state),), duration), rtol, atol)
    return result[:, :width], result[:, width:].reshape(-1, width, width), statistics['calls']
