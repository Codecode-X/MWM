"""Exact input Jacobian for Linear-Tanh-Linear vector fields on CUDA."""
import torch
from torch import nn


def squared_jacobian_norm(field,states):
    """Return ||df/dz||_F^2 per state, without dimension normalization.

    f(z)=W2 tanh(W1 z+b1)+b2, so J=W2 diag(1-tanh(.)^2) W1.
    This exact expression preserves weight gradients without Jacobian probes.
    """
    if states.device.type!='cuda':raise RuntimeError('Jacobian calculations require CUDA')
    if not (isinstance(field,nn.Sequential) and len(field)==3 and
            isinstance(field[0],nn.Linear) and isinstance(field[1],nn.Tanh) and isinstance(field[2],nn.Linear)):
        raise ValueError('exact Jacobian requires Linear-Tanh-Linear field')
    derivative=1-field[1](field[0](states)).square()
    jacobian=(field[2].weight[None,:,:]*derivative[...,None,:])@field[0].weight
    return jacobian.square().sum((-2,-1))


def jacobian_regularization(model,output,batch):
    # Treat visited locations as samples; the regularizer differentiates the
    # vector field parameters, rather than moving encoder states to avoid it.
    states=output['jacobian_states'][batch['valid']].detach()
    if not len(states):raise ValueError('Jacobian loss requires valid generated states')
    values=squared_jacobian_norm(model.forward_ode,states)
    return values.mean(),values.detach().sum(),len(states)
