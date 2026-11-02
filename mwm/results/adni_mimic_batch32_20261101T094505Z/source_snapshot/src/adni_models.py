"""ADNI batched irregular-time adapters of the repository's two ODE models.

The original modules and parameterizations are reused. Each patient has its own
timestamps; padding never causes a GRU jump. Vectorized RK4 integrates the same
autonomous ODE in parallel, with a bounded step and a saved solver check.
"""
import math
import torch

from .model import LatentODE, ODERNN


def integrate(func, h, delta, steps):
    dt = delta[..., None] / steps
    for _ in range(steps):
        k1 = func(None, h)
        k2 = func(None, h + dt * k1 / 2)
        k3 = func(None, h + dt * k2 / 2)
        k4 = func(None, h + dt * k3)
        h = h + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return h


def forward(model, batch, kind, training=False, samples=4, step_multiplier=1):
    values, mask, times = batch['values'], batch['mask'], batch['times']
    context = batch['context_mask']
    steps = batch['steps']
    b, length, _ = values.shape
    if kind == 'ode_rnn':
        h = values.new_zeros(b, model.hidden)
        predictions = []
        for i in range(length):
            if i:
                h = integrate(model.ode, h, times[:, i] - times[:, i-1], steps[i] * step_multiplier)
            pred = model.decoder(h)
            predictions.append(pred)
            feed = context[:, i]
            observed = feed.bool()
            if training:
                observed = observed & (torch.rand(b, 1, device=values.device) < .5)
            filled = torch.where(observed, values[:, i], pred)
            # Future queries receive predictions, never truth or target masks.
            jump = torch.where(batch['is_future'][:, i, None], torch.ones_like(feed), feed)
            jump = jump * batch['valid'][:, i, None]
            h = model.cell(h, filled, jump)
        return {'prediction': torch.stack(predictions, 1)[None]}
    h = values.new_zeros(b, model.encoder.hidden)
    # Causal history encoding; posterior is anchored at the history cutoff,
    # matching this repository's extrapolation implementation.
    for i in range(batch['max_context']):
        if i:
            delta = (times[:, i] - times[:, i-1]) * (i < batch['cutoff'])
            h = integrate(model.encoder.ode, h, delta, steps[i] * step_multiplier)
        h = model.encoder.cell(h, values[:, i], context[:, i])
    mu, raw_scale = model.posterior(h).chunk(2, -1)
    std = torch.nn.functional.softplus(raw_scale) + 1e-5
    z = mu[None] + std[None] * torch.randn(samples, *mu.shape, device=mu.device)
    cutoff_time = times.gather(1, (batch['cutoff']-1)[:, None]).squeeze(1)
    prediction = values.new_zeros(samples, b, length, values.shape[-1])
    previous = cutoff_time
    # History reconstruction is unnecessary for the common future-only task.
    for i in range(length):
        future = batch['is_future'][:, i]
        delta = torch.where(future, times[:, i] - previous, torch.zeros_like(previous))
        n = max(1, math.ceil(batch['max_future_delta'][i] / batch['max_step'])) * step_multiplier
        z = integrate(model.ode, z, delta, n)
        prediction[:, :, i] = model.decoder(z)
        previous = torch.where(future, times[:, i], previous)
    kl = .5 * (mu.square() + std.square() - 1 - 2 * std.log()).sum(-1).mean()
    return {'prediction': prediction, 'kl': kl}


def make_model(kind, features, latent=32, hidden=64, width=64):
    cls = LatentODE if kind == 'latent_ode' else ODERNN
    # Same latent dynamics width and depth; native parameter counts are reported.
    return cls(features=features, latent=latent, hidden=hidden, width=width, layers=2)
