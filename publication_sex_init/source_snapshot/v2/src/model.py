"""Rubanova et al. (1907.03907v1), Algorithms 1/2 and Eq. (5)--(9).

Inputs use a sorted union timeline, values/masks [batch,time,feature]. Missing
values never update the recurrent state. No imputation, time embeddings, or
augmented generative dimensions are used in these two base models.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    def __init__(self, state_dim, width=50, layers=3):
        super().__init__()
        modules = []
        for i in range(layers):
            modules += [nn.Linear(state_dim if i == 0 else width, width), nn.Tanh()]
        modules += [nn.Linear(width, state_dim)]
        self.net = nn.Sequential(*modules)

    def forward(self, t, state):
        return self.net(state)


def solve(func, initial, times, rtol=1e-3, atol=1e-4):
    """Return [time,...,state], including the unchanged initial state."""
    if times.ndim != 1 or len(times) == 0:
        raise ValueError('times must be a nonempty vector')
    if not torch.isfinite(times).all():
        raise ValueError('solver times must be finite')
    delta = times[1:] - times[:-1]
    if len(delta) and not ((delta > 0).all() or (delta < 0).all()):
        raise ValueError('solver times must be strictly monotonic')
    if len(times) == 1:
        return initial.unsqueeze(0)
    return odeint(func, initial, times, method='dopri5', rtol=rtol, atol=atol)


class GRUUpdate(nn.Module):
    """Supplement, section 2.1. Mask is concatenated with observed values."""
    def __init__(self, features, hidden):
        super().__init__()
        self.update = nn.Linear(hidden + 2 * features, hidden)
        self.reset = nn.Linear(hidden + 2 * features, hidden)
        self.candidate = nn.Linear(hidden + 2 * features, hidden)

    def forward(self, h, x, mask):
        x = torch.cat([torch.where(mask.bool(), x, torch.zeros_like(x)), mask], -1)
        hx = torch.cat([h, x], -1)
        z, r = torch.sigmoid(self.update(hx)), torch.sigmoid(self.reset(hx))
        proposal = torch.tanh(self.candidate(torch.cat([r * h, x], -1)))
        updated = (1 - z) * proposal + z * h
        return torch.where(mask.bool().any(-1, keepdim=True), updated, h)


def validate_inputs(values, times, mask):
    if values.ndim != 3 or mask.shape != values.shape:
        raise ValueError('values and feature mask must have shape [batch,time,feature]')
    if times.ndim != 1 or len(times) != values.shape[1] or len(times) == 0:
        raise ValueError('times must be a shared nonempty timeline')
    if not torch.isfinite(times).all() or (times[1:] <= times[:-1]).any():
        raise ValueError('observation times must be finite and strictly increasing')
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError('mask must be binary')
    if not torch.isfinite(values[mask.bool()]).all():
        raise ValueError('observed values must be finite')


class ODERNNEncoder(nn.Module):
    def __init__(self, features, hidden=40, width=50, layers=3):
        super().__init__()
        self.hidden = hidden
        self.ode = ODEFunc(hidden, width, layers)
        self.cell = GRUUpdate(features, hidden)

    def forward(self, values, times, mask, backwards=True):
        validate_inputs(values, times, mask)
        h = values.new_zeros(values.shape[0], self.hidden)
        order = list(range(len(times)))
        if backwards:
            order.reverse()
        previous = order[0]
        states = []
        for j, index in enumerate(order):
            if j:
                h = solve(self.ode, h, times[[previous, index]])[-1]
            h = self.cell(h, values[:, index], mask[:, index])
            states.append(h)
            previous = index
        return h, torch.stack(states, 1)


class LatentODE(nn.Module):
    def __init__(self, features=6, latent=20, hidden=40, width=50, layers=3):
        super().__init__()
        self.encoder = ODERNNEncoder(features, hidden, width, layers)
        self.posterior = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2 * latent))
        self.ode = ODEFunc(latent, width, layers)
        self.decoder = nn.Linear(latent, features)
        self.classifier = nn.Sequential(nn.Linear(latent, 300), nn.ReLU(), nn.Linear(300, 1))

    def forward(self, values, times, mask, query_times=None, samples=3,
                task='interpolation', deterministic=False):
        if task not in ('interpolation', 'extrapolation'):
            raise ValueError('unknown task')
        if samples < 1:
            raise ValueError('samples must be positive')
        backwards = task == 'interpolation'
        h, _ = self.encoder(values, times, mask, backwards=backwards)
        mu, raw_scale = self.posterior(h).chunk(2, -1)
        std = F.softplus(raw_scale) + 1e-5
        z0 = mu.unsqueeze(0).expand(samples, -1, -1)
        if not deterministic:
            z0 = z0 + std.unsqueeze(0) * torch.randn_like(z0)
        query_times = times if query_times is None else query_times
        if query_times.ndim != 1 or len(query_times) == 0 or not torch.isfinite(query_times).all():
            raise ValueError('query times must be a finite nonempty vector')
        anchor = times[0] if backwards else times[-1]
        # Include the actual posterior anchor; sorting queries alone silently
        # assigns z0 to the wrong time when forecasting beyond the observations.
        timeline, inverse = torch.unique(torch.cat([anchor[None], query_times]), sorted=True, return_inverse=True)
        at = int(torch.searchsorted(timeline, anchor))
        right = solve(self.ode, z0, timeline[at:])
        if at:
            left = solve(self.ode, z0, timeline[:at+1].flip(0))
            trajectory = torch.cat([left[1:].flip(0), right], 0)
        else:
            trajectory = right
        z = trajectory[inverse[1:]].permute(1, 2, 0, 3)
        return {'prediction': self.decoder(z), 'mu': mu, 'std': std,
                'z0': z0, 'trajectory': z, 'logits': self.classifier(z0).squeeze(-1)}


class ODERNN(nn.Module):
    """Causal autoregressive model. Predict x_i before consuming x_i.

    Teacher forcing is 0.5 during training (supplement section 1). At test
    time available observations are consumed; missing future observations
    receive the previous model prediction as the next recurrent input.
    """
    def __init__(self, features=6, latent=20, hidden=40, width=50, layers=3):
        super().__init__()
        self.hidden = latent
        self.ode = ODEFunc(latent, width, layers)
        self.cell = GRUUpdate(features, latent)
        self.decoder = nn.Linear(latent, features)
        self.classifier = nn.Sequential(nn.Linear(latent, 300), nn.ReLU(), nn.Linear(300, 1))

    def forward(self, values, times, mask, teacher_forcing=1.0, **unused):
        validate_inputs(values, times, mask)
        if not 0 <= teacher_forcing <= 1:
            raise ValueError('teacher_forcing must be in [0,1]')
        h = values.new_zeros(values.shape[0], self.hidden)
        predictions, states = [], []
        for i in range(len(times)):
            if i:
                h = solve(self.ode, h, times[i-1:i+1])[-1]
            prediction = self.decoder(h)
            predictions.append(prediction)
            observed = mask[:, i].bool()
            if teacher_forcing < 1:
                use_truth = torch.rand(values.shape[0], 1, device=values.device) < teacher_forcing
                observed = observed & use_truth
            filled = torch.where(observed, values[:, i], prediction)
            # At an absent union time, do not perform an RNN jump. Forecast
            # separately marks actual per-patient prediction events.
            h = self.cell(h, filled, mask[:, i])
            states.append(h)
        return {'prediction': torch.stack(predictions, 1).unsqueeze(0),
                'trajectory': torch.stack(states, 1),
                'logits': self.classifier(h).squeeze(-1).unsqueeze(0)}

    def forecast(self, values, times, mask, future_times, future_mask=None):
        """Condition on history and feed predictions at each requested future time."""
        validate_inputs(values, times, mask)
        if future_times.ndim != 1 or len(future_times) == 0 or not torch.isfinite(future_times).all():
            raise ValueError('future times must be a finite nonempty vector')
        if (future_times <= times[-1]).any() or (future_times[1:] <= future_times[:-1]).any():
            raise ValueError('future times must increase strictly after the history')
        history = self(values, times, mask)
        h = history['trajectory'][:, -1]
        previous = times[-1]
        predictions = []
        for index, time in enumerate(future_times):
            h = solve(self.ode, h, torch.stack([previous, time]))[-1]
            prediction = self.decoder(h)
            predictions.append(prediction)
            h = self.cell(h, prediction, torch.ones_like(prediction) if future_mask is None else future_mask[:, index])
            previous = time
        return torch.stack(predictions, 1).unsqueeze(0)


def gaussian_log_likelihood(prediction, target, mask, variance=0.01):
    """Per sample, per sequence sum over observed scalars: [sample,batch]."""
    if variance <= 0:
        raise ValueError('variance must be positive')
    safe_target = torch.where(mask.bool(), target, torch.zeros_like(target))
    logp = -0.5 * ((prediction - safe_target).square() / variance + math.log(2 * math.pi * variance))
    return torch.where(mask.bool().unsqueeze(0), logp, torch.zeros_like(logp)).sum((-1, -2))


def objective(output, target, mask, labels=None, classification_weight=0., variance=0.01):
    """Negative Eq. (9), analytic KL to N(0,I), beta=1, optional BCE."""
    logp = gaussian_log_likelihood(output['prediction'], target, mask, variance)
    nll = -logp.mean()
    kl = nll.new_zeros(())
    if 'mu' in output:
        mu, std = output['mu'], output['std']
        kl = 0.5 * (mu.square() + std.square() - 1 - 2 * std.log()).sum(-1).mean()
    ce = nll.new_zeros(())
    if labels is not None and classification_weight:
        ce = F.binary_cross_entropy_with_logits(output['logits'], labels[None].expand_as(output['logits']))
    safe = torch.where(mask.bool(), target, torch.zeros_like(target))
    mse = torch.where(mask.bool(), (output['prediction'].mean(0) - safe).square(), torch.zeros_like(target)).sum() / mask.sum().clamp_min(1)
    return {'loss': nll + kl + classification_weight * ce, 'nll': nll, 'kl': kl, 'bce': ce, 'mse': mse}


MODELS = {'latent_ode': LatentODE, 'ode_rnn': ODERNN}
