"""Repository Latent ODE / ODE-RNN adapters for the ODE-AE mixed-field task.

Reuse src.model networks. Values and masks are expanded to the same one-hot
width; categorical predictions are logits for scoring and probabilities for
autoregressive feedback. The latent posterior retains its cutoff anchor.
"""
import torch
from torch import nn

from src.adni_models import make_model
from .models import encode_values, input_width
from .solver import integrate


def expanded_mask(mask, schema):
    return torch.cat([mask[..., j:j+1].expand(*mask.shape[:-1], f.get('classes', 1))
                      for j, f in enumerate(schema)], -1)


def feedback(logits, schema):
    parts, at = [], 0
    for field in schema:
        width = field.get('classes', 1)
        value = logits[..., at:at+width]
        parts.append(value.softmax(-1) if field['kind']=='categorical' else value)
        at += width
    return torch.cat(parts, -1)


class BaselineModel(nn.Module):
    def __init__(self, schema, kind, latent=32, hidden=64, width=64, rtol=1e-4, atol=1e-6):
        super().__init__()
        if kind not in ['latent_ode', 'ode_rnn']:
            raise ValueError(kind)
        self.schema, self.kind = schema, kind
        self.rtol, self.atol = rtol, atol
        self.core = make_model(kind, input_width(schema), latent, hidden, width)
        # The previous classification head is unrelated to mixed-field decoding.
        self.core.classifier.requires_grad_(False)

    def solve(self, func, state, delta):
        return integrate(lambda z: func(None, z), state, delta, self.rtol, self.atol)

    def forward(self, batch, sample=False):
        values, ages = batch['values'], batch['ages']
        if values.device.type!='cuda' or next(self.parameters()).device!=values.device:
            raise RuntimeError('baseline model execution requires CUDA')
        context = expanded_mask(batch['context_mask'], self.schema)
        inputs = encode_values(values, batch['context_mask'], self.schema)
        # Preserve the clock used in the earlier ADNI adapters: one unit/decade.
        times = (ages-ages[:, :1])/10.
        b, n, _ = values.shape
        if self.kind=='ode_rnn':
            state = values.new_zeros(b, self.core.hidden)
            predictions = []
            for i in range(n):
                if i:
                    state = self.solve(self.core.ode, state, times[:,i]-times[:,i-1])
                prediction = self.core.decoder(state)
                predictions.append(prediction)
                available = context[:,i].bool()
                if self.training:
                    available = available & (torch.rand(b,1,device=values.device)<.5)
                filled = torch.where(available, inputs[:,i], feedback(prediction, self.schema))
                future = i>=batch['cutoff']
                jump_mask = torch.where(future[:,None], torch.ones_like(context[:,i]), context[:,i])
                jump_mask = jump_mask*batch['valid'][:,i,None]
                state = self.core.cell(state, filled, jump_mask)
            return {'prediction': torch.stack(predictions,1), 'kl': state.new_zeros(())}
        state = values.new_zeros(b, self.core.encoder.hidden)
        for i in range(batch['max_context']):
            if i:
                delta = (times[:,i]-times[:,i-1])*(i<batch['cutoff'])
                state = self.solve(self.core.encoder.ode, state, delta)
            state = self.core.encoder.cell(state, inputs[:,i], context[:,i])
        mu, raw_scale = self.core.posterior(state).chunk(2,-1)
        std = torch.nn.functional.softplus(raw_scale)+1e-5
        z = mu+std*torch.randn_like(mu) if sample else mu
        cutoff_time = times.gather(1,(batch['cutoff']-1)[:,None]).squeeze(1)
        # Reconstruct history by integrating back from the actual posterior anchor.
        z = self.solve(self.core.ode, z, times[:,0]-cutoff_time)
        predictions = [self.core.decoder(z)]
        for i in range(1,n):
            z = self.solve(self.core.ode, z, times[:,i]-times[:,i-1])
            predictions.append(self.core.decoder(z))
        kl = .5*(mu.square()+std.square()-1-2*std.log()).sum(-1).mean()
        return {'prediction': torch.stack(predictions,1), 'mu': mu, 'std': std, 'kl': kl}

    def ode_regularization(self):
        networks = [self.core.ode]
        if self.kind=='latent_ode':
            networks.append(self.core.encoder.ode)
        return sum(p.square().sum() for net in networks for p in net.parameters())
