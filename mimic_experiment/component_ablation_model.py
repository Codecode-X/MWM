"""Exact-budget alpha=.1 component ablations for ADNI and MIMIC."""
import math
import torch
from torch import nn
from paper_odeae.models import PaperModel, paper_model_from_config
from fixed_update_model import FixedUpdate


def unique_fields(model):
    return list({id(f): f for f in [model.backward_ode, model.forward_ode]}.values())


class ComponentModel(PaperModel):
    def ode_regularization(self):
        # Same coefficient per distinct ODE parameter, and same total budget.
        return sum(p.square().sum() for f in unique_fields(self) for p in f.parameters())*self.shared_l2_multiplier


def make_model(schema, cfg, alpha=.1):
    # Allocate the original model first, preserving every retained initial tensor.
    native_cfg = {**cfg, 'shared_vector_field': True}
    model = paper_model_from_config(schema, native_cfg, model_class=ComponentModel).cuda()
    model.update = FixedUpdate(model.update, model.latent, alpha)
    model.variant = cfg.get('variant', 'both'); model.generator_anchor = 'history cutoff'
    if cfg['ablation_component'] == 'separate_fields':
        d, width = model.latent, 94
        original = model.forward_ode
        encoder = nn.Sequential(nn.Linear(d, width), nn.Tanh(), nn.Linear(width, d, bias=False)).cuda()
        generator = nn.Sequential(nn.Linear(d, width), nn.Tanh(), nn.Linear(width, d)).cuda()
        with torch.no_grad():
            for field in [encoder, generator]:
                field[0].weight.copy_(original[0].weight[:width]); field[0].bias.copy_(original[0].bias[:width])
                # Match the original output-weight variance after halving width.
                field[2].weight.copy_(original[2].weight[:, :width]*math.sqrt(2.))
            generator[2].bias.copy_(original[2].bias)
        model.backward_ode, model.forward_ode = encoder, generator
        model.shared_vector_field = False
        assert not ({id(p) for p in encoder.parameters()} & {id(p) for p in generator.parameters()})
    assert model.latent == 28
    assert sum(p.numel() for p in model.parameters()) == cfg['target_parameters']
    assert sum(p.numel() for f in unique_fields(model) for p in f.parameters()) == 10744
    return model
