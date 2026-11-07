"""Dataset-independent fixed UPDATE mixing, including the no-UPDATE endpoint."""
from torch import nn
from paper_odeae.models import paper_model_from_config


class FixedUpdate(nn.Sequential):
    def __init__(self, network, latent, alpha):
        super().__init__(*list(network.children()))
        self.latent, self.alpha = int(latent), float(alpha)
        if not 0 <= self.alpha <= 1: raise ValueError('Invalid alpha')

    def forward(self, inputs):
        state = inputs[..., :self.latent]
        if self.alpha == 0:
            return state
        return (1-self.alpha)*state+self.alpha*super().forward(inputs)


def make_model(schema, cfg, alpha):
    model = paper_model_from_config(schema, cfg).cuda()
    model.update = FixedUpdate(model.update, model.latent, alpha)
    model.variant = cfg.get('variant', 'both')
    model.generator_anchor = cfg.get('generator_anchor', 'history cutoff')
    assert model.forward_ode is model.backward_ode and model.latent == 28
    if cfg.get('target_parameters'):
        assert sum(p.numel() for p in model.parameters()) == cfg['target_parameters']
    return model
