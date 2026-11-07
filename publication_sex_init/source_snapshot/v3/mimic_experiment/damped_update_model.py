"""Keep the original UPDATE layers; mix their proposal with the input state."""
from torch import nn

from paper_odeae.models import paper_model_from_config


class FixedDampedUpdate(nn.Sequential):
    def __init__(self, original, latent, alpha):
        super().__init__(*list(original.children()))
        if not 0 <= alpha <= 1:
            raise ValueError('alpha must lie in [0, 1]')
        self.latent = int(latent)
        self.alpha = float(alpha)

    def forward(self, inputs):
        proposal = super().forward(inputs)
        return (1 - self.alpha) * inputs[..., :self.latent] + self.alpha * proposal


def make_damped_model(schema, cfg, alpha=0.3):
    model = paper_model_from_config(schema, cfg).cuda()
    model.variant = cfg['variant']
    model.generator_anchor = cfg['generator_anchor']
    # Reuse the exact Linear/Tanh/Linear objects and state_dict keys.
    model.update = FixedDampedUpdate(model.update, model.latent, alpha)
    assert model.forward_ode is model.backward_ode
    assert sum(p.numel() for p in model.parameters()) == 43280
    return model
