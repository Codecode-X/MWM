"""Fixed-alpha history direction ablation, with one shared vector field."""
import torch
from paper_odeae.models import PaperModel, paper_model_from_config, encode_values
from fixed_update_model import FixedUpdate


class AlphaDirectionModel(PaperModel):
    def initial_state(self, batch):
        last = batch['ages'].gather(1, (batch['cutoff']-1)[:, None]).squeeze(1)
        return self.init(batch.get('initialization_age', last)[:, None]/100.)

    def causal_priors(self, batch, inputs):
        state = self.initial_state(batch); previous = batch['ages'][:, 0]; priors = []
        rows = torch.arange(len(state), device=state.device); cutoff = batch['cutoff']
        for j in range(batch['max_context']):
            index = torch.minimum(torch.full_like(cutoff, j), cutoff-1); active = j < cutoff
            time = batch['ages'].gather(1, index[:, None]).squeeze(1)
            evolved = self.solve(self.forward_ode, state, torch.where(active, time-previous, torch.zeros_like(time)))
            priors.append(self.decoder(evolved))
            proposal = self.update(torch.cat([evolved, inputs[rows, index]], -1))
            state = torch.where(active[:, None], proposal, state); previous = torch.where(active, time, previous)
        return torch.stack(priors, 1)

    def reverse_state(self, batch, inputs):
        state = self.initial_state(batch); cutoff = batch['cutoff']
        previous = batch['ages'].gather(1, (cutoff-1)[:, None]).squeeze(1)
        rows = torch.arange(len(state), device=state.device)
        for j in range(batch['max_context']):
            index = (cutoff-1-j).clamp_min(0); active = j < cutoff
            time = batch['ages'].gather(1, index[:, None]).squeeze(1)
            evolved = self.solve(self.forward_ode, state, torch.where(active, time-previous, torch.zeros_like(time)))
            proposal = self.update(torch.cat([evolved, inputs[rows, index]], -1))
            state = torch.where(active[:, None], proposal, state); previous = torch.where(active, time, previous)
        return state

    def generate_from_first(self, state, batch):
        previous = batch['ages'][:, 0]; predictions, states = [], []
        for j in range(batch['ages'].shape[1]):
            if j:
                active = batch['valid'][:, j]; time = batch['ages'][:, j]
                state = self.solve(self.forward_ode, state, torch.where(active, time-previous, torch.zeros_like(time)))
                previous = torch.where(active, time, previous)
            predictions.append(self.decoder(state)); states.append(state)
        return torch.stack(predictions, 1), torch.stack(states, 1)

    def forward(self, batch, sample=False):
        if self.history_read_direction == 'forward': return super().forward(batch, sample=sample)
        if batch['values'].device.type != 'cuda': raise RuntimeError('CUDA required')
        inputs = encode_values(batch['values'], batch['context_mask'], self.schema)
        state = self.reverse_state(batch, inputs)
        # Same causal weak-prior supervision, with alpha applied to this pass too.
        # This state is never used as the main reverse reconstruction anchor.
        priors = self.causal_priors(batch, inputs)
        predictions, states = self.generate_from_first(state, batch)
        return {'prediction': predictions, 'mu': state, 'std': torch.zeros_like(state), 'kl': state.new_zeros(()),
                'prior_prediction': priors, 'jacobian_states': states}


def make_model(schema, cfg, alpha):
    model = paper_model_from_config(schema, cfg, model_class=AlphaDirectionModel).cuda()
    model.history_read_direction = cfg['history_read_direction']
    model.update = FixedUpdate(model.update, model.latent, alpha)
    model.variant = cfg.get('variant', 'both')
    if model.history_read_direction == 'reverse':
        model.encoder_direction = 'backward'; model.generator_anchor = 'first history visit'
    else:
        model.generator_anchor = 'history cutoff'
    assert model.forward_ode is model.backward_ode and model.latent == 28
    assert sum(p.numel() for p in model.parameters()) == cfg['target_parameters']
    return model
