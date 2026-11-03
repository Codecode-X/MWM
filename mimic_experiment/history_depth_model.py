"""Original forward values; history-specific encoder BPTT depth limit."""
import torch

from paper_odeae.models import PaperModel, paper_model_from_config, encode_values, input_width


class HistoryDepthModel(PaperModel):
    history_depth = 8

    def history_cutoff(self, batch):
        """Recompute identical encoding, detaching once per long-history row.

        The original encoding graph is retained by the native forward method
        for future prediction and weak prior. Only this history-loss branch
        detaches immediately BEFORE the last `history_depth` ODE/UPDATE steps.
        """
        values, ages, cutoff = batch['values'], batch['ages'], batch['cutoff']
        inputs = encode_values(values, batch['context_mask'], self.schema)
        h = self.init(batch['initialization_age'][:, None]/100.)
        previous = ages[:, 0]
        for j in range(batch['max_context']):
            boundary = (cutoff > self.history_depth) & (j == cutoff-self.history_depth)
            h = torch.where(boundary[:, None], h.detach(), h)
            index = torch.minimum(torch.full_like(cutoff, j), cutoff-1)
            active = j < cutoff
            age = ages.gather(1, index[:, None]).squeeze(1)
            x = inputs[torch.arange(len(cutoff), device=values.device), index]
            evolved = self.solve(self.backward_ode, h, torch.where(active, age-previous, torch.zeros_like(age)))
            proposal = self.update(torch.cat([evolved, x], -1))
            h = torch.where(active[:, None], proposal, h)
            previous = torch.where(active, age, previous)
        return h

    def generate_from_cutoff(self, z, batch, return_states=False):
        historical = z
        if torch.is_grad_enabled() and bool((batch['cutoff'] > self.history_depth).any()):
            tail = self.history_cutoff(batch)
            # Exact original forward values; history gradients use the tail graph.
            historical = z.detach() + (tail-tail.detach())
        ages, cutoff = batch['ages'], batch['cutoff']
        b, n = ages.shape
        anchor = ages.gather(1, (cutoff-1)[:, None]).squeeze(1)
        predictions = [z.new_zeros(b, input_width(self.schema)) for _ in range(n)]
        states = [torch.zeros_like(z) for _ in range(n)] if return_states else None
        state, previous = historical, anchor
        for i in reversed(range(batch['max_context'])):
            active = i < cutoff
            delta = torch.where(active, ages[:, i]-previous, torch.zeros_like(previous))
            state = self.solve(self.forward_ode, state, delta)
            predictions[i] = torch.where(active[:, None], self.decoder(state), predictions[i])
            if return_states:
                states[i] = torch.where(active[:, None], state, states[i])
            previous = torch.where(active, ages[:, i], previous)
        state, previous = z, anchor
        for i in range(n):
            active = (i >= cutoff) & batch['valid'][:, i]
            delta = torch.where(active, ages[:, i]-previous, torch.zeros_like(previous))
            state = self.solve(self.forward_ode, state, delta)
            predictions[i] = torch.where(active[:, None], self.decoder(state), predictions[i])
            if return_states:
                states[i] = torch.where(active[:, None], state, states[i])
            previous = torch.where(active, ages[:, i], previous)
        prediction = torch.stack(predictions, 1)
        return (prediction, torch.stack(states, 1)) if return_states else prediction


def make_depth_model(schema, cfg, depth=8):
    model = paper_model_from_config(schema, cfg, model_class=HistoryDepthModel).cuda()
    model.variant, model.generator_anchor = cfg['variant'], cfg['generator_anchor']
    model.history_depth = int(depth)
    assert model.history_depth >= 0
    assert model.forward_ode is model.backward_ode
    assert sum(p.numel() for p in model.parameters()) == 43280
    return model
