"""Frozen-configuration publication models; all computations require CUDA."""
import hashlib
import math
import torch
from torch import nn
from paper_odeae.models import PaperModel, mlp, encode_values, feature_errors, reconstruction, prior_errors
from paper_odeae.jacobian import squared_jacobian_norm
from mimic_experiment.fixed_update_model import FixedUpdate


def fields(model):
    return list({id(f): f for f in [model.backward_ode, model.forward_ode]}.values())


def jacobian_regularization(model, output, batch):
    states = output["jacobian_states"][batch["valid"]].detach()
    values = torch.stack([squared_jacobian_norm(f, states) for f in fields(model)]).mean(0)
    return values.mean(), values.detach().sum(), len(states)


class PublicationModel(PaperModel):
    def __init__(self, schema, cfg):
        self.cfg = dict(cfg)
        super().__init__(schema, "deterministic", cfg["latent"], cfg["ode_width"], cfg["mlp_width"],
                         cfg["rtol"], cfg["atol"], "forward", "age",
                         init_width=cfg.get("init_width"), update_width=cfg.get("update_width"),
                         decoder_width=cfg.get("decoder_width"), shared_vector_field=True,
                         collect_prior_predictions=cfg["prior_weight"] > 0,
                         collect_jacobian_states=True)
        self.sex_index = next(i for i, f in enumerate(schema) if f["name"] == "sex")
        self.variant = cfg["variant"]
        self.history_order = cfg.get("history_order", "chronological")
        self.generator_anchor = "first history visit" if self.history_order == "reverse" else "history cutoff"
        self.update = FixedUpdate(self.update, self.latent, cfg["update_alpha"])
        if self.variant == "separate_fields":
            assert self.latent == 28 and cfg["ode_width"] == 188
            original = self.forward_ode
            encoder = nn.Sequential(nn.Linear(28, 94), nn.Tanh(), nn.Linear(94, 28, bias=False))
            generator = mlp(28, 28, 94)
            with torch.no_grad():
                for f in [encoder, generator]:
                    f[0].weight.copy_(original[0].weight[:94]); f[0].bias.copy_(original[0].bias[:94])
                    f[2].weight.copy_(original[2].weight[:, :94] * math.sqrt(2))
                generator[2].bias.copy_(original[2].bias)
            self.backward_ode, self.forward_ode = encoder, generator
            self.shared_vector_field = False
        if self.variant == "attention":
            from paper_odeae.history_encoders import CausalTimeAttention
            from paper_odeae.models import input_width
            del self.update
            self.history_attention = CausalTimeAttention(input_width(schema), self.latent,
                                                        "year" if cfg["dataset"] == "adni" else "hour")
        self.encoder_initialization = cfg.get("init_covariate", "sex")

    def initialization_scalar(self, batch):
        source = self.cfg.get("init_covariate", "sex")
        if source == "sex":
            # Cached ADNI labels: M=0/F=1; MIMIC: F=0/M=1.
            value = batch["values"][:, 0, self.sex_index]
            known = batch["mask"][:, 0, self.sex_index].bool()
            if self.cfg["dataset"] == "adni":
                value = 1 - value
            if not bool((~known | ((value == 0) | (value == 1))).all()):
                raise ValueError("Invalid sex category")
            return torch.where(known, value, value.new_full(value.shape, self.cfg["sex_missing_fill"]))
        if source == "zero_scalar":
            return batch["values"].new_zeros(len(batch["values"]))
        cutoff_age = batch.get("initialization_age")
        if cutoff_age is None:
            cutoff_age = batch["ages"].gather(1, (batch["cutoff"] - 1)[:, None]).squeeze(1)
        if source == "cutoff_age":
            return cutoff_age / 100
        if source == "first_age":
            if self.cfg["dataset"] == "adni":
                return batch["ages"][:, 0] / 100
            elapsed = batch["ages"].gather(1, (batch["cutoff"] - 1)[:, None]).squeeze(1) - batch["ages"][:, 0]
            return (cutoff_age - elapsed / 8766) / 100
        raise ValueError(source)

    def initial_state(self, batch):
        return self.init(self.initialization_scalar(batch)[:, None])

    def encode(self, batch, return_trace=False):
        inputs = encode_values(batch["values"], batch["context_mask"], self.schema)
        if self.variant == "attention":
            prefix = self.history_attention(inputs[:, :batch["max_context"]],
                                            batch["ages"][:, :batch["max_context"]],
                                            batch["cutoff"], self.initial_state(batch))
            anchor = prefix[torch.arange(len(prefix), device="cuda"), batch["cutoff"] - 1]
            return anchor, [], []
        state = self.initial_state(batch)
        cuts = batch["cutoff"]; rows = torch.arange(len(cuts), device=state.device)
        previous = batch["ages"].gather(1, (cuts-1)[:, None]).squeeze(1) if self.history_order == "reverse" else batch["ages"][:, 0]
        priors, trace = [], []
        for j in range(batch["max_context"]):
            index = (cuts - 1 - j).clamp_min(0) if self.history_order == "reverse" else torch.minimum(torch.full_like(cuts, j), cuts-1)
            active = j < cuts
            time = batch["ages"].gather(1, index[:, None]).squeeze(1)
            delta = torch.where(active, time - previous, torch.zeros_like(time))
            evolved = state if self.variant == "no_encoder_ode" else self.solve(self.backward_ode, state, delta)
            if self.collect_prior_predictions:
                priors.append(self.decoder(evolved))
            proposed = self.update(torch.cat([evolved, inputs[rows, index]], -1))
            state = torch.where(active[:, None], proposed, state)
            if return_trace:
                trace.append({"index": index, "active": active, "before": evolved, "after": state, "time": time})
            previous = torch.where(active, time, previous)
        return state, priors, trace

    def generate_from_first(self, state, batch):
        previous = batch["ages"][:, 0]; predictions, states = [], []
        for j in range(batch["ages"].shape[1]):
            if j:
                active = batch["valid"][:, j]; time = batch["ages"][:, j]
                state = self.solve(self.forward_ode, state, torch.where(active, time-previous, torch.zeros_like(time)))
                previous = torch.where(active, time, previous)
            predictions.append(self.decoder(state)); states.append(state)
        return torch.stack(predictions, 1), torch.stack(states, 1)

    def forward(self, batch, sample=False):
        if batch["values"].device.type != "cuda":
            raise RuntimeError("CUDA model execution required")
        state, priors, _ = self.encode(batch)
        if self.history_order == "reverse":
            prediction, states = self.generate_from_first(state, batch)
        else:
            prediction, states = self.generate_from_cutoff(state, batch, return_states=True)
        out = {"prediction": prediction, "mu": state, "std": torch.zeros_like(state),
               "kl": state.new_zeros(()), "jacobian_states": states}
        if self.collect_prior_predictions:
            if self.history_order != "chronological":
                raise ValueError("Prior intervention is chronological only")
            out["prior_prediction"] = torch.stack(priors, 1)
        return out


def make_model(schema, cfg):
    family = cfg.get("family", "publication")
    if family == "publication":
        model = PublicationModel(schema, cfg)
    elif family == "matched_baseline":
        from capacity_matched.models import make_model as matched
        model = matched(schema, cfg)
    elif family == "deterministic_baseline":
        model = PaperModel(schema, kind="deterministic", latent=cfg["latent"],
                           ode_width=cfg["ode_width"], mlp_width=cfg["mlp_width"],
                           rtol=cfg["rtol"], atol=cfg["atol"],
                           init_width=cfg.get("init_width"), update_width=cfg.get("update_width"),
                           decoder_width=cfg.get("decoder_width"),
                           encoder_ode_width=cfg.get("encoder_ode_width"),
                           generator_ode_width=cfg.get("generator_ode_width"))
    elif family == "native_baseline":
        if cfg["dataset"] == "mimic":
            from mimic_experiment.baseline_models import make_model as native
            model = native(schema, cfg)
        else:
            from paper_odeae.models import paper_model_from_config
            model = paper_model_from_config(schema, cfg)
    else:
        raise ValueError(family)
    return model.cuda()


def objective(model, out, batch, schema, cfg):
    mse, ce = reconstruction(feature_errors(out["prediction"], batch["values"], batch["mask"], schema),
                             batch["mask"], schema)
    zero = mse.new_zeros(())
    prior = zero
    if cfg["prior_weight"]:
        errors, mask = prior_errors(out, batch, schema)
        a, b = reconstruction(errors, mask, schema); prior = a+b
    jac = jacobian_regularization(model, out, batch)[0] if cfg["jacobian_weight"] else zero
    l2 = model.ode_regularization(); kl = out["kl"]
    loss = mse + ce + cfg["kl_weight"]*kl + cfg["l2_weight"]*l2
    loss = loss + cfg["prior_weight"]*prior + cfg["jacobian_weight"]*jac
    idx = torch.arange(batch["values"].shape[1], device="cuda")[None]
    region_losses = {}
    for key, region in [("history", idx < batch["cutoff"][:, None]),
                        ("future", (idx >= batch["cutoff"][:, None]) & batch["valid"])]:
        mask = batch["mask"] * region[..., None]
        a, b = reconstruction(feature_errors(out["prediction"], batch["values"], mask, schema), mask, schema)
        region_losses[key] = a+b
    return loss, {"mse": mse, "cce": ce, "kl": kl, "prior": prior, "jacobian": jac,
                  "ode_l2": l2, "loss": loss, **region_losses}


def parameter_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
