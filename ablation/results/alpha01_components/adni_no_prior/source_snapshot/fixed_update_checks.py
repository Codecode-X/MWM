"""CUDA checks for the no-UPDATE endpoint and the ADNI fixed-alpha model."""
import hashlib
import torch
from paper_odeae.models import paper_model_from_config, encode_values, feature_errors
from fixed_update_model import make_model


def parameter_hash(model):
    h = hashlib.sha256()
    for p in model.parameters(): h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def gradient(value, params):
    params = tuple(params)
    gs = torch.autograd.grad(value, params, allow_unused=True, retain_graph=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p, g in zip(params, gs)])


def verify(model, batch, schema, cfg, loss_function):
    native = paper_model_from_config(schema, cfg).cuda().eval()
    native.load_state_dict(model.state_dict()); model.eval()
    before = parameter_hash(model); alpha = model.update.alpha
    model.update.alpha = 1
    a, b = native(batch, sample=False), model(batch, sample=False)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert torch.equal(loss_function(native, a, batch)[0], loss_function(model, b, batch)[0])
    model.update.alpha = alpha
    out = model(batch, sample=False)
    errors = feature_errors(out['prediction'], batch['values'], batch['mask'], schema)
    index = torch.arange(errors.shape[1], device='cuda')[None, :]
    factors = errors.new_tensor([cfg['mse_weight'] if f['kind']=='numerical' else cfg['cce_weight'] for f in schema])
    count = batch['mask'].sum((0, 1)).clamp_min(1)
    losses = [((errors*region[..., None]).sum((0, 1))/count*factors).sum()
              for region in [index < batch['cutoff'][:, None], index >= batch['cutoff'][:, None]]]
    norms = {name: {module: float(gradient(value, getattr(model, module).parameters()).norm())
                   for module in ['init', 'update', 'forward_ode', 'decoder']}
             for name, value in zip(['history', 'future'], losses)}
    for record in norms.values():
        assert record['init'] > 0 and record['forward_ode'] > 0 and record['decoder'] > 0
        assert (record['update'] == 0) if alpha == 0 else (record['update'] > 0)
    if alpha == 0:
        altered = {**batch, 'values': torch.zeros_like(batch['values'])}
        other = model(altered, sample=False)
        assert all(torch.equal(out[k], other[k]) for k in out)
    # Local derivative at identical states and observations, in float64.
    inputs = encode_values(batch['values'], batch['context_mask'], schema)
    state = out['mu'][:4].detach().double().requires_grad_(True)
    obs = inputs[:4, 0].detach().double(); model.update.double(); native.update.double()
    x = torch.cat([state, obs], -1); u = native.update(x); y = model.update(x)
    basis = torch.eye(model.latent, device='cuda', dtype=torch.float64)[:, None, :].expand(model.latent, len(state), model.latent)
    ju = torch.autograd.grad(u, state, basis, is_grads_batched=True, retain_graph=True)[0].permute(1, 0, 2)
    jy = torch.autograd.grad(y, state, basis, is_grads_batched=True)[0].permute(1, 0, 2)
    target = (1-alpha)*torch.eye(model.latent, device='cuda', dtype=torch.float64)[None]+alpha*ju
    error = float((jy-target).abs().max()); assert error < 1e-12
    model.update.float(); assert parameter_hash(model) == before
    return {'status': 'passed', 'alpha': alpha, 'physical_gpu': cfg['physical_gpu'],
            'parameter_device': str(next(model.parameters()).device), 'parameters': sum(p.numel() for p in model.parameters()),
            'same_parameter_hash_and_state_dict_keys': list(native.state_dict()) == list(model.state_dict()),
            'alpha1_forward_and_loss_bitwise_equal': True, 'alpha0_returns_input_state': alpha == 0,
            'alpha0_observation_values_do_not_change_outputs': alpha == 0,
            'jacobian_formula_max_absolute_error_float64': error,
            'history_encoder_gradient_stopped': False, 'gradient_norms': norms, 'optimizer_steps': 0,
            'actual_tensor_devices': {k: str(v.device) for k, v in batch.items() if torch.is_tensor(v)},
            'batch_kind': 'fixed training-patient implementation check, not a replay of a historical abnormal batch'}
