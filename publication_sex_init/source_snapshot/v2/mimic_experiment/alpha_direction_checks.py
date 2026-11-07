"""CUDA checks against the established reverse encoder and gated forward model."""
import torch
from torch import nn
from paper_odeae.models import encode_values
from fixed_update_model import make_model as make_forward, FixedUpdate
from fixed_update_checks import parameter_hash, gradient
from alpha_direction_model import make_model
from reference_history_encoders import HistoryEncoderAblation


def verify(model, batch, schema, cfg, loss_function):
    before = parameter_hash(model); model.eval()
    forward = make_forward(schema, cfg, .1).eval(); forward.load_state_dict(model.state_dict())
    assert parameter_hash(forward) == before
    counterpart_cfg = {**cfg, 'history_read_direction': 'reverse' if cfg['history_read_direction']=='forward' else 'forward'}
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    counterpart = make_model(schema, counterpart_cfg, .1)
    assert parameter_hash(counterpart) == before
    torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
    del counterpart
    calls = []; updates = []
    solve = model.solve
    def recorded(field, h, dt):
        calls.append(dt.detach().clone()); return solve(field, h, dt)
    model.solve = recorded
    hook = model.update.register_forward_pre_hook(lambda module, args: updates.append(args[0].detach()))
    out = model(batch, sample=False); hook.remove(); model.solve = solve
    max_context = batch['max_context']; reverse = cfg['history_read_direction']=='reverse'
    if reverse:
        assert len(updates) == 2*max_context
        assert all(bool((dt <= 1e-6).all()) for dt in calls[:max_context])
        assert all(bool((dt >= -1e-6).all()) for dt in calls[max_context:])
        reference = HistoryEncoderAblation(schema, {**cfg, 'history_encoder_variant': 'reverse_shared'}).cuda().eval()
        reference.update = FixedUpdate(reference.update, model.latent, .1)
        reference.load_state_dict(model.state_dict())
        expected = reference(batch, sample=False)
        for key in expected: assert torch.equal(expected[key], out[key]), key
        del reference, expected
    else:
        expected = forward(batch, sample=False)
        for key in expected: assert torch.equal(expected[key], out[key]), key
    forward_output = forward(batch, sample=False)
    assert torch.equal(out['prior_prediction'], forward_output['prior_prediction'])
    # Future targets cannot influence the encoder or its generated predictions.
    index = torch.arange(batch['values'].shape[1], device='cuda')[None, :]
    future = index >= batch['cutoff'][:, None]
    altered = {**batch, 'values': torch.where(future[..., None], torch.zeros_like(batch['values']), batch['values'])}
    other = model(altered, sample=False)
    for key in out: assert torch.equal(out[key], other[key]), key
    # With one historical visit, both directions have the same anchor and outputs.
    one = {**batch, 'cutoff': torch.ones_like(batch['cutoff']), 'max_context': 1,
           'context_mask': batch['context_mask']*(index == 0)[..., None]}
    a, b = model(one, sample=False), forward(one, sample=False)
    assert torch.equal(a['mu'], b['mu'])
    assert torch.equal(a['prediction'][one['valid']], b['prediction'][one['valid']])
    loss, _ = loss_function(model, out, batch)
    norms = {name: float(gradient(loss, getattr(model, name).parameters()).norm()) for name in ['init', 'update', 'forward_ode', 'decoder']}
    assert all(v > 0 for v in norms.values())
    loss.backward(); assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    x = updates[0][:4].detach().double().requires_grad_(True); model.update.double()
    u = nn.Sequential(*list(model.update.children()))(x); y = model.update(x)
    basis = torch.eye(28, device='cuda', dtype=torch.float64)[:, None, :].expand(28, len(x), 28)
    ju = torch.autograd.grad(u, x, basis, is_grads_batched=True, retain_graph=True)[0][:, :, :28].permute(1, 0, 2)
    jy = torch.autograd.grad(y, x, basis, is_grads_batched=True)[0][:, :, :28].permute(1, 0, 2)
    error = float((jy-(.9*torch.eye(28, device='cuda', dtype=torch.float64)[None]+.1*ju)).abs().max()); assert error < 1e-12
    model.update.float(); model.zero_grad(set_to_none=True); assert parameter_hash(model) == before
    return {'status': 'passed', 'dataset': cfg['dataset'], 'history_read_direction': cfg['history_read_direction'],
            'physical_gpu': cfg['physical_gpu'], 'parameter_device': str(next(model.parameters()).device),
            'parameters': sum(p.numel() for p in model.parameters()), 'alpha': .1,
            'same_initial_tensors_for_directions': True, 'reference_forward_or_reverse_output_bitwise_equal': True,
            'causal_prior_bitwise_equal_to_forward_alpha01': True, 'future_observations_do_not_change_outputs': True,
            'single_visit_directions_equal': True, 'update_calls_in_main_and_auxiliary': len(updates),
            'reverse_ode_deltas_nonpositive': reverse, 'generation_and_prior_deltas_nonnegative': reverse,
            'jacobian_formula_max_absolute_error_float64': error, 'total_loss_gradient_norms': norms,
            'optimizer_steps': 0, 'batch_kind': 'fixed training batch implementation checks',
            'actual_tensor_devices': {k: str(v.device) for k, v in batch.items() if torch.is_tensor(v)}}
