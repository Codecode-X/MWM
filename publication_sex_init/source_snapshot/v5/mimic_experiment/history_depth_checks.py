"""CUDA forward, loss-gradient, and per-patient depth-boundary checks."""
import torch

from direction_gradient_common import decompose
from mechanism_common import parameter_hash, relative
from history_depth_model import make_depth_model
from paper_odeae.models import encode_values, prior_errors, reconstruction
from paper_odeae.jacobian import jacobian_regularization
from mimic_experiment.train_full_batch import make_model, loss_function
from mimic_experiment.prepare import atomic


def gradients(value, params):
    params = tuple(params)
    gs = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p, g in zip(params, gs)])


def verify(root, folder, manifest, cfg, payload, gpu):
    saved = torch.load(manifest['abnormal_snapshot'], map_location='cuda', weights_only=False)
    batch = saved['probe_batch']; schema = payload['schema']
    native = make_model(schema, cfg).cuda().eval()
    depth = make_depth_model(schema, cfg).eval()
    for model in [native, depth]:
        model.load_state_dict(saved['state_dict'], strict=True)
    before = parameter_hash(depth)
    original, truncated = native(batch, sample=False), depth(batch, sample=False)
    assert all(torch.equal(original[k], truncated[k]) for k in original)
    assert torch.equal(depth.history_cutoff(batch), original['mu'])
    ln, _ = loss_function(native, original, batch, schema, cfg)
    lt, _ = loss_function(depth, truncated, batch, schema, cfg)
    assert torch.equal(ln, lt)
    hn, fn = decompose(native, original, batch, schema, cfg)
    ht, ft = decompose(depth, truncated, batch, schema, cfg)
    future_error = relative(gradients(ft, depth.parameters()), gradients(fn, native.parameters()))
    assert future_error < 1e-6
    def rest(model, out):
        pe, pm = prior_errors(out, batch, schema); mse, ce = reconstruction(pe, pm, schema)
        jac, _, _ = jacobian_regularization(model, out, batch)
        return cfg['prior_weight']*(mse+ce)+cfg['jacobian_weight']*jac+cfg['l2_weight']*model.ode_regularization()
    rest_error = relative(gradients(rest(depth, truncated), depth.parameters()), gradients(rest(native, original), native.parameters()))
    assert rest_error < 1e-6
    hnorms = {name: float(gradients(ht, getattr(depth, name).parameters()).norm()) for name in ['init', 'update', 'forward_ode', 'decoder']}
    assert hnorms['update'] > 0 and hnorms['forward_ode'] > 0 and hnorms['decoder'] > 0

    # Leaf observation inputs make the exact per-patient boundary directly testable.
    # Keep the real batch's solver grids; differentiate sum(cutoff) w.r.t. each input.
    original_encode_values = encode_values
    import history_depth_model as module
    encoded = original_encode_values(batch['values'], batch['context_mask'], schema).detach().requires_grad_(True)
    module.encode_values = lambda *args: encoded
    try:
        cutoff_state = depth.history_cutoff(batch)
        input_grads = torch.autograd.grad(cutoff_state.sum(), encoded)[0]
    finally:
        module.encode_values = original_encode_values
    idx = torch.arange(encoded.shape[1], device='cuda')[None, :]
    counts = batch['cutoff']; boundary = (counts-8).clamp_min(0)
    prefix = idx < boundary[:, None]
    tail = (idx >= boundary[:, None]) & (idx < counts[:, None])
    assert not bool(input_grads[prefix].count_nonzero())
    tail_norm = input_grads.norm(dim=-1)
    assert bool((tail_norm[tail] > 0).all())
    records = [{'patient_index': i, 'subject_id': str(batch['subject_ids'][i]), 'history_count': int(counts[i]),
                'first_differentiable_visit_0based': int(boundary[i]),
                'observation_gradient_norms': tail_norm[i, :int(counts[i])].detach().cpu().tolist()}
               for i in range(len(counts))]
    # With a depth >= maximum history length, recover the original full gradient.
    depth.history_depth = batch['max_context']
    full = depth(batch, sample=False)
    hf, _ = decompose(depth, full, batch, schema, cfg)
    full_error = relative(gradients(hf, depth.parameters()), gradients(hn, native.parameters()))
    assert full_error < 1e-6
    depth.history_depth = 8
    assert parameter_hash(depth) == before
    atomic(folder/'checks.json', {'status': 'passed', 'physical_gpu': gpu, 'parameter_device': str(next(depth.parameters()).device),
           'actual_tensor_devices': {k: str(v.device) for k, v in batch.items() if torch.is_tensor(v)},
           'parameters': 43280, 'latent': 28, 'history_max_encoder_depth': 8,
           'forward_outputs_bitwise_equal': True, 'scalar_loss_bitwise_equal': True, 'reencoded_cutoff_bitwise_equal': True,
           'future_all_parameter_gradient_relative_error': future_error, 'prior_jacobian_l2_gradient_relative_error': rest_error,
           'unlimited_depth_history_gradient_relative_error': full_error, 'history_gradient_norms': hnorms,
           'prefix_observation_gradients_exactly_zero': True, 'all_last8_observation_gradient_norms_positive': True,
           'patient_boundary_checks': records, 'optimizer_steps': 0, 'snapshot': manifest['abnormal_snapshot'],
           'batch_kind': 'saved verified e46b121 training replay batch'})
