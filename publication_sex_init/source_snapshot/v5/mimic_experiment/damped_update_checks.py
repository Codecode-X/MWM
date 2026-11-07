"""CUDA implementation checks, using saved inputs; no optimizer steps."""
import torch

from direction_gradient_common import decompose
from mechanism_common import parameter_hash
from damped_update_model import make_damped_model
from mimic_experiment.train_full_batch import make_model, loss_function
from mimic_experiment.prepare import atomic


def verify(root, folder, manifest, cfg, payload, gpu):
    snapshot = torch.load(manifest['abnormal_snapshot'], map_location='cuda', weights_only=False)
    batch = snapshot['probe_batch']
    native = make_model(payload['schema'], cfg).cuda().eval()
    mixed = make_damped_model(payload['schema'], cfg).eval()
    for model in [native, mixed]:
        model.load_state_dict(snapshot['state_dict'], strict=True)
    before = parameter_hash(mixed)
    assert parameter_hash(native) == before
    assert list(native.state_dict()) == list(mixed.state_dict())
    mixed.update.alpha = 1.0
    with torch.no_grad():
        original = native(batch, sample=False)
        identity = mixed(batch, sample=False)
        assert all(torch.equal(original[k], identity[k]) for k in original)
        native_loss, _ = loss_function(native, original, batch, payload['schema'], cfg)
        identity_loss, _ = loss_function(mixed, identity, batch, payload['schema'], cfg)
        assert torch.equal(native_loss, identity_loss)
    del original, identity

    # The same saved pre-UPDATE states and observations in both calculations.
    banks = torch.load(root/'module_cross/stable/fixed_banks_and_jacobians.pt', map_location='cuda', weights_only=False)
    bank = banks['A']
    x = torch.cat([bank['preupdate_states'], bank['observations']], -1).detach().double().requires_grad_(True)
    native.double(); mixed.double(); mixed.update.alpha = 0.3
    u = native.update(x)
    y = mixed.update(x)
    expected = .7 * x[:, :28] + .3 * u
    assert torch.equal(y, expected)
    basis = torch.eye(28, device='cuda', dtype=torch.float64)[:, None, :].expand(28, len(x), 28)
    ju = torch.autograd.grad(u, x, basis, is_grads_batched=True, retain_graph=True)[0][:, :, :28].permute(1, 0, 2)
    jy = torch.autograd.grad(y, x, basis, is_grads_batched=True)[0][:, :, :28].permute(1, 0, 2)
    reference = .7 * torch.eye(28, device='cuda', dtype=torch.float64)[None] + .3 * ju
    jacobian_error = float((jy - reference).abs().max())
    assert jacobian_error < 1e-12
    with torch.no_grad():
        mixed.update.alpha = 0
        assert torch.equal(mixed.update(x), x[:, :28])
    del x, u, y, ju, jy

    mixed.float(); mixed.update.alpha = .3
    assert parameter_hash(mixed) == before
    out = mixed(batch, sample=False)
    loss, _ = loss_function(mixed, out, batch, payload['schema'], cfg)
    historical, future = decompose(mixed, out, batch, payload['schema'], cfg)
    def norm(value, parameters):
        grads = torch.autograd.grad(value, tuple(parameters), retain_graph=True)
        return float(torch.cat([g.flatten() for g in grads]).norm())
    history_norms = {name: norm(historical, getattr(mixed, name).parameters()) for name in ['init', 'update', 'forward_ode', 'decoder']}
    future_norms = {name: norm(future, getattr(mixed, name).parameters()) for name in ['init', 'update', 'forward_ode', 'decoder']}
    assert all(v > 0 for v in history_norms.values())
    assert all(v > 0 for v in future_norms.values())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in mixed.parameters())
    assert parameter_hash(mixed) == before
    result = {'status': 'passed', 'physical_gpu': gpu, 'parameter_device': str(next(mixed.parameters()).device),
              'actual_tensor_devices': {k: str(v.device) for k, v in batch.items() if torch.is_tensor(v)},
              'alpha': .3, 'learnable_alpha': False, 'parameters': 43280, 'latent': 28,
              'same_parameter_hash_and_state_dict_keys': True, 'alpha1_forward_and_loss_bitwise_equal': True,
              'alpha0_returns_input_state': True, 'alpha03_update_formula_bitwise_equal': True,
              'jacobian_formula_max_absolute_error_float64': jacobian_error,
              'history_gradient_norms': history_norms, 'future_gradient_norms': future_norms,
              'history_encoder_gradient_stopped': False, 'optimizer_steps': 0,
              'snapshot': manifest['abnormal_snapshot'], 'batch_kind': 'saved verified e46b121 training replay batch',
              'fixed_update_inputs': str(root/'module_cross/stable/fixed_banks_and_jacobians.pt')}
    atomic(folder/'checks.json', result)
    return result
