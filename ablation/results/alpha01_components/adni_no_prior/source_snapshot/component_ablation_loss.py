"""Original objectives, with Jacobian averaged over the actual vector fields."""
import torch
from paper_odeae.models import feature_errors, reconstruction, prior_errors
from paper_odeae.jacobian import squared_jacobian_norm
from component_ablation_model import unique_fields


def jacobian_regularization(model, output, batch):
    states = output['jacobian_states'][batch['valid']].detach()
    values = torch.stack([squared_jacobian_norm(f, states) for f in unique_fields(model)]).mean(0)
    return values.mean(), values.detach().sum(), len(states)


def loss_function(model, output, batch, schema, cfg):
    mse, ce = reconstruction(feature_errors(output['prediction'], batch['values'], batch['mask'], schema), batch['mask'], schema)
    pe, pmask = prior_errors(output, batch, schema); pmse, pce = reconstruction(pe, pmask, schema)
    jac, _, _ = jacobian_regularization(model, output, batch); l2 = model.ode_regularization()
    if cfg['dataset'] == 'adni':
        # Preserve the established ADNI expression and floating-point order.
        loss = cfg['mse_weight']*mse+cfg['cce_weight']*ce+cfg['kl_weight']*output['kl']+cfg['l2_weight']*l2
        loss = loss+cfg['prior_weight']*(pmse+pce)
        loss = loss+cfg['jacobian_weight']*jac
    else:
        loss = cfg['mse_weight']*mse+cfg['cce_weight']*ce+cfg['prior_weight']*(pmse+pce)+cfg['jacobian_weight']*jac+cfg['l2_weight']*l2
    return loss, torch.stack([mse.detach(), ce.detach(), pmse.detach(), pce.detach(), jac.detach(), l2.detach()])
