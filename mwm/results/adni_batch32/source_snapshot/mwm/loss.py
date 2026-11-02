"""Unified supervision; A weights data only, B predicts observation residual scale."""
import torch

from paper_odeae.models import feature_errors
from paper_odeae.jacobian import jacobian_regularization


def numeric_errors(prediction,target,schema):
    rows=[];at=0
    for j,field in enumerate(schema):
        width=field.get('classes',1)
        rows.append((prediction[...,at]-target[...,j]).square() if field['kind']=='numerical' else prediction.new_zeros(target.shape[:2]))
        at+=width
    return torch.stack(rows,-1)


def data_loss(model,out,batch,schema):
    mask=batch['mask'];counts=mask.sum((0,1)).clamp_min(1)
    errors=feature_errors(out['prediction'],batch['values'],mask,schema)
    numerical=torch.tensor([f['kind']=='numerical' for f in schema],device=mask.device)
    weights=out['supervision_weights'][...,None]
    means=errors.sum((0,1))/counts
    mse=means[numerical].sum();cce=means[~numerical].sum()
    weighted=(errors*weights).sum((0,1))/counts
    zero=mse.new_zeros(());scaled_sq=zero;log_sigma=zero
    if model.variant=='B':
        sigma=out['sigma'];square=numeric_errors(out['prediction'],batch['values'],schema)
        scaled=square/(2*sigma.square());penalty=sigma.log().expand_as(square)
        scaled_sq=(scaled*mask).sum((0,1)).div(counts)[numerical].sum()
        log_sigma=(penalty*mask).sum((0,1)).div(counts)[numerical].sum()
        data=scaled_sq+log_sigma+weighted[~numerical].sum()
    else:data=weighted.sum()
    return data,{'unweighted_mse':mse,'unweighted_cce':cce,'data_supervision':data,
                 'nll_scaled_square':scaled_sq,'nll_log_sigma':log_sigma,
                 'average_data_weight':(weights.squeeze(-1)*batch['valid']).sum()/batch['valid'].sum().clamp_min(1)}


def objective(model,out,batch,schema,cfg):
    data,parts=data_loss(model,out,batch,schema)
    jac,_,_=jacobian_regularization(model,out,batch);l2=model.ode_regularization()
    loss=data+cfg['jacobian_weight']*jac+cfg['l2_weight']*l2
    return loss,{**parts,'jacobian_frobenius_squared':jac,'ode_l2':l2,
                 'weighted_jacobian':cfg['jacobian_weight']*jac,'weighted_ode_l2':cfg['l2_weight']*l2}
