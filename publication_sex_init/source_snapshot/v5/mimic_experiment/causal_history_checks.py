"""CUDA gradient checks at the saved high-gain real batch; no updates."""
import argparse
from pathlib import Path

import torch

from direction_gradient_common import initialize,decompose,flat_gradient
from mechanism_common import parameter_hash,relative
from causal_history_model import make_causal_model
from mimic_experiment.prepare import atomic
from mimic_experiment.train_full_batch import make_model,loss_function
from paper_odeae.models import prior_errors,reconstruction
from paper_odeae.jacobian import jacobian_regularization


def gradients(loss,params):
    gs=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(params,gs)])


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    root=Path(a.out);manifest,cfg,data,modes=initialize(root,a.gpu)
    snapshot=torch.load(manifest['abnormal_snapshot'],map_location='cuda',weights_only=False);batch=snapshot['probe_batch'];schema=data['schema']
    control=make_causal_model(schema,cfg);stop=make_causal_model(schema,cfg,True)
    for model in [control,stop]:model.load_state_dict(snapshot['state_dict']);model.eval()
    before=[parameter_hash(m) for m in [control,stop]]
    c=control(batch,sample=False);s=stop(batch,sample=False)
    assert torch.equal(c['prediction'],s['prediction']) and torch.equal(c['mu'],s['mu'])
    assert torch.equal(c['jacobian_states'],s['jacobian_states']) and torch.equal(c['prior_prediction'],s['prior_prediction'])
    hc,fc=decompose(control,c,batch,schema,cfg);hs,fs=decompose(stop,s,batch,schema,cfg)
    lc,_=loss_function(control,c,batch,schema,cfg);ls,_=loss_function(stop,s,batch,schema,cfg)
    assert torch.equal(lc,ls)
    allc,alls=tuple(control.parameters()),tuple(stop.parameters());fieldc,fields=tuple(control.forward_ode.parameters()),tuple(stop.forward_ode.parameters())
    encoder=tuple(stop.init.parameters())+tuple(stop.update.parameters())
    blocked=gradients(hs,encoder);assert float(blocked.norm())==0
    gf_c,gf_s=gradients(fc,allc),gradients(fs,alls);future_error=relative(gf_s,gf_c);assert future_error<1e-6
    def rest(model,output):
        pe,pm=prior_errors(output,batch,schema);mse,ce=reconstruction(pe,pm,schema)
        jac,_,_=jacobian_regularization(model,output,batch)
        return cfg['prior_weight']*(mse+ce)+cfg['jacobian_weight']*jac+cfg['l2_weight']*model.ode_regularization()
    rest_c,rest_s=rest(control,c),rest(stop,s)
    rest_error=relative(gradients(rest_s,alls),gradients(rest_c,allc));assert rest_error<1e-6
    direct=control.generate_from_cutoff(c['mu'].detach(),batch,return_states=False)
    hd,_=decompose(control,{'prediction':direct},batch,schema,cfg)
    gd,ghs=flat_gradient(hd,fieldc),flat_gradient(hs,fields);direct_error=relative(ghs,gd);assert direct_error<1e-6
    decoder=gradients(hs,tuple(stop.decoder.parameters()));assert float(decoder.norm())>0 and float(ghs.norm())>0
    total_c,total_s=flat_gradient(lc,fieldc),flat_gradient(ls,fields)
    result={'status':'passed','physical_gpu':a.gpu,'parameter_device':str(next(stop.parameters()).device),
            'actual_tensor_devices':{k:str(v.device) for k,v in batch.items() if torch.is_tensor(v)},'prediction_device':str(s['prediction'].device),
            'parameters':43280,'latent':28,'history_init_update_gradient_norm':float(blocked.norm()),
            'forward_predictions_bitwise_equal':True,'scalar_losses_bitwise_equal':True,'future_all_parameter_gradient_relative_error':future_error,
            'prior_jacobian_l2_gradient_relative_error':rest_error,'history_field_gradient_vs_original_direct_relative_error':direct_error,
            'original_history_field_gradient_norm':float(flat_gradient(hc,fieldc).norm()),'stopped_history_field_gradient_norm':float(ghs.norm()),
            'stopped_history_decoder_gradient_norm':float(decoder.norm()),'original_total_field_gradient_norm':float(total_c.norm()),
            'stopped_total_field_gradient_norm':float(total_s.norm()),'parameter_hashes_before':before,'optimizer_steps':0,
            'snapshot':manifest['abnormal_snapshot'],'batch_kind':'saved actual e46b121 replay batch'}
    assert [parameter_hash(m) for m in [control,stop]]==before
    atomic(root/'checks.json',result)


if __name__=='__main__':main()
