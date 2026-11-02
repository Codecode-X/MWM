"""Remove past-generation training; retain the original model for common tests."""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys

import torch

from paper_odeae.models import PaperModel,paper_model_from_config,feature_errors,reconstruction,prior_errors,input_width
from paper_odeae.jacobian import jacobian_regularization
from paper_odeae.train import seed_all,device_batch
from mimic_experiment.data import collate
from mimic_experiment.prepare import atomic
from mimic_experiment import train_full_batch as original


class FutureOnlyModel(PaperModel):
    def generate_from_cutoff(self,z,batch,return_states=False):
        if not self.training:return super().generate_from_cutoff(z,batch,return_states)
        ages,cut=batch['ages'],batch['cutoff'];b,n=ages.shape
        previous=ages.gather(1,(cut-1)[:,None]).squeeze(1);state=z
        predictions=[z.new_zeros(b,input_width(self.schema)) for _ in range(n)]
        states=[torch.zeros_like(z) for _ in range(n)]
        for i in range(n):
            active=(i>=cut)&batch['valid'][:,i]
            if not bool(active.any()):continue
            state=self.solve(self.forward_ode,state,torch.where(active,ages[:,i]-previous,torch.zeros_like(previous)))
            predictions[i]=torch.where(active[:,None],self.decoder(state),predictions[i])
            states[i]=torch.where(active[:,None],state,states[i]);previous=torch.where(active,ages[:,i],previous)
        prediction=torch.stack(predictions,1)
        return (prediction,torch.stack(states,1)) if return_states else prediction


def make_model(schema,cfg):
    if cfg['prior_weight']<0 or cfg.get('train_reconstruct_history') is not False:raise ValueError('Require nonnegative weak_prior and future-only main training')
    model=paper_model_from_config(schema,cfg,model_class=FutureOnlyModel)
    model.variant=cfg['variant'];model.generator_anchor=cfg['generator_anchor'];return model


def future_mask(batch):
    index=torch.arange(batch['ages'].shape[1],device=batch['ages'].device)[None,:]
    return (index>=batch['cutoff'][:,None])&batch['valid']


def loss_function(model,out,batch,schema,cfg):
    future=future_mask(batch);mask=batch['mask']*future[...,None]
    mse,cce=reconstruction(feature_errors(out['prediction'],batch['values'],mask,schema),mask,schema)
    pe,pmask=prior_errors(out,batch,schema);pmse,pcce=reconstruction(pe,pmask,schema)
    jac,_,_=jacobian_regularization(model,out,{**batch,'valid':future});l2=model.ode_regularization()
    loss=cfg['mse_weight']*mse+cfg['cce_weight']*cce+cfg['jacobian_weight']*jac+cfg['l2_weight']*l2
    if cfg['prior_weight']>0:loss=loss+cfg['prior_weight']*(pmse+pcce)
    return loss,torch.stack([mse.detach(),cce.detach(),pmse.detach(),pcce.detach(),jac.detach(),l2.detach()])


def check():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--manifest',required=True);args=parser.parse_args()
    manifest=json.loads(Path(args.manifest).read_text());cfg=manifest['config'];control=manifest['control_config']
    if Path(sys.prefix).name!='ode' or os.environ.get('CUDA_VISIBLE_DEVICES')!='3' or not torch.cuda.is_available():raise RuntimeError('Use ode and GPU3')
    torch.set_num_threads(1);torch.cuda.set_device(0)
    payload=pickle.loads(Path(cfg['data']).read_bytes());schema=payload['schema']
    rows=[s for s in payload['sequences'] if s['split']=='train' and len(s['ages'])>=8][:4]
    items=[({**s,**{k:s[k][:8] for k in ['ages','values','mask','dates']}},4) for s in rows];batch=device_batch(collate(items))
    seed_all(0);reference=original.make_model(schema,control).cuda()
    seed_all(0);model=make_model(schema,cfg).cuda()
    for key,value in reference.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
    deltas=[];solve=model.solve
    def recorded(field,state,dt):deltas.append(dt.detach().clone());return solve(field,state,dt)
    model.solve=recorded
    reference.train();model.train();ref=reference(batch);out=model(batch);future=future_mask(batch)
    torch.testing.assert_close(out['prediction'][future],ref['prediction'][future],rtol=0,atol=0)
    torch.testing.assert_close(out['prior_prediction'],ref['prior_prediction'],rtol=0,atol=0)
    if any(bool((dt<0).any()) for dt in deltas):raise AssertionError('No negative time integration is allowed during training')
    objective,parts=loss_function(model,out,batch,schema,cfg)
    prediction_grad=torch.autograd.grad(objective,out['prediction'],retain_graph=True)[0]
    if bool(prediction_grad[~future].count_nonzero()) or not bool(prediction_grad[future].count_nonzero()):raise AssertionError('Only future predictions may receive data-loss gradients')
    objective.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise AssertionError('Require finite parameter gradients')
    model.eval();deltas.clear()
    with torch.no_grad():evaluated=model(batch)
    if not any(bool((dt<0).any()) for dt in deltas):raise AssertionError('Common evaluation must retain retrospective reconstruction')
    torch.testing.assert_close(evaluated['prediction'][future],out['prediction'][future],rtol=0,atol=0)
    result={'status':'passed','parameters':sum(p.numel() for p in model.parameters()),'physical_gpu':3,'device':str(out['prediction'].device),
            'same_initial_parameters':True,'same_initial_future_predictions':True,'no_negative_training_integration':True,
            'history_data_loss_gradient_zero':True,'finite_gradients':True,'evaluation_retains_history':True,
            'training_jacobian_sampling':'future valid observation times only'}
    atomic(Path(args.manifest).parent/'cuda_checks.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    if '--check' in sys.argv:check()
    else:
        original.make_model=make_model;original.loss_function=loss_function;original.main()
