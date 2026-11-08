"""CUDA-only numerical helpers for the saved standard forward model."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import sys

import torch

from mimic_experiment.prepare import atomic
from mimic_experiment.train_full_batch import make_model,loss_function
from mimic_experiment.train_baseline import normalization_and_modes
from paper_odeae.models import feature_errors,prior_errors,encode_values
from paper_odeae.jacobian import jacobian_regularization


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(out,gpu):
    out=Path(out);manifest=json.loads((out/'manifest.json').read_text())
    if Path(sys.prefix).name!='ode' or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu) or not torch.cuda.is_available():raise RuntimeError('Assigned CUDA GPU and ode are required')
    if not Path(inspect.getfile(make_model)).resolve().is_relative_to(Path(manifest['source_snapshot'])):raise RuntimeError('Use the original frozen direction-study source')
    for name,sha in manifest['source_sha256'].items():
        if digest(Path(manifest['source_snapshot'])/name)!=sha:raise RuntimeError('Frozen source changed')
    for name,sha in manifest.get('analysis_source_sha256',{}).items():
        if digest(Path(manifest['analysis_source'])/name)!=sha:raise RuntimeError('Analysis source changed')
    torch.set_num_threads(1);torch.cuda.set_device(0)
    cfg=manifest['reference_config'];raw=Path(cfg['data']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=cfg['dataset_sha256']:raise RuntimeError('Dataset changed')
    payload=pickle.loads(raw);means,modes=normalization_and_modes(payload)
    cfg={**cfg,'raw_scale':payload['metadata']['scale'],'normalized_train_means':means}
    return manifest,cfg,payload,modes


def load_model(entry,cfg,schema,requires_grad=True):
    if digest(entry['path'])!=entry['sha256']:raise RuntimeError('Analysis checkpoint changed')
    saved=torch.load(entry['path'],map_location='cuda',weights_only=False)
    if saved['epoch']!=entry['epoch']:raise RuntimeError('Checkpoint epoch mismatch')
    model=make_model(schema,cfg).cuda();model.load_state_dict(saved['state_dict'],strict=True);model.eval();model.requires_grad_(requires_grad)
    if sum(p.numel() for p in model.parameters())!=43280 or model.backward_ode is not model.forward_ode:raise RuntimeError('Require the original shared 28D field')
    return model,saved


def decompose(model,out,batch,schema,cfg):
    errors=feature_errors(out['prediction'],batch['values'],batch['mask'],schema)
    idx=torch.arange(batch['ages'].shape[1],device='cuda')[None,:];history=idx<batch['cutoff'][:,None]
    counts=batch['mask'].sum((0,1)).clamp_min(1)
    factors=errors.new_tensor([cfg['mse_weight'] if f['kind']=='numerical' else cfg['cce_weight'] for f in schema])
    hist=((errors*history[...,None]).sum((0,1))/counts*factors).sum()
    future=((errors*(~history)[...,None]).sum((0,1))/counts*factors).sum()
    return hist,future


def flat_gradient(value,parameters):
    return torch.cat([x.reshape(-1) for x in torch.autograd.grad(value,parameters,retain_graph=True)])


def cosine(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-20))


def gradient_record(gh,gf):
    nh,nf=gh.norm(),gf.norm()
    return {'history_gradient_norm':float(nh),'future_gradient_norm':float(nf),'history_future_cosine':cosine(gh,gf),
            'history_to_future_norm_ratio':float(nh/nf.clamp_min(1e-20)),
            'gradient_sum_norm':float((gh+gf).norm()),'cancellation_ratio':float((gh+gf).norm()/(nh+nf).clamp_min(1e-20))}


def component_gradients(model,out,batch,schema,cfg,detail=False):
    parameters=tuple(model.forward_ode.parameters());hist,future=decompose(model,out,batch,schema,cfg)
    gh,gf=flat_gradient(hist,parameters),flat_gradient(future,parameters)
    row={'history_loss_contribution':float(hist.detach()),'future_loss_contribution':float(future.detach()),**gradient_record(gh,gf)}
    if detail:
        original,_=loss_function(model,out,batch,schema,cfg)
        pe,pm=prior_errors(out,batch,schema)
        prior=cfg['prior_weight']*(pe.sum((0,1))/pm.sum((0,1)).clamp_min(1)).sum()
        jac,_,_=jacobian_regularization(model,out,batch);reg=cfg['jacobian_weight']*jac+cfg['l2_weight']*model.ode_regularization()
        gp,gr,gt=flat_gradient(prior,parameters),flat_gradient(reg,parameters),flat_gradient(original,parameters)
        error=(gh+gf+gp+gr-gt).norm()/gt.norm().clamp_min(1e-20)
        if float(error)>2e-4:raise RuntimeError('Loss-gradient decomposition mismatch')
        row.update(weighted_prior_gradient_norm=float(gp.norm()),regularizer_gradient_norm=float(gr.norm()),total_gradient_norm=float(gt.norm()),decomposition_relative_error=float(error))
        generated=model.generate_from_cutoff(out['mu'].detach(),batch,return_states=False)
        hd,fd=decompose(model,{'prediction':generated},batch,schema,cfg)
        gd_h,gd_f=flat_gradient(hd,parameters),flat_gradient(fd,parameters)
        row.update(generation_only_history_gradient_norm=float(gd_h.norm()),generation_only_future_gradient_norm=float(gd_f.norm()),
                   generation_only_history_future_cosine=cosine(gd_h,gd_f))
    return row,gh.detach(),gf.detach()


def encode_only(model,batch):
    values,ages,cut=batch['values'],batch['ages'],batch['cutoff'];inputs=encode_values(values,batch['context_mask'],model.schema)
    h=model.init(batch['initialization_age'][:,None]/100.);previous=ages[:,0]
    for j in range(batch['max_context']):
        index=torch.minimum(torch.full_like(cut,j),cut-1);active=j<cut
        time=ages.gather(1,index[:,None]).squeeze(1);x=inputs[torch.arange(len(cut),device='cuda'),index]
        evolved=model.solve(model.forward_ode,h,torch.where(active,time-previous,torch.zeros_like(time)))
        proposal=model.update(torch.cat([evolved,x],-1));h=torch.where(active[:,None],proposal,h);previous=torch.where(active,time,previous)
    return h


def flow_chain(model,x,batch,direction):
    ages,cut=batch['ages'],batch['cutoff'];state=x;previous=ages.gather(1,(cut-1)[:,None]).squeeze(1)
    for i in reversed(range(batch['max_context'])):
        active=i<cut;time=ages[:,i];delta=torch.where(active,time-previous,torch.zeros_like(previous))
        state=model.solve(model.forward_ode,state,delta*direction);previous=torch.where(active,time,previous)
    return state


def flow_jacobian(flow,states):
    x=states.detach().requires_grad_(True);y=flow(x);d=x.shape[-1]
    # Rows are independent given the solver's detached adaptive controller.
    basis=torch.eye(d,device=x.device)[:,None,:].expand(d,len(x),d)
    vjp=torch.autograd.grad(y,x,grad_outputs=basis,is_grads_batched=True)[0]
    return vjp.permute(1,0,2).detach()


def summary(values):
    x=values.detach().float().flatten()
    return {'n':len(x),'mean':float(x.mean()),'median':float(x.median()),'p90':float(torch.quantile(x,.9)),
            'p95':float(torch.quantile(x,.95)),'max':float(x.max()),'min':float(x.min())}


def publish(out,entry):
    path=Path(out)/'checkpoint_index.json';index=json.loads(path.read_text())
    index['entries']=[x for x in index['entries'] if x['id']!=entry['id']]+[entry];atomic(path,index)
