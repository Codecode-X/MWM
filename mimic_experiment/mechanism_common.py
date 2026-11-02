"""CUDA-only fixed-parameter helpers for the instability mechanism study."""
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import random

import numpy as np
import torch

from direction_gradient_common import initialize,encode_only,flow_jacobian,flat_gradient,cosine
from mimic_experiment.data import ICUContexts,collate
from mimic_experiment.train_full_batch import make_model
from paper_odeae.models import feature_errors,encode_values
from paper_odeae.solver import integrate,solver_statistics
from paper_odeae.train import device_batch


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),
            'torch_cuda':[s for s in torch.cuda.get_rng_state_all()]}


def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch_cpu'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in s['torch_cuda']])


@contextmanager
def rng_guard():
    s=rng_state()
    try:yield
    finally:restore_rng(s)


def vector(parameters):return torch.cat([p.detach().flatten() for p in parameters])


def tensor_hash(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def parameter_hash(model):return tensor_hash(vector(tuple(model.parameters())))


def batch_audit(batch,kind,indices=None):
    return {'batch_kind':kind,'patient_indices':indices,'subject_ids':[str(x) for x in batch['subject_ids']],
            'patient_index_namespace':'ICUContexts eligible dataset; train for real_training and validation for fixed_validation_diagnostic',
            'cutoffs':batch['cutoff'].cpu().tolist(),
            'history_span_hours':(batch['ages'].gather(1,(batch['cutoff']-1)[:,None]).squeeze(1)-batch['ages'][:,0]).cpu().tolist(),
            'times_hours':[batch['ages'][i,batch['valid'][i]].cpu().tolist() for i in range(len(batch['cutoff']))],
            'tensor_hashes':{k:tensor_hash(v) for k,v in batch.items() if torch.is_tensor(v)},
            'actual_tensor_devices':{k:str(v.device) for k,v in batch.items() if torch.is_tensor(v)}}


def fixed_cohort(payload,n=8):
    rows=[s for s in payload['sequences'] if s['split']=='validation'];ds=ICUContexts(rows,seed=0)
    # Same selection and ordering as the previous 64-patient diagnostic, take the first n.
    indices=np.random.default_rng(7).choice(len(ds),64,replace=False)[:n]
    items=[ds[int(i)] for i in indices];batch=device_batch(collate(items))
    return batch,batch_audit(batch,'fixed_validation_diagnostic',indices.tolist())


def model_from(path,cfg,schema,requires_grad=True,dtype=torch.float32):
    saved=torch.load(path,map_location='cuda',weights_only=False)
    model=make_model(schema,cfg).cuda().to(dtype);model.load_state_dict(saved['state_dict']);model.eval();model.requires_grad_(requires_grad)
    assert sum(p.numel() for p in model.parameters())==43280
    assert model.forward_ode is model.backward_ode
    return model,saved


def solver_counts(trace):
    calls=trace['calls']
    return {k:sum(c[k] for c in calls) for k in ['accepted','rejected','nfe']}|{'calls':len(calls)}


def brief(x):
    a=x.detach().double().flatten()
    return {'median':float(torch.quantile(a,.5)),'p95':float(torch.quantile(a,.95)),
            'p99':float(torch.quantile(a,.99)),'max':float(a.max()),'min':float(a.min())}


def rk4(field,z,delta,max_step):
    # Fixed physical step cap in hours; each row takes the same count with its own delta.
    steps=max(1,math.ceil(float(delta.abs().max())/max_step));h=delta[:,None]/steps
    for _ in range(steps):
        a=field(z);b=field(z+h*a/2);c=field(z+h*b/2);d=field(z+h*c)
        z=z+h*(a+2*b+2*c+d)/6
    return z,steps


def solve(model,z,delta,method='tsit5',step=None,counts=None):
    if method=='tsit5':return integrate(model.forward_ode,z,delta,model.rtol,model.atol)
    result,n=rk4(model.forward_ode,z,delta,step)
    if counts is not None:counts['fixed_steps']=counts.get('fixed_steps',0)+n;counts['nfe']=counts.get('nfe',0)+4*n
    return result


def history_states(model,z,batch,direction=-1,method='tsit5',step=None,counts=None):
    ages=batch['ages'];cut=batch['cutoff'];previous=ages.gather(1,(cut-1)[:,None]).squeeze(1)
    states=[torch.zeros_like(z) for _ in range(batch['max_context'])];state=z
    for i in reversed(range(batch['max_context'])):
        active=i<cut;dt=torch.where(active,previous-ages[:,i],torch.zeros_like(previous))
        state=solve(model,state,direction*dt,method,step,counts);states[i]=state
        previous=torch.where(active,ages[:,i],previous)
    return torch.stack(states,1)


def endpoint(model,z,batch,direction=-1,method='tsit5',step=None,counts=None):
    return history_states(model,z,batch,direction,method,step,counts)[:,0]


def encode_solver(model,batch,method='tsit5',step=None,counts=None):
    """The original chronological encoder, with an explicit numerical integrator."""
    values,ages,cut=batch['values'],batch['ages'],batch['cutoff']
    inputs=encode_values(values,batch['context_mask'],model.schema)
    h=model.init(batch['initialization_age'][:,None]/100.);previous=ages[:,0]
    for j in range(batch['max_context']):
        index=torch.minimum(torch.full_like(cut,j),cut-1);active=j<cut
        time=ages.gather(1,index[:,None]).squeeze(1);x=inputs[torch.arange(len(cut),device='cuda'),index]
        evolved=solve(model,h,torch.where(active,time-previous,torch.zeros_like(time)),method,step,counts)
        proposal=model.update(torch.cat([evolved,x],-1));h=torch.where(active[:,None],proposal,h);previous=torch.where(active,time,previous)
    return h


def sensitivity(model,z,batch,direction=-1):
    with solver_statistics() as trace:
        jac=flow_jacobian(lambda x:endpoint(model,x,batch,direction),z)
    singular=torch.linalg.svdvals(jac)[:,0]
    return {'gain':brief(singular),'per_patient_gain':singular.cpu().tolist(),'solver':solver_counts(trace)},jac


def update_probe(probe,train_model,batch,z_fixed):
    probe.load_state_dict(train_model.state_dict())
    probe.rtol,probe.atol=1e-4,1e-6
    with torch.no_grad():z_current=encode_only(probe,batch)
    probe.rtol,probe.atol=1e-6,1e-8
    result={}
    for label,z in [('A_reencoded',z_current),('B_fixed_state',z_fixed)]:
        past,_=sensitivity(probe,z,batch,-1);future,_=sensitivity(probe,z,batch,1)
        ratio=np.array(past['per_patient_gain'])/np.array(future['per_patient_gain'])
        result[label]={'past':past,'future':future,'past_future_ratio_median':float(np.median(ratio)),'state_norm':brief(z.norm(dim=-1))}
    return result


def historical_loss(model,z,batch,schema,denominator=None,method='tsit5',step=None,counts=None):
    states=history_states(model,z,batch,-1,method,step,counts)
    predictions=torch.stack([model.decoder(states[:,j]) for j in range(states.shape[1])],1)
    n=states.shape[1];hist=(torch.arange(n,device='cuda')[None,:]<batch['cutoff'][:,None])
    mask=batch['mask'][:,:n]*hist[...,None]
    errors=feature_errors(predictions,batch['values'][:,:n],mask,schema)
    denom=batch['mask'].sum((0,1)).clamp_min(1) if denominator is None else denominator
    per_patient=(errors.sum(1)/denom).sum(-1)
    return per_patient.sum(),per_patient,states,errors,denom


def relative(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
