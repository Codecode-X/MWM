"""CUDA-only geometry, causality and gradient checks for all six matched jobs."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

import torch

from paper_odeae.models import feature_errors,reconstruction
from paper_odeae.train import seed_all,device_batch
from paper_odeae.scaling_study import atomic
from .models import make_model


def run(cfg,payload,output,gpu):
    schema=payload['schema'];mimic=cfg['dataset_name']=='mimic'
    if mimic:
        from mimic_experiment.data import collate
    else:
        from paper_odeae.data import collate
    training=[s for s in payload['sequences'] if s['split']=='train' and len(s['ages'])>=4][:4]
    clipped=[{**s,**{k:s[k][:4] for k in ['ages','values','mask','dates']}} for s in training]
    batch=device_batch(collate(list(zip(clipped,[2,3,1,2]))));seed_all(cfg['seed'])
    model=make_model(schema,cfg).cuda().eval();count=sum(p.numel() for p in model.parameters())
    if count!=cfg['target_parameters'] or count!=sum(p.numel() for p in model.parameters() if p.requires_grad):raise AssertionError('Total/trainable budget mismatch')
    if hasattr(model,'core') and hasattr(model.core,'classifier'):raise AssertionError('Unused classifier remained in matched capacity')
    with torch.no_grad():original=model(batch,sample=False)
    changed={k:v.clone() if torch.is_tensor(v) else v for k,v in batch.items()}
    future=torch.arange(batch['values'].shape[1],device='cuda')[None]>=batch['cutoff'][:,None]
    for j,f in enumerate(schema):
        value=(changed['values'][...,j]+1)%f['classes'] if f['kind']=='categorical' else changed['values'][...,j]+50
        changed['values'][...,j]=torch.where(future,value,changed['values'][...,j])
    changed['mask']=torch.where(future[...,None],torch.zeros_like(changed['mask']),changed['mask'])
    with torch.no_grad():other=model(changed,sample=False)
    torch.testing.assert_close(original['prediction'],other['prediction'],rtol=0,atol=0)
    time_difference=None
    if 'mu' in original:
        for i,cut in enumerate(batch['cutoff'].tolist()):changed['ages'][i,cut:]+=5
        with torch.no_grad():other=model(changed,sample=False)
        time_difference=float((original['mu']-other['mu']).abs().max())
        torch.testing.assert_close(original['mu'],other['mu'],rtol=2e-6,atol=2e-7)
    model.train();result=model(batch,sample=True)
    mse,cce=reconstruction(feature_errors(result['prediction'],batch['values'],batch['mask'],schema),batch['mask'],schema)
    loss=mse+cce+cfg['kl_weight']*result['kl']+cfg['l2_weight']*model.ode_regularization()
    if not torch.isfinite(loss):raise AssertionError('Nonfinite check loss')
    loss.backward()
    for name,p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():raise AssertionError(f'Unused/nonfinite trainable parameter {name}')
    if cfg['model']=='ode_ae':
        assert model.kind=='ode_ae' and model.latent==6 and model.forward_ode is not model.backward_ode
        assert (result['std']>0).all() and result['mu'].shape[-1]==6
    elif cfg['model']=='latent_ode':
        assert model.core.encoder.hidden==48 and result['mu'].shape[-1]==32 and (result['std']>0).all()
        assert model.core.encoder.ode is not model.core.ode
    else:assert model.core.hidden==38 and float(result['kl'])==0.
    controlled=copy.deepcopy(model)
    with torch.no_grad():
        func=controlled.forward_ode if cfg['model']=='ode_ae' else controlled.core.ode
        for p in func.parameters():p.zero_()
        layers=func if cfg['model']=='ode_ae' else func.net;layers[-1].bias.fill_(.2)
        z=torch.zeros(2,cfg['latent'],device='cuda');dt=torch.tensor([1.,3.],device='cuda')
        torch.testing.assert_close(controlled.solve(func,z,dt),z+.2*dt[:,None],rtol=2e-5,atol=2e-6)
    breakdown={name:sum(p.numel() for p in module.parameters()) for name,module in model.named_children()}
    if hasattr(model,'core'):breakdown={name:sum(p.numel() for p in module.parameters()) for name,module in model.core.named_children()}
    atomic(output/'checks.json',{'status':'passed','dataset':cfg['dataset_name'],'model':cfg['model'],'parameters':count,'trainable_parameters':count,
           'parameter_breakdown':breakdown,'physical_gpu':gpu,'device':'cuda:0','environment':Path(sys.prefix).name,
           'all_parameters_receive_finite_gradients':True,'future_values_and_target_masks_do_not_change_predictions':True,
           'probabilistic_posterior_preserved':cfg['model']!='ode_rnn','unused_binary_classifier_removed':True,
           'solver_clock_constant_field_verified':True,'future_times_change_encoded_state_max':time_difference,
           'actual_tensor_devices':{k:str(v.device) for k,v in batch.items() if torch.is_tensor(v)}})
    print(json.dumps({'dataset':cfg['dataset_name'],'model':cfg['model'],'parameters':count,'physical_gpu':gpu,'checks':'passed'}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',required=True);parser.add_argument('--gpu',type=int,required=True);args=parser.parse_args()
    if Path(sys.prefix).name!='ode' or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu) or not torch.cuda.is_available():raise RuntimeError('Explicit assigned CUDA GPU and ode required')
    torch.set_num_threads(1);torch.cuda.set_device(0);manifest=json.loads(Path(args.manifest).read_text());payloads={}
    for job in manifest['jobs']:
        cfg=job['config'];name=cfg['dataset_name'];output=Path(cfg['comparison_output'])/'checks'/cfg['model']/'checks';output.mkdir(parents=True,exist_ok=True)
        if name not in payloads:
            data=Path(cfg['data']).read_bytes()
            if hashlib.sha256(data).hexdigest()!=cfg['dataset_sha256']:raise RuntimeError('Dataset hash changed')
            payloads[name]=pickle.loads(data)
        run(cfg,payloads[name],output,args.gpu);torch.cuda.empty_cache()
    target=Path(args.manifest).parent/'architecture_checks.json'
    atomic(target,{'status':'passed','physical_gpu':args.gpu,'device':'cuda:0','jobs':len(manifest['jobs'])})


if __name__=='__main__':main()
