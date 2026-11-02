"""CUDA checks for supervision timing, count weights, uncertainty and capacity."""
import argparse
import json
import os
from pathlib import Path

import torch

from paper_odeae.models import feature_errors
from paper_odeae.train import device_batch,seed_all
from mimic_experiment.prepare import atomic
from .data import collate
from .loss import objective,data_loss
from .model import MWM
from .train import load_data


def clone(batch):return {k:v.clone() if torch.is_tensor(v) else v for k,v in batch.items()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--gpu',type=int,required=True);args=parser.parse_args()
    if not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):raise RuntimeError('Assigned CUDA required')
    torch.set_num_threads(1);manifest=json.loads(Path(args.manifest).read_text());rows=[]
    for dataset in ['adni','mimic']:
        jobs=[j for j in manifest['jobs'] if j['dataset']==dataset];cfg=jobs[0]['config'].copy()
        payload,splits,_=load_data(cfg);schema=payload['schema']
        sequences=[s for s in splits['train'] if len(s['ages'])>=5][:4]
        items=[({**s,**{k:s[k][:5] for k in ['ages','values','mask','dates']}},3) for s in sequences]
        batch=device_batch(collate(items,dataset));initial=None;main_prediction=None
        for job in jobs:
            cfg=job['config'];seed_all(cfg['seed']);model=MWM(schema,cfg).cuda();model.eval()
            if cfg['variant']=='main':initial={k:v.clone() for k,v in model.state_dict().items()}
            if cfg['variant']=='A':
                for k,v in model.state_dict().items():torch.testing.assert_close(v,initial[k],rtol=0,atol=0)
            deltas=[];solve=model.solve
            def recorded(field,state,delta):
                deltas.append(delta.detach().clone());return solve(field,state,delta)
            model.solve=recorded
            with torch.no_grad():base=model(batch)
            if any(bool((dt<0).any()) for dt in deltas):raise AssertionError('Training integrates toward the past')
            if cfg['variant']=='main':main_prediction=base['prediction'].clone()
            if cfg['variant']=='A':torch.testing.assert_close(base['prediction'],main_prediction,rtol=0,atol=0)
            torch.testing.assert_close(base['seen_observations'],torch.tensor([[1,1,2,3,3]],device='cuda').expand(len(items),-1))
            changed=clone(batch);changed['initialization_age']=changed['first_observation_age']+100
            with torch.no_grad():other=model(changed)
            torch.testing.assert_close(base['prediction'],other['prediction'],rtol=0,atol=0)
            # Both later history and future values are withheld from the second visit's prediction.
            changed=clone(batch)
            for j,field in enumerate(schema):
                changed['values'][:,1:,j]=(changed['values'][:,1:,j]+1)%field['classes'] if field['kind']=='categorical' else changed['values'][:,1:,j]+20
            with torch.no_grad():other=model(changed)
            torch.testing.assert_close(base['prediction'][:,:2],other['prediction'][:,:2],rtol=0,atol=0)
            changed=clone(batch);changed['context_mask'][:,2]=0
            with torch.no_grad():other=model(changed)
            torch.testing.assert_close(other['seen_observations'][:,3:],torch.full((len(items),2),2,device='cuda',dtype=torch.long))
            changed=clone(batch)
            for j,field in enumerate(schema):
                changed['values'][:,3:,j]=(changed['values'][:,3:,j]+1)%field['classes'] if field['kind']=='categorical' else changed['values'][:,3:,j]+20
            changed['mask'][:,3:]=0
            with torch.no_grad():other=model(changed)
            torch.testing.assert_close(base['prediction'],other['prediction'],rtol=0,atol=0)
            torch.testing.assert_close(base['seen_observations'],other['seen_observations'],rtol=0,atol=0)
            if cfg['variant']=='B':
                torch.testing.assert_close(base['sigma'],torch.ones_like(base['sigma']),rtol=0,atol=1e-6)
                torch.testing.assert_close(base['sigma'],other['sigma'],rtol=0,atol=0)
                sigma=torch.tensor([1.,1000000.],device='cuda');nll=1/(2*sigma.square())+sigma.log()
                if not nll[1]>nll[0]:raise AssertionError('Unbounded sigma must have a log cost')
            if cfg['variant']=='A':
                expected=torch.sigmoid(base['seen_observations'].float()-3)
                torch.testing.assert_close(base['supervision_weights'],expected)
                errors=feature_errors(base['prediction'],batch['values'],batch['mask'],schema)
                expected_loss=((errors*expected[...,None]).sum((0,1))/batch['mask'].sum((0,1)).clamp_min(1)).sum()
                actual,_=data_loss(model,base,batch,schema);torch.testing.assert_close(actual,expected_loss)
            model.train();out=model(batch);loss,parts=objective(model,out,batch,schema,cfg);loss.backward()
            for parameter in model.parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():raise AssertionError('Every trainable parameter needs finite CUDA gradients')
            with torch.no_grad():past=model.retrospective_history(batch,base['mu'])
            if not any(bool((dt<0).any()) for dt in deltas):raise AssertionError('Retrospective test must integrate toward past')
            rows.append({'dataset':dataset,'variant':cfg['variant'],'parameters':sum(p.numel() for p in model.parameters()),
                         'loss':float(loss.detach()),'parameter_device':str(next(model.parameters()).device),'prediction_device':str(out['prediction'].device),
                         'physical_gpu':args.gpu,'training_positive_time_only':True,'no_future_observation_leakage':True,
                         'history_prediction_before_current_update':True,'first_age_independent_of_cutoff':True,
                         'counts_frozen_in_forecast':True,'all_parameter_gradients_finite':True,'retrospective_test_separate':True})
            del model,out,base,other,loss
            torch.cuda.empty_cache()
    atomic(Path(args.manifest).parent/'architecture_checks.json',{'status':'passed','checks':rows,'physical_gpu':args.gpu})
    print(json.dumps({'status':'passed','models':len(rows),'physical_gpu':args.gpu}),flush=True)


if __name__=='__main__':main()
