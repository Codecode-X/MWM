"""CUDA verification of weak-prior values and gradients in future-only training."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys

import torch

from mimic_experiment.data import collate
from mimic_experiment.prepare import atomic
from mimic_experiment.train_future_only import make_model,loss_function,future_mask
from paper_odeae.models import prior_errors,reconstruction
from paper_odeae.train import device_batch,seed_all


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);args=parser.parse_args()
    if Path(sys.prefix).name!='ode' or not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise RuntimeError('Use ode and physical GPU3 for this check')
    torch.set_num_threads(1);torch.cuda.set_device(0)
    manifest=json.loads(Path(args.manifest).read_text());base=manifest['reference_config'];jobs=manifest['jobs']
    payload=pickle.loads(Path(base['data']).read_bytes());schema=payload['schema']
    rows=[s for s in payload['sequences'] if s['split']=='train' and len(s['ages'])>=8][:4]
    items=[({**s,**{k:s[k][:8] for k in ['ages','values','mask','dates']}},4) for s in rows];batch=device_batch(collate(items));future=future_mask(batch)
    old_file=Path(manifest['reference_source'])/'mimic_experiment/train_future_only.py'
    spec=importlib.util.spec_from_file_location('completed_future_zero',old_file);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    seed_all(0);completed_model=old.make_model(schema,base).cuda()
    seed_all(0);model=make_model(schema,base).cuda()
    for key,value in completed_model.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
    completed_model.train();model.train();legacy=completed_model(batch);out=model(batch)
    torch.testing.assert_close(legacy['prediction'],out['prediction'],rtol=0,atol=0)
    old_loss,_=old.loss_function(completed_model,legacy,batch,schema,base)
    zero,_=loss_function(model,out,batch,schema,base);torch.testing.assert_close(old_loss,zero,rtol=0,atol=0)
    params=tuple(model.parameters());old_grads=torch.autograd.grad(old_loss,tuple(completed_model.parameters()))
    zero_grads=torch.autograd.grad(zero,params,retain_graph=True)
    for a,b in zip(old_grads,zero_grads):torch.testing.assert_close(a,b,rtol=0,atol=0)
    pe,mask=prior_errors(out,batch,schema);pmse,pcce=reconstruction(pe,mask,schema);prior=pmse+pcce
    prior_grads=torch.autograd.grad(prior,params,retain_graph=True)
    records=[]
    for job in jobs:
        cfg=job['config'];weight=cfg['prior_weight'];seed_all(0);candidate=make_model(schema,cfg).cuda()
        for key,value in model.state_dict().items():torch.testing.assert_close(value,candidate.state_dict()[key],rtol=0,atol=0)
        deltas=[];solve=candidate.solve
        def recorded(field,state,dt):deltas.append(dt.detach().clone());return solve(field,state,dt)
        candidate.solve=recorded;candidate.train();candidate_out=candidate(batch)
        torch.testing.assert_close(candidate_out['prediction'][future],out['prediction'][future],rtol=0,atol=0)
        torch.testing.assert_close(candidate_out['prior_prediction'],out['prior_prediction'],rtol=0,atol=0)
        if any(bool((dt<0).any()) for dt in deltas):raise AssertionError('Training must not integrate toward the past')
        loss,_=loss_function(model,out,batch,schema,cfg)
        torch.testing.assert_close(loss-zero,weight*prior,rtol=3e-4,atol=3e-6)
        grads=torch.autograd.grad(loss,params,retain_graph=True)
        for actual,zg,pg in zip(grads,zero_grads,prior_grads):
            torch.testing.assert_close(actual,zg+weight*pg,rtol=4e-4,atol=4e-6)
            if not torch.isfinite(actual).all():raise AssertionError('Finite parameter gradients required')
        difference=sum((a-z).abs().sum() for a,z in zip(grads,zero_grads))
        if not difference>0:raise AssertionError('Positive prior must affect the model gradient')
        candidate.eval()
        with torch.no_grad():evaluated=candidate(batch)
        if not any(bool((dt<0).any()) for dt in deltas):raise AssertionError('Evaluation must retain common history reconstruction')
        torch.testing.assert_close(evaluated['prediction'][future],out['prediction'][future],rtol=0,atol=0)
        records.append({'id':job['id'],'prior_weight':weight,'parameters':sum(p.numel() for p in candidate.parameters()),
                        'same_initial_weights_and_future_predictions':True,'loss_increment_matches_weighted_prior':True,
                        'gradient_increment_matches_weighted_prior':True,'prior_gradient_l1_increment':float(difference.detach()),
                        'no_negative_training_integration':True,'common_history_evaluation_retained':True,
                        'parameter_device':str(next(candidate.parameters()).device),'prediction_device':str(candidate_out['prediction'].device)})
    result={'status':'passed','physical_gpu':3,'completed_zero_weight_value_and_gradients_unchanged':True,'jobs':records}
    atomic(Path(args.manifest).parent/'cuda_checks.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
