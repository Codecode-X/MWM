"""CUDA check that the new run changes only the weak-prior coefficient."""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys

import torch

from mimic_experiment.data import collate
from mimic_experiment.prepare import atomic
from mimic_experiment.train_full_batch import make_model,loss_function
from paper_odeae.train import seed_all,device_batch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--gpu',required=True,type=int);args=parser.parse_args()
    if Path(sys.prefix).name!='ode' or not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):raise RuntimeError('Assigned CUDA GPU and ode required')
    torch.set_num_threads(1);torch.cuda.set_device(0)
    manifest=json.loads(Path(args.manifest).read_text());base=manifest['reference_config'];cfg=manifest['config']
    differences={k for k in set(base)|set(cfg) if base.get(k)!=cfg.get(k)}
    if differences!={'prior_weight','output','comparison_output'}:raise AssertionError('More than one experimental factor changed')
    if cfg['prior_weight']!=0 or cfg['batch_size']!=128 or cfg['gradient_accumulation_steps']!=1 or not cfg['collect_prior_predictions']:raise AssertionError('Keep original full-batch computation')
    payload=pickle.loads(Path(cfg['data']).read_bytes());schema=payload['schema']
    sequences=[s for s in payload['sequences'] if s['split']=='train' and len(s['ages'])>=8][:4]
    items=[({**s,**{k:s[k][:8] for k in ['ages','values','mask','dates']}},4) for s in sequences]
    batch=device_batch(collate(items))
    seed_all(0);reference=make_model(schema,base).cuda()
    seed_all(0);model=make_model(schema,cfg).cuda()
    for key,value in reference.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
    reference.eval();model.eval();original=reference(batch);out=model(batch)
    torch.testing.assert_close(original['prediction'],out['prediction'],rtol=0,atol=0)
    torch.testing.assert_close(original['prior_prediction'],out['prior_prediction'],rtol=0,atol=0)
    a,ap=loss_function(reference,original,batch,schema,base);b,bp=loss_function(model,out,batch,schema,cfg)
    prior=bp[2]+bp[3]
    torch.testing.assert_close(a-b,.01*prior,rtol=2e-4,atol=2e-6)
    b.backward()
    for p in model.parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():raise AssertionError('Every parameter requires finite CUDA gradients')
    result={'status':'passed','physical_gpu':args.gpu,'parameters':sum(p.numel() for p in model.parameters()),
            'parameter_device':str(next(model.parameters()).device),'prediction_device':str(out['prediction'].device),
            'identical_initial_parameters':True,'identical_initial_predictions':True,'prior_forward_computation_retained':True,
            'loss_difference_is_only_0_01_times_prior':True,'finite_parameter_gradients':True,'full_batch_test':'performed separately by original CUDA smoke entry point'}
    atomic(Path(args.manifest).parent/'cuda_loss_check.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
