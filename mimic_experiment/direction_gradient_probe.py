"""Fixed-cohort CUDA gradient and inverse-flow sensitivity measurements."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from direction_gradient_common import initialize,load_model,component_gradients,encode_only,flow_chain,flow_jacobian,summary
import sys
sys.path=[p for p in sys.path if Path(p or '.').resolve()!=Path(__file__).resolve().parent]
from mimic_experiment.data import ICUContexts,ExactBatchSampler,collate
from mimic_experiment.prepare import atomic
from paper_odeae.train import seed_all,device_batch


def description(items,batch):
    return {'subject_ids':[int(s['subject_id']) for s,c in items], 'cutoffs':[int(c) for s,c in items],
            'context_mask_sha256':hashlib.sha256(batch['context_mask'].detach().cpu().numpy().tobytes()).hexdigest()}


def cohorts(payload):
    seed_all(41);result=[]
    for split,count in [('train',8),('validation',2)]:
        rows=[s for s in payload['sequences'] if s['split']==split]
        dataset=ICUContexts(rows,training=split=='train',seed=0);dataset.set_epoch(41)
        sampler=ExactBatchSampler(len(dataset),128,0,1,0);sampler.set_epoch(41)
        for n,indices in enumerate(sampler):
            if n>=count:break
            items=[dataset[i] for i in indices];batch=device_batch(collate(items,.1 if split=='train' else 0.))
            result.append((split,n,batch,description(items,batch)))
    return result


def gradients(entry,model,payload,cfg,batches,saved):
    result=[]
    for split,index,batch,info in batches:
        output=model(batch,sample=False)
        row,_,_=component_gradients(model,output,batch,payload['schema'],cfg,detail=index==0)
        result.append({'split':split,'batch':index,'patients':len(batch['cutoff']),**info,**row})
        del output;gc.collect()
    if 'probe_batch' in saved:
        batch=saved['probe_batch'];output=model(batch,sample=False)
        row,_,_=component_gradients(model,output,batch,payload['schema'],cfg,detail=True)
        result.append({'split':'actual_peak_batch','batch':saved['batch'],'patients':len(batch['cutoff']),**row})
    return {'batches':result,'field_parameters':sum(p.numel() for p in model.forward_ode.parameters()),
            'normalization':'Each field uses its total valid history+future count, exactly as in the main training loss.'}


def sensitivity(entry,model,payload,cfg):
    rows=[s for s in payload['sequences'] if s['split']=='validation'];dataset=ICUContexts(rows,seed=0)
    indices=np.random.default_rng(7).choice(len(dataset),64,replace=False)
    items=[dataset[int(i)] for i in indices];batch=device_batch(collate(items));info=description(items,batch)
    with torch.no_grad():
        states=encode_only(model,batch);reference=model(batch,sample=False)['mu']
        encoder_error=float((states-reference).abs().max())
        if encoder_error>1e-6:raise RuntimeError('Encode-only helper differs from original model')
    results=[]
    for label,rtol,atol,kind,direction in [
        ('history_native',1e-4,1e-6,'history',1),('matched_future_native',1e-4,1e-6,'history',-1),
        ('history_tight',1e-6,1e-8,'history',1),('matched_future_tight',1e-6,1e-8,'history',-1),
        ('past_6h',1e-6,1e-8,6,-1),('future_6h',1e-6,1e-8,6,1),
        ('past_12h',1e-6,1e-8,12,-1),('future_12h',1e-6,1e-8,12,1),
        ('past_24h',1e-6,1e-8,24,-1),('future_24h',1e-6,1e-8,24,1)]:
        model.rtol,model.atol=rtol,atol
        if kind=='history':flow=lambda x:flow_chain(model,x,batch,direction)
        else:flow=lambda x:model.solve(model.forward_ode,x,x.new_full((len(x),),kind*direction))
        jac=flow_jacobian(flow,states);_,singular,vh=torch.linalg.svd(jac);gain=singular[:,0]
        record={'condition':label,'rtol':rtol,'atol':atol,'maximum_gain':summary(gain),
                'frobenius':summary(jac.norm(dim=(-2,-1))),
                'fraction_above_1':float((gain>1).float().mean()),'fraction_above_10':float((gain>10).float().mean()),
                'per_patient_gain':gain.cpu().tolist(),'per_patient_singular_values':singular.cpu().tolist()}
        if label in ['history_tight','matched_future_tight']:
            checks=[]
            for relative in [1e-4,1e-3]:
                epsilon=relative*states.norm(dim=-1).clamp_min(1.);perturb=epsilon[:,None]*vh[:,0,:]
                with torch.no_grad():observed=(flow(states+perturb)-flow(states-perturb)).norm(dim=-1)/(2*epsilon)
                checks.append({'relative_perturbation':relative,'gain':summary(observed),
                               'relative_error_to_jacobian':summary((observed-gain).abs()/gain.clamp_min(1e-20))})
            record['finite_perturbation_checks']=checks
        results.append(record)
        print(json.dumps({'checkpoint':entry['id'],'condition':label,'gain_median':record['maximum_gain']['median'],'gain_max':record['maximum_gain']['max']}),flush=True)
    model.rtol,model.atol=cfg['rtol'],cfg['atol']
    return {**info,'patients':len(items),'state_dimension':states.shape[-1],'encode_only_maximum_error':encoder_error,
            'actual_tensor_devices':{'states':str(states.device),'jacobian':str(jac.device),'singular_values':str(singular.device)},
            'history_duration_hours':(batch['ages'].gather(1,(batch['cutoff']-1)[:,None]).squeeze(1)-batch['ages'][:,0]).cpu().tolist(),
            'conditions':results,'definition':'Largest singular value of D_z Phi; same absolute intervals in the future control.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--kind',choices=['gradients','sensitivity'],required=True);args=parser.parse_args()
    out=Path(args.out);manifest,cfg,payload,modes=initialize(out,args.gpu);folder=out/args.kind;folder.mkdir(exist_ok=True)
    batches=cohorts(payload) if args.kind=='gradients' else None
    hardware={'physical_gpu':args.gpu,'parameter_device':'cuda:0','tensor_device':'cuda:0','gpu_name':torch.cuda.get_device_name(),
              'torch_version':torch.__version__,'source_snapshot':manifest['source_snapshot'],'dataset_sha256':cfg['dataset_sha256'],
              'analysis_source':manifest.get('analysis_source')}
    atomic(folder/'hardware.json',hardware)
    while True:
        index=json.loads((out/'checkpoint_index.json').read_text())
        pending=[e for e in index['entries'] if not (folder/(e['id']+'.json')).exists()]
        for entry in pending:
            print(json.dumps({'start':entry['id'],'kind':args.kind}),flush=True);begin=time.monotonic()
            model,saved=load_model(entry,cfg,payload['schema'],requires_grad=args.kind=='gradients')
            hardware.update(actual_parameter_device=str(next(model.parameters()).device))
            if batches is not None:hardware.update(actual_tensor_devices={k:str(v.device) for k,v in batches[0][2].items() if torch.is_tensor(v)})
            atomic(folder/'hardware.json',hardware)
            result=gradients(entry,model,payload,cfg,batches,saved) if args.kind=='gradients' else sensitivity(entry,model,payload,cfg)
            atomic(folder/(entry['id']+'.json'),{'checkpoint':entry,'hardware':hardware,'elapsed_seconds':time.monotonic()-begin,**result})
            del model,saved;gc.collect();torch.cuda.empty_cache()
        if index.get('producer_done') and not pending:break
        atomic(folder/'status.json',{'status':'waiting_for_checkpoint','completed':len(list(folder.glob('epoch_*.json')))})
        time.sleep(5)
    atomic(folder/'status.json',{'status':'complete'})


if __name__=='__main__':main()
