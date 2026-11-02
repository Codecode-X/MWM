"""Single-GPU MWM training, with one complete batch per optimizer update."""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import torch
from torch.utils.data import DataLoader

from paper_odeae.train import seed_all,device_batch
from mimic_experiment.data import ExactBatchSampler
from mimic_experiment.prepare import atomic
from mimic_experiment.train_baseline import normalization_and_modes,save_checkpoint
from src.tracking import SwanLabTracker
from .data import contexts,collate
from .evaluation import evaluate_predictive,evaluate_comparable
from .loss import objective
from .model import MWM


def load_data(cfg,smoke=False):
    raw=Path(cfg['data']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=cfg['dataset_sha256']:raise RuntimeError('Dataset changed')
    payload=pickle.loads(raw)
    splits={k:[s for s in payload['sequences'] if s['split']==k] for k in ['train','validation','test']}
    if cfg['dataset_name']=='mimic':
        if payload['metadata']['dataset_revision']!='chronology_v2':raise RuntimeError('Require chronology_v2')
        means,modes=normalization_and_modes(payload)
        cfg.update(normalized_train_means=means,raw_scale=payload['metadata']['scale'])
    else:modes=None
    if smoke:
        if cfg['dataset_name']=='mimic':
            eligible=contexts(splits['train'],'mimic',training=True,seed=cfg['seed'])
            splits['train']=[splits['train'][i] for i in eligible.indices[:128]]
        splits={k:rows[:(128 if cfg['dataset_name']=='mimic' else 8) if k=='train' else 8] for k,rows in splits.items()}
        if cfg['dataset_name']=='adni':
            splits={k:[{**s,**{field:s[field][:4] for field in ['ages','values','mask','dates']}} for s in rows] for k,rows in splits.items()}
    return payload,splits,modes


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=['smoke','training'],default='training');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());smoke=args.stage=='smoke'
    if Path(sys.prefix).name!='ode' or not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(cfg['physical_gpu']):
        raise RuntimeError('MWM requires ode and the assigned physical CUDA GPU')
    if cfg['gradient_accumulation_steps']!=1:raise ValueError('MWM uses no gradient accumulation')
    torch.set_num_threads(1);torch.cuda.set_device(0);seed_all(cfg['seed'])
    out=Path(cfg['comparison_output'])/'checks'/cfg['variant']/'smoke' if smoke else Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():raise RuntimeError('Completed experiment already exists')
    payload,splits,modes=load_data(cfg,smoke);schema=payload['schema'];meta=payload['metadata']
    if smoke:cfg.update(max_epochs=1,min_epochs=0,patience=1)
    cfg['smoke']=smoke;model=MWM(schema,cfg).cuda()
    hardware={'physical_gpu':cfg['physical_gpu'],'gpu_name':torch.cuda.get_device_name(),'environment':Path(sys.prefix).name,
              'parameter_device':str(next(model.parameters()).device),'parameters':sum(p.numel() for p in model.parameters()),
              'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
              'torch_version':torch.__version__,'world_size':1,'training_batch_size':cfg['batch_size'],
              'gradient_accumulation_steps':1,'forward_calls_per_optimizer_step':1,'backward_calls_per_optimizer_step':1,
              'dataset_sha256':cfg['dataset_sha256'],'time_unit':'hour' if cfg['dataset_name']=='mimic' else 'year',
              'initialization_age':'first actual observation age','latent_dimension':28,'variant':cfg['variant'],
              'shared_vector_field':True,'training_negative_time_integration':False,
              'parameter_breakdown':{name:sum(p.numel() for p in child.parameters()) for name,child in model.named_children()}}
    for name,value in [('config',cfg),('hardware',hardware),('schema',schema),('data',meta)]:atomic(out/f'{name}.json',value)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:cfg['lr_decay']**(step/cfg['lr_decay_steps']))
    history=[];best=float('inf');bad=0;steps=0;first=1;elapsed_base=0.
    if (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        history=saved['history'];best=saved['best_score'];bad=saved['bad_epochs'];steps=saved['optimizer_steps'];first=saved['epoch']+1;elapsed_base=saved['elapsed_seconds']
    dataset=contexts(splits['train'],cfg['dataset_name'],training=True,seed=cfg['seed'])
    with SwanLabTracker(out,cfg,'MWM_'+cfg['dataset_name'],model,cfg['seed']) as tracker:
        tracker.step=steps;started=time.monotonic()
        for epoch in range(first,cfg['max_epochs']+1):
            if epoch>cfg['min_epochs'] and bad>=cfg['patience']:break
            seed_all(cfg['seed']+epoch);model.train();begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
            if cfg['dataset_name']=='mimic':
                dataset.set_epoch(epoch);sampler=ExactBatchSampler(len(dataset),cfg['batch_size'],0,1,cfg['seed']);sampler.set_epoch(epoch)
                batches=(collate([dataset[i] for i in indices],'mimic',cfg['input_dropout']) for indices in sampler)
            else:
                batches=DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(cfg['seed']+epoch),
                                   collate_fn=partial(collate,dataset='adni',dropout=cfg['input_dropout']))
            sums={};batches_run=0;minimum=cfg['batch_size'];maximum=0
            for cpu in batches:
                batch=device_batch(cpu);size=len(batch['cutoff']);minimum=min(minimum,size);maximum=max(maximum,size)
                optimizer.zero_grad(set_to_none=True);prediction=model(batch);loss,parts=objective(model,prediction,batch,schema,cfg)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite MWM objective')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                optimizer.step();scheduler.step();steps+=1;batches_run+=1
                scalars={k:float(v.detach()) for k,v in {**parts,'loss':loss,'gradient_norm':norm}.items()}
                for key,value in scalars.items():sums[key]=sums.get(key,0.)+value
                tracker.batch(scalars['loss'],scheduler.get_last_lr()[0],**{k:v for k,v in scalars.items() if k!='loss'},actual_batch_size=size)
                if 'actual_tensor_devices' not in hardware:
                    hardware.update(actual_tensor_devices={k:str(v.device) for k,v in batch.items() if torch.is_tensor(v)},
                                    prediction_device=str(prediction['prediction'].device),first_actual_training_batch=size)
                    atomic(out/'hardware.json',hardware)
            values={k:v/batches_run for k,v in sums.items()}
            validation=evaluate_predictive(model,splits['validation'],schema,cfg);score=validation['overall']['reconstruction_error']
            if not torch.isfinite(torch.tensor(score,device='cuda')):raise FloatingPointError('Nonfinite validation metric')
            row={'epoch':epoch,'optimizer_steps':steps,'training_batches':batches_run,'minimum_batch_size':minimum,'maximum_batch_size':maximum,
                 'gradient_accumulation_steps':1,'training_loss':values['loss'],'training_components':values,'validation':validation,
                 'lr':scheduler.get_last_lr()[0],'epoch_seconds':time.monotonic()-begin,'elapsed_seconds':elapsed_base+time.monotonic()-started,
                 'peak_memory_bytes':torch.cuda.max_memory_allocated()};history.append(row)
            saved={'state_dict':model.state_dict(),'config':cfg,'schema':schema,'metadata':meta,'epoch':epoch,
                   'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()}
            if score<best:best=score;bad=0;save_checkpoint(out/'best.pt',saved)
            else:bad+=1
            save_checkpoint(out/'last.pt',{**saved,'history':history,'best_score':best,'bad_epochs':bad,'optimizer_steps':steps,'elapsed_seconds':row['elapsed_seconds']})
            atomic(out/'history.json',history);tracker.epoch(epoch,values,validation,row['lr'],epoch_seconds=row['epoch_seconds'],peak_memory_bytes=row['peak_memory_bytes'])
            print(json.dumps({'epoch':epoch,'training_loss':values['loss'],'validation_causal_overall':score,
                              'validation_forecast':validation['forecast']['reconstruction_error'],'epoch_seconds':row['epoch_seconds'],
                              'peak_memory_gib':row['peak_memory_bytes']/2**30}),flush=True)
        saved=torch.load(out/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['state_dict'])
        causal_val=evaluate_predictive(model,splits['validation'],schema,cfg)
        difference=abs(causal_val['overall']['reconstruction_error']-history[saved['epoch']-1]['validation']['overall']['reconstruction_error'])
        if difference>1e-6:raise AssertionError('Checkpoint reload changed validation')
        causal_test=evaluate_predictive(model,splits['test'],schema,cfg)
        metrics={'status':'cuda_smoke_passed' if smoke else 'complete','best_epoch':saved['epoch'],'epochs_run':len(history),
                 'checkpoint_selection':'unweighted causal overall MSE + CE; no A weights or B NLL',
                 'causal_validation':causal_val,'causal_test':causal_test,'checkpoint_reload_difference':difference,
                 'training_batch_size':cfg['batch_size'],'gradient_accumulation_steps':1}
        for split in ['validation','test']:
            metrics[split]=evaluate_comparable(model,splits[split],schema,cfg,meta,modes)
            diff=abs(metrics[split]['forecast']['reconstruction_error']-metrics['causal_'+split]['forecast']['reconstruction_error'])
            if diff>1e-6:raise AssertionError('Retrospective adapter changed future predictions')
        if cfg['dataset_name']=='mimic':
            metrics['test_early_contexts']={str(h):evaluate_comparable(model,splits['test'],schema,cfg,meta,modes,horizon=h) for h in [6,12]}
        metrics['elapsed_seconds']=elapsed_base+time.monotonic()-started
        tracker.final(metrics);atomic(out/'metrics.json',metrics);print(json.dumps({'complete':str(out),'status':metrics['status']}),flush=True)


if __name__=='__main__':main()
