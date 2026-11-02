"""Train either direction with one complete 128-patient batch per optimizer step."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch

from paper_odeae.models import paper_model_from_config,feature_errors,reconstruction,prior_errors
from paper_odeae.jacobian import jacobian_regularization
from paper_odeae.train import seed_all,device_batch
from src.tracking import SwanLabTracker
from .data import ICUContexts,ExactBatchSampler,collate
from .evaluation import evaluate
from .prepare import atomic
from .train_baseline import normalization_and_modes,save_checkpoint


def make_model(schema,cfg):
    model=paper_model_from_config(schema,cfg)
    model.variant=cfg['variant'];model.generator_anchor=cfg['generator_anchor']
    return model


def loss_function(model,out,batch,schema,cfg):
    mse,cce=reconstruction(feature_errors(out['prediction'],batch['values'],batch['mask'],schema),batch['mask'],schema)
    pe,pmask=prior_errors(out,batch,schema);pmse,pcce=reconstruction(pe,pmask,schema)
    jac,_,_=jacobian_regularization(model,out,batch);l2=model.ode_regularization()
    loss=cfg['mse_weight']*mse+cfg['cce_weight']*cce+cfg['prior_weight']*(pmse+pcce)+cfg['jacobian_weight']*jac+cfg['l2_weight']*l2
    return loss,torch.stack([mse.detach(),cce.detach(),pmse.detach(),pcce.detach(),jac.detach(),l2.detach()])


def components(vector,cfg):
    result={k:float(v) for k,v in zip(['mse','cce','prior_mse','prior_cce','jacobian_frobenius_squared','ode_l2','gradient_norm'],vector)}
    result.update(kl=0.,weighted_prior=cfg['prior_weight']*(result['prior_mse']+result['prior_cce']),
                  weighted_jacobian=cfg['jacobian_weight']*result['jacobian_frobenius_squared'],weighted_ode_l2=cfg['l2_weight']*result['ode_l2'])
    result['loss']=cfg['mse_weight']*result['mse']+cfg['cce_weight']*result['cce']+result['weighted_prior']+result['weighted_jacobian']+result['weighted_ode_l2']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--stage',choices=['smoke','training'],default='training');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());smoke=args.stage=='smoke'
    if Path(sys.prefix).name!='ode' or not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(cfg['physical_gpu']):raise RuntimeError('Use ode and the assigned physical CUDA GPU')
    if cfg['batch_size']!=128 or cfg['gradient_accumulation_steps']!=1:raise RuntimeError('Use complete batch128 with no gradient accumulation')
    torch.set_num_threads(1);torch.cuda.set_device(0);seed_all(cfg['seed'])
    out=Path(cfg['comparison_output'])/'checks'/cfg['variant']/'smoke' if smoke else Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():raise RuntimeError('Completed experiment already exists')
    path=Path(cfg['data']);raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    if sha!=cfg['dataset_sha256']:raise RuntimeError('MIMIC dataset changed')
    payload=pickle.loads(raw);del raw
    schema=payload['schema'];meta=payload['metadata']
    if meta['dataset_revision']!='chronology_v2':raise RuntimeError('Use chronology_v2 data')
    means,modes=normalization_and_modes(payload);cfg.update(raw_scale=meta['scale'],normalized_train_means=means,smoke=smoke)
    splits={k:[s for s in payload['sequences'] if s['split']==k] for k in ['train','validation','test']}
    if smoke:
        cfg.update(max_epochs=1,min_epochs=0,patience=1)
        splits={k:rows[:128 if k=='train' else 8] for k,rows in splits.items()}
    model=make_model(schema,cfg).cuda();parameters=sum(p.numel() for p in model.parameters())
    if parameters!=43280 or parameters!=sum(p.numel() for p in model.parameters() if p.requires_grad):raise AssertionError('Model capacity changed')
    hardware={'physical_gpu':cfg['physical_gpu'],'gpu_name':torch.cuda.get_device_name(),'environment':Path(sys.prefix).name,'torch_version':torch.__version__,
              'parameter_device':str(next(model.parameters()).device),'parameters':parameters,'trainable_parameters':parameters,
              'world_size':1,'effective_batch_size':128,'training_batch_size':128,'gradient_accumulation_steps':1,
              'forward_calls_per_optimizer_step':1,'backward_calls_per_optimizer_step':1,'gradient_partitions':1,
              'dataset_sha256':sha,'time_unit':'hour','latent_dimension':28,'encoder_direction':model.encoder_direction,
              'generator_anchor':model.generator_anchor,'shared_vector_field':True,'variant':cfg['variant'],
              'prior_weight':cfg['prior_weight'],'jacobian_weight':cfg['jacobian_weight'],'l2_weight':cfg['l2_weight'],
              'evaluation_batch_size':32,'evaluation_protocol':'same four fixed patient shards as the historical reference; no gradients',
              'parameter_breakdown':{name:sum(p.numel() for p in child.parameters()) for name,child in model.named_children()}}
    for name,value in [('config',cfg),('hardware',hardware),('schema',schema),('data',meta)]:atomic(out/f'{name}.json',value)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr']);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:cfg['lr_decay']**(step/cfg['lr_decay_steps']))
    history=[];best=float('inf');bad=0;steps=0;first=1;elapsed_base=0.
    if (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        history=saved['history'];best=saved['best_score'];bad=saved['bad_epochs'];steps=saved['optimizer_steps'];first=saved['epoch']+1;elapsed_base=saved['elapsed_seconds']
    dataset=ICUContexts(splits['train'],training=True,seed=cfg['seed']);sampler=ExactBatchSampler(len(dataset),128,0,1,cfg['seed'])
    with SwanLabTracker(out,cfg,'mimic_direction_full_batch',model,cfg['seed']) as tracker:
        tracker.step=steps;started=time.monotonic()
        for epoch in range(first,cfg['max_epochs']+1):
            if epoch>cfg['min_epochs'] and bad>=cfg['patience']:break
            seed_all(cfg['seed']+epoch);dataset.set_epoch(epoch);sampler.set_epoch(epoch);model.train();begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
            total=torch.zeros(7,device='cuda');batches=0;min_size=128;max_size=0
            for indices in sampler:
                batch=device_batch(collate([dataset[i] for i in indices],cfg['input_dropout']));size=len(indices)
                min_size=min(min_size,size);max_size=max(max_size,size);optimizer.zero_grad(set_to_none=True)
                out_model=model(batch,sample=False);loss,parts=loss_function(model,out_model,batch,schema,cfg)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-batch loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                optimizer.step();scheduler.step();steps+=1;batches+=1;vector=torch.cat([parts,norm.detach()[None]]);total+=vector
                values=components(vector,cfg);tracker.batch(values['loss'],scheduler.get_last_lr()[0],**{k:v for k,v in values.items() if k!='loss'},actual_batch_size=size)
                if 'actual_tensor_devices' not in hardware:
                    hardware.update(actual_tensor_devices={k:str(v.device) for k,v in batch.items() if torch.is_tensor(v)},
                                    prediction_device=str(out_model['prediction'].device),jacobian_device=str(parts[4].device),first_actual_training_batch=size)
                    atomic(out/'hardware.json',hardware)
            validation=evaluate(model,splits['validation'],schema,cfg,meta['raw_train_std'],modes);score=validation['overall']['reconstruction_error'];values=components(total/batches,cfg)
            row={'epoch':epoch,'optimizer_steps':steps,'training_batches':batches,'training_forward_calls':batches,'training_backward_calls':batches,
                 'minimum_batch_size':min_size,'maximum_batch_size':max_size,'gradient_accumulation_steps':1,'training_loss':values['loss'],'training_components':values,
                 'validation':validation,'lr':scheduler.get_last_lr()[0],'epoch_seconds':time.monotonic()-begin,
                 'elapsed_seconds':elapsed_base+time.monotonic()-started,'peak_memory_bytes':torch.cuda.max_memory_allocated()};history.append(row)
            saved={'state_dict':model.state_dict(),'config':cfg,'schema':schema,'metadata':meta,'epoch':epoch,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()}
            if score<best:best=score;bad=0;save_checkpoint(out/'best.pt',saved)
            else:bad+=1
            save_checkpoint(out/'last.pt',{**saved,'history':history,'best_score':best,'bad_epochs':bad,'optimizer_steps':steps,'elapsed_seconds':row['elapsed_seconds']})
            atomic(out/'history.json',history);tracker.epoch(epoch,values,validation,row['lr'],epoch_seconds=row['epoch_seconds'],peak_memory_bytes=row['peak_memory_bytes'])
            print(json.dumps({'epoch':epoch,'training_loss':values['loss'],'validation_overall':score,'validation_forecast':validation['forecast']['reconstruction_error'],
                              'epoch_seconds':row['epoch_seconds'],'max_batch_size':max_size,'gradient_accumulation_steps':1,'peak_memory_gib':row['peak_memory_bytes']/2**30}),flush=True)
        saved=torch.load(out/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['state_dict'])
        validation=evaluate(model,splits['validation'],schema,cfg,meta['raw_train_std'],modes)
        difference=abs(validation['overall']['reconstruction_error']-history[saved['epoch']-1]['validation']['overall']['reconstruction_error'])
        if difference>1e-6:raise AssertionError('Checkpoint reload changed validation')
        test=evaluate(model,splits['test'],schema,cfg,meta['raw_train_std'],modes)
        early={str(h):evaluate(model,splits['test'],schema,cfg,meta['raw_train_std'],modes,horizon=h) for h in [6,12]}
        metrics={'status':'cuda_smoke_passed' if smoke else 'complete','best_epoch':saved['epoch'],'epochs_run':len(history),
                 'validation':validation,'test':test,'test_early_contexts':early,'checkpoint_reload_difference':difference,
                 'elapsed_seconds':elapsed_base+time.monotonic()-started,'training_batch_size':128,'gradient_accumulation_steps':1}
        tracker.final(metrics);atomic(out/'metrics.json',metrics);print(json.dumps({'complete':str(out),'status':metrics['status']}),flush=True)


if __name__=='__main__':main()
