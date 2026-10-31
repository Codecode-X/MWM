"""Single-GPU history ablations with the original four-rank MIMIC loss reductions."""
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

from paper_odeae.models import paper_model_from_config,feature_errors,prior_errors
from paper_odeae.jacobian import jacobian_regularization
from paper_odeae.train import seed_all,device_batch
from src.tracking import SwanLabTracker
from .data import ICUContexts,ExactBatchSampler,collate
from .evaluation import evaluate
from .prepare import atomic
from .train_baseline import normalization_and_modes,save_checkpoint


def partition_batches(dataset,indices,input_dropout,rngs):
    """Reproduce each original rank's padding, patient shard and NumPy dropout."""
    batches=[]
    for rank in range(4):
        local=indices[rank::4]
        if not local:continue
        batch=collate([dataset[i] for i in local])
        if input_dropout:
            keep=rngs[rank].random_sample(tuple(batch['context_mask'].shape))>=input_dropout
            batch['context_mask']*=torch.from_numpy(keep)
        batches.append(device_batch(batch))
    return batches


def denominators(batches):
    counts=sum(batch['mask'].sum((0,1)) for batch in batches)
    prior_counts=torch.zeros_like(counts);states=counts.new_zeros(())
    for batch in batches:
        n=batch['max_context'];i=torch.arange(n,device='cuda')[None]
        mask=batch['mask'][:,:n]*((i>0)&(i<batch['cutoff'][:,None]))[...,None]
        prior_counts+=mask.sum((0,1));states+=batch['valid'].sum()
    return counts,prior_counts,states


def local_loss(model,out,batch,schema,cfg,denoms):
    counts,pcounts,states=denoms
    numeric=torch.tensor([field['kind']=='numerical' for field in schema],device='cuda')
    errors=feature_errors(out['prediction'],batch['values'],batch['mask'],schema)
    terms=errors.sum((0,1))/counts.clamp_min(1);mse=terms[numeric].sum();cce=terms[~numeric].sum()
    pe,_=prior_errors(out,batch,schema);terms=pe.sum((0,1))/pcounts.clamp_min(1)
    pmse=terms[numeric].sum();pcce=terms[~numeric].sum()
    jac,_,n=jacobian_regularization(model,out,batch);jac=jac*n/states.clamp_min(1)
    loss=cfg['mse_weight']*mse+cfg['cce_weight']*cce+cfg['prior_weight']*(pmse+pcce)+cfg['jacobian_weight']*jac
    return loss,torch.stack([mse.detach(),cce.detach(),pmse.detach(),pcce.detach(),jac.detach()])


def components(vector,cfg):
    result={key:float(v) for key,v in zip(['mse','cce','prior_mse','prior_cce','jacobian_frobenius_squared','ode_l2','gradient_norm'],vector)}
    result.update(kl=0.,weighted_prior=cfg['prior_weight']*(result['prior_mse']+result['prior_cce']),
                  weighted_jacobian=cfg['jacobian_weight']*result['jacobian_frobenius_squared'],weighted_ode_l2=cfg['l2_weight']*result['ode_l2'])
    result['loss']=cfg['mse_weight']*result['mse']+cfg['cce_weight']*result['cce']+result['weighted_prior']+result['weighted_jacobian']+result['weighted_ode_l2']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--stage',choices=['smoke','training'],default='training');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());smoke=args.stage=='smoke'
    if Path(sys.prefix).name!='ode' or not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(cfg['physical_gpu']):
        raise RuntimeError('Use ode and the assigned physical CUDA GPU; CPU fallback is disabled')
    torch.set_num_threads(1);torch.cuda.set_device(0);seed_all(cfg['seed'])
    out=Path(cfg['comparison_output'])/'checks'/cfg['variant']/'smoke' if smoke else Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():raise RuntimeError('Completed experiment already exists')
    dataset=Path(cfg['data']);sha=hashlib.sha256(dataset.read_bytes()).hexdigest()
    if sha!=cfg['dataset_sha256']:raise RuntimeError('Reference dataset changed')
    payload=pickle.loads(dataset.read_bytes());schema=payload['schema'];meta=payload['metadata']
    if meta['dataset_revision']!='chronology_v2':raise RuntimeError('Use the audited chronology_v2 MIMIC data')
    means,modes=normalization_and_modes(payload);cfg.update(raw_scale=meta['scale'],normalized_train_means=means,smoke=smoke)
    splits={key:[s for s in payload['sequences'] if s['split']==key] for key in ['train','validation','test']}
    if smoke:
        cfg.update(max_epochs=1,min_epochs=0,patience=1)
        splits={key:rows[:32 if key=='train' else 8] for key,rows in splits.items()}
    model=paper_model_from_config(schema,cfg).cuda();parameters=sum(p.numel() for p in model.parameters())
    if parameters!=cfg['target_parameters'] or parameters!=43280:raise AssertionError('Matched capacity changed')
    hardware={'physical_gpu':cfg['physical_gpu'],'gpu_name':torch.cuda.get_device_name(),'environment':Path(sys.prefix).name,
              'torch_version':torch.__version__,'parameter_device':str(next(model.parameters()).device),'parameters':parameters,
              'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'world_size':1,
              'effective_batch_size':cfg['batch_size'],'max_micro_batch_size':32,'gradient_partitions':4,
              'partition_semantics':'original four rank patient shards, local padding and independent identical NumPy dropout streams',
              'dataset_sha256':sha,'time_unit':'hour','latent_dimension':28,'encoder_variant':model.variant,
              'encoder_direction':model.encoder_direction,'generator_anchor':model.generator_anchor,'shared_vector_field':True,
              'prior_weight':cfg['prior_weight'],'jacobian_weight':cfg['jacobian_weight'],'l2_weight':cfg['l2_weight'],
              'prior_semantics':cfg['prior_semantics'],'parameter_breakdown':{name:sum(p.numel() for p in child.parameters()) for name,child in model.named_children()},
              'age_covariate':'biological age/100 initializes state; separate from ICU hour clock'}
    for name,value in [('config',cfg),('hardware',hardware),('data',meta),('schema',schema)]:atomic(out/f'{name}.json',value)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr']);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:cfg['lr_decay']**(step/cfg['lr_decay_steps']))
    history=[];best=float('inf');bad=0;steps=0;first=1;elapsed_base=0.
    if (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        history=saved['history'];best=saved['best_score'];bad=saved['bad_epochs'];steps=saved['optimizer_steps'];first=saved['epoch']+1;elapsed_base=saved['elapsed_seconds']
    trainset=ICUContexts(splits['train'],training=True,seed=cfg['seed']);sampler=ExactBatchSampler(len(trainset),cfg['batch_size'],0,4,cfg['seed'])
    with SwanLabTracker(out,cfg,'mimic_history_encoder',model,cfg['seed']) as tracker:
        tracker.step=steps;started=time.monotonic()
        for epoch in range(first,cfg['max_epochs']+1):
            if epoch>cfg['min_epochs'] and bad>=cfg['patience']:break
            seed_all(cfg['seed']+epoch);rngs=[np.random.RandomState(cfg['seed']+epoch) for _ in range(4)]
            trainset.set_epoch(epoch);sampler.set_epoch(epoch);model.train();begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
            sums=torch.zeros(7,device='cuda');batches=0
            for indices in sampler.batches():
                partitions=partition_batches(trainset,indices,cfg['input_dropout'],rngs);denoms=denominators(partitions)
                optimizer.zero_grad(set_to_none=True);vector=torch.zeros(7,device='cuda')
                for partition,batch in enumerate(partitions):
                    result=model(batch,sample=True);loss,parts=local_loss(model,result,batch,schema,cfg,denoms)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite history-encoder loss')
                    loss.backward();vector[:5]+=parts
                    if steps==0 and partition==0:
                        hardware.update(actual_tensor_devices={k:str(v.device) for k,v in batch.items() if torch.is_tensor(v)},
                                        prediction_device=str(result['prediction'].device),jacobian_device=str(parts[4].device));atomic(out/'hardware.json',hardware)
                l2=model.ode_regularization();(cfg['l2_weight']*l2).backward();vector[5]=l2.detach()
                vector[6]=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                optimizer.step();scheduler.step();steps+=1;batches+=1;sums+=vector
                values=components(vector,cfg);tracker.batch(values['loss'],scheduler.get_last_lr()[0],**{k:v for k,v in values.items() if k!='loss'})
            validation=evaluate(model,splits['validation'],schema,cfg,meta['raw_train_std'],modes)
            score=validation['overall']['reconstruction_error'];values=components(sums/batches,cfg)
            row={'epoch':epoch,'optimizer_steps':steps,'training_loss':values['loss'],'training_components':values,'validation':validation,
                 'lr':scheduler.get_last_lr()[0],'epoch_seconds':time.monotonic()-begin,'elapsed_seconds':elapsed_base+time.monotonic()-started,
                 'peak_memory_bytes':torch.cuda.max_memory_allocated()};history.append(row)
            saved={'state_dict':model.state_dict(),'config':cfg,'schema':schema,'metadata':meta,'epoch':epoch,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()}
            if score<best:best=score;bad=0;save_checkpoint(out/'best.pt',saved)
            else:bad+=1
            save_checkpoint(out/'last.pt',{**saved,'history':history,'best_score':best,'bad_epochs':bad,'optimizer_steps':steps,'elapsed_seconds':row['elapsed_seconds']})
            atomic(out/'history.json',history);tracker.epoch(epoch,values,validation,row['lr'],epoch_seconds=row['epoch_seconds'],peak_memory_bytes=row['peak_memory_bytes'])
            print(json.dumps({'epoch':epoch,'training_loss':values['loss'],'validation_overall':score,'validation_forecast':validation['forecast']['reconstruction_error'],'epoch_seconds':row['epoch_seconds']}),flush=True)
        saved=torch.load(out/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['state_dict'])
        validation=evaluate(model,splits['validation'],schema,cfg,meta['raw_train_std'],modes)
        difference=abs(validation['overall']['reconstruction_error']-history[saved['epoch']-1]['validation']['overall']['reconstruction_error'])
        if difference>1e-6:raise AssertionError('Checkpoint reload changed the validation score')
        test=evaluate(model,splits['test'],schema,cfg,meta['raw_train_std'],modes)
        extra={str(h):evaluate(model,splits['test'],schema,cfg,meta['raw_train_std'],modes,horizon=h) for h in [6,12]}
        metrics={'status':'cuda_smoke_passed' if smoke else 'complete','best_epoch':saved['epoch'],'epochs_run':len(history),
                 'validation':validation,'test':test,'test_early_contexts':extra,'checkpoint_reload_difference':difference,
                 'elapsed_seconds':elapsed_base+time.monotonic()-started}
        tracker.final(metrics);atomic(out/'metrics.json',metrics);print(json.dumps({'complete':str(out),'status':metrics['status']}),flush=True)


if __name__=='__main__':main()
