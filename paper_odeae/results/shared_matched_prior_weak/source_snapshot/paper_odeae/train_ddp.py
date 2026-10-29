"""CUDA DDP training with globally masked field means and fixed global batch.

Each worker sees one physical GPU. Field counts are reduced before computing
the local loss; multiplying reconstruction terms by world size compensates
for DDP's gradient average. Regularization and per-patient KL remain unscaled.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from src.tracking import SwanLabTracker
from .data import Prefixes, collate
from .models import PaperModel, feature_errors, paper_model_from_config, prior_errors, reconstruction
from .train import device_batch, seed_all


class TrainingPaperModel(PaperModel):
    def forward(self, batch, sample=False):
        output=super().forward(batch,sample)
        # Keep regularized parameters in DDP's forward graph even when all
        # local histories have one visit and the backward ODE is not called.
        output['ode_l2']=self.ode_regularization()
        return output


def save_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def masked_loss(output, batch, schema, model, cfg, world):
    errors=feature_errors(output['prediction'],batch['values'],batch['mask'],schema)
    counts=batch['mask'].sum((0,1))
    dist.all_reduce(counts)
    means=errors.sum((0,1))/counts.clamp_min(1)
    numerical=torch.tensor([f['kind']=='numerical' for f in schema],device='cuda')
    mse,cce=means[numerical].sum(),means[~numerical].sum()
    regularization=output['ode_l2']
    pmse,pcce=mse.new_zeros(()),mse.new_zeros(())
    if 'prior_prediction' in output:
        pe,mask=prior_errors(output,batch,schema);counts=mask.sum((0,1));dist.all_reduce(counts)
        means=pe.sum((0,1))/counts.clamp_min(1)
        pmse,pcce=means[numerical].sum(),means[~numerical].sum()
    loss=world*(cfg['mse_weight']*mse+cfg['cce_weight']*cce+cfg.get('prior_weight',0.)*(pmse+pcce))+cfg['kl_weight']*output['kl']+cfg['l2_weight']*regularization
    metrics=torch.stack([mse.detach(),cce.detach(),output['kl'].detach()/world,regularization.detach()/world,pmse.detach(),pcce.detach()])
    dist.all_reduce(metrics)
    return loss, metrics


@torch.no_grad()
def evaluate(model, sequences, schema, batch_size, rank, world):
    model.eval()
    prefixes=Prefixes(sequences)
    # No padding or repeated examples in evaluation shards.
    subset=Subset(prefixes,range(rank,len(prefixes),world))
    loader=DataLoader(subset,batch_size=batch_size,collate_fn=collate)
    collect=getattr(model,'collect_prior_predictions',False)
    totals=torch.zeros(4 if collect else 3,2,len(schema),device='cuda',dtype=torch.float64)
    for cpu in loader:
        batch=device_batch(cpu);out=model(batch,sample=False)
        errors=feature_errors(out['prediction'],batch['values'],batch['mask'],schema)
        index=torch.arange(batch['values'].shape[1],device='cuda')[None,:]
        regions=[index<batch['cutoff'][:,None],index>=batch['cutoff'][:,None],batch['valid']]
        for i,region in enumerate(regions):
            totals[i,0]+=(errors*region[...,None]).sum((0,1)).double()
            totals[i,1]+=(batch['mask']*region[...,None]).sum((0,1)).double()
        if collect:
            pe,mask=prior_errors(out,batch,schema)
            totals[3,0]+=pe.sum((0,1)).double();totals[3,1]+=mask.sum((0,1)).double()
    dist.all_reduce(totals)
    numerical=torch.tensor([f['kind']=='numerical' for f in schema],device='cuda')
    results={}
    for region,total in zip(['interpolation','forecast','overall','prior'],totals):
        counts=total[1];means=total[0]/counts.clamp_min(1);valid=counts>0
        mse,cce=means[numerical&valid].sum(),means[(~numerical)&valid].sum()
        results[region]={'reconstruction_error':float(mse+cce),'sum_feature_mse':float(mse),
            'sum_feature_cce':float(cce),'observed_fields':int(valid.sum()),'observed_scalars':int(counts.sum()),
            'per_field':{f['name']:{'loss':float(means[j]),'observations':int(counts[j])}
                         for j,f in enumerate(schema) if counts[j]>0}}
    return results


def step_check(model, sequences, schema, cfg, rank, world, output):
    """Compare one distributed update with a single-GPU global-batch update."""
    tiny=[{**s,'ages':s['ages'][:4],'values':s['values'][:4],
           'mask':s['mask'][:4],'dates':s['dates'][:4]} for s in sequences[:32]]
    dataset=Prefixes(tiny);items=[dataset[i] for i in range(32)]
    reference=deepcopy(model.module) if rank==0 else None
    batch=device_batch(collate(items[rank::world]))
    optimizer=torch.optim.SGD(model.parameters(),lr=.001)
    out=model(batch,sample=False)
    loss,metrics=masked_loss(out,batch,schema,model.module,cfg,world)
    loss.backward();optimizer.step()
    if rank==0:
        global_batch=device_batch(collate(items))
        optimizer_ref=torch.optim.SGD(reference.parameters(),lr=.001)
        predicted=reference(global_batch,sample=False)
        errors=feature_errors(predicted['prediction'],global_batch['values'],global_batch['mask'],schema)
        means=errors.sum((0,1))/global_batch['mask'].sum((0,1)).clamp_min(1)
        numerical=torch.tensor([f['kind']=='numerical' for f in schema],device='cuda')
        loss_ref=cfg['mse_weight']*means[numerical].sum()+cfg['cce_weight']*means[~numerical].sum()+cfg['kl_weight']*predicted['kl']+cfg['l2_weight']*reference.ode_regularization()
        if 'prior_prediction' in predicted:
            pe,mask=prior_errors(predicted,global_batch,schema)
            pmse,pcce=reconstruction(pe,mask,schema)
            loss_ref=loss_ref+cfg.get('prior_weight',0.)*(pmse+pcce)
        loss_ref.backward();optimizer_ref.step()
        discrepancy=max(float((a-b).abs().max()) for a,b in zip(model.module.parameters(),reference.parameters()))
        global_loss=cfg['mse_weight']*metrics[0]+cfg['cce_weight']*metrics[1]+cfg['kl_weight']*metrics[2]+cfg['l2_weight']*metrics[3]+cfg.get('prior_weight',0.)*(metrics[4]+metrics[5])
        result={'single_gpu_device':str(next(reference.parameters()).device),'world_size':world,
            'global_batch_size':32,'single_gpu_loss':float(loss_ref),'distributed_loss':float(global_loss),
            'loss_difference':abs(float(loss_ref-global_loss)),'max_parameter_difference_after_sgd':discrepancy}
        save_json(output/'distributed_step_check.json',result)
        if discrepancy>1e-5 or result['loss_difference']>1e-4:
            raise RuntimeError(f'distributed step differs from global masked objective: {result}')
    dist.barrier()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--rank',type=int,required=True)
    parser.add_argument('--world-size',type=int,required=True);parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--check-step',action='store_true')
    args=parser.parse_args();rank,world=args.rank,args.world_size
    if not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') not in ['0','1','2','3']:
        raise RuntimeError('one explicit physical CUDA GPU per DDP worker is required')
    torch.cuda.set_device(0);torch.set_num_threads(1)
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{args.port}',rank=rank,world_size=world,device_id=torch.device('cuda:0'))
    try:
        cfg=json.loads(Path(args.config).read_text());cfg={**cfg,'smoke':args.smoke,'distributed_world_size':world}
        if cfg['batch_size']%world:raise ValueError('global batch must be divisible by world size')
        if cfg['model'] not in ['deterministic','ode_ae']:raise ValueError('only ODE-AE capacity comparisons supported')
        seed_all(cfg['seed'])
        path=Path(cfg['data']);payload=pickle.loads(path.read_bytes());schema=payload['schema']
        if cfg['scope']!=payload['metadata']['scope'] or cfg['normalization']!=payload['metadata']['normalization']:
            raise ValueError('dataset scope/normalization differs')
        splits={k:[s for s in payload['sequences'] if s['split']==k] for k in ['train','validation','test']}
        if args.smoke:
            cfg.update(max_epochs=1,min_epochs=0,patience=1)
            splits={k:[{**s,'ages':s['ages'][:4],'values':s['values'][:4],
                        'mask':s['mask'][:4],'dates':s['dates'][:4]} for s in v[:32]] for k,v in splits.items()}
        output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=True)
        if not args.smoke and ((output/'metrics.json').exists() or (output/'history.json').exists()):
            raise ValueError('preserve previous run; choose a new output directory')
        core=paper_model_from_config(schema,cfg,TrainingPaperModel).cuda()
        actual=sum(p.numel() for p in core.parameters() if p.requires_grad)
        if actual!=cfg['target_parameters']:raise ValueError(f'parameter target mismatch: {actual}')
        model=DistributedDataParallel(core,device_ids=[0],find_unused_parameters=True)
        hardware_local={'rank':rank,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],
            'parameter_device':str(next(core.parameters()).device),'gpu_name':torch.cuda.get_device_name()}
        hardware_workers=[None]*world;dist.all_gather_object(hardware_workers,hardware_local)
        cfg['physical_gpus']=[worker['physical_gpu'] for worker in hardware_workers]
        if args.check_step:
            step_check(model,splits['train'],schema,cfg,rank,world,output);return
        context=SwanLabTracker(output,cfg,'odeae_capacity_match',core,cfg['seed']) if rank==0 else nullcontext(None)
        with context as tracker:
            if rank==0:
                save_json(output/'config.json',cfg);save_json(output/'schema.json',schema)
                save_json(output/'data.json',payload['metadata'])
                save_json(output/'hardware.json',{'physical_gpu':','.join(w['physical_gpu'] for w in hardware_workers),
                    'parameter_device':'cuda:0','parameters':actual,'trainable_parameters':actual,
                    'world_size':world,'global_batch_size':cfg['batch_size'],'per_gpu_batch_size':cfg['batch_size']//world,
                    'workers':hardware_workers,'gpu_name':torch.cuda.get_device_name(),'torch_version':torch.__version__,
                    'encoder_direction':cfg.get('encoder_direction','backward'),
                    'encoder_initialization':core.encoder_initialization,
                    'generator_anchor':'history cutoff' if cfg.get('encoder_direction')=='forward' else 'first observation',
                    'shared_vector_field':core.shared_vector_field,'prior_weight':cfg.get('prior_weight',0.),
                    'collect_prior_predictions':core.collect_prior_predictions,
                    'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'loss_reduction':'global field counts; world-size-scaled local reconstruction; DDP mean gradients'})
            optimizer=torch.optim.Adam(core.parameters(),lr=cfg['lr'])
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:cfg['lr_decay']**(step/cfg['lr_decay_steps']))
            dataset=Prefixes(splits['train'])
            if args.smoke:
                dataset=Subset(dataset,range(len(dataset)//cfg['batch_size']*cfg['batch_size']))
            sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=cfg['seed'])
            if len(dataset)%cfg['batch_size']:
                raise ValueError('this exact-global-batch comparison requires no repeated training examples')
            best=float('inf');bad=0;history=[];steps=0;start=time.monotonic()
            for epoch in range(1,cfg['max_epochs']+1):
                seed_all(cfg['seed']+epoch+100000*rank);sampler.set_epoch(epoch);model.train()
                loader=DataLoader(dataset,batch_size=cfg['batch_size']//world,sampler=sampler,
                                  collate_fn=partial(collate,dropout=cfg['input_dropout']))
                total=0.;batches=0;component_total=torch.zeros(6,device='cuda');epoch_start=time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                for cpu in loader:
                    batch=device_batch(cpu);optimizer.zero_grad(set_to_none=True)
                    out=model(batch,sample=True)
                    if steps==0:
                        save_json(output/f'rank_{rank}_devices.json',{'rank':rank,
                            'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],
                            'tensors':{key:str(batch[key].device) for key in ['values','ages','mask','context_mask']},
                            'prediction':str(out['prediction'].device),'parameter_device':str(next(core.parameters()).device)})
                    loss,metrics=masked_loss(out,batch,schema,core,cfg,world)
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite distributed loss')
                    loss.backward();grad_norm=torch.nn.utils.clip_grad_norm_(core.parameters(),10.,error_if_nonfinite=True)
                    optimizer.step();scheduler.step();steps+=1;batches+=1
                    global_loss=cfg['mse_weight']*metrics[0]+cfg['cce_weight']*metrics[1]+cfg['kl_weight']*metrics[2]+cfg['l2_weight']*metrics[3]+cfg.get('prior_weight',0.)*(metrics[4]+metrics[5])
                    component_total+=metrics
                    total+=float(global_loss)
                    if tracker:
                        tracker.batch(global_loss,scheduler.get_last_lr()[0],mse=metrics[0],cce=metrics[1],kl=metrics[2],l2=metrics[3],prior_mse=metrics[4],prior_cce=metrics[5],weighted_prior=cfg.get('prior_weight',0.)*(metrics[4]+metrics[5]),gradient_norm=grad_norm)
                val=evaluate(core,splits['validation'],schema,cfg['batch_size']//world,rank,world)
                score=val['overall']['reconstruction_error']
                row={'epoch':epoch,'optimizer_steps':steps,'training_loss':total/batches,'validation':val,
                     'lr':scheduler.get_last_lr()[0],'elapsed_seconds':time.monotonic()-start}
                row['training_components']={k:float(v/batches) for k,v in zip(['mse','cce','kl','ode_l2','prior_mse','prior_cce'],component_total)}
                row['training_components']['weighted_prior']=cfg.get('prior_weight',0.)*(row['training_components']['prior_mse']+row['training_components']['prior_cce'])
                row['epoch_seconds']=time.monotonic()-epoch_start
                row['peak_memory_bytes']=torch.cuda.max_memory_allocated()
                history.append(row)
                if score<best:
                    best=score;bad=0
                    if rank==0:torch.save({'state_dict':core.state_dict(),'config':cfg,'schema':schema,'metadata':payload['metadata'],
                        'epoch':epoch,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()},output/'best.pt')
                else:bad+=1
                if rank==0:
                    tracker.epoch(epoch,{'loss':total/batches,**row['training_components']},val,scheduler.get_last_lr()[0],elapsed_seconds=row['elapsed_seconds'])
                    save_json(output/'history.json',history)
                    print(json.dumps({'epoch':epoch,'train_loss':total/batches,'validation_overall':score,'elapsed_seconds':row['elapsed_seconds']}),flush=True)
                if epoch>=cfg['min_epochs'] and bad>=cfg['patience']:break
            dist.barrier()
            checkpoint=torch.load(output/'best.pt',map_location='cuda',weights_only=False)
            core.load_state_dict(checkpoint['state_dict'])
            val=evaluate(core,splits['validation'],schema,cfg['batch_size']//world,rank,world)
            discrepancy=abs(val['overall']['reconstruction_error']-history[checkpoint['epoch']-1]['validation']['overall']['reconstruction_error'])
            if discrepancy>1e-6:raise RuntimeError('distributed checkpoint reload changed validation score')
            test=evaluate(core,splits['test'],schema,cfg['batch_size']//world,rank,world)
            if rank==0:
                result={'status':'cuda_smoke_passed' if args.smoke else 'capacity_matched_ddp_completed','scope':cfg['scope'],
                    'best_epoch':checkpoint['epoch'],'epochs_run':len(history),'validation':val,'test':test,
                    'checkpoint_reload_difference':discrepancy,'elapsed_seconds':time.monotonic()-start,'published_numbers_reproduced':False}
                tracker.final(result);save_json(output/'metrics.json',result)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
