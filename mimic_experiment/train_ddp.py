"""Four-GPU latent2x-both training with exact global masked loss reduction."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader,Subset

from paper_odeae.models import PaperModel,paper_model_from_config,feature_errors,prior_errors,reconstruction
from paper_odeae.jacobian import jacobian_regularization
from paper_odeae.train import device_batch,seed_all
from src.tracking import SwanLabTracker
from .data import ICUContexts,ExactBatchSampler,collate
from .prepare import atomic


class TrainingModel(PaperModel):
    def forward(self,batch,sample=False):
        out=super().forward(batch,sample)
        out['ode_l2']=self.ode_regularization()
        jac,_,count=jacobian_regularization(self,out,batch)
        out['jacobian_sum']=jac*count;out['jacobian_count']=jac.new_tensor(count)
        return out


def masked_loss(out,batch,schema,cfg,world):
    errors=feature_errors(out['prediction'],batch['values'],batch['mask'],schema)
    counts=batch['mask'].sum((0,1));dist.all_reduce(counts)
    means=errors.sum((0,1))/counts.clamp_min(1)
    numeric=torch.tensor([s['kind']=='numerical' for s in schema],device='cuda')
    mse,cce=means[numeric].sum(),means[~numeric].sum()
    pe,mask=prior_errors(out,batch,schema);pcounts=mask.sum((0,1));dist.all_reduce(pcounts)
    prior_means=pe.sum((0,1))/pcounts.clamp_min(1);pmse,pcce=prior_means[numeric].sum(),prior_means[~numeric].sum()
    jcount=out['jacobian_count'].detach().clone();dist.all_reduce(jcount)
    jac=out['jacobian_sum']/jcount.clamp_min(1)
    loss=world*(cfg['mse_weight']*mse+cfg['cce_weight']*cce+cfg['prior_weight']*(pmse+pcce)+cfg['jacobian_weight']*jac)+cfg['l2_weight']*out['ode_l2']
    metrics=torch.stack([mse.detach(),cce.detach(),pmse.detach(),pcce.detach(),jac.detach(),out['ode_l2'].detach()/world])
    dist.all_reduce(metrics)
    return loss,metrics


def objective(metrics,cfg):
    return cfg['mse_weight']*metrics[0]+cfg['cce_weight']*metrics[1]+cfg['prior_weight']*(metrics[2]+metrics[3])+cfg['jacobian_weight']*metrics[4]+cfg['l2_weight']*metrics[5]


def carry_forward(batch,schema,category_modes,numeric_defaults):
    # Predictions at every future query use only history; no target-mask feedback.
    values=batch['values'];b,n,f=values.shape;last=values.new_tensor(numeric_defaults)[None,:].expand(b,-1).clone()
    for j,s in enumerate(schema):
        if s['kind']=='categorical':last[:,j]=category_modes[j]
    for i in range(batch['max_context']):last=torch.where(batch['context_mask'][:,i].bool(),values[:,i],last)
    return last[:,None,:].expand(-1,n,-1)


@torch.no_grad()
def evaluate(core,sequences,schema,cfg,rank,world,raw_std,category_modes,horizon=24):
    core.eval();dataset=ICUContexts(sequences,horizon=horizon);subset=Subset(dataset,range(rank,len(dataset),world))
    loader=DataLoader(subset,batch_size=cfg['batch_size']//world,collate_fn=collate)
    keys=['interpolation','forecast','overall','prior','forecast_6h','forecast_12h','forecast_24h']
    totals=torch.zeros(len(keys),2,len(schema),device='cuda',dtype=torch.float64)
    absolute=torch.zeros(2,len(schema),device='cuda',dtype=torch.float64)
    naive_square=torch.zeros(len(schema),device='cuda',dtype=torch.float64);naive_abs=torch.zeros_like(naive_square)
    accuracy=torch.zeros(2,len(schema),device='cuda',dtype=torch.float64);naive_accuracy=torch.zeros_like(accuracy)
    jacobian=torch.zeros(2,device='cuda',dtype=torch.float64)
    for cpu in loader:
        batch=device_batch(cpu);out=core(batch,sample=False)
        errors=feature_errors(out['prediction'],batch['values'],batch['mask'],schema)
        index=torch.arange(batch['values'].shape[1],device='cuda')[None,:]
        anchor=batch['ages'].gather(1,(batch['cutoff']-1)[:,None]);future=(index>=batch['cutoff'][:,None])&batch['valid']
        regions=[index<batch['cutoff'][:,None],future,batch['valid'],None,
                 future&(batch['ages']<=horizon+6),future&(batch['ages']<=horizon+12),future&(batch['ages']<=horizon+24)]
        for i,region in enumerate(regions):
            if region is None:
                pe,mask=prior_errors(out,batch,schema);totals[i,0]+=pe.sum((0,1)).double();totals[i,1]+=mask.sum((0,1)).double()
            else:
                totals[i,0]+=(errors*region[...,None]).sum((0,1)).double();totals[i,1]+=(batch['mask']*region[...,None]).sum((0,1)).double()
        jacobian[0]+=out['jacobian_sum'].double();jacobian[1]+=out['jacobian_count'].double()
        naive=carry_forward(batch,schema,category_modes,cfg['normalized_train_means']);at=0;observed=batch['mask']*future[...,None]
        for j,s in enumerate(schema):
            width=s.get('classes',1)
            if s['kind']=='numerical':
                difference=out['prediction'][...,at]-batch['values'][...,j]
                absolute[0,j]+=(difference.abs()*observed[...,j]).sum().double();absolute[1,j]+=observed[...,j].sum().double()
                nd=naive[...,j]-batch['values'][...,j]
                naive_square[j]+=(nd.square()*observed[...,j]).sum().double();naive_abs[j]+=(nd.abs()*observed[...,j]).sum().double()
            else:
                guessed=out['prediction'][...,at:at+width].argmax(-1)
                accuracy[0,j]+=((guessed==batch['values'][...,j])*observed[...,j]).sum().double();accuracy[1,j]+=observed[...,j].sum().double()
                naive_accuracy[0,j]+=((naive[...,j]==batch['values'][...,j])*observed[...,j]).sum().double();naive_accuracy[1,j]+=observed[...,j].sum().double()
            at+=width
    for array in [totals,absolute,naive_square,naive_abs,accuracy,naive_accuracy,jacobian]:dist.all_reduce(array)
    numeric=torch.tensor([s['kind']=='numerical' for s in schema],device='cuda');result={}
    for key,total in zip(keys,totals):
        valid=total[1]>0;means=total[0]/total[1].clamp_min(1)
        mse=means[numeric&valid].sum();cce=means[(~numeric)&valid].sum()
        result[key]={'reconstruction_error':float(mse+cce),'sum_feature_mse':float(mse),'sum_feature_cce':float(cce),
                     'observed_scalars':int(total[1].sum()),'per_field':{s['name']:{'loss':float(means[j]),'observations':int(total[1,j])} for j,s in enumerate(schema) if valid[j]}}
    clinical={};naive={};standardized=[];baseline_standardized=[]
    for j,s in enumerate(schema):
        count=totals[1,1,j]
        if not count:continue
        if s['kind']=='numerical':
            scale=cfg['raw_scale'][j];sd=max(raw_std[j],1e-6)
            rmse=float((totals[1,0,j]/count).sqrt())*scale;nrmse=float((naive_square[j]/count).sqrt())*scale
            clinical[s['name']]={'unit':s['unit'],'observations':int(count),'mae':float(absolute[0,j]/count)*scale,'rmse':rmse,'standardized_rmse':rmse/sd}
            naive[s['name']]={'unit':s['unit'],'mae':float(naive_abs[j]/count)*scale,'rmse':nrmse,'standardized_rmse':nrmse/sd}
            standardized.append((rmse/sd)**2);baseline_standardized.append((nrmse/sd)**2)
        else:
            clinical[s['name']]={'observations':int(count),'accuracy':float(accuracy[0,j]/accuracy[1,j])}
            naive[s['name']]={'accuracy':float(naive_accuracy[0,j]/naive_accuracy[1,j])}
    result.update(patients=len(dataset),context_hours=horizon,dynamics={'jacobian_frobenius_squared':float(jacobian[0]/jacobian[1].clamp_min(1)),'states':int(jacobian[1])},
                  clinical_forecast=clinical,carry_forward=naive,mean_standardized_numeric_mse=sum(standardized)/len(standardized) if standardized else None,
                  carry_forward_mean_standardized_numeric_mse=sum(baseline_standardized)/len(baseline_standardized) if baseline_standardized else None)
    return result


def check_step(model,sequences,schema,cfg,rank,world,output):
    short=[{**s,'ages':s['ages'][:8],'values':s['values'][:8],'mask':s['mask'][:8],'dates':s['dates'][:8]} for s in sequences[:16]]
    items=[(s,min(4,len(s['ages'])-1)) for s in short]
    assert len(items)>=world
    reference=deepcopy(model.module) if rank==0 else None
    batch=device_batch(collate(items[rank::world]));optimizer=torch.optim.SGD(model.parameters(),lr=.001)
    out=model(batch);loss,metrics=masked_loss(out,batch,schema,cfg,world);loss.backward();optimizer.step()
    if rank==0:
        whole=device_batch(collate(items));ref_out=reference(whole);errors=feature_errors(ref_out['prediction'],whole['values'],whole['mask'],schema)
        mse,cce=reconstruction(errors,whole['mask'],schema);pe,pm=prior_errors(ref_out,whole,schema);pmse,pcce=reconstruction(pe,pm,schema)
        jac=ref_out['jacobian_sum']/ref_out['jacobian_count'];ref_loss=mse+cce+cfg['prior_weight']*(pmse+pcce)+cfg['jacobian_weight']*jac+cfg['l2_weight']*ref_out['ode_l2']
        opt=torch.optim.SGD(reference.parameters(),lr=.001);ref_loss.backward();opt.step()
        difference=max(float((a-b).abs().max()) for a,b in zip(model.module.parameters(),reference.parameters()))
        loss_difference=abs(float(ref_loss-objective(metrics,cfg)))
        result={'world_size':world,'physical_gpus':cfg['physical_gpus'],'reference_device':'cuda:0','single_gpu_loss':float(ref_loss),
                'distributed_loss':float(objective(metrics,cfg)),'loss_difference':loss_difference,'parameter_step_max_difference':difference,
                'global_batch_patients_checked':len(items),
                'weak_prior_weight':cfg['prior_weight'],'jacobian_weight':cfg['jacobian_weight'],'jacobian_exact_entries':28*28}
        atomic(output/'distributed_step_check.json',result)
        if difference>1e-5 or loss_difference>1e-4:raise RuntimeError('DDP global objective check failed')
    dist.barrier()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--rank',type=int,required=True);p.add_argument('--world-size',type=int,default=4);p.add_argument('--port',type=int,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--check-step',action='store_true')
    args=p.parse_args();rank,world=args.rank,args.world_size
    if Path(sys.prefix).name!='ode' or not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') not in ['0','1','2','3']:raise RuntimeError('Use ode environment and one explicit physical CUDA GPU; no CPU fallback')
    torch.cuda.set_device(0);torch.set_num_threads(1)
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{args.port}',rank=rank,world_size=world,device_id=torch.device('cuda:0'),timeout=timedelta(minutes=60))
    try:
        cfg=json.loads(Path(args.config).read_text());cfg={**cfg,'smoke':args.smoke,'world_size':world}
        dataset=Path(cfg['data']);payload=pickle.loads(dataset.read_bytes());schema=payload['schema'];cfg['raw_scale']=payload['metadata']['scale']
        cfg['normalized_train_means']=[(row['train_mean']-off)/scale if s['kind']=='numerical' else 0.
            for row,off,scale,s in zip(payload['metadata']['feature_profile'],payload['metadata']['offset'],payload['metadata']['scale'],schema)]
        splits={key:[s for s in payload['sequences'] if s['split']==key] for key in ['train','validation','test']}
        if args.smoke:
            cfg.update(max_epochs=1,min_epochs=0,patience=1,batch_size=32)
            # Keep real 48-hour timing and a 24-hour cutoff in end-to-end checks.
            splits={key:values[:32] for key,values in splits.items()}
        seed_all(cfg['seed']);output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=True)
        core=paper_model_from_config(schema,cfg,TrainingModel).cuda();actual=sum(p.numel() for p in core.parameters())
        if actual!=cfg['target_parameters']:raise ValueError('Parameter count mismatch')
        model=DistributedDataParallel(core,device_ids=[0],find_unused_parameters=False)
        workers=[None]*world;dist.all_gather_object(workers,{'rank':rank,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'device':str(next(core.parameters()).device),'gpu':torch.cuda.get_device_name()})
        cfg['physical_gpus']=[w['physical_gpu'] for w in workers]
        if args.check_step:check_step(model,splits['train'],schema,cfg,rank,world,output);return
        if not args.smoke and ((output/'history.json').exists() or (output/'metrics.json').exists()):raise RuntimeError('Preserve existing run')
        if rank==0:
            atomic(output/'config.json',cfg);atomic(output/'schema.json',schema);atomic(output/'data.json',payload['metadata'])
            atomic(output/'hardware.json',{'physical_gpu':','.join(cfg['physical_gpus']),'workers':workers,'world_size':world,'global_batch_size':cfg['batch_size'],
                   'per_gpu_batch_size':cfg['batch_size']//world,'parameters':actual,'trainable_parameters':actual,'parameter_device':'cuda:0',
                   'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),'torch_version':torch.__version__,'time_unit':'hour',
                   'model_source_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'paper_odeae/models.py').read_bytes()).hexdigest(),
                   'init_input':'anchor-derived admission age proxy + elapsed hours/8766, divided by 100','prior_weight':.01,'jacobian_weight':.01,
                   'jacobian_definition':'sum all 784 squared derivatives, mean valid generated states; coordinates detached',
                   'loss_reduction':'global field counts and global valid-state count; DDP mean-gradient compensation'})
        category_modes=[]
        for j,s in enumerate(schema):
            if s['kind']=='categorical':
                values=torch.tensor([v for seq in splits['train'] for v,m in zip(seq['values'][:,j],seq['mask'][:,j]) if m],device='cuda').long()
                category_modes.append(int(torch.bincount(values,minlength=s['classes']).argmax()))
            else:category_modes.append(0)
        tracker_context=SwanLabTracker(output,cfg,'mimic_latent2x_both',core,cfg['seed']) if rank==0 else nullcontext(None)
        with tracker_context as tracker:
            optimizer=torch.optim.Adam(core.parameters(),lr=cfg['lr']);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:cfg['lr_decay']**(step/cfg['lr_decay_steps']))
            trainset=ICUContexts(splits['train'],training=True,seed=cfg['seed']);sampler=ExactBatchSampler(len(trainset),cfg['batch_size'],rank,world,cfg['seed'])
            history=[];best=float('inf');bad=0;steps=0;started=time.monotonic()
            for epoch in range(1,cfg['max_epochs']+1):
                seed_all(cfg['seed']+epoch+100000*rank);trainset.set_epoch(epoch);sampler.set_epoch(epoch);model.train();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
                loader=DataLoader(trainset,batch_sampler=sampler,collate_fn=partial(collate,dropout=cfg['input_dropout']))
                summed=torch.zeros(6,device='cuda');batches=0
                for cpu in loader:
                    batch=device_batch(cpu);optimizer.zero_grad(set_to_none=True);out=model(batch,sample=False)
                    if steps==0:atomic(output/f'rank_{rank}_devices.json',{'rank':rank,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'tensors':{k:str(v.device) for k,v in batch.items() if torch.is_tensor(v)},'prediction':str(out['prediction'].device),'jacobian':str(out['jacobian_sum'].device)})
                    loss,metrics=masked_loss(out,batch,schema,cfg,world)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite MIMIC loss')
                    loss.backward();grad=torch.nn.utils.clip_grad_norm_(core.parameters(),10.,error_if_nonfinite=True);optimizer.step();scheduler.step();steps+=1;batches+=1;summed+=metrics
                    if tracker:tracker.batch(objective(metrics,cfg),scheduler.get_last_lr()[0],mse=metrics[0],cce=metrics[1],prior_mse=metrics[2],prior_cce=metrics[3],weighted_prior=.01*(metrics[2]+metrics[3]),jacobian=metrics[4],weighted_jacobian=.01*metrics[4],ode_l2=metrics[5],gradient_norm=grad)
                val=evaluate(core,splits['validation'],schema,cfg,rank,world,payload['metadata']['raw_train_std'],category_modes)
                score=val['overall']['reconstruction_error'];averages=summed/batches
                components={k:float(v) for k,v in zip(['mse','cce','prior_mse','prior_cce','jacobian_frobenius_squared','ode_l2'],averages)}
                components.update(weighted_prior=.01*(components['prior_mse']+components['prior_cce']),weighted_jacobian=.01*components['jacobian_frobenius_squared'],weighted_ode_l2=cfg['l2_weight']*components['ode_l2'])
                row={'epoch':epoch,'optimizer_steps':steps,'training_loss':float(objective(averages,cfg)),'training_components':components,'validation':val,
                     'lr':scheduler.get_last_lr()[0],'epoch_seconds':time.monotonic()-start,'elapsed_seconds':time.monotonic()-started,'peak_memory_bytes':torch.cuda.max_memory_allocated()};history.append(row)
                if score<best:
                    best=score;bad=0
                    if rank==0:torch.save({'state_dict':core.state_dict(),'config':cfg,'schema':schema,'metadata':payload['metadata'],'epoch':epoch,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()},output/'best.pt')
                else:bad+=1
                if tracker:
                    tracker.epoch(epoch,components,val,row['lr'],epoch_seconds=row['epoch_seconds'],peak_memory_bytes=row['peak_memory_bytes']);atomic(output/'history.json',history)
                    print(json.dumps({'epoch':epoch,'loss':row['training_loss'],'validation_overall':score,'validation_forecast':val['forecast']['reconstruction_error'],'epoch_seconds':row['epoch_seconds']}),flush=True)
                if epoch>=cfg['min_epochs'] and bad>=cfg['patience']:break
            dist.barrier();saved=torch.load(output/'best.pt',map_location='cuda',weights_only=False);core.load_state_dict(saved['state_dict'])
            val=evaluate(core,splits['validation'],schema,cfg,rank,world,payload['metadata']['raw_train_std'],category_modes)
            discrepancy=abs(val['overall']['reconstruction_error']-history[saved['epoch']-1]['validation']['overall']['reconstruction_error'])
            if discrepancy>1e-6:raise RuntimeError('Checkpoint reload changed validation metric')
            test=evaluate(core,splits['test'],schema,cfg,rank,world,payload['metadata']['raw_train_std'],category_modes)
            extras={str(hour):evaluate(core,splits['test'],schema,cfg,rank,world,payload['metadata']['raw_train_std'],category_modes,horizon=hour) for hour in [6,12]}
            if tracker:
                result={'status':'cuda_smoke_passed' if args.smoke else 'complete','best_epoch':saved['epoch'],'epochs_run':len(history),'validation':val,'test':test,
                        'test_early_contexts':extras,'checkpoint_reload_difference':discrepancy,'elapsed_seconds':time.monotonic()-started}
                tracker.final(result);atomic(output/'metrics.json',result);print(json.dumps({'complete':str(output),'best_epoch':saved['epoch']}),flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
