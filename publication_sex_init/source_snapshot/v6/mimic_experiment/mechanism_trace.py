"""Explicit training replay ONLY: record every update without changing the objective."""
import argparse
import json
from pathlib import Path
import time

import torch

from mechanism_common import initialize,rng_state,restore_rng,rng_guard,parameter_hash,batch_audit,vector
from direction_gradient_common import component_gradients,flat_gradient,digest
from mimic_experiment.data import ICUContexts,ExactBatchSampler,collate
from mimic_experiment.prepare import atomic
from mimic_experiment.train_full_batch import make_model,loss_function,components
from mimic_experiment.train_baseline import save_checkpoint
from paper_odeae.train import device_batch,seed_all
from paper_odeae.solver import solver_statistics
from mechanism_common import solver_counts
from src.tracking import SwanLabTracker


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,required=True);args=p.parse_args()
    root=Path(args.out);manifest,cfg,payload,modes=initialize(root,args.gpu);folder=root/'experiment1';folder.mkdir(exist_ok=True)
    (folder/'snapshots').mkdir(exist_ok=True);(folder/'batches').mkdir(exist_ok=True);(folder/'steps').mkdir(exist_ok=True)
    model=make_model(payload['schema'],cfg).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:cfg['lr_decay']**(step/cfg['lr_decay_steps']))
    existing=json.loads((folder/'index.json').read_text()) if (folder/'index.json').exists() else {'steps':[],'producer_done':False}
    resume_path=existing['steps'][-1]['post'] if existing['steps'] else manifest['replay_start']
    saved=torch.load(resume_path,map_location='cuda',weights_only=False);model.load_state_dict(saved['state_dict'])
    optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);first=saved['epoch']+1
    resume_batch=0;resume_rng=None
    if existing['steps'] and saved['batch']<169:first=saved['epoch'];resume_batch=saved['batch'];resume_rng=saved['rng_state']
    del saved
    train=[s for s in payload['sequences'] if s['split']=='train'];dataset=ICUContexts(train,training=True,seed=cfg['seed'])
    sampler=ExactBatchSampler(len(dataset),128,0,1,cfg['seed']);original_steps=first*169-169
    prior=Path(manifest['prior_diagnostics']);previous={epoch:{r['batch']:r for r in json.loads((prior/'replay'/f'epoch_{epoch:03d}_batch_gradients.json').read_text())} for epoch in [45,46]}
    original_logs={}
    log_root=Path(manifest['original_swanlab'])/'logs'
    for name,number in [('loss',1),('gradient_norm',9)]:
        original_logs[name]={}
        for path in (log_root/str(number)).glob('*.log'):
            for line in path.read_text().splitlines():
                row=json.loads(line);original_logs[name][row['index']]=row['data']
    index=existing;index['producer_done']=False;atomic(folder/'index.json',index)
    hardware={'physical_gpu':args.gpu,'parameter_device':str(next(model.parameters()).device),'gpu_name':torch.cuda.get_device_name(),
              'mode':'training_replay_optimizer_steps_allowed','batch_size':128,'gradient_accumulation_steps':1,
              'solver':'frozen CUDA Tsit5 direct autograd; adaptive error controller detached','rtol':cfg['rtol'],'atol':cfg['atol']}
    atomic(folder/'hardware.json',hardware)
    tracker_cfg={**cfg,'output':str(folder),'physical_gpu':args.gpu,'study':'mechanism_seed_reconstructed_replay'}
    with SwanLabTracker(folder,tracker_cfg,'mechanism_seed_reconstructed_replay',model,0) as tracker:
        tracker.step=original_steps
        for epoch in range(first,47):
            if epoch==first and resume_rng is not None:restore_rng(resume_rng)
            else:seed_all(cfg['seed']+epoch)
            dataset.set_epoch(epoch);sampler.set_epoch(epoch);model.train();start=time.monotonic()
            for b,indices in enumerate(sampler,1):
                if epoch==first and b<=resume_batch:continue
                before_collate=rng_state();items=[dataset[i] for i in indices];batch=device_batch(collate(items,cfg['input_dropout']))
                after_collate=rng_state();optimizer.zero_grad(set_to_none=True)
                step_id=f'e{epoch:03d}_b{b:03d}';step_number=(epoch-1)*169+b
                if epoch>=44:
                    data_path=folder/'batches'/(step_id+'.pt')
                    if data_path.exists():raise RuntimeError('Preserve existing unindexed batch artifact before retrying')
                    save_checkpoint(data_path,{'batch':batch,'audit':batch_audit(batch,'real_training',indices),'epoch':epoch,'batch_index':b})
                    before_hash=parameter_hash(model);pre=folder/'snapshots'/(step_id+'_pre.pt')
                    if pre.exists() or (folder/'snapshots'/(step_id+'_post.pt')).exists():raise RuntimeError('Unindexed existing snapshot: preserve it and audit before resuming')
                    save_checkpoint(pre,{'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                                         'config':cfg,'schema':payload['schema'],'metadata':payload['metadata'],'epoch':epoch,'batch':b,'position':'before_update',
                                         'optimizer_step':step_number,'rng_before_collate':before_collate,'rng_after_collate':after_collate,
                                         'parameter_hash':before_hash,'batch_path':str(data_path),'dataset_indices':indices})
                with solver_statistics() as trace:
                    output=model(batch,sample=False);loss,parts=loss_function(model,output,batch,payload['schema'],cfg)
                before={};gh=gf=None
                if epoch>=44:before,gh,gf=component_gradients(model,output,batch,payload['schema'],cfg)
                loss.backward();parameters=tuple(model.parameters());field=tuple(model.forward_ode.parameters())
                field_before=vector(field);field_norm=float(field_before.norm())
                gfield=torch.cat([p.grad.detach().flatten() for p in field]);field_preclip=float(gfield.norm())
                all_preclip=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
                field_postclip=float(torch.cat([p.grad.detach().flatten() for p in field]).norm())
                all_postclip=float(torch.stack([p.grad.detach().square().sum() for p in parameters]).sum().sqrt())
                # The only parameter update in the entire mechanism study.
                optimizer.step();scheduler.step();tracker.step=step_number-1
                vals=components(torch.cat([parts,all_preclip.detach()[None]]),cfg)
                tracker.batch(vals['loss'],scheduler.get_last_lr()[0],**{k:v for k,v in vals.items() if k!='loss'})
                if epoch<44:continue
                native_differences={k:abs(before[k]-previous[epoch][b][k]) for k in ['history_loss_contribution','future_loss_contribution','history_gradient_norm','future_gradient_norm','history_future_cosine']} if epoch in previous else {}
                if epoch in previous:native_differences['all_parameter_preclip_norm']=abs(float(all_preclip)-previous[epoch][b]['all_parameter_preclip_norm'])
                logged={name:abs((vals['loss'] if name=='loss' else float(all_preclip))-rows[step_number]) for name,rows in original_logs.items() if step_number in rows}
                if max(native_differences.values(),default=0.)>1e-5 or (logged and max(logged.values())>1e-5):
                    atomic(folder/'mismatch.json',{'epoch':epoch,'batch':b,'prior_replay_differences':native_differences,'original_sampled_log_differences':logged})
                    raise RuntimeError('Per-batch replay mismatch; causal localization is not established')
                actual_peak_check=None
                if epoch==46 and b==121:
                    peak=torch.load(manifest['existing_abnormal_snapshot'],map_location='cuda',weights_only=False)
                    restored=torch.load(pre,map_location='cuda',weights_only=False)
                    d=max(float((restored['state_dict'][k]-v).abs().max()) for k,v in peak['state_dict'].items())
                    bd={k:bool(torch.equal(batch[k],v)) for k,v in peak['probe_batch'].items() if torch.is_tensor(v)}
                    actual_peak_check={'parameter_max_absolute_difference':d,'batch_tensors_equal':bd,'subject_ids_equal':batch['subject_ids']==peak['probe_batch']['subject_ids']}
                    atomic(folder/'abnormal_batch_reproduction.json',actual_peak_check)
                    if d!=0 or not all(bd.values()) or not actual_peak_check['subject_ids_equal']:raise RuntimeError('Known abnormal snapshot mismatch')
                    del peak,restored
                change=vector(field)-field_before;after_hash=parameter_hash(model)
                post=folder/'snapshots'/(step_id+'_post.pt')
                save_checkpoint(post,{'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                                      'config':cfg,'schema':payload['schema'],'metadata':payload['metadata'],'epoch':epoch,'batch':b,'position':'after_update',
                                      'optimizer_step':step_number,'rng_state':rng_state(),'parameter_hash':after_hash,'batch_path':str(data_path),'dataset_indices':indices})
                with rng_guard():
                    out_after=model(batch,sample=False);loss_after,_=loss_function(model,out_after,batch,payload['schema'],cfg)
                    after,_,_=component_gradients(model,out_after,batch,payload['schema'],cfg)
                    total_after=flat_gradient(loss_after,field)
                row={'id':step_id,'epoch':epoch,'batch':b,'optimizer_step':step_number,'actual_batch_size':len(indices),
                     'batch_kind':'real_training','patient_indices':indices,'subject_ids':[str(s['subject_id']) for s,c in items],
                     'batch_path':str(data_path),'before_snapshot':str(pre),'after_snapshot':str(post),
                     'before_parameter_hash':before_hash,'after_parameter_hash':after_hash,
                     'before':{**before,'total_loss':float(loss.detach()),'field_gradient_norm':field_preclip,
                               'all_parameter_preclip_norm':float(all_preclip),'all_parameter_postclip_norm':all_postclip,'field_postclip_norm':field_postclip},
                     'after':{**after,'total_loss':float(loss_after.detach()),'field_gradient_norm':float(total_after.norm())},
                     'field_update_norm':float(change.norm()),'field_parameter_norm_before':field_norm,'relative_field_update':float(change.norm())/max(field_norm,1e-30),
                     'adam_history_directional_derivative':float(torch.dot(gh,change)),'adam_future_directional_derivative':float(torch.dot(gf,change)),
                     'lr_after':scheduler.get_last_lr()[0],'training_solver_counts':solver_counts(trace),
                     'prior_replay_absolute_differences':native_differences,'original_sampled_log_absolute_differences':logged,
                     'exact_original_rng_checkpoint_available':False,'replay_type':'epoch_seed_reconstructed; per-batch verified against previous replay and sampled original logs'}
                atomic(folder/'steps'/(step_id+'.json'),row);index['steps'].append({'id':step_id,'epoch':epoch,'batch':b,'pre':str(pre),'post':str(post),'before_parameter_hash':before_hash,'after_parameter_hash':after_hash})
                atomic(folder/'index.json',index);atomic(folder/'status.json',{'status':'running','epoch':epoch,'batch':b,'verified_steps':len(index['steps'])})
                del output,out_after,total_after,loss,loss_after
                if b%10==0:print(json.dumps({'epoch':epoch,'batch':b,'norm':float(all_preclip),'cosine':before['history_future_cosine'],'replay_differences_max':max(native_differences.values(),default=0.)}),flush=True)
            atomic(folder/f'epoch_{epoch:03d}_completion.json',{'epoch':epoch,'seconds':time.monotonic()-start})
    index['producer_done']=True;atomic(folder/'index.json',index);atomic(folder/'status.json',{'status':'complete','verified_steps':len(index['steps'])})


if __name__=='__main__':main()
