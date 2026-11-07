"""Paired, single-GPU continuation: only the history anchor gradient differs."""
import argparse
import json
from pathlib import Path
import time

import torch

from causal_history_model import make_causal_model
from direction_gradient_common import initialize,decompose,digest
from mechanism_common import rng_state,restore_rng,parameter_hash,vector
from mimic_experiment.data import ICUContexts,ExactBatchSampler,collate
from mimic_experiment.prepare import atomic
from mimic_experiment.train_full_batch import loss_function,components
from mimic_experiment.train_baseline import save_checkpoint
from mimic_experiment.evaluation import evaluate
from paper_odeae.train import seed_all,device_batch
from src.tracking import SwanLabTracker


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--branch',choices=['control','history_stop'],required=True);a=p.parse_args()
    root=Path(a.out);manifest,cfg,payload,modes=initialize(root,a.gpu);folder=root/a.branch;folder.mkdir(exist_ok=True)
    if (folder/'metrics.json').exists():return
    cfg={**cfg,'output':str(folder),'comparison_output':str(root),'physical_gpu':a.gpu,'max_epochs':81,
         'history_anchor_gradient':'stopped_only_for_history' if a.branch=='history_stop' else 'full',
         'study':'targeted_history_gradient_causal_continuation','intervention_start_epoch':46,'selection_range':[46,81],
         'fixed_epoch_budget':True,'early_stopping':False}
    seed_all(0);model=make_causal_model(payload['schema'],cfg,a.branch=='history_stop')
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr']);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:cfg['lr_decay']**(step/cfg['lr_decay_steps']))
    initial=Path(manifest['training_start']);resume=folder/'last.pt';resume=resume if resume.exists() else initial
    saved=torch.load(resume,map_location='cuda',weights_only=False);model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
    first=saved['epoch']+1;history=saved.get('causal_history',[]);best=saved.get('causal_best_score',float('inf'));steps=saved['epoch']*169
    restore_rng(saved['rng_state']);del saved
    hardware={'physical_gpu':a.gpu,'gpu_name':torch.cuda.get_device_name(),'parameter_device':str(next(model.parameters()).device),
              'parameters':sum(p.numel() for p in model.parameters()),'latent':28,'shared_field_identity':model.backward_ode is model.forward_ode,
              'batch_size':128,'gradient_accumulation_steps':1,'world_size':1,'time_unit':'hour','rtol':cfg['rtol'],'atol':cfg['atol'],
              'starting_checkpoint':str(initial),'starting_checkpoint_sha256':digest(initial),'starting_epoch':45,'resume_path':str(resume),
              'loaded_parameter_hash':parameter_hash(model),'branch':a.branch,'selection_epochs':[46,81],
              'rng_protocol':'complete RNG restored; native seed_all(seed+epoch) remains at every epoch'}
    for name,data in [('config',cfg),('hardware',hardware),('schema',payload['schema'])]:atomic(folder/(name+'.json'),data)
    splits={s:[x for x in payload['sequences'] if x['split']==s] for s in ['train','validation','test']}
    dataset=ICUContexts(splits['train'],training=True,seed=0);sampler=ExactBatchSampler(len(dataset),128,0,1,0)
    old=json.loads(Path(manifest['original_history']).read_text());original={r['epoch']:r for r in old}
    with SwanLabTracker(folder,cfg,'targeted_history_gradient_continuation',model,0) as tracker:
        tracker.step=steps
        for epoch in range(first,82):
            seed_all(epoch);dataset.set_epoch(epoch);sampler.set_epoch(epoch);model.train();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
            total=torch.zeros(7,device='cuda');loss_contributions=torch.zeros(2,device='cuda');batch_rows=[]
            for number,indices in enumerate(sampler,1):
                batch=device_batch(collate([dataset[i] for i in indices],cfg['input_dropout']));optimizer.zero_grad(set_to_none=True)
                if epoch==46 and number==121:
                    save_checkpoint(folder/'epoch046_batch121_pre.pt',{'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                                                                        'rng_state':rng_state(),'config':cfg,'epoch':epoch,'batch':number,'probe_batch':batch})
                    if a.branch=='control':
                        known=torch.load(manifest['abnormal_snapshot'],map_location='cuda',weights_only=False)
                        diff=max(float((model.state_dict()[k]-v).abs().max()) for k,v in known['state_dict'].items())
                        same={k:bool(torch.equal(v,batch[k])) for k,v in known['probe_batch'].items() if torch.is_tensor(v)}
                        atomic(folder/'abnormal_batch_replay_audit.json',{'maximum_parameter_difference':diff,'input_tensors_equal':same,'source':manifest['abnormal_snapshot']})
                        if diff>1e-5 or not all(same.values()):raise RuntimeError('Control no longer matches the known replayed abnormal batch')
                        del known
                output=model(batch,sample=False);loss,parts=loss_function(model,output,batch,payload['schema'],cfg)
                lh,lf=decompose(model,output,batch,payload['schema'],cfg);loss_contributions+=torch.stack([lh.detach(),lf.detach()])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite paired training loss')
                loss.backward();field=tuple(model.forward_ode.parameters());before=vector(field);field_gradient=float(torch.cat([p.grad.detach().flatten() for p in field]).norm())
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                optimizer.step();scheduler.step();steps+=1;total+=torch.cat([parts,norm.detach()[None]])
                change=vector(field)-before
                row={'epoch':epoch,'batch':number,'optimizer_step':steps,'patients':len(indices),'loss':float(loss.detach()),
                     'history_contribution':float(lh.detach()),'future_contribution':float(lf.detach()),'all_parameter_preclip_norm':float(norm),
                     'field_preclip_norm':field_gradient,'relative_field_update':float(change.norm()/before.norm().clamp_min(1e-30))}
                batch_rows.append(row);values=components(torch.cat([parts,norm.detach()[None]]),cfg)
                tracker.batch(values['loss'],scheduler.get_last_lr()[0],**{k:v for k,v in values.items() if k!='loss'},
                              history_contribution=row['history_contribution'],future_contribution=row['future_contribution'],relative_field_update=row['relative_field_update'],actual_batch_size=len(indices))
                if 'actual_tensor_devices' not in hardware:
                    hardware.update(actual_tensor_devices={k:str(v.device) for k,v in batch.items() if torch.is_tensor(v)},prediction_device=str(output['prediction'].device));atomic(folder/'hardware.json',hardware)
                del output,loss,lh,lf
            validation=evaluate(model,splits['validation'],payload['schema'],cfg,payload['metadata']['raw_train_std'],modes)
            values=components(total/len(sampler),cfg)
            row={'epoch':epoch,'optimizer_steps':steps,'training_batches':len(sampler),'training_components':values,'training_loss':values['loss'],
                 'training_history_contribution':float(loss_contributions[0]/len(sampler)),'training_future_contribution':float(loss_contributions[1]/len(sampler)),
                 'maximum_batch_gradient_norm':max(r['all_parameter_preclip_norm'] for r in batch_rows),
                 'batches_above_10':sum(r['all_parameter_preclip_norm']>10 for r in batch_rows),'batches_above_100':sum(r['all_parameter_preclip_norm']>100 for r in batch_rows),
                 'validation':validation,'lr':scheduler.get_last_lr()[0],'epoch_seconds':time.monotonic()-start,'peak_memory_bytes':torch.cuda.max_memory_allocated()}
            if a.branch=='control':
                ref=original[epoch];d={region:abs(validation[region]['reconstruction_error']-ref['validation'][region]['reconstruction_error']) for region in ['interpolation','forecast','overall']}
                d.update(training_loss=abs(values['loss']-ref['training_loss']),gradient_norm=abs(values['gradient_norm']-ref['training_components']['gradient_norm']))
                row['original_history_absolute_differences']=d
                if max(d.values())>1e-5:atomic(folder/'reference_mismatch.json',row);raise RuntimeError('Matched continuation control differs from original epoch history')
            history.append(row);score=validation['overall']['reconstruction_error']
            checkpoint={'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng_state':rng_state(),
                        'config':cfg,'schema':payload['schema'],'metadata':payload['metadata'],'epoch':epoch,'branch':a.branch}
            if score<best:best=score;save_checkpoint(folder/'best.pt',checkpoint)
            save_checkpoint(folder/'last.pt',{**checkpoint,'causal_history':history,'causal_best_score':best})
            atomic(folder/'history.json',history);atomic(folder/f'epoch_{epoch:03d}_batches.json',batch_rows)
            tracker.epoch(epoch,values,validation,row['lr'],maximum_batch_gradient_norm=row['maximum_batch_gradient_norm'])
            print(json.dumps({'branch':a.branch,'epoch':epoch,'validation_history':validation['interpolation']['reconstruction_error'],'validation_future':validation['forecast']['reconstruction_error'],
                              'validation_overall':score,'maximum_batch_gradient_norm':row['maximum_batch_gradient_norm'],'seconds':row['epoch_seconds']}),flush=True)
        best_saved=torch.load(folder/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(best_saved['state_dict'])
        validation=evaluate(model,splits['validation'],payload['schema'],cfg,payload['metadata']['raw_train_std'],modes)
        selected=next(h for h in history if h['epoch']==best_saved['epoch'])
        difference=abs(validation['overall']['reconstruction_error']-selected['validation']['overall']['reconstruction_error']);assert difference<1e-6
        test=evaluate(model,splits['test'],payload['schema'],cfg,payload['metadata']['raw_train_std'],modes)
        early={str(h):evaluate(model,splits['test'],payload['schema'],cfg,payload['metadata']['raw_train_std'],modes,horizon=h) for h in [6,12]}
        metrics={'status':'complete','branch':a.branch,'continued_from_epoch':45,'epochs_run':len(history),'global_last_epoch':81,'best_epoch':best_saved['epoch'],
                 'selection_range':[46,81],'validation':validation,'test':test,'test_early_contexts':early,'checkpoint_reload_difference':difference,
                 'maximum_training_batch_gradient':max(h['maximum_batch_gradient_norm'] for h in history),'training_batch_size':128,'gradient_accumulation_steps':1}
        atomic(folder/'metrics.json',metrics);tracker.final(metrics)


if __name__=='__main__':main()
