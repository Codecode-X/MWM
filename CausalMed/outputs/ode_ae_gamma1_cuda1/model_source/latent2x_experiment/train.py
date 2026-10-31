"""Blind public-only training and sealed counterfactual submission on physical GPU 3."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from src.tracking import SwanLabTracker
from .data import load_public,cutoff,collate,cuda_batch,atomic,public_hashes
from .model import ControlledBoth,objective
from .rollout import predict,policy_action


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def save(path,value):
    temporary=path.with_suffix('.partial.pt');torch.save(value,temporary);temporary.replace(path)


@torch.no_grad()
def evaluate(model,sequences,metadata,batch_size=128):
    model.eval();totals=torch.zeros(3,2,3,device='cuda',dtype=torch.float64)
    absolute=torch.zeros(3,device='cuda',dtype=torch.float64);naive=torch.zeros_like(absolute)
    prior=torch.zeros(2,3,device='cuda',dtype=torch.float64)
    for begin in range(0,len(sequences),batch_size):
        chunk=sequences[begin:begin+batch_size];batch=cuda_batch(collate([(s,cutoff(s)) for s in chunk]))
        out=model(batch);errors=(out['prediction']-batch['values']).square()*batch['mask']
        i=torch.arange(batch['values'].shape[1],device='cuda')[None]
        past=i<batch['cutoff'][:,None];future=(~past)&batch['valid']
        for j,region in enumerate([past,future,batch['valid']]):
            totals[j,0]+=(errors*region[...,None]).sum((0,1)).double();totals[j,1]+=(batch['mask']*region[...,None]).sum((0,1)).double()
        fm=batch['mask']*future[...,None]
        absolute+=((out['prediction']-batch['values']).abs()*fm).sum((0,1)).double()
        last=batch['values'][torch.arange(len(chunk),device='cuda'),batch['cutoff']-1]
        naive+=((last[:,None]-batch['values']).square()*fm).sum((0,1)).double()
        if out.get('prior_prediction') is not None:
            n=out['prior_prediction'].shape[1];pi=torch.arange(n,device='cuda')[None]
            pm=batch['mask'][:,:n]*((pi>0)&(pi<batch['cutoff'][:,None]))[...,None]
            prior[0]+=((out['prior_prediction']-batch['values'][:,:n]).square()*pm).sum((0,1)).double();prior[1]+=pm.sum((0,1)).double()
    result={'patients':len(sequences),'evaluation':'recorded factual treatment path; no future observation updates'}
    for key,total in zip(['interpolation','forecast','overall'],totals):
        mse=total[0]/total[1].clamp_min(1)
        result[key]={'reconstruction_error':float(mse.sum()),'per_field_mse':mse.tolist(),'observations_per_field':total[1].long().tolist()}
    scale=torch.tensor(metadata['scale'],device='cuda',dtype=torch.float64)
    rmse=(totals[1,0]/totals[1,1].clamp_min(1)).sqrt()*scale
    mae=absolute/totals[1,1].clamp_min(1)*scale
    locf=(naive/totals[1,1].clamp_min(1)).sqrt()*scale
    result.update(future_raw_rmse=rmse.tolist(),future_raw_mae=mae.tolist(),carry_forward_future_raw_rmse=locf.tolist(),
                  prior_error=float((prior[0]/prior[1].clamp_min(1)).sum()) if bool(prior[1].sum()) else None)
    return result


def checks(model,sequences,specs,output):
    short=[{**s,**{key:s[key][:8] for key in ['times','values','raw_values','mask','actions']}} for s in sequences[:4]]
    batch=cuda_batch(collate([(s,3) for s in short]));model.eval()
    with torch.no_grad():original=model(batch)
    changed={key:value.clone() if torch.is_tensor(value) else value for key,value in batch.items()}
    changed['values'][:,3:]+=100;changed['mask'][:,3:]=0
    with torch.no_grad():other=model(changed)
    torch.testing.assert_close(other['prediction'],original['prediction'],rtol=0,atol=0)
    changed={**batch,'actions':batch['actions'].clone()};changed['actions'][:,2]=1-changed['actions'][:,2]
    with torch.no_grad():other=model(changed)
    torch.testing.assert_close(other['anchor'],original['anchor'],rtol=0,atol=0)
    changed={**batch,'actions':batch['actions'].clone()};changed['actions'][:,5:]=1-changed['actions'][:,5:]
    with torch.no_grad():other=model(changed)
    torch.testing.assert_close(other['prediction'][:,:6],original['prediction'][:,:6],rtol=0,atol=0)
    state=torch.randn(4,model.latent,device='cuda');action=torch.tensor([0,1,0,1],device='cuda')
    torch.testing.assert_close(model.change_action(state,action,action),state,rtol=0,atol=0)
    torch.testing.assert_close(model.change_action(state,action,1-action,disabled=True),state,rtol=0,atol=0)
    p=torch.tensor([0,1,2,2,3,3,4,4],device='cuda');t=torch.tensor([0,0,.999999999,1,0,0,0,0],device='cuda',dtype=torch.float64)
    risk=torch.tensor([0,0,0,0,specs['thresholds']['3'],specs['thresholds']['3']+1e-8,specs['thresholds']['4'],specs['thresholds']['4']+1e-8],device='cuda',dtype=torch.float64)
    torch.testing.assert_close(policy_action(p,t,risk,specs),torch.tensor([0,1,0,1,0,1,0,1],device='cuda'),rtol=0,atol=0)
    model.train();out=model(batch);loss,parts=objective(model,out,batch)
    if not torch.isfinite(loss):raise FloatingPointError('Check objective nonfinite')
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():raise AssertionError('Missing or nonfinite gradient')
    controlled=copy.deepcopy(model)
    with torch.no_grad():
        for parameter in controlled.forward_ode.parameters():parameter.zero_()
        controlled.forward_ode[-1].bias.fill_(.2)
        start=torch.zeros(2,model.latent,device='cuda');dt=torch.tensor([1.,3.],device='cuda')
        torch.testing.assert_close(controlled.solve(start,dt),start+dt[:,None]*.2,rtol=2e-5,atol=2e-6)
    atomic(output/'checks.json',{'status':'passed','physical_gpu':int(os.environ['CUDA_VISIBLE_DEVICES']),'device':'cuda:0','parameters':sum(p.numel() for p in model.parameters()),
           'future_outcomes_do_not_change_predictions':True,'fork_action_not_consumed_in_history':True,
           'actions_at_or_after_a_target_do_not_change_that_target':True,'unchanged_action_is_identity':True,
           'policy_delayed_and_strict_risk_thresholds_verified':True,'all_gradients_finite':True,'time_unit_verified':'year',
           'model_isolation':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--stage',choices=['checks','smoke','training'],default='training');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());output=Path('/workspace/output');stage=args.stage
    gpu=int(cfg.get('physical_gpu',3));is_odeae=cfg.get('model')=='ode_ae_action_update'
    if Path(sys.prefix).name!='ode' or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu) or not torch.cuda.is_available():raise RuntimeError('Use ode and the configured physical GPU; no CPU fallback')
    if importlib.util.find_spec('synthetic_cf_level3') is not None or Path('/home/xjh/projects/LatentODE/CausalMed').exists():raise RuntimeError('Generator/oracle filesystem must be absent from model namespace')
    torch.cuda.set_device(0);torch.set_num_threads(1);seed_all(cfg['seed'])
    output.mkdir(parents=True,exist_ok=True);payload,test_dataset=load_public('/workspace/public');metadata=payload['metadata']
    atomic(output/'isolation.json',{'generator_importable':False,'original_project_accessible':False,'public_root':'/workspace/public',
           'readable_dataset_files':public_hashes('/workspace/public'),'execution_namespace':'bubblewrap; read-only public data and frozen model source',
           'physical_gpu':gpu,'device':'cuda:0'})
    atomic(output/'data.json',metadata)
    if is_odeae:
        from .odeae_model import ControlledODEAE
        model=ControlledODEAE(cfg).cuda()
    else:model=ControlledBoth(cfg).cuda()
    parameters=sum(p.numel() for p in model.parameters())
    if parameters!=(1843 if is_odeae else 31314):raise AssertionError('Unexpected architecture parameter count')
    if stage=='checks':checks(model,payload['train'],test_dataset.policy_specs,output);return
    if stage=='smoke':
        cfg.update(max_epochs=1,min_epochs=0,patience=1)
        payload['train']=payload['train'][:64];payload['val']=payload['val'][:16];payload['test']=payload['test'][:8]
    hardware={'physical_gpu':gpu,'gpu_name':torch.cuda.get_device_name(),'parameter_device':str(next(model.parameters()).device),
              'parameters':parameters,'trainable_parameters':parameters,'latent_dimension':model.latent,'mlp_width':16 if is_odeae else 159,'ode_width':16 if is_odeae else 188,
              'environment':Path(sys.prefix).name,'torch_version':torch.__version__,'time_unit':'year','batch_size':cfg['batch_size'],
              'event_input_dimensions':{'observations':3,'observation_mask':3,'action_one_hot':2,'event_type':2},
              'prior_weight':0. if is_odeae else .01,'jacobian_weight':0. if is_odeae else .01,'kl_weight':1. if is_odeae else 0.,
              'l2_weight':.02,'jacobian_squared_entries':0 if is_odeae else 784,'normalization':'train_zscore',
              'encoder_direction':'backward' if is_odeae else 'forward','shared_vector_field':not is_odeae,
              'parameter_breakdown':{name:sum(p.numel() for p in module.parameters()) for name,module in model.named_children()}}
    atomic(output/'hardware.json',hardware);atomic(output/'config.json',cfg)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr']);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:cfg['lr_decay']**(step/cfg['lr_decay_steps']))
    history=[];best=float('inf');bad=0;steps=0;first=1;elapsed_base=0.
    if (output/'last.pt').exists() and not (output/'training_complete.json').exists():
        last=torch.load(output/'last.pt',map_location='cuda',weights_only=False);model.load_state_dict(last['state_dict']);optimizer.load_state_dict(last['optimizer']);scheduler.load_state_dict(last['scheduler'])
        history=last['history'];best=last['best_score'];bad=last['bad_epochs'];steps=last['optimizer_steps'];first=last['epoch']+1;elapsed_base=last['elapsed_seconds']
    trained=json.loads((output/'training_complete.json').read_text()) if (output/'training_complete.json').exists() else None
    if not trained:
        with SwanLabTracker(output,cfg,'causalmed_ode_ae' if is_odeae else 'causalmed_latent2x_both',model,cfg['seed']) as tracker:
            tracker.step=steps;started=time.monotonic()
            for epoch in range(first,cfg['max_epochs']+1):
                seed_all(cfg['seed']+epoch);model.train();begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
                order=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch])).permutation(len(payload['train']))
                summed={};loss_sum=0.;batches=0
                for lo in range(0,len(order),cfg['batch_size']):
                    examples=[payload['train'][int(j)] for j in order[lo:lo+cfg['batch_size']]]
                    batch=cuda_batch(collate([(s,cutoff(s,epoch,cfg['seed'])) for s in examples],cfg['input_dropout']))
                    optimizer.zero_grad(set_to_none=True);out=model(batch);loss,parts=objective(model,out,batch)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training objective')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step();scheduler.step();steps+=1;batches+=1
                    loss_sum+=float(loss.detach())
                    for key,value in parts.items():summed[key]=summed.get(key,0.)+float(value.detach())
                    tracker.batch(loss,scheduler.get_last_lr()[0],**parts,gradient_norm=norm)
                    if steps==1:
                        hardware.update(actual_tensor_devices={k:str(v.device) for k,v in batch.items() if torch.is_tensor(v)},prediction_device=str(out['prediction'].device),jacobian_device=str(parts['jacobian'].device));atomic(output/'hardware.json',hardware)
                validation=evaluate(model,payload['val'],metadata,cfg['batch_size']);score=validation['overall']['reconstruction_error']
                components={key:value/batches for key,value in summed.items()};components['loss']=loss_sum/batches
                row={'epoch':epoch,'optimizer_steps':steps,'training_loss':components['loss'],'training_components':components,'validation':validation,
                     'epoch_seconds':time.monotonic()-begin,'elapsed_seconds':elapsed_base+time.monotonic()-started,'lr':scheduler.get_last_lr()[0],
                     'peak_memory_bytes':torch.cuda.max_memory_allocated()};history.append(row)
                checkpoint={'state_dict':model.state_dict(),'config':cfg,'normalization':metadata,'epoch':epoch,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()}
                if score<best:best=score;bad=0;save(output/'best.pt',checkpoint)
                else:bad+=1
                save(output/'last.pt',{**checkpoint,'history':history,'best_score':best,'bad_epochs':bad,'optimizer_steps':steps,'elapsed_seconds':row['elapsed_seconds']})
                atomic(output/'history.json',history);tracker.epoch(epoch,components,validation,row['lr'],epoch_seconds=row['epoch_seconds'],peak_memory_bytes=row['peak_memory_bytes'])
                print(json.dumps({'epoch':epoch,'loss':components['loss'],'validation_overall':score,'validation_future_Y_rmse':validation['future_raw_rmse'][0],'epoch_seconds':row['epoch_seconds']}),flush=True)
                if epoch>=cfg['min_epochs'] and bad>=cfg['patience']:break
            best_checkpoint=torch.load(output/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(best_checkpoint['state_dict'])
            validation=evaluate(model,payload['val'],metadata,cfg['batch_size'])
            difference=abs(validation['overall']['reconstruction_error']-history[best_checkpoint['epoch']-1]['validation']['overall']['reconstruction_error'])
            if difference>1e-6:raise AssertionError('Checkpoint reload changed validation score')
            trained={'status':'training_complete','best_epoch':best_checkpoint['epoch'],'epochs_run':len(history),'validation':validation,
                     'checkpoint_reload_difference':difference,'elapsed_seconds':elapsed_base+time.monotonic()-started}
            tracker.final(trained);atomic(output/'training_complete.json',trained)
    else:
        best_checkpoint=torch.load(output/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(best_checkpoint['state_dict'])
    for parameter in model.parameters():parameter.requires_grad_(False)
    queries=test_dataset.queries();ids={s['patient_id'] for s in payload['test']};queries=queries[queries.patient_id.isin(ids)].copy()
    predict(model,payload['test'],queries,test_dataset.policy_specs,metadata,output/'with_actions',False,cfg['rollout_patient_batch'])
    predict(model,payload['test'],queries,test_dataset.policy_specs,metadata,output/'action_off',True,cfg['rollout_patient_batch'])
    # An action-blind future rollout must assign identical outcomes to all policies.
    import pandas as pd
    off=pd.read_parquet(output/'action_off/predictions.parquet')
    spread=off.groupby(['patient_id','relative_time']).Y_pred.agg(['min','max'])
    difference=float((spread['max']-spread['min']).abs().max())
    if difference>1e-6:raise AssertionError('Action-off trajectories should be policy invariant')
    atomic(output/'submission.json',{'status':'cuda_smoke_passed' if stage=='smoke' else 'predictions_sealed','best_epoch':trained['best_epoch'],
           'epochs_run':trained['epochs_run'],'physical_gpu':gpu,'device':'cuda:0','checkpoint_sha256':hashlib.sha256((output/'best.pt').read_bytes()).hexdigest(),
           'with_actions':json.loads((output/'with_actions/sealed.json').read_text()),'action_off':json.loads((output/'action_off/sealed.json').read_text()),
           'action_off_max_policy_outcome_difference':difference,'checkpoint_reload_difference':trained['checkpoint_reload_difference']})
    print(json.dumps({'complete':str(output),'status':'predictions_sealed','patients':len(payload['test'])}),flush=True)


if __name__=='__main__':main()
