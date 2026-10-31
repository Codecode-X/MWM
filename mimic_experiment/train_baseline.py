"""One baseline per physical GPU, effective batch 128 via four local gradient partitions."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch

from paper_odeae.models import feature_errors,paper_model_from_config
from paper_odeae.train import seed_all,device_batch
from src.tracking import SwanLabTracker
from .baseline_models import make_model
from .data import ICUContexts,ExactBatchSampler,collate
from .evaluation import evaluate
from .prepare import atomic


def save_checkpoint(path,data):
    temporary=path.with_suffix('.partial.pt');torch.save(data,temporary);temporary.replace(path)


def subset_batch(batch,indices):
    local={key:value[indices] if torch.is_tensor(value) else value for key,value in batch.items()}
    local['max_context']=int(local['cutoff'].max())
    return local


def normalization_and_modes(payload):
    meta=payload['metadata'];schema=payload['schema'];modes=[];means=[]
    for j,field in enumerate(schema):
        profile=meta['feature_profile'][j]
        means.append((profile['train_mean']-meta['offset'][j])/meta['scale'][j] if field['kind']=='numerical' else 0.)
        if field['kind']=='categorical':
            values=np.concatenate([s['values'][:,j][s['mask'][:,j].astype(bool)] for s in payload['sequences'] if s['split']=='train']).astype(np.int64)
            modes.append(int(np.bincount(values,minlength=field['classes']).argmax()))
        else:modes.append(0)
    return means,modes


def checks(model,sequences,schema,cfg,output):
    short=[{**s,**{key:s[key][:8] for key in ['ages','values','mask','dates']}} for s in sequences[:4]]
    items=[(s,min(4,len(s['ages'])-1)) for s in short];batch=device_batch(collate(items));model.eval()
    with torch.no_grad():original=model(batch,sample=False)
    changed={key:value.clone() if torch.is_tensor(value) else value for key,value in batch.items()}
    future=torch.arange(batch['values'].shape[1],device='cuda')[None]>=batch['cutoff'][:,None]
    for j,field in enumerate(schema):
        changed['values'][...,j]=torch.where(future,(changed['values'][...,j]+1)%field['classes'] if field['kind']=='categorical' else changed['values'][...,j]+100,changed['values'][...,j])
    changed['mask']=torch.where(future[...,None],torch.zeros_like(changed['mask']),changed['mask'])
    with torch.no_grad():other=model(changed,sample=False)
    torch.testing.assert_close(original['prediction'],other['prediction'],rtol=0,atol=0)
    for i,cutoff in enumerate(changed['cutoff'].tolist()):changed['ages'][i,cutoff:]+=100
    with torch.no_grad():shifted=model(changed,sample=False)
    torch.testing.assert_close(original['mu'],shifted['mu'],rtol=0,atol=0)
    model.train();out=model(batch,sample=True)
    if cfg['model']=='deterministic':
        torch.testing.assert_close(out['std'],torch.zeros_like(out['std']),rtol=0,atol=0)
        if float(out['kl'])!=0.:raise AssertionError('Deterministic model must have zero KL')
    errors=feature_errors(out['prediction'],batch['values'],batch['mask'],schema)
    counts=batch['mask'].sum((0,1));loss=(errors.sum((0,1))/counts.clamp_min(1)).sum()+cfg['kl_weight']*out['kl']+.02*model.ode_regularization()
    if not torch.isfinite(loss):raise FloatingPointError('Baseline check objective is nonfinite')
    loss.backward();used=0
    for parameter in model.parameters():
        if parameter.requires_grad:
            if parameter.grad is None:raise AssertionError('Trainable parameter has no gradient')
            if not torch.isfinite(parameter.grad).all():raise FloatingPointError('Nonfinite baseline gradient')
            used+=parameter.numel()
    model.zero_grad(set_to_none=True);model.eval()
    # Clock checks mutate only this checks-only model; formal runs initialize separately.
    with torch.no_grad():
        if cfg['model']=='healthflux':
            field=model.dynamics
            for parameter in field.parameters():parameter.zero_()
            field[-1].bias.fill_(float(np.arctanh(.2)))
            state=torch.zeros(2,model.h_dim,device='cuda');hour=torch.tensor([1.,3.],device='cuda');delta=torch.tensor([2.,5.],device='cuda')
            context=torch.zeros(2,model.context_dim,device='cuda');age=torch.tensor([50.,70.],device='cuda')
            after=model.evolve_hours(state,hour,delta,context,age)
            reverse=model.evolve_hours(after,hour+delta,-delta,context,age)
            torch.testing.assert_close(reverse,state,rtol=0,atol=2e-6)
        else:
            field=model.forward_ode if cfg['model'] in ['ode_ae','deterministic'] else model.core.ode
            for parameter in field.parameters():parameter.zero_()
            layers=field if cfg['model'] in ['ode_ae','deterministic'] else field.net;layers[-1].bias.fill_(.2)
            dimensions=cfg['latent'];state=torch.zeros(2,dimensions,device='cuda');delta=torch.tensor([2.,5.],device='cuda')
            after=model.solve(field,state,delta)
        torch.testing.assert_close(after,state+.2*delta[:,None],rtol=2e-5,atol=2e-6)
    report={'status':'passed','model':cfg['model'],'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'device':str(next(model.parameters()).device),
            'future_values_and_target_masks_change_predictions_max':0.,'future_times_change_encoded_state_max':0.,
            'finite_trainable_parameter_gradients':used,'ode_time_unit_verified':'hour','full_native_architecture':True,
            'test_objective':float(loss.detach()),'parameters':sum(p.numel() for p in model.parameters()),
            'source_family':cfg.get('source_family',cfg['model']),'parameter_matching_target':cfg.get('target_parameters')}
    atomic(output/'checks.json',report)


def verify_reference(payload,cfg,output,modes):
    root=Path(cfg['reference_output']);saved=torch.load(root/'seed_0/best.pt',map_location='cuda',weights_only=False)
    reference=paper_model_from_config(payload['schema'],saved['config']).cuda();reference.load_state_dict(saved['state_dict'])
    test=[s for s in payload['sequences'] if s['split']=='test']
    result=evaluate(reference,test,payload['schema'],cfg,payload['metadata']['raw_train_std'],modes)
    expected=json.loads((root/'seed_0/metrics.json').read_text())['test'];difference=0.
    for region in ['interpolation','forecast','overall','forecast_6h','forecast_12h','forecast_24h']:
        for key in ['reconstruction_error','sum_feature_mse','sum_feature_cce']:
            difference=max(difference,abs(result[region][key]-expected[region][key]))
        if result[region]['observed_scalars']!=expected[region]['observed_scalars']:raise AssertionError('Reference target count changed')
    for name,row in result['clinical_forecast'].items():
        for key in ['mae','rmse','accuracy']:
            if key in row:difference=max(difference,abs(row[key]-expected['clinical_forecast'][name][key]))
    if difference>1e-6:raise AssertionError(f'Common evaluator differs from reference: {difference}')
    atomic(output/'reference_evaluation_check.json',{'status':'passed','model':'latent2x both','physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],
           'device':'cuda:0','test_patients':len(test),'max_difference':difference,'checkpoint':str(root/'seed_0/best.pt')})
    del reference,saved;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--stage',choices=['checks','smoke','training'],default='training')
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text());stage=args.stage
    expected_env='healthflux' if cfg['model']=='healthflux' else 'ode'
    if Path(sys.prefix).name!=expected_env or not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(cfg['physical_gpu']):raise RuntimeError('Explicit assigned CUDA GPU and configured Conda environment required')
    torch.set_num_threads(1);torch.cuda.set_device(0);seed_all(cfg['seed'])
    output=Path(cfg['output']) if stage=='training' else Path(cfg['comparison_output'])/'checks'/cfg['variant']/stage
    output.mkdir(parents=True,exist_ok=True)
    dataset=Path(cfg['data']);digest=hashlib.sha256(dataset.read_bytes()).hexdigest()
    if digest!=cfg['dataset_sha256']:raise RuntimeError('Dataset hash changed')
    payload=pickle.loads(dataset.read_bytes());schema=payload['schema'];meta=payload['metadata']
    if meta['dataset_revision']!='chronology_v2':raise RuntimeError('Use audited chronology_v2 dataset')
    means,modes=normalization_and_modes(payload);cfg.update(raw_scale=meta['scale'],normalized_train_means=means,smoke=stage=='smoke')
    splits={key:[s for s in payload['sequences'] if s['split']==key] for key in ['train','validation','test']}
    normalization=None
    if cfg['model']=='healthflux':
        from healthflux.model import input_normalization
        normalization=input_normalization(payload)
    if stage=='checks' and cfg.get('verify_reference'):verify_reference(payload,cfg,output,modes)
    model=make_model(schema,cfg,normalization).cuda()
    actual_parameters=sum(p.numel() for p in model.parameters())
    if cfg.get('target_parameters') is not None and actual_parameters!=cfg['target_parameters']:
        raise AssertionError(f"Parameter count {actual_parameters} differs from target {cfg['target_parameters']}")
    if stage=='checks':checks(model,splits['train'],schema,cfg,output);return
    if stage=='smoke':
        cfg.update(max_epochs=1,min_epochs=0,patience=1,batch_size=32)
        splits={key:values[:32 if key=='train' else 8] for key,values in splits.items()}
    if (output/'metrics.json').exists():raise RuntimeError('Completed run already exists')
    hardware={'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'gpu_name':torch.cuda.get_device_name(),'parameter_device':str(next(model.parameters()).device),
              'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
              'environment':Path(sys.prefix).name,'executable':sys.executable,'torch_version':torch.__version__,'world_size':1,
              'effective_batch_size':cfg['batch_size'],'max_micro_batch_size':32,'gradient_partitions':4,'time_unit':'hour',
              'dataset_sha256':digest,'clock_age_separation':True,'history_semantics':cfg['history_semantics'],'future_semantics':cfg['future_semantics'],
              'age_covariate':cfg['age_covariate'],
              'source_family':cfg.get('source_family',cfg['model']),'parameter_matching_target':cfg.get('target_parameters'),
              'encoder_direction':cfg.get('encoder_direction'),'encoder_initialization':cfg.get('encoder_initialization'),
              'shared_vector_field':getattr(model,'shared_vector_field',False),
              'kl_weight':cfg['kl_weight'],'parameter_breakdown':{name:sum(p.numel() for p in module.parameters()) for name,module in model.named_children()}}
    if cfg['model']=='healthflux':hardware['modality_groups']=cfg['modality_groups']
    for name,obj in [('config',cfg),('schema',schema),('data',meta),('hardware',hardware),('input_normalization',normalization)]:atomic(output/f'{name}.json',obj)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:cfg['lr_decay']**(step/cfg['lr_decay_steps']))
    history=[];best=float('inf');bad=0;steps=0;first_epoch=1;elapsed_base=0.
    if (output/'last.pt').exists():
        saved=torch.load(output/'last.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        history=saved['history'];best=saved['best_score'];bad=saved['bad_epochs'];steps=saved['optimizer_steps'];first_epoch=saved['epoch']+1;elapsed_base=saved['elapsed_seconds']
    elif (output/'history.json').exists():raise RuntimeError('History exists without a resume checkpoint')
    trainset=ICUContexts(splits['train'],training=True,seed=cfg['seed']);sampler=ExactBatchSampler(len(trainset),cfg['batch_size'],0,1,cfg['seed'])
    numeric=torch.tensor([f['kind']=='numerical' for f in schema],device='cuda')
    group_ids=[j for j,f in enumerate(schema) if f['name']!='sex']
    with SwanLabTracker(output,cfg,'mimic_baselines',model,cfg['seed']) as tracker:
        tracker.step=steps;started=time.monotonic()
        for epoch in range(first_epoch,cfg['max_epochs']+1):
            seed_all(cfg['seed']+epoch);trainset.set_epoch(epoch);sampler.set_epoch(epoch);model.train();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
            summed=torch.zeros(6,device='cuda');batches=0
            for indices in sampler:
                batch=device_batch(collate([trainset[i] for i in indices],cfg['input_dropout']));counts=batch['mask'].sum((0,1));size=len(indices)
                nodes=batch['context_mask'][...,group_ids].bool().any(-1).sum();optimizer.zero_grad(set_to_none=True)
                metrics=torch.zeros(6,device='cuda')
                for partition in range(4):
                    ids=torch.arange(partition,size,4,device='cuda')
                    if not len(ids):continue
                    local=subset_batch(batch,ids);out=model(local,sample=True)
                    errors=feature_errors(out['prediction'],local['values'],local['mask'],schema)
                    per_field=errors.sum((0,1))/counts.clamp_min(1);mse=per_field[numeric].sum();cce=per_field[~numeric].sum()
                    kl=out['kl_sum']/nodes.clamp_min(1) if cfg['model']=='healthflux' else out['kl']*len(ids)/size
                    loss=cfg['mse_weight']*mse+cfg['cce_weight']*cce+cfg['kl_weight']*kl
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite baseline loss')
                    loss.backward();metrics[:3]+=torch.stack([mse.detach(),cce.detach(),kl.detach()])
                    if steps==0 and partition==0:
                        hardware.update(actual_tensor_devices={k:str(v.device) for k,v in local.items() if torch.is_tensor(v)},prediction_device=str(out['prediction'].device),kl_device=str(out['kl'].device))
                        atomic(output/'hardware.json',hardware)
                l2=model.ode_regularization();(cfg['l2_weight']*l2).backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step();scheduler.step();steps+=1;batches+=1
                metrics[3]=l2.detach();metrics[4]=cfg['mse_weight']*metrics[0]+cfg['cce_weight']*metrics[1]+cfg['kl_weight']*metrics[2]+cfg['l2_weight']*metrics[3];metrics[5]=norm.detach();summed+=metrics
                tracker.batch(metrics[4],scheduler.get_last_lr()[0],mse=metrics[0],cce=metrics[1],kl=metrics[2],weighted_kl=cfg['kl_weight']*metrics[2],ode_l2=metrics[3],weighted_ode_l2=cfg['l2_weight']*metrics[3],gradient_norm=norm)
            validation=evaluate(model,splits['validation'],schema,cfg,meta['raw_train_std'],modes);score=validation['overall']['reconstruction_error'];avg=summed/batches
            components={key:float(value) for key,value in zip(['mse','cce','kl','ode_l2','loss','gradient_norm'],avg)}
            components.update(weighted_kl=cfg['kl_weight']*components['kl'],weighted_ode_l2=cfg['l2_weight']*components['ode_l2'])
            row={'epoch':epoch,'optimizer_steps':steps,'training_loss':components['loss'],'training_components':components,'validation':validation,
                 'lr':scheduler.get_last_lr()[0],'epoch_seconds':time.monotonic()-start,'elapsed_seconds':elapsed_base+time.monotonic()-started,'peak_memory_bytes':torch.cuda.max_memory_allocated()};history.append(row)
            checkpoint={'state_dict':model.state_dict(),'config':cfg,'schema':schema,'metadata':meta,'input_normalization':normalization,'epoch':epoch,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()}
            if score<best:best=score;bad=0;save_checkpoint(output/'best.pt',checkpoint)
            else:bad+=1
            save_checkpoint(output/'last.pt',{**checkpoint,'history':history,'best_score':best,'bad_epochs':bad,'optimizer_steps':steps,'elapsed_seconds':row['elapsed_seconds']})
            atomic(output/'history.json',history);tracker.epoch(epoch,components,validation,row['lr'],epoch_seconds=row['epoch_seconds'],peak_memory_bytes=row['peak_memory_bytes'])
            print(json.dumps({'epoch':epoch,'train_loss':components['loss'],'validation_overall':score,'validation_forecast':validation['forecast']['reconstruction_error'],'epoch_seconds':row['epoch_seconds']}),flush=True)
            if epoch>=cfg['min_epochs'] and bad>=cfg['patience']:break
        saved=torch.load(output/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['state_dict'])
        validation=evaluate(model,splits['validation'],schema,cfg,meta['raw_train_std'],modes)
        difference=abs(validation['overall']['reconstruction_error']-history[saved['epoch']-1]['validation']['overall']['reconstruction_error'])
        if difference>1e-6:raise RuntimeError('Checkpoint reload changed validation score')
        test=evaluate(model,splits['test'],schema,cfg,meta['raw_train_std'],modes)
        extra={str(hour):evaluate(model,splits['test'],schema,cfg,meta['raw_train_std'],modes,horizon=hour) for hour in [6,12]}
        result={'status':'cuda_smoke_passed' if stage=='smoke' else 'complete','best_epoch':saved['epoch'],'epochs_run':len(history),'validation':validation,'test':test,
                'test_early_contexts':extra,'checkpoint_reload_difference':difference,'elapsed_seconds':elapsed_base+time.monotonic()-started}
        tracker.final(result);atomic(output/'metrics.json',result);print(json.dumps({'complete':str(output),'best_epoch':saved['epoch']}),flush=True)


if __name__=='__main__':main()
