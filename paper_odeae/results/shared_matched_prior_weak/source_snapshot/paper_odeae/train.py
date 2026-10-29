"""Train the paper reconstruction, deterministic ablation, or LSTM on CUDA."""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.tracking import SwanLabTracker
from .data import Prefixes,collate
from .models import PaperModel,feature_errors,reconstruction,paper_model_from_config,prior_errors


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def device_batch(batch):
    return {k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}


@torch.no_grad()
def evaluate(model,sequences,schema,batch_size):
    model.eval();width=len(schema)
    totals={k:torch.zeros(2,width,device='cuda',dtype=torch.float64) for k in ['interpolation','forecast','overall']}
    loader=DataLoader(Prefixes(sequences),batch_size=batch_size,shuffle=False,collate_fn=collate)
    for cpu in loader:
        b=device_batch(cpu);out=model(b,sample=False)
        errors=feature_errors(out['prediction'],b['values'],b['mask'],schema)
        index=torch.arange(b['values'].shape[1],device='cuda')[None,:]
        regions={'interpolation':index<b['cutoff'][:,None],
                 'forecast':index>=b['cutoff'][:,None],
                 'overall':b['valid']}
        for key,region in regions.items():
            selected=region[...,None]*b['mask']
            totals[key][0]+=(errors*region[...,None]).sum((0,1)).double()
            totals[key][1]+=selected.sum((0,1)).double()
    results={}
    numeric=torch.tensor([f['kind']=='numerical' for f in schema],device='cuda')
    for key,total in totals.items():
        count=total[1];means=total[0]/count.clamp_min(1);valid=count>0
        mse=means[numeric&valid].sum();cce=means[(~numeric)&valid].sum()
        results[key]={'reconstruction_error':float(mse+cce),'sum_feature_mse':float(mse),
            'sum_feature_cce':float(cce),'observed_fields':int(valid.sum()),
            'observed_scalars':int(count.sum()),'per_field':{
                f['name']:{'loss':float(means[j]),'observations':int(count[j])}
                for j,f in enumerate(schema) if count[j]>0}}
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='paper_odeae/configs/local_available.json')
    p.add_argument('--model',choices=['ode_ae','deterministic','lstm','latent_ode','ode_rnn'])
    p.add_argument('--seed',type=int);p.add_argument('--output');p.add_argument('--smoke',action='store_true')
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text())
    for k in ['model','seed','output']:
        if getattr(args,k) is not None:cfg[k]=getattr(args,k)
    cfg['smoke']=args.smoke
    if not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') not in ['0','1','2','3']:
        raise RuntimeError('assign one physical CUDA GPU 0/1/2/3; CPU fallback is disabled')
    torch.set_num_threads(1);seed_all(cfg['seed'])
    path=Path(cfg['data']);payload=pickle.loads(path.read_bytes())
    schema=payload['schema']
    if cfg['scope'] != payload['metadata']['scope']:
        raise ValueError('configuration scope does not match prepared data')
    if cfg['normalization'] != payload['metadata']['normalization']:
        raise ValueError('configuration normalization does not match prepared data')
    if cfg['max_epochs']<1 or not 0<=cfg['min_epochs']<=cfg['max_epochs']:
        raise ValueError('invalid epoch budget')
    if cfg['scope']=='paper46' and len(schema)!=46:
        raise ValueError('paper46 requires all 46 fields to have training observations')
    splits={k:[s for s in payload['sequences'] if s['split']==k] for k in ['train','validation','test']}
    if cfg['smoke']:
        cfg.update(max_epochs=1,min_epochs=0,patience=1,batch_size=4)
        for key,n in [('train',8),('validation',4),('test',4)]:
            splits[key]=[{**s,'ages':s['ages'][:4],'values':s['values'][:4],
                          'mask':s['mask'][:4],'dates':s['dates'][:4]} for s in splits[key][:n]]
    if any(not seqs for seqs in splits.values()):raise ValueError('each patient split must be nonempty')
    output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=True)
    if cfg['model'] in ['latent_ode','ode_rnn']:
        from .baselines import BaselineModel
        model=BaselineModel(schema,cfg['model'],cfg['latent'],cfg['encoder_hidden'],cfg['ode_width'],cfg['rtol'],cfg['atol']).cuda()
    else:
        model=paper_model_from_config(schema,cfg).cuda()
    if 'target_parameters' in cfg and sum(p.numel() for p in model.parameters() if p.requires_grad)!=cfg['target_parameters']:
        raise ValueError('configured target parameter count does not match model')
    with SwanLabTracker(output, cfg, 'odeae_reproduction', model, cfg['seed']) as tracker:
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:cfg['lr_decay']**(step/cfg['lr_decay_steps']))
        hardware={'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'gpu_name':torch.cuda.get_device_name(),
                  'parameter_device':str(next(model.parameters()).device),'parameters':sum(p.numel() for p in model.parameters()),
                  'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'torch_version':torch.__version__,'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        if cfg.get('study') in ['forward_age_scaling','shared_matched_scaling']:
            import inspect
            hardware['model_source_sha256']=hashlib.sha256(Path(inspect.getfile(PaperModel)).read_bytes()).hexdigest()
            hardware['shared_vector_field']=getattr(model,'shared_vector_field',False)
            hardware['parameter_breakdown']={}
            seen=set()
            for name,module in model.named_children():
                unique=[p for p in module.parameters() if id(p) not in seen]
                hardware['parameter_breakdown'][name]=sum(p.numel() for p in unique)
                seen.update(id(p) for p in unique)
            hardware['encoder_initialization']=model.encoder_initialization
            hardware['generator_anchor']='history cutoff'
        if cfg['model'] in ['latent_ode','ode_rnn']:
            hardware.update(network_source='src.model via paper_odeae.baselines',
                time_unit='decade, relative to first observation',
                posterior_anchor='history cutoff' if cfg['model']=='latent_ode' else 'causal recurrent state',
                interpolation_semantics='cutoff-conditioned reconstruction' if cfg['model']=='latent_ode' else 'causal prediction before consuming current observation')
        (output/'config.json').write_text(json.dumps(cfg,indent=2))
        (output/'data.json').write_text(json.dumps(payload['metadata'],indent=2))
        (output/'schema.json').write_text(json.dumps(schema,indent=2))
        best=float('inf');bad=0;history=[];start=time.monotonic();steps=0
        for epoch in range(1,cfg['max_epochs']+1):
            seed_all(cfg['seed']+epoch);model.train();total=0.;batches=0
            components=torch.zeros(5,device='cuda');epoch_start=time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            generator=torch.Generator().manual_seed(cfg['seed']+epoch)
            loader=DataLoader(Prefixes(splits['train']),batch_size=cfg['batch_size'],shuffle=True,generator=generator,
                              collate_fn=partial(collate,dropout=cfg['input_dropout']))
            for cpu in loader:
                b=device_batch(cpu);optimizer.zero_grad(set_to_none=True)
                out=model(b,sample=True)
                errors=feature_errors(out['prediction'],b['values'],b['mask'],schema)
                mse,cce=reconstruction(errors,b['mask'],schema)
                l2=model.ode_regularization()
                loss=cfg['mse_weight']*mse+cfg['cce_weight']*cce+cfg['kl_weight']*out['kl']+cfg['l2_weight']*l2
                if cfg.get('prior_weight',0.):
                    pe,pmask=prior_errors(out,b,schema)
                    pmse,pcce=reconstruction(pe,pmask,schema)
                    loss=loss+cfg['prior_weight']*(pmse+pcce)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite paper loss')
                loss.backward();grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                optimizer.step();scheduler.step();steps+=1;total+=float(loss.detach());batches+=1
                components+=torch.stack([mse.detach(),cce.detach(),out['kl'].detach(),l2.detach(),grad_norm.detach()])
                tracker.batch(loss, scheduler.get_last_lr()[0], mse=mse, cce=cce, kl=out['kl'], l2=l2, gradient_norm=grad_norm)
                hardware['actual_tensor_devices']={k:str(b[k].device) for k in ['values','ages','mask','context_mask']}
                hardware['prediction_device']=str(out['prediction'].device)
            val=evaluate(model,splits['validation'],schema,cfg['batch_size'])
            score=val['overall']['reconstruction_error']
            row={'epoch':epoch,'optimizer_steps':steps,'training_loss':total/batches,'lr':scheduler.get_last_lr()[0],
                 'validation':val,'elapsed_seconds':time.monotonic()-start}
            row['training_components']={k:float(v/batches) for k,v in zip(['mse','cce','kl','ode_l2','gradient_norm'],components)}
            row['training_components']['weighted_ode_l2']=row['training_components']['ode_l2']*cfg['l2_weight']
            row['epoch_seconds']=time.monotonic()-epoch_start
            row['peak_memory_bytes']=torch.cuda.max_memory_allocated()
            tracker.epoch(epoch, {'loss': total/batches,**row['training_components']}, val, scheduler.get_last_lr()[0], elapsed_seconds=row['elapsed_seconds'],epoch_seconds=row['epoch_seconds'],peak_memory_bytes=row['peak_memory_bytes'])
            history.append(row);print(json.dumps({'epoch':epoch,'train_loss':total/batches,'validation_overall':score}),flush=True)
            if score<best:
                best=score;bad=0
                torch.save({'state_dict':model.state_dict(),'config':cfg,'schema':schema,'metadata':payload['metadata'],
                            'epoch':epoch,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()},output/'best.pt')
            else:bad+=1
            (output/'history.json').write_text(json.dumps(history,indent=2))
            (output/'hardware.json').write_text(json.dumps(hardware,indent=2))
            if epoch>=cfg['min_epochs'] and bad>=cfg['patience']:break
        checkpoint=torch.load(output/'best.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        val=evaluate(model,splits['validation'],schema,cfg['batch_size'])
        discrepancy=abs(val['overall']['reconstruction_error']-history[checkpoint['epoch']-1]['validation']['overall']['reconstruction_error'])
        if discrepancy>1e-6:raise RuntimeError('checkpoint reload changed validation score')
        results={'status':'cuda_smoke_passed' if cfg['smoke'] else 'local_or_paper_contract_run_completed',
                 'scope':cfg['scope'],'best_epoch':checkpoint['epoch'],'epochs_run':len(history),
                 'validation':val,'test':evaluate(model,splits['test'],schema,cfg['batch_size']),
                 'checkpoint_reload_difference':discrepancy,'elapsed_seconds':time.monotonic()-start,
                 'published_numbers_reproduced':False}
        tracker.final(results)
        (output/'metrics.json').write_text(json.dumps(results,indent=2))
        print(json.dumps({'complete':str(output),'status':results['status'],'best_epoch':results['best_epoch']}),flush=True)


if __name__=='__main__':main()
