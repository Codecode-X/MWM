"""Train the HealthFlux ADNI task adapter in the dedicated healthflux environment."""
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
from paper_odeae.data import Prefixes,collate
from paper_odeae.models import feature_errors,reconstruction
from paper_odeae.train import seed_all,device_batch,evaluate
from paper_odeae.scaling_study import atomic
from src.tracking import SwanLabTracker
from .model import HealthFlux,input_normalization


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text())
    if args.output:cfg['output']=args.output
    cfg['smoke']=args.smoke
    if Path(sys.prefix).name!='healthflux':raise RuntimeError('Use the healthflux Conda environment')
    if not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') not in ['0','1','2','3']:
        raise RuntimeError('Assign one physical CUDA GPU 0/1/2/3; no CPU fallback')
    torch.set_num_threads(1);seed_all(cfg['seed'])
    dataset=Path(cfg['data']);payload=pickle.loads(dataset.read_bytes());schema=payload['schema']
    assert cfg['normalization']==payload['metadata']['normalization'] and cfg['scope']==payload['metadata']['scope']
    splits={key:[s for s in payload['sequences'] if s['split']==key] for key in ['train','validation','test']}
    normalization=input_normalization(payload)
    if args.smoke:
        cfg.update(max_epochs=1,min_epochs=0,patience=1,batch_size=4)
        for key,n in [('train',8),('validation',4),('test',4)]:
            splits[key]=[{**s,**{name:s[name][:4] for name in ['ages','values','mask','dates']}} for s in splits[key][:n]]
    output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=True)
    if (output/'history.json').exists() or (output/'metrics.json').exists():raise RuntimeError('Preserve existing run; select a new output')
    model=HealthFlux(schema,cfg,normalization).cuda()
    hardware={'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'gpu_name':torch.cuda.get_device_name(),
              'parameter_device':str(next(model.parameters()).device),'parameters':sum(p.numel() for p in model.parameters()),
              'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
              'environment':Path(sys.prefix).name,'executable':sys.executable,'torch_version':torch.__version__,
              'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
              'model_source_sha256':hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest(),
              'parameter_breakdown':{name:sum(p.numel() for p in module.parameters()) for name,module in model.named_children()},
              'modality_embedding_parameters':model.modality_embedding.numel(),
              'latent_dimensions':{'deterministic':model.h_dim,'stochastic':model.z_dim},
              'modality_groups':{name:[schema[j]['name'] for j in ids] for name,ids in model.groups.items()},
              'solver':'original-paper fixed-step RK4, age/10 clock, max step 10 years, max 4 substeps',
              'future_updates':False,'history_reconstruction':'cutoff-conditioned backward integration; native prior mean at unobserved query',
              'implementation':'paper v3 Methods reimplementation; common ADNI task readout and objective'}
    for name,data in [('config',cfg),('schema',schema),('data',payload['metadata']),('input_normalization',normalization),('hardware',hardware)]:atomic(output/f'{name}.json',data)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:cfg['lr_decay']**(step/cfg['lr_decay_steps']))
    with SwanLabTracker(output,cfg,'healthflux_adni',model,cfg['seed']) as tracker:
        start=time.monotonic();history=[];best=float('inf');bad=0;steps=0
        for epoch in range(1,cfg['max_epochs']+1):
            seed_all(cfg['seed']+epoch);model.train();epoch_start=time.monotonic();torch.cuda.reset_peak_memory_stats()
            generator=torch.Generator().manual_seed(cfg['seed']+epoch)
            loader=DataLoader(Prefixes(splits['train']),batch_size=cfg['batch_size'],shuffle=True,generator=generator,
                              collate_fn=partial(collate,dropout=cfg['input_dropout']))
            sums=torch.zeros(6,device='cuda');batches=0
            for cpu in loader:
                b=device_batch(cpu);optimizer.zero_grad(set_to_none=True);result=model(b,sample=True)
                mse,cce=reconstruction(feature_errors(result['prediction'],b['values'],b['mask'],schema),b['mask'],schema)
                kl=result['kl'];l2=model.ode_regularization()
                loss=cfg['mse_weight']*mse+cfg['cce_weight']*cce+cfg['kl_weight']*kl+cfg['l2_weight']*l2
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite HealthFlux objective')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                optimizer.step();scheduler.step();steps+=1;batches+=1
                sums+=torch.stack([loss.detach(),mse.detach(),cce.detach(),kl.detach(),l2.detach(),norm.detach()])
                tracker.batch(loss,scheduler.get_last_lr()[0],mse=mse,cce=cce,kl=kl,weighted_kl=cfg['kl_weight']*kl,
                              ode_l2=l2,weighted_ode_l2=cfg['l2_weight']*l2,gradient_norm=norm)
                hardware.update(actual_tensor_devices={key:str(b[key].device) for key in ['values','ages','mask','context_mask']},
                                prediction_device=str(result['prediction'].device),kl_device=str(kl.device))
            validation=evaluate(model,splits['validation'],schema,cfg['batch_size']);score=validation['overall']['reconstruction_error']
            components={key:float(value/batches) for key,value in zip(['loss','mse','cce','kl','ode_l2','gradient_norm'],sums)}
            components.update(weighted_kl=components['kl']*cfg['kl_weight'],weighted_ode_l2=components['ode_l2']*cfg['l2_weight'])
            row={'epoch':epoch,'optimizer_steps':steps,'training_loss':components['loss'],'training_components':components,
                 'validation':validation,'lr':scheduler.get_last_lr()[0],'elapsed_seconds':time.monotonic()-start,
                 'epoch_seconds':time.monotonic()-epoch_start,'peak_memory_bytes':torch.cuda.max_memory_allocated()}
            history.append(row)
            tracker.epoch(epoch,components,validation,row['lr'],epoch_seconds=row['epoch_seconds'],peak_memory_bytes=row['peak_memory_bytes'])
            if score<best:
                best=score;bad=0
                torch.save({'state_dict':model.state_dict(),'config':cfg,'schema':schema,'input_normalization':normalization,
                            'epoch':epoch,'metadata':payload['metadata'],'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()},output/'best.pt')
            else:bad+=1
            atomic(output/'history.json',history);atomic(output/'hardware.json',hardware)
            print(json.dumps({'epoch':epoch,'train_loss':components['loss'],'validation_overall':score,'epoch_seconds':row['epoch_seconds']}),flush=True)
            if epoch>=cfg['min_epochs'] and bad>=cfg['patience']:break
        checkpoint=torch.load(output/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(checkpoint['state_dict'])
        val=evaluate(model,splits['validation'],schema,cfg['batch_size'])
        difference=abs(val['overall']['reconstruction_error']-history[checkpoint['epoch']-1]['validation']['overall']['reconstruction_error'])
        if difference>1e-6:raise RuntimeError('Checkpoint reload changed validation score')
        results={'status':'cuda_smoke_passed' if args.smoke else 'adni_common_protocol_complete','best_epoch':checkpoint['epoch'],
                 'epochs_run':len(history),'validation':val,'test':evaluate(model,splits['test'],schema,cfg['batch_size']),
                 'checkpoint_reload_difference':difference,'elapsed_seconds':time.monotonic()-start,
                 'paper_ukbiobank_metrics_reproduced':False,'task':'same cached ADNI 40-field mixed reconstruction/forecast task'}
        tracker.final(results);atomic(output/'metrics.json',results)
        print(json.dumps({'complete':str(output),'best_epoch':checkpoint['epoch'],'elapsed_seconds':results['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
