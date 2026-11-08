"""Resume the saved epoch-41 optimizer and recover instability checkpoints on CUDA."""
import argparse
import json
from pathlib import Path
import time

import torch

from direction_gradient_common import initialize,digest,publish,component_gradients
import sys
# This directory also contains profile.py; leave stdlib profile unshadowed.
sys.path=[p for p in sys.path if Path(p or '.').resolve()!=Path(__file__).resolve().parent]
from mimic_experiment.data import ICUContexts,ExactBatchSampler,collate
from mimic_experiment.evaluation import evaluate
from mimic_experiment.prepare import atomic
from mimic_experiment.train_baseline import save_checkpoint
from mimic_experiment.train_full_batch import make_model,loss_function,components
from paper_odeae.train import seed_all,device_batch
from src.tracking import SwanLabTracker


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--gpu',type=int,required=True);args=parser.parse_args()
    out=Path(args.out);manifest,cfg,payload,modes=initialize(out,args.gpu);replay=out/'replay';replay.mkdir(exist_ok=True)
    original=json.loads(Path(manifest['original_history']).read_text());reference={r['epoch']:r for r in original}
    seed_all(cfg['seed']);model=make_model(payload['schema'],cfg).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:cfg['lr_decay']**(step/cfg['lr_decay_steps']))
    resume=replay/'last.pt';resume=resume if resume.exists() else Path(manifest['initial_checkpoint'])
    saved=torch.load(resume,map_location='cuda',weights_only=False);model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
    first=saved['epoch']+1;history=saved.get('replay_history',[]);steps=reference[saved['epoch']]['optimizer_steps'];del saved
    if any(not r['matches_original_within_1e_5'] for r in history):raise RuntimeError('Cannot resume an unverified replay checkpoint')
    splits={k:[s for s in payload['sequences'] if s['split']==k] for k in ['train','validation']}
    dataset=ICUContexts(splits['train'],training=True,seed=cfg['seed']);sampler=ExactBatchSampler(len(dataset),128,0,1,cfg['seed'])
    selected={42,45,46,47,48,50,60,63,64,65};probed={42,45,46,47,48,50,63,64,65}
    field_parameters=tuple(model.forward_ode.parameters())
    tracker_cfg={**cfg,'output':str(replay),'physical_gpu':args.gpu,'study':'direction_instability_replay','resume_epoch':first-1}
    atomic(replay/'hardware.json',{'physical_gpu':args.gpu,'parameter_device':str(next(model.parameters()).device),'gpu_name':torch.cuda.get_device_name(),
                                 'training_batch_size':128,'gradient_accumulation_steps':1,'initial_checkpoint':str(resume),'source_snapshot':manifest['source_snapshot']})
    with SwanLabTracker(replay,tracker_cfg,'mimic_direction_instability_replay',model,cfg['seed']) as tracker:
        tracker.step=steps
        for epoch in range(first,66):
            begin=time.monotonic();seed_all(cfg['seed']+epoch);dataset.set_epoch(epoch);sampler.set_epoch(epoch);model.train()
            total=torch.zeros(7,device='cuda');batch_rows=[];peak=-1.;peak_info=None
            for batch_number,indices in enumerate(sampler,1):
                batch=device_batch(collate([dataset[i] for i in indices],cfg['input_dropout']));optimizer.zero_grad(set_to_none=True)
                output=model(batch,sample=False);loss,parts=loss_function(model,output,batch,payload['schema'],cfg)
                record=gh=gf=None
                if epoch in probed:
                    record,gh,gf=component_gradients(model,output,batch,payload['schema'],cfg)
                loss.backward()
                if record is not None:
                    field_total=torch.cat([p.grad.detach().flatten() for p in field_parameters]);before=torch.cat([p.detach().flatten() for p in field_parameters])
                    record.update(batch=batch_number,actual_batch_size=len(indices),optimizer_step=steps+1,shared_field_total_gradient_norm=float(field_total.norm()))
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                if epoch in probed and float(norm)>peak:
                    peak=float(norm);peak_path=out/'checkpoints'/f'epoch_{epoch:03d}_peak.pt'
                    # Serialization precedes the update: the exact batch and its gradients refer to this state.
                    peak_info={'state_dict':model.state_dict(),'config':cfg,'schema':payload['schema'],'metadata':payload['metadata'],'epoch':epoch,
                               'batch':batch_number,'optimizer_step':steps+1,'all_parameter_gradient_norm':peak,'probe_batch':batch,
                               'origin':'replayed pre-update peak-gradient batch'}
                    save_checkpoint(peak_path,peak_info)
                optimizer.step();scheduler.step();steps+=1;total+=torch.cat([parts,norm.detach()[None]])
                values=components(torch.cat([parts,norm.detach()[None]]),cfg)
                tracker.batch(values['loss'],scheduler.get_last_lr()[0],**{k:v for k,v in values.items() if k!='loss'},actual_batch_size=len(indices))
                if record is not None:
                    after=torch.cat([p.detach().flatten() for p in field_parameters]);change=after-before
                    record.update(all_parameter_preclip_norm=float(norm),field_update_norm=float(change.norm()),
                                  history_first_order_actual_adam_update=float(torch.dot(gh,change)),future_first_order_actual_adam_update=float(torch.dot(gf,change)))
                    batch_rows.append(record)
                if batch_number==1:atomic(replay/'actual_tensor_devices.json',{k:str(v.device) for k,v in batch.items() if torch.is_tensor(v)})
            validation=evaluate(model,splits['validation'],payload['schema'],cfg,payload['metadata']['raw_train_std'],modes)
            values=components(total/len(sampler),cfg);ref=reference[epoch]
            differences={r:abs(validation[r]['reconstruction_error']-ref['validation'][r]['reconstruction_error']) for r in ['overall','interpolation','forecast']}
            differences['training_loss']=abs(values['loss']-ref['training_loss'])
            differences['training_gradient_norm']=abs(values['gradient_norm']-ref['training_components']['gradient_norm'])
            agreement=max(differences.values())<1e-5
            row={'epoch':epoch,'optimizer_steps':steps,'validation':validation,'training_components':values,'reference_absolute_differences':differences,
                 'matches_original_within_1e_5':agreement,'epoch_seconds':time.monotonic()-begin}
            history.append(row);atomic(replay/'history.json',history);atomic(replay/f'epoch_{epoch:03d}_batch_gradients.json',batch_rows)
            tracker.epoch(epoch,values,validation,scheduler.get_last_lr()[0],epoch_seconds=row['epoch_seconds'])
            saved={'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                   'config':cfg,'schema':payload['schema'],'metadata':payload['metadata'],'epoch':epoch,'replay_history':history}
            if not agreement:
                save_checkpoint(replay/f'epoch_{epoch:03d}_mismatch.pt',saved)
                atomic(replay/'status.json',{'status':'reference_mismatch','epoch':epoch,'differences':differences})
                raise RuntimeError('Replay differs from original; do not label recovered states as original collapse snapshots')
            save_checkpoint(replay/'last.pt',saved)
            if epoch in selected:
                path=out/'checkpoints'/f'epoch_{epoch:03d}_replay.pt';save_checkpoint(path,saved)
                publish(out,{'id':f'epoch_{epoch:03d}_replay','epoch':epoch,'path':str(path),'sha256':digest(path),'origin':'optimizer replay verified against original history','reference_absolute_differences':differences})
            if peak_info is not None:
                publish(out,{'id':f'epoch_{epoch:03d}_peak','epoch':epoch,'batch':peak_info['batch'],'path':str(peak_path),'sha256':digest(peak_path),
                             'origin':'pre-update peak-gradient batch in verified replay','all_parameter_gradient_norm':peak})
            atomic(replay/'status.json',{'status':'running','epoch':epoch,'reference_absolute_differences':differences})
            print(json.dumps({'epoch':epoch,'seconds':row['epoch_seconds'],'matches_original':agreement,'differences':differences}),flush=True)
    atomic(replay/'status.json',{'status':'complete','last_epoch':65})


if __name__=='__main__':main()
