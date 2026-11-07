"""Seed-0 original loss with history-specific encoder recurrence depth 8."""
import argparse
from pathlib import Path
import time

import torch

from direction_gradient_common import initialize,digest
from mechanism_common import parameter_hash,rng_state,restore_rng
from history_depth_model import make_depth_model
from history_depth_checks import verify
from mimic_experiment.train_full_batch import make_model,loss_function,components
from mimic_experiment.data import ICUContexts,ExactBatchSampler,collate
from mimic_experiment.evaluation import evaluate
from mimic_experiment.prepare import atomic
from mimic_experiment.train_baseline import save_checkpoint
from paper_odeae.train import seed_all,device_batch
from src.tracking import SwanLabTracker


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    root=Path(a.out);manifest,cfg,payload,modes=initialize(root,a.gpu);folder=root/'history_depth_8';folder.mkdir(exist_ok=True)
    if (folder/'metrics.json').exists():return
    if not (folder/'checks.json').exists():verify(root,folder,manifest,cfg,payload,a.gpu)
    cfg={**cfg,'output':str(folder),'comparison_output':str(root),'physical_gpu':a.gpu,'history_anchor_gradient':'max_encoder_recurrence_depth_8','history_max_encoder_depth':8,
         'study':'history_recurrence_depth8_from_initialization','selection_range':'epoch1 through original early-stopping endpoint','from_scratch':True}
    cfg['max_epochs']=min(cfg['max_epochs'],150)
    cfg['min_epochs']=min(cfg['min_epochs'],cfg['max_epochs'])
    seed_all(0);native=make_model(payload['schema'],cfg).cuda();initial_native=parameter_hash(native);del native
    seed_all(0);model=make_depth_model(payload['schema'],cfg);initial_stop=parameter_hash(model)
    assert initial_native==initial_stop
    atomic(folder/'initialization_check.json',{'seed':0,'original_constructor_initial_parameter_hash':initial_native,'depth8_constructor_initial_parameter_hash':initial_stop,
                                            'identical':True,'physical_gpu':a.gpu,'parameter_device':str(next(model.parameters()).device),
                                            'source_model':'frozen original PaperModel constructor; no original epoch0 checkpoint was available'})
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr']);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:cfg['lr_decay']**(step/cfg['lr_decay_steps']))
    history=[];best=float('inf');bad=0;steps=0;first=1
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        history=saved['history'];best=saved['best_score'];bad=saved['bad_epochs'];steps=saved['optimizer_steps'];first=saved['epoch']+1;restore_rng(saved['rng_state'])
    else:save_checkpoint(folder/'initial.pt',{'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng_state':rng_state(),'config':cfg,'epoch':0})
    hardware={'physical_gpu':a.gpu,'gpu_name':torch.cuda.get_device_name(),'parameter_device':str(next(model.parameters()).device),
              'parameters':43280,'latent':28,'history_max_encoder_depth':8,'time_unit':'hour','solver':'Tsit5','training_batch_size':128,'gradient_accumulation_steps':1,'world_size':1,'initial_parameter_hash':initial_stop,
              'rtol':cfg['rtol'],'atol':cfg['atol'],'max_epochs':cfg['max_epochs'],'min_epochs':cfg['min_epochs'],'patience':cfg['patience']}
    atomic(folder/'config.json',cfg);atomic(folder/'hardware.json',hardware)
    splits={s:[x for x in payload['sequences'] if x['split']==s] for s in ['train','validation','test']}
    dataset=ICUContexts(splits['train'],training=True,seed=0);sampler=ExactBatchSampler(len(dataset),128,0,1,0)
    with SwanLabTracker(folder,cfg,'history_depth8_from_seed0',model,0) as tracker:
        tracker.step=steps
        for epoch in range(first,cfg['max_epochs']+1):
            if epoch>cfg['min_epochs'] and bad>=cfg['patience']:break
            seed_all(epoch);dataset.set_epoch(epoch);sampler.set_epoch(epoch);model.train();start=time.monotonic();total=torch.zeros(7,device='cuda');norms=[]
            for indices in sampler:
                batch=device_batch(collate([dataset[i] for i in indices],cfg['input_dropout']));optimizer.zero_grad(set_to_none=True)
                output=model(batch,sample=False);loss,parts=loss_function(model,output,batch,payload['schema'],cfg)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite history-depth8 ablation loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                optimizer.step();scheduler.step();steps+=1;norms.append(float(norm));total+=torch.cat([parts,norm.detach()[None]])
                values=components(torch.cat([parts,norm.detach()[None]]),cfg);tracker.batch(values['loss'],scheduler.get_last_lr()[0],**{k:v for k,v in values.items() if k!='loss'},actual_batch_size=len(indices))
                if 'actual_tensor_devices' not in hardware:
                    hardware.update(actual_tensor_devices={k:str(v.device) for k,v in batch.items() if torch.is_tensor(v)},prediction_device=str(output['prediction'].device));atomic(folder/'hardware.json',hardware)
                del output,loss
            validation=evaluate(model,splits['validation'],payload['schema'],cfg,payload['metadata']['raw_train_std'],modes);values=components(total/len(sampler),cfg)
            row={'epoch':epoch,'optimizer_steps':steps,'training_loss':values['loss'],'training_components':values,'validation':validation,
                 'maximum_batch_gradient_norm':max(norms),'batches_above_10':sum(x>10 for x in norms),'batches_above_100':sum(x>100 for x in norms),
                 'lr':scheduler.get_last_lr()[0],'epoch_seconds':time.monotonic()-start};history.append(row)
            saved={'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng_state':rng_state(),
                   'config':cfg,'schema':payload['schema'],'metadata':payload['metadata'],'epoch':epoch}
            if validation['overall']['reconstruction_error']<best:best=validation['overall']['reconstruction_error'];bad=0;save_checkpoint(folder/'best.pt',saved)
            else:bad+=1
            save_checkpoint(folder/'last.pt',{**saved,'history':history,'best_score':best,'bad_epochs':bad,'optimizer_steps':steps})
            atomic(folder/'history.json',history);tracker.epoch(epoch,values,validation,row['lr'],maximum_batch_gradient_norm=max(norms))
            print({'epoch':epoch,'validation_overall':validation['overall']['reconstruction_error'],'validation_history':validation['interpolation']['reconstruction_error'],
                   'validation_future':validation['forecast']['reconstruction_error'],'maximum_batch_gradient_norm':max(norms),'bad_epochs':bad},flush=True)
        saved=torch.load(folder/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['state_dict'])
        validation=evaluate(model,splits['validation'],payload['schema'],cfg,payload['metadata']['raw_train_std'],modes)
        ref=next(h for h in history if h['epoch']==saved['epoch']);difference=abs(validation['overall']['reconstruction_error']-ref['validation']['overall']['reconstruction_error']);assert difference<1e-6
        test=evaluate(model,splits['test'],payload['schema'],cfg,payload['metadata']['raw_train_std'],modes)
        early={str(h):evaluate(model,splits['test'],payload['schema'],cfg,payload['metadata']['raw_train_std'],modes,horizon=h) for h in [6,12]}
        metrics={'status':'complete','from_scratch':True,'history_max_encoder_depth':8,'best_epoch':saved['epoch'],'epochs_run':len(history),'validation':validation,'test':test,
                 'test_early_contexts':early,'checkpoint_reload_difference':difference,'maximum_training_batch_gradient':max(h['maximum_batch_gradient_norm'] for h in history)}
        atomic(folder/'metrics.json',metrics);tracker.final(metrics)


if __name__=='__main__':main()
