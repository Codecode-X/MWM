"""CUDA tests of cutoff causality, complete event removal and loss normalization."""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

from mimic_experiment.prepare import atomic
from mimic_experiment.data import collate as ordinary_collate
from mimic_experiment.supervision_ablation import task_collate,make_model,loss_function,multi_cutoff_loss,prefix_batch,masked_loss
from mimic_experiment.train_future_only import make_model as reference_factory,loss_function as reference_loss,future_mask
from paper_odeae.train import seed_all,device_batch


def clone(batch):return {k:v.clone() if torch.is_tensor(v) else v for k,v in batch.items()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);args=parser.parse_args()
    if Path(sys.prefix).name!='ode' or os.environ.get('CUDA_VISIBLE_DEVICES')!='3' or not torch.cuda.is_available():raise RuntimeError('Run checks on physical GPU3 using ode')
    torch.set_num_threads(1);torch.cuda.set_device(0)
    manifest=json.loads(Path(args.manifest).read_text());control=manifest['reference_config'];payload=pickle.loads(Path(control['data']).read_bytes());schema=payload['schema']
    sequences=[s for s in payload['sequences'] if s['split']=='train' and len(s['ages'])>=20][:4]
    items=[({**s,**{k:s[k][:20] for k in ['ages','values','mask','dates']}},cut) for s,cut in zip(sequences,[2,4,6,8])]
    records=[]
    for job in manifest['jobs']:
        cfg=job['config'];mode=cfg['supervision_ablation'];batch=device_batch(task_collate(items,cfg=cfg))
        seed_all(0);reference=reference_factory(schema,control).cuda()
        seed_all(0);model=make_model(schema,cfg).cuda()
        for key,value in reference.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        deltas=[];solve=model.solve
        def recorded(field,state,dt):deltas.append(dt.detach().clone());return solve(field,state,dt)
        model.solve=recorded;model.train();reference.train();out=model(batch);ref=reference(batch);future=future_mask(batch)
        torch.testing.assert_close(out['prediction'][future],ref['prediction'][future],rtol=0,atol=0)
        if mode=='A':
            if any(bool((d<0).any()) for d in deltas):raise AssertionError('A must integrate toward the future only')
            torch.testing.assert_close(batch['loss_cutoff_weights'].sum(1),torch.ones(len(items),device='cuda'))
            for slot in range(1,batch['loss_cutoffs'].shape[1]):
                changed=clone(batch);cuts=batch['loss_cutoffs'][:,slot]
                later=torch.arange(batch['ages'].shape[1],device='cuda')[None,:]>=cuts[:,None]
                for j,f in enumerate(schema):
                    changed['values'][...,j]=torch.where(later,(changed['values'][...,j]+1)%f['classes'] if f['kind']=='categorical' else changed['values'][...,j]+10,changed['values'][...,j])
                with torch.no_grad():other=model(changed)
                torch.testing.assert_close(out['forecast_paths'][slot]['prediction'],other['forecast_paths'][slot]['prediction'],rtol=0,atol=0)
                expected=ordinary_collate([(s,int(c)) for (s,_),c in zip(items,cuts.cpu().tolist())])['initialization_age'].cuda()
                torch.testing.assert_close(prefix_batch(batch,slot)['initialization_age'],expected,rtol=0,atol=0)
            toy_schema=[{'name':'x','kind':'numerical'}];cuts=torch.tensor([[1,2,3,4]],device='cuda')
            ages=torch.tensor([[1.,4.,10.,20.,36.]],device='cuda');dt=ages[:,None,:]-ages.gather(1,cuts-1)[:,:,None]
            predictions=torch.where(dt<=6,1.,torch.where(dt<=12,2.,3.))[:,:,:,None]
            toy={'values':torch.zeros(1,5,1,device='cuda'),'mask':torch.ones(1,5,1,device='cuda'),'ages':ages,'valid':torch.ones(1,5,device='cuda',dtype=torch.bool),
                 'loss_cutoffs':cuts,'loss_cutoff_weights':torch.full((1,4),.25,device='cuda')}
            toy_out={'forecast_paths':[{'prediction':predictions[:,s]} for s in range(4)]}
            mse,ce,diag=multi_cutoff_loss(model,toy_out,toy,toy_schema,cfg)
            torch.testing.assert_close(mse,torch.tensor(14/3,device='cuda'));torch.testing.assert_close(ce,torch.tensor(0.,device='cuda'))
            for name,value in [('short',1.),('medium',4.),('long',9.)]:torch.testing.assert_close(diag[f'A_{name}_mse'],torch.tensor(value,device='cuda'))
            flags={'each_cutoff_excludes_later_observations':True,'prefix_ages_match_independent_prefix_collation':True,
                   'cutoff_sampling_weights_sum_to_one':True,'equal_horizon_stratum_normalization_verified':True}
        else:
            aux=out['masked_retrospection'];eligible=aux['eligible'];idx=batch['masked_history_index']
            if eligible[0] or not bool(eligible[1:].all()):raise AssertionError('Histories with fewer than3 events have no internal mask')
            if not bool(((idx[eligible]>0)&(idx[eligible]<batch['cutoff'][eligible]-1)).all()):raise AssertionError('Mask only internal history records')
            if sum(bool((d<0).any()) for d in deltas)!=1:raise AssertionError('Only one masked retrospection solve may integrate toward past')
            changed=clone(batch)
            for row in range(len(items)):
                if not bool(eligible[row]):continue
                j=int(idx[row])
                for f,field in enumerate(schema):changed['values'][row,j,f]=(changed['values'][row,j,f]+1)%field['classes'] if field['kind']=='categorical' else changed['values'][row,j,f]+10
            with torch.no_grad():other=model(changed)
            torch.testing.assert_close(aux['anchor'],other['masked_retrospection']['anchor'],rtol=0,atol=0)
            torch.testing.assert_close(aux['prediction'],other['masked_retrospection']['prediction'],rtol=0,atol=0)
            one_items=[items[1]];one=device_batch(task_collate(one_items,cfg=cfg));removed=int(one['masked_history_index'][0]);s,cut=one_items[0]
            keep=[j for j in range(len(s['ages'])) if j!=removed]
            compact={**s,**{k:np.asarray(s[k])[keep] for k in ['ages','values','mask','dates']}}
            compact_batch=device_batch(ordinary_collate([(compact,cut-1)]))
            with torch.no_grad():
                masked_anchor=model.masked_retrospection(one)['anchor'];compressed=reference(compact_batch)['mu']
            torch.testing.assert_close(masked_anchor,compressed,rtol=1e-6,atol=1e-7)
            original_future,original_parts=reference_loss(reference,ref,batch,schema,control)
            same_future,same_parts=reference_loss(model,out,batch,schema,cfg)
            torch.testing.assert_close(same_future,original_future,rtol=0,atol=0)
            reference_grads=torch.autograd.grad(original_future,tuple(reference.parameters()))
            future_grads=torch.autograd.grad(same_future,tuple(model.parameters()),retain_graph=True)
            for a,b in zip(reference_grads,future_grads):torch.testing.assert_close(a,b,rtol=0,atol=0)
            total,_=loss_function(model,out,batch,schema,cfg);mm,mc=masked_loss(out,batch,schema)
            torch.testing.assert_close(total,same_future+cfg['masked_retrospection_weight']*(mm+mc),rtol=0,atol=0)
            flags={'masked_labels_do_not_affect_auxiliary_anchor_or_prediction':True,'skipped_UPDATE_matches_physically_deleted_event':True,
                   'original_future_value_and_gradients_bit_identical':True,'internal_only_and_short_history_skip_verified':True,
                   'masked_loss_coefficient_verified':True}
        total,_=loss_function(model,out,batch,schema,cfg);total.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise AssertionError('Require finite parameter gradients')
        model.eval()
        with torch.no_grad():evaluated=model(batch)
        torch.testing.assert_close(evaluated['prediction'][future],out['prediction'][future],rtol=0,atol=0)
        records.append({'id':job['id'],'mode':mode,'parameters':sum(p.numel() for p in model.parameters()),'parameter_device':str(next(model.parameters()).device),
                        'prediction_device':str(out['prediction'].device),'same_initial_parameters':True,'common_evaluation_retained':True,
                        'finite_gradients':True,**flags})
        del model,reference,out,ref,total
        torch.cuda.empty_cache()
    atomic(Path(args.manifest).parent/'cuda_checks.json',{'status':'passed','physical_gpu':3,'jobs':records})
    print(json.dumps({'status':'passed','physical_gpu':3,'jobs':records}),flush=True)


if __name__=='__main__':main()
