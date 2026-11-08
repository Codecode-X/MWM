"""Change only the scalar INIT input of the standard full-batch forward model."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

from mimic_experiment import train_full_batch as native
from mimic_experiment import evaluation as scoring
from mimic_experiment.data import collate as ordinary_collate
from mimic_experiment.prepare import atomic
from paper_odeae.train import seed_all,device_batch

native_model_factory=native.make_model


def collate_init(items,dropout=0.,*,source,sex_index):
    batch=ordinary_collate(items,dropout)
    if source=='cutoff_age':return batch
    if source=='first_age':
        scalar=np.array([s['admission_age']+float(s['ages'][0])/8766. for s,_ in items],dtype=np.float32)
    elif source=='sex':
        for s,_ in items:
            if not s['mask'][0,sex_index] or s['values'][0,sex_index] not in [0,1]:raise ValueError('Sex must be known in the first history record')
        # The original model divides this interface value by100 before INIT.
        # Compensate here so the actual network receives F=0 or M=1.
        scalar=np.array([100.*s['values'][0,sex_index] for s,_ in items],dtype=np.float32)
    else:raise ValueError(source)
    batch['initialization_age']=torch.from_numpy(scalar)
    return batch


def make_model(schema,cfg,audit=True):
    source=cfg['init_covariate']
    if source not in ['first_age','sex']:raise ValueError('Choose first_age or sex INIT')
    if cfg['prior_weight']!=.01 or cfg['jacobian_weight']!=.01 or cfg['l2_weight']!=.02 or cfg.get('train_reconstruct_history') is False:
        raise ValueError('INIT ablation requires the standard latent2x both objective')
    sex_index=next(j for j,f in enumerate(schema) if f['name']=='sex')
    fn=partial(collate_init,source=source,sex_index=sex_index)
    native.collate=fn;scoring.collate=fn
    model=native_model_factory(schema,cfg)
    if audit:
        out=Path(cfg['comparison_output'])/'checks/forward_shared/smoke' if cfg.get('smoke') else Path(cfg['output'])
        logged=False
        def record(module,args):
            nonlocal logged
            if logged or not model.training:return
            x=args[0]
            if x.device.type!='cuda':raise RuntimeError('INIT model execution requires CUDA')
            if source=='sex' and not bool(((x==0)|(x==1)).all()):raise AssertionError('Actual sex INIT input must be0 or1')
            info={'physical_gpu':cfg['physical_gpu'],'actual_input_device':str(x.device),'input_shape':list(x.shape),
                  'source':source,'scalar_encoding':'first observation age /100' if source=='first_age' else 'F=0, M=1',
                  'minimum':float(x.detach().min()),'maximum':float(x.detach().max()),'mean':float(x.detach().mean()),
                  'first_eight_inputs':x[:8].detach().flatten().cpu().tolist(),'first_forward_training_mode':model.training,
                  'INIT_structure':'1 ->159 ->28','INIT_parameters':sum(p.numel() for p in model.init.parameters())}
            atomic(out/'init_input_audit.json',info);logged=True
        model.init.register_forward_pre_hook(record)
    return model


def check():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--manifest',required=True);args=parser.parse_args()
    if Path(sys.prefix).name!='ode' or os.environ.get('CUDA_VISIBLE_DEVICES')!='3' or not torch.cuda.is_available():raise RuntimeError('Use ode and physical GPU3 for CUDA checks')
    torch.set_num_threads(1);torch.cuda.set_device(0);manifest=json.loads(Path(args.manifest).read_text());base=manifest['reference_config']
    payload=pickle.loads(Path(base['data']).read_bytes());schema=payload['schema'];sex_index=next(j for j,f in enumerate(schema) if f['name']=='sex')
    rows=[]
    for code in [0,1]:rows.extend([s for s in payload['sequences'] if s['split']=='train' and len(s['ages'])>=8 and s['values'][0,sex_index]==code][:2])
    items=[({**s,**{k:s[k][:8] for k in ['ages','values','mask','dates']}},4) for s in rows]
    plain=device_batch(ordinary_collate(items));seed_all(0);reference=native_model_factory(schema,base).cuda()
    records=[]
    for job in manifest['jobs']:
        cfg=job['config'];source=cfg['init_covariate'];seed_all(0);model=make_model(schema,cfg,audit=False).cuda()
        for key,value in reference.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        batch=device_batch(collate_init(items,source=source,sex_index=sex_index))
        for key,value in plain.items():
            if torch.is_tensor(value) and key!='initialization_age':torch.testing.assert_close(value,batch[key],rtol=0,atol=0)
        captured=[];hook=model.init.register_forward_pre_hook(lambda module,inputs:captured.append(inputs[0].detach().clone()))
        model.train();out=model(batch)
        expected=torch.tensor([s['admission_age']+float(s['ages'][0])/8766. for s,_ in items],device='cuda',dtype=torch.float32)/100 if source=='first_age' else torch.tensor([s['values'][0,sex_index] for s,_ in items],device='cuda',dtype=torch.float32)
        torch.testing.assert_close(captured[-1].flatten(),expected,rtol=0,atol=0)
        short=[(s,2) for s,_ in items];short_batch=device_batch(collate_init(short,source=source,sex_index=sex_index))
        torch.testing.assert_close(batch['initialization_age'],short_batch['initialization_age'],rtol=0,atol=0)
        if source=='sex':
            older=[({**s,'admission_age':s['admission_age']+20},cut) for s,cut in items]
            older_batch=device_batch(collate_init(older,source=source,sex_index=sex_index))
            torch.testing.assert_close(batch['initialization_age'],older_batch['initialization_age'],rtol=0,atol=0)
        loss,_=native.loss_function(model,out,batch,schema,cfg);loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise AssertionError('Every parameter needs finite CUDA gradients')
        model.zero_grad(set_to_none=True);model.eval()
        with torch.no_grad():evaluated=model(batch)
        torch.testing.assert_close(evaluated['prediction'],out['prediction'],rtol=0,atol=0)
        hook.remove()
        # Confirm that the common evaluator resolves exactly the same collation.
        torch.testing.assert_close(device_batch(scoring.collate(items))['initialization_age'],batch['initialization_age'],rtol=0,atol=0)
        records.append({'id':job['id'],'init_covariate':source,'parameters':sum(p.numel() for p in model.parameters()),
                        'INIT_parameters':sum(p.numel() for p in model.init.parameters()),'identical_initial_parameters':True,
                        'all_other_batch_tensors_unchanged':True,'initialization_independent_of_history_cutoff':True,
                        'actual_INIT_inputs':captured[0].flatten().cpu().tolist(),'parameter_device':str(next(model.parameters()).device),
                        'prediction_device':str(out['prediction'].device),'finite_gradients':True,'same_train_validation_test_input_policy':True})
    atomic(Path(args.manifest).parent/'cuda_checks.json',{'status':'passed','physical_gpu':3,'jobs':records})
    print(json.dumps({'status':'passed','physical_gpu':3,'jobs':records}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--stage',choices=['smoke','training'],default='training');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());source=cfg['init_covariate'];native.make_model=make_model
    ordinary_atomic=native.atomic
    def record_metadata(path,value):
        if Path(path).name=='hardware.json':
            value={**value,'initialization_input_source':source,'initialization_scalar_encoding':'first observation age /100' if source=='first_age' else 'F=0, M=1',
                   'INIT_structure':'1 ->159 ->28','INIT_parameters':4798,'standard_loss_restored':True}
        return ordinary_atomic(path,value)
    native.atomic=record_metadata;native.main()


if __name__=='__main__':
    if '--check' in sys.argv:check()
    else:main()
