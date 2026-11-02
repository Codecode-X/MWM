"""CUDA checks for a matched fresh start and a single full-batch gradient step."""
import argparse
import copy
import json
import os
from pathlib import Path
import pickle
import sys

import torch

from paper_odeae.train import seed_all,device_batch
from .data import ICUContexts,ExactBatchSampler,collate
from .prepare import atomic
from .train_full_batch import make_model,loss_function


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True);parser.add_argument('--gpu',type=int,required=True);args=parser.parse_args()
    if Path(sys.prefix).name!='ode' or not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):raise RuntimeError('Assigned CUDA GPU and ode required')
    torch.set_num_threads(1);torch.cuda.set_device(0);manifest=json.loads(Path(args.manifest).read_text());cfgs=manifest['configs']
    payload=pickle.loads(Path(cfgs['forward_shared']['data']).read_bytes());schema=payload['schema']
    fields=['latent','ode_width','mlp_width','model','seed','batch_size','micro_batch_size','gradient_accumulation_steps','gradient_partitions',
            'lr','lr_decay','lr_decay_steps','mse_weight','cce_weight','prior_weight','jacobian_weight','l2_weight','max_epochs','min_epochs','patience',
            'input_dropout','normalization','rtol','atol','data','dataset_sha256','encoder_initialization','shared_vector_field','training_context_hours','evaluation_context_hours']
    for field in fields:assert cfgs['forward_shared'][field]==cfgs['reverse_shared'][field],field
    training=[s for s in payload['sequences'] if s['split']=='train']
    dataset=ICUContexts(training[:256],training=True,seed=0);dataset.set_epoch(1);sampler=ExactBatchSampler(len(dataset),128,0,1,0);sampler.set_epoch(1)
    indices=next(iter(sampler));assert len(indices)==128
    seed_all(1);batch=device_batch(collate([dataset[i] for i in indices],.1))
    seed_all(1);same=device_batch(collate([dataset[i] for i in indices],.1))
    torch.testing.assert_close(batch['context_mask'],same['context_mask'],rtol=0,atol=0)
    models={}
    for key,cfg in cfgs.items():
        seed_all(cfg['seed']);models[key]=make_model(schema,cfg).cuda().eval()
        assert sum(p.numel() for p in models[key].parameters())==43280
        assert models[key].forward_ode is models[key].backward_ode
    forward=models['forward_shared'];reverse=models['reverse_shared']
    for key,value in forward.state_dict().items():torch.testing.assert_close(value,reverse.state_dict()[key],rtol=0,atol=0)
    with torch.no_grad():
        a=forward(batch);b=reverse(batch)
    torch.testing.assert_close(a['prior_prediction'],b['prior_prediction'],rtol=0,atol=0)
    changed={key:value.clone() if torch.is_tensor(value) else value for key,value in batch.items()}
    future=torch.arange(batch['values'].shape[1],device='cuda')[None]>=batch['cutoff'][:,None]
    for j,field in enumerate(schema):
        replacement=(changed['values'][...,j]+1)%field['classes'] if field['kind']=='categorical' else changed['values'][...,j]+20
        changed['values'][...,j]=torch.where(future,replacement,changed['values'][...,j])
    changed['mask']=torch.where(future[...,None],torch.zeros_like(changed['mask']),changed['mask'])
    results={}
    for key,model in models.items():
        original=a if key=='forward_shared' else b
        with torch.no_grad():other=model(changed)
        torch.testing.assert_close(original['prediction'],other['prediction'],rtol=0,atol=2e-6)
        model.train();model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
        optimizer=torch.optim.Adam(model.parameters(),lr=cfgs[key]['lr'])
        out=model(batch,sample=False);loss,_=loss_function(model,out,batch,schema,cfgs[key]);assert torch.isfinite(loss)
        loss.backward()
        for name,p in model.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():raise AssertionError(f'Nonfinite or unused parameter: {name}')
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
        results[key]={'status':'passed','parameters':43280,'full_training_batch':len(indices),'forward_calls':1,'backward_calls':1,'optimizer_steps':1,
                      'gradient_accumulation_steps':1,'peak_memory_bytes':torch.cuda.max_memory_allocated()}
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    atomic(output/'checks.json',{'status':'passed','physical_gpu':args.gpu,'device':'cuda:0','matched_training_configuration':True,
           'identical_initial_parameter_tensors':True,'identical_patient_order_and_dropout_masks':True,'same_causal_prior_supervision':True,
           'future_values_do_not_change_predictions':True,'models':results,'source_sha256':manifest['source_sha256']})
    print(json.dumps({'checks':'passed','physical_gpu':args.gpu,'full_batch':128,'gradient_accumulation_steps':1,'models':results}),flush=True)


if __name__=='__main__':main()
