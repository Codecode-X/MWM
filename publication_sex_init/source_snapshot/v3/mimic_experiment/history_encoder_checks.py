"""CUDA checks for the MIMIC variants and their globally reduced both loss."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

from paper_odeae.models import paper_model_from_config,encode_values,feature_errors,prior_errors
from paper_odeae.jacobian import squared_jacobian_norm
from paper_odeae.train import seed_all,device_batch
from .data import collate
from .prepare import atomic
from .train_baseline import normalization_and_modes,verify_reference
from .train_history_encoder import partition_batches,denominators,local_loss


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if Path(sys.prefix).name!='ode' or not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(cfg['physical_gpu']):
        raise RuntimeError('Explicit ode environment and assigned CUDA GPU required')
    torch.set_num_threads(1);torch.cuda.set_device(0)
    payload=pickle.loads(Path(cfg['data']).read_bytes());schema=payload['schema']
    training=[s for s in payload['sequences'] if s['split']=='train']
    clips=[{**s,**{key:s[key][:8] for key in ['ages','values','mask','dates']}} for s in training[:4]]
    items=list(zip(clips,[3,4,2,1]));batch=device_batch(collate(items))
    base=Path(cfg['reference_output']);base_cfg=json.loads((base/'config.json').read_text())
    frozen=base/'source_snapshot/paper_odeae/models.py'
    spec=importlib.util.spec_from_file_location('paper_odeae._frozen_mimic_history_reference',frozen)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert frozen.with_name('solver.py').read_bytes()==Path(__file__).resolve().parents[1].joinpath('paper_odeae/solver.py').read_bytes()
    seed_all(cfg['seed']);reference=module.paper_model_from_config(schema,base_cfg).cuda().eval()
    seed_all(cfg['seed']);current=paper_model_from_config(schema,base_cfg).cuda().eval()
    for name,value in reference.state_dict().items():torch.testing.assert_close(value,current.state_dict()[name],rtol=0,atol=0)
    with torch.no_grad():expected_forward=reference(batch);current_out=current(batch)
    torch.testing.assert_close(expected_forward['prediction'],current_out['prediction'],rtol=0,atol=0)
    del current
    seed_all(cfg['seed']);model=paper_model_from_config(schema,cfg).cuda().eval()
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in model.parameters() if p.requires_grad)==43280
    assert model.forward_ode is model.backward_ode
    for name in ['init','decoder','forward_ode']:
        for x,y in zip(getattr(model,name).parameters(),getattr(reference,name).parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
    with torch.no_grad():original=model(batch)
    extra={}
    if model.variant=='reverse_shared':
        torch.testing.assert_close(original['prior_prediction'],expected_forward['prior_prediction'],rtol=0,atol=0)
        reference.encoder_direction='backward'
        with torch.no_grad():expected=reference(batch)
        torch.testing.assert_close(original['mu'],expected['mu'],rtol=0,atol=0)
        torch.testing.assert_close(original['prediction'],expected['prediction'],rtol=0,atol=0)
        extra.update(reverse_main_state_and_first_visit_generation_verified=True,weak_prior_matches_original_causal_pass=True)
    else:
        assert sum(p.numel() for p in model.history_attention.parameters())==16087
        assert model.history_attention.time_unit=='hour'
        torch.testing.assert_close(model.history_attention.period_hours[[0,-1]],torch.tensor([.5,48.],device='cuda'),rtol=2e-6,atol=2e-6)
        inputs=encode_values(batch['values'],batch['context_mask'],schema)
        solve=model.solve
        def forbidden(*args,**kwargs):raise AssertionError('ODE called inside attention history encoding')
        model.solve=forbidden
        with torch.no_grad():states=model.attention_encoding(batch,inputs)
        model.solve=solve
        changed_inputs=inputs.clone();changed_inputs[:,2:]+=3
        with torch.no_grad():changed_states=model.attention_encoding(batch,changed_inputs)
        torch.testing.assert_close(states[:,:2],changed_states[:,:2],rtol=0,atol=2e-6)
        changed={**batch,'ages':batch['ages'].clone()};changed['ages'][:,1]+=.25
        with torch.no_grad():changed_states=model.attention_encoding(changed,inputs)
        if float((states-changed_states).abs().max())<=1e-6:raise AssertionError('Physical time encoding has no effect')
        changed={**batch,'values':batch['values'].clone()}
        for j,field in enumerate(schema):
            changed['values'][:,1:,j]=(changed['values'][:,1:,j]+1)%field['classes'] if field['kind']=='categorical' else changed['values'][:,1:,j]+3
        with torch.no_grad():altered=model(changed)
        torch.testing.assert_close(original['prior_prediction'][:,:2],altered['prior_prediction'][:,:2],rtol=0,atol=2e-6)
        extra.update(attention_history_has_no_ode_calls=True,attention_prefix_is_causal=True,time_encoding_unit='hour',
                     attention_encoder_parameters=16087,weak_prior_excludes_current_observation=True)
        # Existing ADNI attention checkpoints remain compatible after the schema generalization.
        adni=Path(cfg['adni_output'])/'attention_time/seed_0'
        saved=torch.load(adni/'best.pt',map_location='cuda',weights_only=False)
        restored=paper_model_from_config(saved['schema'],saved['config']).cuda();restored.load_state_dict(saved['state_dict'],strict=True)
        assert sum(p.numel() for p in restored.parameters())==44875
        extra['adni_attention_checkpoint_strict_load_passed']=True
        del restored,saved
    changed={k:v.clone() if torch.is_tensor(v) else v for k,v in batch.items()}
    future=torch.arange(batch['values'].shape[1],device='cuda')[None]>=batch['cutoff'][:,None]
    for j,field in enumerate(schema):
        value=(changed['values'][...,j]+1)%field['classes'] if field['kind']=='categorical' else changed['values'][...,j]+20
        changed['values'][...,j]=torch.where(future,value,changed['values'][...,j])
    changed['mask']=torch.where(future[...,None],torch.zeros_like(changed['mask']),changed['mask'])
    with torch.no_grad():altered=model(changed)
    torch.testing.assert_close(original['prediction'],altered['prediction'],rtol=0,atol=2e-6)
    for i,cut in enumerate(batch['cutoff'].tolist()):changed['ages'][i,cut:]+=20
    with torch.no_grad():altered=model(changed)
    torch.testing.assert_close(original['mu'],altered['mu'],rtol=0,atol=2e-6)
    torch.testing.assert_close(original['prior_prediction'],altered['prior_prediction'],rtol=0,atol=2e-6)
    # Reproduce the original per-rank dropout streams, including a second batch.
    rngs=[np.random.RandomState(cfg['seed']+1) for _ in range(4)]
    expected_rngs=[np.random.RandomState(cfg['seed']+1) for _ in range(4)]
    partitions=None
    for _ in range(2):
        partitions=partition_batches(items,list(range(4)),.1,rngs)
        for rank,part in enumerate(partitions):
            expected=collate([items[rank]]);expected['context_mask']*=torch.from_numpy(expected_rngs[rank].random_sample(tuple(expected['context_mask'].shape))>=.1)
            torch.testing.assert_close(part['context_mask'],expected['context_mask'].cuda(),rtol=0,atol=0)
    model.train();denoms=denominators(partitions);outputs=[model(part) for part in partitions]
    actual=sum(local_loss(model,out,part,schema,cfg,denoms)[0] for out,part in zip(outputs,partitions))+cfg['l2_weight']*model.ode_regularization()
    numeric=torch.tensor([field['kind']=='numerical' for field in schema],device='cuda')
    field_sums=sum(feature_errors(out['prediction'],part['values'],part['mask'],schema).sum((0,1)) for out,part in zip(outputs,partitions))
    means=field_sums/denoms[0].clamp_min(1)
    psums=sum(prior_errors(out,part,schema)[0].sum((0,1)) for out,part in zip(outputs,partitions));pmeans=psums/denoms[1].clamp_min(1)
    locations=torch.cat([out['jacobian_states'][part['valid']].detach() for out,part in zip(outputs,partitions)])
    jac=squared_jacobian_norm(model.forward_ode,locations).mean()
    expected=cfg['mse_weight']*means[numeric].sum()+cfg['cce_weight']*means[~numeric].sum()+cfg['prior_weight']*pmeans.sum()+cfg['jacobian_weight']*jac+cfg['l2_weight']*model.ode_regularization()
    torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-6)
    parameters=list(model.parameters());ga=torch.autograd.grad(actual,parameters,retain_graph=True);gb=torch.autograd.grad(expected,parameters)
    gradient_difference=0.
    for a,b in zip(ga,gb):
        if not torch.isfinite(a).all():raise AssertionError('Nonfinite trainable gradient')
        torch.testing.assert_close(a,b,rtol=5e-5,atol=3e-6);gradient_difference=max(gradient_difference,float((a-b).abs().max()))
    constant=copy.deepcopy(model)
    with torch.no_grad():
        for parameter in constant.forward_ode.parameters():parameter.zero_()
        constant.forward_ode[-1].bias.fill_(.2);z=torch.zeros(2,28,device='cuda');dt=torch.tensor([2.,5.],device='cuda')
        torch.testing.assert_close(constant.solve(constant.forward_ode,z,dt),z+.2*dt[:,None],rtol=2e-5,atol=2e-6)
    if cfg['verify_reference']:
        means,modes=normalization_and_modes(payload);cfg.update(raw_scale=payload['metadata']['scale'],normalized_train_means=means)
        verify_reference(payload,cfg,output,modes)
        extra['full_reference_test_max_difference']=json.loads((output/'reference_evaluation_check.json').read_text())['max_difference']
    atomic(output/'checks.json',{'status':'passed','physical_gpu':cfg['physical_gpu'],'device':'cuda:0','parameters':43280,'trainable_parameters':43280,
           'shared_field_identity_verified':True,'retained_network_initial_tensors_identical':True,'frozen_reference_prediction_equivalence':True,
           'future_observations_do_not_change_predictions':True,'future_times_do_not_change_state':True,'all_parameter_gradients_finite':True,
           'global_loss_difference':float((actual-expected).abs()),'global_gradient_max_difference':gradient_difference,
           'four_original_rank_dropout_streams_reproduced':True,'time_unit_verified':'hour','jacobian_entries':784,
           'parameter_breakdown':{name:sum(p.numel() for p in child.parameters()) for name,child in model.named_children()},
           'actual_tensor_devices':{k:str(v.device) for k,v in batch.items() if torch.is_tensor(v)},**extra})
    print(json.dumps({'checks':'passed','variant':cfg['variant'],'physical_gpu':cfg['physical_gpu'],'parameters':43280}),flush=True)


if __name__=='__main__':main()
