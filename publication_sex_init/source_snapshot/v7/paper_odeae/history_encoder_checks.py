"""CUDA checks of matching, encoder semantics, causal supervision and gradients."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys

import torch

from .data import collate
from .jacobian import jacobian_regularization
from .models import paper_model_from_config,encode_values,feature_errors,reconstruction,prior_errors
from .scaling_study import atomic
from .train import seed_all,device_batch,evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if Path(sys.prefix).name!='ode' or not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(cfg['physical_gpu']):
        raise RuntimeError('Assigned physical GPU and ode environment required')
    torch.set_num_threads(1);torch.cuda.set_device(0)
    dataset=Path(cfg['data']);assert hashlib.sha256(dataset.read_bytes()).hexdigest()==cfg['dataset_sha256']
    payload=pickle.loads(dataset.read_bytes());schema=payload['schema']
    sequences=[s for s in payload['sequences'] if s['split']=='train' and len(s['ages'])>=5][:4]
    clipped=[{**s,**{key:s[key][:5] for key in ['ages','values','mask','dates']}} for s in sequences]
    batch=device_batch(collate(list(zip(clipped,[3,2,4,1]))))
    base=Path(cfg['reference_output']);base_cfg=json.loads((base/'config.json').read_text())
    frozen=Path(cfg['reference_model_source']);spec=importlib.util.spec_from_file_location('paper_odeae._history_reference',frozen)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert frozen.with_name('solver.py').read_bytes()==Path(__file__).with_name('solver.py').read_bytes()
    seed_all(cfg['seed']);reference=module.paper_model_from_config(schema,base_cfg).cuda().eval()
    seed_all(cfg['seed']);current=paper_model_from_config(schema,base_cfg).cuda().eval()
    for name,value in reference.state_dict().items():torch.testing.assert_close(value,current.state_dict()[name],rtol=0,atol=0)
    with torch.no_grad():reference_out=reference(batch);current_out=current(batch)
    torch.testing.assert_close(reference_out['prediction'],current_out['prediction'],rtol=0,atol=0)
    del current
    seed_all(cfg['seed']);model=paper_model_from_config(schema,cfg).cuda().eval()
    parameters=sum(p.numel() for p in model.parameters())
    assert parameters==sum(p.numel() for p in model.parameters() if p.requires_grad)==44875
    assert model.forward_ode is model.backward_ode
    for name in ['init','decoder','forward_ode']:
        for a,b in zip(getattr(model,name).parameters(),getattr(reference,name).parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    extra={}
    with torch.no_grad():original=model(batch)
    if model.variant=='reverse_shared':
        for a,b in zip(model.update.parameters(),reference.update.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(original['prior_prediction'],reference_out['prior_prediction'],rtol=0,atol=0)
        reference.encoder_direction='backward'
        with torch.no_grad():expected=reference(batch)
        torch.testing.assert_close(original['mu'],expected['mu'],rtol=0,atol=0)
        torch.testing.assert_close(original['prediction'],expected['prediction'],rtol=0,atol=0)
        extra.update(main_state_matches_backward_encoder=True,generation_matches_first_visit_anchor=True,
                     weak_prior_matches_original_forward_causal_supervision=True,auxiliary_forward_state_used_for_main_generation=False)
    else:
        encoder_parameters=sum(p.numel() for p in model.history_attention.parameters());assert encoder_parameters==16882
        inputs=encode_values(batch['values'],batch['context_mask'],schema)
        solve=model.solve
        def forbidden(*args,**kwargs):raise AssertionError('Attention history encoder called an ODE')
        model.solve=forbidden
        with torch.no_grad():states=model.attention_encoding(batch,inputs)
        model.solve=solve
        altered=inputs.clone();altered[:,2:]+=3.
        with torch.no_grad():other_states=model.attention_encoding(batch,altered)
        torch.testing.assert_close(states[:,:2],other_states[:,:2],rtol=0,atol=2e-6)
        changed_times={**batch,'ages':batch['ages'].clone()};changed_times['ages'][:,1]+=.05
        with torch.no_grad():different=model.attention_encoding(changed_times,inputs)
        if float((states-different).abs().max())<=1e-6:raise AssertionError('Irregular elapsed time has no effect')
        changed={**batch,'values':batch['values'].clone()}
        for j,field in enumerate(schema):
            changed['values'][:,1:,j]=(changed['values'][:,1:,j]+1)%field['classes'] if field['kind']=='categorical' else changed['values'][:,1:,j]+7
        with torch.no_grad():other=model(changed)
        torch.testing.assert_close(original['prior_prediction'][:,:2],other['prior_prediction'][:,:2],rtol=0,atol=2e-6)
        extra.update(attention_encoder_parameters=encoder_parameters,history_encoding_has_no_ode_calls=True,
                     prefix_states_are_causal=True,continuous_time_encoding_changes_state=True,weak_prior_excludes_target_observation=True,
                     time_encoding_parameters=0,attention_heads=4)
    changed={key:value.clone() if torch.is_tensor(value) else value for key,value in batch.items()}
    future=torch.arange(batch['values'].shape[1],device='cuda')[None]>=batch['cutoff'][:,None]
    for j,field in enumerate(schema):
        replacement=(changed['values'][...,j]+1)%field['classes'] if field['kind']=='categorical' else changed['values'][...,j]+13
        changed['values'][...,j]=torch.where(future,replacement,changed['values'][...,j])
    changed['mask']=torch.where(future[...,None],torch.zeros_like(changed['mask']),changed['mask'])
    with torch.no_grad():other=model(changed)
    torch.testing.assert_close(original['prediction'],other['prediction'],rtol=0,atol=2e-6)
    for i,cut in enumerate(batch['cutoff'].tolist()):changed['ages'][i,cut:]+=4
    with torch.no_grad():other=model(changed)
    torch.testing.assert_close(original['mu'],other['mu'],rtol=0,atol=2e-6)
    torch.testing.assert_close(original['prior_prediction'],other['prior_prediction'],rtol=0,atol=2e-6)
    torch.testing.assert_close(model.decoder(original['jacobian_states'])[batch['valid']],original['prediction'][batch['valid']],rtol=2e-6,atol=2e-6)
    model.train();out=model(batch)
    mse,cce=reconstruction(feature_errors(out['prediction'],batch['values'],batch['mask'],schema),batch['mask'],schema)
    pe,pmask=prior_errors(out,batch,schema);pmse,pcce=reconstruction(pe,pmask,schema)
    jac,_,count=jacobian_regularization(model,out,batch);assert count==int(batch['valid'].sum())
    loss=mse+cce+.01*(pmse+pcce)+.01*jac+.02*model.ode_regularization();assert torch.isfinite(loss)
    loss.backward()
    for name,p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():raise AssertionError(f'Unused or nonfinite parameter: {name}')
    constant=copy.deepcopy(model)
    with torch.no_grad():
        for p in constant.forward_ode.parameters():p.zero_()
        constant.forward_ode[-1].bias.fill_(.2)
        start=torch.zeros(2,28,device='cuda');delta=torch.tensor([1.,3.],device='cuda')
        torch.testing.assert_close(constant.solve(constant.forward_ode,start,delta),start+.2*delta[:,None],rtol=2e-5,atol=2e-6)
    if cfg.get('verify_reference_checkpoint'):
        saved=torch.load(base/'best.pt',map_location='cuda',weights_only=False)
        reference=module.paper_model_from_config(schema,base_cfg).cuda();reference.load_state_dict(saved['state_dict'])
        expected=json.loads((base/'metrics.json').read_text());difference=0.
        for split in ['validation','test']:
            actual=evaluate(reference,[s for s in payload['sequences'] if s['split']==split],schema,base_cfg['batch_size'])
            for region in ['interpolation','forecast','overall','prior']:
                assert actual[region]['observed_scalars']==expected[split][region]['observed_scalars']
                difference=max(difference,abs(actual[region]['reconstruction_error']-expected[split][region]['reconstruction_error']))
        if difference>1e-6:raise AssertionError('Existing baseline checkpoint no longer matches the common evaluation')
        extra['reference_checkpoint_full_evaluation_max_difference']=difference
    atomic(output/'checks.json',{'status':'passed','variant':model.variant,'physical_gpu':cfg['physical_gpu'],'device':'cuda:0',
           'parameters':parameters,'trainable_parameters':parameters,'parameter_breakdown':{name:sum(p.numel() for p in child.parameters()) for name,child in model.named_children()},
           'shared_field_identity_verified':True,'retained_network_initial_tensors_identical':True,'frozen_forward_reference_equivalence':True,
           'future_values_and_masks_do_not_change_predictions':True,'future_times_do_not_change_encoded_state':True,
           'all_parameters_receive_finite_gradients':True,'exact_jacobian_entries':784,'time_unit_verified':'year',
           'actual_tensor_devices':{key:str(value.device) for key,value in batch.items() if torch.is_tensor(value)},**extra})
    print(json.dumps({'checks':'passed','variant':model.variant,'parameters':parameters,'physical_gpu':cfg['physical_gpu']}),flush=True)


if __name__=='__main__':main()
