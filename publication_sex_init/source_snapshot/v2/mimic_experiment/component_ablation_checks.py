"""CUDA tests of exact capacity, loss removal, field independence and masks."""
import torch
from paper_odeae.models import paper_model_from_config, prior_errors, reconstruction
from paper_odeae.jacobian import squared_jacobian_norm
from fixed_update_model import FixedUpdate
from fixed_update_checks import parameter_hash, gradient
from component_ablation_model import unique_fields
from component_ablation_loss import loss_function as objective, jacobian_regularization


def verify(model, batch, schema, cfg, loss_function):
    before = parameter_hash(model); model.eval()
    refcfg = {**cfg, 'shared_vector_field':True, 'prior_weight':.01, 'jacobian_weight':.01}
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    baseline = paper_model_from_config(schema, refcfg).cuda().eval(); baseline.update = FixedUpdate(baseline.update, baseline.latent, .1)
    torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
    assert sum(p.numel() for p in baseline.parameters()) == sum(p.numel() for p in model.parameters()) == cfg['target_parameters']
    for name in ['init','update','decoder']:
        assert all(torch.equal(a,b) for a,b in zip(getattr(baseline,name).parameters(),getattr(model,name).parameters()))
    out = model(batch, sample=False); ref = baseline(batch, sample=False)
    total, parts = objective(model, out, batch, schema, cfg)
    data = {'same_total_parameters':True, 'retained_network_initial_tensors_equal':True}
    fields = unique_fields(model)
    if cfg['ablation_component'] in ['no_prior','no_jacobian']:
        assert len(fields)==1
        assert all(torch.equal(out[k],ref[k]) for k in out)
        full, _ = objective(baseline, ref, batch, schema, refcfg)
        if cfg['ablation_component']=='no_prior':
            pe, mask = prior_errors(ref,batch,schema); a,b = reconstruction(pe,mask,schema); removed=.01*(a+b)
            assert cfg['prior_weight']==0 and cfg['jacobian_weight']==.01
        else:
            removed=.01*jacobian_regularization(baseline,ref,batch)[0]
            assert cfg['prior_weight']==.01 and cfg['jacobian_weight']==0
        expected = gradient(full-removed,baseline.parameters()); actual = gradient(total,model.parameters())
        error = float((actual-expected).norm()/expected.norm().clamp_min(1e-20)); assert error < 2e-4
        data.update(forward_outputs_bitwise_equal_to_main=True, removed_loss_gradient_relative_error=error,
                    removed_loss_scalar_difference=float((total-(full-removed)).abs()))
    else:
        assert len(fields)==2 and model.backward_ode is not model.forward_ode
        enc,gen=model.backward_ode,model.forward_ode
        assert not ({id(p) for p in enc.parameters()} & {id(p) for p in gen.parameters()})
        assert enc[0].out_features==gen[0].out_features==94 and enc[2].bias is None and gen[2].bias.numel()==28
        assert sum(p.numel() for p in enc.parameters())==5358 and sum(p.numel() for p in gen.parameters())==5386
        z=out['mu'].detach();difference=float((enc(z)-gen(z)).abs().max());assert difference < 1e-5
        jac=jacobian_regularization(model,out,batch)[0]
        states=out['jacobian_states'][batch['valid']].detach()
        manual=.5*(squared_jacobian_norm(enc,states).mean()+squared_jacobian_norm(gen,states).mean())
        assert torch.allclose(jac,manual,rtol=1e-6,atol=1e-8)
        jg=[float(gradient(jac,f.parameters()).norm()) for f in fields];assert all(v>0 for v in jg)
        data.update(independent_field_parameters=True, encoder_field_parameters=5358, generator_field_parameters=5386,
                    initial_two_field_max_absolute_difference=difference, jacobian_gradient_norms_per_field=jg,
                    prediction_relative_difference_from_main_initialization=float((out['prediction']-ref['prediction']).norm()/ref['prediction'].norm().clamp_min(1e-20)))
    # Future values must never alter any model output used for inference.
    idx=torch.arange(batch['values'].shape[1],device='cuda')[None,:]
    future=idx>=batch['cutoff'][:,None]
    altered={**batch,'values':torch.where(future[...,None],torch.zeros_like(batch['values']),batch['values'])}
    other=model(altered,sample=False)
    assert all(torch.equal(out[k],other[k]) for k in out)
    norms={name:float(gradient(total,getattr(model,name).parameters()).norm()) for name in ['init','update','backward_ode','forward_ode','decoder']}
    assert all(v>0 for v in norms.values())
    total.backward();assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    model.zero_grad(set_to_none=True);assert parameter_hash(model)==before
    return {'status':'passed','dataset':cfg['dataset'],'ablation_component':cfg['ablation_component'],'alpha':.1,
            'parameters':sum(p.numel() for p in model.parameters()),'physical_gpu':cfg['physical_gpu'],
            'parameter_device':str(next(model.parameters()).device),'actual_tensor_devices':{k:str(v.device) for k,v in batch.items() if torch.is_tensor(v)},
            'unique_fields':len(fields),'unique_ode_parameters':sum(p.numel() for f in fields for p in f.parameters()),
            'future_observations_do_not_change_outputs':True,'total_loss_gradient_norms':norms,'optimizer_steps':0,**data}
