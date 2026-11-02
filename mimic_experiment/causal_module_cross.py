"""Fixed INIT, crossed ODE/UPDATE snapshots and fixed-input UPDATE Jacobians."""
import argparse
from pathlib import Path

import torch

from direction_gradient_common import initialize
from mechanism_common import parameter_hash,tensor_hash,relative,solver_counts
from causal_history_model import make_causal_model
from paper_odeae.models import encode_values
from paper_odeae.solver import solver_statistics
from mimic_experiment.prepare import atomic
from mimic_experiment.train_baseline import save_checkpoint


def prefix(state,prefix):return {k[len(prefix)+1:]:v for k,v in state.items() if k.startswith(prefix+'.')}


def encode_trace(model,batch,patient):
    values,ages,cut=batch['values'],batch['ages'],batch['cutoff'];inputs=encode_values(values,batch['context_mask'],model.schema)
    h=model.init(batch['initialization_age'][:,None]/100.);previous=ages[:,0];posts=[];preupdate=[];observations=[]
    count=int(cut[patient]);initial=h
    for j in range(batch['max_context']):
        index=torch.minimum(torch.full_like(cut,j),cut-1);active=j<cut;time=ages.gather(1,index[:,None]).squeeze(1)
        x=inputs[torch.arange(len(cut),device='cuda'),index]
        evolved=model.solve(model.forward_ode,h,torch.where(active,time-previous,torch.zeros_like(time)))
        h=torch.where(active[:,None],model.update(torch.cat([evolved,x],-1)),h);previous=torch.where(active,time,previous)
        if j<count:posts.append(h);preupdate.append(evolved[patient].detach());observations.append(x[patient].detach())
    params=tuple(model.forward_ode.parameters());basis=torch.eye(28,device='cuda',dtype=h.dtype)
    gs=torch.autograd.grad(h[patient],[initial,*posts],grad_outputs=basis,is_grads_batched=True,retain_graph=True)
    state_gains=[float(torch.linalg.svdvals(g[:,patient,:])[0]) for g in gs]
    gp=torch.autograd.grad(h[patient],params,grad_outputs=basis,is_grads_batched=True)
    field_jac=torch.cat([g.reshape(28,-1) for g in gp],dim=1).detach()
    return {'cutoff_state':h[patient].detach(),'initial_state_gain':state_gains[0],'postupdate_gains':state_gains[1:],
            'field_parameter_jacobian_max_singular_value':float(torch.linalg.svdvals(field_jac)[0]),
            'field_parameter_jacobian_frobenius':float(field_jac.norm()),'preupdate_states':torch.stack(preupdate),
            'observations':torch.stack(observations),'field_parameter_jacobian':field_jac}


def update_jacobian(network,states,observations):
    x=torch.cat([states,observations],-1);d=1-network[1](network[0](x)).square()
    jac=(network[2].weight[None]*d[:,None,:])@network[0].weight[:,:28]
    return jac.detach(),torch.linalg.svdvals(jac)[:,0].detach()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--init-context',choices=['stable','abnormal'],required=True);a=p.parse_args()
    root=Path(a.out);manifest,cfg,payload,modes=initialize(root,a.gpu);folder=root/'module_cross'/a.init_context;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'results.json').exists():return
    saved={name:torch.load(manifest[key],map_location='cuda',weights_only=False) for name,key in [('stable','stable_snapshot'),('abnormal','abnormal_snapshot')]}
    batch=saved['abnormal']['probe_batch'];batch={k:v.double() if torch.is_tensor(v) and v.is_floating_point() else v for k,v in batch.items()};patient=8
    conditions={'A':('stable','stable'),'B':('abnormal','stable'),'C':('stable','abnormal'),'D':('abnormal','abnormal')}
    rows={};banks={};tensors={};schema=payload['schema']
    model=make_causal_model(schema,cfg).double();model.rtol,model.atol=1e-8,1e-10
    reference_other={k:v.clone().double() for k,v in saved['stable']['state_dict'].items() if not k.startswith(('init.','update.','forward_ode.','backward_ode.'))}
    init_hash=tensor_hash(torch.cat([v.double().flatten() for v in prefix(saved[a.init_context]['state_dict'],'init').values()]))
    for label,(ode_source,update_source) in conditions.items():
        model.load_state_dict(saved['stable']['state_dict']);model.init.load_state_dict(prefix(saved[a.init_context]['state_dict'],'init'))
        model.forward_ode.load_state_dict(prefix(saved[ode_source]['state_dict'],'forward_ode'));model.update.load_state_dict(prefix(saved[update_source]['state_dict'],'update'))
        assert model.forward_ode is model.backward_ode
        assert all(torch.equal(model.state_dict()[k],v) for k,v in reference_other.items())
        current_init=tensor_hash(torch.cat([v.flatten() for v in model.init.state_dict().values()]));assert current_init==init_hash
        before=parameter_hash(model)
        try:
            with solver_statistics() as trace:result=encode_trace(model,batch,patient)
            assert parameter_hash(model)==before
            banks[label]={'states':result['preupdate_states'],'observations':result['observations']}
            tensors[label]=result
            rows[label]={'condition':label,'ode_source':ode_source,'update_source':update_source,'init_source':a.init_context,
                         'other_modules_source':'stable','parameter_hash':before,'init_hash':init_hash,
                         'cutoff_state':result['cutoff_state'].cpu().tolist(),'cutoff_state_norm':float(result['cutoff_state'].norm()),
                         'initial_state_gain':result['initial_state_gain'],'postupdate_gains':result['postupdate_gains'],
                         'maximum_postupdate_gain':max(result['postupdate_gains']),
                         'field_parameter_jacobian_max_singular_value':result['field_parameter_jacobian_max_singular_value'],
                         'field_parameter_jacobian_frobenius':result['field_parameter_jacobian_frobenius'],'solver_counts':solver_counts(trace),'status':'complete'}
        except (RuntimeError,FloatingPointError) as e:rows[label]={'status':'failed','error':str(e),'ode_source':ode_source,'update_source':update_source}
        atomic(folder/'conditions_progress.json',rows)
    # On each frozen state bank, apply BOTH UPDATE weights; no ODE re-integration.
    fixed_input={}
    for bank,data in banks.items():
        fixed_input[bank]={}
        for update_source in ['stable','abnormal']:
            model.update.load_state_dict(prefix(saved[update_source]['state_dict'],'update'))
            with torch.no_grad():jac,gain=update_jacobian(model.update,data['states'],data['observations'])
            fixed_input[bank][update_source]={'per_visit_gain':gain.cpu().tolist(),'maximum':float(gain.max()),'median':float(gain.median()),
                                             'states_sha256':tensor_hash(data['states']),'observations_sha256':tensor_hash(data['observations'])}
    interactions={}
    if all(r.get('status')=='complete' for r in rows.values()):
        for key in ['field_parameter_jacobian_max_singular_value','maximum_postupdate_gain','initial_state_gain']:
            aa,bb,cc,dd=[rows[k][key] for k in ['A','B','C','D']]
            interactions[key]={'B_over_A':bb/aa,'C_over_A':cc/aa,'D_over_A':dd/aa,'additive_interaction_D_B_C_A':dd-bb-cc+aa,
                               'multiplicative_interaction_DA_over_BC':dd*aa/(bb*cc)}
    save_checkpoint(folder/'fixed_banks_and_jacobians.pt',tensors)
    atomic(folder/'results.json',{'status':'complete','conditions':rows,'fixed_input_update_comparison':fixed_input,'factorial_effects':interactions,
                                'init_context':a.init_context,'patient_index':patient,'subject_id':str(batch['subject_ids'][patient]),
                                'history_times_hours':batch['ages'][patient,:int(batch['cutoff'][patient])].cpu().tolist(),
                                'history_observations':int(batch['cutoff'][patient]),'history_span_hours':float(batch['ages'][patient,int(batch['cutoff'][patient])-1]-batch['ages'][patient,0]),
                                'physical_gpu':a.gpu,'parameter_device':str(next(model.parameters()).device),'input_device':str(batch['values'].device),
                                'dtype':'float64','rtol':model.rtol,'atol':model.atol,'parameters':sum(p.numel() for p in model.parameters()),'optimizer_steps':0,
                                'interpretation':'module sensitivity under hybrid snapshots; not full causal identification or a trained-model performance comparison'})


if __name__=='__main__':main()
