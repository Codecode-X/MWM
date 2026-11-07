"""Fixed-parameter CUDA solver convergence; no optimizer is constructed."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from mechanism_common import (initialize,model_from,parameter_hash,encode_only,historical_loss,endpoint,
                              solve,flow_jacobian,flat_gradient,solver_counts,relative,brief,tensor_hash,encode_solver)
from mimic_experiment.prepare import atomic
from mimic_experiment.train_baseline import save_checkpoint
from paper_odeae.solver import solver_statistics


def cast_batch(batch,dtype):
    return {k:v.to(dtype=dtype) if torch.is_tensor(v) and v.is_floating_point() else v for k,v in batch.items()}


def run_setting(model,batch,z,schema,method,step,native_tolerance):
    counts={};x=z.detach().requires_grad_(True)
    loss,_,states,_,_=historical_loss(model,x,batch,schema,method=method,step=step,counts=counts)
    gz=torch.autograd.grad(loss,x,retain_graph=True)[0]
    gp=flat_gradient(loss,tuple(model.forward_ode.parameters()))
    # Isolate generation precision while retaining the original encoder's parameter path.
    # For snapshot_reencoded_state this is the native end-to-end gradient at the frozen value.
    requested_tolerance=model.rtol,model.atol;model.rtol,model.atol=native_tolerance
    encoded=encode_only(model,batch)
    encoder_gp=torch.cat([v.flatten() for v in torch.autograd.grad(encoded,tuple(model.forward_ode.parameters()),grad_outputs=gz.detach())]).detach()
    full_gp=gp.detach()+encoder_gp;model.rtol,model.atol=requested_tolerance
    encoded_refined=encode_solver(model,batch,method,step,counts)
    encoder_gp_refined=torch.cat([v.flatten() for v in torch.autograd.grad(encoded_refined,tuple(model.forward_ode.parameters()),grad_outputs=gz.detach())]).detach()
    full_gp_refined=gp.detach()+encoder_gp_refined
    past=states[:,0].detach();del states
    with torch.no_grad():future=endpoint(model,z,batch,1,method,step,counts)
    jpast=flow_jacobian(lambda v:endpoint(model,v,batch,-1,method,step,counts),z)
    jfuture=flow_jacobian(lambda v:endpoint(model,v,batch,1,method,step,counts),z)
    cycles={}
    for hours in [6.,12.,24.]:
        delta=z.new_full((len(z),),hours)
        with torch.no_grad():
            zp=solve(model,z,-delta,method,step,counts);zc=solve(model,zp,delta,method,step,counts)
        cycles[str(hours)]=brief((zc-z).norm(dim=-1)/z.norm(dim=-1).clamp_min(1e-30))
    singular=torch.linalg.svdvals(jpast)[:,0];fsingular=torch.linalg.svdvals(jfuture)[:,0]
    vectors={'past_endpoint':past,'future_endpoint':future.detach(),'past_singular_value':singular,'future_singular_value':fsingular,
             'history_state_gradient':gz.detach(),'history_field_gradient':gp.detach(),
             'history_encoder_field_gradient':encoder_gp,'history_full_field_gradient':full_gp,
             'refined_encoder_state':encoded_refined.detach(),'history_all_solvers_refined_full_field_gradient':full_gp_refined}
    return {'history_loss':float(loss.detach()),'past_gain':brief(singular),'future_gain':brief(fsingular),
            'history_state_gradient_norm':float(gz.norm()),'history_field_gradient_norm':float(gp.norm()),
            'history_full_field_gradient_norm':float(full_gp.norm()),'history_encoder_field_gradient_norm':float(encoder_gp.norm()),
            'history_all_solvers_refined_full_field_gradient_norm':float(full_gp_refined.norm()),
            'encoder_refinement_state_difference':relative(encoded_refined.detach(),encoded.detach()),
            'encoder_vjp_tolerance':list(native_tolerance),'cycle_errors':cycles,'fixed_solver_counts':counts},vectors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--gpu',type=int,required=True);parser.add_argument('--snapshot-id',required=True);args=parser.parse_args()
    out=Path(args.out);manifest,cfg,payload,modes=initialize(out,args.gpu);folder=out/'experiment2'/args.snapshot_id;folder.mkdir(parents=True,exist_ok=True)
    selections=json.loads((out/'experiment2/snapshots.json').read_text());entry=next(x for x in selections if x['id']==args.snapshot_id)
    probe=torch.load(out/'experiment1/fixed_inputs_and_states.pt',map_location='cuda',weights_only=False)
    model32,saved=model_from(entry['path'],cfg,payload['schema']);base_hash=parameter_hash(model32)
    with torch.no_grad():current=encode_only(model32,probe['batch']).detach()
    anchors=[('fixed_epoch41_state',probe['state']),('snapshot_reencoded_state',current)]
    all_rows=[];comparison=[]
    for anchor,z32 in anchors:
        vectors_by_setting={};rows=[]
        settings=[('train32',torch.float32,'tsit5',1e-4,1e-6,None),
                  ('diagnostic32',torch.float32,'tsit5',1e-6,1e-8,None),
                  ('tight32',torch.float32,'tsit5',1e-8,1e-10,None),('tightest32',torch.float32,'tsit5',1e-9,1e-11,None),
                  ('diagnostic64',torch.float64,'tsit5',1e-6,1e-8,None),
                  ('tight64',torch.float64,'tsit5',1e-8,1e-10,None),('tightest64',torch.float64,'tsit5',1e-9,1e-11,None),
                  ('rk4_h0.2',torch.float64,'rk4',None,None,.2),('rk4_h0.1',torch.float64,'rk4',None,None,.1),('rk4_h0.05',torch.float64,'rk4',None,None,.05)]
        model64,_=model_from(entry['path'],cfg,payload['schema'],dtype=torch.float64)
        index=0;rk_previous=None;rk_converged=False
        while index<len(settings):
            name,dtype,method,rtol,atol,step=settings[index];index+=1
            record={'setting':name,'dtype':str(dtype),'solver':method,'rtol':rtol,'atol':atol,'fixed_step_hours':step,
                    'snapshot_id':args.snapshot_id,'parameter_snapshot':entry['path'],'original_parameter_hash':base_hash,
                    'anchor':anchor,'input_state_sha256_float32':tensor_hash(z32),'batch_kind':'fixed_validation_diagnostic',
                    'physical_gpu':args.gpu,'parameter_device':'cuda:0','state_device':'cuda:0','optimizer_steps':0}
            model=model32 if dtype==torch.float32 else model64;model.rtol,model.atol=rtol,atol
            fingerprint=parameter_hash(model);begin=time.monotonic();trace={'calls':[]}
            try:
                with solver_statistics() as trace:
                    result,vs=run_setting(model,cast_batch(probe['batch'],dtype),z32.to(dtype),payload['schema'],method,step,(cfg['rtol'],cfg['atol']))
                if parameter_hash(model)!=fingerprint:raise RuntimeError('Fixed-parameter diagnostic changed weights')
                vs={k:v.detach() for k,v in vs.items()};vectors_by_setting[name]=vs
                save_checkpoint(folder/(anchor+'_'+name+'_vectors.pt'),vs)
                record.update(status='complete',**result)
                if method=='rk4':
                    if rk_previous is not None:
                        differences={k:relative(vs[k],vectors_by_setting[rk_previous][k]) for k in vs}
                        record['step_halving_relative_differences']=differences
                        rk_converged=max(differences.values())<1e-3
                    rk_previous=name
                    if index==len(settings) and not rk_converged and step>.0125:
                        finer=step/2;settings.append((f'rk4_h{finer}',torch.float64,'rk4',None,None,finer))
            except (RuntimeError,FloatingPointError) as exc:
                record.update(status='failed',error_type=type(exc).__name__,error=str(exc))
                gc.collect();torch.cuda.empty_cache()
            record.update(seconds=time.monotonic()-begin,adaptive_solver_counts=solver_counts(trace))
            rows.append(record);atomic(folder/(anchor+'_progress.json'),rows)
            print(json.dumps({'snapshot':args.snapshot_id,'anchor':anchor,'setting':name,'status':record['status'],'seconds':record['seconds']}),flush=True)
        reference=rk_previous
        if reference is not None:
            for row in rows:
                if row['status']=='complete':
                    row['reference_setting']=reference;row['reference_step_halving_converged']=rk_converged
                    row['relative_differences_to_reference']={k:relative(vectors_by_setting[row['setting']][k].double(),vectors_by_setting[reference][k]) for k in vectors_by_setting[reference]}
        all_rows.extend(rows);comparison.append({'anchor':anchor,'rk4_reference':reference,'step_halving_converged':rk_converged})
        del model64,vectors_by_setting;gc.collect();torch.cuda.empty_cache()
    atomic(folder/'results.json',{'snapshot':entry,'settings':all_rows,'convergence':comparison,'optimizer_steps':0,'native_training_rtol':cfg['rtol'],'native_training_atol':cfg['atol']})


if __name__=='__main__':main()
