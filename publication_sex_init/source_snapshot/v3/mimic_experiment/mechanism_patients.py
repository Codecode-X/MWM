"""Read-only patient and chain-rule decomposition on the REAL stored batches."""
import argparse
import json
from pathlib import Path

import torch

from mechanism_common import (initialize,model_from,parameter_hash,historical_loss,flow_jacobian,endpoint,
                              flat_gradient,cosine,relative,batch_audit)
from direction_gradient_common import decompose
from mimic_experiment.prepare import atomic
from mimic_experiment.train_baseline import save_checkpoint
from mimic_experiment.train_full_batch import loss_function
from paper_odeae.models import prior_errors
from paper_odeae.jacobian import jacobian_regularization


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--case',required=True);args=p.parse_args()
    out=Path(args.out);manifest,cfg,payload,modes=initialize(out,args.gpu);folder=out/'experiment3'/args.case;folder.mkdir(parents=True,exist_ok=True)
    case=next(x for x in json.loads((out/'experiment3/cases.json').read_text()) if x['id']==args.case)
    model,saved=model_from(case['snapshot'],cfg,payload['schema']);stored=torch.load(case['batch_path'],map_location='cuda',weights_only=False)
    batch=stored['batch'] if 'batch' in stored else stored['probe_batch']
    fingerprint=parameter_hash(model);schema=payload['schema'];field=tuple(model.forward_ode.parameters())
    native=model(batch,sample=False);nh,nf=decompose(model,native,batch,schema,cfg)
    native_gh=flat_gradient(nh,field).detach();gf=flat_gradient(nf,field).detach()
    pe,pm=prior_errors(native,batch,schema);prior=cfg['prior_weight']*(pe.sum((0,1))/pm.sum((0,1)).clamp_min(1)).sum()
    jac,_,_=jacobian_regularization(model,native,batch);reg=cfg['jacobian_weight']*jac+cfg['l2_weight']*model.ode_regularization()
    gp=flat_gradient(prior,field).detach();gr=flat_gradient(reg,field).detach();z=native['mu'];nh_value=float(nh.detach())
    del native,nh,nf,prior,reg,pe,pm,jac
    loss,per,states,errors,denom=historical_loss(model,z,batch,schema)
    gh=flat_gradient(loss,field).detach();gz=torch.autograd.grad(loss,z,retain_graph=True)[0].detach()
    gs=torch.autograd.grad(loss,states,retain_graph=True)[0].detach()
    first_loss=(errors[:,0]/denom).sum();gz_first=torch.autograd.grad(first_loss,z,retain_graph=True)[0].detach()
    gs_first=torch.autograd.grad(first_loss,states,retain_graph=True)[0][:,0].detach()
    native_difference=relative(gh,native_gh)
    if abs(float(loss.detach())-nh_value)>1e-5 or native_difference>1e-3:raise RuntimeError('Historical path does not reproduce native loss/gradient')
    modules={};all_parameters=tuple(model.parameters());all_grads=torch.autograd.grad(loss,all_parameters,retain_graph=True)
    by_id={id(p):g.detach() for p,g in zip(all_parameters,all_grads)}
    for name,module in [('INIT',model.init),('UPDATE',model.update),('Decoder',model.decoder),('shared_ODE',model.forward_ode)]:
        modules[name]=float(torch.cat([by_id[id(p)].flatten() for p in module.parameters()]).norm())
    # All endpoint sensitivities use the same states and intervals as the native path, tighter tolerance for measurement.
    old_tol=model.rtol,model.atol
    jp_native=flow_jacobian(lambda x:endpoint(model,x,batch,-1),z.detach())
    un,sn,vhn=torch.linalg.svd(jp_native)
    model.rtol,model.atol=1e-6,1e-8
    jp=flow_jacobian(lambda x:endpoint(model,x,batch,-1),z.detach());jf=flow_jacobian(lambda x:endpoint(model,x,batch,1),z.detach())
    u,s,vh=torch.linalg.svd(jp);af=torch.linalg.svdvals(jf)[:,0];model.rtol,model.atol=old_tol
    vectors=[];encoders=[];patients=[];numeric=torch.tensor([f['kind']=='numerical' for f in schema],device='cuda')
    history_mask=batch['mask'][:,:states.shape[1]]*(torch.arange(states.shape[1],device='cuda')[None,:]<batch['cutoff'][:,None])[...,None]
    for i in range(len(per)):
        total=flat_gradient(per[i],field).detach();upstream=torch.zeros_like(gz);upstream[i]=gz[i]
        encoder=torch.cat([v.flatten() for v in torch.autograd.grad(z,field,grad_outputs=upstream,retain_graph=True)]).detach()
        direct=total-encoder;vectors.append(total);encoders.append(encoder)
        k=int(batch['cutoff'][i]);times=batch['ages'][i,:k];den_i=history_mask[i].sum(0)
        independent=(errors[i].sum(0)/den_i.clamp_min(1)).sum()
        interval=(times[1:]-times[:-1]).detach()
        projected=float(torch.dot(total,gh)/gh.square().sum().clamp_min(1e-30))
        true_gain=float(gz_first[i].norm()/gs_first[i].norm().clamp_min(1e-30))
        predicted_pullback=jp_native[i].T@gs_first[i]
        patients.append({'index':i,'subject_id':str(batch['subject_ids'][i]),'history_observations':k,
                         'history_span_hours':float(times[-1]-times[0]),'history_times_hours':times.cpu().tolist(),'intervals_hours':interval.cpu().tolist(),
                         'observed_numeric_scalars':float(history_mask[i,:,numeric].sum()),'observed_categorical_scalars':float(history_mask[i,:,~numeric].sum()),
                         'missing_fraction':1-float(history_mask[i].sum())/(k*len(schema)),
                         'cutoff_state_norm':float(z[i].detach().norm()),'cutoff_state':z[i].detach().cpu().tolist(),
                         'history_loss_original_batch_contribution':float(per[i].detach()),'history_loss_independent_patient_mean':float(independent.detach()),
                         'past_state_max_singular_value':float(s[i,0]),'future_state_max_singular_value':float(af[i]),
                         'native_past_state_max_singular_value':float(sn[i,0]),
                         'decoder_history_gradient_stacked_norm':float(gs[i].norm()),'cutoff_history_gradient_norm':float(gz[i].norm()),
                         'earliest_decoder_gradient_norm':float(gs_first[i].norm()),'earliest_cutoff_gradient_norm':float(gz_first[i].norm()),
                         'earliest_actual_direction_gain':true_gain,'earliest_gain_divided_by_singular_bound':true_gain/max(float(sn[i,0]),1e-30),
                         'decoder_gradient_alignment_with_top_left_singular_vector':abs(cosine(gs_first[i],un[i,:,0])),
                         'native_pullback_chain_relative_difference':relative(gz_first[i],predicted_pullback),
                         'history_field_gradient_norm':float(total.norm()),'encoder_field_gradient_norm':float(encoder.norm()),'direct_generation_field_gradient_norm':float(direct.norm()),
                         'encoder_direct_gradient_cosine':cosine(encoder,direct),'projected_share_of_batch_history_gradient':projected})
        if i%16==0:print(json.dumps({'case':args.case,'patient':i,'history_field_gradient_norm':float(total.norm())}),flush=True)
    vectors=torch.stack(vectors);encoders=torch.stack(encoders);sum_error=relative(vectors.sum(0),gh)
    if sum_error>2e-4:raise RuntimeError('Patient contributions do not sum to native batch gradient')
    conditions=[];n=len(patients)
    for ranking,key in [('longest_history','history_span_hours'),('highest_sensitivity','past_state_max_singular_value'),('highest_history_gradient','history_field_gradient_norm')]:
        order=sorted(range(n),key=lambda i:patients[i][key],reverse=True)
        for number in sorted({1,3,5,max(1,round(.1*n))}):
            removed=order[:number];keep=torch.ones(n,dtype=torch.bool,device='cuda');keep[removed]=False
            g_h=vectors[keep].sum(0);total=g_h+gf+gp+gr
            conditions.append({'selection':ranking,'removed_count':number,'removed_patient_indices':removed,
                               'normalization':'original full batch denominators retained; future/prior/regularizers held fixed',
                               'history_field_gradient_norm':float(g_h.norm()),'total_field_gradient_norm':float(total.norm()),
                               'cosine_to_original_total':cosine(total,gh+gf+gp+gr),'history_future_cosine':cosine(g_h,gf)})
    # Full patient removal, renormalized exactly by the native loss, reported separately.
    for ranking,key in [('longest_history','history_span_hours'),('highest_sensitivity','past_state_max_singular_value')]:
        removed=sorted(range(n),key=lambda i:patients[i][key],reverse=True)[:5]
        modified={**batch,'mask':batch['mask'].clone(),'valid':batch['valid'].clone()};modified['mask'][removed]=0;modified['valid'][removed]=False
        output=model(batch,sample=False);modified_loss,_=loss_function(model,output,modified,schema,cfg)
        modified_grad=flat_gradient(modified_loss,field).detach()
        conditions.append({'selection':ranking,'removed_count':5,'removed_patient_indices':removed,'normalization':'native per-field denominators recomputed after excluding all targets and Jacobian samples of these patients',
                           'total_field_gradient_norm':float(modified_grad.norm()),'cosine_to_original_total':cosine(modified_grad,gh+gf+gp+gr)})
        del output,modified_loss
    conditions.append({'selection':'future_only','removed_count':0,'normalization':'original future contribution, original joint denominators',
                       'total_field_gradient_norm':float(gf.norm()),'cosine_to_original_total':cosine(gf,gh+gf+gp+gr)})
    if parameter_hash(model)!=fingerprint:raise RuntimeError('Patient diagnostic changed parameters')
    save_checkpoint(folder/'patient_gradient_vectors.pt',{'history':vectors,'encoder_path':encoders,'direct_generation':vectors-encoders,
                                                         'batch_history':gh,'batch_future':gf,'batch_prior':gp,'batch_regularizers':gr,
                                                         'cutoff_states':z.detach(),'past_jacobians':jp,'native_past_jacobians':jp_native,'future_jacobians':jf})
    result={'case':case,'batch_audit':batch_audit(batch,case.get('batch_kind','real_training')),
            'physical_gpu':args.gpu,'parameter_device':str(next(model.parameters()).device),'jacobian_device':str(jp.device),'optimizer_steps':0,
            'parameter_hash':fingerprint,'patients':patients,'counterfactual_conditions':conditions,'module_history_gradient_norms':modules,
            'patient_gradient_sum_relative_error':sum_error,'custom_vs_native_history_gradient_relative_error':native_difference,
            'batch':{'history_gradient_norm':float(gh.norm()),'future_gradient_norm':float(gf.norm()),'history_future_cosine':cosine(gh,gf),
                     'total_field_gradient_norm':float((gh+gf+gp+gr).norm()),
                     'encoder_path_norm':float(encoders.sum(0).norm()),'direct_generation_norm':float((vectors-encoders).sum(0).norm())}}
    atomic(folder/'results.json',result)


if __name__=='__main__':main()
