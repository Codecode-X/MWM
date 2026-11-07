"""Continuous Disease Stage and Progression Analysis; frozen checkpoint, CUDA only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np
import torch

from paper_odeae.models import paper_model_from_config
from paper_odeae.data import collate
from paper_odeae.train import device_batch
from .state_space import ROOT, encode
from .clinical_velocity import write_csv

SOURCE=ROOT/'paper_odeae/results/shared_matched_state_space'
DEFAULT=ROOT/'paper_odeae/results/shared_matched_continuous_stage'
HORIZONS=[12,24,36]


def diagnosis_slice(schema):
    at=0
    for field in schema:
        width=field.get('classes',1)
        if field['name']=='diagnosis':
            if field['kind']!='categorical' or width!=3:raise ValueError('Three existing diagnosis logits required')
            return slice(at,at+3)
        at+=width
    raise ValueError('Existing diagnosis head missing')


def probabilities_and_stage(model,z):
    if z.device.type!='cuda':raise RuntimeError('CUDA required for model diagnostics')
    p=model.decoder(z)[...,diagnosis_slice(model.schema)].softmax(-1)
    return p,p[...,1]+2*p[...,2]


def instantaneous_stage(model,z):
    """Autodiff of the existing Softmax head, with gradients only in z."""
    with torch.enable_grad():
        state=z.detach().requires_grad_(True)
        p,s=probabilities_and_stage(model,state)
        gradient=torch.autograd.grad(s.sum(),state)[0]
        velocity=model.forward_ode(state)
        rate=(gradient*velocity).sum(-1)
    return p.detach(),s.detach(),rate.detach(),gradient.detach(),velocity.detach()


def calendar_month(date):
    return int(str(date)[:4])*12+int(str(date)[5:7])


def mci_origin(sequence,dx):
    """Earliest observed MCI, provided no already-known Dementia precedes it."""
    for i in range(len(sequence['ages'])):
        if not sequence['mask'][i,dx]:continue
        disease=int(sequence['values'][i,dx])
        if disease==2:return None
        if disease==1:return i
    return None


def conversion_label(sequence,origin,dx,horizon,support_months=6):
    """Symmetric endpoint support, observed first-conversion and persistent MCI."""
    start=calendar_month(sequence['dates'][origin])
    future=[(i,calendar_month(sequence['dates'][i])-start,int(sequence['values'][i,dx]))
            for i in range(origin+1,len(sequence['ages'])) if sequence['mask'][i,dx]]
    events=[r for r in future if r[2]==2 and r[1]<=horizon]
    support=[r for r in future if horizon<=r[1]<=horizon+support_months]
    result={'label':None,'reason':'no_endpoint_support','support_visit':None,'support_month':None,
            'first_dementia_month':next((r[1] for r in future if r[2]==2),None)}
    if not support:return result
    end=support[0]
    result.update(support_visit=end[0]+1,support_month=end[1])
    if events:result.update(label=1,reason='converter')
    elif all(r[2]==1 for r in future if r[0]<=end[0]):result.update(label=0,reason='stable_mci')
    else:result['reason']='other_diagnosis_or_post_window_dementia'
    return result


@torch.no_grad()
def rollout(model,states):
    """Monthly free integration for five years, using the original adaptive solver."""
    all_states=[];all_prob=[];all_scores=[]
    for at in range(0,len(states),128):
        state=states[at:at+128];zs=[state];p,s=probabilities_and_stage(model,state);ps=[p];ss=[s]
        delta=state.new_full((len(state),),1/12)
        for month in range(1,61):
            state=model.solve(model.forward_ode,state,delta)
            p,s=probabilities_and_stage(model,state);zs.append(state);ps.append(p);ss.append(s)
        all_states.append(torch.stack(zs,1));all_prob.append(torch.stack(ps,1));all_scores.append(torch.stack(ss,1))
        print(f'Free ODE rollout: {min(at+128,len(states))}/{len(states)} states',flush=True)
    return torch.cat(all_states),torch.cat(all_prob),torch.cat(all_scores)


@torch.no_grad()
def run(args):
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU fallback disabled')
    torch.set_num_threads(1)
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    source=Path(args.source).resolve();info=json.loads((source/'statistics.json').read_text())
    checkpoint=Path(info['checkpoint']);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(checkpoint)==info['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cuda',weights_only=False);cfg=saved['config']
    assert cfg['latent']==6 and cfg['shared_vector_field'] and cfg['encoder_direction']=='forward'
    payload=pickle.loads(Path(cfg['data']).read_bytes());assert sha(cfg['data'])==info['dataset_sha256']
    model=paper_model_from_config(saved['schema'],cfg).cuda().eval();model.load_state_dict(saved['state_dict'])
    assert model.backward_ode is model.forward_ode
    for parameter in model.parameters():parameter.requires_grad_(False)
    schema=model.schema;dx=next(i for i,f in enumerate(schema) if f['name']=='diagnosis')
    cached=np.load(source/'states.npz');all_meta=json.loads((source/'metadata.json').read_text())
    indices=[i for i,r in enumerate(all_meta) if r['split']=='test'];metadata=[all_meta[i] for i in indices]
    z=torch.tensor(cached['states'][indices],device='cuda')
    mean=torch.tensor(cached['pca_mean'],device='cuda');basis=torch.tensor(cached['pca_basis'],device='cuda')
    xy=torch.tensor(cached['coordinates'][indices],device='cuda')
    torch.testing.assert_close((z-mean)@basis,xy,rtol=1e-6,atol=1e-6)
    probs,stage,rate,grad,velocity=instantaneous_stage(model,z)
    torch.testing.assert_close(velocity,torch.tensor(cached['velocities'][indices],device='cuda'),rtol=1e-5,atol=1e-6)
    projected=velocity@basis
    assert bool(torch.isfinite(rate).all()) and bool(((stage>=0)&(stage<=2)).all())
    lookup={(r['subject_id'],r['prefix']):i for i,r in enumerate(metadata)}
    sequences=sorted([s for s in payload['sequences'] if s['split']=='test'],key=lambda s:str(s['subject_id']))
    # Explicitly truncate before encoding: future values, dates and ages are absent.
    sample=metadata[::max(1,len(metadata)//16)][:16]
    seq_lookup={str(s['subject_id']):s for s in sequences}
    original=[];truncated=[];expected=[]
    for r in sample:
        seq=seq_lookup[r['subject_id']];cut=r['prefix'];original.append((seq,cut))
        truncated.append(({**seq,**{k:seq[k][:cut] for k in ['ages','values','mask','dates']}},cut))
        expected.append(lookup[(r['subject_id'],cut)])
    full_batch=device_batch(collate(original));cut_batch=device_batch(collate(truncated))
    full_state=encode(model,full_batch);cut_state=encode(model,cut_batch)
    torch.testing.assert_close(full_state,cut_state,rtol=0,atol=0)
    cached_error=float((cut_state-z[expected]).abs().max());torch.testing.assert_close(cut_state,z[expected],rtol=1e-4,atol=1e-4)
    # Independently check the autodiff rate with directional finite differences in float64.
    import copy
    double=copy.deepcopy(model).double();zz=z[::max(1,len(z)//32)][:32].double()
    _,_,check_rate,_,vv=instantaneous_stage(double,zz);epsilon=1e-4
    _,positive=probabilities_and_stage(double,zz+epsilon*vv)
    _,negative=probabilities_and_stage(double,zz-epsilon*vv)
    fd=(positive-negative)/(2*epsilon)
    torch.testing.assert_close(fd,check_rate,rtol=1e-5,atol=1e-8)
    fd_error=float((fd-check_rate).abs().max());del double
    print(f'Autodiff and history isolation verified; {len(z)} test states',flush=True)
    future_z,future_p,future_s=rollout(model,z)
    # Check repeated monthly solving against a direct original-solver integration.
    direct_errors={}
    check_ids=torch.arange(0,len(z),max(1,len(z)//16),device='cuda')[:16]
    for horizon in [6,12,24,36,60]:
        direct=model.solve(model.forward_ode,z[check_ids],z.new_full((len(check_ids),),horizon/12))
        error=float((direct-future_z[check_ids,horizon]).abs().max());direct_errors[str(horizon)]=error
        torch.testing.assert_close(direct,future_z[check_ids,horizon],rtol=3e-3,atol=3e-4)
    latest=np.array([lookup[(str(s['subject_id']),len(s['ages']))] for s in sequences],dtype=np.int64)
    latest_set=set(latest.tolist())
    origin_rows=[];cohorts=[]
    for seq in sequences:
        origin=mci_origin(seq,dx)
        if origin is None:continue
        idx=lookup[(str(seq['subject_id']),origin+1)]
        origin_rows.append({'subject_id':str(seq['subject_id']),'origin_visit':origin+1,'origin_date':str(seq['dates'][origin]),'state_index':idx})
        for horizon in HORIZONS:
            item=conversion_label(seq,origin,dx,horizon)
            cohorts.append({**origin_rows[-1],'horizon_months':horizon,**item})
    zz=z.cpu().numpy();pp=probs.cpu().numpy();ss=stage.cpu().numpy();rr=rate.cpu().numpy()
    fs=future_s.cpu().numpy();fp=future_p.cpu().numpy()
    state_rows=[];forecast_rows=[]
    for i,r in enumerate(metadata):
        state_rows.append({**r,'state_index':i,'pc1':float(xy[i,0]),'pc2':float(xy[i,1]),
                           'p_cn':float(pp[i,0]),'p_mci':float(pp[i,1]),'p_dementia':float(pp[i,2]),
                           'stage':float(ss[i]),'instantaneous_stage_per_year':float(rr[i]),
                           'is_latest_patient_state':i in latest_set})
        for horizon in [0,6,12,24,36,60]:
            forecast_rows.append({'subject_id':r['subject_id'],'prefix':r['prefix'],'state_index':i,'month':horizon,
                                  'p_cn':float(fp[i,horizon,0]),'p_mci':float(fp[i,horizon,1]),'p_dementia':float(fp[i,horizon,2]),
                                  'stage':float(fs[i,horizon]),'delta_stage':float(fs[i,horizon]-ss[i]),
                                  'mean_stage_per_year':float((fs[i,horizon]-ss[i])/(horizon/12)) if horizon else None,
                                  'is_latest_patient_state':i in latest_set})
    for row in cohorts:
        idx=row['state_index'];h=row['horizon_months']
        row.update(stage_0=float(ss[idx]),instantaneous_speed=float(rr[idx]),delta_stage=float(fs[idx,h]-ss[idx]),
                   mean_speed=float((fs[idx,h]-ss[idx])/(h/12)),future_dementia_probability=float(fp[idx,h,2]),current_dementia_probability=float(pp[idx,2]))
    write_csv(out/'test_states.csv',state_rows);write_csv(out/'forecasts.csv',forecast_rows)
    write_csv(out/'mci_origins.csv',origin_rows);write_csv(out/'conversion_cohorts.csv',cohorts)
    np.savez_compressed(out/'continuous_stage.npz',states=zz,coordinates=xy.cpu().numpy(),projected_velocities=projected.cpu().numpy(),
                        probabilities=pp,stage=ss,instantaneous_rate=rr,stage_gradient=grad.cpu().numpy(),
                        months=np.arange(61),future_states=future_z.cpu().numpy(),future_probabilities=fp,future_stage=fs,
                        latest_indices=latest,pca_mean=cached['pca_mean'],pca_basis=cached['pca_basis'])
    assert all(torch.equal(value,saved['state_dict'][name]) for name,value in model.state_dict().items())
    provenance={'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'checkpoint_epoch':saved['epoch'],
                'dataset':cfg['data'],'dataset_sha256':sha(cfg['data']),'source_states':str(source),
                'source_states_sha256':sha(source/'states.npz'),'source_metadata_sha256':sha(source/'metadata.json'),
                'script_sha256':sha(__file__),'physical_gpu':args.gpu,'gpu_name':torch.cuda.get_device_name(),
                'devices':{'model':str(next(model.parameters()).device),'states':str(z.device),'autodiff':str(rate.device),
                           'ode_rollout':str(future_z.device),'pca_projection':str(basis.device)},
                'parameters':sum(p.numel() for p in model.parameters()),'latent_dimensions':6,'new_learnable_parameters':0,
                'training_performed':False,'checkpoint_parameters_unchanged':True,'test_patients':len(sequences),'test_states':len(z),
                'eligible_first_mci_patients':len(origin_rows),'pca_fit':'existing train-only PCA; basis and signs reused exactly',
                'pca_state_variance_retained':info['state_variance_retained_2d'],'diagnosis_output_slice':[diagnosis_slice(schema).start,diagnosis_slice(schema).stop],
                'head_class_order':['CN','MCI','Dementia'],'derivative_method':'torch.autograd.grad(s.sum(), z), contracted with original f(z)',
                'double_finite_difference_max_error':fd_error,'truncated_history_vs_cached_max_error':cached_error,
                'future_information_isolation':'full record vs history-only record: bitwise identical endpoint encoding',
                'monthly_vs_direct_ode_max_state_errors':direct_errors,'integration_step_output':'monthly; original adaptive Tsit5 inside every interval',
                'ode_time_unit':'year; requested month/12','label_time_unit':'calendar-month difference on existing quarter-end visit dates',
                'cohort_origin':'earliest observed MCI, no previously observed Dementia, one origin per patient across all horizons',
                'cohort_endpoint_support':'both groups require diagnosis at horizon through horizon+6 calendar months',
                'cohort_converter':'first observed Dementia after origin at or before horizon',
                'cohort_stable':'all observed diagnoses after origin through first supporting visit are MCI',
                'no_future_updates':True}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(out),'patients':len(sequences),'mci_origins':len(origin_rows),'physical_gpu':args.gpu}),flush=True)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',choices=['0','1','2','3'],default='3')
    parser.add_argument('--source',default=str(SOURCE));parser.add_argument('--output',default=str(DEFAULT))
    args=parser.parse_args();out=run(args)
    from .continuous_stage_report import generate
    generate(out)


if __name__=='__main__':main()
