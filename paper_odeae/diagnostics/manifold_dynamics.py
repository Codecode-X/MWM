"""Frozen-state tangent/metric diagnostics and validation-selected geometric readout probes."""
import argparse
from collections import defaultdict
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from paper_odeae.models import feature_errors
from .manifold_core import (setup,read,atomic,pca,effective_rank,knn,local_pca,project,describe,patient_numbers,
    representatives,limited,load_model,extract_mimic,load_states,provenance,cuda_batch,adni_collate,icu_collate)

STATIC={'sex','age','marital_status','education_years','handedness','apoe4_count'}
DIMS=[6,12,20]


def select_rows(payload,dataset):
    result=[]
    horizons=[1.,2.] if dataset=='adni' else [12.,24.]
    for split,maximum in [('train',512),('validation',1024),('test',2048)]:
        sequences=sorted([s for s in payload['sequences'] if s['split']==split],key=lambda s:__import__('hashlib').sha256(('probe/'+s['subject_id']).encode()).hexdigest())[:maximum]
        for sequence in sequences:
            times=sequence['ages']
            for horizon in horizons:
                found=None
                cuts=list(reversed(range(1,min(3,len(times)-1)+1))) if dataset=='adni' else [int(np.searchsorted(times,24,side='right'))]
                for cutoff in cuts:
                    if cutoff<1 or cutoff>=len(times):continue
                    delta=times-times[cutoff-1]
                    tolerance=.25*horizon if dataset=='adni' else 2.
                    candidates=np.flatnonzero((np.arange(len(times))>=cutoff)&(np.abs(delta-horizon)<=tolerance))
                    if len(candidates):found=(cutoff,int(candidates[np.argmin(np.abs(delta[candidates]-horizon))]));break
                if found:
                    cutoff,target=found;result.append({'sequence':sequence,'subject_id':sequence['subject_id'],'split':split,'cutoff':cutoff,'target':target,
                        'horizon':horizon,'delta':float(times[target]-times[cutoff-1]),'history':cutoff,
                        'coverage':float(sequence['mask'][:cutoff].mean()),'target_coverage':float(sequence['mask'][target].mean()),
                        'age':float(times[cutoff-1]) if dataset=='adni' else float(sequence['admission_age']+times[cutoff-1]/8766.)})
    return result


@torch.no_grad()
def extract_forecasts(model,rows,dataset,schema):
    origin=[];before=[];future=[];values=[];masks=[]
    model.collect_jacobian_states=True
    for start in range(0,len(rows),32):
        group=rows[start:start+32];batch=cuda_batch((adni_collate if dataset=='adni' else icu_collate)([(r['sequence'],r['cutoff']) for r in group]))
        captured=[];hook=model.update.register_forward_pre_hook(lambda module,args:captured.append(args[0][:,:28].detach()))
        try:out=model(batch,sample=False)
        finally:hook.remove()
        index=torch.arange(len(group),device='cuda');targets=torch.tensor([r['target'] for r in group],device='cuda')
        origin.append(out['mu'].double());before.append(torch.stack(captured,1)[index,batch['cutoff']-1].double())
        future.append(out['jacobian_states'][index,targets].double());values.append(batch['values'][index,targets]);masks.append(batch['mask'][index,targets])
    return [torch.cat(parts) for parts in [origin,before,future,values,masks]]


def normalized_std(payload):
    result=[]
    for j,field in enumerate(payload['schema']):
        if field['kind']!='numerical':result.append(1.);continue
        values=np.concatenate([s['values'][:,j][s['mask'][:,j].astype(bool)] for s in payload['sequences'] if s['split']=='train']).astype(np.float64)
        result.append(max(float(values.std()),1e-6))
    return result


@torch.no_grad()
def decoder_metric(model,states,std):
    field_count=sum(field['name'] not in STATIC for field in model.schema);metrics=[]
    w1=model.decoder[0].weight.double();b1=model.decoder[0].bias.double();w2=model.decoder[2].weight.double()
    for part in states.split(128):
        x=part.double();activation=torch.tanh(x@w1.T+b1);jac=(w2[None]*(1-activation.square())[:,None])@w1
        logits=model.decoder(part.float()).double();g=x.new_zeros(len(x),28,28);at=0
        for j,field in enumerate(model.schema):
            width=field.get('classes',1);derivative=jac[:,at:at+width]
            if field['name'] not in STATIC:
                if field['kind']=='numerical':g+=derivative.transpose(1,2)@derivative/(std[j]**2)
                else:
                    p=logits[:,at:at+width].softmax(-1);fisher=torch.diag_embed(p)-p[:,:,None]*p[:,None,:]
                    g+=derivative.transpose(1,2)@fisher@derivative
            at+=width
        metrics.append(g/max(field_count,1))
    return torch.cat(metrics)


def score(errors,masks,indices,schema,std):
    e=errors[indices].double();m=masks[indices].double();counts=m.sum(0);means=e.sum(0)/counts.clamp_min(1)
    numeric=torch.tensor([f['kind']=='numerical' for f in schema],device='cuda');present=counts>0
    mse=means[numeric&present].sum();ce=means[~numeric&present].sum()
    standardized=means[numeric&present]/torch.tensor(std,device='cuda',dtype=torch.float64)[numeric&present].square()
    return {'error':float(mse+ce),'numeric_mse_sum':float(mse),'categorical_ce_sum':float(ce),'mean_standardized_numeric_mse':float(standardized.mean()),
        'rows':len(e),'observed_scalars':int(counts.sum()),'per_field':{f['name']:{'loss':float(means[j]),'observations':int(counts[j])} for j,f in enumerate(schema) if present[j]}}


def bootstrap_delta(candidate,baseline,masks,pids,repeats=500):
    _,inverse=torch.unique(pids,return_inverse=True);n=int(inverse.max())+1;f=masks.shape[1]
    def aggregate(x):
        result=x.new_zeros(n,f,dtype=torch.float64);result.index_add_(0,inverse,x.double());return result
    difference=aggregate(candidate-baseline);counts=aggregate(masks)
    generator=torch.Generator(device='cuda').manual_seed(582);values=[]
    for _ in range(10):
        draws=torch.randint(n,(repeats//10,n),device='cuda',generator=generator)
        values.append((difference[draws].sum(1)/counts[draws].sum(1).clamp_min(1)).sum(1))
    return torch.quantile(torch.cat(values),torch.tensor([.025,.975],device='cuda',dtype=torch.float64)).tolist()


def associations(rows,normal_fraction,drift,error):
    from scipy.stats import spearmanr,rankdata
    results={}
    for horizon in sorted({r['horizon'] for r in rows}):
        indices=[i for i,r in enumerate(rows) if r['split']=='test' and r['horizon']==horizon]
        if len(indices)<12:continue
        y=error[indices].cpu().numpy();cov=np.asarray([[rows[i][key] for key in ['age','history','coverage','target_coverage','delta']] for i in indices])
        design=np.column_stack([np.ones(len(indices))]+[rankdata(cov[:,j]) for j in range(cov.shape[1])])
        yr=rankdata(y);yr-=design@np.linalg.lstsq(design,yr,rcond=None)[0]
        info={'test_patients':len(indices)}
        for key,values in [('normal_velocity',normal_fraction),('future_support_residual',drift)]:
            x=values[indices].cpu().numpy();rank=rankdata(x);residual=rank-design@np.linalg.lstsq(design,rank,rcond=None)[0]
            info[key]={'spearman':float(spearmanr(x,y).statistic),'partial_spearman':float(np.corrcoef(residual,yr)[0,1])}
        results[str(horizon)]=info
    return results


def plots(out,result,rows,arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','svg','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    test=np.array([r['split']=='test' for r in rows]);fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    dims=DIMS;axes[0].boxplot([arrays[f'normal_velocity_d{d}'][test] for d in dims],tick_labels=[str(d) for d in dims],showfliers=False)
    axes[0].plot(np.arange(1,4),[1-d/28 for d in dims],'k--',label='Isotropic direction expectation');axes[0].legend(fontsize=7)
    axes[0].set(xlabel='Local tangent dimensions',ylabel='Normal velocity energy fraction',title='ODE direction relative to local PCA plane')
    horizons=sorted({r['horizon'] for r in rows})
    for h in horizons:
        selected=test&np.array([r['horizon']==h for r in rows]);axes[1].scatter(arrays['future_residual_d12'][selected],arrays['row_error'][selected],s=9,alpha=.3,label=str(h))
    axes[1].set(xlabel='Future normal residual / neighbor radius',ylabel='Mean observed-field loss',title='Support residual and forecast error');axes[1].legend(title='Horizon',fontsize=8)
    axes[2].hist(arrays['metric_effective_rank'],bins=25,color='#4c78a8');axes[2].set(xlabel='Decoder metric effective rank',ylabel='Test origins',title='Clinical metric spectrum')
    save(fig,'dynamics_evidence')
    fig,axes=plt.subplots(1,2,figsize=(13,5));names=list(result['probes']);base=result['probes']['baseline']['test']['error']
    axes[0].barh(np.arange(len(names)),[result['probes'][n]['test']['error']-base for n in names],color=['#4c78a8' if 'local' in n else '#aaaaaa' for n in names]);axes[0].set_yticks(np.arange(len(names)),names);axes[0].axvline(0,color='black',lw=.7)
    axes[0].set(xlabel='Test error change from baseline',title='Frozen readout interventions')
    for i,(key,selected) in enumerate(result['validation_selected'].items()):
        delta=selected['test_error_change'];lo,hi=selected['paired_patient_bootstrap95'];axes[1].errorbar(delta,i,xerr=[[max(delta-lo,0)],[max(hi-delta,0)]],fmt='o',capsize=4,label=selected['variant'])
    axes[1].axvline(0,color='black',lw=.7);axes[1].set_yticks([0,1],list(result['validation_selected']));axes[1].set(xlabel='Paired test error difference with 95% CI',title='Choice made using validation only');axes[1].legend(fontsize=7)
    save(fig,'projection_probe')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=read(args.config);setup();start=time.monotonic();out=Path(cfg['job_output']);out.mkdir(parents=True,exist_ok=True)
    model,saved=load_model(cfg);payload=pickle.loads(Path(cfg['data']).read_bytes());schema=payload['schema'];std=normalized_std(payload)
    if cfg['dataset']=='mimic':extract_mimic(cfg,model,payload,'heldout')
    states,velocities,metadata=load_states(cfg);pids,patient_map=patient_numbers(metadata)
    primary=representatives(metadata,'train',0,cfg['dataset']=='mimic');mean,basis,eig,ratio=pca(states[primary])
    references=torch.tensor([i for i,row in enumerate(metadata) if row['split']=='train'],device='cuda');reference=states[references];reference_pid=pids[references]
    rows=select_rows(payload,cfg['dataset']);origin,before,future,values,masks=extract_forecasts(model,rows,cfg['dataset'],schema)
    row_pid=torch.tensor([patient_map[r['subject_id']] for r in rows],device='cuda');n=len(rows)
    cached={(r['subject_id'],r['prefix']):i for i,r in enumerate(metadata)};difference=0.
    for i,r in enumerate(rows):
        key=(r['subject_id'],r['cutoff'])
        if key in cached:difference=max(difference,float((origin[i]-states[cached[key]]).abs().max()))
    if difference>1e-4:raise AssertionError(f'Cached/native encoder mismatch: {difference}')
    print(f'{cfg["dataset"]}: {n} patient-horizon probes encoded; max cache difference {difference}',flush=True)
    combined=torch.cat([origin,before,future]);combined_pid=row_pid.repeat(3)
    distance,neighbors=knn(combined,reference,80,combined_pid,reference_pid,distinct=True)
    center,local_basis,spectrum=local_pca(reference,neighbors)
    center0,center_pre,center_future=center[:n],center[n:2*n],center[2*n:]
    basis0,basis_pre,basis_future=local_basis[:n],local_basis[n:2*n],local_basis[2*n:]
    field=model.forward_ode(origin.float()).double();jump=origin-before
    arrays={};normal={};residuals={}
    for d in DIMS:
        b=basis0[:,:,:d];tangent=(b@(b.transpose(1,2)@field[:,:,None])).squeeze(-1)
        normal[d]=(field-tangent).square().sum(1)/field.square().sum(1).clamp_min(1e-30)
        normal_jump=jump-(b@(b.transpose(1,2)@jump[:,:,None])).squeeze(-1)
        for label,z,mu,b_,radius in [('origin',origin,center0,basis0,distance[:n,-1]),('pre_update',before,center_pre,basis_pre,distance[n:2*n,-1]),('future',future,center_future,basis_future,distance[2*n:,-1])]:
            residual=(z-project(z,mu,b_,d)).norm(dim=-1)/radius.clamp_min(1e-12);arrays[f'{label}_residual_d{d}']=residual
            if label=='future':residuals[d]=residual
        arrays[f'normal_velocity_d{d}']=normal[d];arrays[f'normal_update_d{d}']=normal_jump.square().sum(1)/jump.square().sum(1).clamp_min(1e-30)
        # Rotating tangent planes are a direct curvature proxy at the forecast endpoints.
        b2=basis_future[:,:,:d];overlap=(b.transpose(1,2)@b2).square().sum((1,2))/d
        arrays[f'tangent_rotation_d{d}']=(1-overlap).clamp_min(0).sqrt()
    strategies={'baseline':future,'mean_shrink_025':.75*future+.25*mean}
    for d in DIMS:
        strategies[f'global_d{d}']=project(future,mean,basis,d)
        local=project(future,center_future,basis_future,d)
        strategies[f'local_d{d}_a025']=future+.25*(local-future);strategies[f'local_d{d}_a100']=local
    errors={}
    for name,z in strategies.items():
        logits=[]
        for part in z.split(256):logits.append(model.decoder(part.float()))
        errors[name]=feature_errors(torch.cat(logits)[:,None],values[:,None],masks[:,None],schema)[:,0]
    split_ids={split:torch.tensor([i for i,r in enumerate(rows) if r['split']==split],device='cuda') for split in ['train','validation','test']}
    result={'probes':{},'validation_selected':{},'tangent_summary':{},'cache_state_max_difference':difference,'counts':{},'metric':{}}
    for split,ids in split_ids.items():result['counts'][split]={'rows':len(ids),'patients':len(set(rows[i]['subject_id'] for i in ids.tolist()))}
    for name,error in errors.items():
        result['probes'][name]={split:score(error,masks,ids,schema,std) for split,ids in split_ids.items()}
        result['probes'][name]['test_by_horizon']={str(h):score(error,masks,torch.tensor([i for i,r in enumerate(rows) if r['split']=='test' and r['horizon']==h],device='cuda'),schema,std) for h in sorted({r['horizon'] for r in rows})}
    # Predefined candidate families; selection never reads test errors.
    for family,names in [('global_or_shrink',['baseline','mean_shrink_025']+[f'global_d{d}' for d in DIMS]),('local',['baseline']+[name for name in strategies if name.startswith('local')])]:
        selected=min(names,key=lambda name:result['probes'][name]['validation']['error']);ids=split_ids['test']
        result['validation_selected'][family]={'variant':selected,'validation_error':result['probes'][selected]['validation']['error'],
             'test_error':result['probes'][selected]['test']['error'],'test_error_change':result['probes'][selected]['test']['error']-result['probes']['baseline']['test']['error'],
             'paired_patient_bootstrap95':bootstrap_delta(errors[selected][ids],errors['baseline'][ids],masks[ids],row_pid[ids])}
    for h in sorted({r['horizon'] for r in rows}):
        ids=torch.tensor([i for i,r in enumerate(rows) if r['split']=='test' and r['horizon']==h],device='cuda')
        calibration=torch.tensor([i for i,r in enumerate(rows) if r['split']=='train' and r['horizon']==h],device='cuda')
        result['tangent_summary'][str(h)]={}
        for d in DIMS:
            threshold=float(torch.quantile(arrays[f'origin_residual_d{d}'][calibration],.95))
            result['tangent_summary'][str(h)][str(d)]={'normal_velocity':describe(normal[d][ids]),'normal_update':describe(arrays[f'normal_update_d{d}'][ids]),
                'origin_residual':describe(arrays[f'origin_residual_d{d}'][ids]),'pre_update_residual':describe(arrays[f'pre_update_residual_d{d}'][ids]),
                'future_residual':describe(residuals[d][ids]),'tangent_rotation':describe(arrays[f'tangent_rotation_d{d}'][ids]),
                'training_origin_q95':threshold,'origin_above_threshold_fraction':float((arrays[f'origin_residual_d{d}'][ids]>threshold).double().mean()),
                'future_above_threshold_fraction':float((residuals[d][ids]>threshold).double().mean())}
    unique_test={}
    for i in split_ids['test'].tolist():unique_test.setdefault(rows[i]['subject_id'],i)
    metric_ids=limited(torch.tensor(list(unique_test.values()),device='cuda'),512,37)
    g=decoder_metric(model,origin[metric_ids],std);eigen=torch.linalg.eigvalsh(g).clamp_min(0)
    eps=eigen.sum(1)/28*1e-4;condition=(eigen[:,-1]+eps)/(eigen[:,0]+eps).clamp_min(1e-20)
    rank=effective_rank(eigen);unit_trace=g/g.diagonal(dim1=-2,dim2=-1).sum(-1)[:,None,None].clamp_min(1e-30)
    gn=decoder_metric(model,reference[neighbors[metric_ids,0]],std);gn=gn/gn.diagonal(dim1=-2,dim2=-1).sum(-1)[:,None,None].clamp_min(1e-30)
    result['metric']={'origins':len(metric_ids),'effective_rank':describe(rank),'ridge_condition_number':describe(condition),
         'nearest_patient_normalized_metric_change':describe((unit_trace-gn).flatten(1).norm(dim=-1)),
         'ridge_rule':'epsilon=1e-4*trace(G)/28','numeric_metric':'fixed unit Gaussian variance in training-standard-deviation coordinates',
         'categorical_metric':'softmax categorical Fisher metric','excluded_static_fields':sorted(STATIC)}
    # Independent automatic-differentiation check of the exact decoder Jacobian.
    z=origin[metric_ids[:1]].float().detach().requires_grad_(True)
    with torch.enable_grad():autograd_jac=torch.autograd.functional.jacobian(lambda q:model.decoder(q)[0],z)[...,0,:]
    act=torch.tanh(model.decoder[0](z));analytic=(model.decoder[2].weight[None]*(1-act.square())[:,None])@model.decoder[0].weight
    torch.testing.assert_close(analytic[0],autograd_jac,rtol=2e-5,atol=2e-5)
    result['metric']['decoder_jacobian_check_max_difference']=float((analytic[0]-autograd_jac).abs().max())
    row_error=errors['baseline'].sum(1)/masks.sum(1).clamp_min(1);arrays['row_error']=row_error;arrays['metric_effective_rank']=rank
    result['associations']=associations(rows,normal[12],residuals[12],row_error)
    result['provenance']=provenance(cfg,'dynamics');result['provenance'].update(elapsed_seconds=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated())
    atomic(out/'summary.json',result)
    metadata_rows=[{key:value for key,value in r.items() if key!='sequence'} for r in rows];atomic(out/'forecast_rows.json',metadata_rows)
    arrays={key:value.detach().cpu().numpy() for key,value in arrays.items()};np.savez_compressed(out/'dynamics_arrays.npz',**arrays)
    plots(out,result,rows,arrays);atomic(out/'complete.json',{'status':'complete'})
    print(f'{cfg["dataset"]} dynamics complete',flush=True)


if __name__=='__main__':
    with torch.no_grad():main()
