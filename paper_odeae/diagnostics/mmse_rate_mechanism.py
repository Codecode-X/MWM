"""Decompose clinical decline into speed, decoder gain and alignment on CUDA."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from paper_odeae.models import paper_model_from_config
from .state_space import ROOT

RESULTS=ROOT/'paper_odeae/results'


def write_csv(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def mechanism(model,z,head,scale):
    if z.device.type!='cuda':raise RuntimeError('Clinical decomposition requires CUDA')
    decoder=model.decoder;hidden=torch.tanh(decoder[0](z));derivative=1-hidden.square()
    grad=(derivative*decoder[2].weight[head])@decoder[0].weight*scale
    velocity=model.forward_ode(z);speed=velocity.norm(dim=-1);gain=grad.norm(dim=-1)
    rate=(grad*velocity).sum(-1);capacity=speed*gain
    cosine=rate/capacity.clamp_min(1e-12)
    weights=decoder[2].weight[head].abs()*decoder[0].weight.norm(dim=1)
    active_gain=(derivative*weights).sum(-1)/weights.sum().clamp_min(1e-12)
    saturation=((hidden.abs()>.95)*weights).sum(-1)/weights.sum().clamp_min(1e-12)
    return {'speed':speed,'gradient_norm':gain,'cosine':cosine,'rate':rate,'capacity':capacity,
            'mmse_tangent_energy_fraction':1-cosine.square(),
            'weighted_tanh_derivative':active_gain,'weighted_saturated_fraction':saturation,
            'angle_degrees':torch.acos(cosine.clamp(-1,1))*180/torch.pi}


def group_summary(rows):
    result=[]
    for dx,label in [(0,'CN'),(1,'MCI'),(2,'Dementia'),(None,'Unknown')]:
        use=[r for r in rows if r['diagnosis']==dx]
        if not use:continue
        record={'diagnosis':label,'intervals':len(use),'patients':len({r['subject_id'] for r in use})}
        fields=['speed','gradient_norm','cosine','rate','capacity','angle_degrees','mmse_tangent_energy_fraction',
                'weighted_tanh_derivative','weighted_saturated_fraction','observed_rate','predicted_rate',
                'path_budget','path_alignment','quadrature_rate','start_prediction','observed_start']
        for key in fields:
            values=np.array([r[key] for r in use]);record[key+'_mean']=float(values.mean());record[key+'_median']=float(np.median(values))
        record['effective_instant_cosine']=record['rate_mean']/record['capacity_mean']
        record['effective_path_decline_alignment']=-record['quadrature_rate_mean']/record['path_budget_mean']
        decline=[r for r in use if r['observed_rate']<0]
        record['observed_decline_intervals']=len(decline)
        record['decline_within_path_budget_fraction']=float(np.mean([abs(r['observed_rate'])<=r['path_budget'] for r in decline])) if decline else None
        result.append(record)
    return result


def enrich_summary(summary,rows):
    """Patient-cluster uncertainty and additive gap accounting on fixed paths."""
    from collections import defaultdict
    summary['groups']=group_summary(rows)
    dementia=[r for r in rows if r['diagnosis']==2]
    sensitivity=[]
    for gap in [0.,.5,.9]:
        use=[r for r in dementia if r['years']>=gap]
        g=group_summary(use)[0]
        sensitivity.append({'minimum_interval_years':gap,**g})
    summary['dementia_interval_sensitivity']=sensitivity
    patient=defaultdict(list)
    for r in dementia:patient[r['subject_id']].append(r)
    keys=['real_decline','model_decline','budget','total_gap','direction_projection_gap','magnitude_budget_gap']
    sums=[];counts=[]
    for values in patient.values():
        arrays=[]
        for r in values:
            real=-r['observed_rate'];pred=-r['quadrature_rate'];budget=r['path_budget']
            arrays.append([real,pred,budget,real-pred,budget-pred,real-budget])
        sums.append(np.array(arrays).sum(0));counts.append(len(values))
    sums=np.array(sums);counts=np.array(counts);mean=sums.sum(0)/counts.sum()
    rng=np.random.default_rng(0);ids=rng.integers(len(patient),size=(1000,len(patient)))
    boot=sums[ids].sum(1)/counts[ids].sum(1)[:,None]
    summary['dementia_gap_decomposition']={'patients':len(patient),'intervals':len(dementia),
        'values':{key:float(value) for key,value in zip(keys,mean)},
        'patient_bootstrap_95ci':{key:np.quantile(boot[:,i],[.025,.975]).tolist() for i,key in enumerate(keys)},
        'direction_projection_share':float(mean[4]/mean[3]),'magnitude_budget_share':float(mean[5]/mean[3]),
        'definition':'fixed original paths: real decline - model decline = (mean speed*gain budget - model decline) + (real decline - budget)'}
    return summary


def plots(out,rows,summary,coords,metadata,all_values):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','svg','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,10))
    fields=[('speed','Latent speed ||f||'),('gradient_norm','MMSE head gradient ||grad g||'),('cosine','Cosine with increasing-MMSE direction'),
            ('capacity','Instantaneous magnitude budget ||f|| ||grad g||'),('rate','Predicted instantaneous MMSE points/year'),('path_alignment','Path-averaged decline alignment')]
    for ax,(key,title) in zip(axes.flat,fields):
        groups=[[r[key] for r in rows if r['diagnosis']==dx] for dx in [0,1,2]]
        ax.boxplot(groups,tick_labels=['CN','MCI','Dementia'],showfliers=False);ax.set_title(title)
        if key in ['cosine','rate']:ax.axhline(0,c='#aaa',lw=1)
    save(fig,'mechanism_by_diagnosis')
    fig,axes=plt.subplots(2,2,figsize=(13,10));test=np.array([r['split']=='test' for r in metadata])
    for ax,key,title in [(axes[0,0],'speed','Latent speed'),(axes[0,1],'gradient_norm','MMSE gradient norm'),(axes[1,0],'cosine','Velocity / MMSE-gradient cosine'),(axes[1,1],'rate','Predicted MMSE points/year')]:
        sc=ax.scatter(*coords[test].T,c=all_values[key][test],cmap='coolwarm' if key in ['cosine','rate'] else 'viridis',s=8,alpha=.6,rasterized=True)
        fig.colorbar(sc,ax=ax);ax.set(title=title,xlabel='Own train-fitted PC1',ylabel='Own train-fitted PC2')
    save(fig,'mechanism_state_space')
    fig,axes=plt.subplots(1,3,figsize=(17,5));d=[r for r in rows if r['diagnosis']==2]
    for ax,key,title in [(axes[0],'speed','Speed'),(axes[1],'gradient_norm','Head sensitivity'),(axes[2],'path_alignment','Path decline alignment')]:
        ax.scatter([r[key] for r in d],[-r['observed_rate'] for r in d],s=12,alpha=.45,label='Observed decline')
        ax.scatter([r[key] for r in d],[-r['predicted_rate'] for r in d],s=12,alpha=.45,label='Model decline')
        ax.set(title='Dementia | '+title,xlabel=title,ylabel='Decline magnitude (MMSE points/year)');ax.legend(fontsize=8)
    save(fig,'dementia_mechanism_vs_decline')


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--variant',choices=['shared_matched','weak_prior'],required=True)
    parser.add_argument('--gpu',choices=['0','1','2','3'],required=True);args=parser.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(1)
    source=RESULTS/(args.variant+'_state_space');clinical=RESULTS/(args.variant+'_clinical_velocity');out=RESULTS/(args.variant+'_mmse_mechanism');out.mkdir(exist_ok=True)
    info=json.loads((source/'statistics.json').read_text());cached=np.load(source/'states.npz');meta=json.loads((source/'metadata.json').read_text())
    checkpoint=Path(info['checkpoint']);assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==info['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cuda',weights_only=False);model=paper_model_from_config(saved['schema'],saved['config']).cuda().eval();model.load_state_dict(saved['state_dict'])
    j=next(i for i,f in enumerate(saved['schema']) if f['name']=='mmse');head=sum(f.get('classes',1) for f in saved['schema'][:j]);scale=float(saved['metadata']['scale'][j]);offset=float(saved['metadata']['offset'][j])
    z=torch.tensor(cached['states'],device='cuda');parts=[]
    for chunk in z.split(512):parts.append(mechanism(model,chunk,head,scale))
    values={key:torch.cat([p[key] for p in parts]) for key in parts[0]}
    actual=torch.tensor(np.load(clinical/'clinical_states.npz')['instantaneous_rates'][:,0],device='cuda')
    torch.testing.assert_close(values['rate'],actual,rtol=1e-5,atol=1e-5)
    reconstruction=values['speed']*values['gradient_norm']*values['cosine'];torch.testing.assert_close(reconstruction,values['rate'],rtol=1e-5,atol=1e-6)
    check=copy.deepcopy(model).double();zz=z[:32].double().requires_grad_(True)
    with torch.enable_grad():
        yy=check.decoder(zz)[:,head]*scale+offset;grad=torch.autograd.grad(yy.sum(),zz)[0];exact=mechanism(check,zz,head,scale)
        torch.testing.assert_close(grad.norm(dim=-1),exact['gradient_norm'],rtol=1e-10,atol=1e-10)
        torch.testing.assert_close((grad*check.forward_ode(zz)).sum(-1),exact['rate'],rtol=1e-10,atol=1e-10)
    print(f'{args.variant}: per-state exact decomposition checked',flush=True)
    raw=list(csv.DictReader((clinical/'interval_rates.csv').open()));rows=[]
    for r in raw:
        if r['marker']=='mmse' and r['split']=='test' and r['adjacent_visit']=='True':
            rows.append({'subject_id':r['subject_id'],'state_index':int(r['state_index']),'start_visit':int(r['start_visit']),'end_visit':int(r['end_visit']),
                         'diagnosis':None if r['diagnosis']=='' else int(float(r['diagnosis'])),'years':float(r['years']),
                         **{key:float(r[key]) for key in ['observed_rate','predicted_rate','observed_start']},'start_prediction':float(r['predicted_start'])})
    arrays={k:v.cpu().numpy() for k,v in values.items()}
    for r in rows:
        for key,value in arrays.items():r[key]=float(value[r['state_index']])
    # Sixteen-point Gauss-Legendre quadrature on each observed follow-up interval.
    nodes,weights=np.polynomial.legendre.leggauss(16);nodes=torch.tensor((nodes+1)/2,device='cuda',dtype=z.dtype);weights=torch.tensor(weights/2,device='cuda',dtype=z.dtype)
    path_arrays=[];maximum_rate_error=0.;endpoint_error=0.
    for start in range(0,len(rows),32):
        chunk=rows[start:start+32];idx=torch.tensor([r['state_index'] for r in chunk],device='cuda');delta=torch.tensor([r['years'] for r in chunk],device='cuda')
        state=z[idx,None,:].expand(-1,16,-1).reshape(-1,6);dt=(delta[:,None]*nodes).reshape(-1)
        path=model.solve(model.forward_ode,state,dt);path_values=mechanism(model,path,head,scale)
        avg={key:(val.reshape(len(chunk),16)*weights).sum(1) for key,val in path_values.items()}
        end=model.solve(model.forward_ode,z[idx],delta)
        end_rate=(model.decoder(end)[:,head]-model.decoder(z[idx])[:,head])*scale/delta
        difference=float((avg['rate']-end_rate).abs().max());maximum_rate_error=max(maximum_rate_error,difference)
        if difference>2e-3:raise RuntimeError(f'quadrature versus endpoint rate error {difference}')
        for i,r in enumerate(chunk):
            endpoint_error=max(endpoint_error,abs(float(end_rate[i])-r['predicted_rate']))
            r.update(path_budget=float(avg['capacity'][i]),path_alignment=float(-avg['rate'][i]/avg['capacity'][i].clamp_min(1e-12)),
                     quadrature_rate=float(avg['rate'][i]),path_mean_speed=float(avg['speed'][i]),path_mean_gradient=float(avg['gradient_norm'][i]))
        path_arrays.append({key:val.reshape(len(chunk),16).cpu().numpy() for key,val in path_values.items()})
    print(f'{args.variant}: path decomposition complete',flush=True)
    train_dementia=[float(r['observed_rate']) for r in raw if r['split']=='train' and r['marker']=='mmse' and r['adjacent_visit']=='True' and r['diagnosis']=='2.0']
    threshold=float(np.quantile(train_dementia,.2));groups=group_summary(rows)
    dementia=[r for r in rows if r['diagnosis']==2];subgroups=[]
    for label,use in [('fast_decline',[r for r in dementia if r['observed_rate']<=threshold]),('other_decline',[r for r in dementia if threshold<r['observed_rate']<0]),('stable_or_increase',[r for r in dementia if r['observed_rate']>=0])]:
        if not use:continue
        subgroup={'name':label,'intervals':len(use)}
        for key in ['observed_rate','predicted_rate','speed','gradient_norm','cosine','path_budget','path_alignment']:
            subgroup[key]=float(np.mean([r[key] for r in use]))
        subgroup['within_path_budget_fraction']=float(np.mean([abs(r['observed_rate'])<=r['path_budget'] for r in use if r['observed_rate']<0])) if any(r['observed_rate']<0 for r in use) else None
        subgroups.append(subgroup)
    from scipy.stats import spearmanr
    relationships={key:float(spearmanr([r[key] for r in dementia],[-r['observed_rate'] for r in dementia]).statistic) for key in ['speed','gradient_norm','cosine','path_alignment','path_budget']}
    summary={'model':args.variant,'groups':groups,'dementia_subgroups':subgroups,'training_dementia_fast_decline_threshold':threshold,'dementia_observed_decline_correlations':relationships,
             'checks':{'factorization_max_error':float((reconstruction-values['rate']).abs().max()),'quadrature_endpoint_max_error':maximum_rate_error,'endpoint_vs_experiment_A_max_error':endpoint_error,
                       'autograd_gradient_check':'passed','all_states':len(z),'test_intervals':len(rows),'checkpoint_sha256':info['checkpoint_sha256'],'physical_gpu':args.gpu,'device':str(z.device)},
             'definition':'MMSE original units; cosine with increasing-MMSE gradient; negative cosine means decline; path budget=mean_t(||grad g|| ||f||); path alignment=-mean_t(rate)/budget'}
    summary=enrich_summary(summary,rows)
    write_csv(out/'interval_decomposition.csv',rows);write_csv(out/'diagnosis_summary.csv',groups)
    np.savez_compressed(out/'state_decomposition.npz',**arrays)
    np.savez_compressed(out/'path_decomposition.npz',nodes=nodes.cpu().numpy(),weights=weights.cpu().numpy(),**{key:np.concatenate([p[key] for p in path_arrays]) for key in path_arrays[0]})
    (out/'metrics.json').write_text(json.dumps(summary,indent=2));plots(out,rows,summary,cached['coordinates'],meta,arrays)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
