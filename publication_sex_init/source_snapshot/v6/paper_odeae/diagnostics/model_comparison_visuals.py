"""Parallel state/flow/case panels for the three completed CUDA checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np
import torch

from paper_odeae.data import collate
from paper_odeae.models import paper_model_from_config
from paper_odeae.train import device_batch
from .state_space import ROOT,COLORS,LABELS,local_field

OUT=ROOT/'paper_odeae/results/odeae_baseline_state_space'
RUNS=[('ode_ae','ODE-AE',OUT,''),
      ('deterministic_22545','Deterministic ODE-AE',OUT,'deterministic_22545_'),
      ('shared_matched','shared_matched',ROOT/'paper_odeae/results/shared_matched_state_space','')]


def save(fig,name):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180)
    plt.close(fig)


def read_runs():
    runs=[]
    for name,title,path,prefix in RUNS:
        data=dict(np.load(path/f'{prefix}states.npz'))
        stats=json.loads((path/f'{prefix}statistics.json').read_text())
        meta=json.loads((path/f'{prefix}metadata.json').read_text())
        runs.append({'name':name,'title':title,'data':data,'stats':stats,'meta':meta})
    keys=lambda r:[(m['split'],m['subject_id'],m['prefix']) for m in r['meta']]
    assert keys(runs[0])==keys(runs[1])==keys(runs[2])
    return runs


def state_panels(runs):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(17,10))
    for col,run in enumerate(runs):
        xy=run['data']['coordinates'];meta=run['meta'];variance=run['stats']['pca_variance_ratio']
        test=np.array([m['split']=='test' for m in meta])
        dx=np.array([-1 if m['diagnosis'] is None else int(m['diagnosis']) for m in meta])
        ax=axes[0,col]
        for k in range(3):
            mask=test&(dx==k);ax.scatter(*xy[mask].T,s=7,alpha=.45,c=COLORS[k],label=LABELS[k],rasterized=True)
        ax.legend(fontsize=8);ax.set_title(run['title']+' | diagnosis')
        mmse=np.array([np.nan if m['mmse'] is None else m['mmse'] for m in meta]);mask=test&np.isfinite(mmse)
        ax=axes[1,col];s=ax.scatter(*xy[mask].T,c=mmse[mask],vmin=0,vmax=30,cmap='viridis',s=7,alpha=.6,rasterized=True)
        fig.colorbar(s,ax=ax,label='Observed MMSE');ax.set_title(run['title']+' | MMSE')
        for ax in axes[:,col]:ax.set(xlabel=f'State PC1 ({variance[0]*100:.2f}%)',ylabel=f'State PC2 ({variance[1]*100:.2f}%)')
    save(fig,'comparison_state_space')


@torch.no_grad()
def velocity_panels(runs):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(17,10));statistics={}
    for col,run in enumerate(runs):
        a=run['data'];meta=run['meta'];train=torch.tensor([m['split']=='train' for m in meta],device='cuda')
        test=np.array([m['split']=='test' for m in meta])
        z=torch.tensor(a['states'],device='cuda');f=torch.tensor(a['velocities'],device='cuda')
        labels=torch.tensor([-1 if m['diagnosis'] is None else int(m['diagnosis']) for m in meta],device='cuda')
        _,_,vh=torch.linalg.svd(f[train],full_matrices=False);basis=vh[:2].T
        mean=z[train].mean(0);vy=(z-mean)@basis;vf=f@basis
        ratio=float(vf.square().sum()/f.square().sum());state_ratio=float(vy[train].square().sum()/(z[train]-mean).square().sum())
        statistics[run['name']]={'state_plane_velocity_energy':run['stats']['velocity_energy_retained_2d'],
                               'velocity_plane_velocity_energy':ratio,'velocity_plane_training_state_variance':state_ratio,'fit_device':str(basis.device)}
        np.savez_compressed(OUT/f"comparison_{run['name']}_velocity_projection.npz",basis=basis.cpu().numpy(),mean=mean.cpu().numpy(),coordinates=vy.cpu().numpy(),velocities=vf.cpu().numpy())
        for row,(xy,v,retention,label) in enumerate([
                (torch.tensor(a['coordinates'],device='cuda'),torch.tensor(a['projected_velocities'],device='cuda'),run['stats']['velocity_energy_retained_2d'],'State PCA'),
                (vy,vf,ratio,'Velocity-aligned')]):
            field=local_field(xy[train],v[train],labels[train]);supported=field['supported'].cpu().numpy()
            grid=field['grid'].cpu().numpy();vel=field['velocity'].cpu().numpy();pts=xy.cpu().numpy()
            ax=axes[row,col];ax.scatter(*pts[test].T,c='#aaa',s=2,alpha=.15,rasterized=True)
            q=ax.quiver(*grid[supported].T,*vel[supported].T,np.linalg.norm(vel[supported],axis=-1),cmap='plasma',angles='xy',scale_units='xy',scale=None,width=.003)
            fig.colorbar(q,ax=ax,label='Projected speed / year')
            ax.set(title=f"{run['title']} | {label}\n{retention*100:.2f}% velocity energy retained",xlabel=label+' axis 1',ylabel=label+' axis 2')
    save(fig,'comparison_velocity_fields')
    return statistics


@torch.no_grad()
def case_panels(runs,payload):
    import matplotlib.pyplot as plt
    cases=json.loads((ROOT/'paper_odeae/results/shared_matched_state_space/patient_paths.json').read_text())
    seqs={str(s['subject_id']):s for s in payload['sequences'] if s['split']=='test'}
    fig,axes=plt.subplots(3,3,figsize=(17,14));clinical,ca=plt.subplots(1,3,figsize=(17,5))
    records=[];verification=[]
    for col,run in enumerate(runs):
        checkpoint=Path(run['stats']['checkpoint']);saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
        model=paper_model_from_config(saved['schema'],saved['config']).cuda().eval();model.load_state_dict(saved['state_dict'])
        basis=torch.tensor(run['data']['pca_basis'],device='cuda');mean=torch.tensor(run['data']['pca_mean'],device='cuda')
        mmse_index=next(j for j,f in enumerate(model.schema) if f['name']=='mmse')
        mmse_output=sum(f.get('classes',1) for f in model.schema[:mmse_index])
        offset=payload['metadata']['offset'][mmse_index];scale=payload['metadata']['scale'][mmse_index]
        for row,case in enumerate(cases):
            seq=seqs[case['subject_id']];cutoff=3
            batch=device_batch(collate([(seq,cutoff)]));out=model(batch,sample=False)
            ages=torch.tensor(seq['ages'],device='cuda');first=ages[0];anchor_age=ages[cutoff-1]
            history=torch.linspace(float(first),float(anchor_age),60,device='cuda')
            future=torch.linspace(float(anchor_age),float(anchor_age+5),80,device='cuda')
            anchor=out['mu'];origin=anchor_age if model.encoder_direction=='forward' else first
            def generate(times):return model.solve(model.forward_ode,anchor.expand(len(times),-1),times-origin)
            past_z=generate(history);future_z=generate(future);visit_z=generate(ages)
            discrepancy=float((model.decoder(visit_z)-out['prediction'][0]).abs().max())
            if discrepancy>2e-3:raise RuntimeError(f'dense solve differs from original generation: {discrepancy}')
            verification.append({'model':run['name'],'subject_id':case['subject_id'],'max_difference_from_original_prediction':discrepancy})
            past=((past_z-mean)@basis).cpu().numpy();fut=((future_z-mean)@basis).cpu().numpy();visits=((visit_z-mean)@basis).cpu().numpy()
            ax=axes[row,col];background=run['data']['coordinates'];test=np.array([m['split']=='test' for m in run['meta']])
            ax.scatter(*background[test].T,s=2,c='#aaa',alpha=.12,rasterized=True)
            ax.plot(*past.T,c='#2171b5',lw=2,label='History reconstruction')
            ax.plot(*fut.T,c='#8b4ba8',lw=2,label='5-year forecast')
            ax.scatter(*visits[:cutoff].T,c='black',s=24,zorder=5,label='Known visit times')
            ax.scatter(*visits[cutoff-1],marker='*',s=120,c='#e87b20',zorder=6,label='History cutoff')
            ax.annotate('',xy=fut[-1],xytext=fut[-5],arrowprops={'arrowstyle':'->','color':'#8b4ba8','lw':2})
            ax.set(title=f"{run['title']} | {case['label']} case",xlabel='Own state PC1',ylabel='Own state PC2')
            if row==0:ax.legend(fontsize=7)
            pred_mmse=(out['prediction'][0,:,mmse_output]*scale+offset).cpu().numpy()
            dense_mmse=(model.decoder(future_z)[:,mmse_output]*scale+offset).cpu().numpy()
            xs=(ages-anchor_age).cpu().numpy()
            ca[row].plot((future-anchor_age).cpu().numpy(),dense_mmse,lw=2,label=run['title'])
            if col==0:
                valid=(seq['mask'][:,mmse_index]>0)&(xs>=0)&(xs<=5)
                actual=seq['values'][:,mmse_index]*scale+offset
                ca[row].scatter(xs[valid],actual[valid],c='black',s=35,zorder=5,label='Actual MMSE')
                ca[row].axvline(0,c='#aaa',ls='--');ca[row].set(title=f"{case['label']} case | {case['subject_id']}",xlabel='Years after 3rd visit',ylabel='MMSE score')
            records.append({'model':run['name'],'subject_id':case['subject_id'],'case_label':case['label'],'context_visits':cutoff,
                            'context_ages':seq['ages'][:cutoff].tolist(),'last_case_diagnosis_label_from':'last available visit; used for case selection only',
                            'visit_ages':seq['ages'].tolist(),'predicted_visit_mmse':pred_mmse.tolist(),
                            'observed_visit_mmse':[float(v*scale+offset) if m else None for v,m in zip(seq['values'][:,mmse_index],seq['mask'][:,mmse_index])],
                            'history_path_2d':past.tolist(),'forecast_path_2d':fut.tolist()})
        del model
    for ax in ca:ax.legend(fontsize=8);ax.set_ylim(0,31)
    save(fig,'comparison_case_trajectories');save(clinical,'comparison_case_mmse')
    (OUT/'comparison_cases.json').write_text(json.dumps(records,indent=2))
    return verification


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gpu',default='1',choices=['0','1','2','3']);args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(1)
    import matplotlib
    matplotlib.use('Agg')
    runs=read_runs();state_panels(runs);print('State panels complete',flush=True)
    stats=velocity_panels(runs);print('Velocity panels complete',flush=True)
    payload=pickle.loads((ROOT/'paper_odeae/cache/local_available.pkl').read_bytes())
    checks=case_panels(runs,payload);print('Patient case panels complete',flush=True)
    result={'physical_gpu':args.gpu,'gpu_name':torch.cuda.get_device_name(),'model_device':'cuda:0','projection_device':'cuda:0',
            'case_prediction_checks':checks,'velocity_projection':stats,'case_selection':'same three test patients as earlier shared_matched examples; first 3 visits observed',
            'checkpoint_sha256':{r['name']:hashlib.sha256(Path(r['stats']['checkpoint']).read_bytes()).hexdigest() for r in runs}}
    (OUT/'comparison_visuals.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
