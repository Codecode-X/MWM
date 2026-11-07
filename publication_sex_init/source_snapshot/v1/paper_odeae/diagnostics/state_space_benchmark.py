"""Frozen-checkpoint state/flow/trajectory diagnostics for eight ADNI models.

All model execution, PCA, distances and neighborhood calculations require CUDA.
The shared comparison report is generated separately from saved arrays.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from paper_odeae.data import Prefixes,collate
from paper_odeae.models import encode_values,paper_model_from_config
from paper_odeae.train import device_batch,evaluate
from .state_space import ROOT as SOURCE_ROOT,encode as shared_encode,local_field,plots,interactive

ROOT=Path(os.environ.get('ADNI_EXPERIMENT_ROOT',str(SOURCE_ROOT))).resolve()

OUT=ROOT/'paper_odeae/results/state_space_model_comparison'
RUNS=[
 ('healthflux_kl','HealthFlux-kl','healthflux','healthflux/results/adni_comparison/paper_kl/seed_0'),
 ('latent2x_baseline','latent2x baseline','shared','paper_odeae/results/shared_matched_latent_scaling/latent_2x/seed_0'),
 ('latent2x_both','latent2x both','shared','paper_odeae/results/latent2x_loss_ablation/both/seed_0'),
 ('latent2x_prior','latent2x prior','shared','paper_odeae/results/latent2x_loss_ablation/weak_prior/seed_0'),
 ('latent2x_jacobian','latent2x jacobian','shared','paper_odeae/results/latent2x_loss_ablation/jacobian/seed_0'),
 ('ode_ae','ODE-AE','ode_ae','paper_odeae/results/local_available/ode_ae/seed_0'),
 ('latent_ode','Latent ODE','latent_ode','paper_odeae/results/baselines_full/native/latent_ode/seed_0'),
 ('ode_rnn','ODE-RNN','ode_rnn','paper_odeae/results/baselines_full/native/ode_rnn/seed_0'),
 ('shared_matched','shared_matched 6D (reference)','shared','paper_odeae/results/forward_age_scaling/shared_matched/seed_0'),
]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Adapter:
    def __init__(self,name):
        self.name,self.title,self.family,relative=next(r for r in RUNS if r[0]==name)
        self.path=ROOT/relative;self.checkpoint=self.path/'best.pt'
        self.saved=torch.load(self.checkpoint,map_location='cuda',weights_only=False)
        self.cfg=self.saved['config'];schema=self.saved['schema']
        if self.family=='healthflux':
            if Path(sys.prefix).name!='healthflux':raise RuntimeError('Use healthflux environment for HealthFlux diagnostics')
            from healthflux.model import HealthFlux
            self.model=HealthFlux(schema,self.cfg,self.saved['input_normalization']).cuda()
        elif self.family in ['latent_ode','ode_rnn']:
            from paper_odeae.baselines import BaselineModel
            self.model=BaselineModel(schema,self.family,self.cfg['latent'],self.cfg['encoder_hidden'],self.cfg['ode_width'],self.cfg['rtol'],self.cfg['atol']).cuda()
        else:self.model=paper_model_from_config(schema,self.cfg).cuda()
        self.model.load_state_dict(self.saved['state_dict']);self.model.eval()
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        self.schema=schema

    def solve(self,state,age,delta,context=None):
        if self.family=='healthflux':return self.model.evolve(state,age,delta,context)
        if self.family in ['latent_ode','ode_rnn']:return self.model.solve(self.model.core.ode,state,delta/10)
        return self.model.solve(self.model.forward_ode,state,delta)

    def velocity(self,state,age,context=None):
        if self.family=='healthflux':return self.model.vector_field(state,age,context)/10
        if self.family in ['latent_ode','ode_rnn']:return self.model.core.ode(None,state)/10
        return self.model.forward_ode(state)

    def decode(self,state,age,context=None,observed_z=None):
        if self.family=='healthflux':return self.model.read(state,age,context,observed_z)
        if self.family in ['latent_ode','ode_rnn']:return self.model.core.decoder(state)
        return self.model.decoder(state)

    def state(self,batch):
        """Actual observed cutoff state in the space advanced by the generating ODE."""
        model=self.model;values=batch['values'];ages=batch['ages'];cutoff=batch['cutoff']
        if self.family=='healthflux':
            h,z,context,_=model.encode(batch,sample=False)
            return h,{'context':context,'observed_z':z,'representation':model.state_norm(torch.cat([h,z],-1))}
        if self.family=='shared':return shared_encode(model,batch),{}
        x=encode_values(values,batch['context_mask'],self.schema)
        if self.family=='ode_ae':
            last=ages.gather(1,(cutoff-1)[:,None]).squeeze(1)
            h=model.init(last[:,None]/100);previous=last
            for j in range(batch['max_context']):
                index=(cutoff-1-j).clamp_min(0);alive=j<cutoff
                age=ages.gather(1,index[:,None]).squeeze(1)
                h0=model.solve(model.backward_ode,h,torch.where(alive,age-previous,torch.zeros_like(age)))
                proposed=model.update(torch.cat([h0,x[torch.arange(len(h),device='cuda'),index]],-1))
                h=torch.where(alive[:,None],proposed,h);previous=torch.where(alive,age,previous)
            mu,logvar=h.chunk(2,-1);std=(.5*logvar.clamp(-12,8)).exp()
            state=mu;previous=ages[:,0]
            for i in range(1,batch['max_context']):
                alive=i<cutoff;delta=torch.where(alive,ages[:,i]-previous,torch.zeros_like(previous))
                state=model.solve(model.forward_ode,state,delta);previous=torch.where(alive,ages[:,i],previous)
            return state,{'initial_mu':mu,'initial_std':std}
        from paper_odeae.baselines import expanded_mask
        masks=expanded_mask(batch['context_mask'],self.schema)
        times=(ages-ages[:,:1])/10
        if self.family=='latent_ode':
            h=values.new_zeros(len(values),model.core.encoder.hidden)
            for i in range(batch['max_context']):
                if i:h=model.solve(model.core.encoder.ode,h,(times[:,i]-times[:,i-1])*(i<cutoff))
                h=model.core.encoder.cell(h,x[:,i],masks[:,i])
            mu,raw=model.core.posterior(h).chunk(2,-1)
            return mu,{'posterior_std':torch.nn.functional.softplus(raw)+1e-5}
        h=values.new_zeros(len(values),model.core.hidden)
        for i in range(batch['max_context']):
            if i:h=model.solve(model.core.ode,h,(times[:,i]-times[:,i-1])*(i<cutoff))
            h=model.core.cell(h,x[:,i],masks[:,i])
        return h,{}

    def verify(self,batch,state,extra):
        captured=[];hook=None
        if self.family=='ode_rnn':hook=self.model.core.cell.register_forward_hook(lambda module,args,out:captured.append(out.detach()))
        try:output=self.model(batch,sample=False)
        finally:
            if hook is not None:hook.remove()
        if self.family=='ode_rnn':
            expected=torch.stack(captured,1)[torch.arange(len(state),device='cuda'),batch['cutoff']-1]
        elif self.family=='ode_ae':
            observed=self.decode(state,batch['ages'].gather(1,(batch['cutoff']-1)[:,None]).squeeze(1))
            expected=output['prediction'][torch.arange(len(state),device='cuda'),batch['cutoff']-1]
            torch.testing.assert_close(observed,expected,rtol=2e-4,atol=2e-4)
            return {'comparison':'cutoff decoder prediction vs original forward','max_error':float((observed-expected).abs().max())}
        else:expected=output['mu']
        torch.testing.assert_close(state,expected,rtol=2e-4,atol=2e-4)
        if self.family=='healthflux':torch.testing.assert_close(extra['observed_z'],output['stochastic_mean'],rtol=0,atol=0)
        return {'comparison':'cutoff latent state vs original forward/hooked update','max_error':float((state-expected).abs().max())}


def metadata_rows(cpu,schema,normalization):
    fields={f['name']:i for i,f in enumerate(schema)};rows=[]
    values=cpu['values'].numpy();masks=cpu['mask'].numpy()
    for i,cut in enumerate(cpu['cutoff'].tolist()):
        row={'split':None,'subject_id':str(cpu['subject_ids'][i]),'prefix':cut,'age':float(cpu['ages'][i,cut-1])}
        for key in ['diagnosis','mmse','cdrsb']:
            k=fields[key]
            row[key]=float(values[i,cut-1,k]*normalization['scale'][k]+normalization['offset'][k]) if masks[i,cut-1,k] else None
        rows.append(row)
    return rows


@torch.no_grad()
def extract(adapter,payload):
    states=[];velocities=[];metadata=[];representations=[];checks=[]
    for split in ['train','validation','test']:
        seqs=[s for s in payload['sequences'] if s['split']==split]
        for cpu in DataLoader(Prefixes(seqs),batch_size=64,collate_fn=collate):
            batch=device_batch(cpu);z,extra=adapter.state(batch)
            ages=batch['ages'].gather(1,(batch['cutoff']-1)[:,None]).squeeze(1)
            if not checks:
                # Small independent check limits redundant full-trajectory generation.
                small=device_batch(collate([Prefixes(seqs)[i] for i in range(4)]));zs,es=adapter.state(small)
                checks.append(adapter.verify(small,zs,es))
                changed={key:value.clone() if torch.is_tensor(value) else value for key,value in small.items()}
                for i,c in enumerate(small['cutoff'].tolist()):
                    for j,field in enumerate(adapter.schema):
                        if field['kind']=='numerical':changed['values'][i,c:,j]+=50
                        else:changed['values'][i,c:,j]=(changed['values'][i,c:,j]+1)%field['classes']
                    changed['mask'][i,c:]=0
                other,_=adapter.state(changed);torch.testing.assert_close(other,zs,rtol=0,atol=0)
                checks.append({'comparison':'future values and masks changed','max_error':0.})
            states.append(z);velocities.append(adapter.velocity(z,ages,extra.get('context')))
            if 'representation' in extra:representations.append(extra['representation'])
            rows=metadata_rows(cpu,adapter.schema,payload['metadata'])
            for row in rows:row['split']=split
            metadata.extend(rows)
        print(f'{adapter.name}: {split} states extracted',flush=True)
    return torch.cat(states),torch.cat(velocities),metadata,torch.cat(representations) if representations else None,checks


def silhouette(state,labels):
    distance=torch.cdist(state,state,compute_mode='donot_use_mm_for_euclid_dist')
    own=torch.zeros(len(state),device=state.device);other=torch.full_like(own,float('inf'))
    for k in labels.unique():
        mask=labels==k;n=int(mask.sum());sums=distance[:,mask].sum(1)
        own=torch.where(mask,sums/max(n-1,1),own)
        other=torch.where(~mask,torch.minimum(other,sums/n),other)
    return float(((other-own)/torch.maximum(own,other).clamp_min(1e-12)).mean())


@torch.no_grad()
def projection(out,z,f,metadata,reuse=None):
    train=torch.tensor([r['split']=='train' for r in metadata],device='cuda')
    if reuse is None:
        mean=z[train].mean(0);_,singular,vh=torch.linalg.svd(z[train]-mean,full_matrices=False);basis=vh[:2].T
    else:
        mean=torch.tensor(reuse['pca_mean'],device='cuda');basis=torch.tensor(reuse['pca_basis'],device='cuda')
        singular=torch.linalg.svdvals(z[train]-mean)
    variance=singular.square()/singular.square().sum().clamp_min(1e-30)
    xy=(z-mean)@basis;vf=f@basis
    labels=torch.tensor([-1 if r['diagnosis'] is None else int(r['diagnosis']) for r in metadata],device='cuda')
    field=local_field(xy[train],vf[train],labels[train])
    prob=variance[variance>1e-12]
    latest={}
    for i,r in enumerate(metadata):
        if r['split']=='test':latest[r['subject_id']]=i
    ids=torch.tensor([i for i in latest.values() if metadata[i]['diagnosis'] is not None],device='cuda')
    supported=field['supported']
    stats={'dimensions':z.shape[1],'pca_variance_ratio':variance.cpu().tolist(),
           'state_variance_retained_2d':float(variance[:2].sum()),
           'velocity_energy_retained_2d':float(vf.square().sum()/f.square().sum().clamp_min(1e-30)),
           'effective_state_rank':float((-prob*prob.log()).sum().exp()),
           'train_total_state_variance':float((z[train]-mean).square().sum()/int(train.sum())),
           'speed_median_native_units_per_year':float(f.norm(dim=-1).median()),
           'local_direction_coherence_median':float(field['direction_coherence'][supported].median()) if bool(supported.any()) else None,
           'supported_grid_fraction':float(supported.float().mean()),'last_patient_diagnosis_silhouette':silhouette(z[ids]-mean,labels[ids]),
           'last_patient_silhouette_n':len(ids),'pca_fit':'training cutoff states only; unscaled; each cutoff weighted equally',
           'devices':{'state':str(z.device),'velocity':str(f.device),'pca':str(basis.device),'neighborhoods':str(field['velocity'].device)}}
    arrays={'states':z.cpu().numpy(),'velocities':f.cpu().numpy(),'coordinates':xy.cpu().numpy(),
            'projected_velocities':vf.cpu().numpy(),'pca_mean':mean.cpu().numpy(),'pca_basis':basis.cpu().numpy()}
    np.savez_compressed(out/'states.npz',**arrays)
    fields={key:value.cpu().numpy() if torch.is_tensor(value) else value for key,value in field.items()}
    np.savez_compressed(out/'projected_field.npz',**fields)
    return arrays,fields,stats,mean,basis


def save(fig,out,name):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=170)
    plt.close(fig)


def raw_plot(out,arrays,metadata,title):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    xy=arrays['coordinates'];v=arrays['projected_velocities'];speed=np.linalg.norm(v,axis=-1)
    train=np.array([r['split']=='train' for r in metadata]);span=np.ptp(np.quantile(xy[train],[.01,.99],axis=0),axis=0).max()
    gain=float(.06*span/max(float(np.quantile(speed[train],.95)),1e-20))
    norm=Normalize(vmin=0,vmax=max(float(speed.max()),1e-20))
    fig,axes=plt.subplots(1,3,figsize=(17,5.5))
    for ax,split in zip(axes,['train','validation','test']):
        m=np.array([r['split']==split for r in metadata])
        q=ax.quiver(*xy[m].T,*v[m].T,speed[m],angles='xy',scale_units='xy',scale=1/gain,cmap='plasma',norm=norm,width=.0018,alpha=.65)
        ax.set(title=f'{title}: {split}, n={m.sum()}',xlabel='Own train PC1',ylabel='Own train PC2')
        ax.set_xlim(float(xy[:,0].min()-.05*span),float(xy[:,0].max()+.05*span));ax.set_ylim(float(xy[:,1].min()-.05*span),float(xy[:,1].max()+.05*span))
        fig.colorbar(q,ax=ax,label='Projected native latent units / year')
    save(fig,out,'velocity_field_raw')
    return gain


def expand_context(context,n):return None if context is None else context.expand(n,-1)


@torch.no_grad()
def segment(adapter,h,start,end,context,count=13):
    times=torch.linspace(float(start),float(end),count,device='cuda');state=adapter.solve(h.expand(count,-1),start.expand(count),times-start,expand_context(context,count))
    endpoint=adapter.solve(h,start.reshape(1),(end-start).reshape(1),context)
    state[-1]=endpoint[0]
    return times,state,endpoint


@torch.no_grad()
def history_path(adapter,seq,mean,basis):
    model=adapter.model;batch=device_batch(collate([(seq,len(seq['ages']))]));ages=batch['ages'][0]
    target,extra=adapter.state(batch);context=extra.get('context');jumps=[];paths=[];post=[]
    if adapter.family=='shared':
        h=model.init(ages[-1:].reshape(1,1)/100);x=encode_values(batch['values'],batch['context_mask'],adapter.schema)[0]
    elif adapter.family=='ode_rnn':
        from paper_odeae.baselines import expanded_mask
        h=ages.new_zeros(1,model.core.hidden);x=encode_values(batch['values'],batch['context_mask'],adapter.schema)[0]
        masks=expanded_mask(batch['context_mask'],adapter.schema)[0]
    elif adapter.family=='healthflux':
        h=model.initial(torch.cat([context,model.age_features(ages[:1])],-1));normalized=batch['values'].clone()
        for j,field in enumerate(adapter.schema):
            if field['kind']=='numerical':normalized[...,j]=((normalized[...,j]-model.input_mean[j])/model.input_std[j]).clamp(-10,10)
    else:
        h=extra['initial_mu'] if adapter.family=='ode_ae' else adapter.solve(target,ages[-1:],(ages[0]-ages[-1]).reshape(1))
    origin=h[0].clone()
    for i,age in enumerate(ages):
        start=ages[i-1] if i else age
        _,dense,h=segment(adapter,h,start,age,context);paths.append(dense);before=h
        if adapter.family=='shared':h=model.update(torch.cat([h,x[i:i+1]],-1))
        elif adapter.family=='ode_rnn':h=model.core.cell(h,x[i:i+1],masks[i:i+1])
        elif adapter.family=='healthflux':
            dx=model.indices['diagnosis'];active=batch['context_mask'][:,i,dx].bool()
            h,_,_,_=model.jump(h,age.reshape(1),context,model.event_embedding(batch['values'][:,i,dx].long()),active,False)
            middle=h
            observation,present=model.pooled_observation(normalized[:,i],batch['context_mask'][:,i])
            h,_,_,_=model.jump(h,age.reshape(1),context,observation,present,False)
            jumps.append(torch.cat([before,middle,h]));post.append(h[0]);continue
        if adapter.family in ['shared','ode_rnn']:jumps.append(torch.cat([before,h]))
        post.append(h[0])
    error=float((h-target).abs().max());torch.testing.assert_close(h,target,rtol=2e-3,atol=2e-3)
    # A pure ODE continuation is drawn for every model. Native ODE-RNN feedback
    # predictions are separately retained below for the first-three-visit case.
    _,future,_=segment(adapter,target,ages[-1],ages[-1]+5,context,61)
    def project(tensor):return ((tensor-mean)@basis).cpu().tolist()
    return {'subject_id':str(seq['subject_id']),'ages':ages.cpu().tolist(),'origin':project(origin),
            'ode_segments':[project(s) for s in paths],'updates':[project(s) for s in jumps],
            'post_update':project(torch.stack(post)),'forecast_5y':project(future),
            'history_kind':'generator reconstruction' if adapter.family in ['ode_ae','latent_ode'] else 'chronological ODE and observation updates',
            'final_state_check_max_error':error,
            'mean_update_norm':float(torch.cat([s[-1:]-s[:1] for s in jumps]).norm(dim=-1).mean()) if jumps else None}


@torch.no_grad()
def forecast_case(adapter,seq,metadata,mean,basis):
    batch=device_batch(collate([(seq,3)]));h,extra=adapter.state(batch);context=extra.get('context')
    ages=batch['ages'][0];anchor=ages[2]
    dx=next(j for j,f in enumerate(adapter.schema) if f['name']=='mmse')
    output=sum(f.get('classes',1) for f in adapter.schema[:dx]);scale=metadata['scale'][dx];offset=metadata['offset'][dx]
    native=adapter.model(batch,sample=False)['prediction'][0,:,output]*scale+offset
    # Keep original visit intervals as integration boundaries; dense display
    # queries never replace the state used at the next real visit.
    times=[anchor];states=[h[0]];current=h;previous=anchor
    ends=[age for age in ages[3:] if bool(age<=anchor+5)]
    if not ends or float(ends[-1])<float(anchor+5):ends.append(anchor+5)
    native_state_predictions=[]
    for end in ends:
        grid,dense,current=segment(adapter,current,previous,end,context,17)
        times.extend(grid[1:]);states.extend(dense[1:]);previous=end
        if bool((ages[3:]==end).any()):
            native_state_predictions.append((int(torch.where(ages==end)[0][0]),adapter.decode(current,end.reshape(1),context)[0,output]*scale+offset))
    tt=torch.stack(times);zz=torch.stack(states);cc=expand_context(context,len(zz))
    curve=adapter.decode(zz,tt,cc)[:,output]*scale+offset
    observed=adapter.decode(h,anchor.reshape(1),context,extra.get('observed_z'))[0,output]*scale+offset
    errors=[float((pred-native[index]).abs()) for index,pred in native_state_predictions] if adapter.family!='ode_rnn' else []
    if errors and max(errors)>.003:raise RuntimeError(f'Native forecasting disagreement: {adapter.name}, {max(errors)}')
    return {'subject_id':str(seq['subject_id']),'context_visits':3,'anchor_age':float(anchor),
            'years':(tt-anchor).cpu().tolist(),'mmse_curve':curve.cpu().tolist(),
            'state_path_2d':((zz-mean)@basis).cpu().tolist(),
            'observed_cutoff_prediction':float(observed),'event_free_cutoff_prediction':float(curve[0]),
            'visit_years':(ages-anchor).cpu().tolist(),'native_visit_mmse':native.cpu().tolist(),
            'actual_visit_mmse':[float(v*scale+offset) if m else None for v,m in zip(seq['values'][:,dx],seq['mask'][:,dx])],
            'native_vs_pure_ode_mmse_max_error':max(errors) if errors else None,
            'ode_rnn_curve_is_pure_ode_only':adapter.family=='ode_rnn'}


def case_plots(out,paths,arrays,metadata,title):
    import matplotlib.pyplot as plt
    test=np.array([r['split']=='test' for r in metadata]);xy=arrays['coordinates']
    fig,axes=plt.subplots(1,3,figsize=(17,5.5))
    for ax,path in zip(axes,paths):
        ax.scatter(*xy[test].T,c='#bbb',s=2,alpha=.15,rasterized=True)
        for j,part in enumerate(path['ode_segments']):ax.plot(*np.asarray(part).T,c='#2171b5',lw=1.4,label='History ODE' if j==0 else None)
        for j,jump in enumerate(path['updates']):ax.plot(*np.asarray(jump).T,'--',c='#e87b20',lw=1.2,label='Observation update' if j==0 else None)
        ax.scatter(*np.asarray(path['post_update']).T,c='black',s=13,zorder=4)
        ax.plot(*np.asarray(path['forecast_5y']).T,c='#8b4ba8',lw=1.7,label='5y pure ODE forecast')
        ax.scatter(*path['origin'],marker='*',c='#e87b20',s=65,zorder=5,label='Initial estimate')
        ax.set(title=f"{title} | {path['label']}",xlabel='Own train PC1',ylabel='Own train PC2');ax.legend(fontsize=6)
    save(fig,out,'patient_paths')


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',required=True,choices=[r[0] for r in RUNS]);parser.add_argument('--gpu',required=True,choices=['0','1','2','3']);parser.add_argument('--output',default=str(OUT))
    args=parser.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(1);torch.manual_seed(0)
    import matplotlib
    matplotlib.use('Agg')
    adapter=Adapter(args.model);out=Path(args.output).resolve()/args.model;out.mkdir(parents=True,exist_ok=True)
    data=Path(adapter.cfg['data']);data=data if data.is_absolute() else ROOT/data
    payload=pickle.loads(data.read_bytes());hardware=json.loads((adapter.path/'hardware.json').read_text());assert sha(data)==hardware['dataset_sha256']
    old_metrics=json.loads((adapter.path/'metrics.json').read_text())
    errors=[]
    for split in ['validation','test']:
        got=evaluate(adapter.model,[s for s in payload['sequences'] if s['split']==split],adapter.schema,adapter.cfg['batch_size'])
        for region in ['interpolation','forecast','overall']:errors.append(abs(got[region]['reconstruction_error']-old_metrics[split][region]['reconstruction_error']))
    if max(errors)>1e-6:raise RuntimeError(f'Original evaluation changed: {max(errors)}')
    print(f'{args.model}: original validation/test metrics reproduced',flush=True)
    reuse=None;reused=False
    if args.model=='shared_matched':
        source=ROOT/'paper_odeae/results/shared_matched_state_space';info=json.loads((source/'statistics.json').read_text())
        assert sha(adapter.checkpoint)==info['checkpoint_sha256']
        reuse=dict(np.load(source/'states.npz'));metadata=json.loads((source/'metadata.json').read_text())
        z=torch.tensor(reuse['states'],device='cuda');f=torch.tensor(reuse['velocities'],device='cuda');representation=None;checks=[{'source':'previous CUDA state extraction; exact checkpoint hash verified'}];reused=True
    else:z,f,metadata,representation,checks=extract(adapter,payload)
    reference=json.loads((ROOT/'paper_odeae/results/shared_matched_state_space/metadata.json').read_text())
    assert metadata==reference,'Patient/cutoff/label order differs'
    arrays,field,stats,mean,basis=projection(out,z,f,metadata,reuse)
    if representation is not None:
        train=torch.tensor([r['split']=='train' for r in metadata],device='cuda');rm=representation[train].mean(0)
        _,rs,rvh=torch.linalg.svd(representation[train]-rm,full_matrices=False);rb=rvh[:2].T;ry=(representation-rm)@rb
        np.savez_compressed(out/'health_representation_384.npz',states=representation.cpu().numpy(),coordinates=ry.cpu().numpy(),pca_mean=rm.cpu().numpy(),pca_basis=rb.cpu().numpy())
        stats['full_health_representation']={'dimensions':384,'pca_state_variance_retained_2d':float(rs[:2].square().sum()/rs.square().sum()),'semantics':'observed LN([h, posterior mean]); state distribution only'}
    plots(out,arrays['coordinates'],arrays['projected_velocities'],metadata,field,None,np.array(stats['pca_variance_ratio']),model_label=adapter.title)
    stats['raw_arrow_display_gain']=raw_plot(out,arrays,metadata,adapter.title)
    interactive(out,arrays['coordinates'],arrays['projected_velocities'],metadata,stats['pca_variance_ratio'],model_label=adapter.title,shared=adapter.family=='shared')
    page=(out/'interactive.html').read_text().replace('原六维向量场','原高维向量场')
    if adapter.family=='healthflux':page=page.replace('生成 ODE','年龄和背景条件 ODE')
    (out/'interactive.html').write_text(page)
    cases=json.loads((ROOT/'paper_odeae/results/shared_matched_state_space/patient_paths.json').read_text())
    seqs={str(s['subject_id']):s for s in payload['sequences'] if s['split']=='test'};paths=[];forecasts=[]
    for case in cases:
        seq=seqs[case['subject_id']];path=history_path(adapter,seq,mean,basis);path['label']=case['label'];paths.append(path)
        forecast=forecast_case(adapter,seq,payload['metadata'],mean,basis);forecast['label']=case['label'];forecasts.append(forecast)
    case_plots(out,paths,arrays,metadata,adapter.title)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2));(out/'patient_paths.json').write_text(json.dumps(paths,indent=2));(out/'forecast_cases.json').write_text(json.dumps(forecasts,indent=2))
    for key,value in adapter.model.state_dict().items():torch.testing.assert_close(value,adapter.saved['state_dict'][key],rtol=0,atol=0)
    stats.update(name=adapter.name,title=adapter.title,family=adapter.family,checkpoint=str(adapter.checkpoint),checkpoint_sha256=sha(adapter.checkpoint),checkpoint_epoch=adapter.saved['epoch'],
                 dataset_sha256=sha(data),parameters=sum(p.numel() for p in adapter.model.parameters()),
                 trainable_parameters=hardware.get('trainable_parameters',hardware['parameters']),
                 physical_gpu=args.gpu,gpu_name=torch.cuda.get_device_name(),environment=Path(sys.prefix).name,
                 model_device=str(next(adapter.model.parameters()).device),training_performed=False,checkpoint_parameters_unchanged=True,
                 states=len(z),counts_by_split={k:sum(r['split']==k for r in metadata) for k in ['train','validation','test']},
                 pca_reference_reused=reused,original_evaluation_max_difference=max(errors),state_checks=checks,
                 state_semantics=('256D deterministic post-update h; conditional field evaluated at actual age/context' if adapter.family=='healthflux' else
                                  'generator state at cutoff; inferred from reverse history' if adapter.family=='ode_ae' else
                                  '32D generator posterior mean at cutoff; 64D encoder not plotted' if adapter.family=='latent_ode' else
                                  'post-observation GRU hidden state; native future prediction includes feedback jumps' if adapter.family=='ode_rnn' else 'post-observation shared latent state'),
                 velocity_time_conversion='native decade derivative /10' if adapter.family in ['healthflux','latent_ode','ode_rnn'] else 'native year derivative',
                 native_metrics=old_metrics,diagnostics_source_sha256=sha(__file__))
    (out/'statistics.json').write_text(json.dumps(stats,indent=2,allow_nan=False))
    (out/'report.zh-CN.md').write_text(f"# {adapter.title} 状态空间\n\n完整横向分析见 [总报告](../report.zh-CN.md)。\n\n[状态图](state_space.png) · [平滑场](velocity_field.png) · [原始场](velocity_field_raw.png) · [患者轨迹](patient_paths.png) · [交互页面](interactive.html) · [CUDA记录](statistics.json)\n")
    print(json.dumps({'complete':adapter.name,'gpu':args.gpu,'dimensions':z.shape[1],'pca_retained':stats['state_variance_retained_2d']}),flush=True)


if __name__=='__main__':main()
