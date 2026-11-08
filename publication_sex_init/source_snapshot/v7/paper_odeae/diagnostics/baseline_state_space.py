"""Visualize probabilistic ODE-AE at the same cutoff times as shared_matched."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from paper_odeae.data import Prefixes,collate
from paper_odeae.models import paper_model_from_config
from paper_odeae.train import device_batch
from .state_space import ROOT,COLORS,LABELS,local_field,plots,interactive


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def extract(model,payload):
    means=[];stds=[];states=[];velocities=[];metadata=[];checks=[]
    schema=model.schema;fields={f['name']:i for i,f in enumerate(schema)}
    offset=payload['metadata']['offset'];scale=payload['metadata']['scale']
    for split in ['train','validation','test']:
        seqs=[s for s in payload['sequences'] if s['split']==split]
        count=0
        for cpu in DataLoader(Prefixes(seqs),batch_size=64,collate_fn=collate):
            batch=device_batch(cpu);decoded=[]
            # Baseline only calls DECODER during trajectory generation. Capture
            # its true states rather than approximating the first-to-cutoff solve.
            hook=model.decoder.register_forward_pre_hook(lambda module,args:decoded.append(args[0].detach()))
            try:out=model(batch,sample=False)
            finally:hook.remove()
            assert len(decoded)==batch['values'].shape[1]
            trajectory=torch.stack(decoded,1)
            z=trajectory[torch.arange(len(batch['cutoff']),device='cuda'),batch['cutoff']-1]
            prediction=model.decoder(z)
            expected=out['prediction'][torch.arange(len(z),device='cuda'),batch['cutoff']-1]
            torch.testing.assert_close(prediction,expected,rtol=0,atol=0)
            if not checks:
                torch.testing.assert_close(trajectory[:,0],out['mu'],rtol=0,atol=0)
                checks.append('decoder-hook states exactly reproduce original cutoff predictions; first state equals '+('deterministic initial state' if model.kind=='deterministic' else 'posterior mean'))
            states.append(z);velocities.append(model.forward_ode(z));means.append(out['mu']);stds.append(out['std'])
            values=cpu['values'].numpy();mask=cpu['mask'].numpy()
            for i,cutoff in enumerate(cpu['cutoff'].tolist()):
                row={'split':split,'subject_id':cpu['subject_ids'][i],'prefix':cutoff,'age':float(cpu['ages'][i,cutoff-1])}
                for name in ['diagnosis','mmse','cdrsb']:
                    j=fields[name]
                    row[name]=float(values[i,cutoff-1,j]*scale[j]+offset[j]) if mask[i,cutoff-1,j] else None
                metadata.append(row)
            count+=len(z)
        print(f'Extracted {split}: {len(seqs)} patients, {count} cutoff states',flush=True)
    return torch.cat(states),torch.cat(velocities),torch.cat(means),torch.cat(stds),metadata,checks


def save(fig,out,name):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=180)
    plt.close(fig)


def additional_plots(out,mu,std,initial_y,meta,stats):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    train=np.array([r['split']=='train' for r in meta]);test=np.array([r['split']=='test' for r in meta])
    labels=np.array([-1 if r['diagnosis'] is None else int(r['diagnosis']) for r in meta])
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    ax=axes[0]
    for k in range(3):
        m=test&(labels==k);ax.scatter(*initial_y[m].T,s=8,alpha=.4,c=COLORS[k],label=LABELS[k],rasterized=True)
    ax.legend();ax.set(title='Posterior means at first visit\nColored by last available diagnosis',xlabel='Initial-mean PC1',ylabel='Initial-mean PC2')
    ax=axes[1];sd=np.array(stats['train_posterior_mean_std_by_dimension']);noise=np.array(stats['train_average_posterior_std_by_dimension'])
    x=np.arange(6);ax.bar(x-.18,sd,.36,label='Between-patient/prefix mean variation');ax.bar(x+.18,noise,.36,label='Average posterior standard deviation')
    ax.set_yscale('log');ax.set(xticks=x,xticklabels=[str(i+1) for i in x],xlabel='Latent dimension',ylabel='Standard deviation (log scale)',title='Signal variation versus posterior spread');ax.legend(fontsize=7)
    ax=axes[2];cov=np.array(stats['average_projected_initial_covariance'])
    vals,vecs=np.linalg.eigh(cov);order=vals.argsort()[::-1];vals=vals[order];vec=vecs[:,order[0]]
    center=initial_y[train].mean(0);ellipse=Ellipse(center,2*np.sqrt(vals[0]),2*np.sqrt(vals[1]),angle=np.degrees(np.arctan2(vec[1],vec[0])),fill=False,color='#555',lw=2,label='Average covariance: 1 SD ellipse')
    ax.add_patch(ellipse);ax.scatter(*initial_y[test].T,s=3,alpha=.3,label='Test posterior means',rasterized=True)
    extent=max(float(np.sqrt(vals.max())*1.25),float(abs(initial_y[test]-center).max()*1.1))
    ax.set(xlim=(center[0]-extent,center[0]+extent),ylim=(center[1]-extent,center[1]+extent),aspect='equal',title='Posterior means versus uncertainty scale',xlabel='Initial-mean PC1',ylabel='Initial-mean PC2');ax.legend(fontsize=7)
    save(fig,out,'posterior_uncertainty')


def comparison(out,coords,meta,variance,model_title='Baseline probabilistic ODE-AE'):
    import matplotlib.pyplot as plt
    path=ROOT/'paper_odeae/results/shared_matched_state_space'
    other=json.loads((path/'metadata.json').read_text());shared=np.load(path/'states.npz')['coordinates']
    assert [(r['split'],r['subject_id'],r['prefix']) for r in meta]==[(r['split'],r['subject_id'],r['prefix']) for r in other]
    var=json.loads((path/'statistics.json').read_text())['pca_variance_ratio']
    test=np.array([r['split']=='test' for r in meta]);labels=np.array([-1 if r['diagnosis'] is None else int(r['diagnosis']) for r in meta])
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,xy,v,title in [(axes[0],coords,variance,model_title),(axes[1],shared,var,'shared_matched')]:
        for k in range(3):
            mask=test&(labels==k);ax.scatter(*xy[mask].T,s=8,alpha=.45,c=COLORS[k],label=LABELS[k],rasterized=True)
        ax.set(title=title,xlabel=f'Own PC1 ({v[0]*100:.2f}%)',ylabel=f'Own PC2 ({v[1]*100:.2f}%)');ax.legend()
    fig.suptitle('Same test patients and history cutoffs; separate train-fitted PCA coordinates')
    save(fig,out,'baseline_vs_shared')


def context_diagnostics(out,payload,meta,coords):
    """Statistical summaries of saved coordinates and recorded covariates."""
    import matplotlib.pyplot as plt
    seqs={str(s['subject_id']):s for s in payload['sequences']}
    fields={f['name']:j for j,f in enumerate(payload['schema'])}
    test=np.array([r['split']=='test' for r in meta]);coordinates=coords[test]
    rows=[r for r in meta if r['split']=='test'];features={}
    for name in ['sex','marital_status','diagnosis']:
        j=fields[name];features[name]=np.array([seqs[str(r['subject_id'])]['values'][r['prefix']-1,j]
            if seqs[str(r['subject_id'])]['mask'][r['prefix']-1,j] else np.nan for r in rows])
    features['history_visits']=np.array([r['prefix'] for r in rows])
    features['history_years']=np.array([r['age']-float(seqs[str(r['subject_id'])]['ages'][0]) for r in rows])
    summary={}
    for name in ['sex','marital_status','diagnosis','history_visits']:
        x=features[name];valid=np.isfinite(x);y=coordinates[valid,1];x=x[valid]
        groups=[{'code':int(code),'states':int((x==code).sum()),'mean_pc2':float(y[x==code].mean())} for code in np.unique(x)]
        between=sum(g['states']*(g['mean_pc2']-y.mean())**2 for g in groups)
        summary[name]={'pc2_between_group_variance_fraction':float(between/((y-y.mean())**2).sum()),'groups':groups}
    fig,axes=plt.subplots(1,4,figsize=(19,4.8))
    for ax,name in zip(axes,['sex','marital_status','history_visits','history_years']):
        x=features[name];valid=np.isfinite(x)
        sc=ax.scatter(*coordinates[valid].T,c=x[valid],cmap='viridis',s=7,alpha=.55,rasterized=True)
        fig.colorbar(sc,ax=ax);ax.set(title=name.replace('_',' '),xlabel='Baseline PC1',ylabel='Baseline PC2')
    save(fig,out,'context_factors')
    (out/'context_statistics.json').write_text(json.dumps(summary,indent=2))


def deterministic_initial_plot(out,initial_y,meta,variance):
    import matplotlib.pyplot as plt
    test=np.array([r['split']=='test' for r in meta]);labels=np.array([-1 if r['diagnosis'] is None else int(r['diagnosis']) for r in meta])
    fig,ax=plt.subplots(figsize=(7,5))
    for k in range(3):
        m=test&(labels==k);ax.scatter(*initial_y[m].T,s=8,alpha=.45,c=COLORS[k],label=LABELS[k],rasterized=True)
    ax.set(title='deterministic_22545: initial states at first visit\nColored by last available diagnosis',
           xlabel=f'Initial PC1 ({variance[0]*100:.2f}%)',ylabel=f'Initial PC2 ({variance[1]*100:.2f}%)');ax.legend()
    save(fig,out,'initial_state')


def deterministic_comparison(root):
    import matplotlib.pyplot as plt
    names=[('','Probabilistic ODE-AE (3,003 parameters)'),('deterministic_22545_','deterministic_22545 (22,545 parameters)')]
    models=[]
    for prefix,title in names:
        meta=json.loads((root/f'{prefix}metadata.json').read_text())
        stats=json.loads((root/f'{prefix}statistics.json').read_text())
        models.append((meta,stats,np.load(root/f'{prefix}states.npz')['coordinates'],title))
    assert [(r['split'],r['subject_id'],r['prefix']) for r in models[0][0]]==[(r['split'],r['subject_id'],r['prefix']) for r in models[1][0]]
    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    for ax,(meta,stats,xy,title) in zip(axes,models):
        test=np.array([r['split']=='test' for r in meta]);dx=np.array([-1 if r['diagnosis'] is None else int(r['diagnosis']) for r in meta])
        for k in range(3):
            m=test&(dx==k);ax.scatter(*xy[m].T,s=8,alpha=.45,c=COLORS[k],label=LABELS[k],rasterized=True)
        variance=stats['pca_variance_ratio'];ax.set(title=title,xlabel=f'Own PC1 ({variance[0]*100:.2f}%)',ylabel=f'Own PC2 ({variance[1]*100:.2f}%)');ax.legend()
    fig.suptitle('Same test patients and history cutoff times; separate train-fitted PCA')
    save(fig,root,'odeae_vs_deterministic_22545')


def publish_prefixed(working,root):
    """Keep the two models' artifacts together, without overwriting ODE-AE."""
    names={p.name:'deterministic_22545_'+p.name for p in working.iterdir() if p.is_file()}
    for old,new in names.items():
        p=working/old
        if p.suffix=='.html':
            text=p.read_text()
            for source,target in names.items():text=text.replace('href="'+source+'"','href="'+target+'"')
            p.write_text(text)
        p.replace(root/new)
    working.rmdir()


@torch.no_grad()
def velocity_projection(out,z,f,meta,stats):
    """Complement state PCA when the dominant flow lies outside its plane."""
    import matplotlib.pyplot as plt
    train=torch.tensor([r['split']=='train' for r in meta],device=z.device)
    mean=z[train].mean(0);_,s,vh=torch.linalg.svd(f[train],full_matrices=False)
    basis=vh[:2].T;y=(z-mean)@basis;v=f@basis
    labels=torch.tensor([-1 if r['diagnosis'] is None else int(r['diagnosis']) for r in meta],device=z.device)
    field=local_field(y[train],v[train],labels[train])
    ratio=float(v.square().sum()/f.square().sum())
    state_ratio=float(y[train].square().sum()/(z[train]-mean).square().sum())
    stats['velocity_aligned_projection']={'fit':'training velocities; uncentered SVD, includes mean flow',
        'velocity_energy_retained_2d':ratio,'train_state_variance_retained_2d':state_ratio,'device':str(basis.device)}
    test=np.array([r['split']=='test' for r in meta]);dx=labels.cpu().numpy();xy=y.cpu().numpy()
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for k in range(3):
        m=test&(dx==k);axes[0].scatter(*xy[m].T,s=8,alpha=.45,c=COLORS[k],label=LABELS[k],rasterized=True)
    axes[0].legend();axes[0].set_title(f'Test states in velocity-aligned plane\nRetains {state_ratio*100:.2f}% of training state variance')
    grid=field['grid'].cpu().numpy();vel=field['velocity'].cpu().numpy();support=field['supported'].cpu().numpy()
    axes[1].scatter(*xy[test].T,c='#999',s=3,alpha=.2,rasterized=True)
    q=axes[1].quiver(*grid[support].T,*vel[support].T,np.linalg.norm(vel[support],axis=-1),cmap='plasma',angles='xy',scale_units='xy',scale=None,width=.003)
    fig.colorbar(q,ax=axes[1],label='Projected speed (latent units/year)')
    axes[1].set_title(f'Generator flow in velocity-aligned plane\nRetains {ratio*100:.2f}% of velocity energy')
    for ax in axes:ax.set(xlabel='Velocity-aligned axis 1',ylabel='Velocity-aligned axis 2')
    save(fig,out,'velocity_aligned')
    np.savez_compressed(out/'velocity_aligned.npz',coordinates=xy,velocities=v.cpu().numpy(),basis=basis.cpu().numpy(),mean=mean.cpu().numpy())


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gpu',default='1',choices=['0','1','2','3'])
    parser.add_argument('--output',default=str(ROOT/'paper_odeae/results/odeae_baseline_state_space'))
    parser.add_argument('--model',choices=['ode_ae','deterministic_22545'],default='ode_ae')
    args=parser.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(1);torch.manual_seed(0)
    import matplotlib
    matplotlib.use('Agg')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    deterministic=args.model=='deterministic_22545'
    out=root/'.deterministic_22545_work' if deterministic else root;out.mkdir(exist_ok=True)
    path=ROOT/('paper_odeae/results/capacity_matched/deterministic_22545/seed_0/best.pt' if deterministic else 'paper_odeae/results/local_available/ode_ae/seed_0/best.pt')
    saved=torch.load(path,map_location='cuda',weights_only=False);cfg=saved['config']
    model=paper_model_from_config(saved['schema'],cfg).cuda().eval();model.load_state_dict(saved['state_dict'])
    assert model.kind==('deterministic' if deterministic else 'ode_ae') and model.encoder_direction=='backward' and not model.shared_vector_field
    data=Path(cfg['data']);data=data if data.is_absolute() else ROOT/data
    payload=pickle.loads(data.read_bytes())
    original_hardware=json.loads((path.parent/'hardware.json').read_text());assert sha(data)==original_hardware['dataset_sha256']
    z,f,mu,std,meta,checks=extract(model,payload)
    train=torch.tensor([r['split']=='train' for r in meta],device='cuda')
    mean=z[train].mean(0);_,s,vh=torch.linalg.svd(z[train]-mean,full_matrices=False);basis=vh[:2].T
    variance=s.square()/s.square().sum();y=(z-mean)@basis;v=f@basis
    initial_mean=mu[train].mean(0);_,s0,v0=torch.linalg.svd(mu[train]-initial_mean,full_matrices=False)
    initial_variance=s0.square()/s0.square().sum();initial_basis=v0[:2].T;initial_y=(mu-initial_mean)@initial_basis
    labels=torch.tensor([-1 if r['diagnosis'] is None else int(r['diagnosis']) for r in meta],device='cuda')
    field=local_field(y[train],v[train],labels[train])
    noise=std[train].square().mean(0)
    stats={'checkpoint':str(path),'checkpoint_sha256':sha(path),'epoch':saved['epoch'],
           'dataset_sha256':sha(data),'parameters':sum(p.numel() for p in model.parameters()),
           'physical_gpu':args.gpu,'gpu_name':torch.cuda.get_device_name(),'model_device':str(next(model.parameters()).device),
           'state_device':str(z.device),'pca_device':str(basis.device),'field_device':str(field['velocity'].device),
           'prefix_states':len(z),'counts_by_split':{k:sum(r['split']==k for r in meta) for k in ['train','validation','test']},
           'pca_fit':'train only, unscaled, all prefix cutoffs equally weighted',
           'model':args.model,'has_probabilistic_posterior':not deterministic,
           'state_semantics':'generator trajectory captured at history cutoff, from '+('deterministic initial state' if deterministic else 'initial posterior mean'),
           'encoder_state_dim':6 if deterministic else 12,'generator_state_dim':6,'shared_field':False,
           'pca_variance_ratio':variance.cpu().tolist(),'state_variance_retained_2d':float(variance[:2].sum()),
           'velocity_energy_retained_2d':float(v.square().sum()/f.square().sum()),
           'initial_mean_pca_variance_ratio':initial_variance.cpu().tolist(),
           'train_posterior_mean_std_by_dimension':mu[train].std(0).cpu().tolist(),
           'train_average_posterior_std_by_dimension':std[train].mean(0).cpu().tolist(),
           'average_projected_initial_covariance':(initial_basis.T@torch.diag(noise)@initial_basis).cpu().tolist(),
           'train_total_mean_variation':float(mu[train].var(0).sum()),'train_total_posterior_variance':float(noise.sum()),
           'generator_speed_median':float(f.norm(dim=-1).median()),'checks':checks}
    if deterministic:
        stats['initial_state_pca_variance_ratio']=stats.pop('initial_mean_pca_variance_ratio')
        stats['train_initial_state_std_by_dimension']=stats.pop('train_posterior_mean_std_by_dimension')
        stats['train_total_initial_state_variation']=stats.pop('train_total_mean_variation')
        for key in ['train_average_posterior_std_by_dimension','average_projected_initial_covariance','train_total_posterior_variance']:
            del stats[key]
    arrays={k:value.cpu().numpy() if torch.is_tensor(value) else value for k,value in field.items()}
    distribution={'initial_states':mu.cpu().numpy()} if deterministic else {'posterior_mean':mu.cpu().numpy(),'posterior_std':std.cpu().numpy()}
    np.savez_compressed(out/'states.npz',states=z.cpu().numpy(),velocities=f.cpu().numpy(),coordinates=y.cpu().numpy(),
                        projected_velocities=v.cpu().numpy(),**distribution,
                        initial_coordinates=initial_y.cpu().numpy(),pca_basis=basis.cpu().numpy(),pca_mean=mean.cpu().numpy(),
                        initial_pca_basis=initial_basis.cpu().numpy(),initial_pca_mean=initial_mean.cpu().numpy())
    np.savez_compressed(out/'projected_field.npz',**arrays)
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));(out/'statistics.json').write_text(json.dumps(stats,indent=2))
    plots(out,y.cpu().numpy(),v.cpu().numpy(),meta,arrays,None,variance.cpu().numpy(),model_label=args.model+' generator ODE')
    if deterministic:
        velocity_projection(out,z,f,meta,stats)
        (out/'statistics.json').write_text(json.dumps(stats,indent=2))
        deterministic_initial_plot(out,initial_y.cpu().numpy(),meta,initial_variance.cpu().numpy())
    else:
        additional_plots(out,mu.cpu().numpy(),std.cpu().numpy(),initial_y.cpu().numpy(),meta,stats)
    comparison(out,y.cpu().numpy(),meta,variance.cpu().numpy(),model_title=args.model)
    context_diagnostics(out,payload,meta,y.cpu().numpy())
    interactive(out,y.cpu().numpy(),v.cpu().numpy(),meta,variance.cpu().numpy(),model_label=args.model,shared=False)
    if deterministic:
        publish_prefixed(out,root)
        deterministic_comparison(root)
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':main()
