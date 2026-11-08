"""Patient-balanced global/local dimension, sampling sensitivity and known-geometry controls."""
import argparse
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from .manifold_core import (setup,read,atomic,pca,effective_rank,dimension,knn,local_pca,id_estimates,
    describe,patient_numbers,representatives,limited,load_model,extract_mimic,load_states,provenance)


def bootstrap_mean(x,repeats=500):
    generator=torch.Generator(device='cuda').manual_seed(37);x=x[torch.isfinite(x)]
    indices=torch.randint(len(x),(repeats,len(x)),device='cuda',generator=generator)
    return torch.quantile(x[indices].mean(1),x.new_tensor([.025,.975])).tolist()


@torch.no_grad()
def cloud_statistics(train,test,train_pid,test_pid,full=True):
    mean,basis,eig,ratio=pca(train);results={'train_patients':len(train),'test_patients':len(test),
        'pca_ratio':ratio.tolist(),'pca_d90':int(dimension(ratio,.9)),'pca_d95':int(dimension(ratio,.95)),
        'pca_d99':int(dimension(ratio,.99)),'entropy_effective_rank':float(effective_rank(eig)),
        'participation_ratio':float(eig.sum().square()/eig.square().sum()),'global_reconstruction':{},'scales':{}}
    total=(test-mean).square().sum().clamp_min(1e-30)
    for d in [2,4,6,8,12,16,20,24,28]:
        residual=(test-mean)-(test-mean)@basis[:,:d]@basis[:,:d].T
        results['global_reconstruction'][str(d)]=float(residual.square().sum()/total)
    summaries={};arrays={'eigenvalues':eig,'pca_mean':mean,'pca_basis':basis}
    transformations={'raw':torch.eye(28,device='cuda',dtype=torch.float64)}
    if full:
        covariance=basis@torch.diag(eig)@basis.T
        transformations['axis_zscore']=torch.diag(covariance.diag().clamp_min(1e-12).rsqrt())
        transformations['whitened']=basis@torch.diag(eig.clamp_min(float(eig[0])*1e-8).rsqrt())
    for name,transform in transformations.items():
        reference=(train-mean)@transform;query=(test-mean)@transform
        distances,neighbors=knn(query,reference,160,test_pid,train_pid)
        estimates=id_estimates(distances)
        summary={key:describe(value) if torch.is_tensor(value) else value for key,value in estimates.items()}
        summary['mle_20_mean_bootstrap95']=bootstrap_mean(estimates['mle_20'])
        summary['local_pca']={}
        if name=='raw':
            for k in [40,80,160]:
                center,local_basis,spectrum=local_pca(reference,neighbors[:,:k]);local_ratio=spectrum/spectrum.sum(-1,keepdim=True)
                d90=dimension(local_ratio,.9);d95=dimension(local_ratio,.95)
                summary['local_pca'][str(k)]={'d90':describe(d90),'d95':describe(d95),'entropy_rank':describe(effective_rank(spectrum))}
                if k==80:
                    arrays['test_local_eigenvalues']=spectrum;arrays['test_local_d90']=d90;arrays['test_local_d95']=d95
                    arrays['test_local_mean']=center;arrays['test_local_basis']=local_basis
            arrays['test_distances']=distances;arrays['test_mle20']=estimates['mle_20'];arrays['test_mle80']=estimates['mle_80']
        results['scales'][name]=summary
    return results,arrays


def controls(train,test):
    generator=torch.Generator(device='cuda').manual_seed(984)
    def random(n,d):return torch.randn(n,d,device='cuda',dtype=torch.float64,generator=generator)
    n,m=len(train),len(test);mean,basis,eig,ratio=pca(train)
    rotation=torch.linalg.qr(random(28,8)).Q
    roll_rotation=torch.linalg.qr(random(28,3)).Q
    def roll(n):
        u=torch.rand(n,2,device='cuda',dtype=torch.float64,generator=generator);t=1.5*torch.pi+3*torch.pi*u[:,0]
        return torch.stack([t*t.cos(),20*u[:,1],t*t.sin()],1)@roll_rotation.T
    return {'isotropic_gaussian28':(random(n,28),random(m,28)),
            'covariance_matched_gaussian28':(mean+(random(n,28)*eig.sqrt())@basis.T,mean+(random(m,28)*eig.sqrt())@basis.T),
            'linear8_in_28':(random(n,8)@rotation.T,random(m,8)@rotation.T),
            'swiss_roll2_in_28':(roll(n),roll(m))}


def plots(out,result,arrays,test,metadata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','svg','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4.7));ratio=np.asarray(result['primary']['pca_ratio']);d=np.arange(1,29)
    axes[0].plot(d,np.cumsum(ratio),'-o',ms=3)
    for threshold in [.9,.95,.99]:axes[0].axhline(threshold,color='gray',lw=.7,ls='--')
    axes[0].set(xlabel='Number of global PCs',ylabel='Cumulative variance',title='One state per training patient',ylim=(0,1.02))
    axes[1].semilogy(d,ratio,'o-');axes[1].set(xlabel='Principal component',ylabel='Variance fraction',title='Global spectrum')
    raw=result['primary']['scales']['raw'];ks=[10,20,40,80,160]
    axes[2].plot(ks,[raw[f'mle_{k}']['median'] for k in ks],'-o',label='Observed states')
    for name in ['covariance_matched_gaussian28','isotropic_gaussian28','linear8_in_28','swiss_roll2_in_28']:
        r=result['controls'][name]['scales']['raw'];axes[2].plot(ks,[r[f'mle_{k}']['median'] for k in ks],'-o',ms=3,label=name)
    axes[2].set(xlabel='Nearest neighbors k',ylabel='Local dimension estimate',title='Matched sample-size controls');axes[2].legend(fontsize=6)
    save(fig,'dimension_evidence')
    fig,axes=plt.subplots(1,3,figsize=(16,4.7));test_np=test.detach().cpu().numpy();mean=arrays['pca_mean'].cpu().numpy();basis=arrays['pca_basis'].cpu().numpy();xy=(test_np-mean)@basis[:,:2]
    for ax,key,title in [(axes[0],'test_mle20','kNN dimension (k=20)'),(axes[1],'test_local_d95','Local PCA d95 (k=80)')]:
        c=arrays[key].cpu().numpy();sc=ax.scatter(*xy.T,c=c,s=14,cmap='viridis',alpha=.7);fig.colorbar(sc,ax=ax);ax.set(title=title,xlabel='PC1',ylabel='PC2')
    dims=[2,4,6,8,12,16,20,24,28];axes[2].plot(dims,[result['primary']['global_reconstruction'][str(d)] for d in dims],'-o')
    axes[2].set(title='Held-out linear compression residual',xlabel='Retained dimensions',ylabel='Residual / total squared distance')
    save(fig,'local_structure')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=read(args.config);setup();start=time.monotonic();out=Path(cfg['job_output']);out.mkdir(parents=True,exist_ok=True)
    if cfg['dataset']=='mimic':
        model,_=load_model(cfg);payload=pickle.loads(Path(cfg['data']).read_bytes());extract_mimic(cfg,model,payload,'train');del model,payload;torch.cuda.empty_cache()
    z,v,metadata=load_states(cfg);pid,_=patient_numbers(metadata);primary=cfg['dataset']=='mimic'
    train_indices=representatives(metadata,'train',0,primary);test_indices=limited(representatives(metadata,'test',0,primary),1024,913)
    train=z[train_indices];test=z[test_indices];tpid=pid[train_indices];qpid=pid[test_indices]
    primary_result,arrays=cloud_statistics(train,test,tpid,qpid)
    result={'primary':primary_result,'controls':{},'prefix_selection_sensitivity':{},'reference_size_sensitivity':{}}
    arrays.update(test_states=test,test_global_ids=test_indices)
    for name,(reference,query) in controls(train,test).items():
        info,_=cloud_statistics(reference,query,torch.arange(len(reference),device='cuda'),torch.arange(len(query),device='cuda')+len(reference),full=False)
        result['controls'][name]=info;print(f'{cfg["dataset"]}: control {name} complete',flush=True)
    for seed in [1,2]:
        indices=representatives(metadata,'train',seed,primary);queries=limited(representatives(metadata,'test',seed,primary),1024,913)
        info,_=cloud_statistics(z[indices],z[queries],pid[indices],pid[queries],full=False)
        result['prefix_selection_sensitivity'][str(seed)]={'pca_d95':info['pca_d95'],'rank':info['entropy_effective_rank'],'mle20':info['scales']['raw']['mle_20'],'twonn':info['scales']['raw']['twonn']}
    for count in sorted({max(200,len(train)//4),max(200,len(train)//2),min(2093,len(train))}):
        ids=limited(torch.arange(len(train),device='cuda'),count,147+count)
        dist,_=knn(test,train[ids],160,qpid,tpid[ids]);est=id_estimates(dist)
        result['reference_size_sensitivity'][str(len(ids))]={'mle20':describe(est['mle_20']),'mle80':describe(est['mle_80']),'twonn':est['twonn']}
    all_train=torch.tensor([i for i,row in enumerate(metadata) if row['split']=='train'],device='cuda');counts=torch.bincount(pid[all_train])
    _,_,weighted_eig,weighted_ratio=pca(z[all_train],1/counts[pid[all_train]])
    result['all_prefix_patient_weighted_pca']={'states':len(all_train),'pca_d95':int(dimension(weighted_ratio,.95)),'entropy_rank':float(effective_rank(weighted_eig)),'pca_ratio':weighted_ratio.tolist()}
    result['provenance']=provenance(cfg,'geometry');result['provenance'].update(elapsed_seconds=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated())
    atomic(out/'summary.json',result);np.savez_compressed(out/'geometry_arrays.npz',**{key:value.cpu().numpy() for key,value in arrays.items()})
    plots(out,result,arrays,test,[metadata[i] for i in test_indices.tolist()]);atomic(out/'complete.json',{'status':'complete'})
    print(f'{cfg["dataset"]} geometry complete',flush=True)


if __name__=='__main__':main()
