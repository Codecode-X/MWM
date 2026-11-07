"""CUDA follow-ups: transported state manifolds, clinical tangent energy and information strata."""
import argparse
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from .manifold_core import (setup,read,atomic,pca,effective_rank,dimension,knn,local_pca,project,describe,
    id_estimates,patient_numbers,representatives,limited,load_states,load_model,provenance)
from .manifold_dynamics import decoder_metric,normalized_std


@torch.no_grad()
def evolve(model,z,delta):
    result=[]
    for part in z.split(128):result.append(model.solve(model.forward_ode,part.float(),torch.full((len(part),),delta,device='cuda')).double())
    return torch.cat(result)


@torch.no_grad()
def flow(cfg,model,z,metadata):
    out=Path(cfg['job_output']);pid,_=patient_numbers(metadata);primary=cfg['dataset']=='mimic'
    train_ids=representatives(metadata,'train',0,primary);test_ids=limited(representatives(metadata,'test',0,primary),1024,913)
    train=z[train_ids];test=z[test_ids];train_pid=pid[train_ids];test_pid=pid[test_ids]
    calibration=limited(torch.arange(len(train),device='cuda'),1024,301)
    base_distance,base_nn=knn(train[calibration],train,80,train_pid[calibration],train_pid)
    base_center,base_basis,_=local_pca(train,base_nn)
    thresholds={d:float(torch.quantile((train[calibration]-project(train[calibration],base_center,base_basis,d)).norm(dim=-1)/base_distance[:,-1],.95)) for d in [6,12,20]}
    result={'rows':{},'training_patients':len(train),'test_patients':len(test)};plot_arrays={}
    for delta in ([0.,1.,2.] if cfg['dataset']=='adni' else [0.,12.,24.]):
        transported_train=train if delta==0 else evolve(model,train,delta)
        transported_test=test if delta==0 else evolve(model,test,delta)
        mean,basis,eig,ratio=pca(transported_train)
        ids_dist,_=knn(transported_test,transported_train,160,test_pid,train_pid);ids=id_estimates(ids_dist)
        row={'pca_d95':int(dimension(ratio,.95)),'entropy_rank':float(effective_rank(eig)),
             'mle20':describe(ids['mle_20']),'twonn':ids['twonn'],'planes':{}}
        for label,reference in [('static_reference',train),('transported_reference',transported_train)]:
            distances,neighbors=knn(transported_test,reference,80,test_pid,train_pid)
            center,local_basis,_=local_pca(reference,neighbors)
            row[label]={}
            for d in [6,12,20]:
                residual=(transported_test-project(transported_test,center,local_basis,d)).norm(dim=-1)/distances[:,-1]
                row[label][str(d)]={'residual':describe(residual),'above_initial_training_q95':float((residual>thresholds[d]).double().mean())}
                if d==12:plot_arrays[f'{label}_{delta}']=residual.cpu().numpy()
        result['rows'][str(delta)]=row
        print(f'{cfg["dataset"]}: transported manifold at {delta} complete',flush=True)
    atomic(out/'summary.json',result);np.savez_compressed(out/'flow_arrays.npz',**plot_arrays)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6));horizons=[float(h) for h in result['rows']]
    for name,color in [('static_reference','#cc6677'),('transported_reference','#4477aa')]:
        axes[0].plot(horizons,[result['rows'][str(h)][name]['12']['residual']['median'] for h in horizons],'-o',label=name,color=color)
        axes[1].plot(horizons,[result['rows'][str(h)][name]['12']['above_initial_training_q95']*100 for h in horizons],'-o',label=name,color=color)
    for ax in axes:ax.set_xlabel('ODE time (years)' if cfg['dataset']=='adni' else 'ODE time (hours)');ax.legend(fontsize=8);ax.grid(alpha=.15)
    axes[0].set(ylabel='Median normal residual / neighbor radius',title='Static versus co-evolved reference')
    axes[1].set(ylabel='Above initial training q95 (%)',title='Support threshold under ODE transport');fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(out/f'flow_transport.{ext}',dpi=180)
    plt.close(fig)


@torch.no_grad()
def strata(cfg,z,metadata,payload):
    primary=cfg['dataset']=='mimic';train=representatives(metadata,'train',0,primary);test=limited(representatives(metadata,'test',0,primary),1024,913)
    lookup={s['subject_id']:s for s in payload['sequences']};field={f['name']:j for j,f in enumerate(payload['schema'])};features={}
    for index in torch.cat([train,test]).tolist():
        row=metadata[index];seq=lookup[row['subject_id']];c=row['prefix'];coverage=float(seq['mask'][:c].mean())
        values={'coverage':coverage,'prefix':c}
        if cfg['dataset']=='adni':
            j=field['diagnosis'];observed=np.flatnonzero(seq['mask'][:c,j]);values['diagnosis']=int(seq['values'][observed[-1],j]) if len(observed) else -1
        features[index]=values
    cuts=torch.quantile(torch.tensor([features[i]['coverage'] for i in train.tolist()],device='cuda',dtype=torch.float64),torch.tensor([.25,.5,.75],device='cuda',dtype=torch.float64)).cpu().tolist()
    group_specs=[]
    for q in range(4):group_specs.append((f'coverage_Q{q+1}',lambda values,q=q:int(np.searchsorted(cuts,values['coverage'],side='right'))==q))
    if cfg['dataset']=='adni':
        for k,label in enumerate(['CN','MCI','Dementia']):group_specs.append(('diagnosis_'+label,lambda values,k=k:values['diagnosis']==k))
        group_specs += [('history_1',lambda v:v['prefix']==1),('history_2to3',lambda v:2<=v['prefix']<=3),('history_4plus',lambda v:v['prefix']>=4)]
    result={'coverage_training_cuts':cuts,'groups':{}};generator=torch.Generator(device='cuda').manual_seed(953)
    for label,predicate in group_specs:
        a=torch.tensor([i for i in train.tolist() if predicate(features[i])],device='cuda',dtype=torch.long)
        b=torch.tensor([i for i in test.tolist() if predicate(features[i])],device='cuda',dtype=torch.long)
        if len(a)<80 or len(b)<12:continue
        reference=z[a];query=z[b];mean,basis,eig,ratio=pca(reference)
        distance,_=knn(query,reference,40);estimate=id_estimates(distance)
        gauss_train=mean+(torch.randn(len(a),28,device='cuda',dtype=torch.float64,generator=generator)*eig.sqrt())@basis.T
        gauss_test=mean+(torch.randn(len(b),28,device='cuda',dtype=torch.float64,generator=generator)*eig.sqrt())@basis.T
        null_dist,_=knn(gauss_test,gauss_train,40);null=id_estimates(null_dist)
        result['groups'][label]={'train_patients':len(a),'test_patients':len(b),'pca_d95':int(dimension(ratio,.95)),
              'mle20':describe(estimate['mle_20']),'covariance_gaussian_mle20':describe(null['mle_20']),'twonn':estimate['twonn']}
    return result


@torch.no_grad()
def clinical(cfg,model,z,metadata,payload):
    rows=read(Path(cfg['output'])/(cfg['dataset']+'_dynamics')/'forecast_rows.json');unique={}
    for row in rows:
        if row['split']=='test':unique.setdefault(row['subject_id'],row)
    lookup={(r['subject_id'],r['prefix']):i for i,r in enumerate(metadata)}
    selected=[row for row in unique.values() if (row['subject_id'],row['cutoff']) in lookup]
    indices=limited(torch.arange(len(selected),device='cuda'),512,37)
    selected=[selected[i] for i in indices.tolist()];qids=torch.tensor([lookup[(r['subject_id'],r['cutoff'])] for r in selected],device='cuda')
    pid,_=patient_numbers(metadata);train=torch.tensor([i for i,row in enumerate(metadata) if row['split']=='train'],device='cuda')
    q=z[qids];distance,neighbors=knn(q,z[train],80,pid[qids],pid[train],distinct=True);center,basis,_=local_pca(z[train],neighbors)
    velocity=model.forward_ode(q.float()).double();g=decoder_metric(model,q,normalized_std(payload))
    energy=(velocity[:,None]@g@velocity[:,:,None]).flatten().clamp_min(1e-20)
    eps=g.diagonal(dim1=-1,dim2=-2).sum(-1)/28*1e-4;result={'patients':len(q),'dimensions':{}}
    for d in [6,12,20]:
        b=basis[:,:,:d];tangent=(b@(b.transpose(1,2)@velocity[:,:,None])).squeeze(-1);normal=velocity-tangent
        tangent_energy=(tangent[:,None]@g@tangent[:,:,None]).flatten().clamp_min(0)
        normal_energy=(normal[:,None]@g@normal[:,:,None]).flatten().clamp_min(0)
        cosine=(velocity[:,None]@g@tangent[:,:,None]).flatten()/(energy*tangent_energy).clamp_min(1e-30).sqrt()
        bg=b.transpose(1,2)@g;bgb=bg@b
        coefficients=torch.linalg.solve(bgb+eps[:,None,None]*torch.eye(d,device='cuda'),bg@velocity[:,:,None])
        riemann_tangent=(b@coefficients).squeeze(-1);residual=velocity-riemann_tangent
        result['dimensions'][str(d)]={'euclidean_normal_energy':describe(normal.square().sum(1)/velocity.square().sum(1)),
             'clinical_normal_component_energy':describe(normal_energy/(normal_energy+tangent_energy).clamp_min(1e-30)),
             'clinical_cosine_after_euclidean_tangency':describe(cosine),
             'clinical_energy_lost_after_metric_orthogonal_tangency':describe((residual[:,None]@g@residual[:,:,None]).flatten()/energy),
             'metric_tangent_speed_inflation':describe(riemann_tangent.norm(dim=-1)/velocity.norm(dim=-1).clamp_min(1e-30))}
    result['conditional_dimension']=strata(cfg,z,metadata,payload)
    atomic(Path(cfg['job_output'])/'summary.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);args=parser.parse_args();cfg=read(args.config)
    setup();started=time.monotonic();out=Path(cfg['job_output']);out.mkdir(parents=True,exist_ok=True)
    model,_=load_model(cfg);z,v,metadata=load_states(cfg)
    if cfg['stage']=='flow':flow(cfg,model,z,metadata)
    else:clinical(cfg,model,z,metadata,pickle.loads(Path(cfg['data']).read_bytes()))
    result=read(out/'summary.json');result['provenance']=provenance(cfg,cfg['stage']);result['provenance'].update(elapsed_seconds=time.monotonic()-started,peak_memory_bytes=torch.cuda.max_memory_allocated())
    atomic(out/'summary.json',result);atomic(out/'complete.json',{'status':'complete'});print(f'{cfg["dataset"]} {cfg["stage"]} complete',flush=True)


if __name__=='__main__':
    with torch.no_grad():main()
