"""CUDA primitives for patient-balanced intrinsic dimension and frozen geometry probes."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from paper_odeae.models import paper_model_from_config,encode_values
from paper_odeae.data import collate as adni_collate
from mimic_experiment.data import ICUContexts,collate as icu_collate


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,obj):
    path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def read(path):return json.loads(Path(path).read_text())


def setup():
    if not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') not in ['0','1','2','3']:raise RuntimeError('Explicit physical CUDA GPU required')
    torch.cuda.set_device(0);torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
    torch.cuda.set_per_process_memory_fraction(.12)
    torch.backends.cuda.matmul.allow_tf32=False


def pca(z,weights=None):
    if z.device.type!='cuda':raise RuntimeError('Geometry must execute on CUDA')
    x=z.double();w=x.new_ones(len(x)) if weights is None else weights.double();w=w/w.sum()
    mean=(x*w[:,None]).sum(0);c=x-mean;cov=(c*w[:,None]).T@c
    eig,basis=torch.linalg.eigh(cov);eig=eig.flip(-1).clamp_min(0);basis=basis.flip(-1)
    ratio=eig/eig.sum().clamp_min(1e-30)
    return mean,basis,eig,ratio


def effective_rank(eig):
    p=eig.clamp_min(0)/eig.clamp_min(0).sum(-1,keepdim=True).clamp_min(1e-30)
    return (-(p*p.clamp_min(1e-30).log()).sum(-1)).exp()


def dimension(ratio,fraction):return (ratio.cumsum(-1)<fraction).sum(-1)+1


@torch.no_grad()
def knn(query,reference,k,query_pid=None,reference_pid=None,distinct=False):
    distances=[];indices=[]
    for start in range(0,len(query),64):
        q=query[start:start+64].double();dist=torch.cdist(q,reference.double(),compute_mode='donot_use_mm_for_euclid_dist')
        if query_pid is not None:dist.masked_fill_(query_pid[start:start+64,None]==reference_pid[None],float('inf'))
        if distinct:
            people=int(reference_pid.max())+1;scatter=reference_pid[None].expand(len(q),-1)
            minimum=dist.new_full((len(q),people),float('inf'));minimum.scatter_reduce_(1,scatter,dist,reduce='amin',include_self=True)
            values,pids=minimum.topk(k,largest=False,sorted=True)
            locations=torch.arange(len(reference),device='cuda')[None].expand(len(q),-1)
            candidates=torch.where(dist==minimum.gather(1,scatter),locations,len(reference))
            argmin=torch.full((len(q),people),len(reference),device='cuda',dtype=torch.long);argmin.scatter_reduce_(1,scatter,candidates,reduce='amin',include_self=True)
            ids=argmin.gather(1,pids)
        else:values,ids=dist.topk(k,largest=False,sorted=True)
        if not torch.isfinite(values).all():raise ValueError('Insufficient distinct patients')
        distances.append(values);indices.append(ids)
    return torch.cat(distances),torch.cat(indices)


def local_pca(reference,neighbors):
    means=[];bases=[];spectra=[]
    for indices in neighbors.split(128):
        x=reference[indices].double();mean=x.mean(1);centered=x-mean[:,None]
        cov=centered.transpose(1,2)@centered/max(x.shape[1]-1,1);eig,basis=torch.linalg.eigh(cov)
        means.append(mean);bases.append(basis.flip(-1));spectra.append(eig.flip(-1).clamp_min(0))
    return torch.cat(means),torch.cat(bases),torch.cat(spectra)


def project(z,mean,basis,d):
    b=basis[...,:d];v=(z-mean).double()
    return mean+(b@(b.transpose(-2,-1)@v[...,None])).squeeze(-1)


def id_estimates(distances):
    result={}
    for k in [10,20,40,80,160]:
        if k>distances.shape[1]:continue
        x=distances[:,:k];logs=(x[:,-1:]/x[:,:-1].clamp_min(1e-15)).log().sum(-1)
        result[f'mle_{k}']=(k-2)/logs.clamp_min(1e-15)
    valid=(distances[:,0]>1e-12)&(distances[:,1]>distances[:,0]*(1+1e-10))
    log_mu=(distances[valid,1]/distances[valid,0]).log();sorted_x=log_mu.sort().values
    cdf=(torch.arange(len(sorted_x),device='cuda',dtype=torch.float64)+.5)/len(sorted_x);keep=cdf<.95
    twonn=(sorted_x[keep]*(-torch.log1p(-cdf[keep]))).sum()/sorted_x[keep].square().sum().clamp_min(1e-30)
    result['twonn']=float(twonn);result['twonn_mle']=float(1/log_mu.mean());result['twonn_valid']=int(valid.sum())
    return result


def describe(x):
    x=x.detach().double();x=x[torch.isfinite(x)]
    if not len(x):return {'n':0,'mean':None,'median':None,'q25':None,'q75':None}
    q=torch.quantile(x,x.new_tensor([.05,.25,.5,.75,.95]))
    return {'n':len(x),'mean':float(x.mean()),'median':float(q[2]),'q05':float(q[0]),'q25':float(q[1]),'q75':float(q[3]),'q95':float(q[4])}


def patient_numbers(metadata):
    ids={pid:i for i,pid in enumerate(sorted({row['subject_id'] for row in metadata}))}
    return torch.tensor([ids[row['subject_id']] for row in metadata],device='cuda'),ids


def representatives(metadata,split,seed=0,primary_only=False):
    groups={}
    for i,row in enumerate(metadata):
        if row['split']==split and (not primary_only or row.get('context_hours')==24):groups.setdefault(row['subject_id'],[]).append(i)
    chosen=[]
    for pid,indices in sorted(groups.items()):
        n=int(hashlib.sha256(f'manifold/{seed}/{pid}'.encode()).hexdigest()[:16],16)
        chosen.append(indices[n%len(indices)])
    return torch.tensor(chosen,device='cuda',dtype=torch.long)


def limited(indices,maximum,seed):
    if len(indices)<=maximum:return indices
    generator=torch.Generator(device='cuda').manual_seed(seed)
    return indices[torch.randperm(len(indices),device='cuda',generator=generator)[:maximum]]


def load_model(cfg):
    saved=torch.load(cfg['checkpoint'],map_location='cuda',weights_only=False)
    model=paper_model_from_config(saved['schema'],saved['config']).cuda();model.load_state_dict(saved['state_dict']);model.eval()
    assert model.latent==28 and model.kind=='deterministic' and model.shared_vector_field and model.encoder_direction=='forward'
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.collect_prior_predictions=False;model.collect_jacobian_states=False
    return model,saved


@torch.no_grad()
def encode(model,batch,return_before=False):
    ages,cutoff=batch['ages'],batch['cutoff'];row=torch.arange(len(cutoff),device='cuda')
    age=batch.get('initialization_age',ages[row,cutoff-1]);h=model.init(age[:,None]/100.)
    x=encode_values(batch['values'],batch['context_mask'],model.schema);previous=ages[:,0];before=h
    for i in range(batch['max_context']):
        index=torch.minimum(torch.full_like(cutoff,i),cutoff-1);alive=i<cutoff;time_=ages[row,index]
        evolved=model.solve(model.forward_ode,h,torch.where(alive,time_-previous,torch.zeros_like(previous)))
        before=torch.where((i==cutoff-1)[:,None],evolved,before)
        updated=model.update(torch.cat([evolved,x[row,index]],-1));h=torch.where(alive[:,None],updated,h);previous=torch.where(alive,time_,previous)
    return (h,before) if return_before else h


def cuda_batch(batch):return {key:value.cuda() if torch.is_tensor(value) else value for key,value in batch.items()}


@torch.no_grad()
def extract_mimic(cfg,model,payload,role):
    out=Path(cfg['output']);path=out/f'mimic_states_{role}.npz';metadata_path=out/f'mimic_metadata_{role}.json'
    if path.exists() and metadata_path.exists():return
    states=[];velocities=[];metadata=[];fields={f['name']:j for j,f in enumerate(payload['schema'])};verified=False
    for split in (['train'] if role=='train' else ['validation','test']):
        sequences=[s for s in payload['sequences'] if s['split']==split]
        for hours in ([6,12,18,24] if role=='train' else [24]):
            dataset=ICUContexts(sequences,horizon=hours)
            for cpu in DataLoader(dataset,batch_size=64,collate_fn=icu_collate):
                batch=cuda_batch(cpu);z=encode(model,batch)
                if not verified:
                    small={key:value[:4] if torch.is_tensor(value) else value for key,value in batch.items()};small['max_context']=int(small['cutoff'].max())
                    actual=encode(model,small);torch.testing.assert_close(actual,model(small)['mu'],rtol=0,atol=0);verified=True
                states.append(z.cpu().numpy());velocities.append(model.forward_ode(z).cpu().numpy())
                for i,c in enumerate(cpu['cutoff'].tolist()):
                    masks=cpu['mask'][i,:c].numpy();metadata.append({'subject_id':cpu['subject_ids'][i],'split':split,'prefix':c,'context_hours':hours,
                        'clock':float(cpu['ages'][i,c-1]),'age':float(cpu['initialization_age'][i]),'coverage':float(masks.mean())})
            print(f'MIMIC {split}, {hours}h states extracted',flush=True)
    tmp=path.with_name(path.stem+'.partial.npz');np.savez_compressed(tmp,states=np.concatenate(states),velocities=np.concatenate(velocities));tmp.replace(path);atomic(metadata_path,metadata)


def load_states(cfg):
    if cfg['dataset']=='adni':
        source=Path(cfg['state_source']);arrays=np.load(source/'states.npz');metadata=read(source/'metadata.json')
        states=torch.tensor(arrays['states'],device='cuda',dtype=torch.float64);velocities=torch.tensor(arrays['velocities'],device='cuda',dtype=torch.float64)
    else:
        out=Path(cfg['output'])
        while not all((out/f'mimic_states_{r}.npz').exists() and (out/f'mimic_metadata_{r}.json').exists() for r in ['train','heldout']):time.sleep(2)
        packs=[np.load(out/f'mimic_states_{r}.npz') for r in ['train','heldout']]
        states=torch.cat([torch.tensor(pack['states'],device='cuda',dtype=torch.float64) for pack in packs])
        velocities=torch.cat([torch.tensor(pack['velocities'],device='cuda',dtype=torch.float64) for pack in packs])
        metadata=read(out/'mimic_metadata_train.json')+read(out/'mimic_metadata_heldout.json')
    return states,velocities,metadata


def provenance(cfg,stage):
    return {'dataset':cfg['dataset'],'stage':stage,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'gpu_name':torch.cuda.get_device_name(),
            'device':'cuda:0','environment':Path(os.sys.prefix).name,'checkpoint':cfg['checkpoint'],'checkpoint_sha256':sha(cfg['checkpoint']),
            'dataset_sha256':sha(cfg['data']),'training_performed':False,'memory_budget_fraction':.12,'geometry_precision':'float64 CUDA',
            'model_precision':'float32 CUDA','torch_version':torch.__version__}
