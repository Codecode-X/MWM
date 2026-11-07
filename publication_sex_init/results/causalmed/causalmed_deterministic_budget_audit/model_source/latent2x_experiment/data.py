"""Public facts only, training-fitted transformations and deterministic history cutoffs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from causalmed_client import PublicDataset

FEATURES=['Y_obs','X1_obs','X2_obs']


def load_public(root):
    datasets={split:PublicDataset(root,split) for split in ['train','val','test']}
    raw={key:list(dataset) for key,dataset in datasets.items()}
    train=np.concatenate([np.asarray(s['observations'],np.float64) for s in raw['train']])
    offset=train.mean(0);scale=np.maximum(train.std(0),1e-6)
    static=np.asarray([s['c'] for s in raw['train']]);c_mean=float(static.mean());c_std=max(float(static.std()),1e-6)
    result={}
    for split,sequences in raw.items():
        converted=[]
        for record in sequences:
            times=np.asarray(record['times'],np.float64);n=len(times);actions=np.full(n,-1,np.int64)
            intervals=record['treatment_intervals']
            expected=n-1 if split=='test' else n
            if len(intervals)!=expected:raise ValueError('One treatment interval per decision, excluding the test fork')
            for i,interval in enumerate(intervals):
                if abs(interval['start']-times[i])>1e-9 or interval['treatment'] not in [0,1]:raise ValueError('Treatment chronology mismatch')
                if i+1<n and abs(interval['end']-times[i+1])>1e-9:raise ValueError('Treatment interval does not end at the next visit')
                actions[i]=interval['treatment']
            values=np.asarray(record['observations'],np.float64)
            if not np.isfinite(values).all() or not np.all(np.diff(times)>0):raise ValueError('Invalid public facts')
            converted.append({'patient_id':record['patient_id'],'times':times,'values':((values-offset)/scale).astype(np.float32),
                              'raw_values':values,'mask':np.asarray(record['observation_mask'],np.float32),
                              'actions':actions,'c':float((record['c']-c_mean)/c_std)})
        result[split]=converted
    identities={key:{s['patient_id'] for s in sequences} for key,sequences in result.items()}
    assert not (identities['train']&identities['val'] or identities['train']&identities['test'] or identities['val']&identities['test'])
    metadata={'patient_counts':{key:len(sequences) for key,sequences in result.items()},'normalization':'train_zscore',
              'offset':offset.tolist(),'scale':scale.tolist(),'static_c_mean':c_mean,'static_c_std':c_std,
              'feature_order':FEATURES,'time_unit':'year','patient_splits_disjoint':True,
              'test_has_no_action_at_fork':all(s['actions'][-1]==-1 for s in result['test'])}
    result['metadata']=metadata
    return result,datasets['test']


def cutoff(sequence,epoch=None,seed=0):
    if epoch is None:return int(np.searchsorted(sequence['times'],3.,side='left'))+1
    choices=sorted({int(np.searchsorted(sequence['times'],year,side='left'))+1 for year in [1.,2.,3.,4.]})
    choices=[c for c in choices if 2<=c<len(sequence['times'])]
    ident=int(sequence['patient_id'].rsplit('_',1)[-1]);rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,ident]))
    return choices[int(rng.integers(len(choices)))]


def collate(items,dropout=0.):
    b=len(items);n=max(len(s['times']) for s,_ in items)
    times=np.zeros((b,n),np.float32);values=np.zeros((b,n,3),np.float32);mask=np.zeros_like(values)
    context=np.zeros_like(values);valid=np.zeros((b,n),bool);actions=np.zeros((b,n),np.int64);cuts=[];static=[]
    for i,(s,c) in enumerate(items):
        size=len(s['times']);times[i,:size]=s['times'];times[i,size:]=s['times'][-1]
        values[i,:size]=s['values'];mask[i,:size]=s['mask'];context[i,:c]=s['mask'][:c]
        valid[i,:size]=True;actions[i,:size]=s['actions'];cuts.append(c);static.append(s['c'])
    if dropout:context*=np.random.random(context.shape)>=dropout
    return {'times':torch.from_numpy(times),'values':torch.from_numpy(values),'mask':torch.from_numpy(mask),
            'context_mask':torch.from_numpy(context),'valid':torch.from_numpy(valid),'actions':torch.from_numpy(actions),
            'cutoff':torch.tensor(cuts),'c':torch.tensor(static,dtype=torch.float32),'max_context':max(cuts),
            'patient_ids':[s['patient_id'] for s,_ in items]}


def cuda_batch(batch):
    return {key:value.cuda() if torch.is_tensor(value) else value for key,value in batch.items()}


def atomic(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def public_hashes(root):
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(root).iterdir()) if p.is_file()}
