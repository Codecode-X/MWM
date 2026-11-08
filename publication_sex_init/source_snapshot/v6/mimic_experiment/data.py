"""Causal ICU contexts, keeping physical age separate from the hourly ODE clock."""
import numpy as np
import torch
from torch.utils.data import Dataset
from paper_odeae.data import collate as base_collate


class ICUContexts(Dataset):
    def __init__(self,sequences,training=False,seed=0,horizon=24):
        self.sequences=sequences;self.training=training;self.seed=seed;self.epoch=0;self.horizon=horizon
        self.indices=[];self.choices=[]
        for i,sequence in enumerate(sequences):
            choices=[]
            for hour in ([6,12,18,24] if training else [horizon]):
                cut=int(np.searchsorted(sequence['ages'],hour,side='right'))
                if cut>=2 and len(sequence['ages'])-cut>=2:choices.append(cut)
            if choices:self.indices.append(i);self.choices.append(choices)
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        choices=self.choices[index]
        if self.training:
            # Choice is invariant to DDP rank, batch composition and worker order.
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,int(self.sequences[self.indices[index]]['subject_id'])]))
            cutoff=choices[int(rng.integers(len(choices)))]
        else:cutoff=choices[0]
        return self.sequences[self.indices[index]],cutoff


def collate(items,dropout=0.):
    batch=base_collate(items,dropout)
    actual=np.array([s['admission_age']+float(s['ages'][cut-1])/(365.25*24) for s,cut in items],np.float32)
    batch['initialization_age']=torch.from_numpy(actual)
    return batch


class ExactBatchSampler:
    """Partition every global batch, without duplicated or discarded patients.

    Rank shards may have different sizes in the last batch; each rank is nonempty.
    A tiny final batch (<world) is merged into the preceding batch.
    """
    def __init__(self,length,batch_size,rank,world,seed=0):
        if length<world or batch_size<world:raise ValueError('Need at least one patient per rank')
        self.length,self.batch_size,self.rank,self.world,self.seed=length,batch_size,rank,world,seed;self.epoch=0
    def set_epoch(self,epoch):self.epoch=epoch
    def batches(self):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch]));order=rng.permutation(self.length).tolist()
        parts=[order[i:i+self.batch_size] for i in range(0,self.length,self.batch_size)]
        if len(parts)>1 and len(parts[-1])<self.world:parts[-2].extend(parts.pop())
        return parts
    def __len__(self):
        n=(self.length+self.batch_size-1)//self.batch_size
        if n>1 and 0<self.length%self.batch_size<self.world:n-=1
        return n
    def __iter__(self):
        for part in self.batches():yield part[self.rank::self.world]
