"""MWM data tasks preserve patient splits while supplying first-observation age."""
import numpy as np
import torch

from paper_odeae.data import Prefixes,collate as adni_collate
from mimic_experiment.data import ICUContexts,collate as mimic_collate


def collate(items,dataset,dropout=0.):
    batch=(mimic_collate if dataset=='mimic' else adni_collate)(items,dropout)
    age=np.array([s['admission_age']+float(s['ages'][0])/8766. if dataset=='mimic' else float(s['ages'][0]) for s,_ in items],dtype=np.float32)
    batch['first_observation_age']=torch.from_numpy(age)
    return batch


def contexts(sequences,dataset,training=False,seed=0,horizon=24):
    return ICUContexts(sequences,training=training,seed=seed,horizon=horizon) if dataset=='mimic' else Prefixes(sequences)
