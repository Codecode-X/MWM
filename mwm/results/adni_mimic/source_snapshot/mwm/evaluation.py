"""Causal validation and the existing cutoff-conditioned reconstruction test."""
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader,Subset

from paper_odeae.models import feature_errors
from paper_odeae.train import device_batch
from .data import collate,contexts


def loaders(sequences,cfg,horizon=24):
    dataset=contexts(sequences,cfg['dataset_name'],horizon=horizon)
    if cfg['dataset_name']=='mimic':
        for shard in range(4):
            yield DataLoader(Subset(dataset,range(shard,len(dataset),4)),batch_size=32,
                             collate_fn=partial(collate,dataset='mimic'))
    else:
        yield DataLoader(dataset,batch_size=32,collate_fn=partial(collate,dataset='adni'))


def summarize(total,schema):
    count=total[1];means=total[0]/count.clamp_min(1)
    numeric=torch.tensor([f['kind']=='numerical' for f in schema],device=total.device)
    mse=means[numeric].sum();cce=means[~numeric].sum()
    return {'reconstruction_error':float(mse+cce),'sum_feature_mse':float(mse),'sum_feature_cce':float(cce),
            'observed_scalars':int(count.sum()),'per_field':{
                f['name']:{'loss':float(means[j]),'observations':int(count[j])}
                for j,f in enumerate(schema) if count[j]>0}}


@torch.no_grad()
def evaluate_predictive(model,sequences,schema,cfg,horizon=24):
    model.eval();keys=['overall','interpolation','bootstrap','history_preupdate','forecast']
    totals={k:torch.zeros(2,len(schema),device='cuda',dtype=torch.float64) for k in keys}
    uncertainty=torch.zeros(5,len(schema),device='cuda',dtype=torch.float64)
    for loader in loaders(sequences,cfg,horizon):
        for cpu in loader:
            batch=device_batch(cpu);out=model(batch)
            errors=feature_errors(out['prediction'],batch['values'],batch['mask'],schema)
            index=torch.arange(batch['values'].shape[1],device='cuda')[None,:]
            history=index<batch['cutoff'][:,None];future=(~history)&batch['valid']
            regions={'overall':batch['valid'],'interpolation':history,'bootstrap':index==0,
                     'history_preupdate':history&(index>0),'forecast':future}
            for key,region in regions.items():
                totals[key][0]+=(errors*region[...,None]).sum((0,1)).double()
                totals[key][1]+=(batch['mask']*region[...,None]).sum((0,1)).double()
            if 'sigma' in out:
                sigma=out['sigma'].squeeze(-1);at=0
                for j,field in enumerate(schema):
                    width=field.get('classes',1)
                    if field['kind']=='numerical':
                        observed=batch['mask'][...,j]*future
                        residual=(out['prediction'][...,at]-batch['values'][...,j]).abs()
                        numbers=[residual.square()/(2*sigma.square())+sigma.log(),
                                 (residual<=sigma).float(),(residual<=1.959964*sigma).float(),sigma,torch.ones_like(sigma)]
                        uncertainty[:,j]+=torch.stack([(x*observed).sum().double() for x in numbers])
                    at+=width
    result={key:summarize(value,schema) for key,value in totals.items()}
    result['history_semantics']='first observation after UPDATE; remaining history before UPDATE'
    if model.variant=='B':
        result['forecast_uncertainty']={f['name']:{
            'nll_without_constant':float(uncertainty[0,j]/uncertainty[4,j]),
            'coverage_68_percent':float(uncertainty[1,j]/uncertainty[4,j]),
            'coverage_95_percent':float(uncertainty[2,j]/uncertainty[4,j]),
            'mean_sigma_normalized':float(uncertainty[3,j]/uncertainty[4,j]),
            'observations':int(uncertainty[4,j])}
            for j,f in enumerate(schema) if uncertainty[4,j]>0}
    return result


class RetrospectiveAdapter(nn.Module):
    """Expose old metrics without letting inverse integration enter training."""
    collect_prior_predictions=False
    collect_jacobian_states=False

    def __init__(self,model,sequences,dataset):
        super().__init__();self.model=model
        self.first_ages={s['subject_id']:float(s['ages'][0])
                         for s in sequences} if dataset=='adni' else {
                             s['subject_id']:s['admission_age']+float(s['ages'][0])/8766. for s in sequences}

    @torch.no_grad()
    def forward(self,batch,sample=False):
        enriched={**batch,'first_observation_age':batch['values'].new_tensor([self.first_ages[x] for x in batch['subject_ids']])}
        out=self.model(enriched);past=self.model.retrospective_history(enriched,out['mu'])
        prediction=out['prediction'].clone();index=torch.arange(past.shape[1],device=prediction.device)[None,:]
        prediction[:,:past.shape[1]]=torch.where((index<batch['cutoff'][:,None])[...,None],past,prediction[:,:past.shape[1]])
        return {'prediction':prediction,'mu':out['mu'],'kl':out['kl']}


def evaluate_comparable(model,sequences,schema,cfg,metadata,modes,horizon=24):
    adapter=RetrospectiveAdapter(model,sequences,cfg['dataset_name'])
    if cfg['dataset_name']=='mimic':
        from mimic_experiment.evaluation import evaluate
        return evaluate(adapter,sequences,schema,cfg,metadata['raw_train_std'],modes,horizon=horizon)
    from paper_odeae.train import evaluate
    return evaluate(adapter,sequences,schema,32)
