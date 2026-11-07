"""Multi-cutoff forecasting and fully masked historical retrospection supervision."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from mimic_experiment import train_full_batch as native
from mimic_experiment.data import ICUContexts,collate as base_collate
from mimic_experiment.prepare import atomic
from mimic_experiment.train_future_only import FutureOnlyModel,future_mask,loss_function as future_only_loss
from paper_odeae.models import paper_model_from_config,encode_values,feature_errors,reconstruction
from paper_odeae.jacobian import jacobian_regularization
from src.tracking import SwanLabTracker


class TaskContexts(ICUContexts):
    def __getitem__(self,index):
        sequence,cut=super().__getitem__(index)
        return ({**sequence,'_supervision_epoch':self.epoch} if self.training else sequence),cut


def task_collate(items,dropout=0.,*,cfg):
    batch=base_collate(items,dropout);b=len(items)
    if cfg['supervision_ablation']=='A':
        slots=1+cfg['additional_cutoffs'];cuts=np.ones((b,slots),np.int64);weights=np.zeros((b,slots),np.float32)
        init=np.zeros((b,slots),np.float32)
        for row,(s,cut) in enumerate(items):
            rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],s.get('_supervision_epoch',0),int(s['subject_id']),1201]))
            q=min(cfg['additional_cutoffs'],cut-1);sample=rng.choice(np.arange(1,cut),q,replace=False)
            cuts[row,0]=cut;weights[row,0]=1./cut
            if q:cuts[row,1:1+q]=sample;weights[row,1:1+q]=(cut-1)/(cut*q)
            for col,c in enumerate(cuts[row]):init[row,col]=s['admission_age']+float(s['ages'][c-1])/8766.
        batch.update(loss_cutoffs=torch.from_numpy(cuts),loss_cutoff_weights=torch.from_numpy(weights),loss_initialization_ages=torch.from_numpy(init))
    else:
        indices=np.full(b,-1,np.int64)
        for row,(s,cut) in enumerate(items):
            if cut>=3:
                rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],s.get('_supervision_epoch',0),int(s['subject_id']),1202]))
                indices[row]=int(rng.integers(1,cut-1))
        batch['masked_history_index']=torch.from_numpy(indices)
    return batch


def prefix_batch(batch,slot):
    cut=batch['loss_cutoffs'][:,slot];index=torch.arange(batch['ages'].shape[1],device=cut.device)[None,:]
    return {**batch,'cutoff':cut,'max_context':int(cut.max()),'context_mask':batch['context_mask']*(index<cut[:,None])[...,None],
            'initialization_age':batch['loss_initialization_ages'][:,slot]}


class SupervisionModel(FutureOnlyModel):
    def masked_retrospection(self,batch):
        index=batch['masked_history_index'];eligible=index>=0;values=batch['values'];n=values.shape[1]
        keep=(torch.arange(n,device=values.device)[None,:]!=index[:,None])&eligible[:,None]
        context=batch['context_mask']*keep[...,None];inputs=encode_values(values,context,self.schema)
        state=self.init(batch['initialization_age'][:,None]/100.);previous=batch['ages'][:,0]
        for k in range(batch['max_context']):
            active=(k<batch['cutoff'])&keep[:,k]
            if not bool(active.any()):continue
            time=batch['ages'][:,k]
            evolved=self.solve(self.forward_ode,state,torch.where(active,time-previous,torch.zeros_like(time)))
            proposal=self.update(torch.cat([evolved,inputs[:,k]],-1))
            state=torch.where(active[:,None],proposal,state);previous=torch.where(active,time,previous)
        anchor=state;target_time=batch['ages'].gather(1,index.clamp_min(0)[:,None]).squeeze(1)
        delta=torch.where(eligible,target_time-previous,torch.zeros_like(previous))
        retrospective=self.solve(self.forward_ode,anchor,delta)
        return {'prediction':self.decoder(retrospective)[:,None,:],'anchor':anchor,'eligible':eligible,'backward_delta':delta}

    def forward(self,batch,sample=False):
        result=super().forward(batch,sample=sample)
        if not self.training:return result
        if self.supervision_cfg['supervision_ablation']=='A':
            paths=[dict(result)]
            for slot in range(1,batch['loss_cutoffs'].shape[1]):
                paths.append(super().forward(prefix_batch(batch,slot),sample=False))
            result['forecast_paths']=paths
        else:result['masked_retrospection']=self.masked_retrospection(batch)
        if not getattr(self,'sampling_audited',False):
            self.sampling_audit={'mode':self.supervision_cfg['supervision_ablation'],'physical_gpu':self.supervision_cfg['physical_gpu'],
                'tensor_device':str(batch['values'].device),'patients':len(batch['cutoff']),'first_eight_original_cutoffs':batch['cutoff'][:8].detach().cpu().tolist()}
            if self.supervision_cfg['supervision_ablation']=='A':
                self.sampling_audit.update(first_eight_path_cutoffs=batch['loss_cutoffs'][:8].detach().cpu().tolist(),
                    first_eight_cutoff_weights=batch['loss_cutoff_weights'][:8].detach().cpu().tolist())
            else:self.sampling_audit['first_eight_masked_indices']=batch['masked_history_index'][:8].detach().cpu().tolist()
        return result


def make_model(schema,cfg):
    if cfg['prior_weight']!=0 or cfg.get('train_reconstruct_history') is not False:raise ValueError('Start from weak_prior=0, future-only')
    model=paper_model_from_config(schema,cfg,model_class=SupervisionModel);model.supervision_cfg=cfg
    model.variant=cfg['variant'];model.generator_anchor=cfg['generator_anchor'];return model


def multi_cutoff_loss(model,out,batch,schema,cfg):
    predictions=torch.stack([path['prediction'] for path in out['forecast_paths']],1)
    b,s,n,width=predictions.shape;f=len(schema)
    predictions=torch.where((batch['loss_cutoff_weights']>0)[:,:,None,None],predictions,torch.zeros_like(predictions))
    targets=batch['values'][:,None].expand(-1,s,-1,-1);observed=batch['mask'][:,None].expand(-1,s,-1,-1)
    errors=feature_errors(predictions.reshape(b*s,n,width),targets.reshape(b*s,n,f),observed.reshape(b*s,n,f),schema).reshape(b,s,n,f)
    cuts=batch['loss_cutoffs'];anchor_times=batch['ages'].gather(1,cuts-1);dt=batch['ages'][:,None,:]-anchor_times[:,:,None]
    is_future=(torch.arange(n,device=cuts.device)[None,None,:]>=cuts[:,:,None])&batch['valid'][:,None,:]
    short,medium=cfg['span_boundaries_hours']
    masks=[is_future&(dt>0)&(dt<=short),is_future&(dt>short)&(dt<=medium),is_future&(dt>medium)]
    numeric=torch.tensor([x['kind']=='numerical' for x in schema],device=cuts.device)
    losses=[];masses=[];diagnostics={}
    for label,region,mass in zip(['short','medium','long'],masks,cfg['span_weights']):
        selected=observed*region[...,None];count=selected.sum(2)
        mean=(errors*region[...,None]).sum(2)/count.clamp_min(1)
        weights=batch['loss_cutoff_weights'][...,None]*(count>0)
        field=(mean*weights).sum((0,1))/weights.sum((0,1)).clamp_min(1e-12)
        mse=field[numeric].sum();ce=field[~numeric].sum()
        if bool(weights.sum()>0):losses.append(torch.stack([mse,ce]));masses.append(mass)
        sampled=batch['loss_cutoff_weights']>0
        diagnostics.update({f'A_{label}_mse':mse.detach(),f'A_{label}_ce':ce.detach(),f'A_{label}_target_scalars':(selected*sampled[:,:,None,None]).sum().detach()})
    if not losses:raise ValueError('Multi-cutoff targets are empty')
    mass=predictions.new_tensor(masses);mse,ce=(torch.stack(losses)*mass[:,None]).sum(0)/mass.sum()
    diagnostics['A_mean_sampled_cutoffs']=(batch['loss_cutoff_weights']>0).float().sum(1).mean()
    return mse,ce,diagnostics


def masked_loss(out,batch,schema):
    aux=out['masked_retrospection'];index=batch['masked_history_index'].clamp_min(0)
    gather=index[:,None,None].expand(-1,1,len(schema))
    target=batch['values'].gather(1,gather);mask=batch['mask'].gather(1,gather)*aux['eligible'][:,None,None]
    mse,ce=reconstruction(feature_errors(aux['prediction'],target,mask,schema),mask,schema)
    return mse,ce


def loss_function(model,out,batch,schema,cfg):
    if cfg['supervision_ablation']=='A':
        mse,ce,diagnostics=multi_cutoff_loss(model,out,batch,schema,cfg)
        future=future_mask(batch);mask=batch['mask']*future[...,None]
        fmse,fce=reconstruction(feature_errors(out['prediction'],batch['values'],mask,schema),mask,schema)
        jac,_,_=jacobian_regularization(model,out,{**batch,'valid':future});l2=model.ode_regularization()
        loss=cfg['mse_weight']*mse+cfg['cce_weight']*ce+cfg['jacobian_weight']*jac+cfg['l2_weight']*l2
        vector=torch.stack([mse.detach(),ce.detach(),fmse.detach(),fce.detach(),jac.detach(),l2.detach()])
    else:
        # Preserve the original future objective, normalization and gradients.
        base,parts=future_only_loss(model,out,batch,schema,cfg)
        mse,ce=masked_loss(out,batch,schema);weight=cfg['masked_retrospection_weight'];loss=base+weight*(mse+ce)
        vector=torch.stack([parts[0],parts[1],mse.detach(),ce.detach(),parts[4],parts[5]])
        aux=out['masked_retrospection'];diagnostics={'B_masked_mse':mse.detach(),'B_masked_ce':ce.detach(),
            'B_eligible_patients':aux['eligible'].sum(),'B_mean_backward_hours':-aux['backward_delta'].sum()/aux['eligible'].sum().clamp_min(1)}
    model.loss_diagnostics=diagnostics
    return loss,vector


def components(vector,cfg):
    a,b,c,d,jac,l2,norm=[float(v) for v in vector]
    if cfg['supervision_ablation']=='A':
        mse,ce=a,b;extra={'multi_cutoff_mse':a,'multi_cutoff_ce':b,'final_future_mse':c,'final_future_ce':d}
    else:
        w=cfg['masked_retrospection_weight'];mse,ce=a+w*c,b+w*d
        extra={'future_mse':a,'future_ce':b,'masked_mse':c,'masked_ce':d,'weighted_masked':w*(c+d)}
    return {'mse':mse,'cce':ce,'prior_mse':None,'prior_cce':None,'weighted_prior':0.,'kl':0.,
            'jacobian_frobenius_squared':jac,'ode_l2':l2,'gradient_norm':norm,
            'weighted_jacobian':cfg['jacobian_weight']*jac,'weighted_ode_l2':cfg['l2_weight']*l2,
            'loss':mse+ce+cfg['jacobian_weight']*jac+cfg['l2_weight']*l2,**extra}


class SupervisionTracker(SwanLabTracker):
    def __enter__(self):
        result=super().__enter__();self.diag_sums={};self.diag_batches=0
        self.diag_history=json.loads((self.output/'supervision_epoch_diagnostics.json').read_text()) if (self.output/'supervision_epoch_diagnostics.json').exists() else []
        return result
    def batch(self,loss,lr,**values):
        if not getattr(self.model,'sampling_audited',False):
            atomic(self.output/'first_batch_supervision_audit.json',self.model.sampling_audit);self.model.sampling_audited=True
        diagnostics={k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in self.model.loss_diagnostics.items()}
        for key,value in diagnostics.items():self.diag_sums[key]=self.diag_sums.get(key,0.)+value
        self.diag_batches+=1
        return super().batch(loss,lr,**values,**diagnostics)
    def epoch(self,epoch,train,validation,lr,**extra):
        diagnostics={k:v/max(self.diag_batches,1) for k,v in self.diag_sums.items()}
        self.diag_history=[x for x in self.diag_history if x['epoch']!=epoch]+[{'epoch':epoch,'batches':self.diag_batches,**diagnostics}]
        atomic(self.output/'supervision_epoch_diagnostics.json',self.diag_history)
        self.diag_sums={};self.diag_batches=0
        return super().epoch(epoch,{**train,**diagnostics},validation,lr,**extra)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--stage',choices=['smoke','training'],default='training');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    native.make_model=make_model;native.loss_function=loss_function;native.components=components
    native.ICUContexts=TaskContexts;native.collate=partial(task_collate,cfg=cfg);native.SwanLabTracker=SupervisionTracker
    original_atomic=native.atomic
    def metadata(path,value):
        if Path(path).name=='hardware.json':value={**value,'supervision_ablation':cfg['supervision_ablation'],
            'training_past_generation':cfg['supervision_ablation']=='B','training_jacobian_sampling':'original final-cutoff future states only',
            'masked_retrospection_weight':cfg.get('masked_retrospection_weight'),'additional_cutoffs':cfg.get('additional_cutoffs')}
        return original_atomic(path,value)
    native.atomic=metadata;native.main()


if __name__=='__main__':main()
