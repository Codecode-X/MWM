"""HealthFlux 256+128D, with physiological groups, hourly ODE and age-in-years inputs."""
import torch
from healthflux.model import HealthFlux
from paper_odeae.models import input_width


class HealthFluxICU(HealthFlux):
    def __init__(self,schema,cfg,normalization=None):
        super().__init__(schema,cfg,normalization)
        # The selected ICU observations have no diagnosis event token.
        self.event_embedding.requires_grad_(False)

    def evolve_hours(self,h,start_hour,delta,context,admission_age):
        if h.device.type!='cuda':raise RuntimeError('CUDA required')
        if not bool(delta.count_nonzero()):return h
        steps=torch.ceil(delta.abs()).clamp(1,4).long();step=delta/steps
        for i in range(int(steps.max())):
            dt=torch.where((i<steps)&(delta!=0),step,torch.zeros_like(step));hour=start_hour+step*i
            age=admission_age+hour/8766.
            k1=self.vector_field(h,age,context)
            k2=self.vector_field(h+dt[:,None]*k1/2,age+dt/17532.,context)
            k3=self.vector_field(h+dt[:,None]*k2/2,age+dt/17532.,context)
            k4=self.vector_field(h+dt[:,None]*k3,age+dt/8766.,context)
            h=h+dt[:,None]*(k1+2*k2+2*k3+k4)/6
        return h

    def encode_icu(self,batch,sample=False):
        values,mask,hours=batch['values'],batch['context_mask'],batch['ages'];cutoff=batch['cutoff']
        anchor_hour=hours.gather(1,(cutoff-1)[:,None]).squeeze(1)
        admission_age=batch['initialization_age']-anchor_hour/8766.
        age=admission_age+hours[:,0]/8766.;context=self.context(batch)
        h=self.initial(torch.cat([context,self.age_features(age)],-1));previous=hours[:,0]
        z=self.gaussian(self.prior,h,age,context)[0];kl_sum=h.new_zeros(());count=h.new_zeros(())
        normalized=values.clone()
        for j,field in enumerate(self.schema):
            if field['kind']=='numerical':normalized[...,j]=((values[...,j]-self.input_mean[j])/self.input_std[j]).clamp(-10,10)
        for i in range(batch['max_context']):
            active=i<cutoff;hour=hours[:,i];age=admission_age+hour/8766.
            h=self.evolve_hours(h,previous,torch.where(active,hour-previous,torch.zeros_like(hour)),context,admission_age)
            observation,present=self.pooled_observation(normalized[:,i],mask[:,i]);observed=active&present
            h,updated_z,kl,nodes=self.jump(h,age,context,observation,observed,sample)
            z=torch.where(observed[:,None],updated_z,z);kl_sum=kl_sum+kl.sum();count=count+nodes.sum()
            prior=self.gaussian(self.prior,h,age,context)[0]
            z=torch.where((active&~observed)[:,None],prior,z);previous=torch.where(active,hour,previous)
        return h,z,context,admission_age,kl_sum,count

    def forward(self,batch,sample=False):
        if batch['values'].device.type!='cuda':raise RuntimeError('CUDA required')
        anchor,z,context,age0,kl_sum,count=self.encode_icu(batch,sample)
        hours,cutoff=batch['ages'],batch['cutoff'];b,n=hours.shape
        anchor_hour=hours.gather(1,(cutoff-1)[:,None]).squeeze(1)
        prediction=[anchor.new_zeros(b,input_width(self.schema)) for _ in range(n)]
        h=anchor;previous=anchor_hour
        for i in reversed(range(batch['max_context'])):
            active=i<cutoff;hour=hours[:,i];age=age0+hour/8766.
            h=self.evolve_hours(h,previous,torch.where(active,hour-previous,torch.zeros_like(hour)),context,age0)
            query_z=self.gaussian(self.prior,h,age,context)[0];query_z=torch.where((i==cutoff-1)[:,None],z,query_z)
            value=self.read(h,age,context,query_z);prediction[i]=torch.where(active[:,None],value,prediction[i]);previous=torch.where(active,hour,previous)
        h=anchor;previous=anchor_hour
        for i in range(n):
            active=(i>=cutoff)&batch['valid'][:,i]
            if not bool(active.any()):continue
            hour=hours[:,i];h=self.evolve_hours(h,previous,torch.where(active,hour-previous,torch.zeros_like(hour)),context,age0)
            value=self.read(h,age0+hour/8766.,context);prediction[i]=torch.where(active[:,None],value,prediction[i]);previous=torch.where(active,hour,previous)
        return {'prediction':torch.stack(prediction,1),'mu':anchor,'stochastic_mean':z,'kl':kl_sum/count.clamp_min(1),'kl_sum':kl_sum,'kl_count':count}
