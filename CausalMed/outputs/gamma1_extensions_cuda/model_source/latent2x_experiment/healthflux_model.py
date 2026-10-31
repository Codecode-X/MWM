"""Native HealthFlux 256+128D with public time/static context and treatment events."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from healthflux.model import HealthFlux,network


class HealthFluxCF(nn.Module):
    kind='healthflux'

    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        schema=[{'name':name,'kind':'numerical'} for name in ['Y_obs','X1_obs','X2_obs']]
        native={**cfg,'modality_groups':{'outcome':['Y_obs'],'biomarker':['X1_obs'],'risk':['X2_obs']}}
        self.core=HealthFlux(schema,native)
        self.h_dim=self.core.h_dim;self.z_dim=self.core.z_dim;self.context_dim=self.core.context_dim;self.latent=self.h_dim
        self.core.context_encoder=network(1,self.context_dim,cfg.get('internal_width',384),cfg.get('dropout',.1))
        self.core.event_embedding=nn.Embedding(2,self.core.observation_dim)

    def unpack(self,state):
        h=state[:,:self.h_dim];z=state[:,self.h_dim:self.h_dim+self.z_dim]
        context=state[:,self.h_dim+self.z_dim:-1];clock=state[:,-1]
        return h,z,context,clock

    def pack(self,h,z,context,clock):return torch.cat([h,z,context,clock[:,None]],-1)

    def time_features(self,time):
        # CausalMed has elapsed years, not chronological patient age.
        t=time/8.;frequency=time.new_tensor([2.**k for k in range(6)])*math.pi;angle=t[:,None]*frequency
        raw=torch.cat([t[:,None],t.square()[:,None],torch.log1p(time.clamp_min(0))[:,None]/math.log(9.),angle.sin(),angle.cos()],-1)
        return self.core.age_encoder(raw)

    def gaussian(self,network_,h,time,context,observation=None):
        inputs=[h,self.time_features(time),context]
        if observation is not None:inputs.append(observation)
        mu,raw=network_(torch.cat(inputs,-1)).chunk(2,-1)
        return mu,F.softplus(raw)+1e-4

    def field(self,h,time,context):
        return self.core.dynamics(torch.cat([h,self.time_features(time),context],-1)).tanh()

    def solve(self,state,delta):
        if state.device.type!='cuda':raise RuntimeError('HealthFlux model computations require CUDA')
        if not bool(delta.count_nonzero()):return state
        h,_,context,time=self.unpack(state)
        steps=torch.ceil(delta.abs()/self.cfg.get('rk4_step_years',.25)).clamp(1,4).long();step=delta/steps
        for i in range(int(steps.max())):
            dt=torch.where((i<steps)&(delta!=0),step,torch.zeros_like(step));t=time+step*i
            k1=self.field(h,t,context);k2=self.field(h+dt[:,None]*k1/2,t+dt/2,context)
            k3=self.field(h+dt[:,None]*k2/2,t+dt/2,context);k4=self.field(h+dt[:,None]*k3,t+dt,context)
            h=h+dt[:,None]*(k1+2*k2+2*k3+k4)/6
        clock=time+delta;z=self.gaussian(self.core.prior,h,clock,context)[0]
        result=self.pack(h,z,context,clock)
        return torch.where((delta!=0)[:,None],result,state)

    def decoder(self,state):
        h,z,_,_=self.unpack(state)
        return self.core.decoder(self.core.state_norm(torch.cat([h,z],-1)))

    def observe(self,state,values,mask,active):
        h,_,context,time=self.unpack(state)
        observation,present=self.core.pooled_observation(values.clamp(-10,10),mask);active=active&present
        pm,ps=self.gaussian(self.core.prior,h,time,context)
        qm,qs=self.gaussian(self.core.posterior,h,time,context,observation)
        sample=qm+qs*torch.randn_like(qm) if self.training else qm
        candidate=self.core.update_norm(self.core.update(torch.cat([observation,sample,context,self.time_features(time)],-1),h))
        updated=self.pack(candidate,qm,context,time);result=torch.where(active[:,None],updated,state)
        kl=(torch.log(ps/qs)+(qs.square()+(qm-pm).square())/(2*ps.square())-.5).sum(-1)
        return result,(kl*active).sum(),active.sum()

    def change_action(self,state,old,new,active=None,disabled=False):
        if disabled:return state
        changed=old!=new
        if active is not None:changed=changed&active
        if not bool(changed.any()):return state
        h,z,context,time=self.unpack(state);tail=torch.cat([z,context,self.time_features(time)],-1)
        def candidate(action):
            token=self.core.event_embedding(action.clamp(0,1).long())
            return self.core.update_norm(self.core.update(torch.cat([token,tail],-1),h))
        moved=h+candidate(new)-candidate(old)
        prior_z=self.gaussian(self.core.prior,moved,time,context)[0]
        result=self.pack(moved,prior_z,context,time)
        return torch.where(changed[:,None],result,state)

    def encode(self,batch,collect_prior=False):
        context=self.core.context_encoder(batch['c'][:,None]);time=batch['times'][:,0]
        h=self.core.initial(torch.cat([context,self.time_features(time)],-1));z=self.gaussian(self.core.prior,h,time,context)[0]
        state=self.pack(h,z,context,time);previous=time;current=torch.zeros(len(h),device=h.device,dtype=torch.long)
        kl_sum=h.new_zeros(());nodes=h.new_zeros(())
        for i in range(batch['max_context']):
            active=i<batch['cutoff'];time=batch['times'][:,i]
            state=self.solve(state,torch.where(active,time-previous,torch.zeros_like(time)))
            state,kl,count=self.observe(state,batch['values'][:,i],batch['context_mask'][:,i],active);kl_sum=kl_sum+kl;nodes=nodes+count
            continuing=i+1<batch['cutoff'];new=batch['actions'][:,i]
            state=self.change_action(state,current,new,continuing);current=torch.where(continuing,new,current);previous=torch.where(active,time,previous)
        return state,current,(kl_sum/nodes.clamp_min(1))

    def forward(self,batch):
        anchor,held,kl=self.encode(batch)
        times,cut=batch['times'],batch['cutoff'];b,n=times.shape
        predictions=[anchor.new_zeros(b,3) for _ in range(n)]
        anchor_time=times.gather(1,(cut-1)[:,None]).squeeze(1)
        before=torch.cat([torch.zeros_like(batch['actions'][:,:1]),batch['actions'][:,:-1]],1)
        state=anchor;current=held;previous=anchor_time
        for i in reversed(range(batch['max_context'])):
            active=i<cut;time=times[:,i];state=self.solve(state,torch.where(active,time-previous,torch.zeros_like(time)))
            reverse=active&(i<cut-1);new=before[:,i];state=self.change_action(state,current,new,reverse);current=torch.where(reverse,new,current)
            predictions[i]=torch.where(active[:,None],self.decoder(state),predictions[i]);previous=torch.where(active,time,previous)
        state=anchor;current=held;previous=anchor_time
        for i in range(n):
            active=(i>=cut)&batch['valid'][:,i]
            if not bool(active.any()):continue
            new=batch['actions'][:,max(0,i-1)];state=self.change_action(state,current,new,active);current=torch.where(active,new,current)
            time=times[:,i];state=self.solve(state,torch.where(active,time-previous,torch.zeros_like(time)))
            predictions[i]=torch.where(active[:,None],self.decoder(state),predictions[i]);previous=torch.where(active,time,previous)
        return {'prediction':torch.stack(predictions,1),'anchor':anchor,'held_action':held,'kl':kl,'prior_prediction':None}

    def ode_regularization(self):return sum(p.square().sum() for p in self.core.dynamics.parameters())
