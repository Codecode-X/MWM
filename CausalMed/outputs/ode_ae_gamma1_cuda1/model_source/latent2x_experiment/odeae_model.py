"""Native 6D probabilistic ODE-AE with a backward encoder and controlled generator."""
import torch
from torch import nn
from torch.nn import functional as F
from paper_odeae.models import mlp
from paper_odeae.solver import integrate


class ControlledODEAE(nn.Module):
    kind='ode_ae'

    def __init__(self,cfg):
        super().__init__();self.cfg=cfg;self.latent=6
        self.init=mlp(1,12,16);self.update=mlp(12+10,12,16)
        self.backward_ode=mlp(12,12,16);self.forward_ode=mlp(6,6,16)
        self.decoder=mlp(6,3,16);self.generator_action=mlp(6+2,6,16)
        for field in [self.backward_ode,self.forward_ode]:
            nn.init.normal_(field[-1].weight,std=.01);nn.init.zeros_(field[-1].bias)

    def solve(self,state,delta):
        return integrate(self.forward_ode,state,delta,self.cfg['rtol'],self.cfg['atol'])

    def encoder_event(self,state,values,mask,action,observation):
        onehot=F.one_hot(action.clamp(0,1).long(),2).to(state.dtype)
        flags=state.new_tensor([1.,0.] if observation else [0.,1.]).expand(len(state),-1)
        return self.update(torch.cat([state,values*mask,mask,onehot,flags],-1))

    def encoder_change(self,state,old,new,active):
        changed=active&(old!=new)
        if not bool(changed.any()):return state
        zeros=state.new_zeros(len(state),3)
        shift=self.encoder_event(state,zeros,zeros,new,False)-self.encoder_event(state,zeros,zeros,old,False)
        return state+torch.where(changed[:,None],shift,torch.zeros_like(shift))

    def change_action(self,state,old,new,active=None,disabled=False):
        if disabled:return state
        changed=old!=new
        if active is not None:changed=changed&active
        if not bool(changed.any()):return state
        onehot_new=F.one_hot(new.clamp(0,1).long(),2).to(state.dtype)
        onehot_old=F.one_hot(old.clamp(0,1).long(),2).to(state.dtype)
        shift=self.generator_action(torch.cat([state,onehot_new],-1))-self.generator_action(torch.cat([state,onehot_old],-1))
        return state+torch.where(changed[:,None],shift,torch.zeros_like(shift))

    def posterior(self,batch):
        if batch['values'].device.type!='cuda':raise RuntimeError('CUDA model execution required')
        cut=batch['cutoff'];times=batch['times'];b=len(cut)
        before=torch.cat([torch.zeros_like(batch['actions'][:,:1]),batch['actions'][:,:-1]],1)
        previous=times.gather(1,(cut-1)[:,None]).squeeze(1)
        current=before.gather(1,(cut-1)[:,None]).squeeze(1);state=self.init(batch['c'][:,None])
        for i in reversed(range(batch['max_context'])):
            active=i<cut;time=times[:,i];delta=torch.where(active,time-previous,torch.zeros_like(time))
            state=integrate(self.backward_ode,state,delta,self.cfg['rtol'],self.cfg['atol'])
            reversing=active&(i<cut-1);old=before[:,i]
            state=self.encoder_change(state,current,old,reversing);current=torch.where(reversing,old,current)
            observed=self.encoder_event(state,batch['values'][:,i],batch['context_mask'][:,i],current,True)
            state=torch.where(active[:,None],observed,state);previous=torch.where(active,time,previous)
        mu,logvar=state.chunk(2,-1);logvar=logvar.clamp(-12.,8.);std=(.5*logvar).exp()
        kl=.5*(mu.square()+logvar.exp()-1-logvar).sum(-1).mean()
        return mu,std,kl

    def encode(self,batch,collect_prior=False):
        # The original posterior is at the first visit. Replaying known historical
        # treatments yields the common pre-decision cutoff state for CF branching.
        mu,_,_=self.posterior(batch);state=mu;current=torch.zeros(len(mu),device=mu.device,dtype=torch.long)
        previous=batch['times'][:,0]
        for i in range(1,batch['max_context']):
            active=i<batch['cutoff'];new=batch['actions'][:,i-1]
            state=self.change_action(state,current,new,active);current=torch.where(active,new,current)
            time=batch['times'][:,i];state=self.solve(state,torch.where(active,time-previous,torch.zeros_like(time)))
            previous=torch.where(active,time,previous)
        return state,current,None

    def forward(self,batch):
        mu,std,kl=self.posterior(batch)
        state=mu+std*torch.randn_like(mu) if self.training else mu
        current=torch.zeros(len(mu),device=mu.device,dtype=torch.long);previous=batch['times'][:,0]
        predictions=[];states=[];anchor=state.clone();held=current.clone()
        for i in range(batch['times'].shape[1]):
            active=batch['valid'][:,i]
            if i:
                new=batch['actions'][:,i-1];state=self.change_action(state,current,new,active);current=torch.where(active,new,current)
                time=batch['times'][:,i];state=self.solve(state,torch.where(active,time-previous,torch.zeros_like(time)))
                previous=torch.where(active,time,previous)
            predictions.append(self.decoder(state));states.append(state)
            at_cutoff=i==batch['cutoff']-1;anchor=torch.where(at_cutoff[:,None],state,anchor);held=torch.where(at_cutoff,current,held)
        return {'prediction':torch.stack(predictions,1),'states':torch.stack(states,1),'prior_prediction':None,
                'anchor':anchor,'held_action':held,'posterior_mu':mu,'posterior_std':std,'kl':kl}

    def ode_regularization(self):
        return sum(p.square().sum() for field in [self.backward_ode,self.forward_ode] for p in field.parameters())
