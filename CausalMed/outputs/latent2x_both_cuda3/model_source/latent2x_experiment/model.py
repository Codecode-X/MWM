"""28D shared ODE with observation updates and residual action-position changes."""
import torch
from torch import nn
from torch.nn import functional as F
from paper_odeae.models import mlp
from paper_odeae.solver import integrate
from paper_odeae.jacobian import squared_jacobian_norm


class ControlledBoth(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg;self.latent=28
        self.init=mlp(1,28,159);self.update=mlp(28+10,28,159)
        self.forward_ode=mlp(28,28,188);self.decoder=mlp(28,3,159)
        nn.init.normal_(self.forward_ode[-1].weight,std=.01);nn.init.zeros_(self.forward_ode[-1].bias)

    def solve(self,state,delta):
        return integrate(self.forward_ode,state,delta,self.cfg['rtol'],self.cfg['atol'])

    def event(self,state,values,mask,action,observation):
        a=F.one_hot(action.clamp(0,1).long(),2).to(state.dtype)
        flags=state.new_tensor([1.,0.] if observation else [0.,1.]).expand(len(state),-1)
        return self.update(torch.cat([state,values*mask,mask,a,flags],-1))

    def change_action(self,state,old,new,active=None,disabled=False):
        if disabled:return state
        changed=old!=new
        if active is not None:changed=changed&active
        if not bool(changed.any()):return state
        zeros=state.new_zeros(len(state),3)
        # Shared UPDATE evaluates the two treatment conditions at this position.
        shift=self.event(state,zeros,zeros,new,False)-self.event(state,zeros,zeros,old,False)
        return state+torch.where(changed[:,None],shift,torch.zeros_like(shift))

    def encode(self,batch,collect_prior=False):
        if batch['values'].device.type!='cuda':raise RuntimeError('Model execution requires CUDA')
        state=self.init(batch['c'][:,None]);current=torch.zeros(len(state),dtype=torch.long,device=state.device)
        previous=batch['times'][:,0];priors=[]
        for i in range(batch['max_context']):
            active=i<batch['cutoff'];time=batch['times'][:,i]
            state=self.solve(state,torch.where(active,time-previous,torch.zeros_like(time)))
            if collect_prior:priors.append(self.decoder(state))
            proposed=self.event(state,batch['values'][:,i],batch['context_mask'][:,i],current,True)
            state=torch.where(active[:,None],proposed,state)
            # The final history observation is read BEFORE its treatment decision.
            continuing=i+1<batch['cutoff'];new=batch['actions'][:,i]
            state=self.change_action(state,current,new,continuing)
            current=torch.where(continuing,new,current);previous=torch.where(active,time,previous)
        return state,current,torch.stack(priors,1) if collect_prior else None

    def forward(self,batch):
        anchor,held,priors=self.encode(batch,True)
        times,cut=batch['times'],batch['cutoff'];b,n=times.shape
        predictions=[anchor.new_zeros(b,3) for _ in range(n)]
        states=[torch.zeros_like(anchor) for _ in range(n)]
        anchor_time=times.gather(1,(cut-1)[:,None]).squeeze(1)
        previous_action=torch.cat([torch.zeros_like(batch['actions'][:,:1]),batch['actions'][:,:-1]],1)
        state=anchor;current=held;previous=anchor_time
        for i in reversed(range(batch['max_context'])):
            active=i<cut;time=times[:,i]
            state=self.solve(state,torch.where(active,time-previous,torch.zeros_like(time)))
            reverse=active&(i<cut-1);old=previous_action[:,i]
            state=self.change_action(state,current,old,reverse)
            current=torch.where(reverse,old,current)
            predictions[i]=torch.where(active[:,None],self.decoder(state),predictions[i]);states[i]=torch.where(active[:,None],state,states[i])
            previous=torch.where(active,time,previous)
        state=anchor;current=held;previous=anchor_time
        for i in range(n):
            active=(i>=cut)&batch['valid'][:,i]
            if not bool(active.any()):continue
            # Recorded factual action at the preceding visit precedes this target.
            new=batch['actions'][:,max(0,i-1)]
            state=self.change_action(state,current,new,active);current=torch.where(active,new,current)
            time=times[:,i];state=self.solve(state,torch.where(active,time-previous,torch.zeros_like(time)))
            predictions[i]=torch.where(active[:,None],self.decoder(state),predictions[i]);states[i]=torch.where(active[:,None],state,states[i])
            previous=torch.where(active,time,previous)
        return {'prediction':torch.stack(predictions,1),'states':torch.stack(states,1),'prior_prediction':priors,'anchor':anchor,'held_action':held}

    def ode_regularization(self):
        return sum(p.square().sum() for p in self.forward_ode.parameters())


def objective(model,out,batch):
    errors=(out['prediction']-batch['values']).square()*batch['mask']
    mse=(errors.sum((0,1))/batch['mask'].sum((0,1)).clamp_min(1)).sum()
    n=out['prior_prediction'].shape[1];i=torch.arange(n,device='cuda')[None]
    mask=batch['mask'][:,:n]*((i>0)&(i<batch['cutoff'][:,None]))[...,None]
    pe=(out['prior_prediction']-batch['values'][:,:n]).square()*mask
    prior=(pe.sum((0,1))/mask.sum((0,1)).clamp_min(1)).sum()
    visited=out['states'][batch['valid']].detach();jac=squared_jacobian_norm(model.forward_ode,visited).mean()
    l2=model.ode_regularization();loss=mse+.01*prior+.01*jac+.02*l2
    return loss,{'mse':mse,'prior_mse':prior,'weighted_prior':.01*prior,'jacobian':jac,'weighted_jacobian':.01*jac,'ode_l2':l2,'weighted_ode_l2':.02*l2}
