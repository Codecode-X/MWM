"""Single-GPU MIMIC task adapters; elapsed hours and biological age are separate."""
import torch
from paper_odeae.models import encode_values,input_width,paper_model_from_config
from paper_odeae.baselines import BaselineModel,expanded_mask,feedback
from src.model import GRUUpdate


class ICUBaseline(BaselineModel):
    """Original repository networks, with an hourly clock and cutoff-anchored queries."""
    def __init__(self,schema,kind,latent=32,hidden=64,width=64,rtol=1e-4,atol=1e-6):
        super().__init__(schema,kind,latent,hidden,width,rtol,atol)
        # Age is available to every model. Add one input covariate while retaining
        # the original 44-field decoder and target definition.
        if kind=='latent_ode':self.core.encoder.cell=GRUUpdate(input_width(schema)+1,self.core.encoder.hidden)
        else:self.core.cell=GRUUpdate(input_width(schema)+1,self.core.hidden)

    def forward(self,batch,sample=False):
        values,ages=batch['values'],batch['ages']
        if values.device.type!='cuda':raise RuntimeError('CUDA required')
        times=ages-ages[:,:1]
        context=expanded_mask(batch['context_mask'],self.schema)
        inputs=encode_values(values,batch['context_mask'],self.schema)
        cutoff_hour=ages.gather(1,(batch['cutoff']-1)[:,None])
        age_covariate=(batch['initialization_age'][:,None]+(ages-cutoff_hour)/8766.)/100.
        b,n,_=values.shape
        if self.kind=='ode_rnn':
            state=values.new_zeros(b,self.core.hidden);predictions=[];anchor_state=state.clone()
            for i in range(n):
                active=batch['valid'][:,i]
                if i:state=self.solve(self.core.ode,state,torch.where(active,times[:,i]-times[:,i-1],torch.zeros_like(times[:,i])))
                prediction=self.core.decoder(state);predictions.append(prediction)
                available=context[:,i].bool()
                if self.training:available=available&(torch.rand(b,1,device=values.device)<.5)
                filled=torch.where(available,inputs[:,i],feedback(prediction,self.schema))
                future=i>=batch['cutoff']
                jump_mask=torch.where(future[:,None],torch.ones_like(context[:,i]),context[:,i])*active[:,None]
                filled=torch.cat([filled,age_covariate[:,i,None]],-1)
                jump_mask=torch.cat([jump_mask,active[:,None].to(jump_mask.dtype)],-1)
                state=self.core.cell(state,filled,jump_mask)
                anchor_state=torch.where((i==batch['cutoff']-1)[:,None],state,anchor_state)
            return {'prediction':torch.stack(predictions,1),'mu':anchor_state,'kl':state.new_zeros(())}
        state=values.new_zeros(b,self.core.encoder.hidden)
        for i in range(batch['max_context']):
            active=i<batch['cutoff']
            if i:state=self.solve(self.core.encoder.ode,state,(times[:,i]-times[:,i-1])*active)
            x=torch.cat([inputs[:,i],age_covariate[:,i,None]],-1)
            mask=torch.cat([context[:,i],active[:,None].to(context.dtype)],-1)
            proposed=self.core.encoder.cell(state,x,mask)
            state=torch.where(active[:,None],proposed,state)
        mu,raw=self.core.posterior(state).chunk(2,-1);std=torch.nn.functional.softplus(raw)+1e-5
        anchor=mu+std*torch.randn_like(mu) if sample else mu
        cutoff_time=times.gather(1,(batch['cutoff']-1)[:,None]).squeeze(1)
        prediction=[anchor.new_zeros(b,input_width(self.schema)) for _ in range(n)]
        z=anchor;previous=cutoff_time
        for i in reversed(range(batch['max_context'])):
            active=i<batch['cutoff'];delta=torch.where(active,times[:,i]-previous,torch.zeros_like(previous))
            z=self.solve(self.core.ode,z,delta);prediction[i]=torch.where(active[:,None],self.core.decoder(z),prediction[i]);previous=torch.where(active,times[:,i],previous)
        z=anchor;previous=cutoff_time
        for i in range(n):
            active=(i>=batch['cutoff'])&batch['valid'][:,i]
            if not bool(active.any()):continue
            z=self.solve(self.core.ode,z,torch.where(active,times[:,i]-previous,torch.zeros_like(previous)))
            prediction[i]=torch.where(active[:,None],self.core.decoder(z),prediction[i]);previous=torch.where(active,times[:,i],previous)
        kl=.5*(mu.square()+std.square()-1-2*std.log()).sum(-1).mean()
        return {'prediction':torch.stack(prediction,1),'mu':mu,'std':std,'kl':kl}


def make_model(schema,cfg,normalization=None):
    if cfg['model']=='healthflux':
        from .healthflux_icu import HealthFluxICU
        return HealthFluxICU(schema,cfg,normalization)
    if cfg['model']=='ode_ae':return paper_model_from_config(schema,cfg)
    return ICUBaseline(schema,cfg['model'],cfg['latent'],cfg['encoder_hidden'],cfg['ode_width'],cfg['rtol'],cfg['atol'])
