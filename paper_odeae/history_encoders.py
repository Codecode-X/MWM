"""Equal-capacity history encoders; age INIT, shared field and decoder stay fixed."""
import math

import torch
from torch import nn

from .models import PaperModel,encode_values,input_width


class CausalTimeAttention(nn.Module):
    """An exactly matched encoder with a fixed continuous physical-time clock."""
    def __init__(self,features=49,latent=28,time_unit='year'):
        super().__init__()
        if latent!=28 or features not in [49,44]:raise ValueError('Matched encoder supports ADNI and MIMIC 28D schemas')
        if time_unit not in ['year','hour']:raise ValueError('Explicit physical time unit required')
        embedding,feedforward,budget={49:(59,156,16882),44:(42,169,16087)}[features]
        self.time_unit=time_unit
        self.embedding=nn.Sequential(nn.Linear(features,embedding),nn.Tanh(),nn.Linear(embedding,latent,bias=False))
        self.attention=nn.MultiheadAttention(latent,4,dropout=0.,batch_first=True)
        self.norm1=nn.LayerNorm(latent);self.norm2=nn.LayerNorm(latent)
        self.feedforward=nn.Sequential(nn.Linear(latent,feedforward),nn.Tanh(),nn.Linear(feedforward,latent))
        low,high=(.25,20.) if time_unit=='year' else (.5,48.)
        self.register_buffer('period_years' if time_unit=='year' else 'period_hours',torch.logspace(math.log10(low),math.log10(high),latent//2))
        if sum(p.numel() for p in self.parameters())!=budget:raise AssertionError('Attention encoder budget changed')

    def time_encoding(self,elapsed):
        periods=self.period_years if self.time_unit=='year' else self.period_hours
        angle=2*math.pi*elapsed[...,None]/periods
        return torch.cat([angle.sin(),angle.cos()],-1)

    def forward(self,observations,times,cutoff,initial):
        n=observations.shape[1];index=torch.arange(n,device=observations.device)
        padding=index[None]>=cutoff[:,None]
        tokens=self.embedding(observations)+self.time_encoding(times-times[:,:1])+initial[:,None]
        causal=index[None,:]>index[:,None]
        attended,_=self.attention(tokens,tokens,tokens,attn_mask=causal,key_padding_mask=padding,need_weights=False)
        h=self.norm1(tokens+attended);h=self.norm2(h+self.feedforward(h))
        return torch.where(padding[...,None],initial[:,None],h)


class HistoryEncoderAblation(PaperModel):
    def __init__(self,schema,cfg):
        # Allocate the original networks in exactly the original RNG order, so
        # all retained modules have identical initial tensors at the same seed.
        super().__init__(schema,kind='deterministic',latent=cfg['latent'],ode_width=cfg['ode_width'],
                         mlp_width=cfg['mlp_width'],rtol=cfg['rtol'],atol=cfg['atol'],
                         encoder_direction='forward',encoder_initialization='age',shared_vector_field=True,
                         collect_prior_predictions=True,collect_jacobian_states=True)
        self.variant=cfg['history_encoder_variant']
        if self.variant not in ['reverse_shared','attention_time']:raise ValueError(self.variant)
        self.encoder_direction='backward' if self.variant=='reverse_shared' else 'attention'
        self.generator_anchor='first history visit' if self.variant=='reverse_shared' else 'history cutoff'
        if self.variant=='attention_time':
            del self.update
            self.history_attention=CausalTimeAttention(input_width(schema),self.latent,cfg.get('attention_time_unit','year'))
        if sum(p.numel() for p in self.parameters())!=cfg['target_parameters']:raise AssertionError('Matched capacity changed')

    def initial_state(self,batch):
        age=batch.get('initialization_age',batch['ages'].gather(1,(batch['cutoff']-1)[:,None]).squeeze(1))
        return self.init(age[:,None]/100.)

    def forward_ode_encoding(self,batch,inputs):
        """Original causal prior supervision, also retained in the reverse group."""
        state=self.initial_state(batch);previous=batch['ages'][:,0];priors=[]
        for j in range(batch['max_context']):
            active=j<batch['cutoff'];time=batch['ages'][:,j]
            evolved=self.solve(self.forward_ode,state,torch.where(active,time-previous,torch.zeros_like(time)))
            priors.append(self.decoder(evolved));proposed=self.update(torch.cat([evolved,inputs[:,j]],-1))
            state=torch.where(active[:,None],proposed,state);previous=torch.where(active,time,previous)
        return state,torch.stack(priors,1)

    def reverse_encoding(self,batch,inputs):
        state=self.initial_state(batch);cut=batch['cutoff'];previous=batch['ages'].gather(1,(cut-1)[:,None]).squeeze(1)
        rows=torch.arange(len(cut),device=state.device)
        for j in range(batch['max_context']):
            index=(cut-1-j).clamp_min(0);active=j<cut;time=batch['ages'].gather(1,index[:,None]).squeeze(1)
            evolved=self.solve(self.forward_ode,state,torch.where(active,time-previous,torch.zeros_like(time)))
            proposed=self.update(torch.cat([evolved,inputs[rows,index]],-1))
            state=torch.where(active[:,None],proposed,state);previous=torch.where(active,time,previous)
        return state

    def attention_encoding(self,batch,inputs):
        """The history representation itself contains no ODE integration."""
        n=batch['max_context']
        return self.history_attention(inputs[:,:n],batch['ages'][:,:n],batch['cutoff'],self.initial_state(batch))

    def attention_priors(self,batch,prefix_states):
        # Predict observation j using only prefix 0..j-1. The shared ODE is used
        # for the retained weak-prior prediction, outside the attention encoder.
        priors=[self.decoder(self.initial_state(batch))]
        for j in range(1,batch['max_context']):
            active=j<batch['cutoff'];delta=batch['ages'][:,j]-batch['ages'][:,j-1]
            evolved=self.solve(self.forward_ode,prefix_states[:,j-1],torch.where(active,delta,torch.zeros_like(delta)))
            priors.append(self.decoder(evolved))
        return torch.stack(priors,1)

    def generate_from_first(self,state,batch):
        previous=batch['ages'][:,0];predictions=[];states=[]
        for j in range(batch['ages'].shape[1]):
            if j:
                active=batch['valid'][:,j];time=batch['ages'][:,j]
                state=self.solve(self.forward_ode,state,torch.where(active,time-previous,torch.zeros_like(time)))
                previous=torch.where(active,time,previous)
            predictions.append(self.decoder(state));states.append(state)
        return torch.stack(predictions,1),torch.stack(states,1)

    def forward(self,batch,sample=False):
        if batch['values'].device.type!='cuda' or next(self.parameters()).device!=batch['values'].device:
            raise RuntimeError('CUDA model execution required')
        inputs=encode_values(batch['values'],batch['context_mask'],self.schema)
        if self.variant=='reverse_shared':
            state=self.reverse_encoding(batch,inputs)
            # The auxiliary causal pass supplies only the unchanged weak-prior
            # loss, never the main reverse state or its generated trajectory.
            _,priors=self.forward_ode_encoding(batch,inputs)
            prediction,states=self.generate_from_first(state,batch)
        else:
            prefix=self.attention_encoding(batch,inputs)
            rows=torch.arange(len(prefix),device=prefix.device);state=prefix[rows,batch['cutoff']-1]
            priors=self.attention_priors(batch,prefix)
            prediction,states=self.generate_from_cutoff(state,batch,return_states=True)
        return {'prediction':prediction,'mu':state,'std':torch.zeros_like(state),'kl':state.new_zeros(()),
                'prior_prediction':priors,'jacobian_states':states}
