"""Equations (2)-(6) of HealthFlux v3, with the common ADNI mixed-field readout.

No author implementation or pretrained UKB weights were found in the audited
public sources. This is a Methods-based reimplementation, not author code.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from paper_odeae.models import encode_values,input_width


def network(inputs,outputs,width,dropout=0.,hidden_layers=2):
    modules=[]
    for _ in range(hidden_layers):
        modules.extend([nn.Linear(inputs,width),nn.SiLU()])
        if dropout:modules.append(nn.Dropout(dropout))
        inputs=width
    return nn.Sequential(*modules,nn.Linear(inputs,outputs))


def field_groups(schema):
    groups={'demographic':[],'cognition':[],'pet':[],'mri':[],'genetic':[],'csf':[]}
    for j,f in enumerate(schema):
        name=f['name']
        if name in ['sex','age','diagnosis']:continue
        key=('demographic' if name in ['marital_status','education_years','handedness'] else
             'pet' if name in ['amyloid_cortical_suvr','tau_cortical_suvr'] else
             'mri' if 'volume' in name else 'genetic' if name=='apoe4_count' else
             'csf' if name.startswith('csf_') else 'cognition')
        groups[key].append(j)
    return {key:indices for key,indices in groups.items() if indices}


class HealthFlux(nn.Module):
    def __init__(self,schema,cfg,normalization=None):
        super().__init__();self.schema=schema;self.cfg=cfg
        self.h_dim=cfg.get('deterministic_dim',256);self.z_dim=cfg.get('stochastic_dim',128)
        self.context_dim=cfg.get('context_dim',128);self.age_dim=cfg.get('age_dim',64)
        self.observation_dim=cfg.get('observation_dim',256);width=cfg.get('internal_width',384)
        dropout=cfg.get('dropout',.1)
        if 'modality_groups' in cfg:
            indices={f['name']:i for i,f in enumerate(schema)}
            self.groups={name:[indices[field] for field in fields] for name,fields in cfg['modality_groups'].items()}
            flat=[j for ids in self.groups.values() for j in ids]
            if len(flat)!=len(set(flat)) or not flat:raise ValueError('Modality groups must be nonempty and disjoint')
        else:self.groups=field_groups(schema)
        self.indices={f['name']:i for i,f in enumerate(schema)}
        self.register_buffer('input_mean',torch.tensor(normalization['mean'] if normalization else [0.]*len(schema)))
        self.register_buffer('input_std',torch.tensor(normalization['std'] if normalization else [1.]*len(schema)))
        self.context_encoder=network(6,self.context_dim,width,dropout)
        self.age_encoder=network(15,self.age_dim,width)
        self.initial=network(self.context_dim+self.age_dim,self.h_dim,width,dropout)
        self.dynamics=network(self.h_dim+self.context_dim+self.age_dim,self.h_dim,width)
        self.event_embedding=nn.Embedding(3,self.observation_dim)
        self.encoders=nn.ModuleDict()
        for name,ids in self.groups.items():
            expanded=sum(schema[j].get('classes',1) for j in ids)
            self.encoders[name]=network(expanded+len(ids),self.observation_dim,width,dropout)
        self.modality_embedding=nn.Parameter(torch.empty(len(self.groups),self.observation_dim));nn.init.normal_(self.modality_embedding,std=.02)
        self.attention=nn.Linear(self.observation_dim,1)
        prior_inputs=self.h_dim+self.context_dim+self.age_dim
        self.prior=network(prior_inputs,2*self.z_dim,width,dropout)
        self.posterior=network(prior_inputs+self.observation_dim,2*self.z_dim,width,dropout)
        self.update=nn.GRUCell(self.observation_dim+self.z_dim+self.context_dim+self.age_dim,self.h_dim)
        self.update_norm=nn.LayerNorm(self.h_dim)
        self.state_norm=nn.LayerNorm(self.h_dim+self.z_dim)
        d1,d2=cfg.get('decoder_widths',[1024,512])
        self.decoder=nn.Sequential(nn.Linear(self.h_dim+self.z_dim,d1),nn.GELU(),nn.LayerNorm(d1),nn.Dropout(.08),
                                   nn.Linear(d1,d2),nn.GELU(),nn.LayerNorm(d2),nn.Dropout(.08),nn.Linear(d2,input_width(schema)))

    def age_features(self,age):
        normalized=age/100.
        frequency=age.new_tensor([2.**k for k in range(6)])*math.pi
        angle=normalized[...,None]*frequency
        raw=torch.cat([normalized[...,None],normalized.square()[...,None],torch.log1p(age.clamp_min(0))[...,None]/math.log(101),
                       angle.sin(),angle.cos()],-1)
        return self.age_encoder(raw)

    def context(self,batch):
        # Context is restricted to the FIRST observed record, never filled from future records.
        mask=batch['context_mask'][:,0];value=batch['values'][:,0]
        sx=self.indices['sex'];ap=self.indices.get('apoe4_count')
        known=mask[:,sx:sx+1]
        sex=F.one_hot(value[:,sx].long(),2).to(value.dtype)*known
        genetic=value[:,ap:ap+1]*mask[:,ap:ap+1] if ap is not None else known.new_zeros(known.shape)
        genetic_mask=mask[:,ap:ap+1] if ap is not None else known.new_zeros(known.shape)
        return self.context_encoder(torch.cat([sex,known,1-known,genetic,genetic_mask],-1))

    def vector_field(self,h,age,context):
        return self.dynamics(torch.cat([h,self.age_features(age),context],-1)).tanh()

    def evolve(self,h,start_age,delta_years,context):
        """Original RK4 clock d h / d(age/10); <=4 steps, max nominal step 10y."""
        if h.device.type!='cuda':raise RuntimeError('HealthFlux ODE requires CUDA')
        if not bool((delta_years!=0).any()):return h
        steps=torch.ceil(delta_years.abs()/10.).clamp(1,4).long()
        # Batch padding is mathematically inactive, including intervals exactly zero.
        step=delta_years/steps
        for j in range(int(steps.max())):
            active=(j<steps)&(delta_years!=0)
            dt=torch.where(active,step,torch.zeros_like(step));clock=start_age+step*j
            k1=self.vector_field(h,clock,context)
            k2=self.vector_field(h+(dt/20)[:,None]*k1,clock+dt/2,context)
            k3=self.vector_field(h+(dt/20)[:,None]*k2,clock+dt/2,context)
            k4=self.vector_field(h+(dt/10)[:,None]*k3,clock+dt,context)
            h=h+(dt/60)[:,None]*(k1+2*k2+2*k3+k4)
        return h

    def gaussian(self,network_,h,age,context,observation=None):
        parts=[h,self.age_features(age),context]
        if observation is not None:parts.append(observation)
        mu,raw=network_(torch.cat(parts,-1)).chunk(2,-1)
        return mu,F.softplus(raw)+1e-4

    def jump(self,h,age,context,observation,active,sample):
        pm,ps=self.gaussian(self.prior,h,age,context)
        qm,qs=self.gaussian(self.posterior,h,age,context,observation)
        z=qm+qs*torch.randn_like(qm) if sample else qm
        candidate=self.update_norm(self.update(torch.cat([observation,z,context,self.age_features(age)],-1),h))
        next_h=torch.where(active[:,None],candidate,h)
        mean=torch.where(active[:,None],qm,pm)
        kl=(torch.log(ps/qs)+(qs.square()+(qm-pm).square())/(2*ps.square())-.5).sum(-1)
        return next_h,mean,kl*active,active.to(h.dtype)

    def pooled_observation(self,values,mask):
        encoded=[];available=[]
        for number,(name,ids) in enumerate(self.groups.items()):
            subset=[self.schema[j] for j in ids]
            x=encode_values(values[:,ids],mask[:,ids],subset)
            emb=self.encoders[name](torch.cat([x,mask[:,ids]],-1))+self.modality_embedding[number]
            encoded.append(emb);available.append(mask[:,ids].bool().any(-1))
        embeddings=torch.stack(encoded,1);valid=torch.stack(available,1)
        weights=self.attention(embeddings).squeeze(-1).masked_fill(~valid,-1e9).softmax(-1)*valid
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-9)
        return (embeddings*weights[...,None]).sum(1),valid.any(-1)

    def encode(self,batch,sample=False):
        values=batch['values'];mask=batch['context_mask'];ages=batch['ages'];cutoff=batch['cutoff']
        normalized=values.clone()
        for j,field in enumerate(self.schema):
            if field['kind']=='numerical':normalized[...,j]=((values[...,j]-self.input_mean[j])/self.input_std[j]).clamp(-10,10)
        context=self.context(batch);h=self.initial(torch.cat([context,self.age_features(ages[:,0])],-1))
        previous=ages[:,0];z=self.gaussian(self.prior,h,previous,context)[0]
        kl_sum=h.new_zeros(());node_count=h.new_zeros(())
        for i in range(batch['max_context']):
            alive=i<cutoff;age=ages[:,i];delta=torch.where(alive,age-previous,torch.zeros_like(previous))
            h=self.evolve(h,previous,delta,context)
            # Paper order: diagnosis event first; simultaneous other modalities pooled second.
            dx=self.indices['diagnosis'];event=alive&mask[:,i,dx].bool()
            h,event_z,kl,count=self.jump(h,age,context,self.event_embedding(values[:,i,dx].long()),event,sample)
            z=torch.where(event[:,None],event_z,z);kl_sum=kl_sum+kl.sum();node_count=node_count+count.sum()
            observation,present=self.pooled_observation(normalized[:,i],mask[:,i])
            active=alive&present
            h,obs_z,kl,count=self.jump(h,age,context,observation,active,sample)
            z=torch.where(active[:,None],obs_z,z);kl_sum=kl_sum+kl.sum();node_count=node_count+count.sum()
            no_observation=alive&~event&~active
            prior_mean=self.gaussian(self.prior,h,age,context)[0]
            z=torch.where(no_observation[:,None],prior_mean,z)
            previous=torch.where(alive,age,previous)
        return h,z,context,kl_sum/node_count.clamp_min(1)

    def read(self,h,age,context,observed_z=None):
        z=self.gaussian(self.prior,h,age,context)[0] if observed_z is None else observed_z
        return self.decoder(self.state_norm(torch.cat([h,z],-1)))

    def forward(self,batch,sample=False):
        if batch['values'].device.type!='cuda' or next(self.parameters()).device!=batch['values'].device:
            raise RuntimeError('HealthFlux model and observations must execute on CUDA')
        anchor,z,context,kl=self.encode(batch,sample)
        ages=batch['ages'];cutoff=batch['cutoff'];anchor_age=ages.gather(1,(cutoff-1)[:,None]).squeeze(1)
        b,n=ages.shape;prediction=[anchor.new_zeros(b,input_width(self.schema)) for _ in range(n)]
        h=anchor;previous=anchor_age
        for i in reversed(range(batch['max_context'])):
            active=i<cutoff;delta=torch.where(active,ages[:,i]-previous,torch.zeros_like(previous))
            h=self.evolve(h,previous,delta,context)
            query_z=self.gaussian(self.prior,h,ages[:,i],context)[0]
            at_anchor=i==cutoff-1;query_z=torch.where(at_anchor[:,None],z,query_z)
            value=self.read(h,ages[:,i],context,query_z)
            prediction[i]=torch.where(active[:,None],value,prediction[i]);previous=torch.where(active,ages[:,i],previous)
        h=anchor;previous=anchor_age
        for i in range(n):
            active=(i>=cutoff)&batch['valid'][:,i]
            if not bool(active.any()):continue
            delta=torch.where(active,ages[:,i]-previous,torch.zeros_like(previous))
            h=self.evolve(h,previous,delta,context)
            value=self.read(h,ages[:,i],context)
            prediction[i]=torch.where(active[:,None],value,prediction[i]);previous=torch.where(active,ages[:,i],previous)
        return {'prediction':torch.stack(prediction,1),'mu':anchor,'stochastic_mean':z,'kl':kl}

    def ode_regularization(self):
        return sum(parameter.square().sum() for parameter in self.dynamics.parameters())


def input_normalization(payload):
    schema=payload['schema'];mean=[];std=[]
    for j,field in enumerate(schema):
        if field['kind']=='categorical':mean.append(0.);std.append(1.);continue
        observed=[s['values'][:,j][s['mask'][:,j].astype(bool)] for s in payload['sequences'] if s['split']=='train']
        # Data preparation only; cached targets remain exactly the common min-max targets.
        import numpy as np
        values=np.concatenate(observed).astype(np.float64)
        mean.append(float(values.mean()));std.append(max(float(values.std()),1e-6))
    return {'mean':mean,'std':std,'fit_split':'train only; internal numeric input z-score, clipped [-10,10]'}
