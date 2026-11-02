"""MWM mean model and two equal-parameter supervision ablations."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from paper_odeae.models import PaperModel,encode_values


class MWM(PaperModel):
    def __init__(self,schema,cfg):
        self.variant=cfg.get('variant','main');uncertain=self.variant=='B'
        super().__init__(schema,kind='deterministic',latent=28,mlp_width=159,
                         ode_width=187 if uncertain else 188,update_width=160 if uncertain else 159,
                         decoder_width=158 if uncertain else 159,init_width=159,
                         rtol=cfg['rtol'],atol=cfg['atol'],encoder_direction='forward',encoder_initialization='age',
                         shared_vector_field=True,collect_prior_predictions=False,collect_jacobian_states=True)
        self.cfg=cfg;self.sigma_min=cfg.get('sigma_min',.01)
        if uncertain:
            self.sigma_head=nn.Linear(28,1)
            nn.init.zeros_(self.sigma_head.weight)
            nn.init.constant_(self.sigma_head.bias,math.log(math.expm1(1-self.sigma_min)))
        self.generator_anchor='history cutoff';self.name='MWM'
        if sum(p.numel() for p in self.parameters())!=cfg['target_parameters']:raise AssertionError('MWM parameter budget changed')

    def observation_sigma(self,state):
        return F.softplus(self.sigma_head(state))+self.sigma_min if self.variant=='B' else None

    def supervision_weight(self,seen):
        if self.variant!='A':return torch.ones_like(seen,dtype=torch.float32)
        return torch.sigmoid(self.cfg.get('count_a',1.)*(seen.float()-self.cfg.get('count_b',3.)))

    def forward(self,batch,sample=False):
        if batch['values'].device.type!='cuda' or next(self.parameters()).device!=batch['values'].device:raise RuntimeError('MWM computations require CUDA')
        values,times,cut=batch['values'],batch['ages'],batch['cutoff'];b,n,_=values.shape
        inputs=encode_values(values,batch['context_mask'],self.schema)
        state=self.init(batch['first_observation_age'][:,None]/100.);previous=times[:,0]
        seen=torch.zeros(b,device=state.device,dtype=torch.long)
        predictions=[state.new_zeros(b,self.decoder[-1].out_features) for _ in range(n)]
        positions=[torch.zeros_like(state) for _ in range(n)];counts=[seen.clone() for _ in range(n)]
        sigmas=[state.new_ones(b,1) for _ in range(n)] if self.variant=='B' else None
        for i in range(batch['max_context']):
            active=i<cut;time=times[:,i]
            evolved=self.solve(self.forward_ode,state,torch.where(active,time-previous,torch.zeros_like(time)))
            observed=active&batch['context_mask'][:,i].bool().any(-1)
            proposal=self.update(torch.cat([evolved,inputs[:,i]],-1))
            updated=torch.where(observed[:,None],proposal,evolved)
            next_seen=seen+observed.long()
            # First record trains its readout after ingestion; later historical
            # records are predicted strictly before their own UPDATE.
            predictive=updated if i==0 else evolved
            effective_seen=next_seen if i==0 else seen
            predictions[i]=torch.where(active[:,None],self.decoder(predictive),predictions[i])
            positions[i]=torch.where(active[:,None],predictive,positions[i]);counts[i]=torch.where(active,effective_seen,counts[i])
            if sigmas is not None:sigmas[i]=torch.where(active[:,None],self.observation_sigma(predictive),sigmas[i])
            state=torch.where(active[:,None],updated,state);seen=torch.where(active,next_seen,seen);previous=torch.where(active,time,previous)
        anchor=state;final_seen=seen;previous=times.gather(1,(cut-1)[:,None]).squeeze(1)
        for i in range(n):
            active=(i>=cut)&batch['valid'][:,i]
            if not bool(active.any()):continue
            time=times[:,i];state=self.solve(self.forward_ode,state,torch.where(active,time-previous,torch.zeros_like(time)))
            predictions[i]=torch.where(active[:,None],self.decoder(state),predictions[i])
            positions[i]=torch.where(active[:,None],state,positions[i]);counts[i]=torch.where(active,final_seen,counts[i])
            if sigmas is not None:sigmas[i]=torch.where(active[:,None],self.observation_sigma(state),sigmas[i])
            previous=torch.where(active,time,previous)
        count=torch.stack(counts,1)
        result={'prediction':torch.stack(predictions,1),'mu':anchor,'kl':anchor.new_zeros(()),
                'jacobian_states':torch.stack(positions,1),'seen_observations':count,'supervision_weights':self.supervision_weight(count),
                'final_seen_observations':final_seen}
        if sigmas is not None:result['sigma']=torch.stack(sigmas,1)
        return result

    @torch.no_grad()
    def retrospective_history(self,batch,anchor):
        """The previous history-reconstruction test, separate from training."""
        times,cut=batch['ages'],batch['cutoff'];state=anchor
        previous=times.gather(1,(cut-1)[:,None]).squeeze(1);values=[anchor.new_zeros(len(anchor),self.decoder[-1].out_features) for _ in range(batch['max_context'])]
        for i in reversed(range(batch['max_context'])):
            active=i<cut;time=times[:,i]
            state=self.solve(self.forward_ode,state,torch.where(active,time-previous,torch.zeros_like(time)))
            values[i]=torch.where(active[:,None],self.decoder(state),values[i]);previous=torch.where(active,time,previous)
        return torch.stack(values,1)
