"""Paper Algorithm 1: age initialization, reverse updates, first-visit anchor."""
import torch
from torch import nn
from torch.nn import functional as F
from .solver import integrate


def mlp(inputs, outputs, width):
    return nn.Sequential(nn.Linear(inputs,width),nn.Tanh(),nn.Linear(width,outputs))


def input_width(schema):
    return sum(f.get('classes',1) for f in schema)


def encode_values(values,mask,schema):
    pieces=[]
    for j,f in enumerate(schema):
        if f['kind']=='categorical':
            x=F.one_hot(values[...,j].long(),f['classes']).to(values.dtype)
        else:
            x=values[...,j,None]
        pieces.append(x*mask[...,j,None])
    return torch.cat(pieces,-1)


class PaperModel(nn.Module):
    def __init__(self,schema,kind='ode_ae',latent=6,ode_width=16,mlp_width=16,
                 rtol=1e-4,atol=1e-6,encoder_direction='backward',encoder_initialization=None,
                 init_width=None,update_width=None,decoder_width=None,
                 encoder_ode_width=None,generator_ode_width=None,
                 shared_vector_field=False,shared_l2_multiplier=1.):
        super().__init__()
        if kind not in ['ode_ae','deterministic','lstm']:
            raise ValueError(kind)
        if encoder_direction not in ['backward','forward']:
            raise ValueError('unknown encoder direction')
        if encoder_direction=='forward' and kind!='deterministic':
            raise ValueError('forward-encoder ablations are restricted to deterministic ODE-AE')
        encoder_initialization=encoder_initialization or ('zero' if encoder_direction=='forward' else 'age')
        if encoder_initialization not in ['zero','age']:
            raise ValueError('unknown encoder initialization')
        if encoder_direction=='backward' and encoder_initialization!='age':
            raise ValueError('backward reference keeps its original age initialization')
        self.schema,self.kind,self.latent=schema,kind,latent
        self.encoder_direction=encoder_direction
        self.encoder_initialization=encoder_initialization
        self.shared_vector_field=shared_vector_field
        self.shared_l2_multiplier=float(shared_l2_multiplier)
        if shared_vector_field and (kind!='deterministic' or encoder_direction!='forward'):
            raise ValueError('shared field study requires forward deterministic model')
        if self.shared_l2_multiplier<=0:raise ValueError('regularization multiplier must be positive')
        self.rtol,self.atol=rtol,atol
        dim=latent if kind=='deterministic' else 2*latent
        width=input_width(schema)
        self.init=mlp(1,dim,init_width or mlp_width)
        self.decoder=mlp(latent,width,decoder_width or mlp_width)
        if kind=='lstm':
            self.encoder_cell=nn.LSTMCell(width+1,dim)
            self.generator_cell=nn.LSTMCell(1,latent)
        else:
            self.update=mlp(dim+width,dim,update_width or mlp_width)
            self.backward_ode=mlp(dim,dim,encoder_ode_width or ode_width)
            self.forward_ode=mlp(latent,latent,generator_ode_width or ode_width)
            # Small initial vector fields keep early adaptive solves well scaled.
            for network in [self.backward_ode,self.forward_ode]:
                nn.init.normal_(network[-1].weight,std=.01)
                nn.init.zeros_(network[-1].bias)
            if shared_vector_field:
                if (encoder_ode_width or ode_width)!=(generator_ode_width or ode_width):
                    raise ValueError('shared encoder/generator widths must agree')
                # A true module alias, not two independently optimized copies.
                self.backward_ode=self.forward_ode
        if self.encoder_initialization=='zero':
            # Construct shared networks with identical RNG consumption first,
            # then remove the age INIT network: zero initialization has no
            # trainable INIT parameters and must not inflate capacity counts.
            del self.init

    def solve(self,func,h,delta):
        return integrate(func,h,delta,self.rtol,self.atol)

    def forward(self,batch,sample=False):
        values,ages,context=batch['values'],batch['ages'],batch['context_mask']
        if values.device.type!='cuda' or next(self.parameters()).device!=values.device:
            raise RuntimeError('model and observations must execute on CUDA')
        cutoff=batch['cutoff']; b,n,_=values.shape
        inputs=encode_values(values,context,self.schema)
        last_age=ages.gather(1,(cutoff-1)[:,None]).squeeze(1)
        h=self.init(last_age[:,None]/100.) if self.encoder_initialization=='age' else values.new_zeros(b,self.latent)
        previous=last_age if self.encoder_direction=='backward' else ages[:,0]
        c=torch.zeros_like(h)
        for j in range(batch['max_context']):
            if self.encoder_direction=='backward':
                index=(cutoff-1-j).clamp_min(0)
            else:
                index=torch.minimum(torch.full_like(cutoff,j),cutoff-1)
            alive=j<cutoff
            age=ages.gather(1,index[:,None]).squeeze(1)
            x=inputs[torch.arange(b,device=values.device),index]
            if self.kind=='lstm':
                proposed,proposed_c=self.encoder_cell(torch.cat([x,age[:,None]/100.],-1),(h,c))
                c=torch.where(alive[:,None],proposed_c,c)
            else:
                delta=torch.where(alive,age-previous,torch.zeros_like(age))
                evolved=self.solve(self.backward_ode,h,delta)
                proposed=self.update(torch.cat([evolved,x],-1))
            h=torch.where(alive[:,None],proposed,h)
            previous=torch.where(alive,age,previous)
        if self.kind=='deterministic':
            mu=h;std=torch.zeros_like(mu);kl=mu.new_zeros(())
        else:
            mu,logvar=h.chunk(2,-1)
            logvar=logvar.clamp(-12.,8.)
            std=(.5*logvar).exp()
            kl=.5*(mu.square()+logvar.exp()-1-logvar).sum(-1).mean()
        z=mu+std*torch.randn_like(mu) if sample and self.kind!='deterministic' else mu
        if self.encoder_direction=='forward':
            return {'prediction':self.generate_from_cutoff(z,batch),'mu':mu,'std':std,'kl':kl}
        previous=ages[:,0];predictions=[];cell=torch.zeros_like(z)
        for i in range(n):
            if self.kind=='lstm':
                proposed,pc=self.generator_cell(ages[:,i,None]/100.,(z,cell))
                z=torch.where(batch['valid'][:,i,None],proposed,z)
                cell=torch.where(batch['valid'][:,i,None],pc,cell)
            elif i:
                delta=torch.where(batch['valid'][:,i],ages[:,i]-previous,torch.zeros_like(previous))
                z=self.solve(self.forward_ode,z,delta)
            predictions.append(self.decoder(z))
            previous=torch.where(batch['valid'][:,i],ages[:,i],previous)
        return {'prediction':torch.stack(predictions,1),'mu':mu,'std':std,'kl':kl}

    def generate_from_cutoff(self,z,batch):
        """Use the final history estimate as the shared past/future anchor."""
        ages,cutoff=batch['ages'],batch['cutoff']
        b,n=ages.shape
        anchor=ages.gather(1,(cutoff-1)[:,None]).squeeze(1)
        predictions=[z.new_zeros(b,input_width(self.schema)) for _ in range(n)]
        state=z;previous=anchor
        for i in reversed(range(batch['max_context'])):
            active=i<cutoff
            delta=torch.where(active,ages[:,i]-previous,torch.zeros_like(previous))
            state=self.solve(self.forward_ode,state,delta)
            predictions[i]=torch.where(active[:,None],self.decoder(state),predictions[i])
            previous=torch.where(active,ages[:,i],previous)
        # Restart at the cutoff estimate, rather than numerically round-tripping
        # the past trajectory before forecasting.
        state=z;previous=anchor
        for i in range(n):
            active=(i>=cutoff)&batch['valid'][:,i]
            delta=torch.where(active,ages[:,i]-previous,torch.zeros_like(previous))
            state=self.solve(self.forward_ode,state,delta)
            predictions[i]=torch.where(active[:,None],self.decoder(state),predictions[i])
            previous=torch.where(active,ages[:,i],previous)
        return torch.stack(predictions,1)

    def ode_regularization(self):
        if self.kind=='lstm':
            return next(self.parameters()).new_zeros(())
        modules=list({id(module):module for module in [self.backward_ode,self.forward_ode]}.values())
        value=sum(p.square().sum() for module in modules for p in module.parameters())
        return value*self.shared_l2_multiplier if self.shared_vector_field else value


def paper_model_from_config(schema,cfg):
    optional=['init_width','update_width','decoder_width','encoder_ode_width','generator_ode_width',
              'shared_vector_field','shared_l2_multiplier']
    return PaperModel(schema,cfg['model'],cfg['latent'],cfg['ode_width'],cfg['mlp_width'],
        cfg['rtol'],cfg['atol'],cfg.get('encoder_direction','backward'),cfg.get('encoder_initialization'),
        **{k:cfg[k] for k in optional if k in cfg})


def feature_errors(prediction,values,mask,schema):
    """One loss per original field, not per expanded one-hot dimension."""
    errors=[];at=0
    for j,f in enumerate(schema):
        width=f.get('classes',1)
        if f['kind']=='categorical':
            logits=prediction[...,at:at+width]
            error=F.cross_entropy(logits.reshape(-1,width),values[...,j].long().reshape(-1),reduction='none').reshape(values.shape[:2])
        else:
            error=(prediction[...,at]-values[...,j]).square()
        errors.append(error*mask[...,j]);at+=width
    return torch.stack(errors,-1)


def reconstruction(errors,mask,schema):
    counts=mask.sum((0,1)); means=errors.sum((0,1))/counts.clamp_min(1)
    numerical=torch.tensor([f['kind']=='numerical' for f in schema],device=errors.device)
    valid=counts>0
    return means[numerical&valid].sum(),means[(~numerical)&valid].sum()
