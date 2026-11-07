"""Only the history anchor's autograd edge changes; all original networks remain."""
import torch

from paper_odeae.models import PaperModel,paper_model_from_config,input_width


class HistoryStopGradient(PaperModel):
    def generate_from_cutoff(self,z,batch,return_states=False):
        ages,cutoff=batch['ages'],batch['cutoff'];b,n=ages.shape
        anchor=ages.gather(1,(cutoff-1)[:,None]).squeeze(1)
        predictions=[z.new_zeros(b,input_width(self.schema)) for _ in range(n)]
        states=[torch.zeros_like(z) for _ in range(n)] if return_states else None
        # The ONLY change to the original generate_from_cutoff implementation.
        state=z.detach();previous=anchor
        for i in reversed(range(batch['max_context'])):
            active=i<cutoff;delta=torch.where(active,ages[:,i]-previous,torch.zeros_like(previous))
            state=self.solve(self.forward_ode,state,delta)
            predictions[i]=torch.where(active[:,None],self.decoder(state),predictions[i])
            if return_states:states[i]=torch.where(active[:,None],state,states[i])
            previous=torch.where(active,ages[:,i],previous)
        # Future prediction keeps its complete encoder graph.
        state=z;previous=anchor
        for i in range(n):
            active=(i>=cutoff)&batch['valid'][:,i];delta=torch.where(active,ages[:,i]-previous,torch.zeros_like(previous))
            state=self.solve(self.forward_ode,state,delta)
            predictions[i]=torch.where(active[:,None],self.decoder(state),predictions[i])
            if return_states:states[i]=torch.where(active[:,None],state,states[i])
            previous=torch.where(active,ages[:,i],previous)
        prediction=torch.stack(predictions,1)
        return (prediction,torch.stack(states,1)) if return_states else prediction


def make_causal_model(schema,cfg,stop=False):
    model=paper_model_from_config(schema,cfg,model_class=HistoryStopGradient if stop else PaperModel).cuda()
    model.variant=cfg['variant'];model.generator_anchor=cfg['generator_anchor']
    assert sum(p.numel() for p in model.parameters())==43280
    assert model.forward_ode is model.backward_ode
    return model
