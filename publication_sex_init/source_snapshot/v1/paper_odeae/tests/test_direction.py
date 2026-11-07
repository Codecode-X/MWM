"""Forward encoder ablations; all model work uses CUDA."""
import json
from pathlib import Path
import unittest

import torch
from paper_odeae.models import PaperModel, encode_values, feature_errors, reconstruction
from paper_odeae.tests.test_models import SCHEMA, batch


class DirectionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU diagnostics')
        torch.set_num_threads(1)

    def model(self,direction):
        torch.manual_seed(19)
        return PaperModel(SCHEMA,'deterministic',encoder_direction=direction).cuda()

    def test_shared_parameters_identical_and_init_removed(self):
        schema=json.loads((Path(__file__).resolve().parents[1]/'reference/features46.json').read_text())
        missing={'ecog_self','ecog_partner','npi','fdg_parietal_suvr','fdg_temporal_suvr','cortical_volume_change_percent'}
        schema=[f for f in schema if f['name'] not in missing]
        models=[]
        for direction in ['backward','forward']:
            torch.manual_seed(4)
            models.append(PaperModel(schema,'deterministic',6,118,154,encoder_direction=direction).cuda())
        self.assertEqual(sum(p.numel() for p in models[0].parameters()),22545)
        self.assertEqual(sum(p.numel() for p in models[1].parameters()),21307)
        self.assertFalse(hasattr(models[1],'init'))
        for name,value in models[1].state_dict().items():
            torch.testing.assert_close(value,models[0].state_dict()[name],rtol=0,atol=0)

    def test_order_intervals_and_zero_initialization(self):
        b=batch();b['cutoff']=torch.tensor([1,3],device='cuda');b['max_context']=3
        index=torch.arange(b['values'].shape[1],device='cuda')[None,:]
        b['context_mask']=b['mask']*(index<b['cutoff'][:,None])[...,None]
        encoded=encode_values(b['values'],b['context_mask'],SCHEMA)
        for direction in ['backward','forward']:
            model=self.model(direction);updates=[];deltas=[];initial=[];states=[]
            model.update.register_forward_pre_hook(lambda module,args:updates.append(args[0][:,model.latent:].detach().clone()))
            model.update.register_forward_pre_hook(lambda module,args:states.append(args[0][:,:model.latent].detach().clone()))
            if direction=='backward':
                model.init.register_forward_pre_hook(lambda module,args:initial.append(args[0].detach().clone()))
            def solve(func,state,delta):
                if func is model.backward_ode:deltas.append(delta.detach().clone())
                return state
            model.solve=solve
            out=model(b)
            if direction=='backward':
                age=b['ages'].gather(1,(b['cutoff']-1)[:,None])/100.
                torch.testing.assert_close(initial[0],age,rtol=0,atol=0)
            else:
                torch.testing.assert_close(states[0],torch.zeros_like(states[0]),rtol=0,atol=0)
            order=[2,1,0] if direction=='backward' else [0,1,2]
            for observed,j in zip(updates,order):
                torch.testing.assert_close(observed[1],encoded[1,j],rtol=0,atol=0)
            self.assertTrue(all(bool((d<=0).all()) for d in deltas) if direction=='backward'
                            else all(bool((d>=0).all()) for d in deltas))
            if direction=='backward':
                torch.testing.assert_close(out['prediction'][:,0],model.decoder(out['mu']),rtol=0,atol=0)
            else:
                at_cutoff=out['prediction'][torch.arange(2,device='cuda'),b['cutoff']-1]
                torch.testing.assert_close(at_cutoff,model.decoder(out['mu']),rtol=0,atol=0)

    def test_future_isolation_cutoff_anchor_padding_and_gradients(self):
        b=batch();forward=self.model('forward')
        out=forward(b)
        future=torch.arange(b['values'].shape[1],device='cuda')[None,:]>=b['cutoff'][:,None]
        changed=b['values'].clone();changed[...,:2]+=1000*future[...,None]
        changed[...,2]=torch.where(future,2-changed[...,2],changed[...,2])
        torch.testing.assert_close(out['prediction'],forward({**b,'values':changed})['prediction'],rtol=0,atol=0)
        at_cutoff=out['prediction'][torch.arange(2,device='cuda'),b['cutoff']-1]
        torch.testing.assert_close(at_cutoff,forward.decoder(out['mu']),rtol=0,atol=0)
        padded=b['values'].clone();padded[0,3:,:2]=999
        torch.testing.assert_close(out['prediction'][0,:3],forward({**b,'values':padded})['prediction'][0,:3],rtol=0,atol=0)
        errors=feature_errors(out['prediction'],b['values'],b['mask'],SCHEMA)
        mse,cce=reconstruction(errors,b['mask'],SCHEMA)
        (mse+cce+.02*forward.ode_regularization()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in forward.parameters() if p.grad is not None))

    def test_generating_past_does_not_move_the_future_anchor(self):
        b=batch();model=self.model('forward')
        values=[]
        def solve(func,state,delta):
            if func is model.forward_ode:
                values.append((state.detach().clone(),delta.detach().clone()))
                return state+delta[:,None]
            return state
        model.solve=solve
        out=model(b)
        # Past reconstruction starts at mu, goes backwards, then future starts
        # afresh at exactly mu, independently of the reconstructed past state.
        torch.testing.assert_close(values[0][0],out['mu'],rtol=0,atol=0)
        torch.testing.assert_close(values[b['max_context']][0],out['mu'],rtol=0,atol=0)
        self.assertTrue(all(bool((delta<=0).all()) for _,delta in values[:b['max_context']]))
        self.assertTrue(all(bool((delta>=0).all()) for _,delta in values[b['max_context']:]))

    def test_age_initialized_forward_version_and_shared_weights(self):
        zero=self.model('forward')
        torch.manual_seed(19)
        age=PaperModel(SCHEMA,'deterministic',encoder_direction='forward',encoder_initialization='age').cuda()
        for name,value in zero.state_dict().items():
            torch.testing.assert_close(value,age.state_dict()[name],rtol=0,atol=0)
        b=batch();seen=[]
        age.init.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
        out=age(b)
        torch.testing.assert_close(seen[0],b['ages'].gather(1,(b['cutoff']-1)[:,None])/100.,rtol=0,atol=0)
        at_cutoff=out['prediction'][torch.arange(2,device='cuda'),b['cutoff']-1]
        torch.testing.assert_close(at_cutoff,age.decoder(out['mu']),rtol=0,atol=0)
        errors=feature_errors(out['prediction'],b['values'],b['mask'],SCHEMA)
        mse,cce=reconstruction(errors,b['mask'],SCHEMA)
        (mse+cce+.02*age.ode_regularization()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in age.parameters() if p.grad is not None))


if __name__=='__main__':unittest.main()
