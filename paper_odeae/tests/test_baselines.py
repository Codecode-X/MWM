"""CUDA checks for adapting repository baselines to mixed-field evaluation."""
import unittest

import torch
from paper_odeae.baselines import BaselineModel, expanded_mask, feedback
from paper_odeae.models import feature_errors, reconstruction
from paper_odeae.tests.test_models import SCHEMA, batch
from paper_odeae.train import evaluate


class BaselineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA checks cannot fall back to CPU')
        torch.set_num_threads(1)

    def test_masks_and_feedback_preserve_field_meaning(self):
        mask = torch.tensor([[[1.,0.,1.]]], device='cuda')
        self.assertEqual(expanded_mask(mask,SCHEMA).tolist(), [[[1.,0.,1.,1.,1.]]])
        prediction = torch.tensor([[[2.,3.,0.,1.,2.]]], device='cuda')
        filled = feedback(prediction,SCHEMA)
        torch.testing.assert_close(filled[...,:2], prediction[...,:2])
        torch.testing.assert_close(filled[...,2:].sum(-1), torch.ones(1,1,device='cuda'))

    def test_future_isolation_and_finite_training_gradients(self):
        for kind in ['latent_ode','ode_rnn']:
            torch.manual_seed(3)
            model = BaselineModel(SCHEMA,kind,latent=6,hidden=12,width=16).cuda().eval()
            b = batch()
            prediction = model(b,sample=False)['prediction']
            future = torch.arange(b['values'].shape[1],device='cuda')[None,:]>=b['cutoff'][:,None]
            changed = b['values'].clone()
            changed[...,:2] += future[...,None]*1000
            changed[...,2] = torch.where(future,2-changed[...,2],changed[...,2])
            with torch.no_grad():
                after = model({**b,'values':changed},sample=False)['prediction']
            torch.testing.assert_close(after,prediction,rtol=0,atol=0)
            model.train(); out=model(b,sample=True)
            errors=feature_errors(out['prediction'],b['values'],b['mask'],SCHEMA)
            mse,cce=reconstruction(errors,b['mask'],SCHEMA)
            (mse+cce+out['kl']+.02*model.ode_regularization()).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_cutoff_posterior_anchor_and_padding(self):
        model=BaselineModel(SCHEMA,'latent_ode',latent=6,hidden=12,width=16).cuda().eval()
        b=batch()
        for p in model.core.ode.parameters():
            p.data.zero_()
        out=model(b,sample=False)
        torch.testing.assert_close(out['prediction'],model.core.decoder(out['mu'])[:,None].expand_as(out['prediction']))
        changed={**b,'values':b['values'].clone()};changed['values'][0,3:,:2]=999
        torch.testing.assert_close(model(changed)['prediction'][0,:3],out['prediction'][0,:3],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
