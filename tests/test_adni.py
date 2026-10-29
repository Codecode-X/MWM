"""Data checks use CPU; all model forward/gradient/solver checks require CUDA."""
import unittest
import numpy as np
import torch

from src.adni_data import numeric
from src.adni_models import integrate, make_model, forward
from src.adni_train import collate, to_cuda
from src.model import solve


class DataChecks(unittest.TestCase):
    def test_missing_codes_and_censoring(self):
        for v in [None, '-4', '>1700', '<200', float('nan'), float('inf')]:
            self.assertIsNone(numeric(v,0,2000))
        self.assertEqual(numeric('25.5',0,30),25.5)

    def test_padding_and_strict_forecast_mask(self):
        a = {'patient_id':'a','times':np.array([.25,.5,1.]),
             'values':np.ones((3,2),np.float32),'mask':np.ones((3,2),np.float32)}
        b = {'patient_id':'b','times':np.array([.25,.75,1.,1.5,2.]),
             'values':np.ones((5,2),np.float32),'mask':np.ones((5,2),np.float32)}
        batch=collate([a,b])
        self.assertEqual(batch['target_mask'][0,3:].sum(),0)
        self.assertEqual(batch['context_mask'][0,1:].sum(),0)
        self.assertEqual(batch['context_mask'][1,2:].sum(),0)
        self.assertTrue(torch.all(batch['times'][:,1:]>=batch['times'][:,:-1]))


@unittest.skipUnless(torch.cuda.is_available(),'CUDA mandatory for model checks')
class GPUChecks(unittest.TestCase):
    def batch(self):
        rng=np.random.default_rng(17)
        seqs=[{'patient_id':str(i),'times':np.array([.25,.5,.75,1.5,2.])[:n],
               'values':rng.normal(size=(n,3)).astype(np.float32),
               'mask':np.ones((n,3),np.float32)} for i,n in enumerate([3,5])]
        return to_cuda(collate(seqs))

    def test_future_values_never_enter_prediction(self):
        for kind in ['latent_ode','ode_rnn']:
            torch.manual_seed(11)
            model=make_model(kind,3,latent=4,hidden=6,width=8).cuda().eval()
            batch=self.batch()
            torch.cuda.manual_seed_all(33)
            before=forward(model,batch,kind)['prediction']
            batch['values']=torch.where(batch['is_future'][...,None],
                                        batch['values']+1000,batch['values'])
            torch.cuda.manual_seed_all(33)
            after=forward(model,batch,kind)['prediction']
            torch.testing.assert_close(before,after,atol=0,rtol=0)

    def test_gradient_and_finite_outputs(self):
        for kind in ['latent_ode','ode_rnn']:
            model=make_model(kind,3,latent=4,hidden=6,width=8).cuda()
            batch=self.batch()
            batch['values'].requires_grad_(True)
            prediction=forward(model,batch,kind,training=True)['prediction']
            self.assertEqual(prediction.device.type,'cuda')
            self.assertTrue(torch.isfinite(prediction).all())
            (prediction.square()*batch['target_mask'][None]).sum().backward()
            self.assertTrue(torch.isfinite(batch['values'].grad).all())
            self.assertEqual(batch['values'].grad[batch['is_future']].abs().sum(),0)

    def test_rk4_agrees_with_original_solver(self):
        model=make_model('ode_rnn',3,latent=4,hidden=6,width=8).cuda()
        h=torch.randn(2,4,device='cuda')
        deltas=torch.tensor([.1,.4],device='cuda')
        got=integrate(model.ode,h,deltas,16)
        expected=torch.cat([solve(model.ode,h[i:i+1],
                              torch.stack([deltas[i]*0,deltas[i]]))[-1] for i in range(2)])
        torch.testing.assert_close(got,expected,atol=1e-4,rtol=1e-4)


if __name__=='__main__':
    unittest.main()
