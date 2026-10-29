"""All model/solver checks execute on CUDA; CPU only prepares toy observations."""
import math
import unittest
import numpy as np
import torch
from paper_odeae.data import Prefixes,collate
from paper_odeae.models import PaperModel,feature_errors,reconstruction
from paper_odeae.solver import integrate

SCHEMA=[{'name':'a','kind':'numerical'},{'name':'b','kind':'numerical'},
        {'name':'diagnosis','kind':'categorical','classes':3}]


def batch():
    rng=np.random.default_rng(4);items=[]
    for j,n in enumerate([3,5]):
        values=rng.normal(size=(n,3)).astype(np.float32);values[:,2]=np.arange(n)%3
        s={'subject_id':str(j),'ages':np.array([70.,70.25,71.,72.,73.],np.float32)[:n],
           'values':values,'mask':np.ones_like(values)}
        items.append((s,2))
    return {k:v.cuda() if torch.is_tensor(v) else v for k,v in collate(items).items()}


class DataChecks(unittest.TestCase):
    def test_all_prefixes_including_full_history(self):
        sequences=[{'ages':np.zeros(n)} for n in [2,3]]
        prefixes=Prefixes(sequences)
        self.assertEqual(len(prefixes),5)
        self.assertEqual([prefixes[i][1] for i in range(5)],[1,2,1,2,3])


class CUDAChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise RuntimeError('CUDA diagnostics cannot fall back to CPU')
        torch.set_num_threads(1)

    def test_tsit5_forward_backward_and_gradients(self):
        y=torch.ones(2,3,device='cuda',requires_grad=True)
        delta=torch.tensor([2.,-1.5],device='cuda')
        result=integrate(lambda z:.3*z,y,delta,rtol=1e-5,atol=1e-7)
        expected=y*(.3*delta[:,None]).exp()
        torch.testing.assert_close(result,expected,rtol=1e-4,atol=1e-5)
        result.sum().backward()
        torch.testing.assert_close(y.grad,(.3*delta[:,None]).exp().expand_as(y),rtol=1e-4,atol=1e-5)

    def test_future_values_do_not_change_prediction(self):
        for kind in ['ode_ae','deterministic','lstm']:
            torch.manual_seed(2);model=PaperModel(SCHEMA,kind).cuda().eval();b=batch()
            before=model(b,sample=False)['prediction']
            future=torch.arange(b['values'].shape[1],device='cuda')[None,:]>=b['cutoff'][:,None]
            values=b['values'].clone();values[...,:2]+=1000*future[...,None]
            values[...,2]=torch.where(future,2-values[...,2],values[...,2])
            b['values']=values
            after=model(b,sample=False)['prediction']
            torch.testing.assert_close(before,after,rtol=0,atol=0)

    def test_first_visit_anchor_and_padding(self):
        b=batch();model=PaperModel(SCHEMA,'deterministic').cuda().eval()
        out=model(b,sample=False)
        torch.testing.assert_close(out['prediction'][:,0],model.decoder(out['mu']),rtol=0,atol=0)
        altered={**b,'values':b['values'].clone()};altered['values'][0,3:,:2]=100
        torch.testing.assert_close(model(altered)['prediction'][0,:3],out['prediction'][0,:3],rtol=0,atol=0)

    def test_sum_of_field_mse_and_category_cross_entropy(self):
        schema=SCHEMA[:2]+[{'name':'c','kind':'categorical','classes':2}]
        pred=torch.tensor([[[1.,2.,0.,0.]]],device='cuda',requires_grad=True)
        values=torch.zeros(1,1,3,device='cuda');mask=torch.ones_like(values)
        errors=feature_errors(pred,values,mask,schema)
        mse,cce=reconstruction(errors,mask,schema)
        self.assertAlmostEqual(float(mse),5.)
        self.assertAlmostEqual(float(cce),math.log(2),places=6)
        (mse+cce).backward();self.assertTrue(torch.isfinite(pred.grad).all())

    def test_three_models_have_finite_training_gradients(self):
        for kind in ['ode_ae','deterministic','lstm']:
            model=PaperModel(SCHEMA,kind).cuda();b=batch();b['values'].requires_grad_(True)
            out=model(b,sample=True)
            losses=feature_errors(out['prediction'],b['values'].detach(),b['mask'],SCHEMA)
            mse,cce=reconstruction(losses,b['mask'],SCHEMA)
            (mse+cce+out['kl']+.02*model.ode_regularization()).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertEqual(out['prediction'].device.type,'cuda')
            self.assertEqual(float(b['values'].grad[:,2:,:2].abs().sum()),0.)


if __name__=='__main__':unittest.main()
