"""CUDA head derivatives, free-ODE semantics and small cohort boundary checks."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from paper_odeae.diagnostics.continuous_stage import (
    probabilities_and_stage,instantaneous_stage,mci_origin,conversion_label)


class ContinuousStageChecks(unittest.TestCase):
    def test_existing_head_autodiff_and_probability_bound(self):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        head=nn.Linear(2,3,bias=False,device='cuda',dtype=torch.float64)
        field=nn.Linear(2,2,bias=False,device='cuda',dtype=torch.float64)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[-1.,0.],[0.,1.],[1.,0.]],device='cuda',dtype=torch.float64))
            field.weight.copy_(torch.eye(2,device='cuda',dtype=torch.float64))
        model=SimpleNamespace(schema=[{'name':'diagnosis','kind':'categorical','classes':3}],decoder=head,forward_ode=field)
        z=torch.tensor([[0.,0.],[1.,0.],[-1.,0.]],device='cuda',dtype=torch.float64)
        p,s,rate,gradient,f=instantaneous_stage(model,z)
        self.assertTrue(bool(((s>=0)&(s<=2)).all()))
        torch.testing.assert_close(p.sum(-1),torch.ones(3,device='cuda',dtype=torch.float64))
        self.assertEqual(float(s[0]),1.);self.assertEqual(float(rate[0]),0.)
        self.assertGreater(float(rate[1]),0.);self.assertLess(float(rate[2]),0.)
        eps=1e-5
        fd=(probabilities_and_stage(model,z+eps*f)[1]-probabilities_and_stage(model,z-eps*f)[1])/(2*eps)
        torch.testing.assert_close(fd,rate,rtol=1e-8,atol=1e-9)
        torch.testing.assert_close(rate,(gradient*f).sum(-1),rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in head.parameters()))

    def sequence(self,diagnoses,dates):
        return {'ages':np.arange(len(diagnoses)),'values':np.array(diagnoses,float)[:,None],
                'mask':np.ones((len(diagnoses),1)), 'dates':dates}

    def test_patient_origin_and_followup_support(self):
        dates=['2020-03-31','2020-09-30','2021-03-31','2021-09-30']
        stable=self.sequence([1,1,1,1],dates)
        self.assertEqual(mci_origin(stable,0),0)
        self.assertEqual(conversion_label(stable,0,0,12)['label'],0)
        converted=self.sequence([1,1,2,2],dates)
        self.assertEqual(conversion_label(converted,0,0,12)['label'],1)
        # The first Dementia observation after month 12 does not label month 12 as conversion.
        boundary=self.sequence([1,1,2],['2020-03-31','2020-09-30','2021-06-30'])
        self.assertIsNone(conversion_label(boundary,0,0,12)['label'])
        # Both classes need endpoint support; an early observed event alone is not sufficient.
        unsupported=self.sequence([1,2],dates[:2])
        self.assertEqual(conversion_label(unsupported,0,0,12)['reason'],'no_endpoint_support')
        reverted=self.sequence([1,0,1,1],dates)
        self.assertIsNone(conversion_label(reverted,0,0,12)['label'])
        previous_dementia=self.sequence([2,1,1,1],dates)
        self.assertIsNone(mci_origin(previous_dementia,0))
        missing=self.sequence([1,1,1,1],dates);missing['mask'][2:,0]=0
        self.assertIsNone(conversion_label(missing,0,0,12)['label'])


if __name__=='__main__':unittest.main()
