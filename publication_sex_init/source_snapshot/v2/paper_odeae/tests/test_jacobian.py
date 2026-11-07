"""CUDA checks of the exact Jacobian value, weight gradients and masks."""
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from torch import nn
from paper_odeae.data import collate
from paper_odeae.models import paper_model_from_config
from paper_odeae.jacobian import squared_jacobian_norm,jacobian_regularization
from paper_odeae.train import device_batch


class JacobianChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(1);torch.manual_seed(9)

    def test_exact_value_and_parameter_derivative(self):
        field=nn.Sequential(nn.Linear(6,13),nn.Tanh(),nn.Linear(13,6)).cuda()
        z=torch.randn(2,6,device='cuda')
        actual=squared_jacobian_norm(field,z).mean()
        reference=torch.stack([torch.autograd.functional.jacobian(field,x,create_graph=True).square().sum() for x in z]).mean()
        torch.testing.assert_close(actual,reference,rtol=2e-6,atol=1e-7)
        aa=torch.autograd.grad(actual,list(field.parameters()),allow_unused=True)
        bb=torch.autograd.grad(reference,list(field.parameters()),allow_unused=True)
        for a,b in zip(aa,bb):
            if a is None or b is None:self.assertIs(a,b)
            else:torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)

    def test_collected_states_mask_and_detach(self):
        ref=Path(__file__).resolve().parents[1]/'results/forward_age_scaling/shared_matched/seed_0'
        cfg=json.loads((ref/'config.json').read_text());schema=json.loads((ref/'schema.json').read_text())
        ordinary=paper_model_from_config(schema,cfg).cuda()
        model=paper_model_from_config(schema,{**cfg,'collect_jacobian_states':True,'jacobian_weight':.01}).cuda()
        model.load_state_dict(ordinary.state_dict())
        items=[];rng=np.random.default_rng(1)
        for n in [3,2]:
            values=rng.random((n,len(schema))).astype(np.float32)
            for j,f in enumerate(schema):
                if f['kind']=='categorical':values[:,j]=np.arange(n)%f['classes']
            items.append(({'subject_id':str(n),'ages':70+np.arange(n,dtype=np.float32),
                           'values':values,'mask':np.ones_like(values)},2))
        batch=device_batch(collate(items));out=model(batch)
        self.assertEqual(sum(p.numel() for p in model.parameters()),22545)
        self.assertIs(model.backward_ode,model.forward_ode)
        torch.testing.assert_close(out['prediction'],ordinary(batch)['prediction'],rtol=0,atol=0)
        decoded=model.decoder(out['jacobian_states'])
        torch.testing.assert_close(decoded[batch['valid']],out['prediction'][batch['valid']],rtol=0,atol=0)
        loss,total,n=jacobian_regularization(model,out,batch)
        self.assertEqual(n,5)
        torch.testing.assert_close(loss,squared_jacobian_norm(model.forward_ode,out['jacobian_states'][batch['valid']].detach()).mean())
        modified={'jacobian_states':out['jacobian_states'].clone()}
        modified['jacobian_states'][~batch['valid']]=1e5
        torch.testing.assert_close(loss,jacobian_regularization(model,modified,batch)[0],rtol=0,atol=0)
        loss.backward()
        self.assertGreater(float(model.forward_ode[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.forward_ode[2].weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in model.init.parameters()))
        self.assertTrue(all(p.grad is None for p in model.update.parameters()))
        self.assertTrue(all(p.grad is None for p in model.decoder.parameters()))


if __name__=='__main__':unittest.main()
