"""CUDA tests of causal pre-update predictions and loss masks."""
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from paper_odeae.data import collate
from paper_odeae.models import paper_model_from_config, prior_errors, reconstruction
from paper_odeae.train import device_batch


class PriorLossChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(1)
        ref=Path(__file__).resolve().parents[1]/'results/forward_age_scaling/shared_matched/seed_0'
        cls.cfg=json.loads((ref/'config.json').read_text());cls.schema=json.loads((ref/'schema.json').read_text())
        rng=np.random.default_rng(5);items=[]
        for cutoff in [2,1]:
            values=rng.random((3,len(cls.schema))).astype(np.float32)
            for j,f in enumerate(cls.schema):
                if f['kind']=='categorical':values[:,j]=np.arange(3)%f['classes']
            items.append(({'subject_id':str(cutoff),'ages':np.array([70.,71.,72.],np.float32),
                           'values':values,'mask':np.ones_like(values)},cutoff))
        cls.batch=device_batch(collate(items))

    def model(self):
        return paper_model_from_config(self.schema,{**self.cfg,'collect_prior_predictions':True}).cuda()

    def test_causality_and_mask_and_gradient(self):
        m=self.model();out=m(self.batch)
        changed={k:v.clone() if torch.is_tensor(v) else v for k,v in self.batch.items()}
        numerical=next(i for i,f in enumerate(self.schema) if f['kind']=='numerical')
        changed['values'][0,1,numerical]+=1
        # The current value changes the update, but cannot change its prior.
        torch.testing.assert_close(out['prior_prediction'][0,1],m(changed)['prior_prediction'][0,1],rtol=0,atol=0)
        pe,mask=prior_errors(out,self.batch,self.schema)
        self.assertEqual(int(mask[:,0].count_nonzero()),0)
        self.assertEqual(int(mask[1].count_nonzero()),0)
        self.assertEqual(int(mask[0,1].count_nonzero()),len(self.schema))
        a,b=reconstruction(pe,mask,self.schema);(.01*(a+b)).backward()
        for module in [m.init,m.update,m.forward_ode,m.decoder]:
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()),0)

    def test_no_capacity_or_prediction_change(self):
        ordinary=paper_model_from_config(self.schema,self.cfg).cuda();collect=self.model()
        collect.load_state_dict(ordinary.state_dict())
        self.assertEqual(sum(p.numel() for p in collect.parameters()),22545)
        self.assertIs(collect.backward_ode,collect.forward_ode)
        torch.testing.assert_close(ordinary(self.batch)['prediction'],collect(self.batch)['prediction'],rtol=0,atol=0)
        single={k:v.clone() if torch.is_tensor(v) else v for k,v in self.batch.items()}
        single['cutoff']=torch.ones_like(single['cutoff']);single['max_context']=1
        pe,mask=prior_errors(collect(single),single,self.schema)
        a,b=reconstruction(pe,mask,self.schema)
        self.assertEqual(float(a+b),0.)
        (a+b).backward()


if __name__=='__main__':unittest.main()
