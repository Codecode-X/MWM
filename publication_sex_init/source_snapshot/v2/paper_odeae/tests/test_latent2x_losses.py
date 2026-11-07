"""CUDA checks for the 28D ablations and equivalence to the frozen baseline."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from paper_odeae.data import collate
from paper_odeae.jacobian import squared_jacobian_norm, jacobian_regularization
from paper_odeae.models import paper_model_from_config, feature_errors, reconstruction, prior_errors
from paper_odeae.train import device_batch


class Latent2xLossChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(1)
        root=Path(__file__).resolve().parents[2]
        base=root/'paper_odeae/results/shared_matched_latent_scaling/latent_2x/seed_0'
        cls.cfg=json.loads((base/'config.json').read_text())
        cls.schema=json.loads((base/'schema.json').read_text())
        old=root/'paper_odeae/results/shared_matched_latent_scaling/source_snapshot/paper_odeae/models.py'
        spec=importlib.util.spec_from_file_location('paper_odeae._frozen_latent2x_models',old)
        cls.frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.frozen)
        cls.assert_solver=(old.parent/'solver.py').read_bytes()==(root/'paper_odeae/solver.py').read_bytes()
        items=[];rng=np.random.default_rng(3)
        for n in [3,2]:
            values=rng.random((n,len(cls.schema))).astype(np.float32)
            for j,f in enumerate(cls.schema):
                if f['kind']=='categorical':values[:,j]=np.arange(n)%f['classes']
            items.append(({'subject_id':str(n),'ages':70+np.arange(n,dtype=np.float32),
                           'values':values,'mask':np.ones_like(values)},2))
        cls.batch=device_batch(collate(items))

    def test_frozen_baseline_equivalence(self):
        self.assertTrue(self.assert_solver)
        torch.manual_seed(0);old=self.frozen.paper_model_from_config(self.schema,self.cfg).cuda()
        torch.manual_seed(0);new=paper_model_from_config(self.schema,self.cfg).cuda()
        self.assertEqual(sum(p.numel() for p in new.parameters()),44875)
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        for key,value in old.state_dict().items():
            torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
        out_old=old(self.batch);out_new=new(self.batch)
        torch.testing.assert_close(out_old['prediction'],out_new['prediction'],rtol=0,atol=0)
        losses=[]
        for model,out in [(old,out_old),(new,out_new)]:
            mse,cce=reconstruction(feature_errors(out['prediction'],self.batch['values'],self.batch['mask'],self.schema),self.batch['mask'],self.schema)
            loss=mse+cce+.02*model.ode_regularization();loss.backward();losses.append(loss)
        torch.testing.assert_close(*losses,rtol=0,atol=0)
        for a,b in zip(old.parameters(),new.parameters()):
            if a.grad is None or b.grad is None:self.assertIs(a.grad,b.grad)
            else:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)

    def test_three_losses_and_exact_28d_jacobian(self):
        torch.manual_seed(0);ordinary=paper_model_from_config(self.schema,self.cfg).cuda()
        ordinary_prediction=ordinary(self.batch)['prediction'].detach()
        for prior,jac in [(.01,.01),(.01,0.),(0.,.01)]:
            with self.subTest(prior=prior,jacobian=jac):
                cfg={**self.cfg,'collect_prior_predictions':True,'collect_jacobian_states':True,
                     'prior_weight':prior,'jacobian_weight':jac}
                torch.manual_seed(0);model=paper_model_from_config(self.schema,cfg).cuda()
                self.assertEqual(sum(p.numel() for p in model.parameters()),44875)
                self.assertIs(model.forward_ode,model.backward_ode)
                out=model(self.batch)
                torch.testing.assert_close(out['prediction'],ordinary_prediction,rtol=0,atol=0)
                decoded=model.decoder(out['jacobian_states'])
                torch.testing.assert_close(decoded[self.batch['valid']],out['prediction'][self.batch['valid']],rtol=0,atol=0)
                mse,cce=reconstruction(feature_errors(out['prediction'],self.batch['values'],self.batch['mask'],self.schema),self.batch['mask'],self.schema)
                pe,mask=prior_errors(out,self.batch,self.schema);pmse,pcce=reconstruction(pe,mask,self.schema)
                jl,_,count=jacobian_regularization(model,out,self.batch);self.assertEqual(count,5)
                z=out['jacobian_states'][self.batch['valid']][0].detach()
                exact=torch.autograd.functional.jacobian(model.forward_ode,z,create_graph=True)
                self.assertEqual(tuple(exact.shape),(28,28))
                analytic=squared_jacobian_norm(model.forward_ode,z[None]).mean()
                torch.testing.assert_close(analytic,exact.square().sum(),rtol=2e-6,atol=1e-7)
                ga=torch.autograd.grad(analytic,model.forward_ode[0].weight,retain_graph=True)[0]
                gb=torch.autograd.grad(exact.square().sum(),model.forward_ode[0].weight,retain_graph=True)[0]
                torch.testing.assert_close(ga,gb,rtol=1e-5,atol=1e-7)
                loss=mse+cce+.02*model.ode_regularization()+prior*(pmse+pcce)+jac*jl
                self.assertTrue(torch.isfinite(loss));loss.backward()
                for p in model.parameters():
                    self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())


if __name__=='__main__':unittest.main()
