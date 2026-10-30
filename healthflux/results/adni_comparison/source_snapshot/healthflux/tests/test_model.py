"""CUDA tests for HealthFlux equations, missingness and temporal isolation."""
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from paper_odeae.data import collate
from paper_odeae.train import device_batch
from paper_odeae.models import feature_errors,reconstruction
from healthflux.model import HealthFlux


class HealthFluxChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(1);torch.manual_seed(0)
        cls.schema=json.loads(Path('paper_odeae/results/forward_age_scaling/shared_matched/seed_0/schema.json').read_text())
        cls.cfg={'deterministic_dim':8,'stochastic_dim':4,'context_dim':6,'age_dim':4,'observation_dim':8,'internal_width':12,'decoder_widths':[16,8],'dropout':0.}
        items=[];rng=np.random.default_rng(0)
        for n,cutoff in [(4,2),(3,1)]:
            values=rng.random((n,len(cls.schema))).astype(np.float32)
            for j,f in enumerate(cls.schema):
                if f['kind']=='categorical':values[:,j]=np.arange(n)%f['classes']
            sequence={'subject_id':str(n),'ages':70+np.arange(n,dtype=np.float32),'values':values,'mask':np.ones_like(values)}
            items.append((sequence,cutoff))
        cls.batch=device_batch(collate(items))

    def test_causal_encoding_and_forecast(self):
        model=HealthFlux(self.schema,self.cfg).cuda().eval();batch=self.batch
        original=model(batch)
        changed={key:value.clone() if torch.is_tensor(value) else value for key,value in batch.items()}
        for i,cutoff in enumerate(batch['cutoff'].tolist()):
            for j,f in enumerate(self.schema):
                if f['kind']=='categorical':changed['values'][i,cutoff:,j]=(changed['values'][i,cutoff:,j]+1)%f['classes']
                else:changed['values'][i,cutoff:,j]+=100
            changed['mask'][i,cutoff:]=0
        other=model(changed)
        torch.testing.assert_close(original['mu'],other['mu'],rtol=0,atol=0)
        torch.testing.assert_close(original['stochastic_mean'],other['stochastic_mean'],rtol=0,atol=0)
        torch.testing.assert_close(original['prediction'],other['prediction'],rtol=0,atol=0)
        # Also changing future ages cannot affect the encoded cutoff state or age INIT.
        for i,cutoff in enumerate(batch['cutoff'].tolist()):changed['ages'][i,cutoff:]+=50
        torch.testing.assert_close(model.encode(changed)[0],original['mu'],rtol=0,atol=0)

    def test_rk4_decade_clock_and_reverse(self):
        model=HealthFlux(self.schema,self.cfg).cuda().eval()
        with torch.no_grad():
            for p in model.dynamics.parameters():p.zero_()
            model.dynamics[-1].bias.fill_(float(np.arctanh(.2)))
        h=torch.zeros(2,8,device='cuda');age=torch.tensor([70.,72.],device='cuda');context=torch.zeros(2,6,device='cuda')
        dt=torch.tensor([5.,20.],device='cuda');future=model.evolve(h,age,dt,context)
        torch.testing.assert_close(future,h+dt[:,None]/10*.2,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(model.evolve(future,age+dt,-dt,context),h,rtol=0,atol=1e-7)
        torch.testing.assert_close(model.evolve(h,age,dt*0,context),h,rtol=0,atol=0)

    def test_missing_nodes_do_not_jump_and_gradients_finite(self):
        model=HealthFlux(self.schema,self.cfg).cuda();batch=self.batch
        h=torch.randn(2,8,device='cuda');age=batch['ages'][:,0];context=model.context(batch)
        out=model.jump(h,age,context,torch.randn(2,8,device='cuda'),torch.zeros(2,dtype=torch.bool,device='cuda'),False)
        torch.testing.assert_close(out[0],h,rtol=0,atol=0);self.assertEqual(float(out[2].sum()),0.)
        result=model(batch,sample=True)
        mse,cce=reconstruction(feature_errors(result['prediction'],batch['values'],batch['mask'],self.schema),batch['mask'],self.schema)
        loss=mse+cce+.01*result['kl']+.02*model.ode_regularization();self.assertTrue(torch.isfinite(loss));loss.backward()
        for module in [model.dynamics,model.prior,model.posterior,model.update,model.decoder]:
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))

    def test_native_dimensions_and_age_conditioning(self):
        model=HealthFlux(self.schema,{}).cuda().eval();self.assertEqual((model.h_dim,model.z_dim),(256,128))
        self.assertEqual(model.decoder[0].in_features,384)
        h=torch.zeros(2,256,device='cuda');c=torch.zeros(2,128,device='cuda')
        v=model.vector_field(h,torch.tensor([60.,80.],device='cuda'),c)
        self.assertGreater(float((v[0]-v[1]).abs().max()),1e-5)
        self.assertLessEqual(float(v.abs().max()),1.)


if __name__=='__main__':unittest.main()
