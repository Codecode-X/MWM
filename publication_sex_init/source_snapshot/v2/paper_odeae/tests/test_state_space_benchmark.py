"""CUDA checks for model-specific cutoff states and annual velocity clocks."""
import os
from pathlib import Path
import pickle
import unittest

import torch
from paper_odeae.data import collate
from paper_odeae.train import device_batch
from paper_odeae.diagnostics.state_space_benchmark import Adapter,metadata_rows,ROOT


class BenchmarkStateChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(1)
        cls.payload=pickle.loads((ROOT/'paper_odeae/cache/local_available.pkl').read_bytes())
        seq=next(s for s in cls.payload['sequences'] if len(s['ages'])>=4)
        cls.cpu=collate([(seq,i) for i in [1,2,3,4]])
        cls.batch=device_batch(cls.cpu)

    def test_cutoff_states_and_annual_field(self):
        names=['healthflux_kl'] if os.environ.get('CHECK_HEALTHFLUX')=='1' else ['latent2x_baseline','ode_ae','latent_ode','ode_rnn']
        for name in names:
            with self.subTest(model=name),torch.no_grad():
                adapter=Adapter(name);state,extra=adapter.state(self.batch);adapter.verify(self.batch,state,extra)
                ages=self.batch['ages'].gather(1,(self.batch['cutoff']-1)[:,None]).squeeze(1)
                adapter.model.double();state=state.double();ages=ages.double()
                context=extra.get('context');context=context.double() if context is not None else None
                delta=torch.full((len(state),),1e-3,device='cuda',dtype=torch.float64)
                plus=adapter.solve(state,ages,delta,context);minus=adapter.solve(state,ages,-delta,context)
                numerical=(plus-minus)/(2e-3);expected=adapter.velocity(state,ages,context)
                torch.testing.assert_close(numerical,expected,rtol=3e-4,atol=1e-7)
                self.assertEqual(state.device.type,'cuda')
                self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))


if __name__=='__main__':unittest.main()
