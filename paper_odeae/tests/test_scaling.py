"""CUDA validation of every planned sharing/width/latent configuration."""
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from paper_odeae.models import paper_model_from_config,feature_errors,reconstruction
from paper_odeae.data import collate
from paper_odeae.train import device_batch
from paper_odeae.scaling_study import DEFAULT


class ScalingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
        torch.set_num_threads(1)
        cls.manifest=json.loads((DEFAULT/'manifest.json').read_text())
        cls.schema=json.loads((Path(__file__).resolve().parents[1]/'results/local_available/deterministic/seed_0/schema.json').read_text())
        rng=np.random.default_rng(17);items=[]
        for i,n in enumerate([3,4]):
            values=rng.random((n,len(cls.schema))).astype(np.float32)
            for j,f in enumerate(cls.schema):
                if f['kind']=='categorical':values[:,j]=np.arange(n)%f['classes']
            items.append(({'subject_id':str(i),'ages':70+np.arange(n,dtype=np.float32)*.5,
                'values':values,'mask':np.ones_like(values)},2))
        cls.batch=device_batch(collate(items))

    def test_all_parameter_targets_and_gradients(self):
        for job in self.manifest['jobs']:
            with self.subTest(variant=job['name']):
                cfg=json.loads(Path(job['config']).read_text());torch.manual_seed(0)
                model=paper_model_from_config(self.schema,cfg).cuda()
                self.assertEqual(sum(p.numel() for p in model.parameters()),job['parameters'])
                self.assertEqual(model.backward_ode is model.forward_ode,bool(cfg.get('shared_vector_field')))
                out=model(self.batch)
                mse,cce=reconstruction(feature_errors(out['prediction'],self.batch['values'],self.batch['mask'],self.schema),self.batch['mask'],self.schema)
                loss=mse+cce+.02*model.ode_regularization();loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                at_cutoff=out['prediction'][torch.arange(2,device='cuda'),self.batch['cutoff']-1]
                torch.testing.assert_close(at_cutoff,model.decoder(out['mu']),rtol=0,atol=0)

    def test_shared_regularizer_and_checkpoint_roundtrip(self):
        job=next(j for j in self.manifest['jobs'] if j['name']=='shared_tied')
        cfg=json.loads(Path(job['config']).read_text());model=paper_model_from_config(self.schema,cfg).cuda().eval()
        manual=sum(p.square().sum() for p in model.forward_ode.parameters())
        torch.testing.assert_close(model.ode_regularization(),manual)
        copied=paper_model_from_config(self.schema,cfg).cuda().eval();copied.load_state_dict(model.state_dict())
        self.assertIs(copied.backward_ode,copied.forward_ode)
        torch.testing.assert_close(model(self.batch)['prediction'],copied(self.batch)['prediction'],rtol=0,atol=0)
        cfg['shared_l2_multiplier']=2.;double=paper_model_from_config(self.schema,cfg).cuda();double.load_state_dict(model.state_dict())
        torch.testing.assert_close(double.ode_regularization(),manual*2)


if __name__=='__main__':unittest.main()
