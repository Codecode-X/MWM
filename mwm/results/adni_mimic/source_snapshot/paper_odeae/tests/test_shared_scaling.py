"""CUDA checks for shared_matched component and latent expansion."""
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from paper_odeae.data import collate
from paper_odeae.models import paper_model_from_config, feature_errors, reconstruction
from paper_odeae.shared_scaling_study import DEFAULT, definitions, parameter_count
from paper_odeae.train import device_batch


class SharedScalingChecks(unittest.TestCase):
    def test_four_capacity_settings_on_cuda(self):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU fallback')
        torch.set_num_threads(1)
        manifest = json.loads((DEFAULT / 'manifest.json').read_text())
        reference = Path(manifest['baseline_reference'])
        base = json.loads((reference / 'config.json').read_text())
        schema = json.loads((reference / 'schema.json').read_text())
        rng = np.random.default_rng(17)
        values = rng.random((3, len(schema))).astype(np.float32)
        for j, field in enumerate(schema):
            if field['kind'] == 'categorical':
                values[:, j] = np.arange(3) % field['classes']
        batch = device_batch(collate([({'subject_id': 'synthetic',
                'ages': np.array([70., 70.5, 71.], np.float32),
                'values': values, 'mask': np.ones_like(values)}, 2)]))
        for job, (_, changes) in zip(manifest['jobs'], definitions(base)):
            with self.subTest(variant=job['name']):
                cfg = json.loads(Path(job['config']).read_text())
                self.assertEqual(job['changes'], changes)
                key, value = next(iter(changes.items()))
                error = abs(parameter_count(cfg) - manifest['target_parameters'])
                for other in [value - 1, value + 1]:
                    self.assertLessEqual(error, abs(parameter_count({**cfg, key: other}) - manifest['target_parameters']))
                for fixed in ['lr', 'l2_weight', 'batch_size', 'seed', 'input_dropout',
                              'encoder_direction', 'encoder_initialization', 'shared_vector_field']:
                    self.assertEqual(cfg[fixed], base[fixed])
                model = paper_model_from_config(schema, cfg).cuda()
                self.assertIs(model.backward_ode, model.forward_ode)
                self.assertEqual(sum(p.numel() for p in model.parameters()), job['parameters'])
                self.assertLess(abs(job['parameters'] / 90180 - 1), .005)
                out = model(batch)
                mse, cce = reconstruction(feature_errors(out['prediction'], batch['values'], batch['mask'], schema), batch['mask'], schema)
                (mse + cce + .02 * model.ode_regularization()).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                torch.testing.assert_close(out['prediction'][:, 1], model.decoder(out['mu']), rtol=0, atol=0)
                self.assertEqual(out['mu'].shape[-1], cfg['latent'])
                copied = paper_model_from_config(schema, cfg).cuda()
                copied.load_state_dict(model.state_dict())
                self.assertIs(copied.backward_ode, copied.forward_ode)
                torch.testing.assert_close(copied(batch)['prediction'], out['prediction'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
