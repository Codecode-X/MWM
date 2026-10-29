"""Numerical, statistical, causality, masking and data-leakage regressions."""
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.model import (solve, GRUUpdate, ODERNNEncoder, LatentODE, ODERNN,
                       gaussian_log_likelihood, objective)
from src.preprocessing import prepare_data, collate_fn, make_context


torch.set_num_threads(1)


class Constant(nn.Module):
    def forward(self, t, x):
        return torch.ones_like(x)


class NumericalTests(unittest.TestCase):
    def test_constant_forward_backward_and_singleton(self):
        initial = torch.tensor([[2.]], requires_grad=True)
        for times in (torch.tensor([0., .13, .8]), torch.tensor([.8, .13, 0.])):
            actual = solve(Constant(), initial, times)
            torch.testing.assert_close(actual[:, 0, 0], 2 + times - times[0], atol=2e-5, rtol=1e-5)
            actual[-1].sum().backward()
            self.assertAlmostEqual(initial.grad.item(), 1., places=4)
            initial.grad = None
        torch.testing.assert_close(solve(Constant(), initial, torch.tensor([5.]))[0], initial)

    def test_exponential_and_parameter_gradient(self):
        class Linear(nn.Module):
            def __init__(self):
                super().__init__()
                self.rate = nn.Parameter(torch.tensor(.3, dtype=torch.float64))
            def forward(self, t, x):
                return self.rate * x
        f = Linear()
        z = torch.ones(1, 1, dtype=torch.float64)
        times = torch.tensor([0., .2, 1.], dtype=torch.float64)
        y = solve(f, z, times, rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(y[-1], torch.exp(f.rate).reshape(1, 1), atol=1e-7, rtol=1e-6)
        y[-1].sum().backward()
        self.assertAlmostEqual(f.rate.grad.item(), math.exp(.3), places=5)

    def test_invalid_solver_times(self):
        for t in ([0., 0.], [0., 1., .5], [], [0., float('nan')]):
            with self.assertRaises(ValueError):
                solve(Constant(), torch.zeros(1, 2), torch.tensor(t))

    def test_gru_exact_equations_and_missing_skip(self):
        cell = GRUUpdate(2, 3)
        for p in cell.parameters():
            nn.init.zeros_(p)
        h = torch.ones(2, 3)
        x = torch.tensor([[2., 9.], [float('nan'), 1.]])
        mask = torch.tensor([[1., 0.], [0., 0.]])
        actual = cell(h, x, mask)
        torch.testing.assert_close(actual[0], h[0] * .5)
        torch.testing.assert_close(actual[1], h[1])

    def test_encoder_direction(self):
        enc = ODERNNEncoder(1, hidden=1)
        enc.ode = Constant()
        # Set update to identity on any observation, isolating time direction.
        class Identity(nn.Module):
            def forward(self, h, x, mask):
                return h
        enc.cell = Identity()
        x, mask = torch.zeros(1, 3, 1), torch.ones(1, 3, 1)
        times = torch.tensor([.1, .3, .9])
        back, _ = enc(x, times, mask, backwards=True)
        forward, _ = enc(x, times, mask, backwards=False)
        torch.testing.assert_close(back, torch.tensor([[-.8]]), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(forward, -back, atol=1e-5, rtol=1e-5)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(2, 4, 2)
        self.t = torch.tensor([.0, .17, .6, 1.])
        self.mask = torch.ones_like(self.x)
        self.mask[0, 1, 1] = 0

    def model(self, cls):
        return cls(features=2, latent=3, hidden=5, width=8, layers=1)

    def test_shapes_and_gradients_both_models(self):
        for cls in (LatentODE, ODERNN):
            model = self.model(cls)
            out = model(self.x, self.t, self.mask)
            self.assertEqual(out['prediction'].shape, (3 if cls is LatentODE else 1, 2, 4, 2))
            loss = objective(out, self.x, self.mask, torch.tensor([0., 1.]), 100.)
            loss['loss'].backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            if cls is ODERNN:
                self.assertEqual(float(loss['kl']), 0.)

    def test_no_target_leakage_causal_ode_rnn(self):
        model = self.model(ODERNN)
        first = model(self.x, self.t, self.mask)['prediction']
        changed = self.x.clone()
        changed[:, 2:] += 100
        second = model(changed, self.t, self.mask)['prediction']
        torch.testing.assert_close(first[:, :, :3], second[:, :, :3])
        self.assertFalse(torch.allclose(first[:, :, 3], second[:, :, 3]))

    def test_masked_nan_never_changes_output(self):
        contaminated = self.x.clone()
        contaminated[~self.mask.bool()] = float('nan')
        for cls in (LatentODE, ODERNN):
            model = self.model(cls)
            a = model(self.x, self.t, self.mask, deterministic=True)
            b = model(contaminated, self.t, self.mask, deterministic=True)
            torch.testing.assert_close(a['prediction'], b['prediction'])
            self.assertTrue(torch.isfinite(objective(b, contaminated, self.mask)['loss']))

    def test_empty_union_timestamp_no_rnn_jump(self):
        for cls in (LatentODE, ODERNN):
            model = self.model(cls)
            a = model(self.x, self.t, self.mask, deterministic=True)
            expanded_t = torch.tensor([0., .1, .17, .6, 1.])
            indices = torch.tensor([0, 2, 3, 4])
            x, mask = torch.zeros(2, 5, 2), torch.zeros(2, 5, 2)
            x[:, indices], mask[:, indices] = self.x, self.mask
            b = model(x, expanded_t, mask, deterministic=True)
            torch.testing.assert_close(a['prediction'], b['prediction'][:, :, indices], atol=3e-4, rtol=3e-4)

    def test_batch_independence(self):
        for cls in (LatentODE, ODERNN):
            model = self.model(cls)
            a = model(self.x, self.t, self.mask, deterministic=True)['prediction']
            b = model(self.x[:1], self.t, self.mask[:1], deterministic=True)['prediction']
            torch.testing.assert_close(a[:, :1], b, atol=3e-4, rtol=3e-4)

    def test_generative_anchor_and_query_order(self):
        model = self.model(LatentODE)
        model.ode = Constant()
        query = torch.tensor([1.3, 1., 1.3])
        out = model(self.x, self.t, self.mask, query_times=query,
                    task='extrapolation', deterministic=True)
        torch.testing.assert_close(out['trajectory'][:, :, 1], out['z0'])
        torch.testing.assert_close(out['trajectory'][:, :, 0], out['z0'] + .3, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(out['prediction'][:, :, 0], out['prediction'][:, :, 2])

    def test_decode_before_and_after_posterior_anchor(self):
        model = self.model(LatentODE)
        model.ode = Constant()
        query = torch.tensor([1.2, -.2, .5, 1., -.2])
        out = model(self.x, self.t, self.mask, query_times=query,
                    task='extrapolation', deterministic=True)
        expected = out['z0'][:, :, None] + (query - 1.)[None, None, :, None]
        torch.testing.assert_close(out['trajectory'], expected, atol=2e-5, rtol=1e-5)

    def test_elapsed_time_changes_recurrent_predictions(self):
        model = self.model(ODERNN)
        model.ode = Constant()
        first = model(self.x, self.t, self.mask)['prediction']
        second = model(self.x, self.t * 2, self.mask)['prediction']
        torch.testing.assert_close(first[:, :, 0], second[:, :, 0])
        self.assertFalse(torch.allclose(first[:, :, 1:], second[:, :, 1:]))

    def test_invalid_inputs_rejected(self):
        model = self.model(LatentODE)
        for times in (torch.tensor([0., 0., .6, 1.]), self.t[None]):
            with self.assertRaises(ValueError):
                model(self.x, times, self.mask)
        with self.assertRaises(ValueError):
            model(self.x, self.t, self.mask * .5)
        with self.assertRaises(ValueError):
            model(self.x * float('nan'), self.t, self.mask)
        with self.assertRaises(ValueError):
            model(self.x, self.t, self.mask, samples=0)

    def test_forecast_rollout_no_future_values(self):
        model = self.model(ODERNN)
        a = model.forecast(self.x[:, :2], self.t[:2], self.mask[:, :2], self.t[2:])
        self.assertEqual(a.shape, (1, 2, 2, 2))
        self.assertTrue(torch.isfinite(a).all())
        with self.assertRaises(ValueError):
            model.forecast(self.x, self.t, self.mask, self.t)

    def test_reparameterization_statistics(self):
        model = self.model(LatentODE)
        out = model(self.x, self.t, self.mask, samples=20000)
        torch.testing.assert_close(out['z0'].mean(0), out['mu'], atol=.025, rtol=.05)
        torch.testing.assert_close(out['z0'].std(0), out['std'], atol=.025, rtol=.05)

    def test_checkpoint_roundtrip(self):
        model = self.model(LatentODE)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(model.state_dict(), path)
            restored = self.model(LatentODE)
            restored.load_state_dict(torch.load(path, weights_only=True))
            a = model(self.x, self.t, self.mask, deterministic=True)['prediction']
            b = restored(self.x, self.t, self.mask, deterministic=True)['prediction']
            torch.testing.assert_close(a, b)

    def test_single_observation(self):
        for cls in (LatentODE, ODERNN):
            out = self.model(cls)(self.x[:, :1], self.t[:1], self.mask[:, :1])
            self.assertEqual(out['prediction'].shape[-2], 1)

    def test_overfit_small_sequence(self):
        for cls in (LatentODE, ODERNN):
            torch.manual_seed(2)
            model = self.model(cls)
            x = torch.ones(2, 4, 2) * .3
            opt = torch.optim.Adam(model.parameters(), lr=.03)
            def loss():
                return objective(model(x, self.t, self.mask, deterministic=True), x, self.mask)['loss']
            before = float(loss())
            for _ in range(35):
                opt.zero_grad()
                current = loss()
                current.backward()
                opt.step()
            self.assertLess(float(loss()), before * .5)


class LossTests(unittest.TestCase):
    def test_likelihood_and_kl_against_torch_distributions(self):
        torch.manual_seed(4)
        pred = torch.randn(3, 2, 4, 2)
        target = torch.randn(2, 4, 2)
        mask = torch.randint(0, 2, target.shape).float()
        actual = gaussian_log_likelihood(pred, target, mask)
        expected = (torch.distributions.Normal(pred, .1).log_prob(target) * mask).sum((-1, -2))
        torch.testing.assert_close(actual, expected)
        mu, std = torch.randn(2, 3), torch.rand(2, 3) + .1
        out = {'prediction': pred, 'mu': mu, 'std': std}
        actual = objective(out, target, mask)
        expected_kl = torch.distributions.kl_divergence(torch.distributions.Normal(mu, std), torch.distributions.Normal(0., 1.)).sum(-1).mean()
        torch.testing.assert_close(actual['kl'], expected_kl)
        torch.testing.assert_close(actual['loss'], -expected.mean() + expected_kl)

    def test_kl_zero_at_prior_and_classification_uses_labels(self):
        target = torch.zeros(2, 1, 1)
        out = {'prediction': target.unsqueeze(0), 'mu': torch.zeros(2, 3),
               'std': torch.ones(2, 3), 'logits': torch.tensor([[4., -4.]])}
        right = objective(out, target, torch.ones_like(target), torch.tensor([1., 0.]), 100.)
        wrong = objective(out, target, torch.ones_like(target), torch.tensor([0., 1.]), 100.)
        self.assertEqual(float(right['kl']), 0.)
        self.assertGreater(float(wrong['loss'] - right['loss']), 399.)

    def test_mask_denominator_counts_features(self):
        pred = torch.ones(1, 2, 3, 4)
        mask = torch.zeros(2, 3, 4)
        mask[0, 0, 0] = mask[1, 1, :2] = 1
        actual = objective({'prediction': pred}, torch.zeros_like(mask), mask)
        self.assertEqual(float(actual['mse']), 1.)


class DataTests(unittest.TestCase):
    def test_split_scaling_and_reproducibility(self):
        a, meta = prepare_data()
        b, second = prepare_data()
        self.assertEqual(meta, second)
        ids = {name: set(meta['patient_ids'][name]) for name in a}
        for first, other in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            self.assertFalse(ids[first] & ids[other])
        self.assertEqual(sum(map(len, ids.values())), 100)
        df = pd.read_csv('data/raw/synthetic_icu_data.csv')
        from src.preprocessing import FEATURES
        train = df[df.patient_id.isin(ids['train'])][FEATURES]
        np.testing.assert_allclose(meta['minimum'], train.min().values)
        np.testing.assert_allclose(meta['scale'], train.max().values - train.min().values)
        torch.testing.assert_close(a['train'][0]['values'], b['train'][0]['values'])

    def test_union_mapping_and_no_mask_on_absent_times(self):
        data, _ = prepare_data()
        sequences = data['train'].sequences[:3]
        batch = collate_fn(sequences)
        self.assertEqual(int(batch['mask'].sum()), sum(int(s['mask'].sum()) for s in sequences))
        for index, seq in enumerate(sequences):
            where = torch.searchsorted(batch['times'], seq['times'])
            torch.testing.assert_close(batch['values'][index, where], seq['values'])
        self.assertTrue((batch['times'][1:] > batch['times'][:-1]).all())

    def test_extrapolation_alignment_preserves_intervals(self):
        data, _ = prepare_data()
        seqs = data['train'].sequences[:4]
        batch = collate_fn(seqs, task='extrapolation')
        for index, seq in enumerate(seqs):
            present = batch['mask'][index].bool().any(-1)
            aligned = batch['times'][present]
            torch.testing.assert_close(aligned[1:] - aligned[:-1], seq['times'][1:] - seq['times'][:-1])
            self.assertEqual(int((aligned <= 0).sum()), len(seq['times']) // 2)

    def test_context_does_not_expose_targets(self):
        data, _ = prepare_data()
        batch = collate_fn(data['train'].sequences[:3])
        context = make_context(batch, 'interpolation', generator=torch.Generator().manual_seed(1))
        self.assertTrue((context['values'][~context['mask'].bool()] == 0).all())
        self.assertLess(context['mask'].sum(), batch['mask'].sum())
        batch = collate_fn(data['train'].sequences[:3], task='extrapolation')
        future = make_context(batch, 'extrapolation')
        self.assertTrue((future['times'] <= 0).all())
        self.assertTrue((future['query_times'] > 0).all())
        changed = {**batch, 'values': batch['values'].clone()}
        changed['values'][:, batch['times'] > 0] += 1000
        second = make_context(changed, 'extrapolation')
        torch.testing.assert_close(future['values'], second['values'])


if __name__ == '__main__':
    unittest.main()
