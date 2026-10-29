"""Fair-comparison safeguards independent of long-running training."""
import argparse
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.comparison import (LegacyAdapter, build_model, training_loss, predict,
                            classification_metrics, choose_threshold, summarize, parser, paper_batch)
from src.preprocessing import prepare_data


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.splits, cls.meta = prepare_data()

    def args(self):
        args = parser().parse_args([])
        args.width, args.layers = 8, 1
        args.eval_samples = 4
        return args

    def test_capacity_profiles_match_legacy_parameter_budget(self):
        args = parser().parse_args([])
        for profile, expected in [('base', [126767, 32727, 15877]),
                                  ('matched', [126767, 126766, 126997])]:
            args.capacity = profile
            counts = [sum(p.numel() for p in build_model(n, args).parameters())
                      for n in ('legacy_transformer', 'latent_ode', 'ode_rnn')]
            self.assertEqual(counts, expected)
            if profile == 'matched':
                self.assertTrue(all(abs(n / counts[0] - 1) < .002 for n in counts))
        args.hidden = 41
        with self.assertRaises(ValueError):
            build_model('latent_ode', args)

    def test_matched_profiles_finite_gradients_and_roundtrip(self):
        args = parser().parse_args([])
        args.capacity = 'matched'
        # Tiny mathematical check; full data experiments run exclusively on CUDA.
        sequences = [{**s, 'times': s['times'][:3], 'values': s['values'][:3],
                      'mask': s['mask'][:3]} for s in self.splits['train'].sequences[:2]]
        for name in ('latent_ode', 'ode_rnn'):
            for loss_name in ('native', 'bce'):
                args.objective = loss_name
                model = build_model(name, args)
                loss = training_loss(model, name, sequences, args)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                clone = build_model(name, args)
                clone.load_state_dict(model.state_dict())
                for a, b in zip(clone.parameters(), model.parameters()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_legacy_endpoint_has_no_batch_max_dependence(self):
        torch.manual_seed(3)
        model = LegacyAdapter().eval()
        seqs = self.splits['train'].sequences[:2]
        with torch.no_grad():
            alone = model([seqs[0]], samples=1, deterministic=True)
            together = model(seqs, samples=1, deterministic=True)
        torch.testing.assert_close(alone['probability'][:, 0], together['probability'][:, 0], rtol=3e-4, atol=3e-4)
        torch.testing.assert_close(alone['prediction'][:, 0], together['prediction'][:, 0], rtol=3e-4, atol=3e-4)

    def test_legacy_scaled_time_solver_matches_individual_ivps(self):
        from src.model import solve
        torch.manual_seed(1)
        model = LegacyAdapter().eval()
        z = torch.randn(2, 32)
        ends = torch.tensor([.2, .8])
        actual = model.evolve(z, ends)
        for i in range(2):
            augmented = torch.cat([z[i:i+1], torch.zeros(1, 5)], -1)
            expected = solve(model.ode.ode_func, augmented, torch.tensor([0., float(ends[i])]))[-1, :, :32]
            torch.testing.assert_close(actual[i:i+1], expected, rtol=5e-4, atol=5e-4)

    def test_all_models_both_losses_backpropagate(self):
        args = self.args()
        sequences = self.splits['train'].sequences[:2]
        for name in ('legacy_transformer', 'latent_ode', 'ode_rnn'):
            for loss_name in ('native', 'bce'):
                args.objective = loss_name
                model = build_model(name, args)
                loss = training_loss(model, name, sequences, args)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                trained = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(trained)
                self.assertTrue(all(torch.isfinite(g).all() for g in trained))

    def test_paper_predictions_have_fixed_boundaries_across_batch_composition(self):
        args = self.args()
        seqs = self.splits['train'].sequences[:2]
        for name in ('latent_ode', 'ode_rnn'):
            model = build_model(name, args).eval()
            alone = paper_batch(seqs[:1], 'cpu')
            together = paper_batch(seqs, 'cpu')
            self.assertEqual(float(alone[1][0]), 0.)
            self.assertEqual(float(alone[1][-1]), 1.)
            with torch.no_grad():
                a = model(*alone, deterministic=True)['logits'].sigmoid()[:, 0]
                b = model(*together, deterministic=True)['logits'].sigmoid()[:, 0]
            torch.testing.assert_close(a, b, atol=3e-4, rtol=3e-4)

    def test_evaluation_replay_and_rng_preservation(self):
        args = self.args()
        from src.preprocessing import SequenceDataset
        dataset = SequenceDataset(self.splits['val'].sequences[:2])
        for name in ('legacy_transformer', 'latent_ode', 'ode_rnn'):
            model = build_model(name, args)
            before = torch.random.get_rng_state().clone()
            a = predict(model, name, dataset, args)
            torch.testing.assert_close(torch.random.get_rng_state(), before)
            b = predict(model, name, dataset, args)
            self.assertEqual(a, b)
            reverse = predict(model, name, SequenceDataset(dataset.sequences[::-1]), args)
            self.assertEqual(a, reverse[::-1])

    def test_threshold_uses_validation_only_and_metrics_match(self):
        records = [{'label': 0., 'probability': .1}, {'label': 1., 'probability': .3},
                   {'label': 0., 'probability': .2}, {'label': 1., 'probability': .4}]
        threshold = choose_threshold(records)
        self.assertAlmostEqual(threshold, .3)
        metrics = classification_metrics(records, threshold)
        self.assertEqual(metrics['auroc'], 1.)
        self.assertEqual(metrics['ap'], 1.)
        self.assertEqual(metrics['f1'], 1.)
        self.assertAlmostEqual(metrics['brier'], (.01 + .49 + .04 + .36) / 4)

    def test_sample_standard_deviation_and_paired_differences(self):
        rows = []
        for name, values in [('legacy_transformer', [.4, .6]), ('latent_ode', [.6, .9])]:
            for seed, value in enumerate(values):
                rows.append({'model': name, 'seed': seed, 'parameters': 1,
                             'test': {k: value for k in ('auroc', 'ap', 'brier', 'accuracy', 'f1', 'precision', 'recall')}})
        with tempfile.TemporaryDirectory() as directory:
            report = summarize(rows, directory)
        actual = report['summary']['legacy_transformer']['test']['auroc']
        self.assertAlmostEqual(actual['mean'], .5)
        self.assertAlmostEqual(actual['std'], np.std([.4, .6], ddof=1))
        paired = report['paired_differences']['latent_ode minus legacy_transformer']['auroc']
        self.assertAlmostEqual(paired['mean'], .25)
        self.assertAlmostEqual(paired['std'], np.std([.2, .3], ddof=1))


if __name__ == '__main__':
    unittest.main()
