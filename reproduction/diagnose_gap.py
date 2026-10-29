"""Post-hoc diagnostics only; never alter benchmark checkpoints or official metrics."""
import argparse
import json
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from src.comparison import build_model, paper_batch, initialize_worker
from src.preprocessing import prepare_data


def diagnose_worker(training_seed):
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    results = {'sampling': [], 'temporal_gradients_bce': [],
               'runtime': {'seed': training_seed, 'tensor_device': str(device),
                           'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}}
    for protocol in ('native', 'bce'):
        root = Path('results/comparison_' + protocol)
        config = json.loads((root / 'protocol.json').read_text())
        splits, _ = prepare_data(config['data'], config['split_seed'])
        args = argparse.Namespace(**config)
        for seed in [training_seed]:
            path = root / 'latent_ode' / f'seed_{seed}'
            checkpoint = torch.load(path / 'best.pt', map_location='cpu', weights_only=False)
            model = build_model('latent_ode', args).to(device).eval()
            model.load_state_dict(checkpoint['state_dict'])
            stored = json.loads((path / 'predictions.json').read_text())
            for split in ('val', 'test'):
                truth, probs = [], {str(k): [] for k in (32, 128, 512, 4096)}
                probs['posterior_mean'] = []
                mc_errors, scales = [], []
                with torch.no_grad():
                    for sequence in splits[split].sequences:
                        values, times, mask = paper_batch([sequence], device)
                        h, _ = model.encoder(values, times, mask, backwards=True)
                        mu, raw = model.posterior(h).chunk(2, -1)
                        std = torch.nn.functional.softplus(raw) + 1e-5
                        truth.append(sequence['label'])
                        scales.append(float(std.mean()))
                        probs['posterior_mean'].append(float(model.classifier(mu).sigmoid()))
                        for count in (32, 128, 512, 4096):
                            with torch.random.fork_rng(devices=[0]):
                                torch.manual_seed(args.eval_seed + sequence['patient_id'])
                                z = mu[None] + std[None] * torch.randn(count, 1, args.latent, device=device)
                                p = model.classifier(z).sigmoid().flatten()
                            probs[str(count)].append(float(p.mean()))
                            if count == 4096:
                                mc_errors.append(float(p.std() / np.sqrt(32)))
                saved = stored['validation' if split == 'val' else 'test']
                # Historical CPU RNG draws differ from CUDA RNG draws. Keep
                # this GPU diagnosis separate; never overwrite official scores.
                if checkpoint['config']['device'].startswith('cuda'):
                    assert np.max(np.abs(np.array(probs['32']) - [r['probability'] for r in saved])) < 1e-6
                results['sampling'].append({'protocol': protocol, 'seed': seed, 'split': split,
                    'auroc': {k: float(roc_auc_score(truth, p)) for k, p in probs.items()},
                    'posterior_std_mean': np.mean(scales),
                    'expected_mc32_probability_standard_error': np.mean(mc_errors),
                    'between_patient_probability_std_mc4096': float(np.std(probs['4096'])),
                    'probabilities': probs})
        if protocol != 'bce':
            continue
        for name in config['models']:
            for seed in [training_seed]:
                checkpoint = torch.load(root / name / f'seed_{seed}' / 'best.pt', map_location='cpu', weights_only=False)
                model = build_model(name, args).to(device).eval()
                model.load_state_dict(checkpoint['state_dict'])
                fractions = []
                for sequence in splits['test'].sequences:
                    base = sequence['values'].clone().requires_grad_(True)
                    seq = {**sequence, 'values': base}
                    if name == 'legacy_transformer':
                        probability = model([seq], samples=1, deterministic=True)['probability'].sum()
                        logit = torch.logit(probability.clamp(1e-6, 1-1e-6))
                    else:
                        values, times, mask = paper_batch([seq], device)
                        if name == 'latent_ode':
                            h, _ = model.encoder(values, times, mask, backwards=True)
                            mu = model.posterior(h).chunk(2, -1)[0]
                            logit = model.classifier(mu).sum()
                        else:
                            logit = model(values, times, mask)['logits'].sum()
                    gradient = torch.autograd.grad(logit, base)[0].abs().sum(-1)
                    groups = torch.tensor_split(gradient, 3)
                    fractions.append([float(g.sum() / gradient.sum().clamp_min(1e-20)) for g in groups])
                results['temporal_gradients_bce'].append({'model': name, 'seed': seed,
                    'path': 'posterior_mean' if name != 'ode_rnn' else 'deterministic',
                    'mean_gradient_fraction_early_middle_late': np.mean(fractions, axis=0).tolist()})
    results['posthoc_logistic_baselines'] = {}
    for portion in ('early_third', 'late_third', 'whole_sequence'):
        features, labels = {}, {}
        for split, dataset in splits.items():
            rows = []
            for sequence in dataset.sequences:
                x = sequence['values'].numpy()
                groups = np.array_split(x, 3)
                selected = groups[0] if portion == 'early_third' else groups[2] if portion == 'late_third' else x
                rows.append(selected.mean(0))
            features[split] = np.array(rows)
            labels[split] = [s['label'] for s in dataset.sequences]
        scaler = StandardScaler().fit(features['train'])
        arrays = {split: torch.tensor(scaler.transform(x), dtype=torch.float64, device=device)
                  for split, x in features.items()}
        y = torch.tensor(labels['train'], dtype=torch.float64, device=device)
        w = torch.zeros(6, dtype=torch.float64, device=device, requires_grad=True)
        b = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
        optimizer = torch.optim.LBFGS([w, b], max_iter=1000, tolerance_grad=1e-7,
                                     tolerance_change=1e-10, line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(arrays['train'] @ w + b, y)
            loss = loss + .5 * w.square().sum() / len(y)
            loss.backward()
            return loss
        optimizer.step(closure)
        results['posthoc_logistic_baselines'][portion] = {
            split: float(roc_auc_score(labels[split], (x @ w + b).sigmoid().detach().cpu().numpy()))
            for split, x in arrays.items()}

    results['parameter_counts'] = {
        name: {key: sum(p.numel() for p in mod.parameters()) for key, mod in build_model(name, args).named_children()}
        for name in args.models}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--output', default='results/comparison_diagnostics_cuda.json')
    args_cli = parser.parse_args()
    if not torch.cuda.is_available() or any(g < 0 or g >= torch.cuda.device_count() for g in args_cli.gpus):
        raise RuntimeError('Four-GPU CUDA diagnostics required; CPU fallback is disabled')
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        raise RuntimeError('Unset CUDA_VISIBLE_DEVICES when specifying physical GPU IDs')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    for gpu in args_cli.gpus:
        queue.put(gpu)
    results = {'sampling': [], 'temporal_gradients_bce': [], 'workers': []}
    with ProcessPoolExecutor(max_workers=len(args_cli.gpus), mp_context=context,
                             initializer=initialize_worker, initargs=(queue,)) as pool:
        futures = [pool.submit(diagnose_worker, seed) for seed in args_cli.seeds]
        for future in as_completed(futures):
            part = future.result()
            results['sampling'].extend(part['sampling'])
            results['temporal_gradients_bce'].extend(part['temporal_gradients_bce'])
            results['workers'].append(part['runtime'])
            results['parameter_counts'] = part['parameter_counts']
            results['posthoc_logistic_baselines'] = part['posthoc_logistic_baselines']
    results['sampling'].sort(key=lambda r: (r['protocol'], r['seed'], r['split']))
    destination = Path(args_cli.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2))
    for protocol in ('native', 'bce'):
        for split in ('val', 'test'):
            rows = [r for r in results['sampling'] if r['protocol'] == protocol and r['split'] == split]
            print(protocol, split, {key: round(float(np.mean([r['auroc'][key] for r in rows])), 4) for key in rows[0]['auroc']})
            print('mean MC32 SE / between-patient SD4096',
                np.mean([r['expected_mc32_probability_standard_error'] for r in rows]),
                np.mean([r['between_patient_probability_std_mc4096'] for r in rows]))
    for name in ('legacy_transformer', 'latent_ode', 'ode_rnn'):
        rows = [r for r in results['temporal_gradients_bce'] if r['model'] == name]
        print(name, 'temporal gradient fraction:', np.mean([r['mean_gradient_fraction_early_middle_late'] for r in rows], axis=0))


if __name__ == '__main__':
    main()
