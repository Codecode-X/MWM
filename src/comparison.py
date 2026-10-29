"""Paired multi-training-seed patient classification benchmark.

No calibration; checkpoint by validation AUROC, threshold by validation F1.
The historical architecture has explicit endpoint/target-index corrections.
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import hashlib
import json
import math
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)
from legacy.model import LatentODEModel
from .model import LatentODE, ODERNN, objective, solve
from .preprocessing import collate_fn, prepare_data
from .train import seed_all
from .tracking import SwanLabTracker

NAMES = ('legacy_transformer', 'latent_ode', 'ode_rnn')


class LegacyAdapter(LatentODEModel):
    """Original modules; use each patient's endpoint, not the batch maximum.

    Independent IVPs are vectorized by ds/dtau=end_time*f(s), tau in [0,1].
    The archived model and checkpoints are never changed.
    """
    def __init__(self):
        super().__init__()

    def encode(self, sequences, device):
        length = max(len(s['times']) for s in sequences)
        values = torch.zeros(len(sequences), length, 6, device=device)
        mask = torch.zeros(len(sequences), length, device=device)
        ends = []
        for i, sequence in enumerate(sequences):
            if not sequence['mask'].bool().all():
                raise ValueError('legacy architecture does not support feature-level missingness')
            n = len(sequence['times'])
            values[i, :n] = sequence['values'].to(device)
            mask[i, :n] = 1
            ends.append(sequence['times'][-1])
        mu, logvar = self.encoder(values, mask)
        return mu, torch.exp(.5 * logvar), torch.stack(ends).to(device) + 12. / 72.

    def evolve(self, z, ends):
        augmented = torch.cat([z, z.new_zeros(*z.shape[:-1], self.ode.augment_dim)], -1)
        def scaled(t, h):
            return self.ode.ode_func(t, h) * ends[..., None]
        return solve(scaled, augmented, z.new_tensor([0., 1.]))[-1][..., :self.latent_dim]

    def forward(self, sequences, samples=3, deterministic=False):
        mu, std, ends = self.encode(sequences, next(self.parameters()).device)
        z = mu[None].expand(samples, -1, -1)
        if not deterministic:
            z = z + std[None] * torch.randn_like(z)
        final = self.evolve(z, ends)
        flat = final.reshape(-1, self.latent_dim)
        alpha, beta = self.classifier(flat)
        alpha, beta = alpha.reshape(samples, -1), beta.reshape(samples, -1)
        return {'prediction': self.decoder(final), 'probability': alpha / (alpha + beta),
                'mu': mu, 'std': std, 'alpha': alpha, 'beta': beta}


def build_model(name, args):
    if name == 'legacy_transformer':
        return LegacyAdapter()
    factory = LatentODE if name == 'latent_ode' else ODERNN
    latent, hidden = args.latent, args.hidden
    if getattr(args, 'capacity', 'base') == 'matched':
        if (args.latent, args.hidden, args.width, args.layers) != (20, 40, 50, 3):
            raise ValueError('matched profile requires default base dimensions')
        if name == 'latent_ode':
            hidden = 143
        else:
            latent = 140
    return factory(latent=latent, hidden=hidden, width=args.width, layers=args.layers)


def paper_batch(sequences, device):
    """Common acquisition boundaries prevent batch-dependent prior/reference times."""
    batch = collate_fn(sequences)
    times = torch.unique(torch.cat([batch['times'], torch.tensor([0., 1.])]), sorted=True)
    values = torch.zeros(len(sequences), len(times), 6)
    mask = torch.zeros_like(values)
    indices = torch.searchsorted(times, batch['times'])
    values[:, indices], mask[:, indices] = batch['values'], batch['mask']
    return values.to(device), times.to(device), mask.to(device)


def training_loss(model, name, sequences, args):
    device = next(model.parameters()).device
    labels = torch.tensor([s['label'] for s in sequences], device=device)
    if name == 'legacy_transformer':
        out = model(sequences, samples=args.train_samples)
        bce = F.binary_cross_entropy(out['probability'], labels[None].expand_as(out['probability']))
        if args.objective == 'bce':
            return bce
        targets = torch.stack([s['values'][-1] for s in sequences]).to(device)
        # Original reconstruction convention sums features, averages patients.
        recon = (out['prediction'] - targets[None]).square().sum(-1).mean()
        mu, std = out['mu'], out['std']
        kl = .5 * (mu.square() + std.square() - 1 - 2 * std.log()).sum(-1).mean()
        evidence = -(out['alpha'] + out['beta']).log().mean()
        return .3 * (recon + .01 * kl + .1 * evidence) + 2 * bce
    values, times, mask = paper_batch(sequences, device)
    if name == 'latent_ode':
        out = model(values, times, mask, samples=args.train_samples)
    else:
        out = model(values, times, mask, teacher_forcing=.5)
    if args.objective == 'bce':
        return F.binary_cross_entropy_with_logits(out['logits'], labels[None].expand_as(out['logits']))
    return objective(out, values, mask, labels, classification_weight=100., variance=.01)['loss']


@torch.no_grad()
def predict(model, name, dataset, args):
    """One patient per evaluation IVP; repeatable local MC draws, no RNG leakage."""
    model.eval()
    device = next(model.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    records = []
    for sequence in dataset.sequences:
        patient_seed = args.eval_seed + int(sequence['patient_id'])
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(patient_seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed(patient_seed)
            deterministic = args.eval_mode == 'mean'
            count = 1 if deterministic else args.eval_samples
            if name == 'legacy_transformer':
                out = model([sequence], samples=count, deterministic=deterministic)
                probability = out['probability'].mean().item()
            else:
                values, times, mask = paper_batch([sequence], device)
                if name == 'latent_ode':
                    # The paper classifier consumes z0. Avoid unnecessary trajectory
                    # decoding when only computing patient classification metrics.
                    h, _ = model.encoder(values, times, mask, backwards=True)
                    mu, raw = model.posterior(h).chunk(2, -1)
                    std = F.softplus(raw) + 1e-5
                    z = mu[None].expand(count, -1, -1)
                    if not deterministic:
                        z = z + std[None] * torch.randn_like(z)
                    probability = model.classifier(z).sigmoid().mean().item()
                else:
                    probability = model(values, times, mask)['logits'].sigmoid().item()
        if not math.isfinite(probability):
            raise FloatingPointError('nonfinite predicted probability')
        records.append({'patient_id': sequence['patient_id'], 'label': sequence['label'],
                        'probability': probability})
    return records


def classification_metrics(records, threshold=.5):
    y = np.array([r['label'] for r in records])
    p = np.array([r['probability'] for r in records])
    if len(np.unique(y)) != 2:
        raise ValueError('each scored split must contain both classes')
    prediction = p >= threshold
    return {'auroc': float(roc_auc_score(y, p)), 'ap': float(average_precision_score(y, p)),
            'brier': float(brier_score_loss(y, p)),
            'accuracy': float(accuracy_score(y, prediction)),
            'f1': float(f1_score(y, prediction, zero_division=0)),
            'precision': float(precision_score(y, prediction, zero_division=0)),
            'recall': float(recall_score(y, prediction, zero_division=0))}


def choose_threshold(validation):
    # Includes all-positive/all-negative candidates; ties prefer proximity to .5.
    probabilities = np.array([r['probability'] for r in validation])
    candidates = sorted(set([0., .5, float(np.nextafter(probabilities.max(), np.inf)), *probabilities.tolist()]))
    labels = [r['label'] for r in validation]
    return max(candidates, key=lambda t: (f1_score(labels, probabilities >= t, zero_division=0), -abs(t - .5), -t))


def configuration(args):
    # Ignore execution-only settings when validating reuse of a completed run.
    config = {k: v for k, v in vars(args).items() if k not in ('resume', 'reevaluate', 'jobs', 'gpus')}
    if config.get('capacity') == 'base':
        config.pop('capacity')  # Preserve historical checkpoint protocol compatibility.
    return {**config, 'paper_time_boundaries': [0., 1.]}


def run_one(name, seed, splits, metadata, args):
    directory = Path(args.output) / name / f'seed_{seed}'
    config = configuration(args)
    config.update(model=name, training_seed=seed)
    if (directory / 'best.pt').exists() and not args.resume and not args.reevaluate:
        raise FileExistsError(f'{directory} already contains a checkpoint; use --resume or a new output directory')
    if (directory / 'best.pt').exists() and (args.resume or args.reevaluate):
        checkpoint = torch.load(directory / 'best.pt', map_location=args.device, weights_only=False)
        if checkpoint['config'] != config or checkpoint['metadata'] != metadata:
            raise ValueError(f'{directory}: configuration/data mismatch; refusing reuse')
        if args.resume and not args.reevaluate and (directory / 'metrics.json').exists():
            result = json.loads((directory / 'metrics.json').read_text())
            history = json.loads((directory / 'history.json').read_text())
            if len(history) != args.epochs:
                raise ValueError('incomplete run: restart in a new output directory')
            return result
        if args.resume and not args.reevaluate:
            raise ValueError('incomplete run: restart in a new output directory')
    elif args.reevaluate:
        raise FileNotFoundError(directory / 'best.pt')
    directory.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    model = build_model(name, args).to(args.device)
    with SwanLabTracker(directory, config, 'classification_comparison', model, seed) as tracker:
        if not args.reevaluate:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            loader = DataLoader(splits['train'], batch_size=args.batch_size, shuffle=True,
                                generator=torch.Generator().manual_seed(seed), collate_fn=list)
            best, history = None, []
            for epoch in range(1, args.epochs + 1):
                seed_all(seed * 100003 + epoch)
                model.train()
                total = count = 0
                for sequences in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = training_loss(model, name, sequences, args)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite training loss')
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, error_if_nonfinite=True)
                    optimizer.step()
                    tracker.batch(loss, optimizer.param_groups[0]['lr'], gradient_norm=grad_norm)
                    total += float(loss.detach()) * len(sequences)
                    count += len(sequences)
                validation = predict(model, name, splits['val'], args)
                metrics = classification_metrics(validation)
                row = {'epoch': epoch, 'train_loss': total / count, 'validation': metrics}
                history.append(row)
                # AUROC first, AP breaks ties; otherwise retain the earliest epoch.
                score = (metrics['auroc'], metrics['ap'])
                if best is None or score > best:
                    best = score
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'config': config,
                                'metadata': metadata, 'validation': metrics}, directory / 'best.pt')
                (directory / 'history.json').write_text(json.dumps(history, indent=2))
                tracker.epoch(epoch, {'loss': total/count}, metrics, optimizer.param_groups[0]['lr'], selection={'best_auroc': best[0]})
                print(json.dumps({'model': name, 'seed': seed, **row}), flush=True)
        checkpoint = torch.load(directory / 'best.pt', map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        validation = predict(model, name, splits['val'], args)
        if classification_metrics(validation) != checkpoint['validation']:
            raise AssertionError('checkpoint validation replay mismatch')
        threshold = choose_threshold(validation)
        # Test probabilities are produced only after checkpoint and threshold selection.
        test = predict(model, name, splits['test'], args)
        result = {'model': name, 'seed': seed, 'split_seed': args.split_seed,
                  'objective': args.objective, 'best_epoch': checkpoint['epoch'],
                  'parameters': sum(p.numel() for p in model.parameters()), 'threshold': threshold,
                  'runtime': {'tensor_device': str(next(model.parameters()).device),
                              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                              'gpu_name': torch.cuda.get_device_name() if next(model.parameters()).is_cuda else None},
                  'validation': classification_metrics(validation, threshold),
                  'test': classification_metrics(test, threshold),
                  'test_fixed_05': classification_metrics(test, .5)}
        (directory / 'predictions.json').write_text(json.dumps({'validation': validation, 'test': test}, indent=2))
        (directory / 'metrics.json').write_text(json.dumps(result, indent=2))
        (directory / 'config.json').write_text(json.dumps(config, indent=2))
        tracker.final(result)
        print(json.dumps(result), flush=True)
        return result


def summarize(results, output):
    output = Path(output)
    summary = {}
    for name in NAMES:
        runs = [r for r in results if r['model'] == name]
        if not runs:
            continue
        fields = {}
        for metric in runs[0]['test']:
            values = np.array([r['test'][metric] for r in runs])
            fields[metric] = {'mean': float(values.mean()),
                              'std': float(values.std(ddof=1)) if len(values) > 1 else None}
        summary[name] = {'n_training_seeds': len(runs), 'test': fields,
                         'parameters': runs[0]['parameters']}
    # Paired differences preserve the shared split/training-seed pairing.
    paired = {}
    for a, b in [('latent_ode', 'legacy_transformer'), ('ode_rnn', 'legacy_transformer'), ('latent_ode', 'ode_rnn')]:
        first = {r['seed']: r for r in results if r['model'] == a}
        second = {r['seed']: r for r in results if r['model'] == b}
        common = sorted(first.keys() & second.keys())
        if common:
            paired[f'{a} minus {b}'] = {}
            for key in ('auroc', 'ap', 'brier'):
                diff = np.array([first[s]['test'][key] - second[s]['test'][key] for s in common])
                paired[f'{a} minus {b}'][key] = {'mean': float(diff.mean()),
                    'std': float(diff.std(ddof=1)) if len(diff) > 1 else None, 'n_pairs': len(diff)}
    report = {'runs': results, 'summary': summary, 'paired_differences': paired}
    (output / 'summary.json').write_text(json.dumps(report, indent=2))
    lines = ['# Three-model classification comparison', '',
             'Mean ± sample standard deviation (ddof=1) across independent training seeds, fixed patient split.',
             'Threshold selected on validation F1. AUROC/AP do not depend on the threshold.', '',
             '| Model | Seeds | Parameters | AUROC | AP | Brier | Accuracy | F1 | Precision | Recall |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in summary.items():
        scores = []
        for key in ('auroc', 'ap', 'brier', 'accuracy', 'f1', 'precision', 'recall'):
            x = row['test'][key]
            scores.append(f"{x['mean']:.4f} ± {x['std']:.4f}" if x['std'] is not None else f"{x['mean']:.4f} (std unavailable)")
        lines.append(f"| {name} | {row['n_training_seeds']} | {row['parameters']} | " + ' | '.join(scores) + ' |')
    lines += ['', 'This estimates training randomness on one fixed split; it is not cross-validation or a clinical generalization estimate.',
              'Native auxiliary losses and parameter counts differ; use --objective bce for a shared classification-loss ablation.']
    with (output / 'runs.csv').open('w', newline='') as handle:
        columns = ['model', 'seed', 'split_seed', 'objective', 'best_epoch', 'parameters', 'threshold', *results[0]['test']]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow({**{key: row[key] for key in columns if key in row}, **row['test']})
    with (output / 'summary.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model', 'n_training_seeds', 'metric', 'mean', 'std_ddof1'])
        for name, row in summary.items():
            for key, value in row['test'].items():
                writer.writerow([name, row['n_training_seeds'], key, value['mean'], value['std']])
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/raw/synthetic_icu_data.csv')
    p.add_argument('--output', default='results/comparison_cuda')
    p.add_argument('--models', nargs='+', choices=NAMES, default=list(NAMES))
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=10)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--weight-decay', type=float, default=0.)
    p.add_argument('--clip-grad', type=float, default=10.)
    p.add_argument('--train-samples', type=int, default=3)
    p.add_argument('--eval-samples', type=int, default=32)
    p.add_argument('--eval-mode', choices=['mc', 'mean'], default='mc')
    p.add_argument('--eval-seed', type=int, default=1000000)
    p.add_argument('--objective', choices=['native', 'bce'], default='native')
    p.add_argument('--capacity', choices=['base', 'matched'], default='base',
                   help='matched: recognition hidden 143 for Latent ODE; state 140 for ODE-RNN')
    p.add_argument('--latent', type=int, default=20)
    p.add_argument('--hidden', type=int, default=40)
    p.add_argument('--width', type=int, default=50)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--device', default='cuda')
    p.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3], help='physical GPU indices, one visible GPU per worker')
    p.add_argument('--jobs', type=int, default=4, help='independent model/seed worker processes')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--resume', action='store_true', help='skip completed matching runs; incomplete runs require a new directory')
    p.add_argument('--reevaluate', action='store_true', help='reload existing checkpoints and reproduce predictions')
    return p


class _SingleGPU:
    def __init__(self, index):
        self.index = index

    def get(self):
        return self.index


def initialize_worker(gpu_queue):
    if gpu_queue is not None:
        # Spawn workers have not initialized CUDA. Isolate one physical GPU
        # before the first CUDA call; cuda:0 then denotes that assigned GPU.
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_queue.get())
        torch.cuda.set_device(0)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def run_pair(name, seed, args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    splits, metadata = prepare_data(args.data, args.split_seed)
    metadata['data_sha256'] = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    return run_one(name, seed, splits, metadata, args)


def main():
    args = parser().parse_args()
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('Project policy requires CUDA model experiments; CPU fallback is disabled')
    if min(args.epochs, args.batch_size, args.train_samples, args.eval_samples, args.jobs) < 1:
        raise ValueError('epochs, batch size and sample counts must be positive')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError('training seeds must be distinct and nonnegative')
    if args.gpus is not None:
        if args.device != 'cuda' or len(set(args.gpus)) != len(args.gpus) or not args.gpus:
            raise ValueError('--gpus requires --device cuda and distinct GPU indices')
        if any(g < 0 or g >= torch.cuda.device_count() for g in args.gpus):
            raise ValueError('requested GPU is not visible')
        if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
            raise ValueError('unset CUDA_VISIBLE_DEVICES when using physical --gpus indices')
        if args.jobs < len(args.gpus):
            raise ValueError('jobs must be at least the number of requested GPUs')
    if len(set(args.models)) != len(args.models):
        raise ValueError('models must be distinct')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    splits, metadata = prepare_data(args.data, args.split_seed)
    metadata['data_sha256'] = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        previous = json.loads((out / 'protocol.json').read_text())
        if previous != configuration(args):
            raise ValueError('output directory belongs to a different protocol; choose a new output')
    if (out / 'data.json').exists() and json.loads((out / 'data.json').read_text()) != metadata:
        raise ValueError('raw data or patient split differs from existing experiment')
    (out / 'protocol.json').write_text(json.dumps(configuration(args), indent=2))
    (out / 'data.json').write_text(json.dumps(metadata, indent=2))
    (out / 'environment.json').write_text(json.dumps({'python': platform.python_version(),
        'torch': torch.__version__, 'numpy': np.__version__, 'device': args.device,
        'threads': args.threads, 'requested_gpus': args.gpus, 'deterministic_algorithms': True}, indent=2))
    results = []
    if args.jobs == 1:
        if args.gpus:
            initialize_worker(_SingleGPU(args.gpus[0]))
        else:
            initialize_worker(None)
        for seed in args.seeds:
            for name in args.models:
                results.append(run_one(name, seed, splits, metadata, args))
                summarize(results, out)
    else:
        context = multiprocessing.get_context('spawn')
        gpu_queue = None
        if args.gpus:
            gpu_queue = context.Queue()
            for index in range(args.jobs):
                gpu_queue.put(args.gpus[index % len(args.gpus)])
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=context,
                                 initializer=initialize_worker, initargs=(gpu_queue,)) as pool:
            pending = [pool.submit(run_pair, name, seed, args) for seed in args.seeds for name in args.models]
            for future in as_completed(pending):
                results.append(future.result())
                results.sort(key=lambda row: (row['seed'], NAMES.index(row['model'])))
                summarize(results, out)



if __name__ == '__main__':
    main()
