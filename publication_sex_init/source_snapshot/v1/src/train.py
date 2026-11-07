"""Train both paper models; checkpoint selection uses validation data only."""
import argparse
from functools import partial
import json
import random
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from .tracking import SwanLabTracker
from .model import MODELS, objective
from .preprocessing import prepare_data, collate_fn, make_context


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_batch(model, batch, args, training=False):
    # Context sampling is performed on CPU for deterministic explicit RNG.
    context = make_context(batch, args.task, args.observed_fraction,
                           generator=None if training else torch.Generator().manual_seed(args.seed + 100))
    device = next(model.parameters()).device
    context = {k: v.to(device) for k, v in context.items()}
    if args.model == 'latent_ode':
        output = model(context['values'], context['times'], context['mask'],
                       query_times=context['query_times'], samples=args.samples,
                       task=args.task, deterministic=False)
        target, mask = context['target'], context['target_mask']
    elif training or args.task == 'interpolation':
        # Paper: AR models train on the interpolation/next-step task, including
        # when the eventual test task is extrapolation. Predict before update.
        # Interpolation consumes the subsampled context, scoring every timestamp.
        values, mask = batch['values'].to(device), batch['mask'].to(device)
        times = batch['times'].to(device)
        if training:
            feed_mask = mask
        else:
            feed_mask = torch.zeros_like(mask)
            indices = torch.searchsorted(times, context['times'])
            feed_mask[:, indices] = context['mask']
        output = model(values, times, feed_mask, teacher_forcing=.5 if training else 1.)
        target = values
    else:
        history = model(context['values'], context['times'], context['mask'])
        prediction = model.forecast(context['values'], context['times'], context['mask'],
                                    context['query_times'], context['target_mask'])
        output = {'prediction': prediction, 'logits': history['logits']}
        target, mask = context['target'], context['target_mask']
    labels = batch['labels'].to(device)
    loss = objective(output, target, mask, labels, args.classification_weight, args.variance)
    return output, loss, labels, int(mask.sum().item())


@torch.no_grad()
def evaluate(model, loader, args):
    model.eval()
    # Fixed Monte Carlo draws, independent of training and loader ordering.
    seed_all(args.seed + 1000)
    totals = dict(loss=0., nll=0., kl=0., bce=0., squared_error=0.)
    count = elements = 0
    probabilities, labels = [], []
    for batch in loader:
        output, losses, truth, observed = model_batch(model, batch, args)
        n = len(truth)
        for key in ('loss', 'nll', 'kl', 'bce'):
            totals[key] += float(losses[key]) * n
        totals['squared_error'] += float(losses['mse']) * observed
        count += n
        elements += observed
        probabilities.extend(output['logits'].sigmoid().mean(0).cpu().tolist())
        labels.extend(truth.cpu().tolist())
    result = {k: totals[k] / count for k in ('loss', 'nll', 'kl', 'bce')}
    result.update(mse=totals['squared_error'] / max(1, elements), patients=count, observed_scalars=elements)
    if args.classification_weight and len(set(labels)) == 2:
        result.update(auroc=float(roc_auc_score(labels, probabilities)),
                      auprc=float(average_precision_score(labels, probabilities)))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, default='latent_ode')
    p.add_argument('--task', choices=['interpolation', 'extrapolation'], default='interpolation')
    p.add_argument('--data', default='data/raw/synthetic_icu_data.csv')
    p.add_argument('--output', default='results/paper/latent_ode_interpolation')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=10)
    p.add_argument('--latent', type=int, default=20)
    p.add_argument('--hidden', type=int, default=40)
    p.add_argument('--width', type=int, default=50)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--samples', type=int, default=3)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--variance', type=float, default=.01)
    p.add_argument('--classification-weight', type=float, default=0.)
    p.add_argument('--observed-fraction', type=float, default=.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=1)
    return p


def main():
    args = parser().parse_args()
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('Project policy requires CUDA experiments; CPU fallback is disabled')
    if args.epochs < 1:
        raise ValueError('epochs must be positive')
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    splits, metadata = prepare_data(args.data, args.seed)
    # A separate generator prevents validation RNG reset from changing shuffling.
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {name: DataLoader(ds, batch_size=args.batch_size, shuffle=name == 'train',
                                generator=generator if name == 'train' else None,
                                collate_fn=partial(collate_fn, task=args.task)) for name, ds in splits.items()}
    model = MODELS[args.model](latent=args.latent, hidden=args.hidden,
                              width=args.width, layers=args.layers).to(args.device)
    (output / 'hardware.json').write_text(json.dumps({
        'tensor_device': str(next(model.parameters()).device),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_name': torch.cuda.get_device_name(next(model.parameters()).device)}, indent=2))
    with SwanLabTracker(output, vars(args), 'paper_baselines', model, args.seed) as tracker:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
        (output / 'data.json').write_text(json.dumps(metadata, indent=2))
        baseline = evaluate(model, loaders['val'], args)
        history, best = [], float('inf')
        for epoch in range(1, args.epochs + 1):
            seed_all(args.seed + epoch)
            model.train()
            total, count = 0., 0
            for batch in loaders['train']:
                optimizer.zero_grad(set_to_none=True)
                _, losses, labels, _ = model_batch(model, batch, args, training=True)
                if not torch.isfinite(losses['loss']):
                    raise FloatingPointError('nonfinite training loss')
                losses['loss'].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                tracker.batch(losses['loss'], optimizer.param_groups[0]['lr'], gradient_norm=grad_norm, components=losses)
                total += float(losses['loss'].detach()) * len(labels)
                count += len(labels)
            val = evaluate(model, loaders['val'], args)
            row = {'epoch': epoch, 'train_loss': total / count, 'validation': val}
            history.append(row)
            tracker.epoch(epoch, {'loss': total / count}, val, optimizer.param_groups[0]['lr'])
            print(json.dumps(row), flush=True)
            # MSE is the paper's reported endpoint for both model families.
            if val['mse'] < best:
                best = val['mse']
                torch.save({'state_dict': model.state_dict(), 'config': vars(args),
                            'epoch': epoch, 'validation': val, 'metadata': metadata}, output / 'best.pt')
            (output / 'history.json').write_text(json.dumps(history, indent=2))
        checkpoint = torch.load(output / 'best.pt', map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        results = {'baseline_validation': baseline, 'best_epoch': checkpoint['epoch'],
                   'validation': evaluate(model, loaders['val'], args),
                   'test': evaluate(model, loaders['test'], args)}
        (output / 'metrics.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2), flush=True)
        tracker.final(results)


if __name__ == '__main__':
    main()
