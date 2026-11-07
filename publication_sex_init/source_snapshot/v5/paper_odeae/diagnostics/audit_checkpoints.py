"""Read-only CUDA diagnostics on completed reproduction checkpoints.

Run four independent workers, one per physical GPU. Only aggregate statistics
are saved; original checkpoints, predictions and training results are untouched.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import DataLoader
from paper_odeae.data import Prefixes, collate
from paper_odeae.models import PaperModel, feature_errors
from paper_odeae.train import device_batch, evaluate

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'paper_odeae/results/discrepancy_audit'
JOBS = [('ode_ae', 0), ('ode_ae', 1), ('deterministic', 0), ('lstm', 0)]


@torch.no_grad()
def worker(index):
    if not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') != str(index):
        raise RuntimeError('explicit physical CUDA assignment required')
    torch.set_num_threads(1)
    torch.manual_seed(20261027)
    model_name, seed = JOBS[index]
    run = ROOT/f'paper_odeae/results/local_available/{model_name}/seed_{seed}'
    checkpoint = torch.load(run/'best.pt', map_location='cuda', weights_only=False)
    cfg = checkpoint['config']
    payload = pickle.loads((ROOT/cfg['data']).read_bytes())
    schema = payload['schema']
    model = PaperModel(schema, model_name, cfg['latent'], cfg['ode_width'], cfg['mlp_width'], cfg['rtol'], cfg['atol']).cuda()
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    sequences = [s for s in payload['sequences'] if s['split'] == 'validation']
    # A fixed, evenly spaced set of patients; all their prefixes are included.
    chosen = [sequences[i] for i in np.linspace(0, len(sequences)-1, 64, dtype=int)]
    loader = DataLoader(Prefixes(chosen), batch_size=32, collate_fn=collate)
    mus, stds, totals = [], [], {}
    for name in ['mean', 'sample', 'no_context']:
        totals[name] = torch.zeros(2, len(schema), device='cuda', dtype=torch.float64)
    for cpu in loader:
        batch = device_batch(cpu)
        out = model(batch, sample=False)
        mus.append(out['mu']); stds.append(out['std'])
        sampled = model(batch, sample=True)
        blank = {**batch, 'context_mask': torch.zeros_like(batch['context_mask'])}
        no_context = model(blank, sample=False)
        for name, prediction in [('mean', out), ('sample', sampled), ('no_context', no_context)]:
            totals[name][0] += feature_errors(prediction['prediction'], batch['values'], batch['mask'], schema).sum((0,1)).double()
            totals[name][1] += batch['mask'].sum((0,1)).double()
    mu, std = torch.cat(mus), torch.cat(stds)
    covariance = (mu-mu.mean(0)).T @ (mu-mu.mean(0)) / (len(mu)-1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    numeric = torch.tensor([f['kind']=='numerical' for f in schema], device='cuda')
    scores = {}
    for name, total in totals.items():
        means = total[0]/total[1].clamp_min(1)
        scores[name] = {'mse': float(means[numeric].sum()), 'cce': float(means[~numeric].sum()), 'overall': float(means.sum())}
    # Compare numerical tolerance at fixed parameters and fixed validation data.
    regular = evaluate(model, chosen, schema, 32)
    model.rtol, model.atol = cfg['rtol']/10, cfg['atol']/10
    tighter = evaluate(model, chosen, schema, 32)
    kl = 0. if model_name == 'deterministic' else float(.5*(mu.square()+std.square()-1-2*std.log()).sum(-1).mean())
    result = {
        'model': model_name, 'seed': seed, 'checkpoint_epoch': checkpoint['epoch'],
        'physical_gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'parameter_device': str(next(model.parameters()).device),
        'gpu_name': torch.cuda.get_device_name(), 'validation_patients': len(chosen), 'prefixes': len(mu),
        'posterior_mu_rms': float(mu.square().mean().sqrt()),
        'posterior_mu_std_across_prefixes': mu.std(0).tolist(),
        'posterior_mu_covariance_eigenvalues': eigenvalues.tolist(),
        'largest_mu_variance_fraction': float(eigenvalues[-1]/eigenvalues.sum().clamp_min(1e-20)),
        'posterior_std_mean': float(std.mean()), 'posterior_kl_per_prefix': kl,
        'ode_l2': float(model.ode_regularization()), 'scores': scores,
        'tolerance_check': {'regular': regular['overall']['reconstruction_error'], 'tighter': tighter['overall']['reconstruction_error'],
            'absolute_difference': abs(regular['overall']['reconstruction_error']-tighter['overall']['reconstruction_error'])},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f'{model_name}_seed_{seed}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker); return
    OUT.mkdir(parents=True, exist_ok=True)
    def launch(index):
        with (OUT/f'gpu_{index}.log').open('w') as stream:
            subprocess.run([sys.executable, '-m', 'paper_odeae.diagnostics.audit_checkpoints', '--worker', str(index)],
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(index), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                stdout=stream, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(launch, range(4)))


if __name__ == '__main__':
    main()
