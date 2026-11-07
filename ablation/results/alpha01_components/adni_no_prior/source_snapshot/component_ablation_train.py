"""Exact-budget component ablations; original data protocols, max150 epochs."""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from paper_odeae.train import seed_all, device_batch, evaluate as evaluate_adni
from paper_odeae.data import Prefixes, collate as collate_adni
from paper_odeae.models import feature_errors, reconstruction, prior_errors
from paper_odeae.jacobian import jacobian_regularization
from src.tracking import SwanLabTracker
from component_ablation_model import make_model, unique_fields
from component_ablation_loss import loss_function as component_loss, jacobian_regularization as component_jacobian
from component_ablation_checks import verify
from fixed_update_checks import parameter_hash


def atomic(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)


def save(path, value):
    tmp = path.with_suffix('.partial.pt'); torch.save(value, tmp); tmp.replace(path)


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(), 'numpy': np.random.get_state(), 'python': random.getstate()}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--stage', choices=['checks','training'],default='training'); args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text()); folder = Path(cfg['output'])
    if Path(sys.prefix).name != 'ode' or not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') != str(cfg['physical_gpu']):
        raise RuntimeError('Assigned CUDA device and ode required')
    if cfg['max_epochs'] > 150 or cfg['gradient_accumulation_steps'] != 1: raise RuntimeError('150 epoch cap; no accumulation')
    torch.set_num_threads(1); torch.cuda.set_device(0); seed_all(cfg['seed'])
    raw = Path(cfg['data']).read_bytes(); assert hashlib.sha256(raw).hexdigest() == cfg['dataset_sha256']
    payload = pickle.loads(raw); schema = payload['schema']
    splits = {s: [x for x in payload['sequences'] if x['split']==s] for s in ['train', 'validation', 'test']}
    if (folder/'metrics.json').exists(): return
    # Evaluation reports the same averaged-field Jacobian diagnostic as training.
    import paper_odeae.jacobian as jac_module
    import paper_odeae.train as train_module
    jac_module.jacobian_regularization = component_jacobian
    train_module.jacobian_regularization = component_jacobian
    if cfg['dataset'] == 'mimic':
        from mimic_experiment.data import ICUContexts, ExactBatchSampler, collate
        from mimic_experiment.train_baseline import normalization_and_modes
        from mimic_experiment.train_full_batch import loss_function as original_loss
        from mimic_experiment.evaluation import evaluate as evaluate_mimic
        means, modes = normalization_and_modes(payload); cfg.update(raw_scale=payload['metadata']['scale'], normalized_train_means=means)
        dataset = ICUContexts(splits['train'], training=True, seed=cfg['seed'])
        sampler = ExactBatchSampler(len(dataset), cfg['batch_size'], 0, 1, cfg['seed'])
        def batches(epoch):
            dataset.set_epoch(epoch); sampler.set_epoch(epoch)
            for indices in sampler: yield device_batch(collate([dataset[i] for i in indices], cfg['input_dropout']))
        def loss_function(model, out, batch): return component_loss(model, out, batch, schema, cfg)
        def evaluate(model, split): return evaluate_mimic(model, splits[split], schema, cfg, payload['metadata']['raw_train_std'], modes)
    else:
        assert cfg['normalization'] == payload['metadata']['normalization'] and cfg['scope'] == payload['metadata']['scope']
        def batches(epoch):
            g = torch.Generator().manual_seed(cfg['seed']+epoch)
            for batch in DataLoader(Prefixes(splits['train']), batch_size=cfg['batch_size'], shuffle=True, generator=g,
                                    collate_fn=partial(collate_adni, dropout=cfg['input_dropout'])):
                yield device_batch(batch)
        def loss_function(model, out, batch): return component_loss(model, out, batch, schema, cfg)
        def evaluate(model, split): return evaluate_adni(model, splits[split], schema, cfg['batch_size'])
    model = make_model(schema, cfg, cfg['update_alpha']); initial_hash = parameter_hash(model)
    # Prove constructor equality without advancing the actual model initialization.
    seed_all(cfg['seed']); native = make_model(schema, cfg, 1.); assert parameter_hash(native) == initial_hash; del native
    atomic(folder/'initialization_check.json', {'identical': True, 'seed': cfg['seed'], 'initial_parameter_hash': initial_hash,
                                              'physical_gpu': cfg['physical_gpu'], 'parameter_device': str(next(model.parameters()).device)})
    if args.stage=='checks' or not (folder/'checks.json').exists():
        probe = next(batches(1)); atomic(folder/'checks.json', verify(model, probe, schema, cfg, loss_function)); del probe
    if args.stage=='checks': return
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: cfg['lr_decay']**(step/cfg['lr_decay_steps']))
    history, best, bad, steps, first = [], float('inf'), 0, 0, 1
    if (folder/'last.pt').exists():
        saved = torch.load(folder/'last.pt', map_location='cuda', weights_only=False); model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        history, best, bad, steps, first = saved['history'], saved['best_score'], saved['bad_epochs'], saved['optimizer_steps'], saved['epoch']+1
    else:
        save(folder/'initial.pt', {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rng_state': rng_state(), 'config': cfg, 'epoch': 0})
    inactive = sum(p.numel() for p in model.update.parameters()) if cfg['update_alpha'] == 0 else 0
    nominal = sum(p.numel() for p in model.parameters())
    hardware = {'physical_gpu': cfg['physical_gpu'], 'gpu_name': torch.cuda.get_device_name(), 'parameter_device': str(next(model.parameters()).device),
                'parameters': nominal, 'task_used_parameters': nominal-inactive, 'inactive_update_parameters': inactive,
                'latent': model.latent, 'alpha': cfg['update_alpha'], 'ablation_component':cfg['ablation_component'],
                'unique_ode_fields':len(unique_fields(model)), 'unique_ode_parameters':sum(p.numel() for f in unique_fields(model) for p in f.parameters()),
                'shared_vector_field':model.shared_vector_field, 'ode_l2_definition':'sum over all distinct ODE parameters; coefficient0.02',
                'jacobian_definition':'mean over distinct fields and detached valid main-generation states', 'training_batch_size': cfg['batch_size'], 'gradient_accumulation_steps': 1,
                'max_epochs': 150, 'solver': 'Tsit5', 'rtol': cfg['rtol'], 'atol': cfg['atol'],
                'time_unit': 'hour' if cfg['dataset']=='mimic' else 'year', 'dataset_sha256': cfg['dataset_sha256']}
    atomic(folder/'config.json', cfg); atomic(folder/'hardware.json', hardware)
    with SwanLabTracker(folder, cfg, 'ablation_alpha01_'+cfg['dataset']+'_'+cfg['ablation_component'], model, cfg['seed']) as tracker:
        tracker.step = steps
        for epoch in range(first, cfg['max_epochs']+1):
            if epoch > cfg['min_epochs'] and bad >= cfg['patience']: break
            seed_all(cfg['seed']+epoch); model.train(); total = torch.zeros(7, device='cuda'); norms = []; count = 0; start = time.monotonic()
            for batch in batches(epoch):
                optimizer.zero_grad(set_to_none=True); out = model(batch, sample=False); loss, parts = loss_function(model, out, batch)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite fixed update loss')
                loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); steps += 1; count += 1; norms.append(float(norm)); total += torch.cat([parts, norm.detach()[None]])
                tracker.batch(loss, scheduler.get_last_lr()[0], gradient_norm=norm, mse=parts[0], cce=parts[1], prior_mse=parts[2], prior_cce=parts[3], jacobian_frobenius_squared=parts[4], ode_l2=parts[5])
                if 'actual_tensor_devices' not in hardware:
                    hardware.update(actual_tensor_devices={k: str(v.device) for k, v in batch.items() if torch.is_tensor(v)}, prediction_device=str(out['prediction'].device)); atomic(folder/'hardware.json', hardware)
                del out, loss
            values = {k: float(v/count) for k, v in zip(['mse', 'cce', 'prior_mse', 'prior_cce', 'jacobian_frobenius_squared', 'ode_l2', 'gradient_norm'], total)}
            values.update(weighted_prior=cfg['prior_weight']*(values['prior_mse']+values['prior_cce']), weighted_jacobian=cfg['jacobian_weight']*values['jacobian_frobenius_squared'], weighted_ode_l2=cfg['l2_weight']*values['ode_l2'])
            training_loss = cfg['mse_weight']*values['mse']+cfg['cce_weight']*values['cce']+values['weighted_prior']+values['weighted_jacobian']+values['weighted_ode_l2']
            validation = evaluate(model, 'validation'); row = {'epoch': epoch, 'optimizer_steps': steps, 'training_loss': training_loss, 'training_components': values, 'validation': validation,
                'maximum_batch_gradient_norm': max(norms), 'batches_above_100': sum(n > 100 for n in norms), 'epoch_seconds': time.monotonic()-start}
            history.append(row); saved = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rng_state': rng_state(), 'config': cfg, 'schema': schema, 'metadata': payload['metadata'], 'epoch': epoch}
            if validation['overall']['reconstruction_error'] < best:
                best = validation['overall']['reconstruction_error']; bad = 0; save(folder/'best.pt', saved)
            else: bad += 1
            save(folder/'last.pt', {**saved, 'history': history, 'best_score': best, 'bad_epochs': bad, 'optimizer_steps': steps})
            atomic(folder/'history.json', history); tracker.epoch(epoch, values, validation, scheduler.get_last_lr()[0], maximum_batch_gradient_norm=max(norms))
            print(json.dumps({'epoch': epoch, 'validation_history': validation['interpolation']['reconstruction_error'], 'validation_future': validation['forecast']['reconstruction_error'], 'validation_overall': validation['overall']['reconstruction_error']}), flush=True)
        saved = torch.load(folder/'best.pt', map_location='cuda', weights_only=False); model.load_state_dict(saved['state_dict'])
        validation = evaluate(model, 'validation'); expected = next(h for h in history if h['epoch']==saved['epoch'])['validation']['overall']['reconstruction_error']
        difference = abs(validation['overall']['reconstruction_error']-expected); assert difference < 1e-6
        metrics = {'status': 'complete', 'dataset': cfg['dataset'], 'alpha': cfg['update_alpha'], 'best_epoch': saved['epoch'], 'epochs_run': len(history),
                   'validation': validation, 'test': evaluate(model, 'test'), 'checkpoint_reload_difference': difference, 'max_epochs': 150, 'ablation_component':cfg['ablation_component']}
        if cfg['dataset']=='mimic':
            metrics['test_early_contexts'] = {str(h): evaluate_mimic(model, splits['test'], schema, cfg, payload['metadata']['raw_train_std'], modes, horizon=h) for h in [6, 12]}
        atomic(folder/'metrics.json', metrics); tracker.final(metrics)


if __name__ == '__main__': main()
