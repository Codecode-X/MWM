"""CUDA-only ADNI longitudinal forecasting, fixed patient splits and paired seeds."""
import argparse
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .tracking import SwanLabTracker
from .adni_data import CACHE
from .adni_models import make_model, forward


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(cache, imaging=False, projection=None):
    with (cache / 'prepared.pkl').open('rb') as f:
        data = pickle.load(f)
    sequences, metadata = data['sequences'], data['metadata']
    mean, scale = np.array(metadata['mean'], np.float32), np.array(metadata['scale'], np.float32)
    feature_count = len(mean)
    if imaging:
        paths = {p.stem: p for p in (cache / 'embeddings').glob('*.npz')}
        train_ids = [s['patient_id'] for s in sequences if s['split'] == 'train' and s['patient_id'] in paths]
        if len(train_ids) < 100:
            raise ValueError('too few MRI training embeddings; extraction must finish first')
        if projection is None:
            from sklearn.decomposition import PCA
            train = np.stack([np.load(paths[s])['embedding'] for s in train_ids])
            pca = PCA(n_components=16, svd_solver='full').fit(train)
            projected = pca.transform(train)
            image_scale = np.maximum(projected.std(0), 1e-5)
            projection = {'mean': pca.mean_, 'components': pca.components_, 'scale': image_scale,
                          'explained_variance_ratio':pca.explained_variance_ratio_}
        metadata['imaging'] = {'encoder': 'frozen MedicalNet original 3DSeg-8 ResNet10',
                               'dimensions': 16, 'pca_fit_patients': len(train_ids),
                               'explained_variance_ratio': np.asarray(projection.get('explained_variance_ratio',[])).tolist(),
                               'embedding_patients': len(paths),
                               'available_inputs_by_split': {}}
    out = {s: [] for s in ['train', 'validation', 'test']}
    for seq in sequences:
        s = dict(seq)
        raw = seq['values']
        mask = np.isfinite(raw)
        values = np.where(mask, (raw - mean) / scale, 0).astype(np.float32)
        if imaging:
            extra = np.zeros((len(raw), 16), np.float32)
            extra_mask = np.zeros_like(extra)
            if s['patient_id'] in paths:
                with np.load(paths[s['patient_id']]) as emb:
                    d = date.fromisoformat(str(emb['date'])).toordinal()
                    acquired = (d - s['start_ordinal']) / 365.25
                    at = int(np.searchsorted(s['times'], acquired, side='left'))
                    if at < len(raw):
                        extra[at] = ((emb['embedding'] - projection['mean']) @ projection['components'].T) / projection['scale']
                        extra_mask[at] = 1
                        if at < max(1, len(raw)//2):
                            counts = metadata['imaging']['available_inputs_by_split']
                            counts[s['split']] = counts.get(s['split'], 0) + 1
            values = np.concatenate([values, extra], -1)
            mask = np.concatenate([mask, extra_mask.astype(bool)], -1)
        s.update(values=values, mask=mask.astype(np.float32))
        out[s['split']].append(s)
    return out, metadata, feature_count, projection


def collate(seqs, training=False, max_step=.05, target_features=None):
    b, n, f = len(seqs), max(len(s['times']) for s in seqs), seqs[0]['values'].shape[-1]
    values, mask, context = np.zeros((b,n,f), np.float32), np.zeros((b,n,f), np.float32), np.zeros((b,n,f), np.float32)
    times, valid, future = np.zeros((b,n), np.float32), np.zeros((b,n), bool), np.zeros((b,n), bool)
    cutoffs = []
    for j, s in enumerate(seqs):
        length = len(s['times'])
        cutoff = random.randrange(1, length) if training else max(1, length//2)
        cutoffs.append(cutoff)
        values[j,:length], mask[j,:length] = s['values'], s['mask']
        context[j,:cutoff] = s['mask'][:cutoff]
        times[j,:length] = s['times']/10.  # ODE clock: decades, actual elapsed dates.
        times[j,length:] = times[j,length-1]
        valid[j,:length], future[j,cutoff:length] = True, True
    # Missingness augmentation affects history only, shared protocol for both.
    if training:
        context *= np.random.random(context.shape) > .1
    target_mask = mask * future[...,None]
    if target_features is not None:
        target_mask[:,:,target_features:] = 0
    delta = np.diff(times, axis=1, prepend=times[:,:1])
    steps = [max(1, int(math.ceil(float(v)/max_step))) for v in delta.max(0)]
    previous = times[np.arange(b), np.array(cutoffs)-1].copy()
    max_future_delta = []
    for i in range(n):
        ds = np.where(future[:,i], times[:,i]-previous, 0)
        max_future_delta.append(float(ds.max()))
        previous = np.where(future[:,i], times[:,i], previous)
    return {'values': torch.from_numpy(values), 'mask': torch.from_numpy(mask),
            'context_mask': torch.from_numpy(context), 'times': torch.from_numpy(times),
            'valid': torch.from_numpy(valid), 'is_future': torch.from_numpy(future),
            'target_mask': torch.from_numpy(target_mask), 'cutoff': torch.tensor(cutoffs),
            'max_context': max(cutoffs), 'steps': steps, 'max_future_delta': max_future_delta,
            'max_step': max_step, 'patient_ids': [s['patient_id'] for s in seqs]}


def to_cuda(batch):
    return {k: v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}


def score(prediction, target, mask):
    error = prediction.mean(0) - target
    squared = (error.square()*mask).sum((0,1))
    absolute = (error.abs()*mask).sum((0,1))
    counts = mask.sum((0,1))
    return squared, absolute, counts


def summarize(squared, absolute, counts, names, scale):
    observed = counts > 0
    mse, mae = squared / np.maximum(counts,1), absolute / np.maximum(counts,1)
    per = {name: {'mse_z': float(mse[i]), 'mae_z': float(mae[i]),
                  'mae_native': float(mae[i]*scale[i]), 'observations': int(counts[i])}
           for i, name in enumerate(names) if observed[i]}
    modalities = {}
    for group in sorted({n.split('.')[0] for n in names}):
        idx = np.array([n.startswith(group+'.') for n in names]) & observed
        if idx.any():
            modalities[group] = {'macro_mse_z': float(mse[idx].mean()),
                                  'macro_mae_z': float(mae[idx].mean()),
                                  'observations': int(counts[idx].sum())}
    return {'macro_mse_z': float(mse[observed].mean()), 'macro_mae_z': float(mae[observed].mean()),
            'micro_mse_z': float(squared.sum()/max(counts.sum(),1)),
            'observed_features': int(observed.sum()), 'observations': int(counts.sum()),
            'per_feature': per, 'per_modality': modalities}


@torch.no_grad()
def evaluate(model, seqs, args, metadata, target_features, save=None, step_multiplier=1):
    model.eval()
    seed_all(args.seed + 10000)
    totals = torch.zeros(3, target_features, device='cuda', dtype=torch.float64)
    baseline = totals.clone()
    horizon_totals = {k: totals.clone() for k in ['0_to_1_year','1_to_3_years','over_3_years']}
    patient_scores = []
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(seqs, batch_size=args.batch_size, shuffle=False, generator=generator,
                        collate_fn=lambda s: collate(s, max_step=args.max_step, target_features=target_features))
    for cpu_batch in loader:
        batch = to_cuda(cpu_batch)
        output = forward(model, batch, args.model, samples=args.eval_samples, step_multiplier=step_multiplier)
        prediction = output['prediction'][...,:target_features]
        target, mask = batch['values'][...,:target_features], batch['target_mask'][...,:target_features]
        totals += torch.stack(score(prediction, target, mask)).double()
        cutoff_time=batch['times'].gather(1,(batch['cutoff']-1)[:,None])
        # Quarter timestamps are exactly on a 0.25-year grid; restore that
        # grid before assigning bins to avoid float32 errors at 1/3 years.
        horizon=((batch['times']-cutoff_time)*40.).round()/4.
        for name,low,high in [('0_to_1_year',0.,1.),('1_to_3_years',1.,3.),('over_3_years',3.,float('inf'))]:
            hm=mask*((horizon>low)&(horizon<=high))[...,None]
            horizon_totals[name] += torch.stack(score(prediction,target,hm)).double()
        # Last observed value per feature; unavailable history uses train mean.
        last = torch.zeros(len(target),target_features,device='cuda')
        for i in range(target.shape[1]):
            available = batch['context_mask'][:,i,:target_features].bool()
            last = torch.where(available, target[:,i], last)
        persistence = last[:,None].expand_as(target)[None]
        baseline += torch.stack(score(persistence, target, mask)).double()
        for j, patient in enumerate(batch['patient_ids']):
            e = prediction.mean(0)[j]-target[j]
            m = mask[j]
            patient_scores.append({'patient_id': patient,
                'squared': (e.square()*m).sum(0).cpu().numpy(),
                'absolute': (e.abs()*m).sum(0).cpu().numpy(),
                'counts': m.sum(0).cpu().numpy()})
    stats = totals.cpu().numpy()
    result = summarize(*stats, metadata['features'], np.array(metadata['scale']))
    result['persistence'] = summarize(*baseline.cpu().numpy(), metadata['features'], np.array(metadata['scale']))
    result['patients'] = len(seqs)
    result['per_horizon']={k:summarize(*v.cpu().numpy(),metadata['features'],np.array(metadata['scale']))
                           for k,v in horizon_totals.items() if v[2].sum()>0}
    if save:
        np.savez_compressed(save, patient_ids=np.array([p['patient_id'] for p in patient_scores]),
                            squared=np.stack([p['squared'] for p in patient_scores]),
                            absolute=np.stack([p['absolute'] for p in patient_scores]),
                            counts=np.stack([p['counts'] for p in patient_scores]))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', default=CACHE)
    p.add_argument('--model', choices=['latent_ode','ode_rnn'], required=True)
    p.add_argument('--imaging', action='store_true')
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--latent', type=int, default=32)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--kl-weight', type=float, default=.001)
    p.add_argument('--max-step', type=float, default=.05)
    p.add_argument('--eval-samples', type=int, default=8)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if not torch.cuda.is_available() or 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise RuntimeError('explicit physical CUDA GPU assignment is required')
    torch.set_num_threads(1)
    seed_all(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    splits, metadata, target_features, projection = load_data(Path(args.cache),args.imaging)
    if args.smoke:
        splits = {k:v[:64] for k,v in splits.items()}
    model = make_model(args.model, splits['train'][0]['values'].shape[-1], args.latent,args.hidden,args.width).cuda()
    config = vars(args)
    (output/'config.json').write_text(json.dumps(config,indent=2))
    (output/'data.json').write_text(json.dumps(metadata,indent=2))
    (output/'hardware.json').write_text(json.dumps({'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],
         'parameter_device': str(next(model.parameters()).device), 'input_device':'cuda:0',
         'gpu_name':torch.cuda.get_device_name(), 'torch_version':torch.__version__,
         'parameters':sum(x.numel() for x in model.parameters()),
         'forecast_network_parameters':sum(x.numel() for k,x in model.named_parameters() if not k.startswith('classifier.')),
         'prepared_sha256':hashlib.sha256((Path(args.cache)/'prepared.pkl').read_bytes()).hexdigest()},indent=2))
    with SwanLabTracker(output, config, 'adni', model, args.seed) as tracker:
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-5)
        history, best, started = [], float('inf'), time.monotonic()
        for epoch in range(1,args.epochs+1):
            seed_all(args.seed+epoch)
            model.train()
            g = torch.Generator().manual_seed(args.seed+epoch)
            loader = DataLoader(splits['train'],batch_size=args.batch_size,shuffle=True,generator=g,
                collate_fn=lambda s:collate(s,training=True,max_step=args.max_step,target_features=target_features))
            total, batches = 0.,0
            for cpu_batch in loader:
                batch = to_cuda(cpu_batch)
                optimizer.zero_grad(set_to_none=True)
                out = forward(model,batch,args.model,training=True,samples=3)
                if epoch == 1 and batches == 0:
                    hardware = json.loads((output/'hardware.json').read_text())
                    hardware['actual_training_tensor_devices'] = {
                        'values': str(batch['values'].device), 'times': str(batch['times'].device),
                        'context_mask': str(batch['context_mask'].device),
                        'prediction': str(out['prediction'].device),
                        'target_mask': str(batch['target_mask'].device)}
                    (output/'hardware.json').write_text(json.dumps(hardware,indent=2))
                mask = batch['target_mask']
                # Expected sample squared error, balanced across observed features;
                # prevents the variational posterior from inflating its variance.
                err = (out['prediction']-batch['values'][None]).square().mean(0)
                counts = mask.sum((0,1))
                eligible = counts>0
                loss = ((err*mask).sum((0,1))/counts.clamp_min(1))[eligible].mean()
                kl = out.get('kl',loss.new_zeros(()))
                loss = loss+args.kl_weight*min(1.,epoch/10.)*kl/args.latent
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite training loss')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                optimizer.step()
                tracker.batch(loss, optimizer.param_groups[0]['lr'], kl=kl, gradient_norm=grad_norm)
                total += float(loss.detach())
                batches += 1
            val = evaluate(model,splits['validation'],args,metadata,target_features)
            row = {'epoch':epoch,'train_loss':total/batches,
                    'validation_macro_mse_z':val['macro_mse_z'],
                    'validation_macro_mae_z':val['macro_mae_z'],
                    'elapsed_seconds':time.monotonic()-started}
            history.append(row)
            tracker.epoch(epoch, {'loss': total/batches}, val, optimizer.param_groups[0]['lr'], elapsed_seconds=row['elapsed_seconds'])
            print(json.dumps(row),flush=True)
            if val['macro_mse_z']<best:
                best=val['macro_mse_z']
                torch.save({'state_dict':model.state_dict(),'config':config,'metadata':metadata,
                             'projection':projection,'epoch':epoch},output/'best.pt')
            (output/'history.json').write_text(json.dumps(history,indent=2))
        c = torch.load(output/'best.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(c['state_dict'])
        result = {'best_epoch':c['epoch'],
                  'validation':evaluate(model,splits['validation'],args,metadata,target_features,output/'validation_patients.npz'),
                  'test':evaluate(model,splits['test'],args,metadata,target_features,output/'test_patients.npz')}
        fine = evaluate(model,splits['validation'],args,metadata,target_features,step_multiplier=2)
        result['solver_check']={'half_step_validation_macro_mse_z':fine['macro_mse_z'],
                                'absolute_metric_change':abs(fine['macro_mse_z']-result['validation']['macro_mse_z'])}
        result['elapsed_seconds']=time.monotonic()-started
        tracker.final(result)
        (output/'metrics.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'complete':str(output),'best_epoch':c['epoch'],
                         'validation':result['validation']['macro_mse_z'],
                         'test':result['test']['macro_mse_z'],'solver_check':result['solver_check']}),flush=True)


if __name__=='__main__':
    main()
