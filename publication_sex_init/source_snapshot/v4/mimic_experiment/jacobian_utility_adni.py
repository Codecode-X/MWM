"""Fixed-checkpoint ADNI tests of what Jacobian regularization changes.

All neural-network, ODE, tangent-flow and sensitivity calculations use CUDA.
CPU operations are limited to data I/O, patient bootstrap and plotting.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from paper_odeae.data import Prefixes, collate
from paper_odeae.models import encode_values, feature_errors
from paper_odeae.train import device_batch, seed_all
from paper_odeae.solver import integrate, solver_statistics
from component_ablation_model import make_model


REGIONS = ['all_future', '0-12m', '12-24m', '24-36m', '36-60m', '>60m']
YEARS = [0., 1., 2., 3., 5.]


def atomic(path, value):
    tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def progress(out, stage, **extra):
    value = {'status':'running', 'stage':stage, 'updated_at_utc':datetime.now(timezone.utc).isoformat(), **extra}
    atomic(out/'progress.json', value); print(json.dumps(value), flush=True)


def numeric_statistics(payload):
    schema = payload['schema']; means = []; std = []
    train = [s for s in payload['sequences'] if s['split']=='train']
    for j, field in enumerate(schema):
        values = np.concatenate([s['values'][:, j][s['mask'][:, j].astype(bool)] for s in train]).astype(np.float64)
        means.append(float(values.mean())); std.append(float(values.std()))
    return np.array(means), np.array(std)


def output_indices(schema):
    result = {}; at = 0
    for f in schema:
        result[f['name']] = at; at += f.get('classes', 1)
    return result


@torch.no_grad()
def forecast_test(model, sequences, schema, std, sigma, out, label):
    """Native prefix order, original targets, noise in observed historical inputs only."""
    path = out/f'forecast_{label}_{sigma:g}.npz'
    if path.exists():
        return dict(np.load(path, allow_pickle=False))
    ids = [s['subject_id'] for s in sequences]; idmap = {pid:i for i,pid in enumerate(ids)}
    f = len(schema); n = len(ids)
    sums = torch.zeros(n, len(REGIONS), f, device='cuda', dtype=torch.float64)
    counts = torch.zeros_like(sums); shifts = torch.zeros_like(sums)
    std_cuda = torch.tensor(std, device='cuda', dtype=torch.float32)
    noise_fields = torch.tensor([x['kind']=='numerical' and x['name'] not in ['age','education_years','apoe4_count'] for x in schema], device='cuda')
    generator = torch.Generator(device='cuda').manual_seed(74821)
    at = output_indices(schema)
    loader = DataLoader(Prefixes(sequences), batch_size=32, shuffle=False, collate_fn=collate)
    for cpu in loader:
        b = device_batch(cpu)
        clean = model(b, sample=False)
        if sigma:
            noise = torch.randn(b['values'].shape, device='cuda', generator=generator)*std_cuda*sigma
            noisy = {**b, 'values':b['values']+noise*b['context_mask']*noise_fields}
            prediction = model(noisy, sample=False)['prediction']
        else:
            prediction = clean['prediction']
        errors = feature_errors(prediction, b['values'], b['mask'], schema)
        anchor = b['ages'].gather(1, (b['cutoff']-1)[:,None])
        elapsed = b['ages']-anchor
        index = torch.arange(b['ages'].shape[1], device='cuda')[None]
        future = (index>=b['cutoff'][:,None]) & b['valid']
        regions = [future, future & (elapsed<=1), future & (elapsed>1) & (elapsed<=2),
                   future & (elapsed>2) & (elapsed<=3), future & (elapsed>3) & (elapsed<=5), future & (elapsed>5)]
        delta = torch.zeros_like(errors)
        for j, field in enumerate(schema):
            if field['kind']=='numerical':
                delta[...,j] = (prediction[...,at[field['name']]]-clean['prediction'][...,at[field['name']]]).square()*b['mask'][...,j]
        batch_ids = torch.tensor([idmap[pid] for pid in b['subject_ids']], device='cuda')
        for r, mask in enumerate(regions):
            sums[:,r].index_add_(0, batch_ids, (errors*mask[...,None]).sum(1).double())
            counts[:,r].index_add_(0, batch_ids, (b['mask']*mask[...,None]).sum(1).double())
            shifts[:,r].index_add_(0, batch_ids, (delta*mask[...,None]).sum(1).double())
    result = {'sums':sums.cpu().numpy(), 'counts':counts.cpu().numpy(), 'shifts':shifts.cpu().numpy()}
    np.savez_compressed(path, **result)
    return result


@torch.no_grad()
def encode(model, b):
    inputs = encode_values(b['values'], b['context_mask'], model.schema)
    last_age = b['ages'].gather(1,(b['cutoff']-1)[:,None]).squeeze(1)
    h = model.init(last_age[:,None]/100.); previous = b['ages'][:,0]
    for j in range(b['max_context']):
        idx = torch.minimum(torch.full_like(b['cutoff'],j), b['cutoff']-1)
        active = j<b['cutoff']; age = b['ages'].gather(1,idx[:,None]).squeeze(1)
        evolved = model.solve(model.backward_ode, h, torch.where(active,age-previous,torch.zeros_like(age)))
        x = inputs[torch.arange(len(h),device='cuda'),idx]
        h = torch.where(active[:,None],model.update(torch.cat([evolved,x],-1)),h)
        previous = torch.where(active,age,previous)
    return h


def field_jacobian(field, z):
    hidden = field[0](z).tanh()
    return torch.einsum('dh,bh,he->bde', field[2].weight, 1-hidden.square(), field[0].weight)


def tangent(field, z, years, rtol=1e-8, atol=1e-10):
    d = z.shape[-1]; eye = torch.eye(d, device='cuda', dtype=z.dtype).expand(len(z),d,d)
    augmented = torch.cat([z,eye.reshape(len(z),-1)],-1)
    def vector(y):
        state = y[:,:d]; matrix = y[:,d:].reshape(-1,d,d)
        return torch.cat([field(state), torch.bmm(field_jacobian(field,state),matrix).reshape(len(y),-1)],-1)
    with solver_statistics() as statistics:
        result = integrate(vector, augmented, z.new_full((len(z),),years),rtol,atol)
    return result[:,:d], result[:,d:].reshape(-1,d,d), statistics['calls']


@torch.no_grad()
def flow_test(model, sequences, schema, out, label):
    path = out/f'flow_{label}.npz'
    if path.exists():
        return dict(np.load(path, allow_pickle=False))
    states = []; encoding_diff = 0.
    items = [(s,min(3,len(s['ages']))) for s in sequences]
    for start in range(0,len(items),32):
        b = device_batch(collate(items[start:start+32])); z = encode(model,b)
        if start==0:
            encoding_diff = float((z-model(b,sample=False)['mu']).abs().max()); assert encoding_diff<1e-6
        states.append(z)
    z = torch.cat(states).double()
    precise = copy.deepcopy(model).double().eval(); field = precise.forward_ode
    jac = field_jacobian(field,z)
    local_frob = jac.square().sum((1,2))
    local_spectral = torch.linalg.svdvals(jac)[:,0]
    local_expansion = torch.linalg.eigvalsh((jac+jac.transpose(1,2))*.5)[:,-1]
    # Exact analytic df/dz checked against CUDA autograd at the same states.
    with torch.enable_grad():
        x = z[:3].detach().requires_grad_(True); y = field(x)
        autograd_j = torch.stack([torch.autograd.grad(y[:,j].sum(),x,retain_graph=True)[0] for j in range(28)],1)
    analytic_error = float((autograd_j-jac[:3]).abs().max()); assert analytic_error<1e-10
    gain = []; state_paths = []; mmse = []; cdrsb = []; shifts = []; stats = []
    indices = output_indices(schema)
    scales = model._diagnostic_scale
    generator = torch.Generator(device='cuda').manual_seed(71683)
    directions = torch.randn(len(z),8,28,device='cuda',dtype=torch.float64,generator=generator)
    directions /= directions.norm(dim=-1,keepdim=True)
    delta = .01*z.norm(dim=-1).clamp_min(1e-8)[:,None,None]*directions
    perturbed = z[:,None]+delta
    for years in YEARS:
        all_end = []; all_gain = []
        for start in range(0,len(z),32):
            endpoint, matrix, calls = tangent(field,z[start:start+32],years)
            all_end.append(endpoint); all_gain.append(torch.linalg.svdvals(matrix)[:,0]); stats.extend(calls)
        endpoint = torch.cat(all_end); state_paths.append(endpoint); gain.append(torch.cat(all_gain))
        prediction = precise.decoder(endpoint)
        mmse.append(prediction[:,indices['mmse']]*scales['mmse'])
        cdrsb.append(prediction[:,indices['cdrsb']]*scales['cdrsb'])
        noisy_end = []
        flat = perturbed.reshape(-1,28)
        for start in range(0,len(flat),256):
            x = flat[start:start+256]
            noisy_end.append(integrate(field,x,x.new_full((len(x),),years),1e-8,1e-10))
        noisy = torch.cat(noisy_end).reshape(len(z),8,28)
        noisy_prediction = precise.decoder(noisy)
        shift_mmse = (noisy_prediction[...,indices['mmse']]-prediction[:,None,indices['mmse']])*scales['mmse']
        shift_cdr = (noisy_prediction[...,indices['cdrsb']]-prediction[:,None,indices['cdrsb']])*scales['cdrsb']
        shifts.append(torch.stack([(noisy-endpoint[:,None]).norm(dim=-1)/delta.norm(dim=-1), shift_mmse.abs(), shift_cdr.abs()],-1))
        progress(out,'flow',model=label,years=years)
    # Tolerance convergence and an independent autograd derivative of the original 28D flow.
    endpoint, matrix, _ = tangent(field,z[:8],5.)
    endpoint2, matrix2, _ = tangent(field,z[:8],5.,1e-10,1e-12)
    relative_endpoint = float((endpoint-endpoint2).norm()/endpoint2.norm())
    relative_matrix = float((matrix-matrix2).norm()/matrix2.norm())
    with torch.enable_grad():
        x = z[:8].detach().requires_grad_(True)
        y = integrate(field,x,x.new_full((len(x),),5.),1e-9,1e-11)
        original_j = torch.stack([torch.autograd.grad(y[:,j].sum(),x,retain_graph=True)[0] for j in range(28)],1)
    relative_autograd = float((original_j-matrix2).norm()/matrix2.norm())
    assert max(relative_endpoint,relative_matrix,relative_autograd)<1e-4
    checks = {'encoding_max_absolute_error':encoding_diff,'field_jacobian_max_absolute_error':analytic_error,
              'five_year_tolerance_endpoint_relative_error':relative_endpoint,
              'five_year_tolerance_flow_jacobian_relative_error':relative_matrix,
              'five_year_original_flow_autograd_relative_error':relative_autograd,
              'rtol':1e-8,'atol':1e-10,'reference_rtol':1e-10,'reference_atol':1e-12,
              'dtype':'float64','physical_gpu':int(os.environ['CUDA_VISIBLE_DEVICES']),
              'solver_calls':len(stats),'nfe':sum(x['nfe'] for x in stats),
              'accepted_steps':sum(x['accepted'] for x in stats),'rejected_steps':sum(x['rejected'] for x in stats)}
    atomic(out/f'flow_checks_{label}.json',checks)
    result = {'cutoff_states':z.cpu().numpy(),'local_frob_squared':local_frob.cpu().numpy(),
              'local_spectral':local_spectral.cpu().numpy(),'local_expansion':local_expansion.cpu().numpy(),
              'flow_gain':torch.stack(gain,1).cpu().numpy(),'paths':torch.stack(state_paths,1).cpu().numpy(),
              'mmse':torch.stack(mmse,1).cpu().numpy(),'cdrsb':torch.stack(cdrsb,1).cpu().numpy(),
              'perturbation':torch.stack(shifts,1).cpu().numpy(),'cutoff_norm':z.norm(dim=-1).cpu().numpy()}
    np.savez_compressed(path,**result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--gpu',type=int,required=True);args=parser.parse_args()
    out=Path(args.output);manifest=json.loads((out/'manifest.json').read_text())
    assert Path(sys.prefix).name=='ode' and torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES')==str(args.gpu)
    torch.set_num_threads(1);torch.cuda.set_device(0);seed_all(0)
    raw=Path(manifest['data']).read_bytes();assert hashlib.sha256(raw).hexdigest()==manifest['dataset_sha256']
    payload=pickle.loads(raw);schema=payload['schema'];sequences=[s for s in payload['sequences'] if s['split']=='test']
    _,std=numeric_statistics(payload)
    atomic(out/'hardware.json',{'physical_gpu':args.gpu,'parameter_device':'cuda:0','gpu_name':torch.cuda.get_device_name(),
                              'model_computation':'CUDA only','concurrent_with_training':True,'optimizer_steps':0,
                              'forecast_dtype':'float32','flow_dtype':'float64'})
    for label,entry in manifest['models'].items():
        progress(out,'loading',model=label)
        checkpoint=out/entry['checkpoint'];assert sha(checkpoint)==entry['checkpoint_sha256']
        saved=torch.load(checkpoint,map_location='cuda',weights_only=False);assert saved['epoch']==30
        cfg=json.loads((out/entry['config']).read_text());cfg['ablation_component']='full_main' if label=='regularized' else 'no_jacobian'
        model=make_model(schema,cfg,.1).eval();model.load_state_dict(saved['state_dict'])
        assert sum(p.numel() for p in model.parameters())==44875
        model._diagnostic_scale={f['name']:payload['metadata']['scale'][j] for j,f in enumerate(schema)}
        before=[p.detach().clone() for p in model.parameters()]
        for sigma in [0.,.01,.05]:
            progress(out,'forecast_noise',model=label,noise_sigma=sigma)
            forecast_test(model,sequences,schema,std,sigma,out,label)
        flow_test(model,sequences,schema,out,label)
        assert all(torch.equal(a,b) for a,b in zip(before,model.parameters()))
        del model,saved,before;torch.cuda.empty_cache()
    # Statistical summaries do not execute the neural networks.
    from jacobian_utility_report import generate
    generate(out,payload)
    atomic(out/'progress.json',{'status':'complete','updated_at_utc':datetime.now(timezone.utc).isoformat(),'report':str(out/'report.zh-CN.md')})
    print(json.dumps({'complete':str(out)}),flush=True)


if __name__=='__main__':main()
