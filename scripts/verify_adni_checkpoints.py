#!/usr/bin/env python3
"""Reload finished checkpoints on their assigned CUDA GPU and verify metrics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


def verify_one(path):
    import hashlib
    import torch
    from src.adni_train import load_data, collate, to_cuda, evaluate
    from src.adni_models import make_model, forward
    output=Path(path)
    args=argparse.Namespace(**json.loads((output/'config.json').read_text()))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA checkpoint evaluation required')
    torch.set_num_threads(1)
    cache=Path(args.cache)
    hw=json.loads((output/'hardware.json').read_text())
    if hashlib.sha256((cache/'prepared.pkl').read_bytes()).hexdigest()!=hw['prepared_sha256']:
        raise ValueError('dataset cache changed after training')
    checkpoint=torch.load(output/'best.pt',map_location='cuda',weights_only=False)
    if args.imaging and checkpoint.get('projection') is None:
        raise ValueError('MRI checkpoint is missing its train-fitted PCA')
    splits,meta,f,projection=load_data(cache,args.imaging,projection=checkpoint.get('projection'))
    meta=checkpoint['metadata']
    model=make_model(args.model,splits['train'][0]['values'].shape[-1],args.latent,args.hidden,args.width).cuda().eval()
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    batch=to_cuda(collate(splits['validation'][:args.batch_size],target_features=f))
    with torch.inference_mode():
        prediction=forward(model,batch,args.model,samples=args.eval_samples)['prediction']
    actual={'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],
            'model_device':str(next(model.parameters()).device),
            'actual_validation_tensor_devices':{k:str(v.device) for k,v in batch.items() if torch.is_tensor(v)},
            'prediction_device':str(prediction.device),
            'gpu_name':torch.cuda.get_device_name(),
            'forecast_network_parameters':sum(v.numel() for k,v in model.named_parameters() if not k.startswith('classifier.'))}
    original=json.loads((output/'metrics.json').read_text())
    checks={}
    for split in ['validation','test']:
        metrics=evaluate(model,splits[split],args,meta,f)
        change=abs(metrics['macro_mse_z']-original[split]['macro_mse_z'])
        if change>1e-6:
            raise ValueError(f'checkpoint metric mismatch {output} {split}: {change}')
        checks[split]={'macro_mse_z':metrics['macro_mse_z'],'absolute_reload_difference':change}
        original[split]['per_horizon']=metrics['per_horizon']
    actual['metric_verification']=checks
    actual['checkpoint_epoch']=checkpoint['epoch']
    (output/'checkpoint_verification.json').write_text(json.dumps(actual,indent=2))
    (output/'metrics.json').write_text(json.dumps(original,indent=2))
    hw['forecast_network_parameters']=actual['forecast_network_parameters']
    (output/'hardware.json').write_text(json.dumps(hw,indent=2))
    print(str(output),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='results/adni_cuda')
    p.add_argument('--one')
    args=p.parse_args()
    if args.one:
        verify_one(args.one)
        return
    root=Path(args.root)
    jobs={str(i):[] for i in range(4)}
    for f in sorted(root.glob('*/*/seed_*/metrics.json')):
        if 'smoke' in str(f):
            continue
        hw=json.loads((f.parent/'hardware.json').read_text())
        jobs[hw['physical_gpu']].append(str(f.parent))
    def worker(gpu,paths):
        for path in paths:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=str(Path.cwd()),OPENBLAS_NUM_THREADS='1')
            subprocess.run([sys.executable,'-u',__file__,'--one',path],env=env,check=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,gpu,paths) for gpu,paths in jobs.items()]
        for future in futures:
            future.result()


if __name__=='__main__':
    main()
