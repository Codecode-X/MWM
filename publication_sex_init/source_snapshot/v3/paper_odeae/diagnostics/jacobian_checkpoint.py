"""Evaluate an existing shared_matched checkpoint with exact Jacobian metrics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import torch
from paper_odeae.models import paper_model_from_config
from paper_odeae.train import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    if not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') not in ['0','1','2','3']:raise RuntimeError('one physical CUDA GPU required')
    torch.set_num_threads(1);path=Path(args.checkpoint)
    saved=torch.load(path,map_location='cuda',weights_only=False)
    cfg={**saved['config'],'collect_jacobian_states':True,'collect_prior_predictions':True,'jacobian_weight':0.,'prior_weight':0.}
    model=paper_model_from_config(saved['schema'],cfg).cuda().eval();model.load_state_dict(saved['state_dict'])
    payload=pickle.loads(Path(cfg['data']).read_bytes())
    results={k:evaluate(model,[s for s in payload['sequences'] if s['split']==k],saved['schema'],cfg['batch_size']) for k in ['validation','test']}
    original=json.loads((path.parent/'metrics.json').read_text())
    difference=max(abs(results[split][region]['reconstruction_error']-original[split][region]['reconstruction_error']) for split in ['validation','test'] for region in ['interpolation','forecast','overall'])
    if difference>1e-6:raise RuntimeError(f'checkpoint evaluation changed: {difference}')
    results.update(checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],device=str(next(model.parameters()).device),existing_metric_difference=difference,training_performed=False)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'physical_gpu':results['physical_gpu'],'validation_jacobian':results['validation']['dynamics'],'test_jacobian':results['test']['dynamics'],'metric_difference':difference}),flush=True)


if __name__=='__main__':main()
