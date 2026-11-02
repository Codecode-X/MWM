"""Evaluate a paused direction-study checkpoint on its frozen CUDA evaluator."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import torch

from mimic_experiment.evaluation import evaluate
from mimic_experiment.prepare import atomic
from mimic_experiment.train_baseline import normalization_and_modes
from mimic_experiment.train_full_batch import make_model
from paper_odeae.train import seed_all


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--epoch',type=int,required=True);parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args();run=Path(args.run).resolve();study=run.parents[1];source=study/'source_snapshot'
    if Path(sys.prefix).name!='ode' or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu) or not torch.cuda.is_available():
        raise RuntimeError('Use ode and the assigned physical CUDA GPU')
    if not (run/'paused_by_user.json').exists():raise RuntimeError('This entry point evaluates paused runs only')
    if not Path(inspect.getfile(make_model)).resolve().is_relative_to(source):raise RuntimeError('Use the original frozen direction-study code')
    for name,sha in json.loads((source/'source_hashes.json').read_text()).items():
        if digest(source/name)!=sha:raise RuntimeError('Frozen direction-study code changed')
    lease=Path(f'/tmp/latentode_gpu_{args.gpu}.lock').open('a')
    fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    apps=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if apps:raise RuntimeError('Assigned GPU already has a CUDA process')
    torch.set_num_threads(1);torch.cuda.set_device(0);seed_all(0);started=time.monotonic()
    folder=run/f'checkpoint_evaluation_epoch_{args.epoch}';folder.mkdir(exist_ok=True)
    if (folder/'metrics.json').exists():raise RuntimeError('Checkpoint already evaluated')
    checkpoint=folder/'checkpoint.pt';shutil.copy2(run/'best.pt',checkpoint);checkpoint_sha=digest(checkpoint)
    saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
    if saved['epoch']!=args.epoch:raise RuntimeError('Best checkpoint epoch does not match requested epoch')
    cfg=saved['config'];raw=Path(cfg['data']).read_bytes();sha=hashlib.sha256(raw).hexdigest()
    if sha!=cfg['dataset_sha256']:raise RuntimeError('Prepared data changed')
    payload=pickle.loads(raw);del raw
    if payload['schema']!=saved['schema']:raise RuntimeError('Checkpoint and dataset schemas differ')
    meta=payload['metadata'];schema=payload['schema']
    means,modes=normalization_and_modes(payload);cfg={**cfg,'raw_scale':meta['scale'],'normalized_train_means':means}
    model=make_model(schema,cfg).cuda();model.load_state_dict(saved['state_dict'],strict=True);model.eval()
    if sum(p.numel() for p in model.parameters())!=43280 or model.backward_ode is not model.forward_ode:raise RuntimeError('Model architecture changed')
    model.requires_grad_(False)
    hardware={'physical_gpu':args.gpu,'gpu_name':torch.cuda.get_device_name(),'parameter_device':str(next(model.parameters()).device),
              'parameters':sum(p.numel() for p in model.parameters()),'environment':Path(sys.prefix).name,'torch_version':torch.__version__,
              'checkpoint_epoch':saved['epoch'],'checkpoint_sha256':checkpoint_sha,'dataset_sha256':sha,'source_snapshot':str(source),
              'driver_sha256':digest(__file__),'evaluation_batch_size':32,'evaluation_protocol':'same four fixed patient shards; 24h history; targets through 48h',
              'model_training':False,'gradient_enabled':False,'optimizer_steps_performed':0}
    original=model.forward
    def recorded(batch,sample=False):
        result=original(batch,sample=sample)
        if 'actual_tensor_devices' not in hardware:
            hardware['actual_tensor_devices']={k:str(v.device) for k,v in batch.items() if torch.is_tensor(v)}
            hardware['prediction_device']=str(result['prediction'].device)
            if set(hardware['actual_tensor_devices'].values())!={'cuda:0'} or hardware['prediction_device']!='cuda:0':raise RuntimeError('All model tensors must be CUDA')
            atomic(folder/'hardware.json',hardware)
        return result
    model.forward=recorded
    atomic(folder/'hardware.json',hardware)
    history=json.loads((run/'history.json').read_text());expected=next(row for row in history if row['epoch']==args.epoch)
    splits={k:[s for s in payload['sequences'] if s['split']==k] for k in ['validation','test']}
    reference=json.loads((study/'forward_shared/seed_0/metrics.json').read_text())
    results={}
    with torch.no_grad():
        for split in ['validation','test']:
            results[split]=evaluate(model,splits[split],schema,cfg,meta['raw_train_std'],modes,horizon=24)
            for region in ['interpolation','forecast','overall']:
                if results[split][region]['observed_scalars']!=reference[split][region]['observed_scalars']:raise RuntimeError('Evaluation targets differ from the forward comparison')
            print(json.dumps({'evaluated':split,'epoch':args.epoch,'overall':results[split]['overall']['reconstruction_error']}),flush=True)
    discrepancy=abs(results['validation']['overall']['reconstruction_error']-expected['validation']['overall']['reconstruction_error'])
    if discrepancy>1e-6:raise RuntimeError('Reloaded checkpoint does not reproduce stored epoch128 validation')
    if digest(run/'best.pt')!=checkpoint_sha:raise RuntimeError('Training checkpoint changed during evaluation')
    metrics={'status':'checkpoint_evaluated_while_paused','training_status':'paused_by_user','best_epoch':args.epoch,
             'epochs_run':len(history),'checkpoint_selection':'stored best validation checkpoint at pause',
             'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_sha,'evaluation_physical_gpu':args.gpu,
             'checkpoint_reload_difference':discrepancy,'same_evaluation_target_counts':True,
             'evaluated_at':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,**results}
    atomic(folder/'metrics.json',metrics)
    print(json.dumps({'status':metrics['status'],'epoch':args.epoch,'test':{k:results['test'][k]['reconstruction_error'] for k in ['interpolation','forecast','overall']},
                      'standardized_numeric_mse':results['test']['mean_standardized_numeric_mse'],'checkpoint_reload_difference':discrepancy}),flush=True)


if __name__=='__main__':main()
