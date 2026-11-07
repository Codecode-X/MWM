"""Run a future-only training ablation beside the stopped weak-prior study."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .prepare import atomic
from .report import read
from .weak_prior_ablation_report import generate

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'mimic_experiment/results/weak_prior_ablation_cuda'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().split();return None if stat[2]=='Z' else stat[21]
    except FileNotFoundError:return None


def prepare(root,out):
    old=read(root/'manifest.json');control=old['config']
    cfg={**control,'train_reconstruct_history':False,'output':str(root/'no_history_reconstruction/seed_0'),'comparison_output':str(out)}
    differences={k for k in set(cfg)|set(control) if cfg.get(k)!=control.get(k)}
    if differences!={'train_reconstruct_history','output','comparison_output'}:raise RuntimeError('Unexpected configuration difference')
    if digest(cfg['data'])!=cfg['dataset_sha256']:raise RuntimeError('Dataset changed')
    source=out/'source_snapshot';original=Path(old['source_snapshot'])
    if not source.exists():
        shutil.copytree(original,source)
        shutil.copy2(ROOT/'mimic_experiment/train_future_only.py',source/'mimic_experiment/train_future_only.py')
        hashes={**old['source_sha256'],'mimic_experiment/train_future_only.py':digest(source/'mimic_experiment/train_future_only.py')}
        atomic(source/'source_hashes.json',hashes)
    hashes=read(source/'source_hashes.json')
    for name,sha in old['source_sha256'].items():
        if hashes[name]!=sha:raise RuntimeError('Original backbone or training loop changed')
    for name,sha in hashes.items():
        if digest(source/name)!=sha:raise RuntimeError('Frozen future-only source changed')
    manifest={'control':control['output'],'control_config':control,'config':cfg,'source_snapshot':str(source),'source_sha256':hashes,
              'training_change':'remove past generation and historical target supervision; Jacobian samples future generated states',
              'unchanged':'backbone, initialization, forward history encoding, dataset, seed, optimizer, lr schedule, full batch128, no accumulation, weak_prior=0, Jacobian0.01, L2 0.02',
              'evaluation':'original complete history and future protocol','checkpoint_selection':'original validation overall reconstruction error',
              'driver_source_sha256':{name:digest(ROOT/name) for name in ['mimic_experiment/history_reconstruction_ablation_study.py','mimic_experiment/weak_prior_ablation_report.py']}}
    if (out/'manifest.json').exists() and read(out/'manifest.json')!=manifest:raise RuntimeError('Preserve existing history ablation manifest')
    atomic(out/'manifest.json',manifest);atomic(out/'config.json',cfg);return source,cfg


def verify(root,out,cfg,stage):
    folder=out/'checks/forward_shared/smoke' if stage=='smoke' else Path(cfg['output']);m=read(folder/'metrics.json');h=read(folder/'hardware.json')
    if not m or m['status']!=('cuda_smoke_passed' if stage=='smoke' else 'complete') or m['checkpoint_reload_difference']>1e-6:raise RuntimeError('Incomplete CUDA stage')
    if h['parameters']!=43280 or h['trainable_parameters']!=43280 or h['prior_weight']!=0 or h['physical_gpu']!=3:raise RuntimeError('Model or device configuration changed')
    if h['first_actual_training_batch']!=128 or h['gradient_accumulation_steps']!=1:raise RuntimeError('Full batch128, no gradient accumulation required')
    if h['prediction_device']!='cuda:0' or set(h['actual_tensor_devices'].values())!={'cuda:0'}:raise RuntimeError('CUDA execution required')
    if stage=='training':
        ref=read(Path(read(root/'manifest.json')['reference'])/'metrics.json')
        for split in ['validation','test']:
            for region in ['interpolation','forecast','overall','prior','forecast_6h','forecast_12h','forecast_24h']:
                if m[split][region]['observed_scalars']!=ref[split][region]['observed_scalars']:raise RuntimeError('Common evaluation target counts differ')
    atomic(folder/'history_ablation_verification.json',{'status':'passed','no_past_training_generation':True,'future_only_data_loss':True,
           'training_jacobian_states':'future only','common_history_and_future_evaluation':True,'parameters':43280,'batch_size':128,
           'gradient_accumulation_steps':1,'physical_gpu':3,'device':'cuda:0'})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=str(DEFAULT));args=parser.parse_args()
    if Path(sys.prefix).name!='ode':raise RuntimeError('Use ode environment')
    root=Path(args.root).resolve();out=root/'history_reconstruction_study';out.mkdir(parents=True,exist_ok=True)
    if (out/'paused_by_user.json').exists():raise RuntimeError('Explicit resume required after pause')
    old=read(out/'queue.json',{})
    if old.get('pid') and identity(old['pid'])==old.get('start_time'):raise RuntimeError('History ablation already running')
    source,cfg=prepare(root,out);state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'waiting_for_gpu','phase':'waiting_for_gpu',
        'physical_gpu':3,'started_at':datetime.now(timezone.utc).isoformat()};atomic(out/'queue.json',state);generate(root)
    handle=None;worker=None
    def stop(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONPATH=str(source),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             SWANLAB_LOG_DIR=str(ROOT/'swanlog'),SWANLAB_EXPERIMENT_NAME='mimic-forward-future-only-no-prior-batch128-seed0')
    def run(command,folder,phase):
        nonlocal worker
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/'run.log').open('a') as log:
            worker=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='running',phase=phase,worker_pid=worker.pid);atomic(out/'queue.json',state)
            print(f'future-only training: {phase}, physical GPU3, PID {worker.pid}',flush=True);last=0.
            while worker.poll() is None:
                if time.monotonic()-last>=30:generate(root);last=time.monotonic()
                time.sleep(2)
            code=worker.returncode;worker=None
            if code:raise RuntimeError(f'{phase} exited {code}; inspect {folder / "run.log"}')
    try:
        while handle is None:
            candidate=Path('/tmp/latentode_gpu_3.lock').open('a')
            try:fcntl.flock(candidate.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:candidate.close();time.sleep(5);continue
            apps=subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if apps:candidate.close();time.sleep(5)
            else:handle=candidate
        if not (out/'cuda_checks.json').exists():
            run([sys.executable,'-u','-m','mimic_experiment.train_future_only','--check','--manifest',str(out/'manifest.json')],out/'checks/loss','cuda_checks')
        if read(out/'cuda_checks.json',{}).get('status')!='passed':raise RuntimeError('CUDA checks failed')
        for stage in ['smoke','training']:
            folder=out/'checks/forward_shared/smoke' if stage=='smoke' else Path(cfg['output'])
            if not (folder/'metrics.json').exists():run([sys.executable,'-u','-m','mimic_experiment.train_future_only','--config',str(out/'config.json'),'--stage',stage],folder,stage)
            verify(root,out,cfg,stage)
        state.update(status='complete',phase='tested',finished_at=datetime.now(timezone.utc).isoformat());atomic(out/'queue.json',state);generate(root)
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc),finished_at=datetime.now(timezone.utc).isoformat())
        atomic(out/'queue.json',state);generate(root);raise
    finally:
        if worker is not None and worker.poll() is None:worker.terminate();worker.wait()
        if handle is not None:handle.close()


if __name__=='__main__':main()
