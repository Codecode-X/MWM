"""Single-GPU weak_prior=0 run using the original full-batch training snapshot."""
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
REFERENCE=ROOT/'mimic_experiment/results/direction_full_batch_cuda/forward_shared/seed_0'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().split()
        return None if stat[2]=='Z' else stat[21]
    except FileNotFoundError:return None


def prepare(out):
    reference_config=read(REFERENCE.parents[1]/'configs/forward_shared.json')
    if not read(REFERENCE/'metrics.json') or not reference_config:raise RuntimeError('Missing completed reference')
    cfg={**reference_config,'prior_weight':0.,'output':str(out/'no_weak_prior/seed_0'),'comparison_output':str(out)}
    differences={key for key in set(reference_config)|set(cfg) if reference_config.get(key)!=cfg.get(key)}
    if differences!={'prior_weight','output','comparison_output'}:raise AssertionError('Only weak_prior weight may change')
    if digest(cfg['data'])!=cfg['dataset_sha256']:raise RuntimeError('Dataset changed')
    original=REFERENCE.parents[1]/'source_snapshot';source=out/'source_snapshot'
    if not source.exists():shutil.copytree(original,source)
    hashes=read(original/'source_hashes.json')
    for name,sha in hashes.items():
        if digest(source/name)!=sha:raise RuntimeError('Full-batch training code differs from reference')
    manifest={'reference':str(REFERENCE),'reference_config':reference_config,'config':cfg,'source_snapshot':str(source),
              'source_sha256':hashes,'changed_training_factor':{'prior_weight':[.01,0.]},
              'controller_source_sha256':{name:digest(ROOT/name) for name in ['mimic_experiment/weak_prior_ablation_study.py','mimic_experiment/weak_prior_ablation_checks.py','mimic_experiment/weak_prior_ablation_report.py']},
              'checkpoint_selection':'same validation overall reconstruction error','seed':0,'physical_gpu':3,
              'trajectory_comparison':'compare common completed epochs; retain reference min50/max300/patience40'}
    if (out/'manifest.json').exists() and read(out/'manifest.json')!=manifest:raise RuntimeError('Preserve existing ablation manifest')
    atomic(out/'manifest.json',manifest);atomic(out/'config.json',cfg);return source,cfg


def verify(out,cfg,stage):
    folder=out/'checks/forward_shared/smoke' if stage=='smoke' else Path(cfg['output'])
    m=read(folder/'metrics.json');h=read(folder/'hardware.json')
    if not m or m['status']!=('cuda_smoke_passed' if stage=='smoke' else 'complete') or m['checkpoint_reload_difference']>1e-6:raise RuntimeError('Missing completed CUDA stage')
    if h['parameters']!=43280 or h['trainable_parameters']!=43280 or h['prior_weight']!=0:raise RuntimeError('Loss coefficient or parameter count changed')
    if h['physical_gpu']!=3 or h['training_batch_size']!=128 or h['first_actual_training_batch']!=128 or h['gradient_accumulation_steps']!=1:raise RuntimeError('A complete single-GPU batch128 is required')
    if h['parameter_device']!='cuda:0' or h['prediction_device']!='cuda:0' or set(h['actual_tensor_devices'].values())!={'cuda:0'}:raise RuntimeError('All model tensors must execute on CUDA')
    if stage=='training':
        ref=read(REFERENCE/'metrics.json')
        for split in ['validation','test']:
            if m[split]['patients']!=ref[split]['patients']:raise RuntimeError('Patient counts differ')
            for region in ['interpolation','forecast','overall','prior','forecast_6h','forecast_12h','forecast_24h']:
                if m[split][region]['observed_scalars']!=ref[split][region]['observed_scalars']:raise RuntimeError('Evaluation target counts differ')
        atomic(folder/'completion_verification.json',{'status':'passed','same_evaluation_counts':True,'parameters':43280,
               'batch_size':128,'gradient_accumulation_steps':1,'prior_weight':0.,'physical_gpu':3,'device':'cuda:0'})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(DEFAULT));args=parser.parse_args()
    if Path(sys.prefix).name!='ode':raise RuntimeError('Use ode environment')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'paused_by_user.json').exists():raise RuntimeError('Explicit resume required after user pause')
    previous=read(out/'queue.json',{})
    if previous.get('pid') and identity(previous['pid'])==previous.get('start_time'):raise RuntimeError('Weak-prior ablation is already running')
    source,cfg=prepare(out);state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'waiting_for_gpu',
         'phase':'waiting_for_gpu','physical_gpu':3,'prior_weight':0.,'started_at':datetime.now(timezone.utc).isoformat()}
    atomic(out/'queue.json',state);generate(out)
    handle=None;worker=None
    def stop(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONPATH=str(source),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             SWANLAB_LOG_DIR=str(ROOT/'swanlog'),SWANLAB_EXPERIMENT_NAME='mimic-forward-no-weak-prior-batch128-seed0')
    def run(command,folder,phase):
        nonlocal worker
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/'run.log').open('a') as log:
            worker=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='running',phase=phase,worker_pid=worker.pid);atomic(out/'queue.json',state)
            print(f'weak_prior=0: {phase}, physical GPU3, PID {worker.pid}',flush=True);last=0.
            while worker.poll() is None:
                if time.monotonic()-last>=30:generate(out);last=time.monotonic()
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
        if not (out/'cuda_loss_check.json').exists():
            run([sys.executable,'-u',str(ROOT/'mimic_experiment/weak_prior_ablation_checks.py'),'--manifest',str(out/'manifest.json'),'--gpu','3'],out/'checks/loss','loss_check')
        if read(out/'cuda_loss_check.json',{}).get('status')!='passed':raise RuntimeError('CUDA loss check failed')
        for stage in ['smoke','training']:
            folder=out/'checks/forward_shared/smoke' if stage=='smoke' else Path(cfg['output'])
            if not (folder/'metrics.json').exists():
                run([sys.executable,'-u','-m','mimic_experiment.train_full_batch','--config',str(out/'config.json'),'--stage',stage],folder,stage)
            verify(out,cfg,stage)
        state.update(status='complete',phase='tested',finished_at=datetime.now(timezone.utc).isoformat());atomic(out/'queue.json',state);generate(out)
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc),finished_at=datetime.now(timezone.utc).isoformat())
        atomic(out/'queue.json',state);generate(out);raise
    finally:
        if worker is not None and worker.poll() is None:worker.terminate();worker.wait()
        if handle is not None:handle.close()


if __name__=='__main__':main()
