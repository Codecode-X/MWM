"""Train multi-cutoff forecast A and masked-retrospection B on separate GPUs."""
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
    completed=read(root/'history_reconstruction_study/manifest.json');base=completed['config']
    if not read(Path(base['output'])/'metrics.json') or base['prior_weight']!=0 or base['train_reconstruct_history'] is not False:
        raise RuntimeError('Use the completed weak_prior=0, future-only reference')
    if digest(base['data'])!=base['dataset_sha256']:raise RuntimeError('Dataset changed')
    original=Path(completed['source_snapshot']);source=out/'source_snapshot'
    if not source.exists():
        shutil.copytree(original,source);shutil.copy2(ROOT/'mimic_experiment/supervision_ablation.py',source/'mimic_experiment/supervision_ablation.py')
        hashes={**completed['source_sha256'],'mimic_experiment/supervision_ablation.py':digest(source/'mimic_experiment/supervision_ablation.py')}
        atomic(source/'source_hashes.json',hashes)
    hashes=read(source/'source_hashes.json')
    for name,sha in completed['source_sha256'].items():
        if hashes.get(name)!=sha:raise RuntimeError('Existing backbone, future objective or trainer changed')
    for name,sha in hashes.items():
        if digest(source/name)!=sha:raise RuntimeError('Frozen supervision source changed')
    jobs=[]
    for mode,name,gpu in [('A','multi_cutoff_forecast',3),('B','masked_historical_retrospection',1)]:
        cfg={**base,'supervision_ablation':mode,'physical_gpu':gpu,'output':str(root/name/'seed_0'),'comparison_output':str(out/mode)}
        if mode=='A':cfg.update(additional_cutoffs=3,span_boundaries_hours=[6.,12.],span_weights=[1.,1.,1.])
        else:cfg['masked_retrospection_weight']=1.
        differences={key:[base.get(key),cfg.get(key)] for key in set(base)|set(cfg) if base.get(key)!=cfg.get(key)}
        allowed={'supervision_ablation','physical_gpu','output','comparison_output','additional_cutoffs','span_boundaries_hours','span_weights','masked_retrospection_weight'}
        if set(differences)-allowed:raise RuntimeError('Unexpected model or optimizer change')
        path=out/'configs'/f'{mode}.json';path.parent.mkdir(exist_ok=True)
        if path.exists() and read(path)!=cfg:raise RuntimeError('Preserve existing configuration')
        atomic(path,cfg);Path(cfg['output']).mkdir(parents=True,exist_ok=True)
        jobs.append({'id':mode,'name':name,'physical_gpu':gpu,'config':cfg,'config_path':str(path),'configuration_differences':differences})
    manifest={'reference':base['output'],'reference_config':base,'source_snapshot':str(source),'source_sha256':hashes,'jobs':jobs,
              'parameters':43280,'latent':28,'batch_size':128,'gradient_accumulation_steps':1,'seed':0,
              'A_cutoff_distribution':'uniform over known history prefixes; final always included with weight1/K; up to3 uniformly sampled internal prefixes share remaining mass',
              'A_target_reduction':'within patient/cutoff/span/field observed-target mean; weighted mean over valid anchors per field; equal span mean',
              'A_regularization':'original final-cutoff future Jacobian0.01 and ODE L2 0.02',
              'B_masking':'uniformly select one internal event per eligible patient; skip its UPDATE and time step; preserve full original history in primary future branch',
              'B_weight':1.,'B_regularization':'unchanged original future branch Jacobian0.01 and ODE L2 0.02; no extra past Jacobian',
              'checkpoint_selection':'original validation overall reconstruction error','fresh_training':True,
              'driver_sha256':{name:digest(ROOT/name) for name in ['mimic_experiment/supervision_ablation_study.py','mimic_experiment/supervision_ablation_checks.py','mimic_experiment/weak_prior_ablation_report.py']}}
    if (out/'manifest.json').exists() and read(out/'manifest.json')!=manifest:raise RuntimeError('Preserve existing study manifest')
    atomic(out/'manifest.json',manifest);return manifest


def folder(job,stage):
    cfg=job['config'];return Path(cfg['output']) if stage=='training' else Path(cfg['comparison_output'])/'checks/forward_shared/smoke'


def verify(manifest,job,stage):
    path=folder(job,stage);m=read(path/'metrics.json');h=read(path/'hardware.json');cfg=job['config']
    if not m or m['status']!=('complete' if stage=='training' else 'cuda_smoke_passed') or m['checkpoint_reload_difference']>1e-6:raise RuntimeError('Incomplete CUDA stage')
    if h['parameters']!=43280 or h['trainable_parameters']!=43280 or h['prior_weight']!=0 or h['supervision_ablation']!=job['id']:raise RuntimeError('Model or supervision configuration mismatch')
    if h['physical_gpu']!=job['physical_gpu'] or h['first_actual_training_batch']!=128 or h['gradient_accumulation_steps']!=1:raise RuntimeError('Require one complete batch128 per assigned GPU')
    if h['prediction_device']!='cuda:0' or set(h['actual_tensor_devices'].values())!={'cuda:0'}:raise RuntimeError('CUDA computations required')
    history=read(path/'history.json')
    for row in history:
        c=row['training_components']
        expected=c['mse']+c['cce']+cfg['jacobian_weight']*c['jacobian_frobenius_squared']+cfg['l2_weight']*c['ode_l2']
        if abs(expected-row['training_loss'])>1e-5:raise RuntimeError('Recorded loss components do not sum to training objective')
    if not read(path/'supervision_epoch_diagnostics.json'):raise RuntimeError('Missing per-epoch supervision diagnostics')
    if stage=='training':
        ref=read(Path(manifest['reference'])/'metrics.json')
        for split in ['validation','test']:
            for region in ['interpolation','forecast','overall','prior','forecast_6h','forecast_12h','forecast_24h']:
                if m[split][region]['observed_scalars']!=ref[split][region]['observed_scalars']:raise RuntimeError('Common evaluation target counts differ')
        atomic(path/'completion_verification.json',{'status':'passed','supervision':job['id'],'parameters':43280,'batch_size':128,
               'gradient_accumulation_steps':1,'physical_gpu':h['physical_gpu'],'same_evaluation_counts':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=str(DEFAULT));args=parser.parse_args()
    if Path(sys.prefix).name!='ode':raise RuntimeError('Use ode environment')
    root=Path(args.root).resolve();out=root/'supervision_study';out.mkdir(parents=True,exist_ok=True)
    if (out/'paused_by_user.json').exists():raise RuntimeError('Explicit resume required after user pause')
    previous=read(out/'queue.json',{})
    if previous.get('pid') and identity(previous['pid'])==previous.get('start_time'):raise RuntimeError('Supervision ablation already running')
    manifest=prepare(root,out);source=Path(manifest['source_snapshot']);jobs={j['id']:j for j in manifest['jobs']}
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'waiting_for_gpu','started_at':datetime.now(timezone.utc).isoformat(),
           'jobs':{key:{'status':'waiting','phase':'cuda_checks','physical_gpu':j['physical_gpu']} for key,j in jobs.items()}}
    atomic(out/'queue.json',state);generate(root);leases={};workers={};logs={}
    def stop(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    def environment(gpu,key):return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(source),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        SWANLAB_LOG_DIR=str(ROOT/'swanlog'),SWANLAB_EXPERIMENT_NAME=f'mimic-supervision-{key}-batch128-seed0')
    def launch(job,stage):
        key=job['id'];path=folder(job,stage);path.mkdir(parents=True,exist_ok=True);stream=(path/'run.log').open('a');logs[key]=stream
        worker=subprocess.Popen([sys.executable,'-u','-m','mimic_experiment.supervision_ablation','--config',job['config_path'],'--stage',stage],
            cwd=source,env=environment(job['physical_gpu'],key),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        workers[key]=(worker,stage);state['jobs'][key].update(status='running',phase=stage,pid=worker.pid);atomic(out/'queue.json',state)
        print(f'{key}: {stage}, physical GPU{job["physical_gpu"]}, PID {worker.pid}',flush=True)
    try:
        for gpu in [3,1]:
            while gpu not in leases:
                candidate=Path(f'/tmp/latentode_gpu_{gpu}.lock').open('a')
                try:fcntl.flock(candidate.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:candidate.close();time.sleep(5);continue
                apps=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                if apps:candidate.close();time.sleep(5)
                else:leases[gpu]=candidate
        if not (out/'cuda_checks.json').exists():
            with (out/'cuda_checks.log').open('a') as stream:
                worker=subprocess.Popen([sys.executable,'-u',str(ROOT/'mimic_experiment/supervision_ablation_checks.py'),'--manifest',str(out/'manifest.json')],
                    cwd=source,env=environment(3,'checks'),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                workers['checks']=(worker,'checks');state['status']='cuda_checks';atomic(out/'queue.json',state);last=0.
                while worker.poll() is None:
                    if time.monotonic()-last>=30:generate(root);last=time.monotonic()
                    time.sleep(2)
                workers.pop('checks')
                if worker.returncode:raise RuntimeError('CUDA checks failed; inspect cuda_checks.log')
        if read(out/'cuda_checks.json',{}).get('status')!='passed':raise RuntimeError('CUDA preflight incomplete')
        state['status']='running'
        for key,job in jobs.items():
            if (folder(job,'training')/'metrics.json').exists():verify(manifest,job,'training');state['jobs'][key].update(status='complete',phase='tested');leases.pop(job['physical_gpu']).close()
            else:
                stage='training' if (folder(job,'smoke')/'metrics.json').exists() else 'smoke'
                if stage=='training':verify(manifest,job,'smoke')
                launch(job,stage)
        last=0.
        while workers:
            for key,(worker,stage) in list(workers.items()):
                code=worker.poll()
                if code is None:continue
                workers.pop(key);logs.pop(key).close();job=jobs[key]
                try:
                    if code:raise RuntimeError(f'Worker exited {code}')
                    verify(manifest,job,stage)
                except Exception as exc:
                    state['jobs'][key].update(status='failed',error=repr(exc));leases.pop(job['physical_gpu']).close()
                else:
                    if stage=='smoke':launch(job,'training')
                    else:state['jobs'][key].update(status='complete',phase='tested');leases.pop(job['physical_gpu']).close()
                atomic(out/'queue.json',state)
            if time.monotonic()-last>=30:generate(root);last=time.monotonic()
            time.sleep(3)
        state.update(status='complete' if all(s['status']=='complete' for s in state['jobs'].values()) else 'failed',finished_at=datetime.now(timezone.utc).isoformat())
        atomic(out/'queue.json',state);generate(root)
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc));atomic(out/'queue.json',state);generate(root);raise
    finally:
        for worker,_ in workers.values():
            if worker.poll() is None:worker.terminate();worker.wait()
        for stream in logs.values():stream.close()
        for lease in leases.values():lease.close()


if __name__=='__main__':main()
