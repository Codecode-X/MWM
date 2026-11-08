"""Two future-only weak-prior coefficients, one complete CUDA job per GPU."""
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
    previous=read(root/'history_reconstruction_study/manifest.json');base=previous['config']
    if not read(Path(base['output'])/'metrics.json'):raise RuntimeError('Keep the completed future-only zero-prior reference')
    original=Path(previous['source_snapshot']);source=out/'source_snapshot'
    old=(original/'mimic_experiment/train_future_only.py').read_text()
    expected=old.replace("    if cfg['prior_weight']!=0 or cfg.get('train_reconstruct_history') is not False:raise ValueError('This experiment requires weak_prior=0 and future-only training')",
                         "    if cfg['prior_weight']<0 or cfg.get('train_reconstruct_history') is not False:raise ValueError('Require nonnegative weak_prior and future-only main training')")
    line="    loss=cfg['mse_weight']*mse+cfg['cce_weight']*cce+cfg['jacobian_weight']*jac+cfg['l2_weight']*l2\n"
    expected=expected.replace(line,line+"    if cfg['prior_weight']>0:loss=loss+cfg['prior_weight']*(pmse+pcce)\n")
    if expected!=(ROOT/'mimic_experiment/train_future_only.py').read_text():raise RuntimeError('Unexpected changes beyond activating the prior coefficient')
    if digest(base['data'])!=base['dataset_sha256']:raise RuntimeError('Dataset changed')
    if not source.exists():
        shutil.copytree(original,source);shutil.copy2(ROOT/'mimic_experiment/train_future_only.py',source/'mimic_experiment/train_future_only.py')
        hashes={name:digest(source/name) for name in previous['source_sha256']};atomic(source/'source_hashes.json',hashes)
    hashes=read(source/'source_hashes.json')
    for name,sha in hashes.items():
        if digest(source/name)!=sha:raise RuntimeError('Frozen source changed')
    changed=[name for name,sha in hashes.items() if sha!=previous['source_sha256'][name]]
    if changed!=['mimic_experiment/train_future_only.py']:raise RuntimeError('Backbone or original training loop changed')
    jobs=[]
    for key,weight,gpu in [('prior_0p01',.01,3),('prior_1',1.,1)]:
        folder=out/key;cfg={**base,'prior_weight':weight,'physical_gpu':gpu,'output':str(root/f'future_{key}/seed_0'),'comparison_output':str(folder)}
        differences={k:[base.get(k),cfg.get(k)] for k in set(base)|set(cfg) if base.get(k)!=cfg.get(k)}
        if set(differences)-{'prior_weight','physical_gpu','output','comparison_output'}:raise RuntimeError('Unmatched training settings')
        config=out/'configs'/f'{key}.json';config.parent.mkdir(exist_ok=True)
        if config.exists() and read(config)!=cfg:raise RuntimeError('Preserve existing configuration')
        atomic(config,cfg);jobs.append({'id':key,'config':cfg,'config_path':str(config),'physical_gpu':gpu,'configuration_differences':differences})
    manifest={'reference':base['output'],'reference_config':base,'reference_source':str(original),'source_snapshot':str(source),
              'source_sha256':hashes,'source_changes':changed,'jobs':jobs,'changed_training_factor':'weak_prior coefficient only',
              'training':'no past reconstruction; future primary supervision plus chronological pre-UPDATE prior supervision',
              'jacobian_states':'future generated states, same as completed future-only zero-prior group',
              'checkpoint_selection':'original validation overall MSE+CE','seed':0,'gradient_accumulation_steps':1,
              'driver_sha256':{name:digest(ROOT/name) for name in ['mimic_experiment/future_prior_ablation_study.py','mimic_experiment/future_prior_ablation_checks.py','mimic_experiment/weak_prior_ablation_report.py']}}
    if (out/'manifest.json').exists() and read(out/'manifest.json')!=manifest:raise RuntimeError('Preserve existing manifest')
    atomic(out/'manifest.json',manifest);return manifest


def folder(job,stage):
    cfg=job['config'];return Path(cfg['output']) if stage=='training' else Path(cfg['comparison_output'])/'checks/forward_shared/smoke'


def verify(manifest,job,stage):
    path=folder(job,stage);m=read(path/'metrics.json');h=read(path/'hardware.json');cfg=job['config']
    if not m or m['status']!=('complete' if stage=='training' else 'cuda_smoke_passed') or m['checkpoint_reload_difference']>1e-6:raise RuntimeError('Incomplete training/test stage')
    if h['parameters']!=43280 or h['trainable_parameters']!=43280 or h['prior_weight']!=cfg['prior_weight']:raise RuntimeError('Parameter budget or prior weight mismatch')
    if h['physical_gpu']!=job['physical_gpu'] or h['first_actual_training_batch']!=128 or h['gradient_accumulation_steps']!=1:raise RuntimeError('Require one complete batch128 per assigned GPU')
    if h['prediction_device']!='cuda:0' or set(h['actual_tensor_devices'].values())!={'cuda:0'}:raise RuntimeError('Model computations must be CUDA')
    history=read(path/'history.json')
    if any(abs(row['training_components']['weighted_prior']-cfg['prior_weight']*(row['training_components']['prior_mse']+row['training_components']['prior_cce']))>1e-5 for row in history):
        raise RuntimeError('Recorded prior weighting is inconsistent')
    if stage=='training':
        ref=read(Path(manifest['reference'])/'metrics.json')
        for split in ['validation','test']:
            for region in ['interpolation','forecast','overall','prior','forecast_6h','forecast_12h','forecast_24h']:
                if m[split][region]['observed_scalars']!=ref[split][region]['observed_scalars']:raise RuntimeError('Common evaluation counts differ')
        atomic(path/'completion_verification.json',{'status':'passed','parameters':43280,'prior_weight':cfg['prior_weight'],
               'batch_size':128,'gradient_accumulation_steps':1,'physical_gpu':h['physical_gpu'],'device':'cuda:0','same_target_counts':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=str(DEFAULT));args=parser.parse_args()
    if Path(sys.prefix).name!='ode':raise RuntimeError('Use ode environment')
    root=Path(args.root).resolve();out=root/'future_prior_scaling_study';out.mkdir(parents=True,exist_ok=True)
    if (out/'paused_by_user.json').exists():raise RuntimeError('Explicit resume required after user pause')
    previous=read(out/'queue.json',{})
    if previous.get('pid') and identity(previous['pid'])==previous.get('start_time'):raise RuntimeError('Future-prior ablation already running')
    manifest=prepare(root,out);source=Path(manifest['source_snapshot']);jobs={j['id']:j for j in manifest['jobs']}
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'waiting_for_gpu','started_at':datetime.now(timezone.utc).isoformat(),
           'jobs':{key:{'status':'waiting','phase':'cuda_checks','physical_gpu':j['physical_gpu']} for key,j in jobs.items()}}
    atomic(out/'queue.json',state);generate(root);leases={};workers={};logs={}
    def stop(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    def environment(gpu,key):return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(source),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                                        SWANLAB_LOG_DIR=str(ROOT/'swanlog'),SWANLAB_EXPERIMENT_NAME=f'mimic-future-only-{key}-batch128-seed0')
    def launch(job,stage):
        key=job['id'];path=folder(job,stage);path.mkdir(parents=True,exist_ok=True);log=(path/'run.log').open('a');logs[key]=log
        worker=subprocess.Popen([sys.executable,'-u','-m','mimic_experiment.train_future_only','--config',job['config_path'],'--stage',stage],
             cwd=source,env=environment(job['physical_gpu'],key),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
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
            with (out/'cuda_checks.log').open('a') as log:
                p=subprocess.Popen([sys.executable,'-u',str(ROOT/'mimic_experiment/future_prior_ablation_checks.py'),'--manifest',str(out/'manifest.json')],
                   cwd=source,env=environment(3,'gradient-checks'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                workers['checks']=(p,'checks');state['status']='cuda_checks';atomic(out/'queue.json',state);last=0.
                while p.poll() is None:
                    if time.monotonic()-last>=30:generate(root);last=time.monotonic()
                    time.sleep(2)
                workers.pop('checks')
                if p.returncode:raise RuntimeError('CUDA gradient checks failed; inspect cuda_checks.log')
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
        state.update(status='complete' if all(j['status']=='complete' for j in state['jobs'].values()) else 'failed',finished_at=datetime.now(timezone.utc).isoformat())
        atomic(out/'queue.json',state);generate(root)
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc));atomic(out/'queue.json',state);generate(root);raise
    finally:
        for worker,_ in workers.values():
            if worker.poll() is None:worker.terminate();worker.wait()
        for stream in logs.values():stream.close()
        for lease in leases.values():lease.close()


if __name__=='__main__':main()
