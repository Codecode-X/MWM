"""Single-GPU standard-model INIT ablations, changing only the scalar covariate."""
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
from .init_covariate_ablation_report import generate

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'mimic_experiment/results/init_covariate_ablation_cuda'
REFERENCE=ROOT/'mimic_experiment/results/direction_full_batch_cuda/forward_shared/seed_0'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().split();return None if stat[2]=='Z' else stat[21]
    except FileNotFoundError:return None


def prepare(out):
    baseline=read(REFERENCE.parents[1]/'configs/forward_shared.json')
    if not read(REFERENCE/'metrics.json') or baseline['prior_weight']!=.01 or baseline['jacobian_weight']!=.01 or baseline['l2_weight']!=.02:
        raise RuntimeError('Use the completed standard forward latent2x both reference')
    if baseline.get('train_reconstruct_history') is False:raise RuntimeError('Restore the complete standard training objective')
    if digest(baseline['data'])!=baseline['dataset_sha256']:raise RuntimeError('Dataset changed')
    original=REFERENCE.parents[1]/'source_snapshot';source=out/'source_snapshot';original_hashes=read(original/'source_hashes.json')
    if not source.exists():
        shutil.copytree(original,source);shutil.copy2(ROOT/'mimic_experiment/init_covariate_ablation.py',source/'mimic_experiment/init_covariate_ablation.py')
        hashes={**original_hashes,'mimic_experiment/init_covariate_ablation.py':digest(source/'mimic_experiment/init_covariate_ablation.py')}
        atomic(source/'source_hashes.json',hashes)
    hashes=read(source/'source_hashes.json')
    for name,sha in original_hashes.items():
        if hashes.get(name)!=sha:raise RuntimeError('Backbone, loss, data clock or original trainer changed')
    for name,sha in hashes.items():
        if digest(source/name)!=sha:raise RuntimeError('Frozen source changed')
    jobs=[]
    for mode,gpu in [('first_age',3),('sex',1)]:
        description='first observed age = ICU admission age + first elapsed hour/8766' if mode=='first_age' else 'F=0, M=1 into scalar INIT; interface compensates original /100'
        cfg={**baseline,'init_covariate':mode,'initialization_age':description,'physical_gpu':gpu,
             'output':str(out/mode/'seed_0'),'comparison_output':str(out/'jobs'/mode)}
        differences={key:[baseline.get(key),cfg.get(key)] for key in set(cfg)|set(baseline) if cfg.get(key)!=baseline.get(key)}
        if set(differences)-{'init_covariate','initialization_age','physical_gpu','output','comparison_output'}:raise RuntimeError('Unexpected experimental factor')
        path=out/'configs'/f'{mode}.json';path.parent.mkdir(exist_ok=True)
        if path.exists() and read(path)!=cfg:raise RuntimeError('Preserve existing INIT configuration')
        atomic(path,cfg);Path(cfg['output']).mkdir(parents=True,exist_ok=True)
        jobs.append({'id':mode,'physical_gpu':gpu,'config':cfg,'config_path':str(path),'configuration_differences':differences})
    manifest={'reference':str(REFERENCE),'reference_config':baseline,'source_snapshot':str(source),'source_sha256':hashes,'jobs':jobs,
              'changed_training_factor':'scalar input to unchanged 1->159->28 INIT network',
              'standard_objective':'history reconstruction + future prediction + 0.01 weak_prior + 0.01 Jacobian + 0.02 ODE L2',
              'parameters':43280,'INIT_parameters':4798,'batch_size':128,'gradient_accumulation_steps':1,'seed':0,
              'checkpoint_selection':'same validation overall data reconstruction error',
              'driver_sha256':{name:digest(ROOT/name) for name in ['mimic_experiment/init_covariate_ablation_study.py','mimic_experiment/init_covariate_ablation_report.py']}}
    if (out/'manifest.json').exists() and read(out/'manifest.json')!=manifest:raise RuntimeError('Preserve existing INIT manifest')
    atomic(out/'manifest.json',manifest);return manifest


def folder(job,stage):
    cfg=job['config'];return Path(cfg['output']) if stage=='training' else Path(cfg['comparison_output'])/'checks/forward_shared/smoke'


def verify(manifest,job,stage):
    path=folder(job,stage);m=read(path/'metrics.json');h=read(path/'hardware.json');audit=read(path/'init_input_audit.json')
    if not m or m['status']!=('complete' if stage=='training' else 'cuda_smoke_passed') or m['checkpoint_reload_difference']>1e-6:raise RuntimeError('Incomplete CUDA training/test stage')
    if h['parameters']!=43280 or h['trainable_parameters']!=43280 or h['prior_weight']!=.01 or h['jacobian_weight']!=.01 or h['l2_weight']!=.02:raise RuntimeError('Restore standard objective and capacity')
    if h['physical_gpu']!=job['physical_gpu'] or h['first_actual_training_batch']!=128 or h['gradient_accumulation_steps']!=1:raise RuntimeError('Require one actual batch128 on the assigned GPU')
    if h['prediction_device']!='cuda:0' or set(h['actual_tensor_devices'].values())!={'cuda:0'}:raise RuntimeError('All model tensors must execute on CUDA')
    if not audit or audit['source']!=job['id'] or audit['actual_input_device']!='cuda:0' or audit['INIT_parameters']!=4798:raise RuntimeError('Actual INIT input audit mismatch')
    if stage=='training':
        ref=read(REFERENCE/'metrics.json')
        for split in ['validation','test']:
            for region in ['interpolation','forecast','overall','prior','forecast_6h','forecast_12h','forecast_24h']:
                if m[split][region]['observed_scalars']!=ref[split][region]['observed_scalars']:raise RuntimeError('Common evaluation target counts differ')
        atomic(path/'completion_verification.json',{'status':'passed','parameters':43280,'INIT_parameters':4798,
               'init_covariate':job['id'],'batch_size':128,'gradient_accumulation_steps':1,'physical_gpu':h['physical_gpu'],'same_target_counts':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(DEFAULT));args=parser.parse_args()
    if Path(sys.prefix).name!='ode':raise RuntimeError('Use ode environment')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'paused_by_user.json').exists():raise RuntimeError('Explicit resume required after user pause')
    previous=read(out/'queue.json',{})
    if previous.get('pid') and identity(previous['pid'])==previous.get('start_time'):raise RuntimeError('INIT ablation already running')
    manifest=prepare(out);source=Path(manifest['source_snapshot']);jobs={j['id']:j for j in manifest['jobs']}
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'waiting_for_gpu','started_at':datetime.now(timezone.utc).isoformat(),
           'jobs':{key:{'status':'waiting','phase':'cuda_checks','physical_gpu':j['physical_gpu']} for key,j in jobs.items()}}
    atomic(out/'queue.json',state);generate(out);leases={};workers={};logs={}
    def stop(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    def environment(gpu,key):return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(source),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        SWANLAB_LOG_DIR=str(ROOT/'swanlog'),SWANLAB_EXPERIMENT_NAME=f'mimic-standard-init-{key}-batch128-seed0')
    def launch(job,stage):
        key=job['id'];path=folder(job,stage);path.mkdir(parents=True,exist_ok=True);stream=(path/'run.log').open('a');logs[key]=stream
        worker=subprocess.Popen([sys.executable,'-u','-m','mimic_experiment.init_covariate_ablation','--config',job['config_path'],'--stage',stage],
            cwd=source,env=environment(job['physical_gpu'],key),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        workers[key]=(worker,stage);state['jobs'][key].update(status='running',phase=stage,pid=worker.pid);atomic(out/'queue.json',state)
        print(f'{key}: {stage}, physical GPU{job["physical_gpu"]}, PID {worker.pid}',flush=True)
    try:
        for gpu in [3,1]:
            while gpu not in leases:
                stream=Path(f'/tmp/latentode_gpu_{gpu}.lock').open('a')
                try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:stream.close();time.sleep(5);continue
                apps=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                if apps:stream.close();time.sleep(5)
                else:leases[gpu]=stream
        if not (out/'cuda_checks.json').exists():
            with (out/'cuda_checks.log').open('a') as stream:
                worker=subprocess.Popen([sys.executable,'-u','-m','mimic_experiment.init_covariate_ablation','--check','--manifest',str(out/'manifest.json')],
                    cwd=source,env=environment(3,'checks'),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                workers['checks']=(worker,'checks');state['status']='cuda_checks';atomic(out/'queue.json',state);last=0.
                while worker.poll() is None:
                    if time.monotonic()-last>=30:generate(out);last=time.monotonic()
                    time.sleep(2)
                workers.pop('checks')
                if worker.returncode:raise RuntimeError('CUDA INIT checks failed; inspect cuda_checks.log')
        if read(out/'cuda_checks.json',{}).get('status')!='passed':raise RuntimeError('CUDA checks incomplete')
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
            if time.monotonic()-last>=30:generate(out);last=time.monotonic()
            time.sleep(3)
        state.update(status='complete' if all(s['status']=='complete' for s in state['jobs'].values()) else 'failed',finished_at=datetime.now(timezone.utc).isoformat())
        atomic(out/'queue.json',state);generate(out)
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc));atomic(out/'queue.json',state);generate(out);raise
    finally:
        for worker,_ in workers.values():
            if worker.poll() is None:worker.terminate();worker.wait()
        for stream in logs.values():stream.close()
        for stream in leases.values():stream.close()


if __name__=='__main__':main()
