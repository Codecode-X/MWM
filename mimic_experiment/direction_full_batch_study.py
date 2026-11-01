"""Fresh, matched full-batch128 direction runs on physical GPUs 3 and 2."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from paper_odeae.scaling_study import identity
from .prepare import atomic
from .report import read
from .direction_full_batch_report import generate

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'mimic_experiment/results/direction_full_batch_cuda'
OLD=ROOT/'mimic_experiment/results/history_encoder_ablation_cuda'
GPUS={'forward_shared':3,'reverse_shared':2}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    base=read(OLD/'configs/reverse_shared.json');sha=digest(base['data'])
    if sha!=base['dataset_sha256']:raise RuntimeError('Dataset changed')
    (out/'configs').mkdir(exist_ok=True);configs={}
    for variant,gpu in GPUS.items():
        cfg={**base,'study':'mimic_direction_full_batch','variant':variant,'physical_gpu':gpu,
             'output':str(out/variant/'seed_0'),'comparison_output':str(out),'batch_size':128,'micro_batch_size':128,
             'gradient_partitions':1,'gradient_accumulation_steps':1,'encoder_direction':'forward' if variant=='forward_shared' else 'backward',
             'generator_anchor':'history cutoff' if variant=='forward_shared' else 'first history visit',
             'prior_semantics':'forward causal one-step prediction' if variant=='forward_shared' else 'same-parameter auxiliary forward causal prediction; main state remains reverse',
             'verify_reference':False,'fresh_relative_to_previous_experiments':True,
             'ablation':'single GPU full batch128, no gradient accumulation; main history direction and corresponding anchor',
             'source_previous_config':str(OLD/'configs/reverse_shared.json')}
        if variant=='forward_shared':cfg.pop('history_encoder_variant',None)
        else:cfg['history_encoder_variant']='reverse_shared'
        path=out/'configs'/f'{variant}.json'
        if path.exists() and read(path)!=cfg:raise RuntimeError('Preserve existing full-batch configuration')
        atomic(path,cfg);configs[variant]=cfg
    source=out/'source_snapshot'
    if not (out/'manifest.json').exists():
        if not source.exists():shutil.copytree(OLD/'source_snapshot',source)
        for name in ['train_full_batch.py','direction_full_batch_checks.py','direction_full_batch_report.py','direction_full_batch_study.py']:
            shutil.copy2(ROOT/'mimic_experiment'/name,source/'mimic_experiment'/name)
        files=sorted(str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() and p.suffix in ['.py','.json'] and p.name!='source_hashes.json')
        atomic(source/'source_hashes.json',{name:digest(source/name) for name in files})
    hashes=read(source/'source_hashes.json')
    for name,sha_ in hashes.items():
        if digest(source/name)!=sha_:raise RuntimeError('Frozen full-batch source changed')
    manifest={'configs':configs,'parameters':43280,'latent':28,'physical_gpus':GPUS,'seed':0,'batch_size':128,
              'gradient_accumulation_steps':1,'gradient_partitions':1,'source_sha256':hashes,'dataset_sha256':sha,
              'source_origin':str(OLD/'source_snapshot'),'reference_four_gpu':str(ROOT/'mimic_experiment/results/latent2x_both_cuda'),
              'fresh_training':True,'old_checkpoint_weights_loaded':False,'common_training_code':'mimic_experiment/train_full_batch.py'}
    if (out/'manifest.json').exists() and read(out/'manifest.json')!=manifest:raise RuntimeError('Manifest changed')
    atomic(out/'manifest.json',manifest);return configs,source


def verify(out,key,stage):
    folder=out/key/'seed_0' if stage=='training' else out/'checks'/key/'smoke'
    m=read(folder/'metrics.json');h=read(folder/'hardware.json');history=read(folder/'history.json')
    assert m['status']==('complete' if stage=='training' else 'cuda_smoke_passed') and m['checkpoint_reload_difference']<=1e-6
    assert h['parameters']==h['trainable_parameters']==43280 and h['world_size']==1 and h['gradient_accumulation_steps']==1
    assert h['first_actual_training_batch']==128 and h['forward_calls_per_optimizer_step']==h['backward_calls_per_optimizer_step']==1
    assert set(h['actual_tensor_devices'].values())=={'cuda:0'} and h['prediction_device']=='cuda:0'
    for row in history:
        assert row['training_batches']==row['training_forward_calls']==row['training_backward_calls']
        assert row['maximum_batch_size']==128 and row['gradient_accumulation_steps']==1
    if stage=='smoke':assert history[0]['optimizer_steps']==1 and history[0]['minimum_batch_size']==128
    else:
        ref=read(ROOT/'mimic_experiment/results/latent2x_both_cuda/seed_0/metrics.json')
        for split in ['validation','test']:
            assert m[split]['patients']==ref[split]['patients']
            for region in ['interpolation','forecast','overall','prior','forecast_6h','forecast_12h','forecast_24h']:
                assert m[split][region]['observed_scalars']==ref[split][region]['observed_scalars']
        atomic(folder/'completion_verification.json',{'status':'passed','same_target_counts':True,'batch_size':128,
               'gradient_accumulation_steps':1,'physical_gpu':h['physical_gpu'],'device':'cuda:0'})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(DEFAULT));args=parser.parse_args()
    if Path(sys.prefix).name!='ode':raise RuntimeError('Use ode environment')
    out=Path(args.output).resolve();out.mkdir(exist_ok=True);previous=read(out/'queue.json',{})
    if previous.get('pid') and identity(previous['pid'])==previous.get('start_time'):raise RuntimeError('Full-batch direction study already running')
    leases={}
    for gpu in [2,3]:
        inherited=os.environ.get(f'MIMIC_FULL_BATCH_GPU{gpu}_FD')
        if inherited is not None:leases[gpu]=os.fdopen(int(inherited),'a')
        else:
            leases[gpu]=Path(f'/tmp/latentode_gpu_{gpu}.lock').open('a');fcntl.flock(leases[gpu].fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    configs,source=prepare(out);workers={};logs={}
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
           'jobs':{key:{'status':'waiting','phase':'checks','physical_gpu':gpu} for key,gpu in GPUS.items()}}
    atomic(out/'queue.json',state);generate(out)
    def env(gpu,name):return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(source),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                                 SWANLAB_LOG_DIR=str(ROOT/'swanlog'),SWANLAB_EXPERIMENT_NAME=f'mimic-fullbatch128-{name}-seed0')
    try:
        check=out/'checks/full_batch';check.mkdir(parents=True,exist_ok=True)
        if not (check/'checks.json').exists():
            while subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(5)
            with (check/'run.log').open('a') as log:
                p=subprocess.Popen([sys.executable,'-u','-m','mimic_experiment.direction_full_batch_checks','--manifest',str(out/'manifest.json'),'--output',str(check),'--gpu','3'],
                                   cwd=source,env=env(3,'checks'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                workers['checks']=(p,'checks');print(f'Full-batch128 CUDA checks on GPU 3, PID {p.pid}',flush=True)
                while p.poll() is None:time.sleep(2)
                workers.pop('checks')
                if p.returncode or not (check/'checks.json').exists():raise RuntimeError('Full-batch checks failed; inspect checks/full_batch/run.log')
        for stage in ['smoke','training']:
            for key,cfg in configs.items():
                folder=Path(cfg['output']) if stage=='training' else out/'checks'/key/'smoke';folder.mkdir(parents=True,exist_ok=True)
                if (folder/'metrics.json').exists():verify(out,key,stage);state['jobs'][key].update(status='complete' if stage=='training' else 'ready',phase='tested' if stage=='training' else 'smoke_passed');continue
                gpu=cfg['physical_gpu']
                while subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(5)
                stream=(folder/'run.log').open('a');logs[key]=stream
                p=subprocess.Popen([sys.executable,'-u','-m','mimic_experiment.train_full_batch','--config',str(out/'configs'/f'{key}.json'),'--stage',stage],
                                   cwd=source,env=env(gpu,key+'-'+stage),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                workers[key]=(p,stage);state['jobs'][key].update(status='running',phase=stage,pid=p.pid);atomic(out/'queue.json',state)
                print(f'{key}: {stage}, physical GPU {gpu}, full batch128, no accumulation, PID {p.pid}',flush=True)
            last=0
            while workers:
                for key,(p,stage_) in list(workers.items()):
                    if p.poll() is None:continue
                    logs.pop(key).close();workers.pop(key)
                    if p.returncode:state['jobs'][key].update(status='failed');raise RuntimeError(f'{key}/{stage_} failed; see run.log')
                    verify(out,key,stage_);state['jobs'][key].update(status='complete' if stage_=='training' else 'ready',phase='tested' if stage_=='training' else 'smoke_passed')
                    if stage_=='training':leases.pop(GPUS[key]).close()
                    atomic(out/'queue.json',state);print(f'{key}: {stage_} passed',flush=True)
                if time.monotonic()-last>=30:generate(out);last=time.monotonic()
                time.sleep(2)
        state.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat());atomic(out/'queue.json',state);generate(out)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),finished_at=datetime.now(timezone.utc).isoformat())
        for key in workers:
            if key in state['jobs']:state['jobs'][key]['status']='interrupted'
        atomic(out/'queue.json',state);generate(out);raise
    finally:
        for p,_ in workers.values():
            if p.poll() is None:p.terminate();p.wait()
        for stream in logs.values():stream.close()
        for handle in leases.values():handle.close()


if __name__=='__main__':main()
