"""Launch four independent single-GPU baseline pipelines and maintain the live comparison."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

from paper_odeae.scaling_study import identity
from .baseline_report import generate,NAMES
from .prepare import DEFAULT,atomic
from .report import read
from .run import verify_data

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/'mimic_experiment/results/baselines_single_cuda'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out,data_root,gpus):
    payload=verify_data(data_root,out);schema=payload['schema'];sha=digest(data_root/'dataset.pkl')
    reference=ROOT/'mimic_experiment/results/latent2x_both_cuda'
    assert read(reference/'seed_0/metrics.json')['status']=='complete'
    assert read(reference/'seed_0/hardware.json')['dataset_sha256']==sha
    originals={
        'healthflux_kl':ROOT/'healthflux/results/adni_comparison/configs/paper_kl.json',
        'ode_ae':ROOT/'paper_odeae/results/local_available/ode_ae/seed_0/config.json',
        'ode_rnn':ROOT/'paper_odeae/results/baselines_full/native/ode_rnn/seed_0/config.json',
        'latent_ode':ROOT/'paper_odeae/results/baselines_full/native/latent_ode/seed_0/config.json'}
    configs={};(out/'configs').mkdir(exist_ok=True)
    groups={'vitals':[f['name'] for f in schema[:9]],'laboratory':[f['name'] for f in schema[9:27]],'gcs':[f['name'] for f in schema[27:30]]}
    for gpu,(variant,source) in zip(gpus,originals.items()):
        cfg={**read(source),'data':str(data_root/'dataset.pkl'),'dataset_sha256':sha,'scope':payload['metadata']['scope'],
             'output':str(out/variant/'seed_0'),'comparison_output':str(out),'reference_output':str(reference),
             'physical_gpu':gpu,'batch_size':128,'micro_batch_size':32,'gradient_partitions':4,'study':'mimic_baselines_single',
             'variant':variant,'time_unit':'hour','training_context_hours':[6,12,18,24],'evaluation_context_hours':24,
             'verify_reference':variant=='latent_ode','source_adni_config':str(source),
             'age_covariate':'biological years/100 supplied as an additional GRU input; decoder targets unchanged' if variant in ['ode_rnn','latent_ode'] else 'age INIT or age-conditioned dynamics',
             'history_semantics':'causal prediction before current observation' if variant=='ode_rnn' else 'history-conditioned reconstruction',
             'future_semantics':'autoregressive predicted-value GRU updates' if variant=='ode_rnn' else 'ODE rollout without future observation updates'}
        if variant=='healthflux_kl':cfg.update(modality_groups=groups,ode_clock='hour',rk4_nominal_step_hours=1,rk4_max_substeps=4,age_condition_unit='year',diagnosis_event_nodes=0)
        elif variant=='ode_ae':cfg.update(encoder_direction='backward',encoder_initialization='age')
        if variant=='ode_rnn':cfg['kl_weight']=0.
        path=out/'configs'/f'{variant}.json'
        if path.exists() and read(path)!=cfg:raise RuntimeError('Existing baseline configuration changed')
        atomic(path,cfg);configs[variant]=cfg
    snapshot=out/'source_snapshot'
    files=['mimic_experiment/__init__.py','mimic_experiment/prepare.py','mimic_experiment/data.py','mimic_experiment/baseline_models.py','mimic_experiment/healthflux_icu.py',
           'mimic_experiment/evaluation.py','mimic_experiment/train_baseline.py','mimic_experiment/baseline_report.py','mimic_experiment/baseline_study.py',
           'mimic_experiment/report.py','mimic_experiment/run.py','mimic_experiment/train_ddp.py',
           'healthflux/__init__.py','healthflux/model.py','paper_odeae/__init__.py','paper_odeae/models.py','paper_odeae/data.py','paper_odeae/train.py',
           'paper_odeae/baselines.py','paper_odeae/solver.py','paper_odeae/jacobian.py','paper_odeae/scaling_study.py',
           'paper_odeae/reference/features46.json','paper_odeae/reference/tsit5_tableau.json','src/__init__.py','src/model.py','src/adni_models.py','src/tracking.py','src/swanlab_local.py']
    if not (snapshot/'source_hashes.json').exists():
        for file in files:
            target=snapshot/file;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/file,target)
        atomic(snapshot/'source_hashes.json',{file:digest(snapshot/file) for file in files})
    for file,sha_ in read(snapshot/'source_hashes.json').items():
        if digest(snapshot/file)!=sha_:raise RuntimeError('Frozen baseline source changed')
    atomic(out/'manifest.json',{'dataset':str(data_root/'dataset.pkl'),'dataset_sha256':sha,'reference_output':str(reference),
           'models':list(configs),'physical_gpus':gpus,'strategy':'one model per GPU; single-GPU gradient accumulation; no DDP',
           'seed':0,'effective_batch_size':128,'max_micro_batch_size':32,'source_hashes':read(snapshot/'source_hashes.json')})
    del payload
    return configs,snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-root',default=str(DEFAULT));parser.add_argument('--output',default=str(DEFAULT_OUT));parser.add_argument('--gpus',type=int,nargs=4,default=[0,1,2,3]);args=parser.parse_args()
    if Path(sys.prefix).name!='ode':raise RuntimeError('Use ode for orchestration')
    if sorted(args.gpus)!=[0,1,2,3]:raise ValueError('Assign all four GPUs, one independent model each')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True);previous=read(out/'queue.json',{})
    if previous.get('pid') and identity(previous['pid'])==previous.get('start_time'):raise RuntimeError('Baseline controller already running')
    configs,snapshot=prepare(out,Path(args.data_root).resolve(),args.gpus)
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'waiting_for_gpus','started_at':datetime.now(timezone.utc).isoformat(),
           'jobs':{key:{'physical_gpu':cfg['physical_gpu'],'status':'waiting','phase':'checks'} for key,cfg in configs.items()}}
    atomic(out/'queue.json',state);generate(out)
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(10)
    running={};streams={};phases=['checks','smoke','training']
    def completed(key,stage):
        folder=Path(configs[key]['output']) if stage=='training' else out/'checks'/key/stage
        return (folder/('checks.json' if stage=='checks' else 'metrics.json')).exists()
    def launch(key,stage):
        cfg=configs[key];environment='healthflux' if cfg['model']=='healthflux' else 'ode'
        python=Path('/home/xjh/miniforge3/envs')/environment/'bin/python'
        if not python.exists():raise RuntimeError(f'Missing Conda interpreter {python}')
        folder=Path(cfg['output']) if stage=='training' else out/'checks'/key/stage;folder.mkdir(parents=True,exist_ok=True)
        stream=(folder/'run.log').open('a');streams[key]=stream
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(cfg['physical_gpu']),PYTHONPATH=str(snapshot),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',SWANLAB_LOG_DIR=str(ROOT/'swanlog'))
        process=subprocess.Popen([str(python),'-u','-m','mimic_experiment.train_baseline','--config',str(out/'configs'/f'{key}.json'),'--stage',stage],
                                 cwd=snapshot,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        running[key]=(process,stage);state['jobs'][key].update(status='running',phase=stage,pid=process.pid,environment=environment,started_at=datetime.now(timezone.utc).isoformat())
        print(f'{key}: {stage}, single GPU {cfg["physical_gpu"]}, PID {process.pid}',flush=True)
    try:
        state['status']='running'
        for key in configs:
            todo=next((stage for stage in phases if not completed(key,stage)),None)
            if todo:launch(key,todo)
            else:state['jobs'][key].update(status='complete',phase='training')
        atomic(out/'queue.json',state);last=0
        while running:
            for key,(process,stage) in list(running.items()):
                code=process.poll()
                if code is None:continue
                streams.pop(key).close();running.pop(key)
                if code!=0 or not completed(key,stage):
                    state['jobs'][key].update(status='failed',exit_code=code,failed_at=datetime.now(timezone.utc).isoformat())
                    print(f'{key}: failed in {stage}; see its run.log',flush=True)
                elif stage!='training':launch(key,phases[phases.index(stage)+1])
                else:state['jobs'][key].update(status='complete',finished_at=datetime.now(timezone.utc).isoformat())
                atomic(out/'queue.json',state)
            if time.monotonic()-last>=30:generate(out);last=time.monotonic()
            time.sleep(2)
        state.update(status='complete' if all(job['status']=='complete' for job in state['jobs'].values()) else 'failed',finished_at=datetime.now(timezone.utc).isoformat())
        atomic(out/'queue.json',state);generate(out)
    finally:
        for process,_ in running.values():
            if process.poll() is None:process.terminate()
        for stream in streams.values():stream.close()


if __name__=='__main__':main()
