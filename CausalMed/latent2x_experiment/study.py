"""Serial GPU-3 blind model experiments, followed by independent oracle scoring."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .report import generate,read
from .isolation import command,PYTHON

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'CausalMed/outputs/latent2x_both_cuda3'
ORDER=['gamma_1']


def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.partial');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def identity(pid):
    try:
        parts=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return parts[19] if parts[0]!='Z' else None
    except FileNotFoundError:return None


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    source=out/'model_source';source.mkdir(exist_ok=True)
    files={
      **{f'CausalMed/latent2x_experiment/{name}':f'latent2x_experiment/{name}' for name in ['__init__.py','data.py','model.py','rollout.py','train.py']},
      **{f'paper_odeae/{name}':f'paper_odeae/{name}' for name in ['__init__.py','models.py','solver.py','jacobian.py','reference/tsit5_tableau.json']},
      **{f'src/{name}':f'src/{name}' for name in ['__init__.py','tracking.py','swanlab_local.py']},
      **{f'CausalMed/model_client/causalmed_client/{name}':f'causalmed_client/{name}' for name in ['__init__.py','dataset_adapter.py','counterfactual_policies.py']}}
    hashes=read(out/'model_source_hashes.json')
    if not hashes:
        for original,target in files.items():
            destination=source/target;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/original,destination)
        hashes={target:digest(source/target) for target in files.values()};atomic(out/'model_source_hashes.json',hashes)
    for name,sha in hashes.items():
        if digest(source/name)!=sha:raise RuntimeError('Frozen model source changed')
    configs={};public_hashes={}
    for key in ORDER:
        public=ROOT/'CausalMed/outputs/level3_suite'/key/'public'
        public_hashes[key]={p.name:digest(p) for p in sorted(public.iterdir()) if p.is_file()}
        cfg={'model':'latent2x_both_action_update','variant':key,'seed':0,'latent':28,'mlp_width':159,'ode_width':188,
             'prior_weight':.01,'jacobian_weight':.01,'l2_weight':.02,'normalization':'train_zscore',
             'lr':.003,'lr_decay':.97,'lr_decay_steps':2000,'input_dropout':.1,'batch_size':128,
             'max_epochs':300,'min_epochs':50,'patience':40,'rtol':1e-4,'atol':1e-6,'rollout_patient_batch':32,
             'selection':'validation factual overall standardized MSE','training_cutoff_years':[1,2,3,4],
             'physical_gpu':3,'action_update':'residual difference between new/old treatment-conditioned shared UPDATE outputs'}
        configs[key]=cfg
    manifest={'model_source_sha256':hashes,'public_data_sha256':public_hashes,'configs':configs,'order':ORDER,
              'physical_gpu':3,'environment':'ode','blind_execution':'bubblewrap namespace; public-only model mounts; separate post-submission oracle evaluator'}
    if (out/'manifest.json').exists() and read(out/'manifest.json')!=manifest:raise RuntimeError('Experiment inputs/configuration changed')
    atomic(out/'manifest.json',manifest)
    return source,configs


def wait_gpu():
    while subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(10)


def ensure_analysis(out):
    existing=read(out/'analysis_queue.json',{});pid=existing.get('pid')
    path=Path(f'/proc/{pid}/cmdline') if pid else None
    if path and path.exists():
        cmd=path.read_bytes()
        if b'CausalMed.latent2x_experiment.analyze' in cmd and str(out).encode() in cmd:return
    with (out/'analysis.log').open('a') as log:
        process=subprocess.Popen([str(PYTHON),'-u','-m','CausalMed.latent2x_experiment.analyze','--output',str(out)],
                                 cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(f'Independent statistics/plots process PID {process.pid}',flush=True)


def launch(out,source,key,cfg,stage,state):
    folder=out/key/('model' if stage=='training' else stage);folder.mkdir(parents=True,exist_ok=True)
    expected='checks.json' if stage=='checks' else 'submission.json'
    if (folder/expected).exists():return
    atomic(folder/'launch_config.json',cfg)
    state['jobs'][key].update(status='running',phase=stage);atomic(out/'queue.json',state);generate(out);wait_gpu()
    public=ROOT/'CausalMed/outputs/level3_suite'/key/'public';env=dict(os.environ,SWANLAB_EXPERIMENT_NAME=f'causalmed-latent2x-both-{key}-{stage}-seed0')
    with (folder/'run.log').open('a') as log:
        worker=subprocess.Popen(command(source,public,folder,ROOT/'swanlog',stage),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
        state['jobs'][key].update(pid=worker.pid);atomic(out/'queue.json',state);print(f'{key}: {stage}, physical GPU 3, PID {worker.pid}',flush=True)
        last=0
        try:
            while worker.poll() is None:
                if time.monotonic()-last>=30:generate(out);last=time.monotonic()
                time.sleep(2)
        finally:
            if worker.poll() is None:worker.terminate();worker.wait()
        if worker.returncode or not (folder/expected).exists():raise RuntimeError(f'{key}/{stage} failed; see {folder}/run.log')
    result=read(folder/expected)
    if stage=='smoke' and result['status']!='cuda_smoke_passed':raise RuntimeError('Smoke submission incomplete')
    if stage=='training' and result['status']!='predictions_sealed':raise RuntimeError('Formal submission incomplete')


def score(out,key,state):
    state['jobs'][key].update(status='running',phase='independent_scoring');atomic(out/'queue.json',state);generate(out)
    for variant in ['with_actions','action_off']:
        model=out/key/'model'/variant;sealed=read(model/'sealed.json')
        for name,sha in sealed['sha256'].items():
            if digest(model/name)!=sha:raise RuntimeError('Sealed predictions changed')
        target=out/key/'evaluation'/variant;target.mkdir(parents=True,exist_ok=True)
        if (target/'metrics.json').exists():continue
        env=dict(os.environ,PYTHONPATH=str(ROOT/'CausalMed'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        args=[str(PYTHON),'-m','synthetic_cf_level3.evaluation','--predictions',str(model/'predictions.parquet'),
              '--predicted-treatments',str(model/'predicted_treatments.parquet'),'--oracle-dir',str(ROOT/'CausalMed/outputs/level3_suite'/key/'oracle_private'),'--out-dir',str(target)]
        with (target/'run.log').open('a') as log:subprocess.run(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    state['jobs'][key].update(status='complete',phase='scored',finished_at=datetime.now(timezone.utc).isoformat());atomic(out/'queue.json',state);generate(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(DEFAULT));args=parser.parse_args()
    if Path(sys.prefix).name!='ode':raise RuntimeError('Use ode environment')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True);previous=read(out/'queue.json',{})
    if previous.get('pid') and identity(previous['pid'])==previous.get('start_time'):raise RuntimeError('CausalMed study already running')
    source,configs=prepare(out)
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
           'physical_gpu':3,'jobs':{key:{'status':'waiting','phase':'waiting'} for key in ORDER}}
    atomic(out/'queue.json',state);generate(out);ensure_analysis(out)
    try:
        for key in ORDER:
            if (out/key/'evaluation/action_off/metrics.json').exists():state['jobs'][key].update(status='complete',phase='scored');continue
            for stage in ['checks','smoke','training']:launch(out,source,key,configs[key],stage,state)
            score(out,key,state)
        state.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat());atomic(out/'queue.json',state);generate(out)
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}')
        for job in state['jobs'].values():
            if job['status']=='running':job['status']='failed'
        atomic(out/'queue.json',state);generate(out);raise


if __name__=='__main__':main()
