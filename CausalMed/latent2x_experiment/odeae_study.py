"""Gamma=1 ODE-AE: one GPU-1 worker, public-only training and independent scoring."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .study import ROOT,PYTHON,atomic,identity,digest,score,ensure_analysis
from .report import generate,read
from .isolation import command

DEFAULT=ROOT/'CausalMed/outputs/ode_ae_gamma1_cuda1'


def prepare(out):
    reference=read(ROOT/'CausalMed/outputs/latent2x_both_cuda3/manifest.json')
    public=ROOT/'CausalMed/outputs/level3_suite/gamma_1/public'
    hashes={p.name:digest(p) for p in sorted(public.iterdir()) if p.is_file()}
    if hashes!=reference['public_data_sha256']['gamma_1']:raise RuntimeError('Comparison public data differ')
    files={
      **{f'CausalMed/latent2x_experiment/{name}':f'latent2x_experiment/{name}' for name in ['__init__.py','data.py','model.py','odeae_model.py','rollout.py','train.py']},
      **{f'paper_odeae/{name}':f'paper_odeae/{name}' for name in ['__init__.py','models.py','solver.py','jacobian.py','reference/tsit5_tableau.json']},
      **{f'src/{name}':f'src/{name}' for name in ['__init__.py','tracking.py','swanlab_local.py']},
      **{f'CausalMed/model_client/causalmed_client/{name}':f'causalmed_client/{name}' for name in ['__init__.py','dataset_adapter.py','counterfactual_policies.py']}}
    source=out/'model_source';source.mkdir(exist_ok=True);source_hashes=read(out/'model_source_hashes.json')
    if not source_hashes:
        for original,target in files.items():
            destination=source/target;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/original,destination)
        source_hashes={target:digest(source/target) for target in files.values()};atomic(out/'model_source_hashes.json',source_hashes)
    for name,sha in source_hashes.items():
        if digest(source/name)!=sha:raise RuntimeError('Frozen ODE-AE source changed')
    cfg={**reference['configs']['gamma_1'],'model':'ode_ae_action_update','latent':6,'mlp_width':16,'ode_width':16,
         'physical_gpu':1,'kl_weight':1.,'prior_weight':0.,'jacobian_weight':0.,
         'action_update':'encoder residual correction; independent 6D generator treatment-conditional residual module',
         'posterior_anchor':'first history visit; mean replayed to cutoff before policy branching',
         'encoder_direction':'backward','shared_vector_field':False}
    manifest={'model_name':'ODE-AE','model_source_sha256':source_hashes,'public_data_sha256':{'gamma_1':hashes},'configs':{'gamma_1':cfg},
              'order':['gamma_1'],'physical_gpu':1,'environment':'ode','reference_output':str(ROOT/'CausalMed/outputs/latent2x_both_cuda3'),
              'blind_execution':'bubblewrap namespace; public-only model mounts; separate post-submission oracle evaluator'}
    if (out/'manifest.json').exists() and read(out/'manifest.json')!=manifest:raise RuntimeError('ODE-AE configuration changed')
    atomic(out/'manifest.json',manifest);return source,cfg


def ensure_comparison(out):
    target=ROOT/'CausalMed/outputs/gamma1_model_comparison';target.mkdir(parents=True,exist_ok=True)
    previous=read(target/'queue.json',{});pid=previous.get('pid')
    if pid and identity(pid)==previous.get('start_time'):return
    with (target/'run.log').open('a') as log:
        process=subprocess.Popen([str(PYTHON),'-u','-m','CausalMed.latent2x_experiment.compare','--odeae-output',str(out),'--output',str(target)],
                                 cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(f'Gamma=1 comparison monitor PID {process.pid}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(DEFAULT));args=parser.parse_args()
    if Path(sys.prefix).name!='ode':raise RuntimeError('Use ode environment')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True);old=read(out/'queue.json',{})
    if old.get('pid') and identity(old['pid'])==old.get('start_time'):raise RuntimeError('ODE-AE study already running')
    source,cfg=prepare(out);state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'running',
         'started_at':datetime.now(timezone.utc).isoformat(),'physical_gpu':1,'jobs':{'gamma_1':{'status':'waiting','phase':'checks'}}}
    atomic(out/'queue.json',state);generate(out);ensure_analysis(out);ensure_comparison(out)
    try:
        for stage in ['checks','smoke','training']:
            folder=out/'gamma_1'/('model' if stage=='training' else stage);folder.mkdir(parents=True,exist_ok=True)
            expected='checks.json' if stage=='checks' else 'submission.json'
            if (folder/expected).exists():continue
            atomic(folder/'launch_config.json',cfg);state['jobs']['gamma_1'].update(status='running',phase=stage);atomic(out/'queue.json',state);generate(out)
            while subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(10)
            env=dict(os.environ,SWANLAB_EXPERIMENT_NAME=f'causalmed-ode-ae-gamma1-{stage}-seed0')
            with (folder/'run.log').open('a') as log:
                worker=subprocess.Popen(command(source,ROOT/'CausalMed/outputs/level3_suite/gamma_1/public',folder,ROOT/'swanlog',stage,1),
                       stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
                state['jobs']['gamma_1']['pid']=worker.pid;atomic(out/'queue.json',state);print(f'ODE-AE: {stage}, GPU 1, PID {worker.pid}',flush=True);last=0
                try:
                    while worker.poll() is None:
                        if time.monotonic()-last>=30:generate(out);last=time.monotonic()
                        time.sleep(2)
                finally:
                    if worker.poll() is None:worker.terminate();worker.wait()
                if worker.returncode or not (folder/expected).exists():raise RuntimeError(f'ODE-AE {stage} failed; see {folder}/run.log')
        score(out,'gamma_1',state);state.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat());atomic(out/'queue.json',state);generate(out)
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');state['jobs']['gamma_1']['status']='failed';atomic(out/'queue.json',state);generate(out);raise


if __name__=='__main__':main()
