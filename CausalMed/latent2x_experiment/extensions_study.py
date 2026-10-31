"""Two independent gamma=1 jobs: matched deterministic ODE-AE and native HealthFlux-KL."""
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

DEFAULT=ROOT/'CausalMed/outputs/gamma1_extensions_cuda'
JOBS={'deterministic_matched':('deterministic_matched_gamma1_cuda1',1,'ode','deterministic_22545（同预算）'),
      'healthflux_kl':('healthflux_kl_gamma1_cuda3',3,'healthflux','HealthFlux-KL')}


def prepare(master):
    ref=read(ROOT/'CausalMed/outputs/latent2x_both_cuda3/manifest.json');public=ROOT/'CausalMed/outputs/level3_suite/gamma_1/public'
    public_sha={p.name:digest(p) for p in sorted(public.iterdir()) if p.is_file()}
    if public_sha!=ref['public_data_sha256']['gamma_1']:raise RuntimeError('Public comparison data changed')
    files={
      **{f'CausalMed/latent2x_experiment/{name}':f'latent2x_experiment/{name}' for name in ['__init__.py','data.py','model.py','odeae_model.py','healthflux_model.py','rollout.py','train.py']},
      **{f'healthflux/{name}':f'healthflux/{name}' for name in ['__init__.py','model.py']},
      **{f'paper_odeae/{name}':f'paper_odeae/{name}' for name in ['__init__.py','models.py','solver.py','jacobian.py','reference/tsit5_tableau.json']},
      **{f'src/{name}':f'src/{name}' for name in ['__init__.py','tracking.py','swanlab_local.py']},
      **{f'CausalMed/model_client/causalmed_client/{name}':f'causalmed_client/{name}' for name in ['__init__.py','dataset_adapter.py','counterfactual_policies.py']}}
    source=master/'model_source';source.mkdir(exist_ok=True);hashes=read(master/'model_source_hashes.json')
    if not hashes:
        for original,target in files.items():
            destination=source/target;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/original,destination)
        hashes={target:digest(source/target) for target in files.values()};atomic(master/'model_source_hashes.json',hashes)
    for filename,sha in hashes.items():
        if digest(source/filename)!=sha:raise RuntimeError('Frozen extension model source changed')
    configurations={};folders={}
    for key,(dirname,gpu,environment,title) in JOBS.items():
        folder=ROOT/'CausalMed/outputs'/dirname;folder.mkdir(parents=True,exist_ok=True);folders[key]=folder
        local_source=folder/'model_source'
        if not local_source.exists():local_source.symlink_to(source,target_is_directory=True)
        base={**ref['configs']['gamma_1'],'prior_weight':0.,'jacobian_weight':0.,'physical_gpu':gpu,'variant':'gamma_1'}
        if key=='deterministic_matched':
            cfg={**base,'model':'deterministic_matched_action_update','latent':6,'mlp_width':411,'update_width':412,'ode_width':317,
                 'kl_weight':0.,'target_parameters':31314,'source_family':'deterministic_22545',
                 'capacity_change':'CausalMed exact 31,314 parameter budget; six-dimensional backward deterministic encoder; two independent fields',
                 'action_update':'encoder conditional residual and independent generator treatment-conditional residual'}
            # 8w_init + 23w_update + 10w_decoder + 15w_action + 26w_ode + 33.
            count=8*411+23*412+10*411+15*411+26*317+33
            if count!=31314:raise AssertionError('Deterministic parameter matching failed')
        else:
            cfg={**base,'model':'healthflux_kl_action_update','kl_weight':.01,'target_parameters':4361412,
                 'deterministic_dim':256,'stochastic_dim':128,'context_dim':128,'age_dim':64,'observation_dim':256,'internal_width':384,
                 'decoder_widths':[1024,512],'dropout':.1,'rk4_step_years':.25,'native_capacity':True,
                 'time_condition':'public elapsed years; 15-feature time embedding; static c is separate',
                 'action_update':'residual difference of treatment-embedding conditioned native GRU updates',
                 'capacity_change':'native 256+128 dimensions; scalar public context and three numerical observation groups'}
        manifest={'model_name':title,'model_source_sha256':hashes,'public_data_sha256':{'gamma_1':public_sha},'configs':{'gamma_1':cfg},
                  'order':['gamma_1'],'physical_gpu':gpu,'environment':environment,
                  'reference_output':str(ROOT/'CausalMed/outputs/latent2x_both_cuda3'),
                  'blind_execution':'bubblewrap public-only model mounts; independent post-submission oracle evaluation'}
        if (folder/'manifest.json').exists() and read(folder/'manifest.json')!=manifest:raise RuntimeError('Extension model configuration changed')
        atomic(folder/'manifest.json',manifest);configurations[key]=cfg
    atomic(master/'manifest.json',{'models':list(JOBS),'source_sha256':hashes,'outputs':{key:str(folder) for key,folder in folders.items()},'gamma':1,
           'deterministic_parameter_target':31314,'healthflux_capacity':'native 256+128','seed':0})
    return source,configurations,folders


def ensure_comparison():
    out=ROOT/'CausalMed/outputs/gamma1_model_comparison';out.mkdir(parents=True,exist_ok=True);previous=read(out/'queue.json',{})
    if previous.get('pid') and identity(previous['pid'])==previous.get('start_time'):return
    with (out/'run.log').open('a') as log:
        process=subprocess.Popen([str(PYTHON),'-u','-m','CausalMed.latent2x_experiment.compare','--output',str(out)],cwd=ROOT,
                     stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(f'Four-model comparison monitor PID {process.pid}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(DEFAULT));args=parser.parse_args()
    if Path(sys.prefix).name!='ode':raise RuntimeError('Use ode for orchestration')
    master=Path(args.output).resolve();master.mkdir(parents=True,exist_ok=True);previous=read(master/'queue.json',{})
    if previous.get('pid') and identity(previous['pid'])==previous.get('start_time'):raise RuntimeError('Extension study already running')
    source,configs,folders=prepare(master);started=datetime.now(timezone.utc).isoformat();running={};streams={};states={}
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'running','started_at':started,'jobs':{}}
    for key,folder in folders.items():
        states[key]={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'running','started_at':started,'physical_gpu':JOBS[key][1],
                     'jobs':{'gamma_1':{'status':'waiting','phase':'checks'}}}
        atomic(folder/'queue.json',states[key]);generate(folder);ensure_analysis(folder)
        state['jobs'][key]={'status':'waiting','phase':'checks','physical_gpu':JOBS[key][1]}
    atomic(master/'queue.json',state);ensure_comparison()
    def done(key,stage):
        folder=folders[key]/'gamma_1'/('model' if stage=='training' else stage)
        return (folder/('checks.json' if stage=='checks' else 'submission.json')).exists()
    def launch(key,stage):
        folder=folders[key]/'gamma_1'/('model' if stage=='training' else stage);folder.mkdir(parents=True,exist_ok=True)
        atomic(folder/'launch_config.json',configs[key]);_,gpu,environment,_=JOBS[key]
        while subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(5)
        stream=(folder/'run.log').open('a');streams[key]=stream
        env=dict(os.environ,SWANLAB_EXPERIMENT_NAME=f'causalmed-{key}-gamma1-{stage}-seed0')
        process=subprocess.Popen(command(source,ROOT/'CausalMed/outputs/level3_suite/gamma_1/public',folder,ROOT/'swanlog',stage,gpu,environment),
                stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
        running[key]=(process,stage);state['jobs'][key].update(status='running',phase=stage,pid=process.pid)
        states[key]['jobs']['gamma_1'].update(status='running',phase=stage,pid=process.pid);atomic(folders[key]/'queue.json',states[key]);atomic(master/'queue.json',state)
        print(f'{key}: {stage}, physical GPU {gpu}, {environment}, PID {process.pid}',flush=True)
    phases=['checks','smoke','training']
    try:
        for key in JOBS:
            if (folders[key]/'gamma_1/evaluation/action_off/metrics.json').exists():
                states[key].update(status='complete');states[key]['jobs']['gamma_1'].update(status='complete',phase='scored');atomic(folders[key]/'queue.json',states[key]);state['jobs'][key].update(status='complete');continue
            stage=next((p for p in phases if not done(key,p)),None)
            if stage:launch(key,stage)
            else:score(folders[key],'gamma_1',states[key]);states[key]['status']='complete';atomic(folders[key]/'queue.json',states[key]);state['jobs'][key]['status']='complete'
        last=0
        while running:
            for key,(process,stage) in list(running.items()):
                code=process.poll()
                if code is None:continue
                streams.pop(key).close();running.pop(key)
                if code or not done(key,stage):
                    state['jobs'][key].update(status='failed',exit_code=code);states[key]['status']='failed';states[key]['jobs']['gamma_1'].update(status='failed',phase=stage)
                    atomic(folders[key]/'queue.json',states[key]);print(f'{key}: failed in {stage}; inspect run.log',flush=True)
                elif stage!='training':launch(key,phases[phases.index(stage)+1])
                else:
                    score(folders[key],'gamma_1',states[key]);states[key].update(status='complete',finished_at=datetime.now(timezone.utc).isoformat());atomic(folders[key]/'queue.json',states[key]);state['jobs'][key].update(status='complete',phase='scored')
                atomic(master/'queue.json',state)
            if time.monotonic()-last>=30:
                for folder in folders.values():generate(folder)
                last=time.monotonic()
            time.sleep(2)
        state.update(status='complete' if all(job['status']=='complete' for job in state['jobs'].values()) else 'failed',finished_at=datetime.now(timezone.utc).isoformat());atomic(master/'queue.json',state)
        for folder in folders.values():generate(folder)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),finished_at=datetime.now(timezone.utc).isoformat());atomic(master/'queue.json',state)
        for key in running:
            states[key].update(status='failed',error=repr(exc));atomic(folders[key]/'queue.json',states[key])
        raise
    finally:
        for process,_ in running.values():
            if process.poll() is None:process.terminate()
        for stream in streams.values():stream.close()


if __name__=='__main__':main()
