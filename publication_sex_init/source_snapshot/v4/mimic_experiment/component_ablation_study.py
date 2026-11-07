"""CUDA0 queue for six exactly matched alpha=.1 component ablations."""
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .prepare import atomic
from . import component_ablation_report


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if Path(sys.prefix).name!='ode':raise RuntimeError('Use ode')
    project=Path(__file__).resolve().parents[1];out=project/'ablation/results/alpha01_components';out.mkdir(parents=True,exist_ok=True)
    if (out/'dispatch_revision.json').exists():
        from .component_ablation_parallel import main as parallel_main
        return parallel_main()
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=project/'mimic_experiment/results/weak_prior_ablation_cuda/targeted_causal_ablation'
    old=json.loads((previous/'manifest.json').read_text())
    refs={'adni':previous/'adni_update_alpha_01','mimic':previous/'update_alpha_01'}
    jobs=[]
    for component in ['no_prior','no_jacobian','separate_fields']:
        for dataset in ['adni','mimic']:
            name=dataset+'_'+component;folder=out/name;folder.mkdir(exist_ok=True);source=folder/'source_snapshot';source.mkdir(exist_ok=True)
            files=['component_ablation_model.py','component_ablation_loss.py','component_ablation_checks.py','component_ablation_train.py','fixed_update_model.py','fixed_update_checks.py']
            for filename in files:
                src=project/'mimic_experiment'/filename;dest=source/filename
                if dest.exists() and dest.read_bytes()!=src.read_bytes():raise RuntimeError('Preserve frozen component source')
                if not dest.exists():shutil.copy2(src,dest)
            reference=json.loads((refs[dataset]/'config.json').read_text());cfg=dict(reference)
            cfg.update(dataset=dataset,update_alpha=.1,physical_gpu=0,output=str(folder),max_epochs=150,
                       min_epochs=min(cfg['min_epochs'],150),gradient_accumulation_steps=1,
                       ablation_component=component,study='main_alpha01_component_ablation',generator_anchor='history cutoff',
                       encoder_direction='forward',prior_weight=0. if component=='no_prior' else .01,
                       jacobian_weight=0. if component=='no_jacobian' else .01,
                       collect_prior_predictions=True,collect_jacobian_states=True,
                       shared_vector_field=component!='separate_fields',target_parameters=44875 if dataset=='adni' else 43280)
            if dataset=='adni':
                m=json.loads((refs[dataset]/'manifest.json').read_text());native=Path(m['native_source_snapshot']);hashes=m['native_source_hashes']
            else:native=Path(old['source_snapshot']);hashes=old['source_sha256']
            for filename,sha in hashes.items():assert digest(native/filename)==sha
            assert digest(Path(cfg['data']))==cfg['dataset_sha256']
            assert all(cfg[k]==reference[k] for k in ['seed','latent','batch_size','input_dropout','lr','lr_decay','lr_decay_steps','mse_weight','cce_weight','l2_weight','rtol','atol','mlp_width','ode_width'])
            assert cfg['update_alpha']==.1 and cfg['latent']==28 and cfg['ode_width']==188
            cfgfile=folder/'launch_config.json'
            if cfgfile.exists():assert json.loads(cfgfile.read_text())==cfg
            atomic(cfgfile,cfg)
            manifest={'dataset':dataset,'component':component,'physical_gpu':0,'max_epochs':150,
                      'reference':str(refs[dataset]),'native_source':str(native),'native_source_hashes':hashes,
                      'source_sha256':{p.name:digest(p) for p in source.glob('*.py')},'same_total_parameters':True,
                      'target_parameters':cfg['target_parameters'],'unique_ode_parameters':10744,
                      'field_architecture':'188 shared' if component!='separate_fields' else 'encoder94 outputbias=False (5358) + generator94 outputbias=True (5386)',
                      'jacobian_scope':'mean over distinct fields at the same detached main-generation states',
                      'l2_scope':'sum of squares of all distinct ODE parameters; same coefficient0.02',
                      'config_differences':{k:{'main':reference.get(k),'ablation':cfg.get(k)} for k in reference.keys()|cfg.keys() if reference.get(k)!=cfg.get(k)}}
            atomic(folder/'manifest.json',manifest)
            jobs.append({'name':name,'dataset':dataset,'component':component,'config':str(cfgfile),'output':str(folder),'source':str(source),'native':str(native)})
    manifest={'main_model':'latent2x, chronological encoding, shared field, weak prior=.01, Jacobian=.01, fixed residual alpha=.1',
              'physical_gpu':0,'max_epochs':150,'jobs':jobs,'references':{k:str(v) for k,v in refs.items()},
              'direction_study':str(previous/'direction_alpha01'),'document':str(project/'ablation/Ablation.zh-CN.md'),
              'same_parameter_budget':{'adni':44875,'mimic':43280},'unique_ode_parameter_budget':10744}
    atomic(out/'manifest.json',manifest)
    state={'status':'running','controller_pid':os.getpid(),'physical_gpu':0,'phase':'waiting_for_GPU','jobs':{j['name']:{'status':'queued'} for j in jobs}}
    proc=None
    def report():atomic(out/'queue.json',state);importlib.reload(component_ablation_report).generate(out)
    def stop(signum,frame):
        if proc is not None and proc.poll() is None:proc.terminate();proc.wait()
        if state.get('current_job'):state['jobs'][state['current_job']]['status']='stopped_by_user'
        state['status']='stopped_by_user';atomic(out/'queue.json',state);raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);report()
    gpu=Path('/tmp/latentode_gpu_0.lock').open('a')
    while True:
        try:fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:time.sleep(10);continue
        if subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            fcntl.flock(gpu,fcntl.LOCK_UN);time.sleep(10);continue
        break
    # Preflight every dataset/component before committing to long training jobs.
    for stage in ['checks','training']:
        state['phase']=stage
        for job in jobs:
            folder=Path(job['output']);q=state['jobs'][job['name']]
            if (folder/'metrics.json').exists():q['status']='complete';continue
            if stage=='checks' and (folder/'checks.json').exists():q['status']='checks_passed';continue
            if stage=='training' and not (folder/'checks.json').exists():q['status']='failed_checks';continue
            state['current_job']=job['name'];q.update(status='checking' if stage=='checks' else 'running')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':job['native'],'SWANLAB_LOG_DIR':str(project/'swanlog'),'PYTHONUNBUFFERED':'1'}
            with (folder/(stage+'.log')).open('a') as stream:
                proc=subprocess.Popen([sys.executable,str(Path(job['source'])/'component_ablation_train.py'),'--config',job['config'],'--stage',stage],
                                      cwd=job['native'],env=env,stdout=stream,stderr=subprocess.STDOUT)
            q.update(pid=proc.pid);report();last_report=time.monotonic()
            while proc.poll() is None:
                time.sleep(2)
                if time.monotonic()-last_report>30:report();last_report=time.monotonic()
            q.update(status=('checks_passed' if stage=='checks' else 'complete') if proc.returncode==0 else 'failed_'+stage,exit_code=proc.returncode);report()
    gpu.close();state['phase']='finished';state['status']='complete' if all(q['status']=='complete' for q in state['jobs'].values()) else 'finished_with_failures';report()


if __name__=='__main__':main()
