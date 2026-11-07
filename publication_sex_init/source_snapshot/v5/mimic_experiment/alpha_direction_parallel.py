"""Adopt the running ADNI job; train MIMIC on GPU3 and reuse old forward."""
import fcntl
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .prepare import atomic
from . import causal_history_report


def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z'


def main():
    if Path(sys.prefix).name!='ode':raise RuntimeError('Use ode')
    project=Path(__file__).resolve().parents[1];root=project/'mimic_experiment/results/weak_prior_ablation_cuda/targeted_causal_ablation';study=root/'direction_alpha01'
    lock=(study/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    revision=json.loads((study/'execution_revision.json').read_text());manifest=json.loads((study/'manifest.json').read_text())
    adni=study/'adni_reverse';mimic=study/'mimic_reverse';adni_pid=revision['adopted_adni_worker_pid']
    state={'status':'running','controller_pid':os.getpid(),'physical_gpus':{'adni_reverse':2,'mimic_reverse':3},
           'jobs':{'adni_reverse':{'status':'running','pid':adni_pid,'physical_gpu':2,'adopted_without_restart':True},
                   'mimic_reverse':{'status':'queued','physical_gpu':3},'mimic_forward_150':{'status':'cancelled_by_user'},
                   'mimic_forward_existing':{'status':'queued_evaluation_only','physical_gpu':3}}}
    handles={};proc=None;evaluation=None
    def report():
        atomic(study/'queue.json',state)
        for name,q in state['jobs'].items():
            folder=study/name;folder.mkdir(exist_ok=True);atomic(folder/'queue.json',{**q,'controller_pid':os.getpid()})
        importlib.reload(causal_history_report).generate(root)
    def stop(signum,frame):
        for child in [proc,evaluation]:
            if child is not None and child.poll() is None:child.terminate();child.wait()
        if alive(adni_pid):os.kill(adni_pid,signal.SIGTERM)
        state['status']='stopped_by_user';atomic(study/'queue.json',state);raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if alive(adni_pid):
        assert str(adni) in Path(f'/proc/{adni_pid}/cmdline').read_bytes().decode()
        h=Path('/tmp/latentode_gpu_2.lock').open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);handles['adni']=h
    else:state['jobs']['adni_reverse']['status']='complete' if (adni/'metrics.json').exists() else 'failed'
    report()
    h=Path('/tmp/latentode_gpu_3.lock').open('a')
    while True:
        try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:time.sleep(10);continue
        if subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            fcntl.flock(h,fcntl.LOCK_UN);time.sleep(10);continue
        break
    handles['mimic']=h
    m=json.loads((mimic/'manifest.json').read_text());env={**os.environ,'CUDA_VISIBLE_DEVICES':'3','PYTHONPATH':m['native_source_snapshot'],'SWANLAB_LOG_DIR':str(project/'swanlog'),'PYTHONUNBUFFERED':'1'}
    if not (mimic/'metrics.json').exists():
        with (mimic/'run.log').open('a') as stream:
            proc=subprocess.Popen([sys.executable,str(mimic/'source_snapshot/alpha_direction_train.py'),'--config',str(mimic/'launch_config.json')],
                                  cwd=m['native_source_snapshot'],env=env,stdout=stream,stderr=subprocess.STDOUT)
        state['jobs']['mimic_reverse'].update(status='running',pid=proc.pid)
    else:state['jobs']['mimic_reverse']['status']='complete'
    report()
    while True:
        if not alive(adni_pid):
            state['jobs']['adni_reverse']['status']='complete' if (adni/'metrics.json').exists() else 'failed'
            if 'adni' in handles:handles.pop('adni').close()
        if proc is not None and proc.poll() is not None:state['jobs']['mimic_reverse'].update(status='complete' if proc.returncode==0 else 'failed',exit_code=proc.returncode)
        if state['jobs']['mimic_reverse']['status'] in ['complete','failed'] and evaluation is None:
            dest=study/'mimic_forward_existing';dest.mkdir(exist_ok=True)
            if (dest/'metrics.json').exists():state['jobs']['mimic_forward_existing']['status']='complete'
            else:
                base=json.loads((root/'manifest.json').read_text());ev_env={**env,'PYTHONPATH':base['analysis_source']+':'+str(root/'update_alpha_01/source_snapshot')+':'+base['source_snapshot']}
                with (dest/'evaluation.log').open('a') as stream:
                    evaluation=subprocess.Popen([sys.executable,str(project/'mimic_experiment/evaluate_existing_alpha01.py'),'--out',str(root),'--gpu','3'],
                                                cwd=base['source_snapshot'],env=ev_env,stdout=stream,stderr=subprocess.STDOUT)
                state['jobs']['mimic_forward_existing'].update(status='evaluating_only',pid=evaluation.pid)
        if evaluation is not None and evaluation.poll() is not None:
            state['jobs']['mimic_forward_existing'].update(status='complete' if evaluation.returncode==0 else 'failed',exit_code=evaluation.returncode)
        if state['jobs']['mimic_forward_existing']['status'] in ['complete','failed'] and 'mimic' in handles:handles.pop('mimic').close()
        report()
        if all(state['jobs'][name]['status'] in ['complete','failed'] for name in ['adni_reverse','mimic_reverse','mimic_forward_existing']):break
        time.sleep(30)
    state['status']='finished';report()


if __name__=='__main__':main()
