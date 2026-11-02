"""GPU orchestration for the two targeted causal experiments."""
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
from . import causal_history_report


def main():
    if Path(sys.prefix).name!='ode':raise RuntimeError('Use the ode Conda environment')
    project=Path(__file__).resolve().parents[1];out=project/'mimic_experiment/results/weak_prior_ablation_cuda/targeted_causal_ablation';out.mkdir(exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=out.parent/'mechanism_diagnosis';previous=json.loads((prior/'manifest.json').read_text())
    files=['direction_gradient_common.py','mechanism_common.py','causal_history_model.py','causal_history_checks.py','causal_history_train.py','causal_module_cross.py']
    hashes={name:hashlib.sha256((project/'mimic_experiment'/name).read_bytes()).hexdigest() for name in files}
    version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16];analysis=out/'analysis_source'/version;analysis.mkdir(parents=True,exist_ok=True)
    for name in files:
        dest=analysis/name
        if not dest.exists():shutil.copy2(project/'mimic_experiment'/name,dest)
    initial=prior/'experiment1/snapshots/e045_b169_post.pt'
    manifest={'reference_config':previous['reference_config'],'source_snapshot':previous['source_snapshot'],'source_sha256':previous['source_sha256'],
              'analysis_source':str(analysis),'analysis_source_sha256':hashes,'training_start':str(initial),'training_start_sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),
              'stable_snapshot':previous['stable_45'],'abnormal_snapshot':previous['existing_abnormal_snapshot'],
              'original_history':str(project/'mimic_experiment/results/direction_full_batch_cuda/forward_shared/seed_0/history.json'),
              'original_metrics':str(project/'mimic_experiment/results/direction_full_batch_cuda/forward_shared/seed_0/metrics.json'),
              'training_budget':{'from_epoch':46,'through_epoch':81,'additional_epochs':36,'batch':128,'gradient_accumulation':1,'selection':'best validation overall within46..81','early_stopping':False},
              'module_cross':{'patient_index':8,'batch':'saved actual e46b121 replay batch','dtype':'float64','rtol':1e-8,'atol':1e-10,
                              'fixed_init_contexts':['stable','abnormal'],'non_exchanged_modules':'stable','factorial_interaction':'D*A/(B*C), reported alongside additive D-B-C+A'},
              'physical_gpus':{'control':0,'history_stop':1,'cross_stable_init':2,'cross_abnormal_init':3},'model_parameters':43280,'latent':28}
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())!=manifest:raise RuntimeError('Existing targeted study configuration changed')
    else:atomic(out/'manifest.json',manifest)
    state={'controller_pid':os.getpid(),'status':'running','phase':'cuda_checks','jobs':{}};processes={};locks={};streams={}
    def report():atomic(out/'queue.json',state);importlib.reload(causal_history_report).generate(out)
    def stop(signum,frame):
        for proc in processes.values():
            if proc.poll() is None:proc.terminate()
        state['status']='stopped';report();raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(name,script,gpu,extra):
        while True:
            h=Path(f'/tmp/latentode_gpu_{gpu}.lock').open('a')
            try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:h.close();state['jobs'][name]={'status':'waiting_for_GPU','physical_gpu':gpu};report();time.sleep(20);continue
            busy=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if busy:h.close();report();time.sleep(20);continue
            locks[name]=h;break
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':previous['source_snapshot'],'SWANLAB_LOG_DIR':str(project/'swanlog'),'PYTHONUNBUFFERED':'1'}
        command=[sys.executable,str(analysis/script),'--out',str(out),'--gpu',str(gpu),*extra];streams[name]=(out/(name+'.log')).open('a')
        proc=subprocess.Popen(command,cwd=previous['source_snapshot'],env=env,stdout=streams[name],stderr=subprocess.STDOUT);processes[name]=proc
        state['jobs'][name]={'status':'running','physical_gpu':gpu,'pid':proc.pid,'command':command};report();return proc
    if not (out/'checks.json').exists():
        proc=launch('checks','causal_history_checks.py',3,[])
        while proc.poll() is None:report();time.sleep(10)
        locks.pop('checks').close();state['jobs']['checks']['status']='complete' if proc.returncode==0 else 'failed'
        if proc.returncode:state['status']='failed_checks';report();raise RuntimeError('CUDA gradient checks failed before any training')
    state['phase']='paired_training_and_module_cross'
    specifications=[('control','causal_history_train.py',0,['--branch','control'],out/'control/metrics.json'),
                    ('history_stop','causal_history_train.py',1,['--branch','history_stop'],out/'history_stop/metrics.json'),
                    ('cross_stable_init','causal_module_cross.py',2,['--init-context','stable'],out/'module_cross/stable/results.json'),
                    ('cross_abnormal_init','causal_module_cross.py',3,['--init-context','abnormal'],out/'module_cross/abnormal/results.json')]
    for name,script,gpu,extra,done in specifications:
        if done.exists():state['jobs'][name]={'status':'complete','physical_gpu':gpu,'reused_existing_artifact':True};continue
        launch(name,script,gpu,extra)
    while True:
        for name,proc in processes.items():
            if name=='checks' or proc.poll() is None:continue
            state['jobs'][name].update(status='complete' if proc.returncode==0 else 'failed',exit_code=proc.returncode)
            if name in locks:locks.pop(name).close()
        report()
        if all(p.poll() is not None for p in processes.values()):break
        time.sleep(30)
    state['status']='complete' if all(v['status']=='complete' for v in state['jobs'].values()) else 'finished_with_failures';state['phase']='finished';report()


if __name__=='__main__':main()
