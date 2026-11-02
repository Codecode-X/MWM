"""One new seed-0 method run, reusing the already completed original control."""
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .prepare import atomic
from . import causal_history_report


def main():
    project=Path(__file__).resolve().parents[1];root=project/'mimic_experiment/results/weak_prior_ablation_cuda/targeted_causal_ablation';folder=root/'fresh_history_stop';folder.mkdir(exist_ok=True)
    lock=(folder/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((root/'manifest.json').read_text());source=folder/'source_snapshot';source.mkdir(exist_ok=True)
    for name in ['causal_history_fresh.py','causal_history_model.py']:
        path=project/'mimic_experiment'/name;dest=source/name
        if dest.exists() and dest.read_bytes()!=path.read_bytes():raise RuntimeError('Preserve the frozen from-scratch source')
        if not dest.exists():shutil.copy2(path,dest)
    atomic(folder/'manifest.json',{'control':'existing original seed0 complete run','original_config':manifest['reference_config'],'original_metrics':manifest['original_metrics'],
                                 'single_change':'detach cutoff state only for history generation for every training step from epoch1','physical_gpu':2,
                                 'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}})
    state={'status':'waiting_for_GPU','physical_gpu':2,'controller_pid':os.getpid()};atomic(folder/'queue.json',state)
    if (folder/'metrics.json').exists():state['status']='complete';atomic(folder/'queue.json',state);return
    gpu_lock=Path('/tmp/latentode_gpu_2.lock').open('a')
    while True:
        try:fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:time.sleep(10);continue
        busy=subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if busy:fcntl.flock(gpu_lock,fcntl.LOCK_UN);time.sleep(10);continue
        break
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','PYTHONPATH':manifest['analysis_source']+':'+manifest['source_snapshot'],'SWANLAB_LOG_DIR':str(project/'swanlog'),'PYTHONUNBUFFERED':'1'}
    with (root/'fresh_history_stop.log').open('a') as f:
        proc=subprocess.Popen([sys.executable,str(source/'causal_history_fresh.py'),'--out',str(root),'--gpu','2'],cwd=manifest['source_snapshot'],env=env,stdout=f,stderr=subprocess.STDOUT)
        state.update(status='running',pid=proc.pid);atomic(folder/'queue.json',state)
        while proc.poll() is None:importlib.reload(causal_history_report).generate(root);time.sleep(30)
    gpu_lock.close();state.update(status='complete' if proc.returncode==0 else 'failed',exit_code=proc.returncode);atomic(folder/'queue.json',state);importlib.reload(causal_history_report).generate(root)


if __name__=='__main__':main()
