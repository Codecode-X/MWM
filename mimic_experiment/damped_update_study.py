"""Run the fixed alpha=.3 ablation on physical GPU 3 and update the report."""
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
    if Path(sys.prefix).name != 'ode':
        raise RuntimeError('Use the ode Conda environment')
    project = Path(__file__).resolve().parents[1]
    root = project/'mimic_experiment/results/weak_prior_ablation_cuda/targeted_causal_ablation'
    folder = root/'update_alpha_03'
    folder.mkdir(exist_ok=True)
    lock = (folder/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder/'stop_request.json').exists():
        atomic(folder/'queue.json', {'status':'stopped_by_user','physical_gpu':3,'reason':'preserved user stop request'});return
    original = json.loads((root/'manifest.json').read_text())
    source = folder/'source_snapshot'; source.mkdir(exist_ok=True)
    for name in ['damped_update_model.py', 'damped_update_checks.py', 'damped_update_train.py']:
        path = project/'mimic_experiment'/name; dest = source/name
        if dest.exists() and dest.read_bytes() != path.read_bytes():
            raise RuntimeError('Preserve the frozen alpha03 source')
        if not dest.exists():
            shutil.copy2(path, dest)
    spec = {'control': 'existing original seed0 complete run', 'physical_gpu': 3, 'alpha': .3,
            'original_config': original['reference_config'], 'original_metrics': original['original_metrics'],
            'history_gradient': 'full original gradient; no stop-gradient',
            'single_change': 'post-UPDATE state = 0.7*pre-UPDATE state + 0.3*original UPDATE network output',
            'from_scratch': True, 'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}}
    if (folder/'manifest.json').exists() and json.loads((folder/'manifest.json').read_text()) != spec:
        raise RuntimeError('Existing ablation configuration changed')
    atomic(folder/'manifest.json', spec)
    state = {'status': 'waiting_for_GPU', 'physical_gpu': 3, 'controller_pid': os.getpid()}
    def report():
        atomic(folder/'queue.json', state)
        importlib.reload(causal_history_report).generate(root)
    report()
    if (folder/'metrics.json').exists():
        state['status'] = 'complete'; report(); return
    proc = None
    def stop(signum, frame):
        if proc is not None and proc.poll() is None:
            proc.terminate(); proc.wait()
        state['status'] = 'stopped'; atomic(folder/'queue.json', state); raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    gpu_lock = Path('/tmp/latentode_gpu_3.lock').open('a')
    while True:
        try:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            time.sleep(10); continue
        busy = subprocess.check_output(['nvidia-smi', '-i', '3', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if busy:
            fcntl.flock(gpu_lock, fcntl.LOCK_UN); time.sleep(10); continue
        break
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '3', 'PYTHONPATH': original['analysis_source']+':'+original['source_snapshot'],
           'SWANLAB_LOG_DIR': str(project/'swanlog'), 'PYTHONUNBUFFERED': '1'}
    with (root/'update_alpha_03.log').open('a') as stream:
        proc = subprocess.Popen([sys.executable, str(source/'damped_update_train.py'), '--out', str(root), '--gpu', '3'],
                                cwd=original['source_snapshot'], env=env, stdout=stream, stderr=subprocess.STDOUT)
        state.update(status='running', pid=proc.pid); report()
        while proc.poll() is None:
            time.sleep(30); report()
    gpu_lock.close()
    state.update(status='complete' if proc.returncode == 0 else 'failed', exit_code=proc.returncode)
    report()


if __name__ == '__main__':
    main()
