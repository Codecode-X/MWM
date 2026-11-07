"""Additional fixed-alpha jobs, one CUDA GPU per coefficient."""
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, choices=[.1, .2, .4, .5], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    assignments = {.1: 1, .2: 0, .4: 2, .5: 0}
    if Path(sys.prefix).name != 'ode' or args.gpu != assignments[args.alpha]:
        raise RuntimeError('Use ode and the assigned physical GPU: alpha .1->1, .2->0, .4->2, .5->0')
    project = Path(__file__).resolve().parents[1]
    root = project/'mimic_experiment/results/weak_prior_ablation_cuda/targeted_causal_ablation'
    tag = f'{round(args.alpha*10):02d}'; folder = root/('update_alpha_'+tag)
    folder.mkdir(exist_ok=True)
    lock = (folder/'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder/'stop_request.json').exists() or (folder/'epoch_cap_stop.json').exists():
        return
    original = json.loads((root/'manifest.json').read_text())
    source = folder/'source_snapshot'; source.mkdir(exist_ok=True)
    previous = json.loads((folder/'manifest.json').read_text()) if (folder/'manifest.json').exists() else None
    files = ['damped_update_model.py', 'damped_update_sweep_checks.py', 'damped_update_sweep_train.py']
    for name in files:
        path = project/'mimic_experiment'/name; dest = source/name
        if dest.exists():
            if previous and hashlib.sha256(dest.read_bytes()).hexdigest() != previous['source_sha256'][name]:
                raise RuntimeError('Frozen alpha sweep source hash changed')
            if not previous and dest.read_bytes() != path.read_bytes():
                raise RuntimeError('Unmanifested source differs from the current worker')
        else:
            shutil.copy2(path, dest)
    spec = {'control': 'existing original seed0 complete run', 'physical_gpu': args.gpu, 'alpha': args.alpha,
            'reference_alpha03_config': str(root/'update_alpha_03/config.json'),
            'reference_alpha03_source': str(root/'update_alpha_03/source_snapshot'),
            'original_config': original['reference_config'], 'original_metrics': original['original_metrics'],
            'single_change': 'fixed UPDATE alpha; full original history, future and regularizer gradients',
            'from_scratch': True, 'seed': 0, 'training_batch_size': 128, 'gradient_accumulation_steps': 1,
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}}
    if (folder/'manifest.json').exists() and json.loads((folder/'manifest.json').read_text()) != spec:
        raise RuntimeError('Existing sweep configuration changed')
    atomic(folder/'manifest.json', spec)
    state = {'status': 'waiting_for_GPU', 'physical_gpu': args.gpu, 'alpha': args.alpha, 'controller_pid': os.getpid()}
    def report():
        atomic(folder/'queue.json', state); importlib.reload(causal_history_report).generate(root)
    report()
    if (folder/'metrics.json').exists():
        state['status'] = 'complete'; report(); return
    proc = None
    def stop(signum, frame):
        if proc is not None and proc.poll() is None: proc.terminate(); proc.wait()
        state['status'] = 'stopped'; atomic(folder/'queue.json', state); raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    gpu_lock = Path(f'/tmp/latentode_gpu_{args.gpu}.lock').open('a')
    while True:
        try: fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: time.sleep(10); continue
        busy = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if busy: fcntl.flock(gpu_lock, fcntl.LOCK_UN); time.sleep(10); continue
        break
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu), 'PYTHONPATH': original['analysis_source']+':'+original['source_snapshot'],
           'SWANLAB_LOG_DIR': str(project/'swanlog'), 'PYTHONUNBUFFERED': '1'}
    with (root/f'update_alpha_{tag}.log').open('a') as stream:
        proc = subprocess.Popen([sys.executable, str(source/'damped_update_sweep_train.py'), '--out', str(root),
                                 '--gpu', str(args.gpu), '--alpha', str(args.alpha)], cwd=original['source_snapshot'],
                                env=env, stdout=stream, stderr=subprocess.STDOUT)
        state.update(status='running', pid=proc.pid); report()
        while proc.poll() is None: time.sleep(30); report()
    gpu_lock.close(); state.update(status='complete' if proc.returncode == 0 else 'failed', exit_code=proc.returncode); report()


if __name__ == '__main__':
    main()
