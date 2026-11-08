"""Launch MIMIC no-UPDATE and ADNI alpha=.1 on the freed GPUs."""
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
    if Path(sys.prefix).name != 'ode': raise RuntimeError('Use ode')
    project = Path(__file__).resolve().parents[1]
    root = project/'mimic_experiment/results/weak_prior_ablation_cuda/targeted_causal_ablation'
    lock = (root/'fixed_update_additions.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    original = json.loads((root/'manifest.json').read_text())
    jobs = [('mimic', 0., 1, 'update_alpha_00'), ('adni', .1, 3, 'adni_update_alpha_01')]
    processes, locks, state = {}, {}, {'status': 'running', 'controller_pid': os.getpid(), 'jobs': {}}
    def report():
        atomic(root/'fixed_update_additions_queue.json', state); importlib.reload(causal_history_report).generate(root)
    def stop(signum, frame):
        for name, proc in processes.items():
            if proc.poll() is None: proc.terminate(); proc.wait()
            q = state['jobs'][name]; q['status'] = 'stopped'; atomic(root/name/'queue.json', q)
        state['status'] = 'stopped'; atomic(root/'fixed_update_additions_queue.json', state); raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    for dataset, alpha, gpu, name in jobs:
        folder = root/name; folder.mkdir(exist_ok=True)
        source = folder/'source_snapshot'; source.mkdir(exist_ok=True)
        for filename in ['fixed_update_model.py', 'fixed_update_checks.py', 'fixed_update_train.py']:
            src = project/'mimic_experiment'/filename; dest = source/filename
            if dest.exists() and dest.read_bytes() != src.read_bytes(): raise RuntimeError('Frozen source changed')
            if not dest.exists(): shutil.copy2(src, dest)
        if dataset == 'mimic':
            cfg = dict(original['reference_config']); native = Path(original['source_snapshot']); hashes = original['source_sha256']
            reference = original['original_metrics']; target = 43280
        else:
            baseline = project/'paper_odeae/results/latent2x_loss_ablation/both/seed_0'
            cfg = json.loads((baseline/'config.json').read_text()); native = baseline.parent.parent/'source_snapshot'
            hashes = json.loads((native/'source_hashes.json').read_text()); reference = str(baseline/'metrics.json'); target = 44875
        for filename, sha in hashes.items():
            assert hashlib.sha256((native/filename).read_bytes()).hexdigest() == sha
        cfg.update(dataset=dataset, update_alpha=alpha, physical_gpu=gpu, output=str(folder), target_parameters=target,
                   max_epochs=150, min_epochs=min(cfg['min_epochs'], 150), gradient_accumulation_steps=1,
                   generator_anchor='history cutoff', study='fixed_update_budget150', learnable_update_alpha=False,
                   dataset_sha256=hashlib.sha256(Path(cfg['data']).read_bytes()).hexdigest())
        cfgfile = folder/'launch_config.json'
        if cfgfile.exists(): assert json.loads(cfgfile.read_text()) == cfg
        atomic(cfgfile, cfg)
        manifest = {'dataset': dataset, 'alpha': alpha, 'physical_gpu': gpu, 'max_epochs': 150,
                    'reference_metrics': reference, 'reference_config': cfg, 'native_source_snapshot': str(native),
                    'native_source_hashes': hashes, 'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')},
                    'single_change': 'fixed UPDATE alpha; original dataset-specific training/evaluation protocol'}
        atomic(folder/'manifest.json', manifest)
        q = {'status': 'waiting_for_GPU', 'controller_pid': os.getpid(), 'physical_gpu': gpu, 'dataset': dataset, 'alpha': alpha}
        state['jobs'][name] = q; atomic(folder/'queue.json', q); report()
        if (folder/'metrics.json').exists(): q['status'] = 'complete'; atomic(folder/'queue.json', q); continue
        handle = Path(f'/tmp/latentode_gpu_{gpu}.lock').open('a')
        while True:
            try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: time.sleep(10); continue
            if subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
                fcntl.flock(handle, fcntl.LOCK_UN); time.sleep(10); continue
            break
        locks[name] = handle
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONPATH': str(native), 'SWANLAB_LOG_DIR': str(project/'swanlog'), 'PYTHONUNBUFFERED': '1'}
        with (root/(name+'.log')).open('a') as stream:
            proc = subprocess.Popen([sys.executable, str(source/'fixed_update_train.py'), '--config', str(cfgfile)],
                                    cwd=native, env=env, stdout=stream, stderr=subprocess.STDOUT)
        processes[name] = proc; q.update(status='running', pid=proc.pid); atomic(folder/'queue.json', q); report()
    while True:
        for name, proc in processes.items():
            if proc.poll() is not None:
                state['jobs'][name].update(status='complete' if proc.returncode==0 else 'failed', exit_code=proc.returncode)
                atomic(root/name/'queue.json', state['jobs'][name])
                if name in locks: locks.pop(name).close()
        report()
        if all(p.poll() is not None for p in processes.values()): break
        time.sleep(30)
    state['status'] = 'complete' if all(q['status']=='complete' for q in state['jobs'].values()) else 'finished_with_failures'; report()


if __name__ == '__main__': main()
