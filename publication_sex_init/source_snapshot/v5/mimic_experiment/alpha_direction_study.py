"""Prepare direction experiments, honoring subsequent GPU/queue revisions."""
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


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if Path(sys.prefix).name != 'ode': raise RuntimeError('Use ode')
    project = Path(__file__).resolve().parents[1]
    root = project/'mimic_experiment/results/weak_prior_ablation_cuda/targeted_causal_ablation'
    study = root/'direction_alpha01'; study.mkdir(exist_ok=True)
    if (study/'execution_revision.json').exists():
        from .alpha_direction_parallel import main as revised_main
        return revised_main()
    lock = (study/'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base = json.loads((root/'manifest.json').read_text())
    forward_paths = {'adni': root/'adni_update_alpha_01', 'mimic': root/'update_alpha_01'}
    specs = [('adni_reverse', 'adni', 'reverse'), ('mimic_reverse', 'mimic', 'reverse'), ('mimic_forward_150', 'mimic', 'forward')]
    jobs = []
    for name, dataset, direction in specs:
        folder = study/name; folder.mkdir(exist_ok=True)
        source = folder/'source_snapshot'; source.mkdir(exist_ok=True)
        for filename in ['alpha_direction_model.py', 'alpha_direction_checks.py', 'alpha_direction_train.py', 'fixed_update_model.py', 'fixed_update_checks.py']:
            original = project/'mimic_experiment'/filename; dest = source/filename
            if dest.exists() and dest.read_bytes() != original.read_bytes(): raise RuntimeError('Frozen direction source changed')
            if not dest.exists(): shutil.copy2(original, dest)
        reference_encoder = Path(base['source_snapshot'])/'paper_odeae/history_encoders.py'
        reference_text = reference_encoder.read_text().replace('from .models import ', 'from paper_odeae.models import ')
        dest = source/'reference_history_encoders.py'
        if dest.exists() and dest.read_text() != reference_text: raise RuntimeError('Frozen reference encoder changed')
        if not dest.exists(): dest.write_text(reference_text)
        forward = forward_paths[dataset]; original_cfg = json.loads((forward/'config.json').read_text())
        cfg = {**original_cfg, 'output': str(folder), 'physical_gpu': 2, 'dataset': dataset,
               'history_read_direction': direction, 'effective_encoder_direction': 'backward' if direction=='reverse' else 'forward',
               'encoder_direction': 'forward', 'generator_anchor': 'first history visit' if direction=='reverse' else 'history cutoff',
               'max_epochs': 150, 'min_epochs': min(original_cfg['min_epochs'], 150), 'update_alpha': .1,
               'gradient_accumulation_steps': 1, 'study': 'alpha01_direction_budget150',
               'target_parameters': 44875 if dataset=='adni' else 43280}
        if direction == 'forward': cfg['replay_reference_history'] = str(forward/'history.json')
        cfgfile = folder/'launch_config.json'
        if cfgfile.exists(): assert json.loads(cfgfile.read_text()) == cfg
        atomic(cfgfile, cfg)
        if dataset == 'adni':
            forward_manifest = json.loads((forward/'manifest.json').read_text())
            native = Path(forward_manifest['native_source_snapshot']); hashes = forward_manifest['native_source_hashes']
        else:
            native = Path(base['source_snapshot']); hashes = base['source_sha256']
        for filename, sha in hashes.items(): assert digest(native/filename) == sha
        changes = {k: {'forward_reference': original_cfg.get(k), 'current': cfg.get(k)}
                   for k in original_cfg.keys() | cfg.keys() if original_cfg.get(k) != cfg.get(k)}
        assert all(cfg[k] == original_cfg[k] for k in ['seed', 'batch_size', 'lr', 'lr_decay', 'lr_decay_steps', 'latent', 'mlp_width', 'ode_width',
                     'prior_weight', 'jacobian_weight', 'l2_weight', 'mse_weight', 'cce_weight', 'input_dropout', 'rtol', 'atol', 'data'])
        manifest = {'dataset': dataset, 'direction': direction, 'physical_gpu': 2, 'max_epochs': 150,
                    'forward_reference': str(forward), 'native_source_snapshot': str(native), 'native_source_hashes': hashes,
                    'reference_reverse_encoder': str(reference_encoder), 'reference_reverse_encoder_sha256': digest(reference_encoder),
                    'source_sha256': {p.name: digest(p) for p in source.glob('*.py')}, 'config_changes': changes,
                    'architecture_and_training_hyperparameters_equal': True,
                    'reverse_semantics': 'read latest to earliest; generate without UPDATE from first visit toward future; unchanged causal weak-prior auxiliary pass with alpha=.1'}
        atomic(folder/'manifest.json', manifest)
        jobs.append({'name': name, 'dataset': dataset, 'direction': direction, 'output': str(folder), 'config': str(cfgfile), 'native': str(native), 'source': str(source)})
    manifest = {'jobs': jobs, 'physical_gpu': 2, 'execution': 'one worker at a time', 'max_epochs': 150,
                'ADNI_forward_reused': str(forward_paths['adni']), 'MIMIC_fair_forward': str(study/'mimic_forward_150'),
                'reason_for_MIMIC_forward_replay': 'previous alpha=.1 forward ended at237; best/last checkpoints are beyond150 and no best-within150 snapshot was retained'}
    atomic(study/'manifest.json', manifest)
    state = {'status': 'running', 'controller_pid': os.getpid(), 'physical_gpu': 2, 'jobs': {j['name']: {'status':'queued'} for j in jobs}}
    def report():
        atomic(study/'queue.json', state)
        for j in jobs: atomic(Path(j['output'])/'queue.json', {**state['jobs'][j['name']], 'physical_gpu':2, 'controller_pid':os.getpid()})
        importlib.reload(causal_history_report).generate(root)
    proc = None
    def stop(signum, frame):
        if proc is not None and proc.poll() is None: proc.terminate(); proc.wait()
        state['status'] = 'stopped_by_user'; atomic(study/'queue.json', state); raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    report()
    handle = Path('/tmp/latentode_gpu_2.lock').open('a')
    while True:
        try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: time.sleep(10); continue
        if subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True).strip():
            fcntl.flock(handle, fcntl.LOCK_UN); time.sleep(10); continue
        break
    for job in jobs:
        folder = Path(job['output'])
        if (folder/'metrics.json').exists(): state['jobs'][job['name']]['status']='complete'; report(); continue
        state['current_job'] = job['name']
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'2', 'PYTHONPATH':job['native'], 'SWANLAB_LOG_DIR':str(project/'swanlog'), 'PYTHONUNBUFFERED':'1'}
        with (folder/'run.log').open('a') as stream:
            proc = subprocess.Popen([sys.executable, str(Path(job['source'])/'alpha_direction_train.py'), '--config', job['config']],
                                    cwd=job['native'], env=env, stdout=stream, stderr=subprocess.STDOUT)
        state['jobs'][job['name']].update(status='running',pid=proc.pid); report()
        while proc.poll() is None: time.sleep(30); report()
        state['jobs'][job['name']].update(status='complete' if proc.returncode==0 else 'failed',exit_code=proc.returncode); report()
    handle.close(); state['status']='complete' if all(v['status']=='complete' for v in state['jobs'].values()) else 'finished_with_failures';report()


if __name__ == '__main__': main()
