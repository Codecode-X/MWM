"""Schedule the existing exact-budget jobs on idle physical CUDA devices.

Only orchestration changes: frozen model, loss, checks and trainer are reused.
An active worker can be adopted without interrupting its training process.
"""
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .prepare import atomic
from . import component_ablation_report


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def alive(pid):
    path = Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split(') ', 1)[1].split()[0] != 'Z'


def worker_matches(pid, job):
    if not pid or not alive(pid):
        return False
    argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    return (str(Path(job['source'])/'component_ablation_train.py').encode() in argv
            and str(job['config']).encode() in argv)


def gpu_is_idle(gpu):
    return not subprocess.check_output(
        ['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
         '--format=csv,noheader,nounits'], text=True).strip()


def main():
    if Path(sys.prefix).name != 'ode':
        raise RuntimeError('Use the existing ode environment')
    project = Path(__file__).resolve().parents[1]
    out = project/'ablation/results/alpha01_components'
    lock = (out/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    revision = read(out/'dispatch_revision.json')
    manifest = read(out/'manifest.json')
    previous = read(out/'queue.json', {})
    jobs = {j['name']: j for j in manifest['jobs']}
    priority = revision['priority']
    assert set(priority) == set(jobs) and len(priority) == len(jobs)
    gpus = revision['physical_gpus']
    assert set(gpus) == {0, 1, 2, 3}
    available = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'], text=True)
    devices = {int(row[0]): row[1].strip() for row in csv.reader(available.splitlines())}
    assert all('4090' in devices[gpu] for gpu in gpus)
    # Files and statistics only; this controller never imports a model or torch.
    hashes = {}
    for name, job in jobs.items():
        folder = Path(job['output']); cfg = read(Path(job['config']))
        m = read(folder/'manifest.json')
        assert cfg['max_epochs'] == 150 and cfg['gradient_accumulation_steps'] == 1
        assert cfg['target_parameters'] == manifest['same_parameter_budget'][job['dataset']]
        for base, entries in [(Path(job['source']), m['source_sha256']),
                              (Path(job['native']), m['native_source_hashes']),
                              (Path(cfg['data']).parent, {Path(cfg['data']).name: cfg['dataset_sha256']})]:
            for filename, expected in entries.items():
                path = base/filename
                if path not in hashes:
                    hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
                assert hashes[path] == expected, str(path)
    manifest.update(physical_gpus=gpus, execution='one worker per idle physical GPU',
                    dispatch_priority=priority, dispatch_revision=str(out/'dispatch_revision.json'))
    manifest['physical_gpu'] = None
    atomic(out/'manifest.json', manifest)
    state = {'status': 'running', 'controller_pid': os.getpid(),
             'physical_gpus': gpus, 'phase': 'parallel_training',
             'priority': priority, 'jobs': {}, 'started_at_utc': now()}
    running = {}

    def report():
        state['current_jobs'] = list(running)
        state['updated_at_utc'] = now()
        atomic(out/'queue.json', state)
        importlib.reload(component_ablation_report).generate(out)

    def stop(signum, frame):
        for name, worker in running.items():
            if worker_matches(worker['pid'], jobs[name]):
                os.kill(worker['pid'], signal.SIGTERM)
            state['jobs'][name]['status'] = 'interrupted'
        state['status'] = 'interrupted'
        atomic(out/'queue.json', state)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for name, job in jobs.items():
        folder = Path(job['output'])
        cfg = read(Path(job['config']))
        old = previous.get('jobs', {}).get(name, {})
        q = state['jobs'][name] = {'status': 'queued', 'physical_gpu': cfg['physical_gpu']}
        if worker_matches(old.get('pid'), job):
            gpu = cfg['physical_gpu']
            environment = Path(f"/proc/{old['pid']}/environ").read_bytes().split(b'\0')
            assert f'CUDA_VISIBLE_DEVICES={gpu}'.encode() in environment
            handle = Path(f'/tmp/latentode_gpu_{gpu}.lock').open('a')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert not any(w['gpu'] == gpu for w in running.values())
            stage = old.get('stage', 'checks' if old.get('status') == 'checking' else 'training')
            running[name] = {'pid': old['pid'], 'proc': None, 'gpu': gpu,
                             'lock': handle, 'stage': stage}
            q.update(status='running' if stage == 'training' else 'checking',
                     pid=old['pid'], stage=stage, adopted_without_restart=True,
                     adopted_at_utc=now())
        elif (folder/'metrics.json').exists():
            q['status'] = 'complete'
        elif read(folder/'stop_request.json', {}).get('prevent_automatic_resume'):
            q['status'] = 'stopped_by_user'
    report()

    def launch(name, gpu, handle, stage):
        job = jobs[name]; folder = Path(job['output'])
        q = state['jobs'][name]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONPATH': job['native'],
               'SWANLAB_LOG_DIR': str(project/'swanlog'), 'PYTHONUNBUFFERED': '1'}
        with (folder/(stage+'.log')).open('a') as stream:
            stream.write(f'\nParallel dispatch {now()} physical CUDA {gpu}, stage={stage}\n')
            stream.flush()
            proc = subprocess.Popen(
                [sys.executable, str(Path(job['source'])/'component_ablation_train.py'),
                 '--config', job['config'], '--stage', stage],
                cwd=job['native'], env=env, stdout=stream, stderr=subprocess.STDOUT)
        running[name] = {'pid': proc.pid, 'proc': proc, 'gpu': gpu, 'lock': handle, 'stage': stage}
        q.update(status='checking' if stage == 'checks' else 'running', pid=proc.pid,
                 physical_gpu=gpu, stage=stage, dispatched_at_utc=now())

    def assign(name, gpu, handle):
        job = jobs[name]; folder = Path(job['output']); path = Path(job['config'])
        cfg = read(path)
        if cfg['physical_gpu'] != gpu:
            # Preserve original metadata before changing only device assignment.
            for filename in ['launch_config.json', 'manifest.json', 'checks.json']:
                p = folder/filename; backup = folder/(filename+'.before_parallel')
                if p.exists() and not backup.exists():
                    backup.write_bytes(p.read_bytes())
            before = dict(cfg); cfg['physical_gpu'] = gpu
            assert {k:v for k,v in cfg.items() if k != 'physical_gpu'} == {
                k:v for k,v in before.items() if k != 'physical_gpu'}
            atomic(path, cfg)
            m = read(folder/'manifest.json'); m['physical_gpu'] = gpu
            m['config_differences']['physical_gpu'] = {'main': 0, 'ablation': gpu}
            m['dispatch_revision'] = str(out/'dispatch_revision.json')
            atomic(folder/'manifest.json', m)
        check = read(folder/'checks.json', {})
        stage = 'training' if check.get('status') == 'passed' and check.get('physical_gpu') == gpu else 'checks'
        launch(name, gpu, handle, stage)

    last_report = time.monotonic()
    while True:
        changed = False
        for name, job in jobs.items():
            request = read(Path(job['output'])/'stop_request.json', {})
            q = state['jobs'][name]
            if not request.get('prevent_automatic_resume') or q['status'] in ['complete', 'stopped_by_user', 'stopping_by_user']:
                continue
            if name in running:
                worker = running[name]
                if worker_matches(worker['pid'], job):
                    os.kill(worker['pid'], signal.SIGTERM)
                q['status'] = 'stopping_by_user'
            else:
                q['status'] = 'stopped_by_user'
            q['stop_request'] = str(Path(job['output'])/'stop_request.json')
            changed = True
        for name, worker in list(running.items()):
            proc = worker['proc']
            if (proc is not None and proc.poll() is None) or (proc is None and alive(worker['pid'])):
                continue
            q = state['jobs'][name]; folder = Path(jobs[name]['output'])
            code = proc.returncode if proc is not None else None
            if read(folder/'stop_request.json', {}).get('prevent_automatic_resume'):
                q.update(status='complete' if (folder/'metrics.json').exists() else 'stopped_by_user',
                         exit_code=code, finished_at_utc=now())
                worker['lock'].close(); del running[name]; changed = True
                continue
            if worker['stage'] == 'checks':
                check = read(folder/'checks.json', {})
                success = (code in [0, None] and check.get('status') == 'passed'
                           and check.get('physical_gpu') == worker['gpu'])
                if success:
                    launch(name, worker['gpu'], worker['lock'], 'training')
                    changed = True
                    continue
            else:
                success = (folder/'metrics.json').exists() and code in [0, None]
            q.update(status='complete' if success else 'failed_'+worker['stage'],
                     exit_code=code, finished_at_utc=now())
            worker['lock'].close(); del running[name]; changed = True
        pending = [name for name in priority if state['jobs'][name]['status'] == 'queued']
        for gpu in gpus:
            if not pending or any(w['gpu'] == gpu for w in running.values()):
                continue
            handle = Path(f'/tmp/latentode_gpu_{gpu}.lock').open('a')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close(); continue
            if not gpu_is_idle(gpu):
                handle.close(); continue
            name = pending.pop(0); assign(name, gpu, handle); changed = True
        if changed or time.monotonic()-last_report >= 30:
            report(); last_report = time.monotonic()
        if not running and not pending:
            break
        time.sleep(3)
    state['status'] = ('complete' if all(q['status'] == 'complete' for q in state['jobs'].values()) else
                       'finished_with_user_stops' if all(q['status'] in ['complete', 'stopped_by_user'] for q in state['jobs'].values()) else 'finished_with_failures')
    state['phase'] = 'finished'; report()


if __name__ == '__main__':
    main()
