"""Stop the two pre-policy frozen workers at 150; evaluate saved best on CUDA."""
import fcntl
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime,timezone
from .prepare import atomic
from . import causal_history_report


def main():
    project=Path(__file__).resolve().parents[1];root=project/'mimic_experiment/results/weak_prior_ablation_cuda/targeted_causal_ablation'
    lock=(root/'epoch_cap_watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((root/'manifest.json').read_text());state={'controller_pid':os.getpid(),'max_epochs':150,'jobs':{}}
    processes={};handles={};done=set();last_report=0
    while len(done)<2:
        for tag,gpu in [('02',0),('04',2)]:
            folder=root/('update_alpha_'+tag)
            if tag in done:continue
            if (folder/'metrics_150.json').exists():done.add(tag);state['jobs'][tag]={'status':'complete_at_150'};continue
            if tag in processes:
                proc=processes[tag]
                if proc.poll() is not None:
                    state['jobs'][tag].update(status='complete_at_150' if proc.returncode==0 else 'evaluation_failed',exit_code=proc.returncode)
                    handles.pop(tag).close();done.add(tag)
                continue
            history=json.loads((folder/'history.json').read_text());epoch=history[-1]['epoch']
            state['jobs'][tag]={'status':'watching','latest_epoch':epoch,'physical_gpu':gpu}
            if (folder/'metrics.json').exists() and epoch<=150:
                done.add(tag);state['jobs'][tag]['status']='completed_before_cap';continue
            if epoch<150:continue
            if not (folder/'epoch_cap_stop.json').exists():
                q=json.loads((folder/'queue.json').read_text())
                atomic(folder/'epoch_cap_stop.json',{'requested_at':datetime.now(timezone.utc).isoformat(),'max_epochs':150,
                       'last_completed_epoch':epoch,'reason':'apply user epoch cap to already-running frozen worker','controller_pid':q['controller_pid'],'worker_pid':q['pid']})
                try:
                    cmd=Path(f"/proc/{q['controller_pid']}/cmdline").read_bytes().decode()
                    if 'damped_update' not in cmd:raise RuntimeError('Unexpected controller')
                    # These pre-policy controllers have a reentrant reporting
                    # signal handler. Stop the worker, then release its parent
                    # locks without entering that legacy handler.
                    worker_cmd=Path(f"/proc/{q['pid']}/cmdline").read_bytes().decode()
                    if 'damped_update' not in worker_cmd:raise RuntimeError('Unexpected worker')
                    os.kill(q['pid'],signal.SIGTERM)
                    os.kill(q['controller_pid'],signal.SIGKILL)
                    q.update(status='stopped_at_epoch150',stop_reason='user epoch cap');atomic(folder/'queue.json',q)
                except FileNotFoundError:pass
            handle=Path(f'/tmp/latentode_gpu_{gpu}.lock').open('a')
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:handle.close();continue
            if subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():handle.close();continue
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':manifest['analysis_source']+':'+str(folder/'source_snapshot')+':'+manifest['source_snapshot']}
            with (folder/'epoch150_evaluation.log').open('a') as stream:
                proc=subprocess.Popen([sys.executable,str(project/'mimic_experiment/evaluate_alpha_cap.py'),'--out',str(root),'--tag',tag,'--gpu',str(gpu)],
                                      cwd=manifest['source_snapshot'],env=env,stdout=stream,stderr=subprocess.STDOUT)
            processes[tag]=proc;handles[tag]=handle;state['jobs'][tag].update(status='evaluating_best_within150',pid=proc.pid)
        atomic(root/'epoch_cap_watch.json',state)
        if time.monotonic()-last_report>30:importlib.reload(causal_history_report).generate(root);last_report=time.monotonic()
        time.sleep(1)
    state['status']='finished';atomic(root/'epoch_cap_watch.json',state);importlib.reload(causal_history_report).generate(root)


if __name__=='__main__':main()
