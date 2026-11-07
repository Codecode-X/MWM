"""Reserve both direction-study GPUs before handing them to the full-batch study."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

from paper_odeae.scaling_study import identity
from .prepare import atomic
from .report import read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--gpu',type=int,default=3);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];old=root/'mimic_experiment/results/history_encoder_ablation_cuda'
    lease3=int(os.environ['MIMIC_GPU_LEASE_FD']);lease2=Path('/tmp/latentode_gpu_2.lock').open('a')
    fcntl.flock(lease2.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);os.set_inheritable(lease2.fileno(),True)
    q=read(old/'queue.json');pid=q['jobs']['reverse_shared']['pid'];start=identity(pid)
    if start is not None:
        cmd=Path('/proc',str(pid),'cmdline').read_bytes()
        if b'configs/reverse_shared.json' not in cmd:raise RuntimeError('Unexpected reverse worker')
        os.kill(pid,signal.SIGTERM)
        for _ in range(100):
            if identity(pid)!=start:break
            time.sleep(.1)
        else:raise RuntimeError('Reverse worker did not stop')
    run=old/'reverse_shared/seed_0';history=read(run/'history.json',[]);best=min(history,key=lambda x:x['validation']['overall']['reconstruction_error']) if history else None
    stamp=datetime.now(timezone.utc).isoformat()
    stopped={'status':'stopped_by_user','stopped_at':stamp,'physical_gpu':2,'worker_pid':pid,'epochs_completed':len(history),
             'best_epoch':best['epoch'] if best else None,'reason':'User requested retraining both directions from scratch with a full 128-patient batch and no gradient accumulation.',
             'preserved_checkpoints':[str(p) for p in [run/'best.pt',run/'last.pt'] if p.exists()]}
    atomic(run/'stopped_by_user.json',stopped)
    swan=read(run/'swanlab.json',{});swan.update(status='stopped_by_user',stopped_at=stamp);atomic(run/'swanlab.json',swan)
    q['jobs']['reverse_shared'].update(status='stopped_by_user',phase='user_stopped',stopped_at=stamp)
    q.update(status='stopped_by_user',stopped_at=stamp);atomic(old/'queue.json',q)
    out=root/'mimic_experiment/results/direction_full_batch_cuda';out.mkdir(exist_ok=True)
    atomic(out/'reservation.json',{'pid':os.getpid(),'start_time':identity(os.getpid()),'physical_gpus':[2,3],
           'status':'reserved_for_full_batch_retraining','old_reverse_stop':stopped,'lease_fds':{'2':lease2.fileno(),'3':lease3}})
    print(json.dumps({'old_reverse_stopped':stopped,'gpus_2_3_reserved':True}),flush=True)
    while not (out/'start_training.json').exists():time.sleep(1)
    env=dict(os.environ,MIMIC_FULL_BATCH_GPU2_FD=str(lease2.fileno()),MIMIC_FULL_BATCH_GPU3_FD=str(lease3))
    os.execve(sys.executable,[sys.executable,'-u','-m','mimic_experiment.direction_full_batch_study','--output',str(out)],env)


if __name__=='__main__':main()
