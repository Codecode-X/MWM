"""Import saved history into SwanLab, optionally following an existing process.

This reads JSON statistics only. It does not load checkpoints or execute models.
"""
import argparse
import json
from pathlib import Path
import time
from .tracking import SwanLabTracker


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return None  # Training can be in the middle of a non-atomic JSON write.


def process_start(pid):
    try:
        raw=Path(f'/proc/{pid}/stat').read_text()
        tail=raw.rsplit(')',1)[1].split()
        if tail[0]=='Z':return None
        return tail[19]
    except (FileNotFoundError,ProcessLookupError):
        return None


def follow(directory,pid=None,interval=5.):
    directory=Path(directory)
    if (directory/'swanlab.json').exists():
        raise FileExistsError('run already has SwanLab monitoring; refusing a second writer')
    config=read_json(directory/'config.json')
    if config is None:raise ValueError('config.json must exist before history import')
    runtime=read_json(directory/'hardware.json') or {}
    owner=process_start(pid) if pid else None
    if pid and owner is None:raise ValueError('training process is not running')
    config={**config,'monitor_source':'existing_history_json', 'watched_training_pid':pid}
    with SwanLabTracker(directory,config,'odeae_reproduction',seed=config.get('seed'),runtime=runtime) as tracker:
        emitted=0
        while True:
            history=read_json(directory/'history.json')
            if history is not None:
                for row in history:
                    epoch=row['epoch']
                    if epoch<=emitted:continue
                    tracker.step=max(tracker.step,row.get('optimizer_steps',epoch))
                    tracker.epoch(epoch,{'loss':row.get('training_loss',row.get('train_loss'))},
                                  row.get('validation',{k.removeprefix('validation_'):v for k,v in row.items() if k.startswith('validation_')}),row.get('lr',config.get('lr',0.)),
                                  elapsed_seconds=row.get('elapsed_seconds',0.))
                    emitted=epoch
                results=read_json(directory/'metrics.json')
                if results is not None:
                    tracker.final(results)
                    return
            if not pid:return
            if process_start(pid)!=owner:
                # Allow a last filesystem flush before marking a failed owner.
                results=read_json(directory/'metrics.json')
                if results is not None:
                    tracker.final(results);return
                raise RuntimeError('training process ended without completed metrics.json')
            time.sleep(interval)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--pid',type=int)
    p.add_argument('--interval',type=float,default=5.)
    a=p.parse_args()
    if a.interval<1:raise ValueError('polling interval must be at least one second')
    follow(a.run,a.pid,a.interval)


if __name__=='__main__':main()
