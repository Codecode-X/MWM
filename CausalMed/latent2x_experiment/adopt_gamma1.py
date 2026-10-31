"""Finish the existing gamma=1 worker while cancelling the earlier multi-gamma queue."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import signal
import time

from .study import atomic,identity,score,ensure_analysis,DEFAULT
from .report import read,generate


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(DEFAULT));args=parser.parse_args();out=Path(args.output).resolve()
    handoff=read(out/'scope_handoff.json');original=handoff['original_controller'];oldpid=original['pid'];worker=handoff['running_worker_preserved'];worker_start=identity(worker)
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'running','started_at':original['started_at'],
           'physical_gpu':3,'active_scope':['gamma_1'],'adopted_at':datetime.now(timezone.utc).isoformat(),
           'previous_controller':oldpid,'jobs':{'gamma_1':{**original['jobs']['gamma_1'],'status':'running','phase':'training'}}}
    atomic(out/'queue.json',state);generate(out);ensure_analysis(out)
    try:
        while worker_start is not None and identity(worker)==worker_start:
            generate(out);time.sleep(10)
        submission=read(out/'gamma_1/model/submission.json',{})
        if submission.get('status')!='predictions_sealed':raise RuntimeError('Adopted worker exited before sealed predictions; restart the gamma=1-only entry point to resume')
        if identity(oldpid)==original['start_time']:os.kill(oldpid,signal.SIGKILL)
        score(out,'gamma_1',state)
        state.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat());atomic(out/'queue.json',state);generate(out)
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');atomic(out/'queue.json',state);generate(out);raise


if __name__=='__main__':main()
