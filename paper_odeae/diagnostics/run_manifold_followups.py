"""Reproduce the four transported-manifold and clinical-tangent follow-ups."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .manifold_core import read,atomic,sha

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(ROOT/'paper_odeae/results/latent2x_both_manifold'));args=parser.parse_args()
    if Path(sys.prefix).name!='ode':raise RuntimeError('Use existing ode environment')
    out=Path(args.output).resolve();old=out/'source_snapshot';snapshot=out/'followup_source_snapshot';state={};processes={};streams={}
    if not snapshot.exists():
        for name in read(old/'source_hashes.json'):
            target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/name,target)
        name='paper_odeae/diagnostics/manifold_followup.py';shutil.copy2(ROOT/name,snapshot/name)
        atomic(snapshot/'source_hashes.json',{str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file() and p.suffix in ['.py','.json']})
    for name,digest in read(snapshot/'source_hashes.json').items():
        if sha(snapshot/name)!=digest:raise RuntimeError('Frozen follow-up source changed')
    for gpu,(dataset,stage) in enumerate([('adni','flow'),('mimic','flow'),('adni','clinical'),('mimic','clinical')]):
        key=dataset+'_'+stage;folder=out/key;folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():state[key]={'physical_gpu':gpu,'status':'complete'};continue
        cfg=read(out/'configs'/f'{dataset}_dynamics.json');cfg.update(stage=stage,physical_gpu=gpu,job_output=str(folder))
        path=out/'configs'/f'{key}.json';atomic(path,cfg)
        stream=(folder/'run.log').open('w');streams[key]=stream
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(snapshot),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        p=subprocess.Popen([sys.executable,'-u','-m','paper_odeae.diagnostics.manifold_followup','--config',str(path)],cwd=snapshot,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        processes[key]=p;state[key]={'pid':p.pid,'physical_gpu':gpu,'status':'running'}
    atomic(out/'followup_queue.json',state)
    try:
        while processes:
            for key,p in list(processes.items()):
                code=p.poll()
                if code is None:continue
                if code!=0 or not (out/key/'complete.json').exists():raise RuntimeError(f'{key} failed; inspect run.log')
                state[key].update(status='complete',exit_code=code);processes.pop(key);streams.pop(key).close();atomic(out/'followup_queue.json',state)
            time.sleep(2)
    finally:
        for p in processes.values():
            if p.poll() is None:p.terminate()
        for stream in streams.values():stream.close()
    atomic(out/'followup_manifest.json',{'status':'complete','finished_at':datetime.now(timezone.utc).isoformat(),
           'source_hashes':read(snapshot/'source_hashes.json'),'jobs':state,'training_performed':False})


if __name__=='__main__':main()
