"""Orchestrate CUDA-only diagnostics and faithful checkpoint replay."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .prepare import atomic
from .direction_gradient_report import generate


def main():
    root=Path(__file__).resolve().parents[1];base=root/'mimic_experiment/results/direction_full_batch_cuda'
    out=base/'gradient_sensitivity_analysis';out.mkdir(exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    source=base/'source_snapshot';cfg=json.loads((base/'configs/forward_shared.json').read_text())
    hashes=json.loads((source/'source_hashes.json').read_text())
    if not (out/'manifest.json').exists():
        atomic(out/'manifest.json',{'reference_config':cfg,'source_snapshot':str(source),'source_sha256':hashes,
                                   'initial_checkpoint':str(base/'forward_shared/seed_0/best.pt'),
                                   'original_history':str(base/'forward_shared/seed_0/history.json'),
                                   'physical_gpu_assignments':{'gradients':1,'sensitivity':2,'replay':3},
                                   'fixed_gradients':{'train_batches':8,'validation_batches':2,'batch_size':128,'context_selection_epoch':41},
                                   'sensitivity':{'validation_patients':64,'selection_seed':7,'dimension':28},
                                   'replay_epochs':[42,65],'gradient_accumulation_steps':1})
        entries=[]
        for epoch,name in [(41,'best'),(81,'last')]:
            path=out/'checkpoints'/f'epoch_{epoch:03d}_original.pt';shutil.copy2(base/f'forward_shared/seed_0/{name}.pt',path)
            entries.append({'id':f'epoch_{epoch:03d}_original','epoch':epoch,'path':str(path),
                            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'origin':'original saved checkpoint'})
        atomic(out/'checkpoint_index.json',{'entries':entries,'producer_done':False})
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    analysis_names=['direction_gradient_common.py','direction_gradient_probe.py','direction_gradient_replay.py']
    analysis_hashes={n:hashlib.sha256((root/'mimic_experiment'/n).read_bytes()).hexdigest() for n in analysis_names}
    version=hashlib.sha256(json.dumps(analysis_hashes,sort_keys=True).encode()).hexdigest()[:16]
    analysis=out/'analysis_source'/version;analysis.mkdir(parents=True,exist_ok=True)
    for name in analysis_names:shutil.copy2(root/'mimic_experiment'/name,analysis/name)
    manifest=json.loads((out/'manifest.json').read_text());manifest.update(analysis_source=str(analysis),analysis_source_sha256=analysis_hashes);atomic(out/'manifest.json',manifest)
    index=json.loads((out/'checkpoint_index.json').read_text());index.update(producer_done=False);index.pop('producer_exit_code',None);atomic(out/'checkpoint_index.json',index)
    processes={};locks={};logs={};queue={'status':'running','controller_pid':os.getpid(),'jobs':{}}
    def stop(signum,frame):
        for p in processes.values():
            if p.poll() is None:p.terminate()
        queue['status']='stopped';atomic(out/'queue.json',queue)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for name,gpu in [('replay',3),('gradients',1),('sensitivity',2)]:
        handle=Path(f'/tmp/latentode_gpu_{gpu}.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks[name]=handle
        usage=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if usage:raise RuntimeError(f'GPU {gpu} is occupied: {usage}')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':str(source),'SWANLAB_LOG_DIR':str(root/'swanlog'),'PYTHONUNBUFFERED':'1'}
        script=analysis/('direction_gradient_replay.py' if name=='replay' else 'direction_gradient_probe.py')
        command=[sys.executable,str(script),'--out',str(out),'--gpu',str(gpu)]
        if name!='replay':command+=['--kind',name]
        logs[name]=(out/f'{name}.log').open('a')
        processes[name]=subprocess.Popen(command,cwd=source,env=env,stdout=logs[name],stderr=subprocess.STDOUT)
        queue['jobs'][name]={'physical_gpu':gpu,'pid':processes[name].pid,'status':'running'}
    atomic(out/'queue.json',queue);generate(out)
    while True:
        for name,p in processes.items():
            if p.poll() is not None:
                queue['jobs'][name].update(status='complete' if p.returncode==0 else 'failed',exit_code=p.returncode)
                if name in locks:locks.pop(name).close()
        if processes['replay'].poll() is not None:
            index=json.loads((out/'checkpoint_index.json').read_text())
            if not index.get('producer_done'):
                index.update(producer_done=True,producer_exit_code=processes['replay'].returncode);atomic(out/'checkpoint_index.json',index)
        atomic(out/'queue.json',queue);generate(out)
        if all(p.poll() is not None for p in processes.values()):break
        time.sleep(20)
    queue['status']='complete' if all(p.returncode==0 for p in processes.values()) else 'failed';atomic(out/'queue.json',queue);generate(out)


if __name__=='__main__':main()
