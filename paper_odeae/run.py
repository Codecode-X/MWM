"""Four-GPU queue for independent model/seed jobs in the ode environment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from queue import Queue,Empty
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='paper_odeae/configs/local_available.json')
    p.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--models',nargs='+',choices=['ode_ae','deterministic','lstm'],default=['ode_ae','deterministic','lstm'])
    p.add_argument('--output');p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if len(set(a.gpus))!=len(a.gpus) or set(a.gpus)-{0,1,2,3}:raise ValueError('use unique physical GPUs from 0,1,2,3')
    cfg=json.loads(Path(a.config).read_text())
    root=Path(a.output or (str(Path(__file__).parent/'results/smoke') if a.smoke else cfg['output']))
    queue=Queue()
    jobs=[(m,s) for s in a.seeds for m in a.models]
    if a.smoke:jobs=[('ode_ae',0),('deterministic',0),('lstm',0),('ode_ae',1)]
    for job in jobs:queue.put(job)
    expected={**cfg,'smoke':a.smoke}
    if a.smoke:expected.update(max_epochs=1,min_epochs=0,patience=1,batch_size=4)
    def worker(gpu):
        while True:
            try:model,seed=queue.get_nowait()
            except Empty:return
            output=root/model/f'seed_{seed}'
            if (output/'metrics.json').exists():
                saved=json.loads((output/'config.json').read_text())
                request={**expected,'model':model,'seed':seed,'output':str(output)}
                if saved!=request:raise ValueError(f'existing configuration differs: {output}')
                hw=json.loads((output/'hardware.json').read_text())
                if hw['dataset_sha256']!=hashlib.sha256(Path(cfg['data']).read_bytes()).hexdigest():
                    raise ValueError('cached dataset changed; choose a new output directory')
                continue
            output.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            cmd=[sys.executable,'-u','-m','paper_odeae.train','--config',a.config,'--model',model,'--seed',str(seed),'--output',str(output)]
            if a.smoke:cmd.append('--smoke')
            print(f'physical GPU {gpu}: {model}, seed {seed}',flush=True)
            with (output/'train.log').open('w') as f:
                subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        futures=[pool.submit(worker,gpu) for gpu in a.gpus]
        for future in futures:future.result()
    subprocess.run([sys.executable,'-m','paper_odeae.report','--root',str(root)],check=True)


if __name__=='__main__':main()
