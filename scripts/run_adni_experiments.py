#!/usr/bin/env python3
"""Run one independent training worker per physical GPU, then summarize."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    p.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3])
    p.add_argument('--imaging',action='store_true')
    p.add_argument('--matched',action='store_true',help='ODE-RNN active parameter count matched to native Latent ODE')
    p.add_argument('--output',default='results/adni_cuda')
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    variant = 'medicalnet' if args.imaging else 'structured'
    if args.matched:
        variant += '_matched'
    jobs = [(model,seed) for seed in args.seeds for model in ['latent_ode','ode_rnn']]
    if args.matched:
        jobs = [('ode_rnn',seed) for seed in args.seeds]
    queue=Queue()
    for job in jobs:
        queue.put(job)
    def worker(gpu):
        while True:
            try:
                model,seed=queue.get_nowait()
            except Empty:
                return
            output = Path(args.output)/variant/model/f'seed_{seed}'
            if (output/'metrics.json').exists():
                saved=json.loads((output/'config.json').read_text())
                if (saved['epochs']!=args.epochs or saved['seed']!=seed
                        or saved['model']!=model or saved['imaging']!=args.imaging):
                    raise ValueError(f'Existing experiment configuration differs: {output}')
                continue
            output.mkdir(parents=True,exist_ok=True)
            cmd = [sys.executable,'-u','-m','src.adni_train','--model',model,
                   '--seed',str(seed),'--epochs',str(args.epochs),'--output',str(output)]
            if args.imaging:
                cmd.append('--imaging')
            if args.matched:
                features = len(json.loads(Path('data/adni_cache/metadata.json').read_text())['features']) + (16 if args.imaging else 0)
                target = 41632 + 417*features
                choices = sorted((abs(3*l*l+(7*features+4+2*w)*l+w*w+2*w+features-target),
                                  abs(w-64),l,w) for l in range(64,97) for w in range(48,81))
                _,_,latent,width = choices[0]
                cmd.extend(['--latent',str(latent),'--width',str(width)])
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            print(f'GPU {gpu}: {variant} {model} seed {seed}',flush=True)
            with (output/'train.log').open('w') as log:
                subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(f'completed GPU {gpu}: {variant} {model} seed {seed}',flush=True)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker,gpu) for gpu in args.gpus]
        for future in futures:
            future.result()


if __name__=='__main__':
    main()
