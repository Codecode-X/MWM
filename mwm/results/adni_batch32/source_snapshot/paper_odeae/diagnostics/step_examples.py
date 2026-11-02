"""Collect actual adaptive-step traces on four CUDA GPUs."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import torch
from torch.utils.data import DataLoader
from paper_odeae.models import PaperModel
from paper_odeae.data import Prefixes,collate
from paper_odeae.train import device_batch
from paper_odeae.solver import integrate,solver_statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper_odeae/results/step_examples'


@torch.no_grad()
def worker(gpu):
    if not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise RuntimeError('CUDA required')
    torch.set_num_threads(1)
    examples=[]
    if gpu>=2:
        rtol=1e-4 if gpu==2 else 1e-5
        with solver_statistics(trace=True) as stats:
            integrate(lambda z:3*z,torch.ones(1,1,device='cuda'),torch.tensor([2.],device='cuda'),rtol=rtol,atol=rtol/100)
        examples=[{'label':f'Synthetic dz/dt=3z, rtol={rtol}','synthetic':True,'physical_gpu':gpu,
                   'time_unit':'arbitrary, 2 units total','rtol':rtol,**stats['calls'][0]}]
    else:
        run=ROOT/('paper_odeae/results/local_available/deterministic/seed_0' if gpu==0 else 'paper_odeae/results/capacity_matched/deterministic_22545/seed_0')
        checkpoint=torch.load(run/'best.pt',map_location='cuda',weights_only=False);cfg=checkpoint['config']
        payload=pickle.loads((ROOT/cfg['data']).read_bytes())
        model=PaperModel(payload['schema'],'deterministic',cfg['latent'],cfg['ode_width'],cfg['mlp_width'],cfg['rtol'],cfg['atol']).cuda().eval()
        model.load_state_dict(checkpoint['state_dict']);original=model.solve;active=None
        def tagged(func,state,delta):
            active['phase']='encoder' if func is model.backward_ode else 'generator'
            return original(func,state,delta)
        model.solve=tagged
        loader=DataLoader(Prefixes([s for s in payload['sequences'] if s['split']=='test']),batch_size=32,collate_fn=collate)
        found=set()
        for batch_index,cpu in enumerate(loader):
            with solver_statistics(trace=True) as stats:
                active=stats;model(device_batch(cpu),sample=False)
            for call in stats['calls']:
                if not call['attempts']:continue
                wanted=[]
                if gpu==0 and call['phase']=='generator':
                    short=next((i for i,d in enumerate(call['delta']) if .2<d<.3),None)
                    long=next((i for i,d in enumerate(call['delta']) if 1.8<d<2.2),None)
                    if short is not None:wanted.append(('real_short',short))
                    if long is not None:wanted.append(('real_long',long))
                if gpu==0 and call['phase']=='encoder':
                    negative=next((i for i,d in enumerate(call['delta']) if d<0),None)
                    if negative is not None:wanted.append(('real_backward',negative))
                if gpu==1 and call['accepted']>=3:
                    wanted.append(('real_three_steps',max(range(len(call['delta'])),key=lambda i:abs(call['delta'][i]))))
                for label,index in wanted:
                    if label in found:continue
                    found.add(label)
                    # No patient IDs, ages, dates, states or outcomes exported.
                    examples.append({'label':label,'synthetic':False,'physical_gpu':gpu,
                        'model_parameters':sum(p.numel() for p in model.parameters()),'checkpoint_epoch':checkpoint['epoch'],
                        'batch_index':batch_index,'selected_interval_years':call['delta'][index],
                        'selected_batch_position':index,'rtol':cfg['rtol'],**call})
            if (gpu==0 and len(found)==3) or (gpu==1 and found):break
    (OUT/f'gpu_{gpu}.json').write_text(json.dumps(examples,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.worker is not None:worker(args.worker);return
    def launch(gpu):
        with (OUT/f'gpu_{gpu}.log').open('w') as stream:
            subprocess.run([sys.executable,'-m','paper_odeae.diagnostics.step_examples','--worker',str(gpu)],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1'),stdout=stream,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(launch,range(4)))
    examples=sum([json.loads((OUT/f'gpu_{gpu}.json').read_text()) for gpu in range(4)],[])
    (OUT/'traces.json').write_text(json.dumps(examples,indent=2)+'\n')
    for x in examples:
        print(x['label'],x.get('selected_interval_years'),[(round(a['step'],6),round(a['error_ratio'],6),a['accepted']) for a in x['attempts']])


if __name__=='__main__':main()
