"""Small CUDA integration checks for all five training entry points.

Each physical GPU gets one short check worker. Existing formal results are not
reused or overwritten. This script reads ADNI caches but never trains on CPU.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def command(parts):
    subprocess.run([sys.executable,'-u',*parts],check=True)


def worker(index,output):
    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('integration checks require CUDA')
    data=str(output/'tiny_icu.csv')
    if index==0:
        for model in ['latent_ode','ode_rnn']:
            command(['-m','src.train','--model',model,'--data',data,'--epochs','1','--batch-size','8',
                     '--samples','1','--latent','4','--hidden','8','--width','8','--layers','1',
                     '--output',str(output/'base'/model)])
        import numpy as np
        from torch.utils.data import DataLoader
        from legacy.preprocessing import ICUDataset,collate_fn
        from legacy.model import LatentODEModel
        from legacy.train import ImprovedTrainer
        work=output/'legacy';work.mkdir(parents=True,exist_ok=True);os.chdir(work)
        Path('results').mkdir(exist_ok=True)
        sequences=[{'patient_id':i,'times':np.array([0.,.1,.2]),'vitals':np.full((3,6),.2+i*.01),
                    'time_deltas':np.array([0.,.1,.1]),'label':float(i%2),'seq_length':3} for i in range(4)]
        loader=DataLoader(ICUDataset(sequences),batch_size=4,collate_fn=collate_fn)
        model=LatentODEModel(hidden_dim=16,latent_dim=4).cuda()
        trainer=ImprovedTrainer(model,loader,loader,device='cuda',tracking_output=work/'tracking')
        trainer.train(n_epochs=1)
    elif index==1:
        from src.comparison import parser,run_one,initialize_worker
        from src.preprocessing import prepare_data
        initialize_worker(None)
        args=parser().parse_args(['--data',data,'--output',str(output/'comparison'),'--epochs','1',
              '--batch-size','8','--train-samples','1','--eval-samples','1','--objective','bce',
              '--latent','4','--hidden','8','--width','8','--layers','1'])
        splits,metadata=prepare_data(data,args.split_seed)
        for model in ['legacy_transformer','latent_ode','ode_rnn']:
            run_one(model,0,splits,metadata,args)
    elif index==2:
        command(['-m','src.adni_train','--model','ode_rnn','--epochs','1','--smoke','--output',str(output/'adni')])
    elif index==3:
        for model in ['ode_ae','deterministic','lstm']:
            command(['-m','paper_odeae.train','--model',model,'--smoke','--output',str(output/'paper'/model)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='results/swanlab_monitoring/checks')
    p.add_argument('--worker',type=int,choices=range(4))
    args=p.parse_args();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    if args.worker is not None:
        worker(args.worker,output);return
    import csv
    fields=['patient_id','time_hours','heart_rate','sbp','dbp','spo2','temperature','respiratory_rate','adverse_event']
    with (output/'tiny_icu.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(fields)
        for patient in range(24):
            for t in [0,12,36,72]:
                w.writerow([patient,t,70+patient+t*.1,110+patient,70+patient*.2,97,36.5,15+patient*.1,patient%2])
    def check(gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),SWANLAB_MODE='local',SWANLAB_LOG_EVERY='1',
                 SWANLAB_PROJECT='LatentODE-Monitoring-Checks',OMP_NUM_THREADS='1')
        with (output/f'gpu_{gpu}.log').open('w') as log:
            subprocess.run([sys.executable,'-u',__file__,'--output',str(output),'--worker',str(gpu)],
                           env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in [pool.submit(check,gpu) for gpu in range(4)]:f.result()
    runs=[]
    for path in output.rglob('swanlab.json'):
        record=json.loads(path.read_text());assert record['status']=='success',path
        assert record['parameter_device'].startswith('cuda'),path
        runs.append({'path':str(path.parent),**record})
    assert len(runs)==10,len(runs)
    assert {r['physical_gpu'] for r in runs}=={'0','1','2','3'}
    summary={'checked_training_entries':5,'completed_cuda_runs':len(runs),'runs':runs}
    (output.parent/'integration_checks.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'checked_training_entries':5,'completed_cuda_runs':len(runs)},indent=2))


if __name__=='__main__':main()
