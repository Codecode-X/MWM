"""Prepare MIMIC, verify four-GPU losses, then train/test latent2x both."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
from paper_odeae.scaling_study import identity
from .prepare import DEFAULT,atomic,chronology_support
from .report import generate,read

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'mimic_experiment/results/latent2x_both_cuda'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_data(root,out):
    payload=pickle.loads((root/'dataset.pkl').read_bytes());schema=payload['schema'];sequences=payload['sequences'];meta=payload['metadata']
    subjects=[s['subject_id'] for s in sequences];assert len(subjects)==len(set(subjects))
    subject_sets={key:{s['subject_id'] for s in sequences if s['split']==key} for key in ['train','validation','test']}
    assert not (subject_sets['train']&subject_sets['validation'] or subject_sets['train']&subject_sets['test'] or subject_sets['validation']&subject_sets['test'])
    import pandas as pd
    cohort=pd.read_csv(root/'cohort.csv')
    assert meta['dataset_revision']=='chronology_v2' and chronology_support(cohort).all()
    assert set(cohort.subject_id.astype(str))==set(subjects)
    checked=0
    for s in sequences:
        assert np.all(np.diff(s['ages'])>0) and s['ages'].min()>=1 and s['ages'].max()<=48
        assert np.isfinite(s['values']).all() and set(np.unique(s['mask']))<={0.,1.}
        assert np.searchsorted(s['ages'],24,side='right')>=6 and (s['ages']>24).sum()>=6
        for j,f in enumerate(schema):
            values=s['values'][:,j][s['mask'][:,j].astype(bool)]
            if f['kind']=='categorical':assert np.all(values==np.rint(values)) and np.all((values>=0)&(values<f['classes']))
        checked+=len(s['ages'])
    # Check fitted normalization against raw hourly training aggregates reconstructed
    # from the cached normalized values. This is data preparation, not model execution.
    for j,f in enumerate(schema):
        if f['kind']!='numerical':continue
        train=np.concatenate([s['values'][:,j][s['mask'][:,j].astype(bool)] for s in sequences if s['split']=='train'])
        assert abs(float(train.min()))<1e-5 and abs(float(train.max())-1)<1e-5
    report={'status':'passed','dataset_sha256':digest(root/'dataset.pkl'),'schema_fields':len(schema),'expanded_features':sum(s.get('classes',1) for s in schema),
            'patients':len(sequences),'patient_counts':{k:len(v) for k,v in subject_sets.items()},'observed_hour_rows':checked,
            'patient_partitions_disjoint':True,'complete_living_hospital_48h_window':True,'dataset_revision':meta['dataset_revision'],
            'strictly_increasing_hour_clock':True,'finite_values':True,'categorical_codes_verified':True,
            'training_only_minmax_endpoints_verified':True,'availability_rule':meta['availability_time'],
            'normalization':'train_minmax','source_tables':read(root/'source/manifest.json')}
    atomic(out/'data_verification.json',report)
    return payload


def configuration(root,out,payload):
    base=read(ROOT/'paper_odeae/results/latent2x_loss_ablation/configs/both.json')
    f=sum(s.get('classes',1) for s in payload['schema'])
    cfg={**base,'data':str(root/'dataset.pkl'),'output':str(out/'seed_0'),'scope':payload['metadata']['scope'],
         'batch_size':128,'study':'mimic_iv31_latent2x_both','target_parameters':29244+319*f,
         'capacity_change':'dataset schema only; latent 28, MLP 159, ODE 188 retained','time_unit':'hours',
         'initialization_age':'admission_age plus elapsed hours/8766; independent of ODE clock',
         'training_context_hours':[6,12,18,24],'evaluation_context_hours':24}
    path=out/'config.json'
    if path.exists() and read(path)!=cfg:raise RuntimeError('Configuration changed; preserve prior run')
    atomic(path,cfg)
    files=['mimic_experiment/__init__.py','mimic_experiment/prepare.py','mimic_experiment/data.py','mimic_experiment/train_ddp.py','mimic_experiment/report.py','mimic_experiment/run.py',
           'paper_odeae/__init__.py','paper_odeae/models.py','paper_odeae/train.py','paper_odeae/data.py','paper_odeae/solver.py','paper_odeae/jacobian.py',
           'paper_odeae/reference/features46.json','paper_odeae/reference/tsit5_tableau.json','src/__init__.py','src/tracking.py','src/swanlab_local.py']
    snapshot=out/'source_snapshot'
    if not (snapshot/'source_hashes.json').exists():
        for name in files:
            target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        atomic(snapshot/'source_hashes.json',{name:digest(snapshot/name) for name in files})
    for name,sha in read(snapshot/'source_hashes.json').items():
        if digest(snapshot/name)!=sha:raise RuntimeError('Frozen source changed')
    return cfg,snapshot


def all_gpus_free():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=15)
    return not text.strip()


def launch(stage,cfg,snapshot,out,root,state,flags=()):
    folder=out/'checks'/stage if stage!='training' else out/'seed_0';folder.mkdir(parents=True,exist_ok=True)
    expected='distributed_step_check.json' if stage=='step' else 'metrics.json'
    if (folder/expected).exists():return
    if stage=='training' and (folder/'history.json').exists():raise RuntimeError('Preserve interrupted formal training')
    local={**cfg,'output':str(folder)};config=folder/'launch_config.json';atomic(config,local)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    processes=[];streams=[]
    try:
        for rank in range(4):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),PYTHONPATH=str(snapshot),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NCCL_P2P_DISABLE='1',
                     SWANLAB_LOG_DIR=str(ROOT/'swanlog'),TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
            stream=(folder/f'rank_{rank}.log').open('w');streams.append(stream)
            command=[sys.executable,'-u','-m','mimic_experiment.train_ddp','--config',str(config),'--rank',str(rank),'--world-size','4','--port',str(port),*flags]
            processes.append(subprocess.Popen(command,cwd=snapshot,env=env,stdout=stream,stderr=subprocess.STDOUT))
        state.update(status=stage,workers=[{'rank':i,'physical_gpu':i,'pid':p.pid} for i,p in enumerate(processes)]);atomic(out/'queue.json',state);generate(out,root)
        print(f'Started {stage} on GPUs 0,1,2,3',flush=True);last=0
        while True:
            status=[p.poll() for p in processes]
            if any(code is not None and code!=0 for code in status):raise RuntimeError(f'{stage} worker failed; exit codes {status}; see {folder}')
            if all(code is not None for code in status):break
            if time.monotonic()-last>=30:generate(out,root);last=time.monotonic()
            time.sleep(2)
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for stream in streams:stream.close()
    if not (folder/expected).exists():raise RuntimeError(f'{stage}: result missing')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-root',default=str(DEFAULT));parser.add_argument('--output',default=str(OUT));args=parser.parse_args()
    if Path(sys.prefix).name!='ode':raise RuntimeError('Use existing ode environment')
    root=Path(args.data_root).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True);old=read(out/'queue.json',{})
    if old.get('pid') and identity(old['pid'])==old.get('start_time'):raise RuntimeError('MIMIC experiment already running')
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'preparing','started_at':datetime.now(timezone.utc).isoformat(),'physical_gpus':[0,1,2,3]}
    atomic(out/'queue.json',state);generate(out,root)
    try:
        while not ((root/'dataset.pkl').exists() and (root/'dataset_summary.json').exists()):
            previous=read(root/'prepare_pid.json',{});pid=previous.get('pid')
            active=bool(pid and Path(f'/proc/{pid}/cmdline').exists() and b'mimic_experiment.prepare' in Path(f'/proc/{pid}/cmdline').read_bytes())
            if not active:
                root.mkdir(parents=True,exist_ok=True)
                with (root/'prepare.log').open('a') as log:
                    process=subprocess.Popen([sys.executable,'-u','-m','mimic_experiment.prepare','--root',str(root)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                atomic(root/'prepare_pid.json',{'pid':process.pid});print(f'Data preparation PID {process.pid}',flush=True)
            generate(out,root);time.sleep(20)
        payload=verify_data(root,out);cfg,snapshot=configuration(root,out,payload);del payload
        atomic(out/'manifest.json',{'dataset':str(root/'dataset.pkl'),'dataset_sha256':digest(root/'dataset.pkl'),'model':'latent2x both','seed':0,
               'physical_gpus':[0,1,2,3],'training_strategy':'four-GPU DDP; exact global masked loss','configuration':str(out/'config.json')})
        state['status']='waiting_for_gpus';atomic(out/'queue.json',state)
        while not all_gpus_free():generate(out,root);time.sleep(20)
        launch('step',cfg,snapshot,out,root,state,['--check-step'])
        launch('smoke',cfg,snapshot,out,root,state,['--smoke'])
        smoke=read(out/'checks/smoke/metrics.json');assert smoke['status']=='cuda_smoke_passed' and smoke['checkpoint_reload_difference']<=1e-6
        launch('training',cfg,snapshot,out,root,state)
        result=read(out/'seed_0/metrics.json');assert result['status']=='complete' and result['checkpoint_reload_difference']<=1e-6
        state.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat());atomic(out/'queue.json',state);generate(out,root)
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');atomic(out/'queue.json',state);generate(out,root);raise


if __name__=='__main__':main()
