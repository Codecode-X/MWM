"""Execute independent frozen-model diagnostics across physical GPUs 0-3."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait
from datetime import datetime,timezone
import json
import os
from pathlib import Path
from queue import Queue,Empty
import shutil
import subprocess
import sys
import threading

from paper_odeae.scaling_study import atomic,identity
from paper_odeae.prior_study import read
from .state_space_benchmark import ROOT,OUT,RUNS,sha


def progress(out,state):
    done=sum((out/name/'statistics.json').exists() for name,_,_,_ in RUNS)
    lines=['# 八组模型状态空间横向比较：计算进度','',f'已完成 {done}/{len(RUNS)} 组（含原六维 shared_matched 参照）。全部完成后此页自动替换为图文比较报告。','',
           '| 模型 | 状态 | 物理GPU | 日志 |','|---|---|---:|---|']
    for name,title,_,_ in RUNS:
        job=state['jobs'].get(name,{})
        lines.append(f"| {title} | {job.get('status','queued')} | {job.get('physical_gpu','—')} | [日志]({name}/run.log) |")
    (out/'report.zh-CN.md').write_text('\n'.join(lines)+'\n')


def prepare(out):
    out.mkdir(parents=True,exist_ok=True);models=[]
    for name,title,family,relative in RUNS:
        folder=ROOT/relative
        if not (folder/'metrics.json').exists():raise RuntimeError(f'Training not complete: {name}')
        models.append({'name':name,'title':title,'family':family,'checkpoint':str(folder/'best.pt'),
                       'checkpoint_sha256':sha(folder/'best.pt'),'original_metrics_sha256':sha(folder/'metrics.json'),
                       'environment':'healthflux' if family=='healthflux' else 'ode'})
    dataset=ROOT/'paper_odeae/cache/local_available.pkl'
    manifest={'models':models,'dataset':str(dataset),'dataset_sha256':sha(dataset),'gpus':[0,1,2,3],
              'training_performed':False,'pca':'own training states; original shared_matched PCA reused',
              'primary_healthflux_state':'256D h; additional observed 384D representation',
              'native_ode_rnn_forecast':'feedback updates; shown separately from pure ODE diagnostic continuation',
              'case_source':str(ROOT/'paper_odeae/results/shared_matched_state_space/patient_paths.json')}
    if (out/'manifest.json').exists() and read(out/'manifest.json')!=manifest:raise RuntimeError('Manifest changed; preserve old run')
    atomic(out/'manifest.json',manifest)
    files=['paper_odeae/__init__.py','paper_odeae/data.py','paper_odeae/models.py','paper_odeae/train.py','paper_odeae/baselines.py',
           'paper_odeae/solver.py','paper_odeae/jacobian.py','paper_odeae/scaling_study.py','paper_odeae/prior_study.py',
           'paper_odeae/reference/features46.json','paper_odeae/reference/tsit5_tableau.json',
           'paper_odeae/diagnostics/state_space.py','paper_odeae/diagnostics/state_space_benchmark.py',
           'paper_odeae/diagnostics/state_space_benchmark_report.py','paper_odeae/diagnostics/run_state_space_benchmark.py',
           'healthflux/__init__.py','healthflux/model.py','src/__init__.py','src/model.py','src/adni_models.py','src/tracking.py','src/swanlab_local.py',
           'paper_odeae/tests/test_state_space_benchmark.py']
    snapshot=out/'source_snapshot'
    if not (snapshot/'source_hashes.json').exists():
        for name in files:
            target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        atomic(snapshot/'source_hashes.json',{name:sha(snapshot/name) for name in files})
    for name,digest in read(snapshot/'source_hashes.json').items():
        if sha(snapshot/name)!=digest:raise RuntimeError('Frozen source changed')
    return manifest,snapshot


def velocity_pass(out,snapshot):
    """Supplement the state PCA with an accurate training-velocity projection."""
    supplement=out/'velocity_source_snapshot'
    if not (supplement/'source_hashes.json').exists():
        shutil.copytree(snapshot,supplement,dirs_exist_ok=True)
        name='paper_odeae/diagnostics/benchmark_velocity_plane.py'
        shutil.copy2(ROOT/name,supplement/name)
        files=[p for p in supplement.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='source_hashes.json']
        atomic(supplement/'source_hashes.json',{str(p.relative_to(supplement)):sha(p) for p in files})
    for name,digest in read(supplement/'source_hashes.json').items():
        if sha(supplement/name)!=digest:raise RuntimeError('Supplement source changed')
    shader=sha(supplement/'paper_odeae/diagnostics/benchmark_velocity_plane.py');jobs=Queue()
    for row in RUNS:jobs.put(row)
    def worker(gpu):
        while True:
            try:name,title,family,_=jobs.get_nowait()
            except Empty:return
            folder=out/name;old=read(folder/'velocity_plane.json',{})
            if old.get('diagnostics_source_sha256')==shader and (folder/'velocity_plane_paths.json').exists():continue
            if old and not (folder/'velocity_plane.float32_initial.json').exists():atomic(folder/'velocity_plane.float32_initial.json',old)
            python=f"/home/xjh/miniforge3/envs/{'healthflux' if family=='healthflux' else 'ode'}/bin/python"
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(supplement),ADNI_EXPERIMENT_ROOT=str(ROOT),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            cmd=[python,'-u','-m','paper_odeae.diagnostics.benchmark_velocity_plane','--model',name,'--gpu',str(gpu),'--output',str(out)]
            with (folder/'velocity_plane.log').open('w') as stream:subprocess.run(cmd,cwd=supplement,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,range(4)))
    checks=[]
    for name,_,_,_ in RUNS:
        s=read(out/name/'velocity_plane.json')
        assert s['device']==s['case_model_device']=='cuda:0' and s['basis_orthogonality_max_error']<1e-10
        assert s['source_states_sha256']==sha(out/name/'states.npz')
        assert s['diagnostics_source_sha256']==shader and 0<=s['velocity_energy_retained_2d']<=1
        checks.append(s)
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(OUT));args=parser.parse_args()
    out=Path(args.output).resolve();old=read(out/'queue.json',{})
    if old.get('pid') and identity(old['pid'])==old.get('start_time'):raise RuntimeError('Diagnostics already running')
    manifest,snapshot=prepare(out);lock=threading.RLock();jobs=Queue();errors=[]
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
           'jobs':{row['name']:{'status':'queued'} for row in manifest['models']}}
    for row in manifest['models']:jobs.put(row)
    def update(name,**kwargs):
        with lock:state['jobs'][name].update(kwargs);atomic(out/'queue.json',state)
    def worker(gpu):
        while True:
            try:job=jobs.get_nowait()
            except Empty:return
            name=job['name'];folder=out/name;folder.mkdir(parents=True,exist_ok=True)
            try:
                saved=read(folder/'statistics.json')
                if saved:
                    assert saved['checkpoint_sha256']==job['checkpoint_sha256'] and saved['dataset_sha256']==manifest['dataset_sha256']
                    update(name,status='complete',physical_gpu=saved['physical_gpu']);continue
                update(name,status='running',physical_gpu=gpu)
                python=f"/home/xjh/miniforge3/envs/{job['environment']}/bin/python"
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(snapshot),ADNI_EXPERIMENT_ROOT=str(ROOT),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
                command=[python,'-u','-m','paper_odeae.diagnostics.state_space_benchmark','--model',name,'--gpu',str(gpu),'--output',str(out)]
                print(f'GPU {gpu}: {name} ({job["environment"]})',flush=True)
                with (folder/'run.log').open('w') as stream:subprocess.run(command,cwd=snapshot,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                update(name,status='complete');(folder/'failure.json').unlink(missing_ok=True)
            except Exception as error:
                with lock:errors.append(f'{name}: {error}')
                atomic(folder/'failure.json',{'error':str(error)});update(name,status='failed',error=str(error))
            finally:jobs.task_done()
    atomic(out/'queue.json',state);progress(out,state)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(worker,gpu) for gpu in range(4)}
        while pending:
            _,pending=wait(pending,timeout=30);progress(out,state)
    if errors:
        state.update(status='failed',errors=errors);atomic(out/'queue.json',state);progress(out,state);raise RuntimeError('; '.join(errors))
    verification=[]
    for job in manifest['models']:
        stats=read(out/job['name']/'statistics.json')
        assert stats['checkpoint_sha256']==job['checkpoint_sha256'] and stats['dataset_sha256']==manifest['dataset_sha256']
        assert stats['original_evaluation_max_difference']<=1e-6 and stats['checkpoint_parameters_unchanged']
        assert stats['environment']==job['environment'] and stats['model_device']=='cuda:0'
        assert all(device=='cuda:0' for device in stats['devices'].values())
        assert stats['counts_by_split']=={'train':12544,'validation':2219,'test':2402}
        assert sha(job['checkpoint'])==job['checkpoint_sha256']
        verification.append({'name':job['name'],'physical_gpu':stats['physical_gpu'],'environment':stats['environment'],
                             'devices':stats['devices'],'original_evaluation_max_difference':stats['original_evaluation_max_difference'],
                             'checkpoint_parameters_unchanged':True,'state_checks':stats['state_checks']})
    state['status']='velocity_projection';atomic(out/'queue.json',state)
    velocity_checks=velocity_pass(out,snapshot)
    atomic(out/'verification.json',{'status':'passed','runs':verification,'training_performed':False,'velocity_projection_checks':velocity_checks})
    print('CUDA extraction complete; assembling comparison figures',flush=True)
    state['status']='plotting';atomic(out/'queue.json',state)
    from .state_space_benchmark_report import generate
    generate(out)
    state.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat());atomic(out/'queue.json',state)
    print(f'Complete: {out/"report.zh-CN.md"}',flush=True)


if __name__=='__main__':main()
