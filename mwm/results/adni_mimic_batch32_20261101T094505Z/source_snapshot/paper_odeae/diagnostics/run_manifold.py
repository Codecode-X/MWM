"""Four bounded-memory frozen CUDA diagnostics, preserving all existing training jobs."""
from datetime import datetime,timezone
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .manifold_core import sha,atomic,read
from .manifold_report import generate

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(ROOT/'paper_odeae/results/latent2x_both_manifold'));args=parser.parse_args()
    if Path(sys.prefix).name!='ode':raise RuntimeError('Use existing ode environment')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'queue.json').exists():raise RuntimeError('Preserve the existing manifold experiment')
    adni=ROOT/'paper_odeae/results/latent2x_loss_ablation/both/seed_0'
    mimic=ROOT/'mimic_experiment/results/latent2x_both_cuda/seed_0'
    state_source=ROOT/'paper_odeae/results/state_space_model_comparison/latent2x_both'
    stats=read(state_source/'statistics.json');assert sha(adni/'best.pt')==stats['checkpoint_sha256']
    assert sha(ROOT/'paper_odeae/cache/local_available.pkl')==stats['dataset_sha256']
    files=['paper_odeae/__init__.py','paper_odeae/models.py','paper_odeae/solver.py','paper_odeae/data.py',
           'paper_odeae/reference/features46.json','paper_odeae/reference/tsit5_tableau.json',
           'paper_odeae/diagnostics/manifold_core.py','paper_odeae/diagnostics/manifold_geometry.py','paper_odeae/diagnostics/manifold_dynamics.py',
           'paper_odeae/diagnostics/manifold_report.py','paper_odeae/diagnostics/run_manifold.py','mimic_experiment/__init__.py','mimic_experiment/data.py']
    snapshot=out/'source_snapshot'
    for file in files:
        target=snapshot/file;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/file,target)
    hashes={file:sha(snapshot/file) for file in files};atomic(snapshot/'source_hashes.json',hashes)
    checkpoints={'adni':str(adni/'best.pt'),'mimic':str(mimic/'best.pt')}
    configs={};(out/'configs').mkdir(exist_ok=True)
    for gpu,(dataset,kind) in enumerate([('adni','geometry'),('adni','dynamics'),('mimic','geometry'),('mimic','dynamics')]):
        key=dataset+'_'+kind;folder=out/key;folder.mkdir(exist_ok=True)
        cfg={'dataset':dataset,'stage':kind,'physical_gpu':gpu,'checkpoint':checkpoints[dataset],
             'data':str(ROOT/'paper_odeae/cache/local_available.pkl') if dataset=='adni' else '/data/MIMIC/derived/latentode_iv31/dataset.pkl',
             'state_source':str(state_source),'output':str(out),'job_output':str(folder),'memory_fraction':.12,'training_performed':False}
        atomic(out/'configs'/f'{key}.json',cfg);configs[key]=cfg
    atomic(out/'manifest.json',{'started_at':datetime.now(timezone.utc).isoformat(),'model':'latent2x both','checkpoints':{k:{'path':v,'sha256':sha(v)} for k,v in checkpoints.items()},
           'configs':configs,'source_hashes':hashes,'existing_training_preserved':True,'per_job_gpu_memory_fraction':.12,
           'geometry_and_model_execution':'CUDA only','cpu_tasks':'I/O, metadata preparation, statistical summaries, figures and report'})
    state={'pid':os.getpid(),'status':'running','jobs':{}};processes={};streams={}
    try:
        for key,cfg in configs.items():
            stream=(out/key/'run.log').open('w');streams[key]=stream
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(cfg['physical_gpu']),PYTHONPATH=str(snapshot),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            process=subprocess.Popen([sys.executable,'-u','-m','paper_odeae.diagnostics.manifold_'+cfg['stage'],'--config',str(out/'configs'/f'{key}.json')],
                env=env,cwd=snapshot,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            processes[key]=process;state['jobs'][key]={'pid':process.pid,'physical_gpu':cfg['physical_gpu'],'status':'running'}
        atomic(out/'queue.json',state);generate(out);last=0
        while processes:
            for key,process in list(processes.items()):
                code=process.poll()
                if code is None:continue
                state['jobs'][key].update(status='complete' if code==0 and (out/key/'complete.json').exists() else 'failed',exit_code=code)
                processes.pop(key);streams.pop(key).close();atomic(out/'queue.json',state)
                if state['jobs'][key]['status']=='failed':raise RuntimeError(f'{key} failed; inspect run.log')
            if time.monotonic()-last>=30:generate(out);last=time.monotonic()
            time.sleep(2)
        for dataset,path in checkpoints.items():assert sha(path)==read(out/'manifest.json')['checkpoints'][dataset]['sha256']
        state.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat());atomic(out/'queue.json',state);generate(out)
    except BaseException as error:
        state.update(status='failed',error=str(error));atomic(out/'queue.json',state);generate(out);raise
    finally:
        for process in processes.values():
            if process.poll() is None:process.terminate()
        for stream in streams.values():stream.close()


if __name__=='__main__':main()
