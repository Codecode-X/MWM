"""Queue six parameter-matched jobs exclusively on currently idle physical GPUs."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from paper_odeae.scaling_study import atomic,identity
from mimic_experiment.report import read
from .models import definition,parameter_count
from .report import report

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'capacity_matched/results/adni_mimic'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    datasets={
        'adni':{'output':str(ROOT/'paper_odeae/results/baselines_matched_latent2x'),
                'reference':str(ROOT/'paper_odeae/results/latent2x_loss_ablation/both/seed_0'),'target':44875,
                'native':{'ode_ae':str(ROOT/'paper_odeae/results/local_available/ode_ae/seed_0'),
                          'ode_rnn':str(ROOT/'paper_odeae/results/baselines_full/native/ode_rnn/seed_0'),
                          'latent_ode':str(ROOT/'paper_odeae/results/baselines_full/native/latent_ode/seed_0')}},
        'mimic':{'output':str(ROOT/'mimic_experiment/results/baselines_matched_latent2x'),
                 'reference':str(ROOT/'mimic_experiment/results/latent2x_both_cuda/seed_0'),'target':43280,
                 'native':{kind:str(ROOT/'mimic_experiment/results/baselines_single_cuda'/kind/'seed_0') for kind in ['ode_ae','ode_rnn','latent_ode']}}}
    jobs=[];configs={}
    for dataset,meta in datasets.items():
        folder=Path(meta['output']);folder.mkdir(parents=True,exist_ok=True);(folder/'configs').mkdir(exist_ok=True)
        ref=Path(meta['reference']);reference_hw=read(ref/'hardware.json');reference_cfg=read(ref/'config.json')
        if not read(ref/'metrics.json') or reference_hw['trainable_parameters']!=meta['target']:raise RuntimeError('Missing completed reference')
        data=Path(reference_cfg['data']);data=data if data.is_absolute() else ROOT/data
        sha=digest(data)
        if sha!=reference_hw['dataset_sha256']:raise RuntimeError('Reference dataset changed')
        meta.update(dataset=str(data),dataset_sha256=sha)
        for kind,path in meta['native'].items():
            source=Path(path);base=read(source/'config.json')
            if not read(source/'metrics.json'):raise RuntimeError(f'Native baseline incomplete: {source}')
            if read(source/'hardware.json')['dataset_sha256']!=sha:raise RuntimeError('Native comparison data differ')
            if parameter_count(dataset,kind)!=meta['target']:raise AssertionError('Integer capacity design failed')
            cfg={**base,**definition(dataset,kind),'capacity_matched':True,'dataset_name':dataset,'study':'latent2x_budget_matched',
                 'data':str(data),'dataset_sha256':sha,'target_parameters':meta['target'],'seed':0,'physical_gpu':None,
                 'output':str(folder/kind/'seed_0'),'comparison_output':str(folder),'variant':kind,'resume':True,
                 'prior_weight':0.,'jacobian_weight':0.,'collect_prior_predictions':False,'collect_jacobian_states':False,
                 'kl_weight':0. if kind=='ode_rnn' else 1.,'native_reference':str(source),'verify_reference':False,
                 'capacity_change':'exact total/trainable budget; native algorithm and training objective; no unused binary classifier'}
            if dataset=='mimic':cfg['reference_output']=str(ref.parent)
            config=folder/'configs'/f'{kind}.json'
            if config.exists() and read(config)!=cfg:raise RuntimeError('Preserve existing matched configuration')
            atomic(config,cfg);configs[dataset+'_'+kind]={'id':dataset+'_'+kind,'dataset':dataset,'kind':kind,'config':cfg,'config_path':str(config)}
    # Interleave datasets, so newly available GPUs serve both comparison tasks.
    for kind in ['ode_ae','ode_rnn','latent_ode']:
        for dataset in ['adni','mimic']:jobs.append(configs[dataset+'_'+kind])
    source=out/'source_snapshot';source.mkdir(exist_ok=True)
    files=list(read(ROOT/'mimic_experiment/results/baselines_single_cuda/manifest.json')['source_hashes'])
    files += ['capacity_matched/__init__.py','capacity_matched/models.py','capacity_matched/checks.py','capacity_matched/report.py','capacity_matched/study.py',
              'paper_odeae/history_encoders.py']
    if not (source/'source_hashes.json').exists():
        for name in files:
            target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        atomic(source/'source_hashes.json',{name:digest(source/name) for name in files})
    hashes=read(source/'source_hashes.json')
    for name,sha in hashes.items():
        if digest(source/name)!=sha:raise RuntimeError('Frozen matched source changed')
    manifest={'datasets':datasets,'jobs':jobs,'source_sha256':hashes,'seed':0,'strategy':'one job per idle physical GPU; never preempt existing processes',
              'matching':'total = trainable = task-used parameters','preflight':'all six architectures checked on one available CUDA GPU before training'}
    if (out/'manifest.json').exists() and read(out/'manifest.json')!=manifest:raise RuntimeError('Preserve existing manifest')
    atomic(out/'manifest.json',manifest)
    return source,manifest


def take_gpu(gpus,owned):
    for gpu in gpus:
        if gpu in owned:continue
        stream=Path(f'/tmp/latentode_gpu_{gpu}.lock').open('a')
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:stream.close();continue
        apps=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if not apps:return gpu,stream
        stream.close()
    return None


def expected(job,stage):
    cfg=job['config'];folder=Path(cfg['output']) if stage=='training' else Path(cfg['comparison_output'])/'checks'/job['kind']/'smoke'
    return folder,folder/'metrics.json'


def verify(job,stage):
    folder,path=expected(job,stage);m=read(path);h=read(folder/'hardware.json')
    wanted='cuda_smoke_passed' if stage=='smoke' else 'complete' if job['dataset']=='mimic' else 'local_or_paper_contract_run_completed'
    if not m or m['status']!=wanted or m['checkpoint_reload_difference']>1e-6:raise RuntimeError('Incomplete training/test stage')
    if h['parameters']!=h['trainable_parameters'] or h['parameters']!=job['config']['target_parameters']:raise RuntimeError('Capacity mismatch')
    if h['parameter_device']!='cuda:0' or h['prediction_device']!='cuda:0' or set(h['actual_tensor_devices'].values())!={'cuda:0'}:raise RuntimeError('Model computations must be CUDA')
    if stage=='training':
        reference=read(Path(job['config']['native_reference'])/'metrics.json')
        for split in ['validation','test']:
            for region in ['interpolation','forecast','overall']:
                if m[split][region]['observed_scalars']!=reference[split][region]['observed_scalars']:raise RuntimeError('Evaluation target counts differ')
        atomic(folder/'completion_verification.json',{'status':'passed','same_evaluation_target_counts':True,'parameters':h['parameters'],
               'trainable_parameters':h['trainable_parameters'],'physical_gpu':h['physical_gpu'],'device':'cuda:0'})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(DEFAULT));parser.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3]);args=parser.parse_args()
    if Path(sys.prefix).name!='ode' or not args.gpus or len(set(args.gpus))!=len(args.gpus) or set(args.gpus)-{0,1,2,3}:raise RuntimeError('Use ode and valid physical GPUs')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True);previous=read(out/'queue.json',{})
    if previous.get('pid') and identity(previous['pid'])==previous.get('start_time'):raise RuntimeError('Matched baseline queue already running')
    source,manifest=prepare(out);workers={};logs={};leases={};owned={};jobs={j['id']:j for j in manifest['jobs']}
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'waiting_for_idle_gpu','started_at':datetime.now(timezone.utc).isoformat(),
           'gpus':args.gpus,'jobs':{j['id']:{'status':'waiting','phase':'architecture_checks'} for j in manifest['jobs']}}
    atomic(out/'queue.json',state);report(out,manifest)
    def environment(gpu,name):return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(source),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                                        SWANLAB_LOG_DIR=str(ROOT/'swanlog'),SWANLAB_EXPERIMENT_NAME=f'capacity-matched-{name}-seed0')
    def launch(job,stage,gpu):
        key=job['id'];folder,_=expected(job,stage);folder.mkdir(parents=True,exist_ok=True)
        cfg={**job['config'],'physical_gpu':gpu};config=folder/'launch_config.json';atomic(config,cfg)
        if job['dataset']=='mimic':cmd=[sys.executable,'-u','-m','mimic_experiment.train_baseline','--config',str(config),'--stage',stage]
        else:cmd=[sys.executable,'-u','-m','paper_odeae.train','--config',str(config),'--output',str(folder)]+(['--smoke'] if stage=='smoke' else [])
        stream=(folder/'run.log').open('a');logs[key]=stream
        worker=subprocess.Popen(cmd,cwd=source,env=environment(gpu,key+'-'+stage),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        workers[key]=(worker,stage,gpu);owned[gpu]=key;state['jobs'][key].update(status='running',phase=stage,physical_gpu=gpu,pid=worker.pid)
        atomic(out/'queue.json',state);print(f'{key}: {stage}, physical GPU {gpu}, PID {worker.pid}',flush=True)
    try:
        if not (out/'architecture_checks.json').exists():
            lease=None
            while lease is None:
                lease=take_gpu(args.gpus,{})
                if lease is None:report(out,manifest);time.sleep(10)
            gpu,handle=lease;leases[gpu]=handle;owned[gpu]='architecture_checks'
            state.update(status='architecture_checks',architecture_checks_gpu=gpu);atomic(out/'queue.json',state)
            with (out/'architecture_checks.log').open('a') as log:
                worker=subprocess.Popen([sys.executable,'-u','-m','capacity_matched.checks','--manifest',str(out/'manifest.json'),'--gpu',str(gpu)],
                        cwd=source,env=environment(gpu,'architecture-checks'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                workers['architecture_checks']=(worker,'checks',gpu);print(f'All six architecture checks: physical GPU {gpu}, PID {worker.pid}',flush=True)
                while worker.poll() is None:time.sleep(2)
                workers.pop('architecture_checks')
                if worker.returncode or not (out/'architecture_checks.json').exists():raise RuntimeError('Architecture checks failed; inspect architecture_checks.log')
            leases.pop(gpu).close();owned.pop(gpu)
        if read(out/'architecture_checks.json')['status']!='passed':raise RuntimeError('Architecture preflight incomplete')
        state['status']='running'
        for job in manifest['jobs']:
            _,done=expected(job,'training')
            if done.exists():verify(job,'training');state['jobs'][job['id']].update(status='complete',phase='tested')
            else:state['jobs'][job['id']].update(status='waiting',phase='waiting_for_idle_gpu')
        atomic(out/'queue.json',state);last=0
        while any(s['status'] in ['waiting','running'] for s in state['jobs'].values()):
            for key,(worker,stage,gpu) in list(workers.items()):
                code=worker.poll()
                if code is None:continue
                logs.pop(key).close();workers.pop(key)
                try:
                    if code:raise RuntimeError(f'Worker exited {code}')
                    verify(jobs[key],stage)
                except Exception as exc:
                    state['jobs'][key].update(status='failed',error=repr(exc),exit_code=code);leases.pop(gpu).close();owned.pop(gpu)
                    print(f'{key}: failed in {stage}; inspect run.log',flush=True)
                else:
                    if stage=='smoke':launch(jobs[key],'training',gpu)
                    else:
                        state['jobs'][key].update(status='complete',phase='tested',finished_at=datetime.now(timezone.utc).isoformat())
                        leases.pop(gpu).close();owned.pop(gpu);print(f'{key}: complete',flush=True)
                atomic(out/'queue.json',state)
            for job in manifest['jobs']:
                key=job['id']
                if state['jobs'][key]['status']!='waiting':continue
                lease=take_gpu(args.gpus,owned)
                if lease is None:break
                gpu,handle=lease;leases[gpu]=handle
                _,smoked=expected(job,'smoke');stage='training' if smoked.exists() else 'smoke'
                if smoked.exists():verify(job,'smoke')
                launch(job,stage,gpu)
            if time.monotonic()-last>=30:report(out,manifest);last=time.monotonic()
            time.sleep(5)
        state.update(status='complete' if all(s['status']=='complete' for s in state['jobs'].values()) else 'failed',finished_at=datetime.now(timezone.utc).isoformat())
        atomic(out/'queue.json',state);report(out,manifest)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),finished_at=datetime.now(timezone.utc).isoformat())
        for key in workers:
            if key in state['jobs']:state['jobs'][key]['status']='interrupted'
        atomic(out/'queue.json',state);report(out,manifest);raise
    finally:
        for worker,_,_ in workers.values():
            if worker.poll() is None:worker.terminate();worker.wait()
        for stream in logs.values():stream.close()
        for handle in leases.values():handle.close()


if __name__=='__main__':main()
