"""Run six MWM jobs on idle GPUs, sharing existing project GPU locks."""
import argparse
from datetime import datetime,timezone
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

from mimic_experiment.prepare import atomic
from .report import read,report

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'mwm/results/adni_mimic'


def identity(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[21]
    except FileNotFoundError:return None


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    jobs=[]
    for variant in ['main','A','B']:
        for dataset in ['mimic','adni']:
            reference=ROOT/('mimic_experiment/results/direction_full_batch_cuda/forward_shared/seed_0' if dataset=='mimic' else 'paper_odeae/results/latent2x_loss_ablation/both/seed_0')
            if not (reference/'metrics.json').exists():raise RuntimeError(f'Missing completed reference {reference}')
            base=read(reference/'config.json');data=Path(base['data']);data=data if data.is_absolute() else ROOT/data
            folder=ROOT/('mimic_experiment' if dataset=='mimic' else 'paper_odeae')/'results/mwm';folder.mkdir(parents=True,exist_ok=True)
            cfg={**base,'model':'MWM','variant':variant,'dataset_name':dataset,'study':'mwm_causal_supervision','seed':0,
                 'target_parameters':43280 if dataset=='mimic' else 44875,'batch_size':128 if dataset=='mimic' else 32,
                 'gradient_accumulation_steps':1,'prior_weight':0.,'jacobian_weight':.01,'l2_weight':.02,
                 'count_a':1.,'count_b':3.,'sigma_min':.01,'first_observation_bootstrap':'post-update',
                 'max_epochs':300,'min_epochs':50,'patience':40,'data':str(data),'dataset_sha256':digest(data),
                 'checkpoint_selection':'unweighted causal overall MSE + CE','initialization_age':'first actual observation age',
                 'reference_output':str(reference),'physical_gpu':None,'output':str(folder/variant/'seed_0'),
                 'comparison_output':str(folder),'collect_prior_predictions':False,'collect_jacobian_states':True,'kl_weight':0.,
                 'init_width':159,'update_width':160 if variant=='B' else 159,'decoder_width':158 if variant=='B' else 159,
                 'ode_width':187 if variant=='B' else 188,'encoder_ode_width':187 if variant=='B' else 188,
                 'generator_ode_width':187 if variant=='B' else 188,'capacity_change':'B scalar sigma head with exact hidden-width budget compensation' if variant=='B' else 'none',
                 'ablation':'MWM unified causal supervision; A count weight; B numerical observation NLL','smoke':False}
            if digest(data)!=read(reference/'hardware.json')['dataset_sha256']:raise RuntimeError('Reference data changed')
            config=folder/'configs'/f'{variant}.json';config.parent.mkdir(exist_ok=True)
            if config.exists() and read(config)!=cfg:raise RuntimeError('Existing MWM configuration changed')
            atomic(config,cfg);jobs.append({'id':dataset+'_'+variant,'dataset':dataset,'config':cfg,'config_path':str(config)})
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    if not (snapshot/'source_hashes.json').exists():
        paths=[]
        for name in ['mwm','paper_odeae','mimic_experiment','src']:
            paths += [p for p in (ROOT/name).rglob('*.py') if not any(x in p.parts for x in ['results','__pycache__'])]
        paths += list((ROOT/'paper_odeae/reference').glob('*.json'))
        for path in paths:
            target=snapshot/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        atomic(snapshot/'source_hashes.json',{str(p.relative_to(ROOT)):digest(p) for p in paths})
    hashes=read(snapshot/'source_hashes.json')
    for name,sha in hashes.items():
        if digest(snapshot/name)!=sha:raise RuntimeError('Frozen MWM sources changed')
    manifest={'project_root':str(ROOT),'jobs':jobs,'source_sha256':hashes,'strategy':'single full batch; one idle GPU per job; shared flock; no preemption',
              'seed':0,'validation_selection':'unweighted causal overall','first_observation':'first-age INIT, first UPDATE, first readout'}
    if (out/'manifest.json').exists() and read(out/'manifest.json')!=manifest:raise RuntimeError('Preserve existing MWM manifest')
    atomic(out/'manifest.json',manifest);return snapshot,manifest


def take_gpu(gpus,owned):
    for gpu in gpus:
        if gpu in owned:continue
        handle=Path(f'/tmp/latentode_gpu_{gpu}.lock').open('a')
        try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:handle.close();continue
        apps=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if not apps:return gpu,handle
        handle.close()
    return None


def expected(job,stage):
    cfg=job['config'];return Path(cfg['output']) if stage=='training' else Path(cfg['comparison_output'])/'checks'/cfg['variant']/'smoke'


def verify(job,stage):
    folder=expected(job,stage);metrics=read(folder/'metrics.json');hardware=read(folder/'hardware.json')
    if not metrics or metrics['status']!=('complete' if stage=='training' else 'cuda_smoke_passed'):raise RuntimeError('Missing completed metrics')
    if metrics['checkpoint_reload_difference']>1e-6:raise RuntimeError('Checkpoint reload mismatch')
    if hardware['parameters']!=job['config']['target_parameters'] or hardware['trainable_parameters']!=hardware['parameters']:raise RuntimeError('Parameter budget mismatch')
    if hardware['parameter_device']!='cuda:0' or hardware['prediction_device']!='cuda:0' or set(hardware['actual_tensor_devices'].values())!={'cuda:0'}:raise RuntimeError('Every model tensor must execute on CUDA')
    if hardware['gradient_accumulation_steps']!=1:raise RuntimeError('Unexpected gradient accumulation')
    if stage=='training':
        reference=read(Path(job['config']['reference_output'])/'metrics.json')
        for split in ['validation','test']:
            for region in ['interpolation','forecast','overall']:
                if metrics[split][region]['observed_scalars']!=reference[split][region]['observed_scalars']:raise RuntimeError('Evaluation target count differs from reference')
        atomic(folder/'completion_verification.json',{'status':'passed','same_evaluation_target_counts':True,'cuda_only':True,
               'parameters':hardware['parameters'],'physical_gpu':hardware['physical_gpu'],'gradient_accumulation_steps':1})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(DEFAULT));parser.add_argument('--gpus',type=int,nargs='+',default=[2,0,1,3]);args=parser.parse_args()
    if Path(sys.prefix).name!='ode' or not args.gpus or set(args.gpus)-{0,1,2,3}:raise RuntimeError('Use ode and the four physical GPUs')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True);previous=read(out/'queue.json',{})
    if previous.get('pid') and identity(previous['pid'])==previous.get('start_time'):raise RuntimeError('MWM queue already running')
    snapshot,manifest=prepare(out);workers={};leases={};owned={};logs={};jobs={j['id']:j for j in manifest['jobs']}
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'waiting_for_idle_gpu','gpus':args.gpus,
           'started_at':datetime.now(timezone.utc).isoformat(),'jobs':{key:{'status':'waiting','phase':'architecture_checks'} for key in jobs}}
    atomic(out/'queue.json',state);report(out,manifest)
    def stop(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    def environment(gpu,name):return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(snapshot),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                                         SWANLAB_LOG_DIR=str(ROOT/'swanlog'),SWANLAB_EXPERIMENT_NAME=f'MWM-{name}-seed0')
    def launch(job,stage,gpu):
        key=job['id'];folder=expected(job,stage);folder.mkdir(parents=True,exist_ok=True)
        config=folder/'launch_config.json';atomic(config,{**job['config'],'physical_gpu':gpu})
        stream=(folder/'run.log').open('a');logs[key]=stream
        worker=subprocess.Popen([sys.executable,'-u','-m','mwm.train','--config',str(config),'--stage',stage],cwd=snapshot,
                                env=environment(gpu,key+'-'+stage),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        workers[key]=(worker,stage,gpu);owned[gpu]=key;state['jobs'][key].update(status='running',phase=stage,physical_gpu=gpu,pid=worker.pid)
        atomic(out/'queue.json',state);print(f'{key}: {stage}, GPU {gpu}, PID {worker.pid}',flush=True)
    try:
        if not (out/'architecture_checks.json').exists():
            lease=None
            while lease is None:
                lease=take_gpu(args.gpus,{})
                if lease is None:time.sleep(10)
            gpu,handle=lease;leases[gpu]=handle;owned[gpu]='architecture_checks'
            state.update(status='architecture_checks',architecture_checks_gpu=gpu);atomic(out/'queue.json',state)
            with (out/'architecture_checks.log').open('a') as stream:
                worker=subprocess.Popen([sys.executable,'-u','-m','mwm.checks','--manifest',str(out/'manifest.json'),'--gpu',str(gpu)],
                                        cwd=snapshot,env=environment(gpu,'checks'),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                workers['checks']=(worker,'checks',gpu)
                while worker.poll() is None:time.sleep(2)
                workers.pop('checks')
                if worker.returncode or read(out/'architecture_checks.json',{}).get('status')!='passed':raise RuntimeError('CUDA preflight failed; inspect architecture_checks.log')
            leases.pop(gpu).close();owned.pop(gpu)
        state['status']='running'
        for job in manifest['jobs']:
            if (expected(job,'training')/'metrics.json').exists():verify(job,'training');state['jobs'][job['id']].update(status='complete',phase='tested')
            else:state['jobs'][job['id']].update(phase='waiting_for_idle_gpu')
        atomic(out/'queue.json',state);last=0.
        while any(s['status'] in ['waiting','running'] for s in state['jobs'].values()):
            for key,(worker,stage,gpu) in list(workers.items()):
                code=worker.poll()
                if code is None:continue
                workers.pop(key);logs.pop(key).close()
                try:
                    if code:raise RuntimeError(f'Worker exited {code}')
                    verify(jobs[key],stage)
                except Exception as exc:
                    state['jobs'][key].update(status='failed',error=repr(exc),exit_code=code);leases.pop(gpu).close();owned.pop(gpu)
                    print(f'{key}: failed in {stage}, inspect run.log',flush=True)
                else:
                    if stage=='smoke':launch(jobs[key],'training',gpu)
                    else:
                        state['jobs'][key].update(status='complete',phase='tested',finished_at=datetime.now(timezone.utc).isoformat())
                        leases.pop(gpu).close();owned.pop(gpu)
                atomic(out/'queue.json',state)
            for job in manifest['jobs']:
                if state['jobs'][job['id']]['status']!='waiting':continue
                lease=take_gpu(args.gpus,owned)
                if lease is None:break
                gpu,handle=lease;leases[gpu]=handle
                stage='training' if (expected(job,'smoke')/'metrics.json').exists() else 'smoke'
                if stage=='training':verify(job,'smoke')
                launch(job,stage,gpu)
            if time.monotonic()-last>=30:report(out,manifest);last=time.monotonic()
            time.sleep(5)
        state.update(status='complete' if all(s['status']=='complete' for s in state['jobs'].values()) else 'failed',finished_at=datetime.now(timezone.utc).isoformat())
        atomic(out/'queue.json',state);report(out,manifest)
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc))
        atomic(out/'queue.json',state);report(out,manifest);raise
    finally:
        for worker,_,_ in workers.values():
            if worker.poll() is None:worker.terminate();worker.wait()
        for stream in logs.values():stream.close()
        for handle in leases.values():handle.close()


if __name__=='__main__':main()
