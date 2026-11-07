"""Add one parameter-matched deterministic ODE-AE on physical GPU 1."""
import argparse
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from paper_odeae.scaling_study import identity
from .baseline_report import generate,NAMES
from .prepare import atomic
from .report import read

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/'mimic_experiment/results/baselines_with_deterministic_cuda'
BASELINES=ROOT/'mimic_experiment/results/baselines_single_cuda'
KEY='deterministic_matched'
TITLE='deterministic_22545（同预算）'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    reference=ROOT/'mimic_experiment/results/latent2x_both_cuda'
    native=ROOT/'paper_odeae/results/capacity_matched/deterministic_22545/seed_0/config.json'
    base=read(BASELINES/'configs/ode_ae.json');target=read(reference/'seed_0/hardware.json')['parameters']
    dataset=Path(base['data']);sha=digest(dataset)
    audit=read(BASELINES/'data_verification.json')
    if audit['status']!='passed' or sha!=audit['dataset_sha256'] or sha!=base['dataset_sha256']:
        raise RuntimeError('Use the same audited MIMIC chronology_v2 data as the completed reference')
    if read(reference/'seed_0/metrics.json')['status']!='complete':raise RuntimeError('Missing completed reference')
    schema=read(reference/'seed_0/schema.json');features=sum(field.get('classes',1) for field in schema)
    # Six-dimensional states; age INIT; two independent ODEs. Match the total
    # using only hidden widths, keeping the MLP/ODE allocation close to the source.
    count=8*322+(features+13)*318+(features+7)*318+26*242+features+24
    if target!=43280 or features!=44 or count!=target:raise RuntimeError('Recompute the exact capacity for a changed schema')
    cfg={**base,'model':'deterministic','variant':KEY,'study':'mimic_deterministic_matched_single',
         'output':str(out/KEY/'seed_0'),'comparison_output':str(out),'physical_gpu':1,
         'latent':6,'mlp_width':318,'init_width':322,'update_width':318,'decoder_width':318,
         'ode_width':242,'encoder_ode_width':242,'generator_ode_width':242,
         'encoder_direction':'backward','encoder_initialization':'age','shared_vector_field':False,
         'collect_prior_predictions':False,'collect_jacobian_states':False,
         'kl_weight':0.,'prior_weight':0.,'jacobian_weight':0.,'target_parameters':target,
         'source_family':'deterministic_22545','source_adni_config':str(native),'verify_reference':False,
         'capacity_change':'six-dimensional deterministic state; age INIT 322; UPDATE/DECODER 318; independent ODEs 242; exact 43,280 parameters',
         'history_semantics':'history-conditioned reconstruction','future_semantics':'ODE rollout without future observation updates',
         'age_covariate':'age initialization; biological years/100, separate from hourly ODE clock'}
    config=out/'configs'/f'{KEY}.json';config.parent.mkdir(exist_ok=True)
    if config.exists() and read(config)!=cfg:raise RuntimeError('Preserve existing experiment configuration')
    atomic(config,cfg);atomic(out/'data_verification.json',{**audit,'reused_audit':str(BASELINES/'data_verification.json'),'dataset_hash_rechecked':True})
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    files=list(read(BASELINES/'manifest.json')['source_hashes'])+['mimic_experiment/matched_study.py']
    if not (snapshot/'source_hashes.json').exists():
        for name in files:
            path=snapshot/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,path)
        atomic(snapshot/'source_hashes.json',{name:digest(snapshot/name) for name in files})
    hashes=read(snapshot/'source_hashes.json')
    for name,sha_ in hashes.items():
        if digest(snapshot/name)!=sha_:raise RuntimeError('Frozen experiment source changed')
    model_outputs={key:str(BASELINES/key/'seed_0') for key in NAMES};model_outputs[KEY]=str(out/KEY/'seed_0')
    model_queues={key:str(BASELINES/'queue.json') for key in NAMES};model_queues[KEY]=str(out/'queue.json')
    atomic(out/'manifest.json',{'dataset':str(dataset),'dataset_sha256':sha,'reference_output':str(reference),
           'additional_model_names':{KEY:TITLE},'model_outputs':model_outputs,'model_queues':model_queues,
           'source_adni_config':str(native),'source_adni_config_sha256':digest(native),'source_hashes':hashes,
           'target_parameters':target,'physical_gpu':1,'environment':'ode','seed':0,'effective_batch_size':128,
           'strategy':'new deterministic model on one GPU; reuse existing model results in the six-model comparison',
           'parameter_breakdown':{'init':2582,'update':18132,'decoder':16262,'backward_ode':3152,'forward_ode':3152}})
    return cfg,snapshot,config


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(DEFAULT_OUT));args=parser.parse_args()
    if Path(sys.prefix).name!='ode':raise RuntimeError('Use the ode environment')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True);previous=read(out/'queue.json',{})
    if previous.get('pid') and identity(previous['pid'])==previous.get('start_time'):raise RuntimeError('Matched experiment already running')
    cfg,snapshot,config=prepare(out)
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
           'jobs':{KEY:{'status':'waiting','phase':'checks','physical_gpu':1,'environment':'ode'}}}
    atomic(out/'queue.json',state);generate(out);worker=None
    try:
        for stage in ['checks','smoke','training']:
            folder=Path(cfg['output']) if stage=='training' else out/'checks'/KEY/stage
            expected=folder/('checks.json' if stage=='checks' else 'metrics.json')
            if expected.exists():
                result=read(expected)
                if result['status']!=('passed' if stage=='checks' else 'cuda_smoke_passed' if stage=='smoke' else 'complete'):
                    raise RuntimeError('Incomplete previous stage')
                continue
            state['jobs'][KEY].update(status='waiting',phase=stage);atomic(out/'queue.json',state)
            while subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                generate(out);time.sleep(10)
            folder.mkdir(parents=True,exist_ok=True)
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONPATH=str(snapshot),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                             SWANLAB_LOG_DIR=str(ROOT/'swanlog'),SWANLAB_EXPERIMENT_NAME=f'mimic-deterministic-matched-{stage}-seed0')
            with (folder/'run.log').open('a') as log:
                worker=subprocess.Popen([sys.executable,'-u','-m','mimic_experiment.train_baseline','--config',str(config),'--stage',stage],
                         cwd=snapshot,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                state['jobs'][KEY].update(status='running',phase=stage,pid=worker.pid);atomic(out/'queue.json',state)
                print(f'{KEY}: {stage}, physical GPU 1, ode, PID {worker.pid}',flush=True)
                last=0
                while worker.poll() is None:
                    if time.monotonic()-last>=30:generate(out);last=time.monotonic()
                    time.sleep(2)
                if worker.returncode or not expected.exists():raise RuntimeError(f'{stage} failed; inspect {folder}/run.log')
                result=read(expected)
                expected_status='passed' if stage=='checks' else 'cuda_smoke_passed' if stage=='smoke' else 'complete'
                if result['status']!=expected_status:raise RuntimeError(f'{stage}: incomplete output')
                if stage=='checks' and result['parameters']!=cfg['target_parameters']:raise RuntimeError('Capacity check failed')
                print(f'{stage}: passed',flush=True)
        state['jobs'][KEY].update(status='complete',phase='tested',finished_at=datetime.now(timezone.utc).isoformat())
        state['status']='reporting_other_models';atomic(out/'queue.json',state);generate(out)
        # Keep a single report current while the earlier independent baselines finish.
        while True:
            queue=read(BASELINES/'queue.json',{});alive=queue.get('pid') and identity(queue['pid'])==queue.get('start_time')
            if queue.get('status') in ['complete','failed'] or not alive:break
            generate(out);time.sleep(30)
        state.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat());atomic(out/'queue.json',state);generate(out)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));state['jobs'][KEY].update(status='failed')
        atomic(out/'queue.json',state);generate(out);raise
    finally:
        if worker is not None and worker.poll() is None:worker.terminate();worker.wait()


if __name__=='__main__':main()
