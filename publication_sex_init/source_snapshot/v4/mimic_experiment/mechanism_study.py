"""Ordered mechanism study: recovery, update trace, precision, then patients."""
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .prepare import atomic
from . import mechanism_report


def main():
    root=Path(__file__).resolve().parents[1];out=root/'mimic_experiment/results/weak_prior_ablation_cuda/mechanism_diagnosis';out.mkdir(exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=root/'mimic_experiment/results/direction_full_batch_cuda/gradient_sensitivity_analysis'
    source=prior.parent/'source_snapshot';cfg=json.loads((prior.parent/'configs/forward_shared.json').read_text())
    names=['direction_gradient_common.py','mechanism_common.py','mechanism_trace.py','mechanism_sensitivity.py','mechanism_precision.py','mechanism_patients.py']
    hashes={n:hashlib.sha256((root/'mimic_experiment'/n).read_bytes()).hexdigest() for n in names}
    version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16];analysis=out/'analysis_source'/version;analysis.mkdir(parents=True,exist_ok=True)
    for name in names:
        target=analysis/name
        if not target.exists():shutil.copy2(root/'mimic_experiment'/name,target)
    manifest={'reference_config':cfg,'source_snapshot':str(source),'source_sha256':json.loads((source/'source_hashes.json').read_text()),
              'analysis_source':str(analysis),'analysis_source_sha256':hashes,'prior_diagnostics':str(prior),
              'replay_start':str(prior/'checkpoints/epoch_042_replay.pt'),'stable_41':str(prior/'checkpoints/epoch_041_original.pt'),
              'stable_45':str(prior/'checkpoints/epoch_045_replay.pt'),'restored_81':str(prior/'checkpoints/epoch_081_original.pt'),
              'existing_abnormal_snapshot':str(prior/'checkpoints/epoch_046_peak.pt'),
              'original_swanlab':str(root/'swanlog/run-20261031_230012-vunx57yz'),
              'original_rng_checkpoint_available':False,'replay_semantics':'per-epoch seed reconstruction, verified separately against previous replay each batch and original sampled SwanLab logs',
              'training_updates_only_in':'mechanism_trace.py; no optimizer constructed by any other worker',
              'fixed_validation_sample':{'previous_sample':64,'first_n':8,'seed':7},
              'trace_epochs':[44,45,46],'sensitivity_tolerance':[1e-6,1e-8],
              'thresholds_predeclared':{'stable_reference_epochs':[41,42,45],'stable_per_patient_quantile':.99,'single_update_fold_increase':5.},
              'reference_rk4_step_hours':[.2,.1,.05],'maximum_refinement_step_hours':.0125,'reference_step_halving_rtol':.001}
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text())
        if old!=manifest:raise RuntimeError('Preserve the existing mechanism-study manifest; use a new directory for changed settings')
    else:atomic(out/'manifest.json',manifest)
    state={'controller_pid':os.getpid(),'status':'running','phase':'waiting_for_available_GPUs','jobs':{}}
    active={};handles={};logs={}
    def report():
        atomic(out/'queue.json',state);importlib.reload(mechanism_report).generate(out)
    def stop(signum,frame):
        for proc in active.values():
            if proc.poll() is None:proc.terminate()
        state['status']='stopped';report();raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def group(jobs,phase):
        state['phase']=phase;running={};ok=True
        for name,script,gpu,extra,done in jobs:
            if done.exists():state['jobs'][name]={'status':'complete','physical_gpu':gpu,'reused_existing_result':True};continue
            while True:
                h=Path(f'/tmp/latentode_gpu_{gpu}.lock').open('a')
                try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:h.close();state['jobs'][name]={'status':'waiting_for_GPU','physical_gpu':gpu};report();time.sleep(20);continue
                occupied=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
                if occupied:h.close();report();time.sleep(20);continue
                handles[name]=h;break
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':str(source),'SWANLAB_LOG_DIR':str(root/'swanlog'),'PYTHONUNBUFFERED':'1'}
            logs[name]=(out/(name+'.log')).open('a');command=[sys.executable,str(analysis/script),'--out',str(out),'--gpu',str(gpu),*extra]
            proc=subprocess.Popen(command,cwd=source,env=env,stdout=logs[name],stderr=subprocess.STDOUT)
            active[name]=proc;running[name]=proc;state['jobs'][name]={'status':'running','physical_gpu':gpu,'pid':proc.pid,'command':command};report()
        while running:
            for name,proc in list(running.items()):
                if proc.poll() is None:continue
                state['jobs'][name].update(status='complete' if proc.returncode==0 else 'failed',exit_code=proc.returncode)
                if proc.returncode!=0:ok=False
                handles.pop(name).close();running.pop(name);active.pop(name)
                if name=='trace':
                    path=out/'experiment1/index.json'
                    index=json.loads(path.read_text()) if path.exists() else {'steps':[]}
                    index.update(producer_done=True,producer_exit_code=proc.returncode);atomic(path,index)
            report()
            if running:time.sleep(20)
        return ok
    (out/'experiment1').mkdir(exist_ok=True)
    if not (out/'experiment1/index.json').exists():atomic(out/'experiment1/index.json',{'steps':[],'producer_done':False})
    complete1=out/'experiment1/complete.json'
    if not complete1.exists():
        ok=group([('trace','mechanism_trace.py',3,[],complete1),('update_sensitivity','mechanism_sensitivity.py',1,[],complete1)],'P0_experiment1_update_trace')
        atomic(complete1,{'status':'complete' if ok else 'partial_fallback_to_fixed_snapshots','original_rng_available':False})
    (out/'experiment2').mkdir(exist_ok=True)
    post=out/'experiment1/snapshots/e046_b121_post.pt'
    snapshots=[{'id':'epoch41','path':manifest['stable_41']},{'id':'epoch45','path':manifest['stable_45']},
               {'id':'epoch46_batch121_pre','path':manifest['existing_abnormal_snapshot']},{'id':'epoch81','path':manifest['restored_81']}]
    if post.exists():snapshots.insert(3,{'id':'epoch46_batch121_post','path':str(post)})
    for snap in snapshots:snap['file_sha256']=hashlib.sha256(Path(snap['path']).read_bytes()).hexdigest()
    reproduction=json.loads((out/'experiment1/abnormal_batch_reproduction.json').read_text()) if (out/'experiment1/abnormal_batch_reproduction.json').exists() else None
    if reproduction and reproduction['parameter_max_absolute_difference']==0:
        next(s for s in snapshots if s['id']=='epoch46_batch121_pre')['equivalent_full_adam_rng_snapshot']=str(out/'experiment1/snapshots/e046_b121_pre.pt')
    atomic(out/'experiment2/snapshots.json',snapshots)
    for offset in range(0,len(snapshots),3):
        jobs=[(f"precision_{s['id']}",'mechanism_precision.py',gpu,['--snapshot-id',s['id']],out/'experiment2'/s['id']/'results.json') for s,gpu in zip(snapshots[offset:offset+3],[1,2,3])]
        group(jobs,'P0_experiment2_solver_convergence')
    (out/'experiment3').mkdir(exist_ok=True)
    real_batch=out/'experiment1/batches/e046_b121.pt';real_batch=real_batch if real_batch.exists() else Path(manifest['existing_abnormal_snapshot'])
    cases=[{'id':'abnormal_real_e46b121','snapshot':manifest['existing_abnormal_snapshot'],'batch_path':str(real_batch),'batch_kind':'real_training_e46b121'},
           {'id':'stable_matched_same_patients','snapshot':manifest['stable_45'],'batch_path':str(real_batch),'batch_kind':'fixed_diagnostic_reuse_of_e46b121_inputs_at_stable_e45_parameters'}]
    stable_pre=out/'experiment1/snapshots/e045_b121_pre.pt'
    if stable_pre.exists():cases.append({'id':'stable_real_e45b121','snapshot':str(stable_pre),'batch_path':str(out/'experiment1/batches/e045_b121.pt'),'batch_kind':'real_training_e45b121'})
    atomic(out/'experiment3/cases.json',cases)
    group([(f"patients_{s['id']}",'mechanism_patients.py',gpu,['--case',s['id']],out/'experiment3'/s['id']/'results.json') for s,gpu in zip(cases,[1,2,3])],'P1_experiment3_patient_decomposition')
    state['phase']='finished';state['status']='complete' if all(j['status']=='complete' for j in state['jobs'].values()) else 'finished_with_failed_jobs';report()


if __name__=='__main__':main()
