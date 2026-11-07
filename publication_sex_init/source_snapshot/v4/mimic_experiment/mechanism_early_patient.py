"""After P0, diagnose the first clipped real batch on the otherwise free GPU 0."""
import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .prepare import atomic
from . import mechanism_report


def main():
    root=Path(__file__).resolve().parents[1];out=root/'mimic_experiment/results/weak_prior_ablation_cuda/mechanism_diagnosis'
    lock=(out/'early_patient_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'status':'waiting_for_P0_completion','physical_gpu':0,'controller_pid':os.getpid(),
           'selection_rule':'first epoch46 real batch with all-parameter preclip norm > original training clipping threshold10'}
    atomic(out/'early_patient_queue.json',state)
    while True:
        q=json.loads((out/'queue.json').read_text())
        if q['phase'] in ['P1_experiment3_patient_decomposition','finished']:break
        if q.get('status')=='stopped':state['status']='stopped_with_main';atomic(out/'early_patient_queue.json',state);return
        time.sleep(5)
    records=[json.loads(p.read_text()) for p in sorted((out/'experiment1/steps').glob('e046*.json'))]
    selected=next((r for r in records if r['before']['all_parameter_preclip_norm']>10),None)
    if selected is None:state['status']='complete';state['reason']='no clipped epoch46 real batch';atomic(out/'early_patient_queue.json',state);return
    case={'id':f"first_clipped_e46_b{selected['batch']:03d}",'snapshot':selected['before_snapshot'],'batch_path':selected['batch_path'],
          'batch_kind':f"real_training_e46b{selected['batch']}",'selection_rule':state['selection_rule']}
    path=out/'experiment3/cases.json';cases=json.loads(path.read_text())
    if not any(c['id']==case['id'] for c in cases):atomic(path,cases+[case])
    done=out/'experiment3'/case['id']/'results.json'
    if done.exists():state.update(status='complete',case=case);atomic(out/'early_patient_queue.json',state);return
    gpu_lock=Path('/tmp/latentode_gpu_0.lock').open('a')
    while True:
        try:fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:time.sleep(5);continue
        busy=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if busy:fcntl.flock(gpu_lock,fcntl.LOCK_UN);time.sleep(5);continue
        break
    manifest=json.loads((out/'manifest.json').read_text());env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':manifest['source_snapshot'],'PYTHONUNBUFFERED':'1'}
    with (out/'early_patient.log').open('a') as f:
        worker=subprocess.Popen([sys.executable,str(Path(manifest['analysis_source'])/'mechanism_patients.py'),'--out',str(out),'--gpu','0','--case',case['id']],cwd=manifest['source_snapshot'],env=env,stdout=f,stderr=subprocess.STDOUT)
        state.update(status='running',pid=worker.pid,case=case);atomic(out/'early_patient_queue.json',state)
        result=worker.wait()
    gpu_lock.close();state.update(status='complete' if result==0 else 'failed',exit_code=result);atomic(out/'early_patient_queue.json',state)
    importlib.reload(mechanism_report).generate(out)


if __name__=='__main__':main()
