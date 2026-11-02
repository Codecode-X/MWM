"""Read-only CUDA measurements for every before/after Adam snapshot."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from mechanism_common import initialize,fixed_cohort,model_from,encode_only,update_probe,parameter_hash
from mimic_experiment.prepare import atomic
from mimic_experiment.train_baseline import save_checkpoint


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--gpu',type=int,required=True);args=parser.parse_args()
    out=Path(args.out);manifest,cfg,payload,modes=initialize(out,args.gpu);folder=out/'experiment1';cache=folder/'sensitivity';cache.mkdir(exist_ok=True)
    batch,audit=fixed_cohort(payload,8);atomic(folder/'fixed_cohort.json',audit)
    probe,saved=model_from(manifest['stable_41'],cfg,payload['schema'],False)
    with torch.no_grad():fixed=encode_only(probe,batch).detach()
    save_checkpoint(folder/'fixed_inputs_and_states.pt',{'batch':batch,'state':fixed,'state_source':manifest['stable_41'],'audit':audit})
    base=update_probe(probe,probe,batch,fixed);atomic(folder/'baseline_41.json',base)
    stable=[]
    prior=Path(manifest['prior_diagnostics'])
    for name in ['epoch_041_original','epoch_042_replay','epoch_045_replay']:
        old=json.loads((prior/'sensitivity'/(name+'.json')).read_text())
        stable.extend(next(c['per_patient_gain'] for c in old['conditions'] if c['condition']=='history_tight'))
    thresholds={'stable_reference':'64 fixed patients at saved epochs 41,42,45; defined before inspecting new update trace',
                'stable_p99':float(np.quantile(stable,.99)),'adjacent_update_fold_increase':5.,'fixed_sample_count':8,
                'sensitivity_rtol':1e-6,'sensitivity_atol':1e-8,'fixed_z_source':manifest['stable_41']}
    atomic(folder/'thresholds.json',thresholds)
    atomic(folder/'sensitivity_hardware.json',{'physical_gpu':args.gpu,'parameter_device':str(next(probe.parameters()).device),
                                             'state_device':str(fixed.device),'mode':'read_only_no_optimizer','gpu_name':torch.cuda.get_device_name()})
    while True:
        index=json.loads((folder/'index.json').read_text()) if (folder/'index.json').exists() else {'steps':[]}
        for step in index['steps']:
            file=cache/(step['id']+'.json')
            if file.exists():continue
            result={}
            for key,path,hashkey in [('before',step['pre'],'before_parameter_hash'),('after',step['post'],'after_parameter_hash')]:
                h=step[hashkey];cached=cache/('state_'+h+'.json')
                if cached.exists():result[key]=json.loads(cached.read_text());continue
                saved=torch.load(path,map_location='cuda',weights_only=False);probe.load_state_dict(saved['state_dict'])
                if parameter_hash(probe)!=h:raise RuntimeError('Parameter hash mismatch')
                start=time.monotonic();data=update_probe(probe,probe,batch,fixed)
                if parameter_hash(probe)!=h:raise RuntimeError('Diagnostic changed parameters')
                data.update(parameter_hash=h,snapshot=path,elapsed_seconds=time.monotonic()-start)
                atomic(cached,data);result[key]=data;del saved
            flags={}
            for name in ['A_reencoded','B_fixed_state']:
                a=result['before'][name]['past']['gain']['median'];b=result['after'][name]['past']['gain']['median']
                flags[name]={'after_before_ratio':b/max(a,1e-30),'above_stable_p99_before':a>thresholds['stable_p99'],
                             'above_stable_p99_after':b>thresholds['stable_p99'],'crosses_stable_p99':a<=thresholds['stable_p99']<b,
                             'single_update_fivefold':b>=5*a}
            atomic(file,{'step':step,**result,'flags':flags})
            atomic(folder/'sensitivity_status.json',{'status':'running','epoch':step['epoch'],'batch':step['batch'],'measured_updates':len(list(cache.glob('e*.json')))})
            if step['batch']%10==0:print(json.dumps({'id':step['id'],'A_after':result['after']['A_reencoded']['past']['gain']['median'],'B_after':result['after']['B_fixed_state']['past']['gain']['median']}),flush=True)
        if index.get('producer_done'):break
        time.sleep(2)
    atomic(folder/'sensitivity_status.json',{'status':'complete','measured_updates':len(list(cache.glob('e*.json')))})


if __name__=='__main__':main()
