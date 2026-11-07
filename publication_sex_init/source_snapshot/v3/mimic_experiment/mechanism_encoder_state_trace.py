"""Read-only Jacobians through each original encoder ODE/UPDATE transition."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

from mechanism_common import initialize,model_from,parameter_hash
sys.path=[p for p in sys.path if Path(p or '.').resolve()!=Path(__file__).resolve().parent]
from paper_odeae.models import encode_values
from mimic_experiment.prepare import atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    out=Path(a.out);manifest,cfg,payload,modes=initialize(out,a.gpu);folder=out/'encoder_path_followup'
    results={};case=json.loads((out/'experiment3/abnormal_real_e46b121/results.json').read_text())
    patient=8;batch=torch.load(case['case']['batch_path'],map_location='cuda',weights_only=False)['batch'];count=int(batch['cutoff'][patient])
    for label,path in [('stable',manifest['stable_45']),('abnormal',manifest['existing_abnormal_snapshot'])]:
        model,saved=model_from(path,cfg,payload['schema']);before=parameter_hash(model)
        values,ages,cut=batch['values'],batch['ages'],batch['cutoff'];inputs=encode_values(values,batch['context_mask'],model.schema)
        h=model.init(batch['initialization_age'][:,None]/100.);initial=h;previous=ages[:,0];evolved_states=[];updated_states=[];local_update=[]
        for j in range(batch['max_context']):
            index=torch.minimum(torch.full_like(cut,j),cut-1);active=j<cut;time=ages.gather(1,index[:,None]).squeeze(1)
            x=inputs[torch.arange(len(cut),device='cuda'),index];evolved=model.solve(model.forward_ode,h,torch.where(active,time-previous,torch.zeros_like(time)))
            net_input=torch.cat([evolved,x],-1);proposal=model.update(net_input);h=torch.where(active[:,None],proposal,h)
            evolved_states.append(evolved);updated_states.append(h)
            with torch.no_grad():
                derivative=1-model.update[1](model.update[0](net_input[patient])).square()
                ju=(model.update[2].weight*derivative[None,:])@model.update[0].weight[:,:28]
                local_update.append(float(torch.linalg.svdvals(ju)[0]))
            previous=torch.where(active,time,previous)
        states=[initial]+evolved_states[:count]+updated_states[:count]
        gradients=torch.autograd.grad(h[patient],states,grad_outputs=torch.eye(28,device='cuda'),is_grads_batched=True)
        gains=[float(torch.linalg.svdvals(g[:,patient,:])[0]) for g in gradients]
        rows=[]
        for j in range(count):
            rows.append({'visit_index':j,'time_hours':float(ages[patient,j]),'delta_hours':0. if j==0 else float(ages[patient,j]-ages[patient,j-1]),
                         'cutoff_wrt_preupdate_state_gain':gains[1+j],'cutoff_wrt_postupdate_state_gain':gains[1+count+j],
                         'local_update_state_jacobian_largest_singular_value':local_update[j]})
        if parameter_hash(model)!=before:raise RuntimeError('Encoder state trace changed parameters')
        results[label]={'snapshot':path,'parameter_hash':before,'cutoff_wrt_init_state_gain':gains[0],'visits':rows}
    atomic(folder/'state_trace.json',{'models':results,'patient_index':patient,'physical_gpu':a.gpu,'parameter_device':'cuda:0','optimizer_steps':0,
                                      'rtol':cfg['rtol'],'atol':cfg['atol'],'batch_kind':'same real e46b121 inputs at stable and abnormal parameter snapshots',
                                      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__=='__main__':main()
