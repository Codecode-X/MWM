"""Read-only exact encoder parameter Jacobians and crossed gradient directions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

from mechanism_common import initialize,model_from,parameter_hash,encode_only,historical_loss,relative,tensor_hash
sys.path=[p for p in sys.path if Path(p or '.').resolve()!=Path(__file__).resolve().parent]
from mimic_experiment.prepare import atomic
from mimic_experiment.train_baseline import save_checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,required=True);args=p.parse_args()
    out=Path(args.out);manifest,cfg,payload,modes=initialize(out,args.gpu);folder=out/'encoder_path_followup';folder.mkdir(exist_ok=True)
    atomic(folder/'status.json',{'status':'running','physical_gpu':args.gpu,'optimizer_steps':0})
    case=json.loads((out/'experiment3/abnormal_real_e46b121/results.json').read_text())
    patient=max(case['patients'],key=lambda r:r['history_field_gradient_norm'])['index']
    batch=torch.load(case['case']['batch_path'],map_location='cuda',weights_only=False)['batch']
    native_vectors=torch.load(out/'experiment3/abnormal_real_e46b121/patient_gradient_vectors.pt',map_location='cuda',weights_only=False)
    results={};matrices={};directions={}
    for label,path in [('stable',manifest['stable_45']),('abnormal',manifest['existing_abnormal_snapshot'])]:
        model,saved=model_from(path,cfg,payload['schema']);before=parameter_hash(model);z=encode_only(model,batch)
        loss,_,_,_,_=historical_loss(model,z,batch,payload['schema']);gz=torch.autograd.grad(loss,z,retain_graph=True)[0][patient].detach()
        groups=[('INIT',tuple(model.init.parameters())),('UPDATE',tuple(model.update.parameters())),('ODE',tuple(model.forward_ode.parameters()))]
        params=tuple(p for name,ps in groups for p in ps)
        jacobians=torch.autograd.grad(z[patient],params,grad_outputs=torch.eye(28,device='cuda'),is_grads_batched=True,retain_graph=True)
        row={'snapshot':path,'parameter_hash':before,'cutoff_state_norm':float(z[patient].detach().norm()),'state_gradient_norm':float(gz.norm()),'groups':{}}
        cursor=0
        for name,ps in groups:
            jac=torch.cat([g.reshape(28,-1) for g in jacobians[cursor:cursor+len(ps)]],dim=1).detach();cursor+=len(ps)
            u,s,vh=torch.linalg.svd(jac,full_matrices=False);gradient=jac.T@gz;gain=float(gradient.norm()/gz.norm())
            row['groups'][name]={'jacobian_largest_singular_value':float(s[0]),'jacobian_frobenius_norm':float(jac.norm()),
                                 'actual_vjp_gain':gain,'actual_over_operator_bound':gain/float(s[0]),
                                 'gradient_alignment_with_top_left_singular_vector':float(torch.dot(gz/gz.norm(),u[:,0]).abs()),
                                 'patient_history_parameter_gradient_norm':float(gradient.norm())}
            if name=='ODE':
                matrices[label]=jac;directions[label]=gz/gz.norm()
                if label=='abnormal':
                    err=relative(gradient,native_vectors['encoder_path'][patient]);row['field_vjp_vs_patient_decomposition_relative_error']=err
                    if err>1e-3:raise RuntimeError('Encoder parameter Jacobian does not reproduce patient VJP')
        if parameter_hash(model)!=before:raise RuntimeError('Read-only encoder analysis changed weights')
        results[label]=row
    crossed={jac_label:{direction_label:float((jac.T@direction).norm()) for direction_label,direction in directions.items()} for jac_label,jac in matrices.items()}
    save_checkpoint(folder/'matrices.pt',{'encoder_field_jacobians':matrices,'unit_state_gradient_directions':directions})
    atomic(folder/'analysis.json',{'physical_gpu':args.gpu,'parameter_device':'cuda:0','optimizer_steps':0,'patient_index':patient,
                                  'patient_batch_kind':'real_training_e46b121; same inputs at stable and abnormal parameters',
                                  'model_rtol':cfg['rtol'],'model_atol':cfg['atol'],'models':results,'crossed_unit_vjp_gains':crossed,
                                  'state_gradient_direction_cosine':float(torch.dot(directions['stable'],directions['abnormal'])),
                                  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    atomic(folder/'status.json',{'status':'complete','physical_gpu':args.gpu,'optimizer_steps':0})


if __name__=='__main__':main()
