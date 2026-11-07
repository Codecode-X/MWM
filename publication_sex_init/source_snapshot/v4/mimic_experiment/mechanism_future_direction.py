"""Read-only history/future excitation of the same unstable encoder direction."""
import argparse
import json
from pathlib import Path
import sys

import torch

from mechanism_common import initialize,model_from,parameter_hash,flat_gradient,relative,cosine
sys.path=[p for p in sys.path if Path(p or '.').resolve()!=Path(__file__).resolve().parent]
from paper_odeae.models import feature_errors
from mimic_experiment.prepare import atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    out=Path(a.out);manifest,cfg,payload,modes=initialize(out,a.gpu);folder=out/'encoder_path_followup'
    batch=torch.load(out/'experiment1/batches/e046_b121.pt',map_location='cuda',weights_only=False)['batch']
    v=torch.load(out/'experiment3/abnormal_real_e46b121/patient_gradient_vectors.pt',map_location='cuda',weights_only=False)
    matrices=torch.load(folder/'matrices.pt',map_location='cuda',weights_only=False);jac=matrices['encoder_field_jacobians']['abnormal']
    model,saved=model_from(manifest['existing_abnormal_snapshot'],cfg,payload['schema']);before=parameter_hash(model)
    prediction=model(batch,sample=False);errors=feature_errors(prediction['prediction'],batch['values'],batch['mask'],payload['schema'])
    future=torch.arange(batch['values'].shape[1],device='cuda')[None,:]>=batch['cutoff'][:,None]
    per=(errors*future[...,None]).sum(1)/batch['mask'].sum((0,1)).clamp_min(1);loss=per.sum(-1)[8]
    gz=torch.autograd.grad(loss,prediction['mu'],retain_graph=True)[0][8].detach()
    full=flat_gradient(loss,tuple(model.forward_ode.parameters())).detach();enc=jac.T@gz;direct=full-enc
    u,s,vh=torch.linalg.svd(jac,full_matrices=False);history_direction=matrices['unit_state_gradient_directions']['abnormal']
    source=json.loads((out/'experiment3/abnormal_real_e46b121/results.json').read_text())['patients'][8]
    result={'physical_gpu':a.gpu,'parameter_device':str(next(model.parameters()).device),'parameter_hash':before,'optimizer_steps':0,
            'patient_index':8,'history_loss_state_gradient_norm':source['cutoff_history_gradient_norm'],'future_loss_state_gradient_norm':float(gz.norm()),
            'state_gradient_history_future_cosine':cosine(history_direction,gz),
            'history_state_gradient_alignment_with_encoder_top_direction':float(torch.dot(history_direction,u[:,0])),
            'future_state_gradient_alignment_with_encoder_top_direction':float(torch.dot(gz/gz.norm(),u[:,0])),
            'history_encoder_field_gradient_norm':source['encoder_field_gradient_norm'],'future_encoder_field_gradient_norm':float(enc.norm()),
            'future_direct_field_gradient_norm':float(direct.norm()),'future_total_patient_field_gradient_norm':float(full.norm()),
            'patient_history_to_patient_future_field_gradient_cosine':cosine(v['history'][8],full),
            'history_encoder_to_future_encoder_cosine':cosine(v['encoder_path'][8],enc)}
    if parameter_hash(model)!=before:raise RuntimeError('Future direction analysis changed parameters')
    atomic(folder/'future_direction.json',result)


if __name__=='__main__':main()
