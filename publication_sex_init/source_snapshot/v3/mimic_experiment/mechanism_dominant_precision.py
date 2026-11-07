"""Read-only numerical convergence for the dominant real patient's full encoder path."""
import argparse
import json
from pathlib import Path
import sys
import gc

import torch

from mechanism_common import initialize,model_from,encode_only,encode_solver,historical_loss,flat_gradient,relative,parameter_hash
sys.path=[p for p in sys.path if Path(p or '.').resolve()!=Path(__file__).resolve().parent]
from mimic_experiment.prepare import atomic
from mimic_experiment.train_baseline import save_checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    out=Path(a.out);manifest,cfg,payload,modes=initialize(out,a.gpu);folder=out/'dominant_patient_precision';folder.mkdir(exist_ok=True)
    atomic(folder/'status.json',{'status':'running','physical_gpu':a.gpu,'optimizer_steps':0})
    original=torch.load(out/'experiment3/abnormal_real_e46b121/patient_gradient_vectors.pt',map_location='cuda',weights_only=False)
    case=json.loads((out/'experiment3/abnormal_real_e46b121/results.json').read_text());batch=torch.load(case['case']['batch_path'],map_location='cuda',weights_only=False)['batch'];patient=8
    m32,saved=model_from(manifest['existing_abnormal_snapshot'],cfg,payload['schema']);m64,_=model_from(manifest['existing_abnormal_snapshot'],cfg,payload['schema'],dtype=torch.float64)
    with torch.no_grad():native=encode_only(m32,batch).detach()
    settings=[('train32',torch.float32,'tsit5',1e-4,1e-6,None),('diagnostic32',torch.float32,'tsit5',1e-6,1e-8,None),
              ('strict64',torch.float64,'tsit5',1e-9,1e-11,None),('rk4_h0.2',torch.float64,'rk4',None,None,.2),
              ('rk4_h0.1',torch.float64,'rk4',None,None,.1),('rk4_h0.05',torch.float64,'rk4',None,None,.05)]
    rows=[];vs={};i=0;previous=None;converged=False
    while i<len(settings):
        name,dtype,method,rtol,atol,step=settings[i];i+=1;model=m32 if dtype==torch.float32 else m64;model.rtol,model.atol=rtol,atol
        b={k:v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v for k,v in batch.items()};fingerprint=parameter_hash(model)
        encoded=encode_solver(model,b,method,step);result={'setting':name,'rtol':rtol,'atol':atol,'dtype':str(dtype),'step_hours':step,'modes':{}}
        vectors={}
        for mode in ['locked_native_cutoff','end_to_end']:
            z=encoded+(native.to(dtype)-encoded).detach() if mode=='locked_native_cutoff' else encoded
            loss,per,states,_,_=historical_loss(model,z,b,payload['schema'],method=method,step=step)
            g=flat_gradient(per[patient],tuple(model.forward_ode.parameters())).detach()
            vectors[mode+'_field_gradient']=g;vectors[mode+'_past_endpoint']=states[patient,0].detach()
            result['modes'][mode]={'history_loss_contribution':float(per[patient].detach()),'field_gradient_norm':float(g.norm()),
                                   'native_patient_gradient_relative_difference':relative(g.double(),original['history'][patient].double())}
            del loss,per,states
        vectors['encoded_state']=encoded[patient].detach();vs[name]=vectors
        if method=='rk4':
            if previous is not None:
                differences={k:relative(vectors[k],vs[previous][k]) for k in vectors};result['halving_differences']=differences;converged=max(differences.values())<1e-3
            previous=name
            if i==len(settings) and not converged and step>.0125:
                finer=step/2;settings.append((f'rk4_h{finer}',torch.float64,'rk4',None,None,finer))
        if parameter_hash(model)!=fingerprint:raise RuntimeError('Dominant-patient convergence changed parameters')
        rows.append(result);atomic(folder/'progress.json',rows);del encoded;gc.collect();torch.cuda.empty_cache()
    for row in rows:row['relative_differences_to_reference']={k:relative(vs[row['setting']][k].double(),vs[previous][k]) for k in vs[previous]}
    save_checkpoint(folder/'vectors.pt',vs)
    atomic(folder/'analysis.json',{'patient_index':patient,'original_real_batch_size':len(batch['cutoff']),'batch_kind':'original complete real e46b121 batch; patient loss retains its original per-field batch denominators',
                                  'snapshot':manifest['existing_abnormal_snapshot'],'physical_gpu':a.gpu,'parameter_device':'cuda:0','optimizer_steps':0,
                                  'reference':previous,'reference_halving_converged':converged,'settings':rows})
    atomic(folder/'status.json',{'status':'complete','physical_gpu':a.gpu,'optimizer_steps':0})


if __name__=='__main__':main()
