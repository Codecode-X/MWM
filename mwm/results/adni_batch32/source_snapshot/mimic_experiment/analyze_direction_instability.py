"""Read-only CUDA conditioning probes of saved direction-study checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--gpu',type=int,default=3);args=parser.parse_args()
    root=Path(args.root).resolve();sys.path.insert(0,str(root/'source_snapshot'))
    import torch
    from mimic_experiment.data import ICUContexts,collate
    from mimic_experiment.train_full_batch import make_model
    from paper_odeae.train import device_batch
    from paper_odeae.models import feature_errors,prior_errors
    from paper_odeae.jacobian import jacobian_regularization
    if Path(sys.prefix).name!='ode' or not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):raise RuntimeError('Explicit assigned CUDA GPU required')
    torch.set_num_threads(1);torch.cuda.set_device(0)
    output=root/'instability_analysis';output.mkdir(exist_ok=True);copies=output/'checkpoint_copies';copies.mkdir(exist_ok=True)
    cfg=json.loads((root/'configs/forward_shared.json').read_text());payload=pickle.loads(Path(cfg['data']).read_bytes());schema=payload['schema']
    validation=[s for s in payload['sequences'] if s['split']=='validation'][:8]
    contexts=ICUContexts(validation,horizon=24);batch=device_batch(collate([contexts[i] for i in range(len(contexts))]))
    results={'physical_gpu':args.gpu,'device':'cuda:0','environment':Path(sys.prefix).name,'patients':[s['subject_id'] for s in validation],
             'selection':'first 8 validation patients; no checkpoint selection by these probes','models':[]}
    def summary(x):
        x=x.detach().float();return {'mean':float(x.mean()),'median':float(x.median()),'min':float(x.min()),'max':float(x.max())}
    for variant in ['forward_shared','reverse_shared']:
        for kind in ['best','last']:
            original=root/variant/'seed_0'/f'{kind}.pt';copy=copies/f'{variant}_{kind}.pt';shutil.copy2(original,copy)
            saved=torch.load(copy,map_location='cuda',weights_only=False);model=make_model(schema,saved['config']).cuda().eval();model.load_state_dict(saved['state_dict'])
            for parameter in model.parameters():parameter.requires_grad_(True)
            with torch.no_grad():out=model(batch,sample=False)
            states=out['jacobian_states'];cutoff=states[torch.arange(len(states),device='cuda'),batch['cutoff']-1]
            visited=states[batch['valid']]
            with torch.no_grad():
                derivative=1-model.forward_ode[1](model.forward_ode[0](visited)).square()
                J=(model.forward_ode[2].weight[None]*derivative[:,None])@model.forward_ode[0].weight
                eigen=torch.linalg.eigvals(J).real;H=(J+J.transpose(-1,-2))/2;he=torch.linalg.eigvalsh(H)
            row={'variant':variant,'checkpoint':kind,'epoch':saved['epoch'],'checkpoint_sha256':hashlib.sha256(copy.read_bytes()).hexdigest(),
                 'field_jacobian_squared':summary(J.square().sum((-2,-1))),'minimum_real_eigenvalue':summary(eigen.min(-1).values),
                 'maximum_real_eigenvalue':summary(eigen.max(-1).values),'forward_local_growth_bound':summary(he[:,-1]),
                 'backward_local_growth_bound':summary(-he[:,0]),'flow_sensitivity':[]}
            model.rtol=1e-6;model.atol=1e-8
            for sample in range(2):
                z=cutoff[sample].detach().requires_grad_(True)
                for delta in [-24.,-12.,12.,24.]:
                    dt=torch.tensor([delta],device='cuda')
                    def flow(x):return model.solve(model.forward_ode,x[None],dt)[0]
                    flow_jac=torch.autograd.functional.jacobian(flow,z,create_graph=False,vectorize=True)
                    singular=torch.linalg.svdvals(flow_jac)
                    row['flow_sensitivity'].append({'patient':sample,'delta_hours':delta,'max_singular_value':float(singular[0]),'frobenius_norm':float(flow_jac.norm())})
            model.rtol=saved['config']['rtol'];model.atol=saved['config']['atol']
            out=model(batch,sample=False);errors=feature_errors(out['prediction'],batch['values'],batch['mask'],schema)
            idx=torch.arange(batch['values'].shape[1],device='cuda')[None];history=idx<batch['cutoff'][:,None]
            denominator=batch['mask'].sum((0,1)).clamp_min(1)
            hist=(errors*history[...,None]).sum((0,1)).div(denominator).sum()
            future=(errors*(~history)[...,None]).sum((0,1)).div(denominator).sum()
            pe,pm=prior_errors(out,batch,schema);prior=.01*(pe.sum((0,1))/pm.sum((0,1)).clamp_min(1)).sum()
            jac,_,_=jacobian_regularization(model,out,batch);regularizer=.01*jac+.02*model.ode_regularization()
            parameters=list(model.forward_ode.parameters())
            def gradient(value):
                return torch.cat([p.reshape(-1) for p in torch.autograd.grad(value,parameters,retain_graph=True)])
            gh=gradient(hist);gf=gradient(future);gp=gradient(prior);gr=gradient(regularizer)
            def cosine(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-12))
            row['shared_field_gradients']={'history_norm':float(gh.norm()),'future_norm':float(gf.norm()),'prior_norm':float(gp.norm()),'regularizer_norm':float(gr.norm()),
                                          'history_future_cosine':cosine(gh,gf),'history_prior_cosine':cosine(gh,gp)}
            results['models'].append(row)
            (output/'cuda_probes.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps({'variant':variant,'checkpoint':kind,'epoch':saved['epoch'],'jacobian_mean':row['field_jacobian_squared']['mean'],
                              'flow':row['flow_sensitivity'],'gradients':row['shared_field_gradients']}),flush=True)
            del model,saved,out,states,visited,J,H,flow_jac,gh,gf,gp,gr
            torch.cuda.empty_cache()


if __name__=='__main__':main()
