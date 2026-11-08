"""CUDA velocity-aligned SVD of the saved high-dimensional benchmark fields."""
import argparse
import json
import os
from pathlib import Path
import pickle

import numpy as np
import torch
from .state_space_benchmark import OUT,RUNS,ROOT,sha,save,Adapter,history_path
from .state_space import local_field,COLORS,LABELS


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',required=True,choices=[r[0] for r in RUNS]);parser.add_argument('--gpu',required=True,choices=['0','1','2','3']);parser.add_argument('--output',default=str(OUT))
    args=parser.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(1);folder=Path(args.output)/args.model
    data=np.load(folder/'states.npz');meta=json.loads((folder/'metadata.json').read_text());old=json.loads((folder/'statistics.json').read_text())
    # Near-constant fields can have condition numbers beyond float32 precision.
    # Use accurate CUDA float64 SVD and energy accumulation for this projection.
    z=torch.tensor(data['states'],device='cuda',dtype=torch.float64);v=torch.tensor(data['velocities'],device='cuda',dtype=torch.float64)
    train=torch.tensor([r['split']=='train' for r in meta],device='cuda')
    labels=torch.tensor([-1 if r['diagnosis'] is None else int(r['diagnosis']) for r in meta],device='cuda')
    _,singular,vh=torch.linalg.svd(v[train],full_matrices=False,driver='gesvd');basis=vh[:2].T;mean=z[train].mean(0)
    xy=(z-mean)@basis;projected=v@basis
    field=local_field(xy[train].float(),projected[train].float(),labels[train])
    weight=singular.square()/singular.square().sum().clamp_min(1e-30);prob=weight[weight>1e-12]
    retained=float(projected.square().sum()/v.square().sum().clamp_min(1e-30))
    assert 0<=retained<=1+1e-10
    orthogonality=float((basis.T@basis-torch.eye(2,device='cuda',dtype=torch.float64)).abs().max())
    assert orthogonality<1e-10
    stats={'name':args.model,'checkpoint_sha256':old['checkpoint_sha256'],'source_states_sha256':sha(folder/'states.npz'),
           'physical_gpu':args.gpu,'device':str(basis.device),'method':'uncentered SVD of training velocities; includes mean flow',
           'velocity_energy_retained_2d':min(1.,retained),'projection_dtype':'CUDA float64; gesvd',
           'basis_orthogonality_max_error':orthogonality,
           'state_variance_retained_2d':float(xy[train].square().sum()/(z[train]-mean).square().sum().clamp_min(1e-30)),
           'train_mean_velocity_energy_fraction':float(int(train.sum())*v[train].mean(0).square().sum()/v[train].square().sum().clamp_min(1e-30)),
           'effective_uncentered_velocity_rank':float((-prob*prob.log()).sum().exp()),
           'diagnostics_source_sha256':sha(__file__),'model_training_performed':False,
           'neighborhood_dtype':'CUDA float32; projections and energy use float64'}
    arrays={'coordinates':xy.cpu().numpy(),'projected_velocities':projected.cpu().numpy(),'pca_mean':mean.cpu().numpy(),'velocity_basis':basis.cpu().numpy(),
            **{'field_'+key:value.cpu().numpy() if torch.is_tensor(value) else value for key,value in field.items()}}
    np.savez_compressed(folder/'velocity_plane.npz',**arrays)
    adapter=Adapter(args.model)
    assert sha(adapter.checkpoint)==old['checkpoint_sha256']
    dataset=Path(adapter.cfg['data']);dataset=dataset if dataset.is_absolute() else ROOT/dataset
    payload=pickle.loads(dataset.read_bytes());seqs={str(s['subject_id']):s for s in payload['sequences'] if s['split']=='test'}
    cases=json.loads((ROOT/'paper_odeae/results/shared_matched_state_space/patient_paths.json').read_text());paths=[]
    for case in cases:
        path=history_path(adapter,seqs[case['subject_id']],mean,basis);path['label']=case['label'];paths.append(path)
    (folder/'velocity_plane_paths.json').write_text(json.dumps(paths,indent=2)+'\n')
    stats['case_model_device']=str(next(adapter.model.parameters()).device)
    (folder/'velocity_plane.json').write_text(json.dumps(stats,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5.5));test=np.array([r['split']=='test' for r in meta]);dx=labels.cpu().numpy();points=arrays['coordinates']
    for k,label in enumerate(LABELS):
        m=test&(dx==k);axes[0].scatter(*points[m].T,c=COLORS[k],s=7,alpha=.5,label=label,rasterized=True)
    axes[0].legend(fontsize=8);axes[0].set_title(f"{old['title']}: test states\n{stats['state_variance_retained_2d']*100:.2f}% state variance")
    supported=arrays['field_supported'];grid=arrays['field_grid'][supported];vel=arrays['field_velocity'][supported]
    axes[1].scatter(*points[test].T,c='#bbb',s=2,alpha=.15,rasterized=True)
    if len(grid):
        q=axes[1].quiver(*grid.T,*vel.T,np.linalg.norm(vel,axis=-1),cmap='plasma',angles='xy',scale_units='xy',scale=None,width=.003)
        fig.colorbar(q,ax=axes[1],label='Own projected units / year')
    axes[1].set_title(f"Velocity-aligned field\n{stats['velocity_energy_retained_2d']*100:.2f}% velocity energy")
    for ax in axes:ax.set(xlabel='Training velocity axis 1',ylabel='Training velocity axis 2')
    save(fig,folder,'velocity_aligned')
    print(json.dumps(stats),flush=True)


if __name__=='__main__':main()
