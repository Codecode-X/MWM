"""Direct unsmoothed shared_matched velocities at every recorded state."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from paper_odeae.models import paper_model_from_config
from .state_space import ROOT


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',default='1',choices=['0','1','2','3'])
    parser.add_argument('--output',default=str(ROOT/'paper_odeae/results/shared_matched_state_space'))
    args=parser.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(1);out=Path(args.output)
    info=json.loads((out/'statistics.json').read_text());metadata=json.loads((out/'metadata.json').read_text())
    checkpoint=Path(info['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==info['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
    model=paper_model_from_config(saved['schema'],saved['config']).cuda().eval()
    model.load_state_dict(saved['state_dict']);assert model.backward_ode is model.forward_ode
    a=np.load(out/'states.npz');z=torch.tensor(a['states'],device='cuda')
    basis=torch.tensor(a['pca_basis'],device='cuda');mean=torch.tensor(a['pca_mean'],device='cuda')
    velocity=model.forward_ode(z);projected=velocity@basis;coordinates=(z-mean)@basis
    torch.testing.assert_close(velocity,torch.tensor(a['velocities'],device='cuda'),rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(projected,torch.tensor(a['projected_velocities'],device='cuda'),rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(coordinates,torch.tensor(a['coordinates'],device='cuda'),rtol=1e-5,atol=1e-6)
    verification={'checkpoint_sha256':info['checkpoint_sha256'],'physical_gpu':args.gpu,'gpu_name':torch.cuda.get_device_name(),
                  'model_device':str(next(model.parameters()).device),'state_device':str(z.device),'velocity_device':str(velocity.device),
                  'projection_device':str(projected.device),'states':len(z),'smoothing':False,'subsampling':False,
                  'max_velocity_difference_from_saved':float((velocity-torch.tensor(a['velocities'],device='cuda')).abs().max()),
                  'counts_by_split':{k:sum(r['split']==k for r in metadata) for k in ['train','validation','test']},
                  'method':'arrows are P^T f(z_i) at every actual state z_i; no neighbor averaging or grid interpolation',
                  'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    xy=coordinates.cpu().numpy();v=projected.cpu().numpy();speed=np.linalg.norm(v,axis=1)
    np.savez_compressed(out/'velocity_field_raw.npz',coordinates=xy,velocities_6d=velocity.cpu().numpy(),
                        projected_velocities=v,projected_speed=speed)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    fig,axes=plt.subplots(1,3,figsize=(19,6.5),layout='constrained')
    norm=Normalize(vmin=float(speed.min()),vmax=float(speed.max()))
    low=xy.min(0);high=xy.max(0);pad=(high-low)*.05
    splits=np.array([r['split'] for r in metadata]);names={'train':'Training','validation':'Validation','test':'Test'}
    for ax,split in zip(axes,['train','validation','test']):
        selected=splits==split
        ax.scatter(*xy[selected].T,s=2,c='#999',alpha=.15,rasterized=True)
        q=ax.quiver(*xy[selected].T,*v[selected].T,speed[selected],cmap='plasma',norm=norm,
                    angles='xy',scale_units='xy',scale=1,width=.0015,alpha=.5 if split=='train' else .75,
                    headwidth=3.5,headlength=4.5,minlength=0)
        ax.quiverkey(q,.10,.04,.25,'0.25 PC units/year',labelpos='E',coordinates='axes',fontproperties={'size':8},color='#333')
        ax.set(title=f'{names[split]}: {selected.sum():,} states, all arrows',xlabel='State PC1',ylabel='State PC2',
               xlim=(low[0]-pad[0],high[0]+pad[0]),ylim=(low[1]-pad[1],high[1]+pad[1]),aspect='equal')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='plasma'),ax=axes,label='Actual projected speed (PC units/year)',shrink=.7)
    fig.suptitle('shared_matched: unsmoothed velocities at actual patient states',fontsize=15)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'velocity_field_raw.{ext}',dpi=200)
    plt.close(fig)
    (out/'velocity_field_raw.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__':main()
