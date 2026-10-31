"""Verify completed CUDA diagnostics and draw the consolidated evidence figure."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil

import numpy as np
import torch

from .manifold_core import setup,read,atomic,sha

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(ROOT/'paper_odeae/results/latent2x_both_manifold'));args=parser.parse_args()
    setup();out=Path(args.output);manifest=read(out/'manifest.json');summary={};checks={}
    for dataset in ['adni','mimic']:
        for kind in ['geometry','dynamics','flow','clinical']:
            key=dataset+'_'+kind;assert (out/key/'complete.json').exists();summary[key]=read(out/key/'summary.json')
            provenance=summary[key]['provenance'];assert provenance['device']=='cuda:0' and provenance['environment']=='ode' and not provenance['training_performed']
            assert provenance['checkpoint_sha256']==manifest['checkpoints'][dataset]['sha256']==sha(manifest['checkpoints'][dataset]['path'])
        arrays=np.load(out/(dataset+'_geometry')/'geometry_arrays.npz')
        error=0.
        for name in ['pca_basis','test_local_basis']:
            basis=torch.tensor(arrays[name],device='cuda');ident=torch.eye(28,device='cuda',dtype=torch.float64)
            error=max(error,float((basis.transpose(-1,-2)@basis-ident).abs().max()))
        assert error<1e-10
        geometry=summary[dataset+'_geometry'];controls=geometry['controls']
        assert controls['linear8_in_28']['scales']['raw']['local_pca']['80']['d95']['median']==8
        assert controls['swiss_roll2_in_28']['scales']['raw']['local_pca']['80']['d95']['median']==2
        dynamics=summary[dataset+'_dynamics'];probes=dynamics['probes']
        for family,candidates in [('global_or_shrink',['baseline','mean_shrink_025','global_d6','global_d12','global_d20']),('local',['baseline']+[key for key in probes if key.startswith('local')])]:
            chosen=dynamics['validation_selected'][family]['variant']
            assert probes[chosen]['validation']['error']==min(probes[key]['validation']['error'] for key in candidates)
        assert dynamics['metric']['decoder_jacobian_check_max_difference']<2e-5
        checks[dataset]={'basis_orthogonality_max_error':error,'synthetic_flat8_and_swissroll2_calibration':'passed',
                         'validation_selection_recomputed':True,'decoder_autograd_check_max_error':dynamics['metric']['decoder_jacobian_check_max_difference']}
    for folder in ['source_snapshot','followup_source_snapshot']:
        for name,digest in read(out/folder/'source_hashes.json').items():assert sha(out/folder/name)==digest
    atomic(out/'final_verification.json',{'status':'passed','checked_at':datetime.now(timezone.utc).isoformat(),
           'checkpoints_unchanged':True,'all_eight_jobs_completed':True,'model_and_geometry_execution':'CUDA',
           'primary_and_followup_source_hashes_verified':True,'checks':checks,'verification_device':'cuda:0','physical_gpu':__import__('os').environ['CUDA_VISIBLE_DEVICES'],
           'verification_source_sha256':sha(__file__)})
    if not (out/'metrics_report.zh-CN.md').exists():shutil.copy2(out/'report.zh-CN.md',out/'metrics_report.zh-CN.md')
    plot(out,summary)
    print(checks)


def plot(out,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,10));x=np.arange(2);width=.34
    real=[summary[d+'_geometry']['primary']['scales']['raw']['mle_20']['median'] for d in ['adni','mimic']]
    null=[summary[d+'_geometry']['controls']['covariance_matched_gaussian28']['scales']['raw']['mle_20']['median'] for d in ['adni','mimic']]
    axes[0,0].bar(x-width/2,real,width,label='Observed states',color='#4477AA');axes[0,0].bar(x+width/2,null,width,label='Covariance-matched Gaussian',color='#CC6677')
    axes[0,0].set_xticks(x,['ADNI','MIMIC']);axes[0,0].set(ylabel='Median kNN dimension (k=20)',title='Local structure beyond global covariance');axes[0,0].legend(fontsize=8)
    for dataset,color in [('adni','#4477AA'),('mimic','#CC6677')]:
        ratio=np.array(summary[dataset+'_geometry']['primary']['pca_ratio']);axes[0,1].plot(np.arange(1,29),np.cumsum(ratio),label=dataset.upper(),color=color)
    axes[0,1].axhline(.95,c='gray',ls='--');axes[0,1].set(xlabel='Global principal components',ylabel='Cumulative variance',title='Global compression');axes[0,1].legend()
    static=[];moved=[]
    for d,h in [('adni','2.0'),('mimic','24.0')]:
        r=summary[d+'_flow']['rows'][h];static.append(100*r['static_reference']['12']['above_initial_training_q95']);moved.append(100*r['transported_reference']['12']['above_initial_training_q95'])
    axes[1,0].bar(x-width/2,static,width,label='Initial reference',color='#CC6677');axes[1,0].bar(x+width/2,moved,width,label='Co-evolved reference',color='#4477AA')
    axes[1,0].set_xticks(x,['ADNI +2 years','MIMIC +24 hours']);axes[1,0].set(ylabel='Above training support threshold (%)',title='The reference manifold moves with the ODE');axes[1,0].legend(fontsize=8)
    for i,dataset in enumerate(['adni','mimic']):
        selected=summary[dataset+'_dynamics']['validation_selected']['local'];v=selected['test_error_change'];lo,hi=selected['paired_patient_bootstrap95']
        axes[1,1].errorbar(v,i,xerr=[[max(v-lo,0)],[max(hi-v,0)]],fmt='o',capsize=4,color=['#4477AA','#CC6677'][i])
    axes[1,1].axvline(0,c='gray',ls='--');axes[1,1].set_yticks(x,['ADNI: weak local 20D','MIMIC: keep baseline']);axes[1,1].set(xlabel='Test error change; paired patient 95% CI',title='Validation-selected frozen projection')
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(out/f'key_findings.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
