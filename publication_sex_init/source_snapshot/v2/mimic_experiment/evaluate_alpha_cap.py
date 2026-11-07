"""CUDA test of an existing alpha checkpoint after the 150-epoch cap."""
import argparse
import json
from pathlib import Path
import torch
from direction_gradient_common import initialize, digest
from damped_update_model import make_damped_model
from mimic_experiment.evaluation import evaluate
from mimic_experiment.prepare import atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--tag',required=True);a=p.parse_args()
    root=Path(a.out);manifest,cfg,payload,modes=initialize(root,a.gpu);folder=root/('update_alpha_'+a.tag)
    cfg={**cfg,**json.loads((folder/'config.json').read_text()),'max_epochs':150}
    best=torch.load(folder/'best.pt',map_location='cuda',weights_only=False)
    assert best['epoch']<=150
    history=[r for r in json.loads((folder/'history.json').read_text()) if r['epoch']<=150]
    model=make_damped_model(payload['schema'],cfg,alpha=cfg['update_alpha']);model.load_state_dict(best['state_dict'])
    splits={s:[x for x in payload['sequences'] if x['split']==s] for s in ['validation','test']}
    def ev(split,horizon=None):
        kwargs={} if horizon is None else {'horizon':horizon}
        return evaluate(model,splits[split],payload['schema'],cfg,payload['metadata']['raw_train_std'],modes,**kwargs)
    validation=ev('validation');expected=next(r for r in history if r['epoch']==best['epoch'])['validation']['overall']['reconstruction_error']
    difference=abs(validation['overall']['reconstruction_error']-expected);assert difference<1e-6
    result={'status':'complete_at_epoch_cap','alpha':cfg['update_alpha'],'max_epochs':150,'epochs_run':len(history),'best_epoch':best['epoch'],
            'validation':validation,'test':ev('test'),'test_early_contexts':{str(h):ev('test',h) for h in [6,12]},
            'checkpoint_reload_difference':difference,'checkpoint_sha256':digest(folder/'best.pt'),'physical_gpu':a.gpu,
            'parameter_device':str(next(model.parameters()).device),'optimizer_steps':0}
    atomic(folder/'metrics_150.json',result)


if __name__=='__main__':main()
