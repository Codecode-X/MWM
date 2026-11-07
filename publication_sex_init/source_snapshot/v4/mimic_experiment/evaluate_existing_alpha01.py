"""CUDA evaluation of the existing 237-epoch forward run; no training."""
import argparse
import json
from pathlib import Path
import torch
from direction_gradient_common import initialize, digest
from damped_update_model import make_damped_model
from mimic_experiment.evaluation import evaluate
from mimic_experiment.prepare import atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,required=True);args=p.parse_args()
    root=Path(args.out);manifest,cfg,payload,modes=initialize(root,args.gpu)
    forward=root/'update_alpha_01';dest=root/'direction_alpha01/mimic_forward_existing';dest.mkdir(exist_ok=True)
    cfg={**cfg,**json.loads((forward/'config.json').read_text())}
    checkpoint=torch.load(forward/'best.pt',map_location='cuda',weights_only=False)
    history=json.loads((forward/'history.json').read_text())
    model=make_damped_model(payload['schema'],cfg,alpha=.1);model.load_state_dict(checkpoint['state_dict'])
    splits={s:[x for x in payload['sequences'] if x['split']==s] for s in ['validation','test']}
    def ev(split,horizon=None):
        kwargs={} if horizon is None else {'horizon':horizon}
        return evaluate(model,splits[split],payload['schema'],cfg,payload['metadata']['raw_train_std'],modes,**kwargs)
    validation=ev('validation');expected=next(h for h in history if h['epoch']==checkpoint['epoch'])['validation']['overall']['reconstruction_error']
    difference=abs(validation['overall']['reconstruction_error']-expected);assert difference<1e-6
    result={'status':'evaluated_existing_checkpoint','dataset':'mimic','alpha':.1,'epochs_run':len(history),'best_epoch':checkpoint['epoch'],
            'validation':validation,'test':ev('test'),'test_early_contexts':{str(h):ev('test',h) for h in [6,12]},
            'checkpoint_reload_difference':difference,'checkpoint':str(forward/'best.pt'),'checkpoint_sha256':digest(forward/'best.pt'),
            'physical_gpu':args.gpu,'parameter_device':str(next(model.parameters()).device),'optimizer_steps':0,
            'training_restarted':False}
    atomic(dest/'metrics.json',result)


if __name__=='__main__':main()
