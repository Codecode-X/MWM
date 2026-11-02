"""CPU statistical summaries and plots of the already recorded training metrics."""
from datetime import datetime,timezone
import csv
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(__file__).resolve().parent/'results/direction_full_batch_cuda';out=root/'instability_analysis';out.mkdir(exist_ok=True)
    histories={}
    for key,limit in [('forward_shared',81),('reverse_shared',111)]:
        histories[key]=json.loads((root/key/'seed_0/history.json').read_text())[:limit]
    (out/'history_snapshot.json').write_text(json.dumps({'created_at':datetime.now(timezone.utc).isoformat(),'histories':histories},indent=2)+'\n')
    records=[]
    for key,history in histories.items():
        for r in history:
            v=r['validation'];c=r['training_components']
            records.append({'model':key,'epoch':r['epoch'],'training_loss':r['training_loss'],'preclip_gradient_norm':c['gradient_norm'],
                            **{f'validation_{region}':v[region]['reconstruction_error'] for region in ['interpolation','forecast','overall','prior']},
                            'validation_jacobian_squared':v['dynamics']['jacobian_frobenius_squared'],
                            'validation_history_cce':v['interpolation']['sum_feature_cce'],'training_weighted_jacobian':c['weighted_jacobian'],
                            'training_weighted_prior':c['weighted_prior'],'learning_rate':r['lr']})
    with (out/'epoch_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    specifications=[('validation_overall','Validation overall',False),('validation_interpolation','Validation history',False),
                    ('validation_forecast','Validation future',False),('preclip_gradient_norm','Training gradient norm (before clipping)',True),
                    ('validation_jacobian_squared','Validation mean ||J_f||_F^2',True),('validation_history_cce','Validation history: categorical cross entropy',False)]
    for ax,(field,title,log) in zip(axes.ravel(),specifications):
        for key,color,label in [('forward_shared','#d95f02','Forward'),('reverse_shared','#1b9e77','Reverse')]:
            rows=[r for r in records if r['model']==key];ax.plot([r['epoch'] for r in rows],[r[field] for r in rows],color=color,label=label,linewidth=1.4)
        for a,b in [(46,48),(63,65)]:ax.axvspan(a,b,color='#e41a1c',alpha=.12)
        ax.axvline(30,color='gray',linestyle=':',alpha=.6);ax.axvline(41,color='#d95f02',linestyle='--',alpha=.45)
        ax.set(title=title,xlabel='Epoch',ylabel='Recorded value');ax.grid(alpha=.15)
        if log:ax.set_yscale('log')
        ax.legend(fontsize=8)
    fig.suptitle('MIMIC full-batch128: instability begins at epoch 46; forward best checkpoint is epoch 41',fontsize=12)
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(out/f'instability_evidence.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps({'statistics_only':True,'histories':{k:len(v) for k,v in histories.items()},'output':str(out)}))


if __name__=='__main__':main()
