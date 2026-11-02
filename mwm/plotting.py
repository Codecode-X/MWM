"""Plot recorded training statistics; no model/checkpoint execution."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mimic_experiment.prepare import atomic


def save(fig,folder,name):
    fig.tight_layout()
    for ext in ['png','svg','pdf']:
        target=folder/f'{name}.{ext}';temporary=folder/f'{name}.partial.{ext}'
        fig.savefig(temporary,dpi=165);temporary.replace(target)
    plt.close(fig)


def training_curves(folder,rows,previous_signature):
    signature=[(r['id'],r['epochs'],r['status']) for r in rows]
    signature=[list(x) for x in signature]
    if not any(r['history'] for r in rows):return []
    if signature==previous_signature:return ['training_curves']+(['batch_comparison_curves'] if any(r['config']['batch_size']==32 for r in rows) else [])
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for row in rows:
        history=row['history']
        if not history:continue
        cfg=row['config'];label=f"{cfg['variant']} | batch {cfg['batch_size']}"
        epoch=[h['epoch'] for h in history]
        data=[h['training_components']['unweighted_mse']+h['training_components']['unweighted_cce'] for h in history]
        axes[0,0].plot(epoch,data,label=label)
        axes[0,1].plot(epoch,[h['validation']['overall']['reconstruction_error'] for h in history],label=label)
        axes[1,0].plot(epoch,[h['validation']['forecast']['reconstruction_error'] for h in history],label=label)
        axes[1,1].plot([h['optimizer_steps'] for h in history],[h['validation']['forecast']['reconstruction_error'] for h in history],label=label)
    for ax,title,xlabel in zip(axes.flat,['Training unweighted data error','Validation causal overall error','Validation forecast error','Validation forecast vs optimizer steps'],['Epoch','Epoch','Epoch','Optimizer steps']):
        ax.set(title=title,xlabel=xlabel,ylabel='Sum of per-field MSE + CE');ax.grid(alpha=.2);ax.legend(fontsize=8)
    save(fig,folder,'training_curves');names=['training_curves']
    if any(r['config']['batch_size']==32 for r in rows):
        fig,axes=plt.subplots(2,3,figsize=(17,8.5))
        for line,variant in enumerate(['A','B']):
            for row in rows:
                if row['config']['variant']!=variant or not row['history']:continue
                history=row['history'];batch=row['config']['batch_size'];label=f'batch {batch}';epoch=[h['epoch'] for h in history]
                axes[line,0].plot(epoch,[h['training_loss'] for h in history],label=label)
                axes[line,1].plot(epoch,[h['validation']['overall']['reconstruction_error'] for h in history],label=label)
                axes[line,2].plot([h['optimizer_steps'] for h in history],[h['validation']['forecast']['reconstruction_error'] for h in history],label=label)
            for j,title in enumerate(['Training objective','Validation causal overall','Validation forecast vs updates']):
                axes[line,j].set(title=f'MWM {variant}: {title}',xlabel='Optimizer steps' if j==2 else 'Epoch',ylabel='Objective' if j==0 else 'MSE + CE')
                axes[line,j].grid(alpha=.2)
                if axes[line,j].lines:axes[line,j].legend()
        save(fig,folder,'batch_comparison_curves');names.append('batch_comparison_curves')
    atomic(folder/'curve_signature.json',signature)
    return names
