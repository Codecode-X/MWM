"""Figures from already-recorded statistics; never executes a model."""
import hashlib
import json
from pathlib import Path


def stability_figure(out):
    source=Path(out)/'analysis/mimic/stability/metrics.json'
    if not source.exists():
        return
    signature=hashlib.sha256(source.read_bytes()).hexdigest()
    stamp=source.parent/'figure_source.json'
    if stamp.exists() and json.loads(stamp.read_text()).get('sha256')==signature:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=json.loads(source.read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for family,title in [('main','alpha=0.1'),('alpha1','alpha=1')]:
        selected={r['checkpoint_sha256']:r for r in rows if r['model']==family or r['model'].startswith(family+'_')}
        selected=sorted(selected.values(),key=lambda r:r['checkpoint_epoch'])
        x=[r['checkpoint_epoch'] for r in selected]
        sequences=[
            [r['history_gradient_norm']/max(r['future_gradient_norm'],1e-30) for r in selected],
            [r['gradient_cosine'] for r in selected],
            [r['recursive_state_gain']['p95'] for r in selected],
            [r['backward_gain']['median']/max(r['forward_gain']['median'],1e-30) for r in selected]]
        for ax,y in zip(axes.flat,sequences):
            ax.plot(x,y,'o-',label=title)
    titles=['History / future ODE gradient norm','History / future gradient cosine',
            'Recursive state gain: P95','Past / future flow gain: ratio of medians']
    for ax,title in zip(axes.flat,titles):
        ax.set(xlabel='Checkpoint epoch',title=title);ax.grid(alpha=.25);ax.legend()
    axes[0,1].axhline(0,c='gray',lw=.8)
    fig.tight_layout();fig.savefig(source.parent/'stability_diagnostics.png',dpi=180);plt.close(fig)
    stamp.write_text(json.dumps({'sha256':signature,'source':str(source),'model_execution':False,
                                'repeated_checkpoint_hashes_deduplicated':True},indent=2))
