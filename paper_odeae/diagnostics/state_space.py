"""CUDA state extraction and PCA projection of the learned shared ODE field."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from paper_odeae.data import Prefixes, collate
from paper_odeae.models import encode_values, paper_model_from_config
from paper_odeae.train import device_batch

ROOT = Path(__file__).resolve().parents[2]
COLORS = ['#238b45', '#e6a21a', '#c63d47']
LABELS = ['CN', 'MCI', 'Dementia']


@torch.no_grad()
def encode(model, batch):
    """Same encoder as PaperModel.forward; omit unused trajectory decoding."""
    ages, cutoff = batch['ages'], batch['cutoff']
    idx = torch.arange(len(cutoff), device='cuda')
    h = model.init(ages[idx, cutoff-1, None]/100.)
    x = encode_values(batch['values'], batch['context_mask'], model.schema)
    previous = ages[:, 0]
    for i in range(batch['max_context']):
        index = torch.minimum(torch.full_like(cutoff, i), cutoff-1)
        alive = i < cutoff
        age = ages[idx, index]
        delta = torch.where(alive, age-previous, torch.zeros_like(age))
        evolved = model.solve(model.backward_ode, h, delta)
        proposed = model.update(torch.cat([evolved, x[idx, index]], -1))
        h = torch.where(alive[:, None], proposed, h)
        previous = torch.where(alive, age, previous)
    return h


@torch.no_grad()
def extract(model, payload):
    states, velocities, metadata = [], [], []
    schema = model.schema
    fields = {f['name']: j for j, f in enumerate(schema)}
    offsets = payload['metadata']['offset']; scales = payload['metadata']['scale']
    for split in ['train', 'validation', 'test']:
        seqs = [s for s in payload['sequences'] if s['split'] == split]
        count = 0
        for cpu in DataLoader(Prefixes(seqs), batch_size=64, shuffle=False, collate_fn=collate):
            b = device_batch(cpu); z = encode(model, b)
            if not states:
                torch.testing.assert_close(z, model(b)['mu'], rtol=0, atol=0)
            states.append(z); velocities.append(model.forward_ode(z))
            vals = cpu['values'].numpy(); mask = cpu['mask'].numpy()
            for i, cutoff in enumerate(cpu['cutoff'].tolist()):
                j = cutoff-1
                row = {'split': split, 'subject_id': cpu['subject_ids'][i], 'prefix': cutoff,
                       'age': float(cpu['ages'][i, j])}
                for name in ['diagnosis', 'mmse', 'cdrsb']:
                    k = fields[name]
                    row[name] = float(vals[i, j, k]*scales[k]+offsets[k]) if mask[i, j, k] else None
                metadata.append(row)
            count += len(z)
        print(f'Extracted {split}: {len(seqs)} patients, {count} prefix endpoint states', flush=True)
    return torch.cat(states), torch.cat(velocities), metadata


@torch.no_grad()
def local_field(train_y, train_v, train_labels):
    low = torch.quantile(train_y, .005, dim=0)
    high = torch.quantile(train_y, .995, dim=0)
    pad = (high-low)*.05
    gx = torch.linspace(float(low[0]-pad[0]), float(high[0]+pad[0]), 32, device='cuda')
    gy = torch.linspace(float(low[1]-pad[1]), float(high[1]+pad[1]), 28, device='cuda')
    yy, xx = torch.meshgrid(gy, gx, indexing='ij'); grid = torch.stack([xx.flatten(), yy.flatten()], -1)
    # Density threshold uses training points only. No off-manifold inverse PCA.
    sample = train_y[::max(1, len(train_y)//1000)]
    radii = []
    for part in sample.split(128):
        radii.append(torch.cdist(part, train_y).topk(21, largest=False).values[:, -1])
    radius = torch.quantile(torch.cat(radii), .95)
    vs, probs, support, coherence = [], [], [], []
    for part in grid.split(128):
        dist, indices = torch.cdist(part, train_y).topk(30, largest=False)
        w = torch.exp(-dist.square() / (2*dist[:, -1:].square().clamp_min(1e-12)))
        w /= w.sum(1, keepdim=True)
        v = (train_v[indices]*w[..., None]).sum(1)
        vs.append(v)
        lab = train_labels[indices]; valid = lab >= 0
        p = torch.stack([(w*(lab == k)).sum(1) for k in range(3)], -1)
        p /= (w*valid).sum(1, keepdim=True).clamp_min(1e-9)
        probs.append(p)
        support.append((dist[:, 0] <= radius) & (dist[:, -1] <= radius*3))
        coherence.append(v.norm(dim=-1)/(train_v[indices].norm(dim=-1)*w).sum(1).clamp_min(1e-12))
    return {'grid': grid, 'velocity': torch.cat(vs), 'class_probability': torch.cat(probs),
            'supported': torch.cat(support), 'direction_coherence': torch.cat(coherence),
            'radius': float(radius), 'shape': (len(gy), len(gx))}


@torch.no_grad()
def examples(model, payload, mean, basis):
    result = []
    dx = next(i for i,f in enumerate(model.schema) if f['name']=='diagnosis')
    for disease in range(3):
        candidates = [s for s in payload['sequences'] if s['split']=='test' and len(s['ages'])>=4
                      and s['mask'][-1, dx] and int(s['values'][-1, dx])==disease]
        # Deterministic selection, not selected for attractive trajectories.
        seq = sorted(candidates, key=lambda s:str(s['subject_id']))[0]
        b = device_batch(collate([(seq, len(seq['ages']))])); ages=b['ages'][0]
        x = encode_values(b['values'], b['context_mask'], model.schema)[0]
        state = model.init(ages[-1:].reshape(1,1)/100.)
        ode_segments, jumps, post = [], [], []
        for i in range(len(ages)):
            old = state
            previous = ages[i-1] if i else ages[i]
            segment = [state[0]]
            for _ in range(12):
                state = model.solve(model.forward_ode, state, ((ages[i]-previous)/12).reshape(1))
                segment.append(state[0])
            before = state
            state = model.update(torch.cat([state, x[i:i+1]], -1))
            ode_segments.append(torch.stack(segment)); jumps.append(torch.cat([before, state]))
            post.append(state[0])
        torch.testing.assert_close(state, encode(model,b), rtol=2e-3, atol=2e-4)
        future=[state[0]]
        for _ in range(60):
            state=model.solve(model.forward_ode,state,torch.full((1,),5/60,device='cuda'))
            future.append(state[0])
        def project(z): return ((z-mean)@basis).cpu().numpy().tolist()
        result.append({'label':LABELS[disease], 'subject_id':str(seq['subject_id']), 'ages':ages.cpu().tolist(),
                       'ode_segments':[project(s) for s in ode_segments], 'updates':[project(s) for s in jumps],
                       'post_update':project(torch.stack(post)), 'forecast_5y':project(torch.stack(future)),
                       'update_norms_6d':[float((s[1]-s[0]).norm()) for s in jumps],
                       'ode_displacements_6d':[float((s[-1]-s[0]).norm()) for s in ode_segments]})
    return result


def plots(out, y, v, meta, field, paths, variance,model_label='Shared ODE'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    def save(fig, name):
        fig.tight_layout()
        for ext in ['png','svg','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    labels=np.array([-1 if r['diagnosis'] is None else int(r['diagnosis']) for r in meta])
    test=np.array([r['split']=='test' for r in meta])
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for k in range(3):
        mask=test&(labels==k);axes[0].scatter(*y[mask].T,s=8,alpha=.45,c=COLORS[k],label=f'{LABELS[k]} (n={mask.sum()})',rasterized=True)
    axes[0].legend(fontsize=8);axes[0].set_title('Held-out test states: observed diagnosis')
    for ax,key,title in [(axes[1],'mmse','Observed MMSE'),(axes[2],'age','Age (years)')]:
        color=np.array([np.nan if r[key] is None else r[key] for r in meta]);mask=test&np.isfinite(color)
        sc=ax.scatter(*y[mask].T,c=color[mask],cmap='viridis',s=8,alpha=.65,rasterized=True)
        fig.colorbar(sc,ax=ax);ax.set_title(title)
    for ax in axes:ax.set(xlabel=f'PC1 ({variance[0]*100:.2f}% state variance)',ylabel=f'PC2 ({variance[1]*100:.2f}%)')
    save(fig,'state_space')
    grid=field['grid'];supported=field['supported'];fv=field['velocity'];probs=field['class_probability']
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    ax=axes[0]
    ax.scatter(*y[test].T,s=3,c='#bbbbbb',alpha=.2,rasterized=True)
    lengths=np.linalg.norm(fv[supported],axis=-1)
    q=ax.quiver(*grid[supported].T,*fv[supported].T,lengths,cmap='plasma',angles='xy',scale_units='xy',scale=None,width=.003)
    fig.colorbar(q,ax=ax,label='Projected ODE speed (PC units/year)')
    ax.set_title(f'{model_label}: locally averaged projected velocity\nArrow lengths use a common display scale')
    ax=axes[1]
    region_colors=probs[supported]@np.array([to_rgb(c) for c in COLORS])
    ax.scatter(*grid[supported].T,c=region_colors.clip(0,1),marker='s',s=85,alpha=.8)
    for k in range(3):ax.scatter([],[],c=COLORS[k],marker='s',label=LABELS[k])
    ax.legend(title='Nearby training diagnoses; mixtures blend',fontsize=8)
    ax.set_title('Disease composition near each location\nGreen=CN, yellow=MCI, red=dementia')
    for ax in axes:ax.set(xlabel='PC1',ylabel='PC2')
    save(fig,'velocity_field')
    if paths is None:return
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for ax,p in zip(axes,paths):
        ax.scatter(*y[test].T,s=2,c='#aaaaaa',alpha=.1,rasterized=True)
        for j,(seg,jump) in enumerate(zip(p['ode_segments'],p['updates'])):
            a=np.array(seg);b=np.array(jump)
            ax.plot(*a.T,c='#2171b5',lw=2,label='ODE evolution' if j==0 else None)
            ax.plot(*b.T,'--',c='#e87b20',lw=1.5,label='Observation update' if j==0 else None)
        post=np.array(p['post_update']);future=np.array(p['forecast_5y'])
        ax.scatter(*post.T,s=22,c='black',zorder=3,label='After each update')
        ax.plot(*future.T,c='#8b4ba8',lw=2,label='5-year ODE forecast')
        ax.scatter(*np.array(p['updates'][0])[0],marker='*',s=100,c='#e87b20',label='Age INIT',zorder=4)
        ax.set(title=f"{p['label']} example: {len(p['ages'])} visits",xlabel='PC1',ylabel='PC2');ax.legend(fontsize=7)
    save(fig,'patient_paths')


def interactive(out, y, v, meta, variance,model_label='shared_matched',shared=True):
    """Self-contained canvas viewer; visualization only, no browser inference."""
    points=[[round(float(a),5),round(float(b),5),r['split'],r['subject_id'],r['prefix'],
             r['diagnosis'],r['mmse'],round(r['age'],2),round(float(c),5),round(float(d),5)]
            for (a,b),(c,d),r in zip(y,v,meta)]
    template=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>shared_matched 状态空间</title>
<style>body{font:16px system-ui;max-width:1150px;margin:25px auto;color:#263238;background:#fafafa}canvas{width:100%;border:1px solid #bbb;background:white;touch-action:none}select,button,label{margin:8px}#tip{position:fixed;display:none;background:white;border:1px solid #777;padding:10px;pointer-events:none;font-size:13px;white-space:pre-line}small{display:block;margin:10px 0;line-height:1.7}</style>
<h1>shared_matched：患者状态与共享 ODE</h1><p>滚轮缩放，按住拖动。悬停查看检查信息。每个点是读取一段历史后的状态估计。</p>
<label>患者划分 <select id="split"><option value="test">测试集</option><option value="train">训练集</option><option value="validation">验证集</option><option value="all">全部</option></select></label>
<label>着色 <select id="color"><option value="diagnosis">诊断</option><option value="mmse">MMSE</option><option value="age">年龄</option></select></label>
<label><input type="checkbox" id="flow">显示各点的真实投影速度箭头</label><button id="reset">重置视图</button>
<canvas id="plot" width="1100" height="660"></canvas><div id="tip"></div><small id="legend"></small>
<small>坐标采用训练状态拟合的 PCA，保留约 __VAR__% 状态变化。箭头对应原六维向量场在该状态的二维投影，方向与长度使用共同显示比例。图中相近位置可能在其他维度上不同。诊断和 MMSE 也是输入字段。</small>
<p><a href="report.zh-CN.md">查看完整说明</a> · <a href="state_space.png">状态空间图</a> · <a href="velocity_field.png">局部平均速度场</a> · <a href="patient_paths.png">患者路径</a></p>
<script>const data=__DATA__;const canvas=document.getElementById('plot'),ctx=canvas.getContext('2d'),tip=document.getElementById('tip');let view,drag=null,shown=[];
const dxnames=['CN（认知正常）','MCI','痴呆'],colors=['#238b45','#e6a21a','#c63d47'];
function reset(){const xs=data.map(p=>p[0]),ys=data.map(p=>p[1]);let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys),dx=Math.max(x1-x0,.001)*.06,dy=Math.max(y1-y0,.001)*.06;view={x0:x0-dx,x1:x1+dx,y0:y0-dy,y1:y1+dy};draw()}
function screen(p){return [60+(p[0]-view.x0)/(view.x1-view.x0)*1000,610-(p[1]-view.y0)/(view.y1-view.y0)*560]}
function draw(){ctx.clearRect(0,0,1100,660);shown=[];const split=document.getElementById('split').value,mode=document.getElementById('color').value,flow=document.getElementById('flow').checked;
ctx.save();ctx.beginPath();ctx.rect(60,50,1000,560);ctx.clip();
for(const p of data){if(split!='all'&&p[2]!=split)continue;const q=screen(p);if(q[0]<60||q[0]>1060||q[1]<50||q[1]>610)continue;
let c;if(mode=='diagnosis')c=p[5]==null?'#aaa':colors[p[5]];else{let val=mode=='mmse'?p[6]:p[7];if(val==null)c='#aaa';else{let t=mode=='mmse'?val/30:(val-50)/50;t=Math.max(0,Math.min(1,t));c=`hsl(${260-210*t},70%,45%)`}}
ctx.globalAlpha=.55;ctx.fillStyle=c;ctx.beginPath();ctx.arc(q[0],q[1],2.4,0,Math.PI*2);ctx.fill();shown.push([q,p]);
if(flow){const r=screen([p[0]+p[8]*.7,p[1]+p[9]*.7]);ctx.strokeStyle='#375a7f';ctx.globalAlpha=.25;ctx.beginPath();ctx.moveTo(...q);ctx.lineTo(...r);let a=Math.atan2(r[1]-q[1],r[0]-q[0]);ctx.moveTo(r[0]-4*Math.cos(a-.5),r[1]-4*Math.sin(a-.5));ctx.lineTo(...r);ctx.lineTo(r[0]-4*Math.cos(a+.5),r[1]-4*Math.sin(a+.5));ctx.stroke()}}
ctx.restore();ctx.globalAlpha=1;ctx.fillStyle='#333';ctx.font='14px system-ui';ctx.strokeStyle='#333';ctx.strokeRect(60,50,1000,560);
for(let k=0;k<=5;k++){ctx.fillText((view.x0+(view.x1-view.x0)*k/5).toFixed(1),50+200*k,633);ctx.fillText((view.y0+(view.y1-view.y0)*k/5).toFixed(1),16,615-112*k)}ctx.fillText('PC1',540,655);ctx.fillText('PC2',10,30);
document.getElementById('legend').textContent=(mode=='diagnosis'?'绿色：CN；黄色：MCI；红色：痴呆；灰色：缺失。':mode=='mmse'?'颜色从紫到橙，对应 MMSE 从 0 到 30 分。':'颜色从紫到橙，对应年龄从 50 到 100 岁。')+' 当前显示 '+shown.length+' 个历史截断状态。'}
function coords(e){let r=canvas.getBoundingClientRect();return [(e.clientX-r.left)*1100/r.width,(e.clientY-r.top)*660/r.height]}
canvas.addEventListener('wheel',e=>{e.preventDefault();const q=coords(e),u=(q[0]-60)/1000,w=(610-q[1])/560,x=view.x0+u*(view.x1-view.x0),y=view.y0+w*(view.y1-view.y0),s=e.deltaY>0?1.15:1/1.15;view={x0:x+(view.x0-x)*s,x1:x+(view.x1-x)*s,y0:y+(view.y0-y)*s,y1:y+(view.y1-y)*s};draw()},{passive:false});
canvas.onpointerdown=e=>{drag={q:coords(e),v:{...view}};canvas.setPointerCapture(e.pointerId)};canvas.onpointerup=()=>drag=null;
canvas.onpointermove=e=>{const q=coords(e);if(drag){let dx=(q[0]-drag.q[0])/1000*(drag.v.x1-drag.v.x0),dy=(q[1]-drag.q[1])/560*(drag.v.y1-drag.v.y0);view={x0:drag.v.x0-dx,x1:drag.v.x1-dx,y0:drag.v.y0+dy,y1:drag.v.y1+dy};tip.style.display='none';draw();return}let best=null,dist=64;for(const [pos,p] of shown){let d=(pos[0]-q[0])**2+(pos[1]-q[1])**2;if(d<dist){dist=d;best=p}}if(!best){tip.style.display='none';return}tip.textContent=`患者 ${best[3]} | ${best[2]}\n历史检查数：${best[4]}\n年龄：${best[7]}\n诊断：${best[5]==null?'缺失':dxnames[best[5]]}\nMMSE：${best[6]==null?'缺失':best[6].toFixed(1)}`.replaceAll(String.fromCharCode(92)+"n",String.fromCharCode(10));tip.style.display='block';tip.style.left=(e.clientX+15)+'px';tip.style.top=(e.clientY+15)+'px'};
canvas.onpointerleave=()=>tip.style.display='none';for(const id of ['split','color','flow'])document.getElementById(id).onchange=draw;document.getElementById('reset').onclick=reset;reset();</script></html>'''
    html=template.replace('__DATA__',json.dumps(points,separators=(',',':'))).replace('__VAR__',f'{sum(variance[:2])*100:.1f}')
    html=html.replace('shared_matched',model_label)
    if not shared:
        html=html.replace('共享 ODE','生成 ODE').replace(' · <a href="patient_paths.png">患者路径</a>','')
    (out/'interactive.html').write_text(html)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',default='0',choices=['0','1','2','3'])
    parser.add_argument('--output',default=str(ROOT/'paper_odeae/results/shared_matched_state_space'))
    parser.add_argument('--checkpoint',default=str(ROOT/'paper_odeae/results/forward_age_scaling/shared_matched/seed_0/best.pt'))
    parser.add_argument('--model-label',default='shared_matched')
    args=parser.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(1);torch.manual_seed(0)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(args.checkpoint).resolve()
    saved=torch.load(checkpoint,map_location='cuda',weights_only=False);cfg=saved['config']
    model=paper_model_from_config(saved['schema'],cfg).cuda().eval();model.load_state_dict(saved['state_dict'])
    assert model.backward_ode is model.forward_ode
    payload=pickle.loads(Path(cfg['data']).read_bytes())
    z,f,meta=extract(model,payload)
    train=torch.tensor([r['split']=='train' for r in meta],device='cuda')
    mean=z[train].mean(0);_,s,vh=torch.linalg.svd(z[train]-mean,full_matrices=False)
    basis=vh[:2].T;variance=s.square()/s.square().sum()
    y=(z-mean)@basis;v=f@basis
    labels=torch.tensor([-1 if r['diagnosis'] is None else int(r['diagnosis']) for r in meta],device='cuda')
    field=local_field(y[train],v[train],labels[train]);paths=examples(model,payload,mean,basis)
    retained=float(v.square().sum()/f.square().sum())
    # All dimensionality reduction and nearest-neighbor model diagnostics stay on CUDA.
    stats={'model_label':args.model_label,'pca_variance_ratio':variance.cpu().tolist(),'state_variance_retained_2d':float(variance[:2].sum()),
           'velocity_energy_retained_2d':retained,'speed_6d_median':float(f.norm(dim=-1).median()),
           'speed_projected_median':float(v.norm(dim=-1).median()),
           'grid_supported_fraction':float(field['supported'].float().mean()),
           'local_direction_coherence_median':float(field['direction_coherence'][field['supported']].median()),
           'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
           'checkpoint_epoch':saved['epoch'],'dataset_sha256':hashlib.sha256(Path(cfg['data']).read_bytes()).hexdigest(),
           'physical_gpu':args.gpu,'gpu_name':torch.cuda.get_device_name(),'model_device':str(next(model.parameters()).device),
           'state_device':str(z.device),'pca_device':str(basis.device),'field_device':str(field['velocity'].device),
           'prefix_endpoint_states':len(z),'patients':len(payload['sequences']),
           'counts_by_split':{split:sum(r['split']==split for r in meta) for split in ['train','validation','test']},
           'pca_fit_split':'train only; visits weighted equally','trajectory_selection':'first eligible test subject ID per last-visit diagnosis; >=4 visits',
           'encoder_equivalence_check':'exact equality to original forward mu on first batch'}
    np.savez_compressed(out/'states.npz',states=z.cpu().numpy(),velocities=f.cpu().numpy(),coordinates=y.cpu().numpy(),
                        projected_velocities=v.cpu().numpy(),pca_mean=mean.cpu().numpy(),pca_basis=basis.cpu().numpy())
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    (out/'patient_paths.json').write_text(json.dumps(paths,indent=2))
    (out/'statistics.json').write_text(json.dumps(stats,indent=2))
    arrays={k:val.cpu().numpy() if torch.is_tensor(val) else val for k,val in field.items()}
    np.savez_compressed(out/'projected_field.npz',**arrays)
    plots(out,y.cpu().numpy(),v.cpu().numpy(),meta,arrays,paths,variance.cpu().numpy(),model_label=args.model_label)
    interactive(out,y.cpu().numpy(),v.cpu().numpy(),meta,variance.cpu().numpy(),model_label=args.model_label)
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':main()
