"""Experiment B: distinct-patient 6D neighborhoods and future MMSE error."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np
import torch

from paper_odeae.models import paper_model_from_config
from paper_odeae.jacobian import squared_jacobian_norm
from paper_odeae.data import collate
from paper_odeae.train import device_batch
from .state_space import ROOT

SOURCE=ROOT/'paper_odeae/results/shared_matched_state_space'
OUT=ROOT/'paper_odeae/results/shared_matched_local_dynamics'
REGIONS=['CN core','CN/MCI boundary','MCI core','MCI/Dementia boundary','Dementia core','Other mixed','Insufficient labels']


@torch.no_grad()
def distinct_patient_knn(query,reference,query_pid,reference_pid,k,total_patients):
    if query.device.type!='cuda':raise RuntimeError('Neighborhood diagnostics require CUDA')
    indices=[];distances=[]
    for start in range(0,len(query),128):
        q=query[start:start+128];pid=query_pid[start:start+128]
        d=torch.cdist(q,reference,compute_mode='donot_use_mm_for_euclid_dist')
        d.masked_fill_(pid[:,None]==reference_pid[None,:],float('inf'))
        scatter=reference_pid[None,:].expand(len(q),-1)
        minimum=d.new_full((len(q),total_patients),float('inf'))
        minimum.scatter_reduce_(1,scatter,d,reduce='amin',include_self=True)
        values,patients=minimum.topk(k,largest=False,sorted=True)
        if not torch.isfinite(values).all():raise ValueError('too few distinct reference patients')
        local=torch.arange(len(reference),device=query.device)[None,:].expand(len(q),-1)
        candidates=torch.where(d==minimum.gather(1,scatter),local,len(reference))
        argmin=torch.full((len(q),total_patients),len(reference),device=query.device,dtype=torch.long)
        argmin.scatter_reduce_(1,scatter,candidates,reduce='amin',include_self=True)
        indices.append(argmin.gather(1,patients));distances.append(values)
    return torch.cat(indices),torch.cat(distances)


def pair_metrics(z,v,neighbors):
    dz=z[:,None]-z[neighbors];dv=v[:,None]-v[neighbors]
    distance=dz.norm(dim=-1);cosine=(v[:,None]*v[neighbors]).sum(-1)/(v.norm(dim=-1)[:,None]*v[neighbors].norm(dim=-1)+1e-12)
    sensitivity=dv.norm(dim=-1)/(distance+1e-8)
    return cosine,sensitivity,distance


def region_assignment(labels):
    known=labels>=0;coverage=known.float().mean(1)
    counts=torch.stack([(labels==k).sum(1) for k in range(3)],-1)
    p=counts/counts.sum(1,keepdim=True).clamp_min(1)
    result=torch.full((len(labels),),5,device=labels.device,dtype=torch.long)
    result[(p[:,0]+p[:,1]>=.8)&(p[:,0]>=.2)&(p[:,1]>=.2)]=1
    result[(p[:,1]+p[:,2]>=.8)&(p[:,1]>=.2)&(p[:,2]>=.2)]=3
    for k,code in [(0,0),(1,2),(2,4)]:result[p[:,k]>=.8]=code
    result[coverage<.8]=6
    return result,p,coverage


@torch.no_grad()
def random_patient_control(z,v,pid,reference):
    # Uniformly sample one visit per training patient, then select 20 distinct
    # reference patients per query. This is a nonlocal directional control.
    ids=pid[reference].cpu().numpy();ref=reference.cpu().numpy();rng=np.random.default_rng(0)
    unique=np.unique(ids);representatives=np.array([rng.choice(ref[ids==p]) for p in unique])
    people=torch.tensor(unique,device=z.device);states=torch.tensor(representatives,device=z.device)
    generator=torch.Generator(device=z.device).manual_seed(0);neighbors=[]
    for start in range(0,len(z),128):
        sample=torch.rand((min(128,len(z)-start),len(unique)),generator=generator,device=z.device)
        sample.masked_fill_(pid[start:start+128,None]==people[None,:],-1.)
        chosen=sample.topk(20,dim=1).indices;neighbors.append(states[chosen])
    neighbors=torch.cat(neighbors);C,D,_=pair_metrics(z,v,neighbors)
    return neighbors,C.mean(1),D.mean(1)


def write_csv(path,rows):
    if not rows:return
    with path.open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


@torch.no_grad()
def check_forecasts(model,sequences,z,queries,mmse_head,scale,offset):
    selected=[]
    for horizon in [12,24]:
        candidates=[r for r in queries if r['split']=='test' and r['horizon_months']==horizon]
        selected+=candidates[::max(1,len(candidates)//4)][:4]
    batch=device_batch(collate([(sequences[r['subject_id']],r['start_visit']) for r in selected]))
    original=model(batch)
    indices=torch.tensor([r['state_index'] for r in selected],device=z.device)
    torch.testing.assert_close(original['mu'],z[indices],rtol=1e-4,atol=1e-4)
    actual=torch.stack([original['prediction'][i,r['target_visit']-1,mmse_head]*scale+offset for i,r in enumerate(selected)])
    direct=torch.tensor([r['predicted_mmse'] for r in selected],device=z.device)
    difference=float((actual-direct).abs().max())
    if difference>5e-3:raise RuntimeError(f'horizon solve differs from original model: {difference}')
    changed={k:v.clone() if torch.is_tensor(v) else v for k,v in batch.items()}
    for i,cutoff in enumerate(batch['cutoff'].tolist()):
        for j,field in enumerate(model.schema):
            if field['kind']=='numerical':changed['values'][i,cutoff:,j]+=9
    torch.testing.assert_close(model(changed)['prediction'],original['prediction'],rtol=0,atol=0)
    return {'checked_origins':len(selected),'max_direct_vs_original_mmse_difference':difference,
            'max_independent_cached_state_difference':float((original['mu']-z[indices]).abs().max()),'future_value_isolation':'exact equality'}


def correlation(x,y):
    from scipy.stats import pearsonr,spearmanr
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<4 or x.std()<1e-12 or y.std()<1e-12:return {'pearson':None,'spearman':None}
    return {'pearson':float(pearsonr(x,y).statistic),'spearman':float(spearmanr(x,y).statistic)}


def partial_rank(x,y,covariates):
    from scipy.stats import rankdata
    x=rankdata(x);y=rankdata(y);c=np.asarray(covariates,float)
    c=np.stack([rankdata(c[:,j]) for j in range(c.shape[1])],1)
    c=(c-c.mean(0))/np.maximum(c.std(0),1e-10);design=np.column_stack([np.ones(len(c)),c])
    xr=x-design@np.linalg.lstsq(design,x,rcond=None)[0]
    yr=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
    if xr.std()<1e-10 or yr.std()<1e-10:return None
    return float(np.corrcoef(xr,yr)[0,1])


def summarize_forecasts(rows,out):
    from collections import defaultdict
    from threadpoolctl import threadpool_limits
    from scipy.stats import spearmanr
    result={};patient_rows=[];quartile_rows=[];region_rows=[]
    with threadpool_limits(limits=1):
        for horizon in [12,24]:
            selected=[r for r in rows if r['horizon_months']==horizon]
            grouped=defaultdict(list)
            for r in selected:grouped[(r['split'],r['subject_id'])].append(r)
            patients=[]
            numerical=['D10','D20','D40','J','C20','mmse_error','persistence_error','age','years','history_visits',
                       'last_mmse','last_mmse_missing','last_mmse_lag','sex','dx_cn','dx_mci','dx_dem','dx_missing']
            for (split,pid),values in grouped.items():
                row={'horizon_months':horizon,'split':split,'subject_id':pid,'origins':len(values)}
                for key in numerical:
                    a=[v[key] for v in values if v[key] is not None];row[key]=float(np.mean(a)) if a else None
                patients.append(row)
            patient_rows+=patients
            train=[r for r in patients if r['split']=='train'];test=[r for r in patients if r['split']=='test']
            x=np.array([r['D20'] for r in test]);y=np.array([r['mmse_error'] for r in test]);jac=np.array([r['J'] for r in test])
            cov_keys=['age','years','history_visits','last_mmse','last_mmse_missing','last_mmse_lag','sex','dx_cn','dx_mci','dx_dem','dx_missing']
            cov=np.array([[r[k] for k in cov_keys] for r in test])
            info={'test_patients':len(test),'test_origins':sum(r['origins'] for r in test),'mmse_mae_patient_weighted':float(y.mean()),
                  'mmse_mae_origin_weighted':float(np.mean([r['mmse_error'] for r in selected if r['split']=='test'])),
                  'persistence_mae_patient_weighted':float(np.mean([r['persistence_error'] for r in test if r['persistence_error'] is not None])),
                  'sensitivity_error':correlation(x,y),'jacobian_error':correlation(jac,y),
                  'adjusted_sensitivity_error':partial_rank(x,y,cov),'adjusted_jacobian_error':partial_rank(jac,y,cov),
                  'covariates':cov_keys,'knn_sensitivity':{str(k):correlation([r[f'D{k}'] for r in test],y) for k in [10,20,40]}}
            rng=np.random.default_rng(12+horizon);boot={k:[] for k in ['D','J','adjusted_D','adjusted_J']}
            for _ in range(1000):
                ids=rng.integers(len(test),size=len(test))
                boot['D'].append(float(spearmanr(x[ids],y[ids]).statistic));boot['J'].append(float(spearmanr(jac[ids],y[ids]).statistic))
                for key,xx in [('adjusted_D',x),('adjusted_J',jac)]:
                    value=partial_rank(xx[ids],y[ids],cov[ids])
                    if value is not None:boot[key].append(value)
            info['patient_bootstrap_95ci']={k:np.quantile(v,[.025,.975]).tolist() for k,v in boot.items()}
            cuts=np.quantile([r['D20'] for r in train],[.25,.5,.75]);info['training_patient_D_quartile_cuts']=cuts.tolist()
            for q in range(4):
                group=[r for r in test if np.searchsorted(cuts,r['D20'],side='right')==q]
                quartile_rows.append({'horizon_months':horizon,'sensitivity_quartile':q+1,'patients':len(group),
                    'mean_D':float(np.mean([r['D20'] for r in group])) if group else None,
                    'mmse_mae':float(np.mean([r['mmse_error'] for r in group])) if group else None})
            for region in REGIONS:
                group=[r for r in selected if r['split']=='test' and r['region']==region]
                region_rows.append({'horizon_months':horizon,'region':region,'origins':len(group),'patients':len({r['subject_id'] for r in group}),
                                    'mmse_mae':float(np.mean([r['mmse_error'] for r in group])) if group else None})
            result[str(horizon)]=info
    write_csv(out/'patient_forecast_summary.csv',patient_rows);write_csv(out/'sensitivity_quartiles.csv',quartile_rows);write_csv(out/'region_forecast_errors.csv',region_rows)
    return {'horizons':result,'sensitivity_quartiles':quartile_rows,'region_forecast_errors':region_rows}


def plots(out,xy,meta,regions,C,D,J,projection,forecast_rows,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,Normalize
    test=np.array([m['split']=='test' for m in meta]);train=np.array([m['split']=='train' for m in meta])
    colors=['#238b45','#7fc97f','#e6a21a','#ed9975','#c63d47','#888','#ddd']
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','svg','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    axes[0,0].scatter(*xy[test].T,c=regions[test],cmap=ListedColormap(colors),vmin=0,vmax=6,s=9,alpha=.6,rasterized=True)
    for i,name in enumerate(REGIONS):axes[0,0].scatter([],[],c=colors[i],label=name)
    axes[0,0].legend(fontsize=7);axes[0,0].set_title('Regions defined by training neighbors in full 6D')
    for ax,value,title in [(axes[0,1],C,'6D direction consistency C'),(axes[1,0],D,'6D local sensitivity D'),(axes[1,1],J,'Exact Jacobian Frobenius norm')]:
        lo,hi=np.quantile(value[train],[.01,.99]);sc=ax.scatter(*xy[test].T,c=value[test],cmap='viridis',norm=Normalize(lo,hi),s=9,alpha=.65,rasterized=True)
        fig.colorbar(sc,ax=ax,extend='both');ax.set_title(title)
    for ax in axes.flat:ax.set(xlabel='Original state PC1',ylabel='Original state PC2')
    save(fig,'local_dynamics_map')
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    metrics=[C[test],projection['C_2d_neighbors_full_velocity'][test],projection['C_2d_neighbors_projected_velocity'][test]]
    axes[0].boxplot(metrics,tick_labels=['6D NN / 6D v','2D NN / 6D v','2D NN / 2D v'],showfliers=False);axes[0].set(title='Direction consistency under projection',ylabel='Mean cosine similarity')
    axes[1].hist(projection['neighbor_patient_overlap'][test],bins=20,color='#4c78a8');axes[1].set(title='6D versus 2D neighbor patient overlap',xlabel='Fraction of common neighbors',ylabel='Test states')
    axes[2].scatter(C[test],projection['C_2d_neighbors_projected_velocity'][test],s=7,alpha=.3);axes[2].plot([-1,1],[-1,1],'k--',lw=1)
    axes[2].set(xlabel='Full-space C',ylabel='2D-neighbor / projected-velocity C',title='Projected versus full direction agreement')
    save(fig,'projection_consistency')
    patient=list(csv.DictReader((out/'patient_forecast_summary.csv').open()));fig,axes=plt.subplots(2,3,figsize=(17,10))
    for row,h in enumerate([12,24]):
        use=[r for r in patient if int(r['horizon_months'])==h and r['split']=='test']
        x=np.array([float(r['D20']) for r in use]);y=np.array([float(r['mmse_error']) for r in use]);j=np.array([float(r['J']) for r in use])
        for ax,values,name in [(axes[row,0],x,'Mean sensitivity D'),(axes[row,1],j,'Mean Jacobian norm')]:
            ax.scatter(values,y,s=14,alpha=.4);ax.set(xlabel=name,ylabel='Patient mean absolute MMSE error',title=f'{h}-month forecast')
        groups=[r for r in summary['sensitivity_quartiles'] if r['horizon_months']==h]
        axes[row,2].bar([r['sensitivity_quartile'] for r in groups],[r['mmse_mae'] if r['mmse_mae'] is not None else np.nan for r in groups],color='#4c78a8')
        axes[row,2].set(title=f'{h}-month error by sensitivity quartile',xlabel='Training-defined D quartile',ylabel='Patient mean absolute MMSE error')
    save(fig,'sensitivity_forecast_error')


def report(out,summary,provenance):
    def f(x,n=3):return '—' if x is None else f'{x:.{n}f}'
    lines=['# 实验 B：高维局部速度一致性与长期预测误差','',
      '## 1. 从六维邻域研究相似患者的演化','',
      f"本实验使用 {provenance.get('model_label','原始 shared_matched')}（22,545参数，第{provenance.get('checkpoint_epoch',56)}轮检查点）及其17,165个状态。每个状态对应一位患者某次历史截止点的估计。近邻距离直接在完整六维潜在空间中计算。",'',
      '主邻域由20位不同训练患者构成：每位患者仅保留距查询状态最近的一个状态，查询患者自身的所有随访排除。验证集与测试集的近邻均来自训练集。另计算k=10和40的结果。','',
      '方向一致性C是查询状态与各邻居原始六维速度的余弦相似度平均值；局部敏感性D是速度差范数除以状态距离后的平均值。Jacobian范数使用共享向量场的完整6×6精确导数。','',
      r'\[\bar C_i=\frac1k\sum_j\frac{f(z_i)^\top f(z_j)}{\|f(z_i)\|\|f(z_j)\|+\epsilon},\qquad D_i=\frac1k\sum_j\frac{\|f(z_i)-f(z_j)\|}{\|z_i-z_j\|+\epsilon}.\]','',
      '## 2. 五类核心区与交界区','',
      '区域由六维训练邻域中的诊断比例定义。某类占比至少80%为该类核心区；CN/MCI或MCI/痴呆两类合计至少80%、且各占至少20%为交界区，核心区优先。诊断有效邻居至少占80%。其余状态列为其他混合区或标签不足。','',
      '![六维局部动力学在原PCA坐标中的分布](local_dynamics_map.png)','',
      '图中的坐标只用于显示，区域和数值都在六维空间中计算。色条范围取训练状态数值的1%–99%分位。','',
      '| 测试区域 | 状态数 | 患者数 | 平均C | C第10分位 | 平均D | 平均Jacobian范数 | 反向邻居占比 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['regions']:
        lines.append(f"| {r['region']} | {r['states']} | {r['patients']} | {f(r['C_mean'])} | {f(r['C_p10'])} | {f(r['D_mean'])} | {f(r['J_mean'])} | {f(100*r['opposite_pair_fraction']) if r['opposite_pair_fraction'] is not None else '—'}% |")
    if 'low_consistency_states' in summary:
        low=summary['low_consistency_states']
        lines+=['',f"以C<0.9标记局部方向分歧，测试集中共有 {low['states']} 个状态，占 {low['fraction']*100:.2f}%。具体为："+'、'.join(f"{k} {v}个" for k,v in low['by_region'].items())+'。']
    lines+=['','同一患者可在不同历史截止点进入不同区域。反向邻居指六维速度余弦小于0的邻居。','',
      '## 3. 二维投影怎样改变局部一致性','',
      '分别在六维状态和原PCA二维坐标中寻找20位不同患者。对于二维邻居，同时计算其六维原始速度和二维投影速度的方向一致性。','',
      '![二维与六维近邻的一致性比较](projection_consistency.png)','',
      '| 近邻空间 / 速度空间 | 测试平均C | C第10分位 |','|---|---:|---:|']
    for r in summary['projection_comparison']['conditions']:lines.append(f"| {r['name']} | {r['mean']:.4f} | {r['p10']:.4f} |")
    proj=summary['projection_comparison'];lines+=['',f"两种近邻列表中相同患者的平均占比为 **{proj['mean_neighbor_overlap']*100:.2f}%**。二维坐标保留原状态{provenance.get('state_variance_retained_2d',.5301)*100:.2f}%的方差，其邻域组成与六维邻域存在上述差异。",'',
      '随机参照从20位不同训练患者中取状态，排除查询患者本人；每位训练患者的代表状态以固定种子随机选择。它用于比较相邻患者与一般患者之间的方向一致性。','',
      '## 4. 用未来12、24个月的真实MMSE评价预测','',
      '对每个历史截止点，在目标月份±3个月窗口内选择最接近目标日期、且有真实MMSE的下一次检查。模型只使用截止点及之前的历史，沿六维ODE直接推进到目标检查的实际时间。误差为预测MMSE与真实MMSE的绝对差，单位为分。时间沿用原季度聚合数据。','',
      '主相关分析先按患者汇总：D和Jacobian取该患者各有效起点的平均值，E取相应预测绝对误差的平均值。按患者重抽样1000次给出95%区间。调整相关性使用秩残差，控制年龄、实际间隔、历史次数、最近MMSE及其缺失和时间间隔、性别、当前诊断构成。','',
      '| 预测时长 | 测试患者 / 起点 | MMSE MAE（患者等权） | D与误差Spearman | 95%区间 | 调整后相关 | 95%区间 |','|---|---:|---:|---:|---|---:|---|']
    for h in ['12','24']:
        r=summary['forecast']['horizons'][h];ci=r['patient_bootstrap_95ci'];a=ci['D'];b=ci['adjusted_D']
        lines.append(f"| {h}个月 | {r['test_patients']} / {r['test_origins']} | {r['mmse_mae_patient_weighted']:.3f} | {f(r['sensitivity_error']['spearman'])} | [{a[0]:.3f}, {a[1]:.3f}] | {f(r['adjusted_sensitivity_error'])} | [{b[0]:.3f}, {b[1]:.3f}] |")
    lines+=['','![局部敏感性与未来预测误差](sensitivity_forecast_error.png)','',
      '图中每个散点是一位患者。四分组的D边界由相应时长的训练患者确定；柱高为测试患者的平均MMSE绝对误差。','',
      '| 预测时长 | Jacobian与误差Spearman | 95%区间 | 调整后相关 | 95%区间 |','|---|---:|---|---:|---|']
    for h in ['12','24']:
        r=summary['forecast']['horizons'][h];ci=r['patient_bootstrap_95ci'];a=ci['J'];b=ci['adjusted_J']
        lines.append(f"| {h}个月 | {f(r['jacobian_error']['spearman'])} | [{a[0]:.3f}, {a[1]:.3f}] | {f(r['adjusted_jacobian_error'])} | [{b[0]:.3f}, {b[1]:.3f}] |")
    lines+=['','下面按预测起点列出区域误差，单位为MMSE分。与局部一致性及敏感性表结合，可以比较各区域的动力学与预测表现。','',
      '| 测试区域 | 12个月起点数 | 12个月MAE | 24个月起点数 | 24个月MAE |','|---|---:|---:|---:|---:|']
    for region in REGIONS:
        a=next(r for r in summary['forecast']['region_forecast_errors'] if r['horizon_months']==12 and r['region']==region)
        b=next(r for r in summary['forecast']['region_forecast_errors'] if r['horizon_months']==24 and r['region']==region)
        lines.append(f"| {region} | {a['origins']} | {f(a['mmse_mae'])} | {b['origins']} | {f(b['mmse_mae'])} |")
    lines+=['','## 5. 邻域大小与区域误差','',
      '| k | 测试平均C | 测试平均D | 12个月D与误差相关 | 24个月D与误差相关 |','|---|---:|---:|---:|---:|']
    for k in [10,20,40]:
        m=summary['neighbor_size'][str(k)];lines.append(f"| {k} | {m['C_mean']:.4f} | {m['D_mean']:.4f} | {f(summary['forecast']['horizons']['12']['knn_sensitivity'][str(k)]['spearman'])} | {f(summary['forecast']['horizons']['24']['knn_sensitivity'][str(k)]['spearman'])} |")
    lines+=['','区域的12/24个月误差见 [region_forecast_errors.csv](region_forecast_errors.csv)，敏感性四组见 [sensitivity_quartiles.csv](sensitivity_quartiles.csv)。','',
      '## 6. 本实验对动力学规整化的启示','',
      '以调整后的D—误差相关及患者bootstrap区间判断局部敏感性与长期误差的联系，再结合区域误差和不同k值的结果。正相关区间整体高于零，表示在所控制的临床信息之外，敏感性更高的患者同时具有更大误差。区间跨零则呈现较弱或不稳定的关联。','',
      '## 7. 原始结果与复现','',
      f"所有近邻搜索、向量场、Jacobian及ODE预测在物理GPU {provenance['physical_gpu']}（{provenance['gpu_name']}）上完成。CPU用于数据整理、统计汇总、bootstrap及绘图。模型没有重新训练。",'',
      '[状态级数值](state_local_metrics.csv) · [近邻索引与完整数值](neighborhoods.npz) · [预测起点与误差](forecast_errors.csv) · [患者级汇总](patient_forecast_summary.csv) · [完整指标](metrics.json) · [检查与设备记录](provenance.json)','',
      '```bash',f"CUDA_VISIBLE_DEVICES={provenance['physical_gpu']} /home/xjh/miniforge3/envs/ode/bin/python "+chr(92),
      f"  -m paper_odeae.diagnostics.local_dynamics --gpu {provenance['physical_gpu']} "+chr(92),
      f"  --source {provenance.get('state_source',SOURCE)} "+chr(92),f'  --output {out}','```','']
    (out/'report.zh-CN.md').write_text('\n'.join(lines))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',default='2',choices=['0','1','2','3']);p.add_argument('--output',default=str(OUT));p.add_argument('--source',default=str(SOURCE));args=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(1);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    source=Path(args.source);a=np.load(source/'states.npz');meta=json.loads((source/'metadata.json').read_text());info=json.loads((source/'statistics.json').read_text())
    checkpoint=Path(info['checkpoint']);assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==info['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cuda',weights_only=False);cfg=saved['config'];model=paper_model_from_config(saved['schema'],cfg).cuda().eval();model.load_state_dict(saved['state_dict'])
    payload=pickle.loads(Path(cfg['data']).read_bytes());assert hashlib.sha256(Path(cfg['data']).read_bytes()).hexdigest()==info['dataset_sha256']
    z=torch.tensor(a['states'],device='cuda');xy=torch.tensor(a['coordinates'],device='cuda');v=model.forward_ode(z)
    torch.testing.assert_close(v,torch.tensor(a['velocities'],device='cuda'),rtol=1e-5,atol=1e-6)
    ids={pid:i for i,pid in enumerate(sorted({r['subject_id'] for r in meta}))};pid=torch.tensor([ids[r['subject_id']] for r in meta],device='cuda')
    train=torch.tensor([r['split']=='train' for r in meta],device='cuda');test=np.array([r['split']=='test' for r in meta]);reference=train.nonzero().squeeze(1)
    labels=torch.tensor([-1 if r['diagnosis'] is None else int(r['diagnosis']) for r in meta],device='cuda')
    neighbors,_=distinct_patient_knn(z,z[train],pid,pid[train],40,len(ids));neighbors=reference[neighbors]
    assert (pid[neighbors]!=pid[:,None]).all();assert (pid[neighbors].sort(1).values.diff(dim=1)!=0).all()
    Cij,Dij,dist=pair_metrics(z,v,neighbors);region,prob,coverage=region_assignment(labels[neighbors[:,:20]])
    J=squared_jacobian_norm(model.forward_ode,z).sqrt();print('Full 6D distinct-patient neighbors and Jacobians complete',flush=True)
    nn2,_=distinct_patient_knn(xy,xy[train],pid,pid[train],20,len(ids));nn2=reference[nn2]
    c2,_,_=pair_metrics(z,v,nn2);pv=torch.tensor(a['projected_velocities'],device='cuda');c2p,_,_=pair_metrics(xy,pv,nn2)
    overlap=(pid[neighbors[:,:20]][:,:,None]==pid[nn2][:,None,:]).any(2).float().mean(1)
    random_neighbors,random_C,random_D=random_patient_control(z,v,pid,reference)
    np.savez_compressed(out/'random_neighbors_control.npz',neighbors=random_neighbors.cpu().numpy(),C=random_C.cpu().numpy(),D=random_D.cpu().numpy())
    projection={'C_2d_neighbors_full_velocity':c2.mean(1).cpu().numpy(),'C_2d_neighbors_projected_velocity':c2p.mean(1).cpu().numpy(),'neighbor_patient_overlap':overlap.cpu().numpy()}
    C=Cij[:,:20].mean(1).cpu().numpy();D=Dij[:,:20].mean(1).cpu().numpy();jac=J.cpu().numpy();regions=region.cpu().numpy()
    d_by_k={k:Dij[:,:k].mean(1).cpu().numpy() for k in [10,20,40]}
    state_rows=[{'subject_id':r['subject_id'],'split':r['split'],'prefix':r['prefix'],'region':REGIONS[regions[i]],'C20':float(C[i]),'D20':float(D[i]),'J':float(jac[i]),'distance20':float(dist[i,:20].mean()),'cn_fraction':float(prob[i,0]),'mci_fraction':float(prob[i,1]),'dementia_fraction':float(prob[i,2]),'neighbor_label_coverage':float(coverage[i])} for i,r in enumerate(meta)]
    write_csv(out/'state_local_metrics.csv',state_rows)
    np.savez_compressed(out/'neighborhoods.npz',neighbors_6d=neighbors.cpu().numpy(),neighbors_2d=nn2.cpu().numpy(),C_pairs=Cij.cpu().numpy(),D_pairs=Dij.cpu().numpy(),distances_6d=dist.cpu().numpy(),jacobian_norm=jac,regions=regions,**projection)
    region_summary=[]
    cij=Cij[:,:20].cpu().numpy()
    for code,name in enumerate(REGIONS):
        mask=test&(regions==code)
        region_summary.append({'region':name,'states':int(mask.sum()),'patients':len({meta[i]['subject_id'] for i in np.flatnonzero(mask)}),
          'C_mean':float(C[mask].mean()) if mask.any() else None,'C_p10':float(np.quantile(C[mask],.1)) if mask.any() else None,
          'D_mean':float(D[mask].mean()) if mask.any() else None,'J_mean':float(jac[mask].mean()) if mask.any() else None,
          'opposite_pair_fraction':float((cij[mask]<0).mean()) if mask.any() else None})
    print('2D projection comparison complete',flush=True)
    schema=saved['schema'];fields={f['name']:i for i,f in enumerate(schema)};mmse=fields['mmse'];head=sum(f.get('classes',1) for f in schema[:mmse])
    scale=payload['metadata']['scale'][mmse];offset=payload['metadata']['offset'][mmse]
    sequences={str(s['subject_id']):s for s in payload['sequences']};lookup={(r['subject_id'],r['prefix']):i for i,r in enumerate(meta)}
    training_mmse=[float(s['values'][i,mmse]*scale+offset) for s in payload['sequences'] if s['split']=='train' for i in np.flatnonzero(s['mask'][:,mmse])]
    impute=float(np.median(training_mmse));queries=[]
    for subject,s in sequences.items():
        for i in range(len(s['ages'])-1):
            past=np.flatnonzero(s['mask'][:i+1,mmse]);last=int(past[-1]) if len(past) else None
            for horizon in [12,24]:
                target=horizon/12;choices=[j for j in range(i+1,len(s['ages'])) if s['mask'][j,mmse] and abs(float(s['ages'][j]-s['ages'][i])-target)<=.25]
                if not choices:continue
                j=min(choices,key=lambda j:(abs(float(s['ages'][j]-s['ages'][i])-target),j));at=lookup[(subject,i+1)]
                last_value=float(s['values'][last,mmse]*scale+offset) if last is not None else None
                dx=meta[at]['diagnosis'];sex=s['values'][i,fields['sex']] if s['mask'][i,fields['sex']] else .5
                queries.append({'subject_id':subject,'split':s['split'],'horizon_months':horizon,'start_visit':i+1,'target_visit':j+1,'state_index':at,
                  'years':float(s['ages'][j]-s['ages'][i]),'age':float(s['ages'][i]),'history_visits':i+1,'last_mmse':last_value if last_value is not None else impute,
                  'last_mmse_missing':int(last is None),'last_mmse_lag':float(s['ages'][i]-s['ages'][last]) if last is not None else 0.,'sex':float(sex),
                  'dx_cn':int(dx==0),'dx_mci':int(dx==1),'dx_dem':int(dx==2),'dx_missing':int(dx is None),
                  'observed_mmse':float(s['values'][j,mmse]*scale+offset),'last_actual_mmse':last_value,
                  'region':REGIONS[regions[at]],'C20':float(C[at]),'D10':float(d_by_k[10][at]),'D20':float(D[at]),'D40':float(d_by_k[40][at]),'J':float(jac[at])})
    for start in range(0,len(queries),64):
        batch=queries[start:start+64];indices=torch.tensor([q['state_index'] for q in batch],device='cuda');delta=torch.tensor([q['years'] for q in batch],device='cuda')
        end=model.solve(model.forward_ode,z[indices],delta);pred=(model.decoder(end)[:,head]*scale+offset).cpu().numpy()
        for r,value in zip(batch,pred):
            r['predicted_mmse']=float(value);r['mmse_error']=abs(float(value)-r['observed_mmse'])
            r['persistence_error']=abs(r['last_actual_mmse']-r['observed_mmse']) if r['last_actual_mmse'] is not None else None
    forecast_check=check_forecasts(model,sequences,z,queries,head,scale,offset)
    write_csv(out/'forecast_errors.csv',queries);print(f'Long-horizon forecasts complete: {len(queries)} origins across all splits',flush=True)
    forecast=summarize_forecasts(queries,out)
    conditions=[]
    for name,value in [('6D neighbors / 6D velocities',C),('2D neighbors / 6D velocities',projection['C_2d_neighbors_full_velocity']),('2D neighbors / projected velocities',projection['C_2d_neighbors_projected_velocity'])]:
        conditions.append({'name':name,'mean':float(value[test].mean()),'p10':float(np.quantile(value[test],.1))})
    conditions.append({'name':'Random distinct training patients / 6D velocities','mean':float(random_C.cpu().numpy()[test].mean()),'p10':float(np.quantile(random_C.cpu().numpy()[test],.1))})
    summary={'regions':region_summary,'forecast':forecast,'projection_comparison':{'conditions':conditions,'mean_neighbor_overlap':float(projection['neighbor_patient_overlap'][test].mean())},
             'neighbor_size':{str(k):{'C_mean':float(Cij[:,:k].mean(1).cpu().numpy()[test].mean()),'D_mean':float(d_by_k[k][test].mean())} for k in [10,20,40]}}
    low=test&(C<.9)
    summary['low_consistency_states']={'states':int(low.sum()),'fraction':float(low.sum()/test.sum()),
        'by_region':{name:int((low&(regions==i)).sum()) for i,name in enumerate(REGIONS) if (low&(regions==i)).any()}}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    plots(out,a['coordinates'],meta,regions,C,D,jac,projection,queries,forecast)
    provenance={'model_label':info.get('model_label','shared_matched'),'checkpoint_epoch':saved['epoch'],'state_source':str(source),'state_variance_retained_2d':info['state_variance_retained_2d'],
                'checkpoint_sha256':info['checkpoint_sha256'],'dataset_sha256':info['dataset_sha256'],'physical_gpu':args.gpu,'gpu_name':torch.cuda.get_device_name(),
                'model_device':str(next(model.parameters()).device),'knn_device':str(neighbors.device),'jacobian_device':str(J.device),'forecast_device':str(end.device),
                'states':len(z),'reference_states':len(reference),'reference_patients':int(pid[train].unique().numel()),'same_patient_excluded':True,'unique_neighbor_patients':True,
                'k':20,'k_sensitivity':[10,40],'forecast_origins':len(queries),'horizon_tolerance_years':.25,'minimum_speed_norm':float(v.norm(dim=-1).min()),
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'training_performed':False}
    provenance['forecast_check']=forecast_check
    provenance.update(random_control_seed=0,cosine_epsilon=1e-12,sensitivity_epsilon=1e-8)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2));report(out,summary,provenance)
    print(json.dumps({'regions':region_summary,'horizons':forecast['horizons'],'projection':summary['projection_comparison']},indent=2),flush=True)


if __name__=='__main__':main()
