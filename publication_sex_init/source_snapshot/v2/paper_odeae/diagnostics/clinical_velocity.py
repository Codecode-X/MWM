"""Experiment A: map latent ODE velocities to clinical score rates on CUDA."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np
import torch
from torch import nn
from .state_space import ROOT
from paper_odeae.models import paper_model_from_config
from paper_odeae.data import collate
from paper_odeae.train import device_batch

SOURCE=ROOT/'paper_odeae/results/shared_matched_state_space'
DEFAULT=ROOT/'paper_odeae/results/shared_matched_clinical_velocity'
MARKERS=['mmse','cdrsb']+[f'adas_q{i}' for i in range(1,14)]


def clinical_values_and_rates(model,z,indices,scales,offsets):
    """Exact g(z) and J_g(z)f(z), with physical-unit transforms."""
    if z.device.type!='cuda':raise RuntimeError('Clinical dynamics require CUDA')
    head=model.decoder
    if not (len(head)==3 and isinstance(head[0],nn.Linear) and isinstance(head[1],nn.Tanh) and isinstance(head[2],nn.Linear)):
        raise ValueError('Exact clinical JVP requires Linear-Tanh-Linear decoder')
    f=model.forward_ode(z);hidden=head[1](head[0](z))
    hidden_rate=(1-hidden.square())*torch.nn.functional.linear(f,head[0].weight)
    normalized_rate=torch.nn.functional.linear(hidden_rate,head[2].weight)
    values=head(z)[:,indices]*scales+offsets
    rates=normalized_rate[:,indices]*scales
    return values,rates,f


def write_csv(path,rows):
    if not rows:return
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def numeric_metrics(real,pred):
    from scipy.stats import pearsonr,spearmanr
    real=np.asarray(real,float);pred=np.asarray(pred,float)
    if not len(real):return {'n':0}
    corr=lambda fn:float(fn(real,pred).statistic) if len(real)>2 and real.std()>1e-10 and pred.std()>1e-10 else None
    nonzero=real!=0
    return {'n':len(real),'mae':float(np.abs(real-pred).mean()),'rmse':float(np.square(real-pred).mean()**.5),
            'bias':float((pred-real).mean()),'pearson':corr(pearsonr),'spearman':corr(spearmanr),
            'real_mean':float(real.mean()),'pred_mean':float(pred.mean()),'real_std':float(real.std()),'pred_std':float(pred.std()),
            'up_down_accuracy_nonzero':float((np.sign(real[nonzero])==np.sign(pred[nonzero])).mean()) if nonzero.any() else None,
            'nonzero_intervals':int(nonzero.sum()),'three_way_sign_accuracy':float((np.sign(real)==np.sign(pred)).mean()),
            'stable_band_0_5_per_year_accuracy':float((np.where(real<-.5,-1,np.where(real>.5,1,0))==np.where(pred<-.5,-1,np.where(pred>.5,1,0))).mean()),
            'real_direction_counts':{'decrease':int((real<0).sum()),'unchanged':int((real==0).sum()),'increase':int((real>0).sum())}}


def bootstrap_mae(rows,repeats=1000):
    # Resample patients as clusters, preserving their within-patient intervals.
    from collections import defaultdict
    values=defaultdict(list)
    for r in rows:values[r['subject_id']].append(abs(r['predicted_rate']-r['observed_rate']))
    arrays=list(values.values())
    if len(arrays)<2:return None
    sums=np.array([sum(a) for a in arrays]);counts=np.array([len(a) for a in arrays]);rng=np.random.default_rng(0)
    index=rng.integers(len(arrays),size=(repeats,len(arrays)))
    result=sums[index].sum(1)/counts[index].sum(1)
    return np.quantile(result,[.025,.975]).tolist()


def summarize(rows,out,metadata,instant,coords,projected):
    from collections import defaultdict
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    metrics={};groups=[]
    for marker in MARKERS:
        marker_rows=[r for r in rows if r['marker']==marker];metrics[marker]={}
        for split in ['train','validation','test']:
            selected=[r for r in marker_rows if r['split']==split]
            result={}
            for name,filter_ in [('adjacent_visit',lambda r:r['adjacent_visit']),
                                  ('next_observed',lambda r:r['next_observed']),
                                  ('adjacent_gap_ge_0_5y',lambda r:r['adjacent_visit'] and r['years']>=.5),
                                  ('adjacent_gap_ge_0_9y',lambda r:r['adjacent_visit'] and r['years']>=.9)]:
                use=[r for r in selected if filter_(r)]
                item=numeric_metrics([r['observed_rate'] for r in use],[r['predicted_rate'] for r in use])
                item['patients']=len({r['subject_id'] for r in use})
                if marker=='mmse' and split=='test':item['mae_patient_cluster_bootstrap_95ci']=bootstrap_mae(use)
                if use:
                    real=np.array([r['observed_rate'] for r in use]);item['zero_change_baseline_mae']=float(abs(real).mean())
                result[name]=item
            metrics[marker][split]=result
        if marker=='mmse':
            for split in ['validation','test']:
                for diagnosis in [0,1,2,None]:
                    use=[r for r in marker_rows if r['split']==split and r['adjacent_visit'] and r['diagnosis']==diagnosis]
                    group={'split':split,'diagnosis':'未记录' if diagnosis is None else ['CN','MCI','Dementia'][diagnosis],**numeric_metrics([r['observed_rate'] for r in use],[r['predicted_rate'] for r in use])}
                    group['patients']=len({r['subject_id'] for r in use});group['instantaneous_predicted_mean']=float(np.mean([r['instantaneous_rate'] for r in use])) if use else None;groups.append(group)
    mmse=[r for r in rows if r['marker']=='mmse' and r['adjacent_visit']]
    def patient_means(split):
        patients=defaultdict(list)
        for r in mmse:
            if r['split']==split:patients[r['subject_id']].append(r)
        return [{'subject_id':pid,'intervals':len(rs),'years':sum(r['years'] for r in rs),
                 'observed_rate':sum(r['observed_rate']*r['years'] for r in rs)/sum(r['years'] for r in rs),
                 'predicted_rate':sum(r['predicted_rate']*r['years'] for r in rs)/sum(r['years'] for r in rs),
                 'initial_predicted_rate':rs[0]['predicted_rate'],'initial_observed_mmse':rs[0]['observed_start'],
                 'initial_diagnosis':rs[0]['diagnosis'],
                 'duration_weighted_observed_mmse':sum(r['observed_start']*r['years'] for r in rs)/sum(r['years'] for r in rs)} for pid,rs in patients.items()]
    training=patient_means('train');test=patient_means('test');threshold=float(np.quantile([r['observed_rate'] for r in training],.2))
    yy=np.array([r['observed_rate'] for r in test]);pp=np.array([r['predicted_rate'] for r in test]);fast=yy<=threshold
    ranking={'training_patient_rate_20th_percentile':threshold,'test_patients':len(test),'test_fast_decliners':int(fast.sum()),
             'patient_metrics':numeric_metrics(yy,pp),'fast_decline_auc':float(roc_auc_score(fast,-pp)) if len(np.unique(fast))==2 else None,
             'zero_change_baseline_mae':float(abs(yy).mean())}
    # Patient-level ranking controls: early forecast uses only the first eligible
    # history; repeated forecast aggregates the matching pre-interval estimates.
    scores={'repeated_interval_ode':-pp,'repeated_current_mmse':-np.array([r['duration_weighted_observed_mmse'] for r in test]),
            'initial_interval_ode':-np.array([r['initial_predicted_rate'] for r in test]),
            'initial_mmse':-np.array([r['initial_observed_mmse'] for r in test])}
    ranking['auc_controls']={name:float(roc_auc_score(fast,score)) for name,score in scores.items()}
    valid_dx=np.array([r['initial_diagnosis'] is not None for r in test])
    ranking['auc_controls']['initial_diagnosis']=float(roc_auc_score(fast[valid_dx],np.array([r['initial_diagnosis'] or 0 for r in test])[valid_dx]))
    ranking['initial_diagnosis_valid_patients']=int(valid_dx.sum())
    rng=np.random.default_rng(1);ci={}
    for name,score in scores.items():
        estimates=[]
        for _ in range(1000):
            ids=rng.integers(len(test),size=len(test))
            if len(np.unique(fast[ids]))==2:estimates.append(roc_auc_score(fast[ids],score[ids]))
        ci[name]=np.quantile(estimates,[.025,.975]).tolist()
    ranking['auc_patient_bootstrap_95ci']=ci
    train_direction=[r['observed_rate'] for r in mmse if r['split']=='train' and r['observed_rate']!=0]
    majority=-1 if (np.array(train_direction)<0).mean()>=.5 else 1
    nonzero=[r for r in mmse if r['split']=='test' and r['observed_rate']!=0]
    ranking['training_majority_direction']=majority
    ranking['test_nonzero_majority_direction_accuracy']=float(np.mean([np.sign(r['observed_rate'])==majority for r in nonzero]))
    mci=np.array([r['split']=='test' and r['diagnosis']==1 for r in metadata]);upper_left=(projected[:,0]<0)&(projected[:,1]>0)
    direction=[]
    for name,mask in [('upper_left',mci&upper_left),('other_directions',mci&~upper_left)]:
        indices=np.flatnonzero(mask);index_set=set(indices.tolist());use=[r for r in mmse if r['split']=='test' and r['state_index'] in index_set]
        direction.append({'group':name,'test_mci_states':len(indices),'mean_instantaneous_mmse_rate':float(instant[indices,0].mean()) if len(indices) else None,
                          'fraction_instantaneous_decline':float((instant[indices,0]<0).mean()) if len(indices) else None,
                          'adjacent_intervals':len(use),'mean_observed_rate':float(np.mean([r['observed_rate'] for r in use])) if use else None,
                          'fraction_observed_decline':float(np.mean([r['observed_rate']<0 for r in use])) if use else None})
    summary={'metrics':metrics,'mmse_diagnosis_groups':groups,'patient_ranking':ranking,'mci_upper_left':direction}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    write_csv(out/'mmse_diagnosis_groups.csv',[{k:v for k,v in g.items() if not isinstance(v,dict)} for g in groups]);write_csv(out/'patient_mmse_rates.csv',test)
    return summary


def plots(out,rows,coords,projected,velnorm,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','svg','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    train=[r for r in rows if r['marker']=='mmse' and r['split']=='train' and r['adjacent_visit']]
    test=[r for r in rows if r['marker']=='mmse' and r['split']=='test' and r['adjacent_visit']]
    limit=max(1.,float(np.quantile(np.abs([r['observed_rate'] for r in train]),.95)))
    norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit);idx=np.array([r['state_index'] for r in test]);xy=coords[idx];v=projected[idx]
    fig,axes=plt.subplots(1,3,figsize=(19,6))
    for ax,values,title,clinical in [(axes[0],velnorm[idx],'Latent ODE speed',False),
          (axes[1],np.array([r['instantaneous_rate'] for r in test]),'Instantaneous predicted MMSE rate',True),
          (axes[2],np.array([r['observed_rate'] for r in test]),'Observed adjacent-visit MMSE rate',True)]:
        ax.scatter(*xy.T,s=3,c='#bbb',alpha=.2)
        q=ax.quiver(*xy.T,*v.T,values,angles='xy',scale_units='xy',scale=1,width=.0025,cmap='coolwarm' if clinical else 'plasma',norm=norm if clinical else None)
        fig.colorbar(q,ax=ax,label='MMSE points/year' if clinical else 'Latent units/year',extend='both' if clinical else 'neither')
        ax.set(title=title,xlabel='Original state PC1',ylabel='Original state PC2',aspect='equal')
    save(fig,'clinical_velocity_fields')
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    x=np.array([r['observed_rate'] for r in test]);y=np.array([r['predicted_rate'] for r in test]);gap=np.array([r['years'] for r in test])
    sc=axes[0].scatter(x,y,c=gap,cmap='viridis',s=10,alpha=.45);fig.colorbar(sc,ax=axes[0],label='Visit interval (years)')
    low=min(x.min(),y.min());high=max(x.max(),y.max());axes[0].plot([low,high],[low,high],'k--',lw=1)
    axes[0].set(title='Interval rates on held-out test patients',xlabel='Observed MMSE points/year',ylabel='ODE-predicted MMSE points/year')
    for diagnosis,label,color in [(0,'CN','#238b45'),(1,'MCI','#e6a21a'),(2,'Dementia','#c63d47')]:
        use=[r for r in test if r['diagnosis']==diagnosis];axes[1].scatter([r['observed_rate'] for r in use],[r['predicted_rate'] for r in use],s=10,alpha=.4,c=color,label=label)
    axes[1].axhline(0,c='#aaa',lw=1);axes[1].axvline(0,c='#aaa',lw=1);axes[1].legend();axes[1].set(title='Rates grouped by current diagnosis',xlabel='Observed rate',ylabel='Predicted rate')
    patient=list(csv.DictReader((out/'patient_mmse_rates.csv').open()));xx=np.array([float(r['observed_rate']) for r in patient]);yy=np.array([float(r['predicted_rate']) for r in patient])
    axes[2].scatter(xx,yy,s=13,alpha=.5);axes[2].axvline(summary['patient_ranking']['training_patient_rate_20th_percentile'],ls='--',c='#c63d47',label='Training bottom-20% threshold');axes[2].legend(fontsize=8)
    axes[2].set(title='Patient-level duration-weighted rates',xlabel='Observed rate',ylabel='Predicted rate')
    save(fig,'rate_prediction_accuracy')
    from sklearn.metrics import roc_curve,roc_auc_score
    observed=np.array([float(r['observed_rate']) for r in patient]);fast=observed<=summary['patient_ranking']['training_patient_rate_20th_percentile']
    fig,ax=plt.subplots(figsize=(7,6))
    scores=[('Initial-history ODE rate',-np.array([float(r['initial_predicted_rate']) for r in patient])),
            ('Initial MMSE',-np.array([float(r['initial_observed_mmse']) for r in patient]))]
    for label,score in scores:
        fpr,tpr,_=roc_curve(fast,score);ax.plot(fpr,tpr,lw=2,label=f'{label}: AUC={roc_auc_score(fast,score):.3f}')
    valid=np.array([r['initial_diagnosis'] not in ['',None] for r in patient]);score=np.array([float(r['initial_diagnosis'] or 0) for r in patient])
    fpr,tpr,_=roc_curve(fast[valid],score[valid]);ax.plot(fpr,tpr,lw=2,label=f'Initial diagnosis: AUC={roc_auc_score(fast[valid],score[valid]):.3f}')
    ax.plot([0,1],[0,1],'k--',lw=1);ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Faster decline: prediction from first eligible history');ax.legend(fontsize=9)
    save(fig,'fast_decline_ranking')
    return limit


def report(out,summary,provenance,limit):
    metrics=summary['metrics'];mmse=metrics['mmse']['test'];main=mmse['adjacent_visit'];rank=summary['patient_ranking']
    def f(x,n=3):return '—' if x is None else f'{x:.{n}f}'
    lines=['# 实验 A：从潜在 ODE 速度到真实临床变化速度','',
      '## 1. 用已有临床预测头读取潜在动力学','',
      f"本实验使用 {provenance.get('model_label','原始 shared_matched')}（22,545参数，第{provenance.get('checkpoint_epoch',56)}轮检查点）。它的 DECODER 已输出 MMSE、CDR-SB 和 ADAS 各题分数，因此直接使用现有预测头。数据和模型保持原样。",'',
      '临床预测值先从训练集 min-max 尺度还原到原始分数。瞬时变化率通过链式法则计算：','',
      r'\[\frac{d\widehat{\mathrm{MMSE}}}{dt}=\nabla_z g(z)^\top f(z).\]','',
      'MMSE 的单位为分/年，负值表示模型预测评分下降，正值表示预测评分上升。CDR-SB 和 ADAS 题目同样使用原始题目分数/年；各题单独报告，未拼接为 ADAS-Cog 总分。','',
      '## 2. 在相同检查间隔上评价','',
      '对患者第i次检查，模型只读取截至这次检查的历史，得到更新后的状态。随后独立沿ODE推进到第i+1次检查，中间不读入新观测。真实与预测年变化率分别为：','',
      r'\[r_i^{real}=\frac{x_{i+1}-x_i}{\Delta t},\qquad r_i^{pred}=\frac{g(\Phi_{\Delta t}(z_i^+))-g(z_i^+)}{\Delta t}.\]','',
      '主评价使用数据中相邻检查且两端均有真实MMSE的间隔。补充评价连接相邻的有效MMSE记录，并按间隔长度分组。重复访视属于同一患者，MAE置信区间按患者整组重抽样。','',
      '时间轴沿用模型数据中的自然季度聚合：同季度数值检查取平均，时间记在季度末，间隔按该时间戳换算成年。所有真实与预测变化率使用同一时间轴。','',
      '## 3. 测试集临床速度场','',
      '三幅图使用该检查点训练状态拟合的PCA坐标，并使用相同的相邻MMSE检查起点。每支箭头的方向和长度来自该状态的真实潜在速度投影：左图按六维潜在速度大小着色，中图按预测瞬时MMSE年变化率着色，右图按真实相邻检查年变化率着色。','',
      '![潜在速度与临床变化速度](clinical_velocity_fields.png)','',
      f'临床两图共用色条，负值为蓝色、正值为红色。色条范围为 ±{limit:.2f} 分/年，取训练集真实年变化率绝对值的95%分位；超出范围的值使用端点颜色，原始数值完整保留。','',
      '## 4. MMSE年变化率预测指标','',
      '| 评价间隔 | 间隔数 | 患者数 | MAE（分/年） | Pearson | Spearman | 上升/下降准确率 | 三方向准确率 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,key in [('相邻检查，全部间隔','adjacent_visit'),('相邻检查，至少0.5年','adjacent_gap_ge_0_5y'),('相邻检查，至少0.9年','adjacent_gap_ge_0_9y'),('相邻有效MMSE记录','next_observed')]:
        m=mmse[key];lines.append(f"| {name} | {m['n']} | {m['patients']} | {f(m.get('mae'))} | {f(m.get('pearson'))} | {f(m.get('spearman'))} | {f(100*m['up_down_accuracy_nonzero']) if m.get('up_down_accuracy_nonzero') is not None else '—'}% | {f(100*m['three_way_sign_accuracy']) if m.get('three_way_sign_accuracy') is not None else '—'}% |")
    ci=main['mae_patient_cluster_bootstrap_95ci']
    lines+=['',f"主评价MAE为 **{main['mae']:.3f}分/年**，患者整组bootstrap的95%区间为 **[{ci[0]:.3f}, {ci[1]:.3f}]**。零变化基线的MAE为 {main['zero_change_baseline_mae']:.3f}分/年。真实变化率标准差为 {main['real_std']:.3f}，模型预测为 {main['pred_std']:.3f}分/年。",'',
      f"上升/下降准确率仅评价真实变化非零的 {main['nonzero_intervals']} 个间隔；三方向准确率将下降、不变、上升一起评价。真实方向数量为：下降 {main['real_direction_counts']['decrease']}、不变 {main['real_direction_counts']['unchanged']}、上升 {main['real_direction_counts']['increase']}。训练集多数方向基线在测试非零变化间隔上的准确率为 {rank['test_nonzero_majority_direction_accuracy']*100:.2f}%。",'',
      f"模型在年变化率MAE上较零变化基线改善 {(main['zero_change_baseline_mae']-main['mae'])/main['zero_change_baseline_mae']*100:.2f}%。预测变化率标准差为真实值的 {main['pred_std']/main['real_std']*100:.2f}%；方向准确率较上述多数方向基线相差 {(main['up_down_accuracy_nonzero']-rank['test_nonzero_majority_direction_accuracy'])*100:+.2f} 个百分点。",'',
      '![区间变化率预测与真实值](rate_prediction_accuracy.png)','',
      '## 5. 不同诊断区域的实际与预测速度','',
      '| 当前诊断 | 测试间隔数 | 真实平均变化率 | 预测区间平均变化率 | 预测瞬时平均变化率 | MAE |','|---|---:|---:|---:|---:|---:|']
    for g in summary['mmse_diagnosis_groups']:
        if g['split']=='test':lines.append(f"| {g['diagnosis']} | {g['n']} | {f(g.get('real_mean'))} | {f(g.get('pred_mean'))} | {f(g.get('instantaneous_predicted_mean'))} | {f(g.get('mae'))} |")
    lines+=['','## 6. MCI区域向左上移动意味着什么','',
      '沿用原PCA方向，将横向速度小于0、纵向速度大于0定义为“左上”。下面分别列出这些MCI状态的预测瞬时变化，以及有后续MMSE观测的间隔：','',
      '| MCI速度方向 | 测试状态数 | 平均预测瞬时MMSE速度 | 预测下降占比 | 有真实相邻MMSE的间隔数 | 真实平均变化率 | 真实下降占比 |','|---|---:|---:|---:|---:|---:|---:|']
    for d in summary['mci_upper_left']:
        lines.append(f"| {'左上' if d['group']=='upper_left' else '其他方向'} | {d['test_mci_states']} | {f(d['mean_instantaneous_mmse_rate'])} | {f(100*d['fraction_instantaneous_decline']) if d['fraction_instantaneous_decline'] is not None else '—'}% | {d['adjacent_intervals']} | {f(d['mean_observed_rate'])} | {f(100*d['fraction_observed_decline']) if d['fraction_observed_decline'] is not None else '—'}% |")
    lines+=['','## 7. 能否识别变化较快的患者','',
      '对每位患者，将相邻间隔的变化量相加，再除以总间隔年数，得到按时间加权的平均年变化率。以训练患者真实速率最低20%的分位点定义本实验的“较快下降”组。','',
      f"阈值为 {rank['training_patient_rate_20th_percentile']:.3f}分/年。测试集共 {rank['test_patients']} 位有效患者，其中 {rank['test_fast_decliners']} 位属于该组；模型按下降速度排序的 AUROC 为 **{f(rank['fast_decline_auc'])}**，患者级 Spearman 相关性为 **{f(rank['patient_metrics']['spearman'])}**。",'',
      '将多次历史下的预测汇总后，模型可以区分总体变化较快与较慢的患者。下面用相同患者上的已有MMSE作排序参照，并单独列出只使用首次有效历史的早期预测：','',
      '| 排序方式 | 有效患者数 | AUROC | 患者bootstrap 95%区间 |','|---|---:|---:|---|']
    for key,label in [('repeated_interval_ode','各间隔起点ODE预测，按时长汇总'),('repeated_current_mmse','各间隔起点实际MMSE，按时长汇总'),('initial_interval_ode','首次有效历史的ODE预测'),('initial_mmse','首次有效历史的实际MMSE')]:
        bounds=rank['auc_patient_bootstrap_95ci'][key]
        lines.append(f"| {label} | {rank['test_patients']} | {rank['auc_controls'][key]:.3f} | [{bounds[0]:.3f}, {bounds[1]:.3f}] |")
    lines+=[f"| 首次有效历史的诊断代码 | {rank['initial_diagnosis_valid_patients']} | {rank['auc_controls']['initial_diagnosis']:.3f} | — |",'',
      '首次有效历史指该患者首个可评价MMSE区间起点之前的全部检查。其后的实际随访用于计算患者平均真实变化率。AUROC越高，下降较快的患者越倾向于排在前面。','',
      '![首次有效历史的较快下降排序](fast_decline_ranking.png)','',
      '## 8. 其他临床指标','',
      '| 指标 | 测试相邻间隔数 | MAE（原始分/年） | Pearson | Spearman |','|---|---:|---:|---:|---:|']
    for marker in MARKERS[1:]:
        m=metrics[marker]['test']['adjacent_visit'];lines.append(f"| {marker} | {m['n']} | {f(m.get('mae'))} | {f(m.get('pearson'))} | {f(m.get('spearman'))} |")
    lines+=['','## 9. 计算与数据记录','',
      f"计算在物理GPU {provenance['physical_gpu']}（{provenance['gpu_name']}）上完成。模型、链式法则与ODE求解均在CUDA上执行。采用原检查点和原训练集PCA；统计汇总与绘图使用CPU。",'',
      '[逐间隔原始数据](interval_rates.csv) · [完整指标](metrics.json) · [患者平均速率](patient_mmse_rates.csv) · [模型与数值核对](provenance.json)','',
      '```bash',f"CUDA_VISIBLE_DEVICES={provenance['physical_gpu']} /home/xjh/miniforge3/envs/ode/bin/python "+chr(92),
      f"  -m paper_odeae.diagnostics.clinical_velocity --gpu {provenance['physical_gpu']} "+chr(92),
      f"  --source {provenance.get('state_source',SOURCE)} "+chr(92),f'  --output {out}','```','']
    (out/'report.zh-CN.md').write_text('\n'.join(lines))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',default='2',choices=['0','1','2','3']);p.add_argument('--output',default=str(DEFAULT));p.add_argument('--source',default=str(SOURCE));args=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(1);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    source=Path(args.source);info=json.loads((source/'statistics.json').read_text());cached=np.load(source/'states.npz');metadata=json.loads((source/'metadata.json').read_text())
    checkpoint=Path(info['checkpoint']);assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==info['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cuda',weights_only=False);cfg=saved['config'];payload=pickle.loads(Path(cfg['data']).read_bytes())
    assert hashlib.sha256(Path(cfg['data']).read_bytes()).hexdigest()==info['dataset_sha256']
    model=paper_model_from_config(saved['schema'],cfg).cuda().eval();model.load_state_dict(saved['state_dict'])
    schema=saved['schema'];fields={f['name']:i for i,f in enumerate(schema)}
    output_indices={f['name']:sum(g.get('classes',1) for g in schema[:i]) for i,f in enumerate(schema)}
    indices=torch.tensor([output_indices[m] for m in MARKERS],device='cuda');scales=torch.tensor([payload['metadata']['scale'][fields[m]] for m in MARKERS],device='cuda');offsets=torch.tensor([payload['metadata']['offset'][fields[m]] for m in MARKERS],device='cuda')
    z=torch.tensor(cached['states'],device='cuda');values=[];instant=[];velocity=[]
    for chunk in z.split(512):
        a,b,c=clinical_values_and_rates(model,chunk,indices,scales,offsets);values.append(a);instant.append(b);velocity.append(c)
    values=torch.cat(values);instant=torch.cat(instant);velocity=torch.cat(velocity)
    torch.testing.assert_close(velocity,torch.tensor(cached['velocities'],device='cuda'),rtol=1e-5,atol=1e-6)
    print('Clinical heads and instantaneous rates complete',flush=True)
    # Numerical check: double-precision autograd and central differences on GPU.
    import copy
    check=copy.deepcopy(model).double();check_z=z[:16].double().requires_grad_(True)
    with torch.enable_grad():
        head=check.decoder(check_z)[:,int(indices[0])]*scales[0].double()+offsets[0].double()
        grad=torch.autograd.grad(head.sum(),check_z)[0];field=check.forward_ode(check_z)
        autograd_rate=(grad*field).sum(-1)
        _,analytic,_=clinical_values_and_rates(check,check_z,indices,scales.double(),offsets.double())
        torch.testing.assert_close(autograd_rate,analytic[:,0],rtol=1e-10,atol=1e-10)
        eps=1e-5
        plus=check.decoder(check_z+eps*field)[:,int(indices[0])]*scales[0].double()
        minus=check.decoder(check_z-eps*field)[:,int(indices[0])]*scales[0].double()
        fd=(plus-minus)/(2*eps);torch.testing.assert_close(fd,analytic[:,0],rtol=1e-5,atol=1e-7)
    derivative_check=float((fd-analytic[:,0]).abs().max());del check
    lookup={(r['subject_id'],r['prefix']):i for i,r in enumerate(metadata)}
    queries={};sequences={str(s['subject_id']):s for s in payload['sequences']}
    for pid,s in sequences.items():
        for marker in MARKERS:
            k=fields[marker];obs=np.flatnonzero(s['mask'][:,k])
            for i,j in zip(obs[:-1],obs[1:]):queries.setdefault((pid,int(i),int(j)),[]).append(marker)
    keys=list(queries);start_index=torch.tensor([lookup[(pid,i+1)] for pid,i,j in keys],device='cuda')
    delta=torch.tensor([float(sequences[pid]['ages'][j]-sequences[pid]['ages'][i]) for pid,i,j in keys],device='cuda');assert (delta>0).all()
    endpoints=[]
    for at in range(0,len(keys),64):
        zz=model.solve(model.forward_ode,z[start_index[at:at+64]],delta[at:at+64])
        endpoints.append(model.decoder(zz)[:,indices]*scales+offsets)
    endpoints=torch.cat(endpoints);start=values[start_index];predicted=(endpoints-start)/delta[:,None]
    # Check cached prefix states by independently encoding representative histories.
    selected_keys=keys[::max(1,len(keys)//8)][:8];items=[(sequences[pid],i+1) for pid,i,j in selected_keys]
    batch=device_batch(collate(items));encoded=model(batch)['mu'];expected=z[torch.tensor([lookup[(pid,i+1)] for pid,i,j in selected_keys],device='cuda')]
    torch.testing.assert_close(encoded,expected,rtol=1e-4,atol=1e-4)
    changed={k:v.clone() if torch.is_tensor(v) else v for k,v in batch.items()}
    for b,cutoff in enumerate(batch['cutoff'].tolist()):
        for k,f in enumerate(schema):
            if f['kind']=='numerical':changed['values'][b,cutoff:,k]+=7
    torch.testing.assert_close(model(changed)['mu'],encoded,rtol=0,atol=0)
    print(f'Interval ODE forecasts complete: {len(keys)} unique patient intervals',flush=True)
    vi=values.cpu().numpy();ins=instant.cpu().numpy();pred=predicted.cpu().numpy();end=endpoints.cpu().numpy();dt=delta.cpu().numpy();rows=[]
    for q,(pid,i,j) in enumerate(keys):
        seq=sequences[pid];state_index=lookup[(pid,i+1)];diagnosis=metadata[state_index]['diagnosis']
        for marker in queries[(pid,i,j)]:
            col=MARKERS.index(marker);k=fields[marker];scale=payload['metadata']['scale'][k];offset=payload['metadata']['offset'][k]
            real_start=float(seq['values'][i,k]*scale+offset);real_end=float(seq['values'][j,k]*scale+offset)
            # MMSE/CDR-SB exam scores are reconstructed from float32 cache;
            # remove transform roundoff before deciding exact sign / equality.
            if marker in ['mmse','cdrsb'] or marker.startswith('adas_q'):
                real_start=round(real_start,4);real_end=round(real_end,4)
            rows.append({'subject_id':pid,'split':seq['split'],'marker':marker,'start_visit':i+1,'end_visit':j+1,'years':float(dt[q]),
                         'adjacent_visit':j==i+1,'next_observed':True,'state_index':state_index,'diagnosis':diagnosis,
                         'observed_start':real_start,'observed_end':real_end,'predicted_start':float(vi[state_index,col]),'predicted_end':float(end[q,col]),
                         'observed_rate':(real_end-real_start)/float(dt[q]),'predicted_rate':float(pred[q,col]),'instantaneous_rate':float(ins[state_index,col])})
    write_csv(out/'interval_rates.csv',rows)
    np.savez_compressed(out/'clinical_states.npz',markers=np.array(MARKERS),predicted_values=vi,instantaneous_rates=ins,coordinates=cached['coordinates'],projected_velocities=cached['projected_velocities'])
    summary=summarize(rows,out,metadata,ins,cached['coordinates'],cached['projected_velocities'])
    limit=plots(out,rows,cached['coordinates'],cached['projected_velocities'],velocity.norm(dim=-1).cpu().numpy(),summary)
    provenance={'model_label':info.get('model_label','shared_matched'),'checkpoint_epoch':saved['epoch'],'state_source':str(source),
                'checkpoint':str(checkpoint),'checkpoint_sha256':info['checkpoint_sha256'],'dataset_sha256':info['dataset_sha256'],
                'physical_gpu':args.gpu,'gpu_name':torch.cuda.get_device_name(),'model_device':str(next(model.parameters()).device),'derivative_device':str(z.device),
                'derivative_check_max_error':derivative_check,'cached_state_check_max_error':float((encoded-expected).abs().max()),'future_observation_isolation':'exact equality',
                'states':len(z),'unique_intervals':len(keys),'clinical_interval_rows':len(rows),'training_performed':False,'mmse_color_limit':limit,
                'pca':'source model train-only PCA reused','marker_heads':{m:{'field_index':fields[m],'output_index':output_indices[m],'scale':float(scales[k]),'offset':float(offsets[k])} for k,m in enumerate(MARKERS)}}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2));report(out,summary,provenance,limit)
    print(json.dumps({'test_mmse':summary['metrics']['mmse']['test'],'ranking':summary['patient_ranking'],'mci':summary['mci_upper_left']},indent=2),flush=True)


if __name__=='__main__':main()
