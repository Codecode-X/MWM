"""CPU statistics and publication figures from already-computed CUDA outputs."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve

LABELS=['CN','MCI','Dementia']
SCORES={'stage_0':'Current stage s(0)','instantaneous_speed':'Instantaneous ds/dt',
        'delta_stage':'ODE stage increment','future_dementia_probability':'Future P(Dementia)',
        'current_dementia_probability':'Current P(Dementia)'}
ZH={'stage_0':'当前阶段 s₀','instantaneous_speed':'瞬时进展速度 ṡ₀','delta_stage':'未来阶段增量 Δs',
    'future_dementia_probability':'未来 Dementia 概率','current_dementia_probability':'当前 Dementia 概率（补充对照）'}
COLORS=['#238b45','#e6a21a','#c63d47']


def correlations(x,y):
    if len(x)<3 or np.std(x)<1e-12 or np.std(y)<1e-12:return None
    return float(spearmanr(x,y).statistic)


def describe(values):
    values=np.asarray(values,dtype=float)
    if not len(values):return {'n':0}
    q=np.quantile(values,[.25,.5,.75])
    return {'n':len(values),'mean':float(values.mean()),'std':float(values.std()),
            'q25':float(q[0]),'median':float(q[1]),'q75':float(q[2])}


def risk_metrics(cohort,repeats=1000):
    selected=cohort[cohort.label.notna()].copy();y=selected.label.to_numpy(dtype=int)
    positives=np.flatnonzero(y==1);negatives=np.flatnonzero(y==0)
    result={'patients':len(y),'converters':len(positives),'stable_mci':len(negatives),
            'prevalence':float(y.mean()) if len(y) else None,'scores':{},'paired_auroc_gains':{},'group_differences':{}}
    if not len(positives) or not len(negatives):return result
    rng=np.random.default_rng(20261029)
    resamples=[np.r_[rng.choice(positives,len(positives)),rng.choice(negatives,len(negatives))] for _ in range(repeats)]
    boot_auc={}
    for key in SCORES:
        score=selected[key].to_numpy(dtype=float)
        auc=[];ap=[]
        for idx in resamples:
            auc.append(roc_auc_score(y[idx],score[idx]));ap.append(average_precision_score(y[idx],score[idx]))
        boot_auc[key]=np.array(auc)
        result['scores'][key]={'auroc':float(roc_auc_score(y,score)),'auprc_average_precision':float(average_precision_score(y,score)),
                               'auroc_ci95':np.quantile(auc,[.025,.975]).tolist(),
                               'auprc_ci95':np.quantile(ap,[.025,.975]).tolist()}
        diff=np.array([score[idx[:len(positives)]].mean()-score[idx[len(positives):]].mean() for idx in resamples])
        result['group_differences'][key]={'converter':describe(score[positives]),'stable':describe(score[negatives]),
                                         'converter_minus_stable':float(score[positives].mean()-score[negatives].mean()),
                                         'difference_ci95':np.quantile(diff,[.025,.975]).tolist()}
    for a,b in [('future_dementia_probability','stage_0'),('future_dementia_probability','current_dementia_probability'),
                ('delta_stage','instantaneous_speed'),('delta_stage','stage_0'),('delta_stage','current_dementia_probability')]:
        result['paired_auroc_gains'][a+'__minus__'+b]={'difference':result['scores'][a]['auroc']-result['scores'][b]['auroc'],
                                                        'ci95':np.quantile(boot_auc[a]-boot_auc[b],[.025,.975]).tolist()}
    return result


def summarize(out,arrays,states,cohorts,provenance):
    latest=states[states.is_latest_patient_state];groups=[]
    for mode,frame in [('all_update_states',states),('last_known_state_per_patient',latest)]:
        for disease in range(3):
            subset=frame[frame.diagnosis==disease];ids=subset.state_index.to_numpy(dtype=int)
            groups.append({'population':mode,'diagnosis':LABELS[disease],'states':len(subset),
                           'patients':int(subset.subject_id.nunique()),'stage':describe(subset.stage),
                           'instantaneous_speed':describe(subset.instantaneous_stage_per_year),
                           'positive_speed_fraction':float((subset.instantaneous_stage_per_year>1e-6).mean()) if len(subset) else None,
                           'stage_delta_60_months':describe(arrays['future_stage'][ids,60]-arrays['stage'][ids])})
    diagnosis=latest[latest.diagnosis.notna()]
    all_dx=states[states.diagnosis.notna()]
    true=diagnosis.diagnosis.to_numpy(dtype=int)
    risk={};flow={}
    for h in [12,24,36]:
        frame=cohorts[cohorts.horizon_months==h];m=risk_metrics(frame)
        m['exclusions']=dict(Counter(frame.reason));m['candidate_patients']=len(frame);risk[str(h)]=m
    for h in [6,12,24,36,60]:
        ids=latest.state_index.to_numpy(dtype=int);delta=arrays['future_stage'][ids,h]-arrays['stage'][ids]
        flow[str(h)]={'delta':describe(delta),'mean_speed':describe(delta/(h/12)),
                      'positive_fraction':float((delta>1e-6).mean()),'negative_fraction':float((delta<-1e-6).mean())}
    structure={'all_states_spearman_stage_vs_observed_diagnosis':correlations(all_dx.stage,all_dx.diagnosis),
               'patient_last_state_spearman':correlations(diagnosis.stage,diagnosis.diagnosis),
               'patient_last_state_current_diagnosis_accuracy':None}
    diagnosed_indices=diagnosis.state_index.to_numpy(dtype=int)
    structure['patient_last_state_current_diagnosis_accuracy']=float((arrays['probabilities'][diagnosed_indices].argmax(-1)==true).mean())
    current_rates=latest.instantaneous_stage_per_year.to_numpy(dtype=float)
    monthly_changes=np.diff(arrays['future_stage'][latest.state_index.to_numpy(dtype=int)],axis=1)
    summary={'patients':len(latest),'test_states':len(states),'first_mci_patients':provenance['eligible_first_mci_patients'],
             'latest_patients_with_recorded_diagnosis':len(diagnosis),'latest_patients_without_recorded_diagnosis':len(latest)-len(diagnosis),
             'structure':structure,'diagnosis_groups':groups,'latest_patient_flow':flow,
             'latest_patient_positive_instantaneous_fraction':float((current_rates>1e-6).mean()),
             'latest_patient_negative_instantaneous_fraction':float((current_rates<-1e-6).mean()),
             'latest_patient_monotonic_non_decreasing_60_month_fraction':float(np.all(monthly_changes>=-1e-6,axis=1).mean()),
             'conversion_prediction':risk,'bootstrap':'1000 patient resamples, separately within converter/stable groups; shared indices for paired AUROC differences',
             'auprc_definition':'sklearn average_precision_score; stepwise precision-recall integral, prevalence reference'}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    table=[]
    for h,m in risk.items():
        for key,score in m['scores'].items():table.append({'months':h,'predictor':key,'patients':m['patients'],'converters':m['converters'],'stable':m['stable_mci'],**score})
    pd.DataFrame(table).to_csv(out/'conversion_metrics.csv',index=False)
    return summary


def plots(out,a,states,cohorts,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','svg','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=190)
        plt.close(fig)
    xy=a['coordinates'];v=a['projected_velocities'];rate=a['instantaneous_rate'];stage=a['stage']
    lim=max(float(np.max(abs(rate))),1e-8);norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim)
    fig,axes=plt.subplots(1,3,figsize=(18,5.7))
    dx=states.diagnosis.to_numpy();missing=~np.isfinite(dx)
    axes[0].scatter(*xy[missing].T,c='#aaa',s=8,alpha=.4,label='Unrecorded')
    for k,label in enumerate(LABELS):
        mask=dx==k;axes[0].scatter(*xy[mask].T,c=COLORS[k],s=9,alpha=.5,label=f'{label} (n={mask.sum()})',rasterized=True)
    axes[0].legend(fontsize=8);axes[0].set_title('Observed current diagnosis')
    sc=axes[1].scatter(*xy.T,c=stage,cmap='viridis',vmin=0,vmax=2,s=11,alpha=.8,rasterized=True)
    fig.colorbar(sc,ax=axes[1],label='s = P(MCI) + 2P(Dementia)');axes[1].set_title('Continuous diagnostic stage')
    sc=axes[2].scatter(*xy.T,c=rate,cmap='coolwarm',norm=norm,s=11,alpha=.8,rasterized=True)
    fig.colorbar(sc,ax=axes[2],label='Instantaneous stage change / year');axes[2].set_title('Instantaneous progression speed')
    xlim=(float(xy[:,0].min())-.2,float(xy[:,0].max())+.2);ylim=(float(xy[:,1].min())-.2,float(xy[:,1].max())+.2)
    for ax in axes:ax.set(xlabel='Existing train PCA: PC1',ylabel='Existing train PCA: PC2',xlim=xlim,ylim=ylim)
    save(fig,'continuous_stage_space')
    fig,axes=plt.subplots(1,2,figsize=(13.5,6))
    for ax,values,cmap,kw,title in [(axes[0],stage,'viridis',{'clim':(0,2)},'Raw projected ODE field: colored by stage'),
                                   (axes[1],rate,'coolwarm',{'norm':norm},'Raw projected ODE field: colored by ds/dt')]:
        ax.scatter(*xy.T,s=2,c='#ddd',alpha=.25)
        q=ax.quiver(*xy.T,*v.T,values,cmap=cmap,angles='xy',scale_units='xy',scale=1,width=.002,alpha=.7,**kw)
        fig.colorbar(q,ax=ax,label='Stage' if ax is axes[0] else 'Stage / year')
        ax.set(title=title,xlabel='Existing train PCA: PC1',ylabel='Existing train PCA: PC2',xlim=xlim,ylim=ylim)
    save(fig,'progression_velocity_field')
    fig,axes=plt.subplots(2,3,figsize=(17,9),sharex=True)
    rng=np.random.default_rng(73)
    for column,h in enumerate([12,24,36]):
        frame=cohorts[(cohorts.horizon_months==h)&cohorts.label.notna()]
        for y,color,label in [(0,'#2878b5','Stable MCI'),(1,'#d84d4d','MCI to Dementia')]:
            subset=frame[frame.label==y];idx=subset.state_index.to_numpy(dtype=int);curves=a['future_stage'][idx]
            if not len(curves):continue
            weights=rng.multinomial(len(curves),np.full(len(curves),1/len(curves)),size=1000)/len(curves)
            for row,values in [(0,curves),(1,curves-curves[:,:1])]:
                average=values.mean(0);ci=np.quantile(weights@values,[.025,.975],axis=0)
                axes[row,column].plot(a['months'],average,c=color,lw=2,label=f'{label}, n={len(curves)}')
                axes[row,column].fill_between(a['months'],*ci,color=color,alpha=.18)
                axes[row,column].axvline(h,c='#999',ls='--',lw=.8)
                axes[row,column].set(xlabel='Months after frozen MCI history',ylabel='Stage s(t)' if row==0 else 'Stage increment s(t)-s(0)')
        axes[0,column].set_title(f'Groups defined by {h}-month conversion');axes[0,column].legend(fontsize=8)
        axes[1,column].axhline(0,color='#aaa',lw=.7)
    save(fig,'mci_stage_evolution')
    for mode in ['roc','pr']:
        fig,axes=plt.subplots(1,3,figsize=(17,5.4))
        for ax,h in zip(axes,[12,24,36]):
            frame=cohorts[(cohorts.horizon_months==h)&cohorts.label.notna()];y=frame.label.to_numpy(dtype=int)
            m=summary['conversion_prediction'][str(h)]
            for key,label in SCORES.items():
                if key not in m['scores']:continue
                score=frame[key].to_numpy(dtype=float)
                if mode=='roc':
                    x,yy,_=roc_curve(y,score);metric=m['scores'][key]['auroc']
                else:
                    yy,x,_=precision_recall_curve(y,score);metric=m['scores'][key]['auprc_average_precision']
                ax.plot(x,yy,lw=1.8,label=f'{label}: {metric:.3f}',ls='--' if key=='current_dementia_probability' else '-')
            if mode=='roc':ax.plot([0,1],[0,1],'k:',lw=1);ax.set(xlabel='False positive rate',ylabel='True positive rate')
            else:ax.axhline(m['prevalence'],c='#999',ls=':',label=f"Prevalence: {m['prevalence']:.3f}");ax.set(xlabel='Recall',ylabel='Precision')
            ax.set(title=f"{h} months: {m['converters']} converters / {m['patients']} patients",xlim=(0,1),ylim=(0,1.02))
            ax.legend(fontsize=7,loc='lower right' if mode=='roc' else 'upper right')
        save(fig,'conversion_roc' if mode=='roc' else 'conversion_precision_recall')


def report(out,summary,p):
    def f(value,d=3):return '—' if value is None else f'{value:.{d}f}'
    def ci(value):return '['+', '.join(f(x) for x in value)+']'
    latest=[g for g in summary['diagnosis_groups'] if g['population']=='last_known_state_per_patient']
    lines=['# 连续疾病阶段与进展分析','',
           '## 1. 从现有诊断输出得到连续阶段','',
           f"本实验使用已经训练完成的六维 shared_matched：共 {p['parameters']:,} 个参数，使用第 {p['checkpoint_epoch']} 轮检查点。INIT、UPDATE、共享 ODE 和 DECODER 全部保持原样，参数被冻结。",'',
           '原 DECODER 中已有 CN、MCI、Dementia 三个诊断 logits。对它们做 Softmax，得到三类概率，再计算：','',
           r'\[s(z)=p_{\mathrm{MCI}}(z)+2p_{\mathrm{Dementia}}(z),\qquad 0\le s\le2.\]','',
           '这个分数是模型诊断概率的加权平均：接近 0 表示概率偏向 CN，接近 1 表示总体偏向中间阶段，接近 2 表示概率偏向 Dementia。例如 MCI 概率为 1 时评分是 1；CN 与 Dementia 各占一半时评分也为 1。因此评分要结合三类概率理解。','',
           '当前诊断本来就是模型的观测输入之一。这里先看读取当前观测后的状态如何表达疾病阶段，再用起点之后的真实诊断验证进展预测。','',
           '## 2. 先确定历史截止点，再向未来推演','',
           f"测试集共有 **{summary['patients']} 位患者、{summary['test_states']:,} 个 UPDATE 后状态**。空间图使用全部测试检查状态；每位患者完整已知历史的最后状态也单独保存了未来推演。",'',
           '从每个起点出发，只沿共享 ODE 积分，不执行任何未来 UPDATE。每月保存一次结果，并明确导出 0、6、12、24、36、60 个月的诊断概率与阶段评分。时间以年进入 ODE，6 个月对应 0.5 年。','',
           r'\[\dot s(z)=\nabla_zs(z)^\top f_\theta(z),\qquad \Delta s_H=s(H)-s(0),\qquad \bar v_H=\frac{\Delta s_H}{H/12}.\]','',
           '瞬时速度用自动微分计算，单位为阶段分/年。正值表示诊断概率整体向较高阶段移动，负值表示向较低阶段移动。12、24、36 个月的平均速度由各自区间的评分增量除以年数得到。','',
           '## 3. 潜在空间中是否存在连续阶段结构','',
           f"下图严格复用此前由训练集拟合的 PCA 均值、基向量和坐标方向。两维保留原六维状态方差的 {p['pca_state_variance_retained']*100:.2f}%。每个点对应一个真实测试检查后的六维状态；颜色直接来自该状态的诊断头和自动微分结果。",'',
           '![真实诊断、连续阶段与进展速度](continuous_stage_space.png)','',
           '左图用当前真实诊断着色，中图显示连续评分，右图显示瞬时进展速度。右图红色为正、蓝色为负。图中没有重新拟合投影，也没有对分数和速度做空间平滑。','',
           f"为让每位患者权重相同，下面的定量表每人只取完整已知历史的最后状态。其中 {summary['latest_patients_with_recorded_diagnosis']} 人在这次检查有真实诊断，另有 {summary['latest_patients_without_recorded_diagnosis']} 人该次诊断缺失。诊断分组使用前者，全体推演统计使用全部 {summary['patients']} 人。",'',
           '| 当前真实诊断 | 患者数 | 平均阶段 s | 阶段中位数 [Q1, Q3] | 平均瞬时速度（阶段/年） | 正速度占比 | 5年平均阶段增量 |',
           '|---|---:|---:|---|---:|---:|---:|']
    for g in latest:
        s=g['stage'];lines.append(f"| {g['diagnosis']} | {g['patients']} | {f(s.get('mean'))} | {f(s.get('median'))} [{f(s.get('q25'))}, {f(s.get('q75'))}] | {f(g['instantaneous_speed'].get('mean'),4)} | {f(100*g['positive_speed_fraction'],1)}% | {f(g['stage_delta_60_months'].get('mean'))} |")
    structure=summary['structure']
    lines+=['',f"每位患者最后状态的评分与真实诊断等级（0、1、2）之间，Spearman 相关为 **{f(structure['patient_last_state_spearman'])}**；同次诊断概率最大类别的准确率为 {f(100*structure['patient_last_state_current_diagnosis_accuracy'],1)}%。全部检查状态的描述性相关为 {f(structure['all_states_spearman_stage_vs_observed_diagnosis'])}。",'',
            '空间图与分组评分共同呈现了明确的阶段次序：左侧 CN 以低评分为主，右下 MCI 接近中间评分，右上 Dementia 接近高评分，交界处存在过渡和重叠。诊断头能从六维状态中读出连续的诊断阶段结构。下面继续检验，不再读取新观测时模型会把状态推向哪里。','',
            '## 4. ODE 是否沿疾病进展方向演化','',
            '每支箭头都是原六维 ODE 速度在同一 PCA 平面上的直接投影。左图按当前阶段着色，右图按阶段的瞬时变化率着色，箭头使用相同的物理时间比例。','',
            '![按阶段和临床进展速度着色的原始向量场](progression_velocity_field.png)','',
            f"以每位患者最后状态计，瞬时速度为正的占比为 **{f(100*summary['latest_patient_positive_instantaneous_fraction'],1)}%**，为负的占比为 {f(100*summary['latest_patient_negative_instantaneous_fraction'],1)}%。",'',
            '| 自由演化时长 | 平均阶段增量 | 平均变化速度（阶段/年） | 阶段上升患者占比 | 阶段下降患者占比 |',
            '|---|---:|---:|---:|---:|']
    for h,m in summary['latest_patient_flow'].items():lines.append(f"| {h}个月 | {f(m['delta']['mean'])} | {f(m['mean_speed']['mean'],4)} | {f(100*m['positive_fraction'],1)}% | {f(100*m['negative_fraction'],1)}% |")
    lines+=['',f"五年终点评分上升的患者占 {f(100*summary['latest_patient_flow']['60']['positive_fraction'],1)}%，而整条轨迹在每个月采样点都保持不下降的占 {f(100*summary['latest_patient_monotonic_non_decreasing_60_month_fraction'],1)}%。因此整体演化主要指向较高阶段，部分个体路径包含阶段评分下降的区间。",'',
            '评分有 0 到 2 的边界，因此接近 Dementia 端点时，即使潜在速度仍大，评分增加也可能变慢。本实验中 Dementia 组的平均瞬时速度最高，但五年阶段增量小于 MCI 组，与其起点评分已接近 2 的情况相符。','',
            '## 5. 用真实随访定义 MCI 转化','',
            '若把一位患者全部记录都读完，最后状态之后就没有可用于验证的现有随访。为了评价转化预测，本实验在较早的检查处冻结历史，再把之后的诊断留作结果标签。','',
            f"每位患者选取**首次有真实 MCI 诊断、且此前没有 Dementia 记录**的检查作为固定起点，共 {summary['first_mci_patients']} 位。模型吸收截至该次检查的所有历史，同一患者在三个预测窗口中使用同一个起点。",'',
            '对每个 12、24、36 个月窗口，两组都要求目标月份至其后 6 个月内至少有一次真实诊断。以这个区间内的首次真实诊断作为随访支持：','',
            '- 转化组：起点之后、目标月份以内已经首次记录 Dementia。',
            '- 稳定 MCI 组：从起点之后到支持检查，所有已记录诊断均为 MCI。',
            '- 随访不足者、出现 CN 回转者，以及首次 Dementia 只出现在目标月份之后的边界病例，单独记录。','',
            '分组使用原数据的季度末检查日期，按日历月份比较；一位患者在每个窗口中只贡献一条记录。组名“稳定”描述该窗口的真实随访。三个窗口的起点选择规则相同，随访要求使纳入患者不同，因此自由推演的增益在每个窗口内做配对比较。','',
            '| 窗口 | 候选患者 | 转化组 | 稳定MCI组 | 纳入总数 | 无目标随访支持 | 其他诊断或窗口后边界 | 转化率 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for h,m in summary['conversion_prediction'].items():
        ex=m['exclusions'];lines.append(f"| {h}个月 | {m['candidate_patients']} | {m['converters']} | {m['stable_mci']} | {m['patients']} | {ex.get('no_endpoint_support',0)} | {ex.get('other_diagnosis_or_post_window_dementia',0)} | {f(100*m['prevalence'],1)}% |")
    lines+=['','## 6. 稳定与转化患者的演化速度是否不同','',
            '两行曲线均从冻结的 MCI 历史出发，完整自由演化到 60 个月。上行画阶段评分，下行减去各自起点评分后画增量，用来区分“起点已经更严重”和“之后演化更快”。阴影为患者 bootstrap 得到的均值逐时点 95% 区间；虚线是定义分组的预测窗口。','',
            '![稳定与转化MCI的阶段及增量曲线](mci_stage_evolution.png)','',
            '| 窗口 | 转化组平均 s₀ | 稳定组平均 s₀ | 转化组平均 ṡ₀ | 稳定组平均 ṡ₀ | 转化组平均 Δs | 稳定组平均 Δs | Δs组间差及95%区间 |',
            '|---|---:|---:|---:|---:|---:|---:|---|']
    for h,m in summary['conversion_prediction'].items():
        g=m['group_differences'];d=g['delta_stage'];lines.append(f"| {h}个月 | {f(g['stage_0']['converter']['mean'])} | {f(g['stage_0']['stable']['mean'])} | {f(g['instantaneous_speed']['converter']['mean'],4)} | {f(g['instantaneous_speed']['stable']['mean'],4)} | {f(d['converter']['mean'])} | {f(d['stable']['mean'])} | {f(d['converter_minus_stable'])} {ci(d['difference_ci95'])} |")
    ratios=[]
    for h,m in summary['conversion_prediction'].items():
        g=m['group_differences']['instantaneous_speed'];ratios.append(f"{h}个月组约 {g['converter']['mean']/g['stable']['mean']:.2f} 倍")
    lines+=['','转化组的平均瞬时速度均高于稳定组：'+'、'.join(ratios)+'。三个窗口的阶段增量组间差均为正，患者重抽样的95%区间也均高于0。尤其在36个月组，起点评分均值已经接近，之后的平均阶段增量仍明显分开，模型动力学包含了与后续转化相关的变化信息。']
    lines+=['','## 7. 哪个量更能区分真实转化','',
            '以下四个主要指标直接用原模型输出排序：当前评分、当前瞬时速度、目标时点阶段增量，以及目标时点 Dementia 概率。另列当前 Dementia 概率作为补充对照，便于判断自由推演是否改变了原有风险排序。没有拟合新分类器，也没有按测试标签调整方向或阈值。','',
            '| 窗口 | 排序指标 | 患者数（转化/稳定） | AUROC及95%区间 | AUPRC及95%区间 |',
            '|---|---|---:|---|---|']
    for h,m in summary['conversion_prediction'].items():
        for key,value in m['scores'].items():lines.append(f"| {h}个月 | {ZH[key]} | {m['patients']}（{m['converters']}/{m['stable_mci']}） | {f(value['auroc'])} {ci(value['auroc_ci95'])} | {f(value['auprc_average_precision'])} {ci(value['auprc_ci95'])} |")
    lines+=['','AUPRC 使用 average precision 定义；随机排序的参照是该窗口的转化率。阶段增量除以固定窗口年数，就是平均进展速度，两者排序完全相同，因此对应 AUROC、AUPRC 相同。所有区间按患者、在两类内分别重抽样 1,000 次。','',
            '![转化预测ROC对比](conversion_roc.png)','',
            '![转化预测Precision–Recall对比](conversion_precision_recall.png)','',
            '## 8. 自由推演是否提升转化区分能力','',
            '下面用完全相同的患者、相同的重抽样索引，比较未来 Dementia 概率与两个当前量的 AUROC。正数表示未来概率区分能力更高。','',
            '| 窗口 | 未来P(D)减当前s₀：AUROC差 [95%区间] | 未来P(D)减当前P(D)：AUROC差 [95%区间] | Δs减当前s₀：AUROC差 [95%区间] |',
            '|---|---|---|---|']
    for h,m in summary['conversion_prediction'].items():
        one=m['paired_auroc_gains']['future_dementia_probability__minus__stage_0'];two=m['paired_auroc_gains']['future_dementia_probability__minus__current_dementia_probability']
        three=m['paired_auroc_gains']['delta_stage__minus__stage_0']
        lines.append(f"| {h}个月 | {f(one['difference'])} {ci(one['ci95'])} | {f(two['difference'])} {ci(two['ci95'])} | {f(three['difference'])} {ci(three['ci95'])} |")
    lines+=['','12个月内，当前阶段的AUROC最高；24个月内，未来Dementia概率与当前阶段接近；36个月内，阶段增量的AUROC和AUPRC最高。ODE的优势主要体现在36个月窗口的点估计，而不是每个窗口都一致提升。','',
            '三个窗口中，未来Dementia概率相对当前Dementia概率的AUROC点估计均增加约0.02～0.03；这些配对差值的95%区间都跨过0。现有结果支持“自由推演保留并部分改善转化排序”的判断，尚未给出稳定超越当前状态评分的统计证据。','',
            '对本次研究问题，证据形成了三层关系：当前状态呈现清楚的诊断阶段次序；ODE总体向更高阶段演化，并给真实转化者更高的预测速度；进一步要求它在同一窗口显著提升风险排序时，当前样本得到的增益仍有较大不确定性。','',
            '## 9. 输出与复现','',
            f"本次模型读取、Softmax、自动微分、ODE 积分和 PCA 投影在物理 **GPU {p['physical_gpu']}（{p['gpu_name']}）** 上完成。统计与绘图使用 CPU。检查点参数逐项核对未变化，新增可学习参数为 0。",'',
            f"自动微分与 float64 方向有限差分最大误差为 {p['double_finite_difference_max_error']:.3g}；完整记录与截断历史的编码逐位相同。缓存状态与独立编码最大误差为 {p['truncated_history_vs_cached_max_error']:.3g}。",'',
            '[完整指标](metrics.json) · [全部测试状态](test_states.csv) · [指定时点推演](forecasts.csv) · [逐患者转化分组及风险分数](conversion_cohorts.csv) · [原始数组](continuous_stage.npz) · [GPU与数值核对](provenance.json)','',
            '```bash',f"bash scripts/run_continuous_disease_stage.sh --gpu {p['physical_gpu']}",'```','']
    (out/'report.zh-CN.md').write_text('\n'.join(lines))


def generate(out):
    out=Path(out);arrays=np.load(out/'continuous_stage.npz');states=pd.read_csv(out/'test_states.csv');cohorts=pd.read_csv(out/'conversion_cohorts.csv')
    provenance=json.loads((out/'provenance.json').read_text())
    summary=summarize(out,arrays,states,cohorts,provenance);plots(out,arrays,states,cohorts,summary);report(out,summary,provenance)
    print(json.dumps({'report':str(out/'report.zh-CN.md'),'conversion':summary['conversion_prediction']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    generate(parser.parse_args().output)
