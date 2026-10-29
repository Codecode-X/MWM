"""Compare completed original and weak_prior research A/B artifacts."""
import csv
import json
from pathlib import Path
import numpy as np
from .state_space import ROOT

RESULTS=ROOT/'paper_odeae/results'
OUT=RESULTS/'weak_prior_research_AB'
VARIANTS=[('shared_matched','Original shared_matched',RESULTS/'shared_matched'),('weak_prior','weak_prior',RESULTS/'weak_prior')]


def read(path):return json.loads(path.read_text())
def csv_rows(path):return list(csv.DictReader(path.open()))


def paired_error(rows1,rows2,key,error):
    a={key(r):r for r in rows1};b={key(r):r for r in rows2};assert set(a)==set(b)
    ids=sorted(a);patients={pid:i for i,pid in enumerate(sorted({r['subject_id'] for r in rows1}))}
    sums=np.zeros(len(patients));counts=np.zeros(len(patients))
    for k in ids:
        r=a[k];i=patients[r['subject_id']];sums[i]+=error(a[k])-error(b[k]);counts[i]+=1
    rng=np.random.default_rng(0);sample=rng.integers(len(patients),size=(1000,len(patients)))
    bootstrap=sums[sample].sum(1)/counts[sample].sum(1)
    return {'rows':len(ids),'patients':len(patients),'error_reduction':float(sums.sum()/counts.sum()),'paired_patient_bootstrap_95ci':np.quantile(bootstrap,[.025,.975]).tolist()}


def visual_comparison(models):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm,Normalize
    fieldfig,axes=plt.subplots(2,3,figsize=(18,11));localfig,localaxes=plt.subplots(2,3,figsize=(18,11))
    limits=[]
    for model in models:
        model['state']=dict(np.load(model['source']/'states.npz'));model['meta']=read(model['source']/'metadata.json')
        model['Arows']=csv_rows(model['a']/'interval_rates.csv');model['Brows']=csv_rows(model['b']/'state_local_metrics.csv')
    for metric in ['C20','D20','J']:
        values=[float(r[metric]) for m in models for r in m['Brows'] if r['split']=='train']
        limits.append(np.quantile(values,[.01,.99]))
    for row,model in enumerate(models):
        use=[r for r in model['Arows'] if r['split']=='test' and r['marker']=='mmse' and r['adjacent_visit']=='True']
        idx=np.array([int(r['state_index']) for r in use]);a=model['state'];xy=a['coordinates'][idx];v=a['projected_velocities'][idx]
        colors=[np.linalg.norm(a['velocities'][idx],axis=-1),np.array([float(r['instantaneous_rate']) for r in use]),np.array([float(r['observed_rate']) for r in use])]
        limit=read(model['a']/'provenance.json')['mmse_color_limit']
        for col,(value,title) in enumerate(zip(colors,['Latent speed','Predicted instantaneous MMSE rate','Observed MMSE interval rate'])):
            ax=axes[row,col];q=ax.quiver(*xy.T,*v.T,value,angles='xy',scale_units='xy',scale=1,width=.0025,cmap='plasma' if col==0 else 'coolwarm',norm=None if col==0 else TwoSlopeNorm(0,-limit,limit))
            fieldfig.colorbar(q,ax=ax,label='Latent units/year' if col==0 else 'MMSE points/year');ax.set(title=model['label']+'\n'+title,xlabel='Own train-fitted PC1',ylabel='Own train-fitted PC2',aspect='equal')
        test=np.array([r['split']=='test' for r in model['Brows']]);xy=a['coordinates']
        for col,metric in enumerate(['C20','D20','J']):
            value=np.array([float(r[metric]) for r in model['Brows']]);ax=localaxes[row,col]
            sc=ax.scatter(*xy[test].T,c=value[test],norm=Normalize(*limits[col]),cmap='viridis',s=9,alpha=.65,rasterized=True)
            localfig.colorbar(sc,ax=ax,extend='both');ax.set(title=model['label']+' | '+metric,xlabel='Own train-fitted PC1',ylabel='Own train-fitted PC2')
    for fig,name in [(fieldfig,'comparison_A_fields'),(localfig,'comparison_B_maps')]:
        fig.tight_layout()
        for ext in ['png','svg','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180)
        plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True);models=[]
    for name,label,stem in VARIANTS:
        model={'name':name,'label':label,'source':Path(str(stem)+'_state_space'),'a':Path(str(stem)+'_clinical_velocity'),'b':Path(str(stem)+'_local_dynamics')}
        model['A']=read(model['a']/'metrics.json');model['B']=read(model['b']/'metrics.json')
        model['Aprovenance']=read(model['a']/'provenance.json');model['Bprovenance']=read(model['b']/'provenance.json')
        models.append(model)
    original,weak=models
    assert original['Aprovenance']['dataset_sha256']==weak['Aprovenance']['dataset_sha256']
    meta1=read(original['source']/'metadata.json');meta2=read(weak['source']/'metadata.json');assert meta1==meta2
    arows=[[r for r in csv_rows(m['a']/'interval_rates.csv') if r['split']=='test' and r['marker']=='mmse' and r['adjacent_visit']=='True'] for m in models]
    key=lambda r:(r['subject_id'],r['start_visit'],r['end_visit'])
    aa={key(r):r for r in arows[0]};bb={key(r):r for r in arows[1]}
    assert set(aa)==set(bb)
    assert all(aa[k]['observed_rate']==bb[k]['observed_rate'] and aa[k]['years']==bb[k]['years'] for k in aa)
    paired_A=paired_error(*arows,key,lambda r:abs(float(r['predicted_rate'])-float(r['observed_rate'])))
    paired_B={}
    for horizon in [12,24]:
        raw=[[r for r in csv_rows(m['b']/'patient_forecast_summary.csv') if r['split']=='test' and int(r['horizon_months'])==horizon] for m in models]
        paired_B[str(horizon)]=paired_error(*raw,lambda r:r['subject_id'],lambda r:float(r['mmse_error']))
    from sklearn.metrics import roc_auc_score
    pp=[{r['subject_id']:r for r in csv_rows(m['a']/'patient_mmse_rates.csv')} for m in models];assert set(pp[0])==set(pp[1]);ids=sorted(pp[0])
    y=np.array([float(pp[0][pid]['observed_rate']) for pid in ids]);assert all(pp[0][pid]['observed_rate']==pp[1][pid]['observed_rate'] for pid in ids)
    threshold=original['A']['patient_ranking']['training_patient_rate_20th_percentile'];assert threshold==weak['A']['patient_ranking']['training_patient_rate_20th_percentile']
    truth=y<=threshold;score=[-np.array([float(p[pid]['initial_predicted_rate']) for pid in ids]) for p in pp]
    rng=np.random.default_rng(1);bootstrap=[]
    for _ in range(1000):
        sample=rng.integers(len(ids),size=len(ids))
        if len(np.unique(truth[sample]))==2:bootstrap.append(roc_auc_score(truth[sample],score[1][sample])-roc_auc_score(truth[sample],score[0][sample]))
    paired_auc={'auc_increase':float(roc_auc_score(truth,score[1])-roc_auc_score(truth,score[0])),'paired_patient_bootstrap_95ci':np.quantile(bootstrap,[.025,.975]).tolist()}
    visual_comparison(models)
    summary={'experiment_A':{m['name']:m['A'] for m in models},'experiment_B':{m['name']:m['B'] for m in models},
             'paired_A_MAE':paired_A,'paired_A_initial_AUC':paired_auc,'paired_B_MAE':paired_B,
             'checks':{'same_dataset':True,'same_state_metadata':True,'same_MMSE_test_intervals_and_targets':True,'same_test_patients':True,'model_training_performed':False}}
    (OUT/'comparison.json').write_text(json.dumps(summary,indent=2))
    def f(x,n=3):return '—' if x is None else f'{x:.{n}f}'
    baseA=original['A']['metrics']['mmse']['test']['adjacent_visit'];newA=weak['A']['metrics']['mmse']['test']['adjacent_visit']
    lines=['# weak_prior 版本：重做实验 A 与实验 B','',
      '## 1. 比较对象和统一条件','',
      '原始 shared_matched 使用第56轮检查点，weak_prior 使用第70轮检查点。两者都是22,545参数、六维确定性状态和共享向量场；weak_prior 的训练额外加入0.01倍校正前预测损失。此次只执行状态提取与评价。','',
      '患者划分、17,165个历史状态、临床单位换算、检查间隔、近邻排除规则、区域阈值、bootstrap次数和随机种子均保持一致。每个模型只用自身训练状态拟合PCA，原始图与新图并列展示。实验A/B的数值比较使用同一患者、同一历史起点和同一真实目标。','',
      '## 2. 实验 A：临床变化速度','',
      '| 测试指标 | 原始 shared_matched | weak_prior |','|---|---:|---:|']
    for label,key in [('相邻MMSE间隔数','n'),('患者数','patients'),('MAE（分/年）','mae'),('RMSE（分/年）','rmse'),('Pearson','pearson'),('Spearman','spearman'),('预测年变化率标准差','pred_std')]:
        values=[str(int(m[key])) if key in ['n','patients'] else f(m[key]) for m in [baseA,newA]]
        lines.append(f'| {label} | {values[0]} | {values[1]} |')
    for label,key in [('上升/下降准确率','up_down_accuracy_nonzero'),('下降/不变/上升准确率','three_way_sign_accuracy')]:lines.append(f'| {label} | {baseA[key]*100:.2f}% | {newA[key]*100:.2f}% |')
    for label,key in [('首次有效历史的快速下降AUROC','initial_interval_ode')]:lines.append(f"| {label} | {original['A']['patient_ranking']['auc_controls'][key]:.3f} | {weak['A']['patient_ranking']['auc_controls'][key]:.3f} |")
    ci=paired_A['paired_patient_bootstrap_95ci'];aci=paired_auc['paired_patient_bootstrap_95ci']
    lines+=['',f"按同一患者配对，weak_prior 的年变化率MAE减少 {paired_A['error_reduction']:.4f}分/年，95%区间 [{ci[0]:.4f}, {ci[1]:.4f}]；首次有效历史AUROC变化 {paired_auc['auc_increase']:+.4f}，95%区间 [{aci[0]:.4f}, {aci[1]:.4f}]。",'',
      '![原始与weak_prior临床速度场](comparison_A_fields.png)','',
      '上排原始模型，下排weak_prior。三列分别按潜在速度、预测MMSE瞬时年变化率、真实随访MMSE年变化率着色。临床两列使用相同色条，箭头保留各模型真实潜在速度投影。','',
      '| 诊断组 | 原始真实平均速率 | 原始预测平均速率 | weak_prior预测平均速率 | 原始MAE | weak_prior MAE |','|---|---:|---:|---:|---:|---:|']
    for dx in ['CN','MCI','Dementia']:
        g=[next(r for r in m['A']['mmse_diagnosis_groups'] if r['split']=='test' and r['diagnosis']==dx) for m in models]
        lines.append(f"| {dx} | {g[0]['real_mean']:.3f} | {g[0]['pred_mean']:.3f} | {g[1]['pred_mean']:.3f} | {g[0]['mae']:.3f} | {g[1]['mae']:.3f} |")
    lines+=['','## 3. 实验 B：高维局部动力学','',
      '主邻域仍为20位不同训练患者，排除同一患者的所有其他状态。相同规则在两套六维表示上分别寻找近邻；Jacobian 使用各模型共享向量场的完整导数。','',
      '| 指标 | 原始 shared_matched | weak_prior |','|---|---:|---:|']
    for label,path in [('六维局部平均方向一致性',('neighbor_size','20','C_mean')),('六维局部平均敏感性D',('neighbor_size','20','D_mean')),('二维/六维近邻患者重合率',('projection_comparison','mean_neighbor_overlap'))]:
        values=[]
        for m in models:
            value=m['B']
            for key in path:value=value[key]
            values.append(value)
        lines.append(f'| {label} | {values[0]:.4f} | {values[1]:.4f} |')
    lines+=['','![原始与weak_prior的局部一致性和敏感性](comparison_B_maps.png)','',
      '上排原始模型，下排weak_prior，三列为方向一致性C、敏感性D和Jacobian范数。每列颜色范围统一，坐标是各自模型的训练状态PCA。','',
      '| 预测时长 | 模型 | 患者数 | MMSE MAE | D—误差Spearman | 调整后相关 | 调整后95%区间 |','|---|---|---:|---:|---:|---:|---|']
    for horizon in ['12','24']:
        for m in models:
            r=m['B']['forecast']['horizons'][horizon];ci=r['patient_bootstrap_95ci']['adjusted_D']
            lines.append(f"| {horizon}个月 | {m['name']} | {r['test_patients']} | {r['mmse_mae_patient_weighted']:.3f} | {f(r['sensitivity_error']['spearman'])} | {f(r['adjusted_sensitivity_error'])} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    for horizon in ['12','24']:
        paired=paired_B[horizon];ci=paired['paired_patient_bootstrap_95ci']
        lines+=['',f"{horizon}个月患者等权MAE减少 {paired['error_reduction']:.4f}分，配对患者bootstrap 95%区间 [{ci[0]:.4f}, {ci[1]:.4f}]。",'']
    lines+=['## 4. 这些变化说明什么','',
      f"weak_prior使MMSE年变化率MAE降低 {paired_A['error_reduction']/baseA['mae']*100:.2f}%，相关性略有提升。预测变化率的标准差由 {baseA['pred_std']:.3f} 增至 {newA['pred_std']:.3f} 分/年，真实变化率标准差为 {baseA['real_std']:.3f} 分/年，预测幅度仍明显更小。方向准确率和患者排序表现与原模型接近。",'',
      '12和24个月的预测误差点估计均下降，相应配对区间在表中列出。weak_prior的局部方向一致性略低、D相近；控制诊断和最近MMSE等信息后，两个时长的D—误差相关区间都覆盖零。校正前监督带来了小幅临床预测改善，实验B仍未呈现局部敏感性越高、长期误差越大的稳定正相关。','',
      '## 5. 独立报告和复现','',
      '[weak_prior实验A完整报告](../weak_prior_clinical_velocity/report.zh-CN.md) · [weak_prior实验B完整报告](../weak_prior_local_dynamics/report.zh-CN.md) · [weak_prior状态空间](../weak_prior_state_space/report.zh-CN.md)','',
      '[原始实验A](../shared_matched_clinical_velocity/report.zh-CN.md) · [原始实验B](../shared_matched_local_dynamics/report.zh-CN.md) · [配对比较数据](comparison.json)','',
      'weak_prior状态提取及实验A在GPU 2上完成，实验B在GPU 3上完成；模型与ODE计算使用CUDA。数据整理、统计及绘图使用CPU。已有模型训练继续运行。','']
    (OUT/'report.zh-CN.md').write_text('\n'.join(lines))
    asection=['## 10. 与原始 shared_matched 的同任务比较','',
      '| 指标 | 原始模型 | weak_prior |','|---|---:|---:|',
      f"| 年变化率MAE（分/年） | {baseA['mae']:.3f} | {newA['mae']:.3f} |",
      f"| Pearson | {baseA['pearson']:.3f} | {newA['pearson']:.3f} |",
      f"| 上升/下降准确率 | {baseA['up_down_accuracy_nonzero']*100:.2f}% | {newA['up_down_accuracy_nonzero']*100:.2f}% |",'',
      '[查看并排速度场、患者配对区间和实验B比较](../weak_prior_research_AB/report.zh-CN.md)','']
    bsection=['## 8. 与原始 shared_matched 的同任务比较','',
      '| 时长 | 原始MMSE MAE | weak_prior MMSE MAE | 原始调整后D—误差相关 | weak_prior调整后相关 |','|---|---:|---:|---:|---:|']
    for h in ['12','24']:
        a=original['B']['forecast']['horizons'][h];b=weak['B']['forecast']['horizons'][h]
        bsection.append(f"| {h}个月 | {a['mmse_mae_patient_weighted']:.3f} | {b['mmse_mae_patient_weighted']:.3f} | {a['adjusted_sensitivity_error']:.3f} | {b['adjusted_sensitivity_error']:.3f} |")
    bsection+=['','[查看患者配对区间、局部动力学并排图和实验A比较](../weak_prior_research_AB/report.zh-CN.md)','']
    for path,section in [(weak['a']/'report.zh-CN.md',asection),(weak['b']/'report.zh-CN.md',bsection)]:
        text=path.read_text().split(section[0])[0].rstrip()
        path.write_text(text+'\n\n'+'\n'.join(section))
    print(json.dumps({'paired_A_MAE':paired_A,'paired_A_initial_AUC':paired_auc,'paired_B_MAE':paired_B},indent=2))


if __name__=='__main__':main()
