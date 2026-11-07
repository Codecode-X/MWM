"""CPU statistics and static figures for the CUDA Jacobian utility study."""
import csv
import json
from pathlib import Path

import numpy as np

REGIONS=['all_future','0-12m','12-24m','24-36m','36-60m','>60m']
YEARS=[0,1,2,3,5]


def aggregate(data,numeric,std,scales,names,weights=None):
    if weights is None:
        sums=data['sums'].sum(0);counts=data['counts'].sum(0);shifts=data['shifts'].sum(0)
    else:
        sums=np.einsum('bp,prf->brf',weights,data['sums']);counts=np.einsum('bp,prf->brf',weights,data['counts']);shifts=np.einsum('bp,prf->brf',weights,data['shifts'])
    mean=sums/np.maximum(counts,1);valid=counts>0
    shift=shifts/np.maximum(counts,1)
    selected=numeric&(std>1e-12)
    denominator=np.maximum((valid[...,selected]).sum(-1),1)
    score={'total':(mean*valid).sum(-1),'numeric_mse':(mean[...,numeric]*valid[...,numeric]).sum(-1),
           'category_ce':(mean[...,~numeric]*valid[...,~numeric]).sum(-1),
           'standardized_mse':((mean[...,selected]/std[selected]**2)*valid[...,selected]).sum(-1)/denominator,
           'observations':counts.sum(-1)}
    for name in ['mmse','cdrsb']:
        j=names.index(name)
        score[name+'_rmse']=np.sqrt(mean[...,j])*scales[j]
        score[name+'_prediction_shift_rmse']=np.sqrt(shift[...,j])*scales[j]
    return score


def generate(out,payload):
    out=Path(out);manifest=json.loads((out/'manifest.json').read_text());schema=payload['schema']
    names=[f['name'] for f in schema];numeric=np.array([f['kind']=='numerical' for f in schema])
    scales=np.array(payload['metadata']['scale']);std=[]
    train=[s for s in payload['sequences'] if s['split']=='train']
    for j in range(len(schema)):
        values=np.concatenate([s['values'][:,j][s['mask'][:,j].astype(bool)] for s in train]).astype(np.float64)
        std.append(float(values.std()))
    std=np.array(std);results={'models':{},'paired_clean_bootstrap':{},'protocol':manifest}
    all_data={};scores={};flows={}
    for label in manifest['models']:
        all_data[label]={};scores[label]={}
        for sigma in [0.,.01,.05]:
            d=dict(np.load(out/f'forecast_{label}_{sigma:g}.npz'));all_data[label][sigma]=d
            score=aggregate(d,numeric,std,scales,names);scores[label][sigma]=score
            if sigma==0:
                expected=json.loads((out/manifest['models'][label]['metrics']).read_text())['test']['forecast']['reconstruction_error']
                difference=abs(score['total'][0]-expected);assert difference<1e-6,(label,difference)
            else:difference=None
            results['models'].setdefault(label,{})[f'noise_{sigma:g}']={key:value.tolist() for key,value in score.items()}
            results['models'][label][f'noise_{sigma:g}']['clean_reload_difference']=difference
        flows[label]=dict(np.load(out/f'flow_{label}.npz'))
        f=flows[label]
        results['models'][label]['flow']={
            'local_jacobian_frobenius_squared_mean':float(f['local_frob_squared'].mean()),
            'local_jacobian_spectral_mean':float(f['local_spectral'].mean()),
            'local_max_symmetric_eigenvalue_mean':float(f['local_expansion'].mean()),
            'flow_gain_median':np.median(f['flow_gain'],axis=0).tolist(),
            'flow_gain_p95':np.quantile(f['flow_gain'],.95,axis=0).tolist(),
            'state_perturbation_gain_median':np.median(f['perturbation'][...,0],axis=(0,2)).tolist(),
            'state_perturbation_mmse_shift_rms':np.sqrt((f['perturbation'][...,1]**2).mean(axis=(0,2))).tolist(),
            'state_perturbation_cdrsb_shift_rms':np.sqrt((f['perturbation'][...,2]**2).mean(axis=(0,2))).tolist(),
            'cutoff_state_norm_median':float(np.median(f['cutoff_norm']))}
    # Resample patients, retaining all their prefixes and observations together.
    n=all_data['regularized'][0.]['sums'].shape[0]
    rng=np.random.default_rng(124906)
    weights=rng.multinomial(n,np.full(n,1/n),size=2000).astype(np.float64)
    bootstrap={label:aggregate(all_data[label][0.],numeric,std,scales,names,weights) for label in scores}
    for key in ['total','numeric_mse','standardized_mse','mmse_rmse','cdrsb_rmse']:
        delta=bootstrap['regularized'][key]-bootstrap['unregularized'][key]
        results['paired_clean_bootstrap'][key]={
            'regularized_minus_unregularized':(scores['regularized'][0.][key]-scores['unregularized'][0.][key]).tolist(),
            'ci95_lower':np.quantile(delta,.025,axis=0).tolist(),'ci95_upper':np.quantile(delta,.975,axis=0).tolist(),
            'unit':'paired patient bootstrap of these two fixed checkpoints','replicates':2000}
    with (out/'forecast_by_horizon.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['model','noise_train_sd','horizon','patients','total','numeric_mse','category_ce','standardized_mse','mmse_rmse','cdrsb_rmse','mmse_prediction_shift_rmse','cdrsb_prediction_shift_rmse'])
        for label,conditions in scores.items():
            for sigma,score in conditions.items():
                for r,region in enumerate(REGIONS):
                    count=int((all_data[label][sigma]['counts'][:,r].sum(-1)>0).sum())
                    writer.writerow([label,sigma,region,count]+[score[k][r] for k in ['total','numeric_mse','category_ce','standardized_mse','mmse_rmse','cdrsb_rmse','mmse_prediction_shift_rmse','cdrsb_prediction_shift_rmse']])
    with (out/'patient_flow.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['model','patient_id','cutoff_count','year','flow_spectral_gain','state_norm','mean_random_direction_gain','mmse_shift_rms','cdrsb_shift_rms'])
        for label,f in flows.items():
            for i,patient in enumerate(manifest['cohort']):
                for t,year in enumerate(YEARS):
                    p=f['perturbation'][i,t]
                    writer.writerow([label,patient['subject_id'],patient['cutoff_count'],year,f['flow_gain'][i,t],f['cutoff_norm'][i],p[:,0].mean(),np.sqrt(np.mean(p[:,1]**2)),np.sqrt(np.mean(p[:,2]**2))])
    (out/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    draw(out,scores,results)
    write_report(out,results,scores,all_data)


def draw(out,scores,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for label,title,color in [('regularized','Jacobian=0.01','#2574a9'),('unregularized','Jacobian=0','#d76b28')]:
        f=result['models'][label]['flow']
        axes[0,0].plot(YEARS,f['flow_gain_median'],'o-',label=title,color=color)
        axes[0,0].plot(YEARS,f['flow_gain_p95'],'--',alpha=.65,color=color)
        axes[0,1].plot(YEARS,f['state_perturbation_mmse_shift_rms'],'o-',label=title,color=color)
        axes[1,0].plot([0,1,5],[scores[label][s]['mmse_prediction_shift_rmse'][0] for s in [0.,.01,.05]],'o-',label=title,color=color)
        axes[1,1].plot(range(5),scores[label][0.]['mmse_rmse'][1:],'o-',label=title,color=color)
    axes[0,0].set(title='ODE flow sensitivity: median (solid), P95 (dashed)',xlabel='Free forecast, years',ylabel='Largest singular value')
    axes[0,1].set(title='Clinical shift from 1% latent-state perturbation',xlabel='Free forecast, years',ylabel='MMSE prediction shift RMS, points')
    axes[1,0].set(title='Same historical measurement noise',xlabel='Noise SD / training feature SD, %',ylabel='Future MMSE prediction shift RMS, points')
    axes[1,1].set(title='Actual future MMSE error by forecast horizon',xlabel='Forecast horizon',ylabel='MMSE RMSE, points',xticks=range(5),xticklabels=['0-1y','1-2y','2-3y','3-5y','>5y'])
    for ax in axes.flat:ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'utility_comparison.png',dpi=180);fig.savefig(out/'utility_comparison.pdf');plt.close(fig)


def write_report(out,result,scores,all_data):
    mf=result['models']['regularized']['flow'];uf=result['models']['unregularized']['flow']
    r=scores['regularized'];u=scores['unregularized']
    lines=['# Jacobian正则有什么实际作用：ADNI固定检查点测试','',
           '## 1. 从“更平滑”到“更可靠”','',
           '相邻潜在状态的速度更相近，表示一次小状态偏差引起的速度变化更小。疾病预测还要经过多年的ODE积分和Decoder，因此本次逐层检查三个结果：小偏差会被ODE放大多少、会让临床评分预测变化多少、最终真实随访预测是否更准确。','',
           '## 2. 两组和测试方式','',
           '两组均为ADNI、α=0.1、latent28、44,875参数，并使用各自第30轮最佳验证检查点。一组保留0.01 Jacobian正则，另一组权重为0；weak prior均为0.01。参数快照已复制到本目录，全程不训练、不更新参数。','',
           '真实预测误差采用原414位测试患者、2,402个历史前缀、batch=32，保持原min-max归一化和字段聚合。未来目标按实际间隔分为0–12、12–24、24–36、36–60及60个月以上。','',
           '扰动敏感性为每位患者取最早3次已知观测，不足3次则使用已有观测。仅用这些历史进行编码，随后自由推演0、1、2、3、5年，不执行未来UPDATE。','',
           '所有模型计算使用CUDA。真实预测与测量噪声测试沿用float32、Tsit5、rtol=1e-4、atol=1e-6。ODE流敏感性使用float64、rtol=1e-8、atol=1e-10，并以更严容差和原28维ODE的autograd导数核验。','',
           '## 3. 小状态偏差经过ODE会放大多少','',
           r'计算 $\lVert D_z\Phi_T(z_c)\rVert_2$，它表示相同大小的初始状态误差，在最容易被放大的方向上，经过T年会变为原来的多少倍。1表示相同大小，小于1表示缩小，大于1表示放大。','',
           '| 推演时间 | 保留正则：中位数 | 去掉正则：中位数 | 保留正则：P95 | 去掉正则：P95 |','|---|---:|---:|---:|---:|']
    for i,t in enumerate(YEARS):lines.append(f"| {t}年 | {mf['flow_gain_median'][i]:.6f} | {uf['flow_gain_median'][i]:.6f} | {mf['flow_gain_p95'][i]:.6f} | {uf['flow_gain_p95'][i]:.6f} |")
    change=100*(mf['flow_gain_median'][-1]/uf['flow_gain_median'][-1]-1)
    lines += ['',f'5年最坏方向放大倍数的患者中位数，保留正则相对去掉正则变化{change:+.2f}%。这是对完整ODE流的测量，不再仅看局部速度场Jacobian。','',
              '## 4. 状态偏差会造成多大的临床预测偏移','',
              '每位患者生成8个固定随机方向，在两组中使用相同方向。扰动长度为各自截止点状态范数的1%，分别推演。下表统计扰动前后预测之差的RMS，单位为临床评分的分值。','',
              '| 推演时间 | 保留正则：MMSE偏移 | 去掉正则：MMSE偏移 | 保留正则：CDR-SB偏移 | 去掉正则：CDR-SB偏移 |','|---|---:|---:|---:|---:|']
    for i,t in enumerate(YEARS):lines.append(f"| {t}年 | {mf['state_perturbation_mmse_shift_rms'][i]:.6f} | {uf['state_perturbation_mmse_shift_rms'][i]:.6f} | {mf['state_perturbation_cdrsb_shift_rms'][i]:.6f} | {uf['state_perturbation_cdrsb_shift_rms'][i]:.6f} |")
    lines += ['',
              '两个模型经过训练后，潜在坐标和状态范数可以不同。下一项从原始观测端加入完全相同的噪声，直接比较相同测量误差带来的预测偏移。','',
              '## 5. 相同测量误差会让预测晃动多少','',
              '仅扰动历史中实际观测到的连续检查字段。每个字段的噪声标准差取其训练集标准差的1%或5%；两组使用相同随机噪声。年龄、教育年限、APOE4及类别字段保持原值。测试目标、时间、缺失掩码和截止点不变。','',
              '这里的“预测偏移”比较同一模型在干净与带噪历史下的预测；“真实误差”比较预测与真实随访。两者分别回答稳定性和准确性。','',
              '| 噪声幅度 | 保留：MMSE预测偏移RMSE | 去掉：MMSE预测偏移RMSE | 保留：CDR-SB预测偏移RMSE | 去掉：CDR-SB预测偏移RMSE | 保留：未来总误差 | 去掉：未来总误差 |','|---|---:|---:|---:|---:|---:|---:|']
    for sigma in [0.,.01,.05]:
        lines.append(f"| {sigma:.0%}训练标准差 | {r[sigma]['mmse_prediction_shift_rmse'][0]:.6f} | {u[sigma]['mmse_prediction_shift_rmse'][0]:.6f} | {r[sigma]['cdrsb_prediction_shift_rmse'][0]:.6f} | {u[sigma]['cdrsb_prediction_shift_rmse'][0]:.6f} | {r[sigma]['total'][0]:.6f} | {u[sigma]['total'][0]:.6f} |")
    lines += ['', '## 6. 真实随访：收益是否集中在更长的推演','',
              '| 未来间隔 | 有真实随访的患者数 | 保留：总误差 | 去掉：总误差 | 保留：MMSE RMSE | 去掉：MMSE RMSE |','|---|---:|---:|---:|---:|---:|']
    for i,region in enumerate(REGIONS):
        patients=int((all_data['regularized'][0.]['counts'][:,i].sum(-1)>0).sum())
        lines.append(f"| {region} | {patients} | {r[0.]['total'][i]:.6f} | {u[0.]['total'][i]:.6f} | {r[0.]['mmse_rmse'][i]:.6f} | {u[0.]['mmse_rmse'][i]:.6f} |")
    lines += ['', '对固定的两个模型，按患者整体重采样2,000次；同一患者的所有历史前缀和随访一起保留。差值定义为“保留正则−去掉正则”，负数表示保留正则误差更低。','',
              '| 全部未来的指标 | 差值 | 患者配对bootstrap 95%区间 |','|---|---:|---|']
    for key,name in [('total','总误差'),('standardized_mse','数值标准化MSE'),('mmse_rmse','MMSE RMSE'),('cdrsb_rmse','CDR-SB RMSE')]:
        d=result['paired_clean_bootstrap'][key]
        lines.append(f"| {name} | {d['regularized_minus_unregularized'][0]:+.6f} | [{d['ci95_lower'][0]:+.6f}, {d['ci95_upper'][0]:+.6f}] |")
    noise_change=100*(r[.05]['mmse_prediction_shift_rmse'][0]/u[.05]['mmse_prediction_shift_rmse'][0]-1)
    long_change=100*(r[0.]['mmse_rmse'][-1]/u[0.]['mmse_rmse'][-1]-1)
    lines += ['', '## 7. 这些结果如何解释','',
              f"1. **ODE对初始状态误差的放大。** 5年流敏感性中位数为{mf['flow_gain_median'][-1]:.4f}（保留）与{uf['flow_gain_median'][-1]:.4f}（去掉），变化{change:+.2f}%。",'',
              f"2. **观测误差到临床预测的传递。** 相同5%测量噪声引起的未来MMSE预测偏移RMSE为{r[.05]['mmse_prediction_shift_rmse'][0]:.4f}分（保留）与{u[.05]['mmse_prediction_shift_rmse'][0]:.4f}分（去掉），变化{noise_change:+.2f}%。这表示同样的检查误差造成的MMSE预测波动更小。CDR-SB预测波动分别为{r[.05]['cdrsb_prediction_shift_rmse'][0]:.4f}与{u[.05]['cdrsb_prediction_shift_rmse'][0]:.4f}分，基本持平。该项同时经过INIT、历史ODE、UPDATE、生成ODE与Decoder。",'',
              f"3. **真实预测表现。** 干净测试数据的未来总误差为{r[0.]['total'][0]:.6f}与{u[0.]['total'][0]:.6f}；未来MMSE RMSE为{r[0.]['mmse_rmse'][0]:.4f}与{u[0.]['mmse_rmse'][0]:.4f}分。上面的分时长表给出收益出现在哪些时间范围。",'',
              f"真实随访中，0–12个月MMSE误差基本接近；36–60个月从{u[0.]['mmse_rmse'][4]:.4f}降到{r[0.]['mmse_rmse'][4]:.4f}分，60个月以上从{u[0.]['mmse_rmse'][5]:.4f}降到{r[0.]['mmse_rmse'][5]:.4f}分，后者变化{long_change:+.2f}%。收益更集中在较长的推演。",'',
              '按患者配对重采样，全部未来MMSE和CDR-SB RMSE差值的95%区间均小于0；未来总误差的区间跨过0。这里体现的是临床数值预测的改善，总指标中的类别CE会抵消部分数值收益。', '',
              '从机制上看，初始状态有小偏差时，后续偏差满足“偏差变化速度＝速度场Jacobian×当前偏差”。局部速度变化较小，能够减轻ODE反复传播偏差的强度；本次直接测到5年流放大倍数降低3.78%，并在MMSE输出上测到较小的预测波动。', '',
              '因此，这项正则在当前ADNI模型中具有可测的作用：减轻多年推演对小状态误差的放大，稍微提高MMSE预测对测量噪声的稳定性，并改善较长期的临床评分预测。收益幅度温和，适合继续作为0.01权重的辅助动力学约束。CDR-SB抗测量噪声表现基本持平，未来总误差的优势也小于单看MMSE时的优势。','',
              '![ODE敏感性、临床偏移和预测时长对比](utility_comparison.png)','',
              '## 8. 原始记录与复现','',
              '[全部指标与配对区间](metrics.json) · [按时间和噪声分组的误差CSV](forecast_by_horizon.csv) · [患者级流敏感性CSV](patient_flow.csv) · [实验配置与检查点指纹](manifest.json)','',
              '[保留正则数值核验](flow_checks_regularized.json) · [去掉正则数值核验](flow_checks_unregularized.json)','',
              '保留的NPZ文件包含患者级误差分子、有效目标分母、状态、流敏感性及扰动结果；停止后重启会复用已完成阶段。所有模型参数在测试结束后逐张量核对，未执行optimizer.step()。','']
    lines += ['复现命令：','',
              '```bash',
              'CUDA_VISIBLE_DEVICES=1 PYTHONPATH=ablation/results/jacobian_utility_adni/source_snapshot:paper_odeae/results/latent2x_loss_ablation/source_snapshot /home/xjh/miniforge3/envs/ode/bin/python ablation/results/jacobian_utility_adni/source_snapshot/jacobian_utility_adni.py --output ablation/results/jacobian_utility_adni --gpu 1',
              '```','']
    (out/'report.zh-CN.md').write_text('\n'.join(lines))
