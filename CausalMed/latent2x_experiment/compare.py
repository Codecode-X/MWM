"""Gamma=1 comparison from sealed predictions and offline scores; no model execution."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import time

import pandas as pd

from .report import read
from .study import ROOT,atomic,identity


def collect(paths):
    rows=[]
    for name,folder,gpu in paths:
        queue=read(folder/'queue.json',{});run=folder/'gamma_1/model';history=read(run/'history.json',[]);trained=read(run/'training_complete.json')
        best=min(history,key=lambda row:row['validation']['overall']['reconstruction_error']) if history else None
        rows.append({'name':name,'folder':str(folder),'gpu':gpu,'status':queue.get('status','waiting'),
                     'phase':queue.get('jobs',{}).get('gamma_1',{}).get('phase','waiting'),'epochs':len(history),
                     'best_epoch':trained['best_epoch'] if trained else best['epoch'] if best else None,
                     'validation':trained['validation'] if trained else best['validation'] if best else None,
                     'hardware':read(run/'hardware.json',{}),'with_actions':read(folder/'gamma_1/evaluation/with_actions/metrics.json'),
                     'action_off':read(folder/'gamma_1/evaluation/action_off/metrics.json')})
    return rows


def case_plots(out,rows):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    predictions=[]
    for row in rows:
        folder=Path(row['folder'])/'gamma_1/model/with_actions';frame=pd.read_parquet(folder/'predictions.parquet')
        sealed=read(folder/'sealed.json')
        from .study import digest
        if digest(folder/'predictions.parquet')!=sealed['sha256']['predictions.parquet']:raise AssertionError('Prediction seal changed')
        frame['relative_time']=frame.relative_time.round(9);predictions.append(frame)
    keys=['patient_id','policy_id','relative_time']
    if set(map(tuple,predictions[0][keys].values))!=set(map(tuple,predictions[1][keys].values)):raise ValueError('Models must predict identical test queries')
    truth=pd.read_parquet(ROOT/'CausalMed/outputs/level3_suite/gamma_1/oracle_private/cf_trajectories.parquet');truth['relative_time']=truth.relative_time.round(9)
    ids=sorted(predictions[0].patient_id.unique())[:3];colors=['#4477AA','#EE6677','#228833','#CCBB44','#AA3377'];names=['Never','Always','Delayed','Aggressive','Conservative']
    fig,axes=plt.subplots(3,3,figsize=(16,10),squeeze=False)
    for i,pid in enumerate(ids):
        parts=[truth[truth.patient_id==pid]]+[frame[frame.patient_id==pid] for frame in predictions]
        limits=[]
        for j,part in enumerate(parts):
            field='Y_cf_true' if j==0 else 'Y_pred';limits.extend([float(part[field].min()),float(part[field].max())])
            for policy,color in enumerate(colors):
                curve=part[part.policy_id==policy].sort_values('relative_time');axes[i,j].plot(curve.relative_time,curve[field],color=color,label=names[policy])
            axes[i,j].set(title=f"{pid}: {'ground truth' if j==0 else rows[j-1]['name']}",xlabel='Years after fork',ylabel='Outcome Y');axes[i,j].grid(alpha=.15)
        lo,hi=min(limits),max(limits);margin=max((hi-lo)*.07,1.)
        for ax in axes[i]:ax.set_ylim(lo-margin,hi+margin)
    axes[0,2].legend(fontsize=8,ncol=2);fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(out/f'parallel_cases.{ext}',dpi=180)
    plt.close(fig)
    true=truth[truth.relative_time==3.].pivot(index='patient_id',columns='policy_id',values='Y_cf_true');tau=true[1]-true[0]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,row,frame in zip(axes,rows,predictions):
        p=frame[frame.relative_time==3.].pivot(index='patient_id',columns='policy_id',values='Y_pred');estimate=(p[1]-p[0]).reindex(tau.index)
        ax.scatter(tau,estimate,s=7,alpha=.35,color='#4477AA');lo=min(tau.min(),estimate.min());hi=max(tau.max(),estimate.max());ax.plot([lo,hi],[lo,hi],'--',color='gray')
        ax.set(title=row['name'],xlabel='True 3-year effect: Always - Never',ylabel='Predicted 3-year effect');ax.grid(alpha=.15)
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(out/f'parallel_individual_effects.{ext}',dpi=180)
    plt.close(fig)
    atomic(out/'case_metadata.json',{'patients':ids,'selection':'first three public test IDs','all_test_query_keys_identical':True,'model_execution_performed':False})


def report(out,rows):
    lines=['# CausalMed γ=1：latent2x both 与 ODE-AE 的反事实推断比较','',
           '## 1. 同一任务下比较什么','',
           '两种模型读取相同的10,000位训练患者、2,000位验证患者和2,000位测试患者。观测字段为Y、X1、X2，静态资料为c，归一化参数仅拟合训练集。测试在首次达到3年的检查后分叉，执行相同五策略并推演至8年边界。','',
           'latent2x both在GPU 3继续原训练；新增ODE-AE在空闲GPU 1单卡训练。本轮反事实实验只使用γ=1。','',
           '## 2. 模型结构与训练设置','',
           '| 模型 | 潜在表示 | 历史编码 | 生成ODE | 参数量 | 正则 |','|---|---|---|---|---:|---|',
           '| latent2x both | 28维确定性状态 | 正向逐次更新 | 编码与生成共享，宽188 | 31,314 | 0.01 prior＋0.01 Jacobian＋0.02 ODE L2 |',
           '| ODE-AE | 6维Gaussian潜在状态 | 12维反向编码，宽16 | 独立6维生成ODE，宽16 | 1,843 | 1×KL＋0.02两场ODE L2 |','',
           '治疗切换都采用“新治疗条件输出减去旧治疗条件输出”的残差位移，同一治疗保持时位移为零。ODE-AE的编码状态为12维、生成状态为6维，生成阶段增加一个宽16的治疗校正模块；观测编码仍使用原反向UPDATE。','',
           '两种模型使用相同的seed 0、batch 128、Adam学习率0.003及衰减、输入dropout 0.1、300轮上限、50轮下限和40轮早停耐心。验证整体标准化事实误差选模，最终使用最佳检查点；ODE-AE测试使用后验均值。','',
           '## 3. 当前训练进度','',
           '| 模型 | GPU | 阶段 | 已训练 / 最佳轮次 | 验证整体 | 验证未来Y RMSE |','|---|---:|---|---:|---:|---:|']
    for row in rows:
        v=row['validation'];lines.append('| '+' | '.join([row['name'],str(row['gpu']),row['status']+' / '+row['phase'],f"{row['epochs']} / {row['best_epoch'] or '—'}",f"{v['overall']['reconstruction_error']:.5f}" if v else '—',f"{v['future_raw_rmse'][0]:.5f}" if v else '—'])+' |')
    lines+=['','## 4. 反事实轨迹与终点误差','',
            '轨迹CF-RMSE使用配对的无噪声Y真值，终点对应精确1/2/3年。动作关闭对照使用同一检查点和分叉状态，只关闭未来治疗校正。','',
            '| 模型 | 版本 | 全轨迹CF-RMSE | 1年 | 2年 | 3年 |','|---|---|---:|---:|---:|---:|']
    for row in rows:
        for key,label in [('with_actions','动作输入'),('action_off','关闭未来动作')]:
            m=row[key];values=[f"{m['cf_rmse_trajectory']:.5f}"]+[f"{m['horizons'][str(t)]['cf_rmse']:.5f}" for t in [1.,2.,3.]] if m else ['—']*4
            lines.append('| '+' | '.join([row['name'],label]+values)+' |')
    lines+=['','## 5. 个体治疗效应与策略选择','',
            '下表将Always与Never比较。策略效用为Y−3×治疗年数，预测成本来自模型自己生成的治疗区间。','',
            '| 模型 | 年限 | sqrt(PEHE) | ATE Error | 真实ATE | 预测ATE | Policy Regret | 最优策略准确率 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if not row['with_actions']:continue
        for t,h in row['with_actions']['horizons'].items():
            e=h['effects_vs_never']['1'];lines.append(f"| {row['name']} | {t} | {e['sqrt_pehe']:.5f} | {e['ate_error']:.5f} | {e['true_ate']:.5f} | {e['predicted_ate']:.5f} | {h['policy_regret']:.5f} | {100*h['optimal_policy_accuracy']:.2f}% |")
    lines+=['','## 6. 同一患者的平行轨迹','',
            '完整评分后，取测试名单按编号排序的前三位患者，按“真实轨迹—latent2x both—ODE-AE”平行展示。每行使用相同纵轴范围。','']
    if (out/'parallel_cases.png').exists():lines+=['![五策略反事实平行比较](parallel_cases.png)','','![三年个体治疗效应平行比较](parallel_individual_effects.png)','']
    lines+=['','## 7. 输出与来源','',
            '[方法与实验说明](../../latent2x_experiment/README.zh-CN.md) · [结构化比较数据](comparison.json)','']
    for row in rows:
        p=Path(row['folder']);lines.append(f"- [{row['name']}实验记录]({os.path.relpath(p/'summary.md',out)})")
    lines+=['','模型训练和推演仅挂载公开数据。每个模型先封存预测与自身治疗区间，再由同一个独立评分器读取配对真值。SwanLab记录两条训练任务，hardware.json保存实际GPU和张量设备。','']
    (out/'summary.md').write_text('\n'.join(lines));atomic(out/'comparison.json',{'gamma':1,'models':rows})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--both-output',default=str(ROOT/'CausalMed/outputs/latent2x_both_cuda3'));parser.add_argument('--odeae-output',default=str(ROOT/'CausalMed/outputs/ode_ae_gamma1_cuda1'));parser.add_argument('--output',default=str(ROOT/'CausalMed/outputs/gamma1_model_comparison'));args=parser.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'monitoring','started_at':datetime.now(timezone.utc).isoformat()};atomic(out/'queue.json',state)
    paths=[('latent2x both',Path(args.both_output).resolve(),3),('ODE-AE',Path(args.odeae_output).resolve(),1)]
    while True:
        rows=collect(paths)
        if all(row['with_actions'] and row['action_off'] for row in rows) and not (out/'case_metadata.json').exists():case_plots(out,rows)
        report(out,rows)
        if all(row['status'] in ['complete','failed'] for row in rows):break
        time.sleep(20)
    state.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat());atomic(out/'queue.json',state)


if __name__=='__main__':main()
