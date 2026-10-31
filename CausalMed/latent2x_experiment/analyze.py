"""Separate post-submission statistics and plots; never constructs or runs a model."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .report import read

ROOT=Path(__file__).resolve().parents[2]
COLORS=['#4477AA','#EE6677','#228833','#CCBB44','#AA3377']
NAMES=['Never','Always','Delayed','Aggressive','Conservative']


def atomic(path,data):
    temporary=path.with_suffix(path.suffix+'.partial');temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def plots(out,key,truth_directory=None):
    folder=out/key;model=folder/'model'
    truth_dir=Path(truth_directory) if truth_directory else ROOT/'CausalMed/outputs/level3_suite'/key/'oracle_private'
    for name in ['with_actions','action_off']:
        sealed=read(model/name/'sealed.json')
        for filename,sha in sealed['sha256'].items():
            if hashlib.sha256((model/name/filename).read_bytes()).hexdigest()!=sha:raise RuntimeError('Post-submission file changed')
    pred=pd.read_parquet(model/'with_actions/predictions.parquet');truth=pd.read_parquet(truth_dir/'cf_trajectories.parquet')
    for frame in [pred,truth]:frame['relative_time']=frame.relative_time.round(9)
    paired=truth.merge(pred,on=['patient_id','policy_id','relative_time'],how='outer',indicator=True,validate='one_to_one')
    if not paired['_merge'].eq('both').all():raise ValueError('Plot rows must exactly match sealed prediction coverage')
    selected=sorted(pred.patient_id.unique())[:3]
    curves=paired[paired.patient_id.isin(selected)].copy();curves.to_csv(folder/'evaluation/three_patient_curves.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,2,figsize=(12,10),squeeze=False)
    for i,pid in enumerate(selected):
        part=curves[curves.patient_id==pid];lo=min(part.Y_cf_true.min(),part.Y_pred.min());hi=max(part.Y_cf_true.max(),part.Y_pred.max());margin=max((hi-lo)*.08,1.)
        for p,color in enumerate(COLORS):
            g=part[part.policy_id==p].sort_values('relative_time')
            axes[i,0].plot(g.relative_time,g.Y_cf_true,color=color,label=NAMES[p]);axes[i,1].plot(g.relative_time,g.Y_pred,color=color,label=NAMES[p])
        for ax in axes[i]:ax.set(xlabel='Years after fork',ylabel='Outcome Y',ylim=(lo-margin,hi+margin));ax.grid(alpha=.15)
        axes[i,0].set_title(f'{pid}: paired ground truth');axes[i,1].set_title(f'{pid}: model predictions')
    axes[0,1].legend(fontsize=8,ncol=2);fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(folder/'evaluation'/f'three_patient_cases.{ext}',dpi=180)
    plt.close(fig)
    endpoint=paired[paired.relative_time==3.].copy();true=endpoint.pivot(index='patient_id',columns='policy_id',values='Y_cf_true')
    estimated=endpoint.pivot(index='patient_id',columns='policy_id',values='Y_pred');tau=true[1]-true[0];estimate=estimated[1]-estimated[0]
    correlation=float(np.corrcoef(tau,estimate)[0,1]) if min(float(tau.std()),float(estimate.std()))>1e-10 else None
    fig,ax=plt.subplots(figsize=(6.5,5.5));ax.scatter(tau,estimate,s=8,alpha=.35,color='#4477AA')
    lo=min(float(tau.min()),float(estimate.min()));hi=max(float(tau.max()),float(estimate.max()));ax.plot([lo,hi],[lo,hi],'--',color='gray')
    ax.set(xlabel='True 3-year effect: Always - Never',ylabel='Predicted 3-year effect',title=f'{key}: individual treatment effects');ax.grid(alpha=.15);fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(folder/'evaluation'/f'individual_effects.{ext}',dpi=180)
    plt.close(fig)
    pd.DataFrame({'patient_id':tau.index,'true_3y_effect':tau.values,'predicted_3y_effect':estimate.reindex(tau.index).values}).to_csv(folder/'evaluation/individual_effects.csv',index=False)
    pid=selected[0];part=pred[pred.patient_id==pid];intervals=pd.read_parquet(model/'with_actions/predicted_treatments.parquet');intervals=intervals[intervals.patient_id==pid]
    specs=read(ROOT/'CausalMed/outputs/level3_suite'/key/'public/policy_specs.json')
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for p,color in enumerate(COLORS):
        g=part[part.policy_id==p].sort_values('relative_time');a=intervals[intervals.policy_id==p]
        duration=[float((np.maximum(0,np.minimum(a.t_end,t)-np.maximum(a.t_start,0))*a.treatment).sum()) for t in g.relative_time]
        axes[0].plot(g.relative_time,g.Y_pred,color=color,label=NAMES[p]);axes[1].plot(g.relative_time,g.X2_pred,color=color)
        axes[2].plot(g.relative_time,duration,color=color)
    axes[1].axhline(specs['thresholds']['3'],color='gray',linestyle='--',label='q50');axes[1].axhline(specs['thresholds']['4'],color='black',linestyle=':',label='q80')
    axes[0].set(ylabel='Predicted Y',title=f'{pid}: predicted outcome, risk and treatment');axes[1].set(ylabel='Predicted X2');axes[2].set(ylabel='Cumulative years treated',xlabel='Years after fork')
    axes[0].legend(ncol=3,fontsize=8);axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.15)
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(folder/'evaluation'/f'policy_feedback_case.{ext}',dpi=180)
    plt.close(fig)
    audit={'status':'complete','source':'sealed model predictions and independent oracle outcomes after scoring','patients':int(pred.patient_id.nunique()),
           'case_selection':'first three patient IDs in sorted public test roster','cases':selected,'three_year_ite_pearson':correlation,
           'model_execution_performed':False,'plotted_at':datetime.now(timezone.utc).isoformat()}
    atomic(folder/'evaluation/analysis.json',audit)


def report(out):
    lines=['# CausalMed：反事实预测结果与病例分析','',
           '先查看整体误差和个体效应，再沿同一患者的五条分支观察预测曲线、风险和治疗时长。图表由封存的GPU预测与独立评分真值生成。','',
           '[训练进度与完整指标表](summary.md) · [模型与测试方法](../../latent2x_experiment/README.zh-CN.md)','']
    available=[]
    for key in ['gamma_1','gamma_0','gamma_0.5','gamma_2']:
        m=read(out/key/'evaluation/with_actions/metrics.json');off=read(out/key/'evaluation/action_off/metrics.json');audit=read(out/key/'evaluation/analysis.json')
        if not (m and off and audit):continue
        available.append((key,m,off))
        h=m['horizons']['3.0'];other=off['horizons']['3.0']
        lines += [f"## gamma={key[6:]}：3年反事实能力",'',
                  '| 项目 | 动作输入 | 关闭未来动作 |','|---|---:|---:|',
                  f"| 全轨迹CF-RMSE | {m['cf_rmse_trajectory']:.5f} | {off['cf_rmse_trajectory']:.5f} |",
                  f"| 3年CF-RMSE | {h['cf_rmse']:.5f} | {other['cf_rmse']:.5f} |",
                  f"| Always相对Never的sqrt(PEHE) | {h['effects_vs_never']['1']['sqrt_pehe']:.5f} | {other['effects_vs_never']['1']['sqrt_pehe']:.5f} |",
                  f"| ATE Error | {h['effects_vs_never']['1']['ate_error']:.5f} | {other['effects_vs_never']['1']['ate_error']:.5f} |",
                  f"| 策略Regret | {h['policy_regret']:.5f} | {other['policy_regret']:.5f} |",
                  f"| 最优策略准确率 | {h['optimal_policy_accuracy']*100:.2f}% | {other['optimal_policy_accuracy']*100:.2f}% |",'',
                  '病例取测试名单按编号排序的前三位。每行左右是同一患者，颜色对应相同策略，纵轴范围在该行内保持一致。','',
                  f'![真实和预测反事实曲线]({key}/evaluation/three_patient_cases.png)','',
                  '下面逐患者比较“Always减去Never”的3年治疗效应，虚线表示预测与真值相同。','',
                  f'![个体治疗效应]({key}/evaluation/individual_effects.png)','',
                  '同一病例的预测结局、预测风险与累计治疗年数如下。动态策略根据每次未来检查时刻的风险预测更新动作。','',
                  f'![风险与治疗闭环]({key}/evaluation/policy_feedback_case.png)','']
        if audit['three_year_ite_pearson'] is not None:lines.append(f"3年个体治疗效应的Pearson相关为 {audit['three_year_ite_pearson']:.4f}。\n")
    if not available:lines+=['当前正在训练。每组完整反事实预测和配对评分完成后，这里会自动加入该组的结果与病例图。','']
    (out/'analysis.zh-CN.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();out=Path(args.output).resolve()
    state={'pid':os.getpid(),'status':'waiting_for_sealed_evaluations','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};atomic(out/'analysis_queue.json',state)
    while True:
        for key in ['gamma_1','gamma_0','gamma_0.5','gamma_2']:
            if (out/key/'evaluation/action_off/metrics.json').exists() and not (out/key/'evaluation/analysis.json').exists():plots(out,key)
        report(out);queue=read(out/'queue.json',{})
        if queue.get('status') in ['complete','failed']:break
        time.sleep(20)
    state.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat());atomic(out/'analysis_queue.json',state)


if __name__=='__main__':main()
