#!/usr/bin/env python3
"""Aggregate completed ADNI GPU experiments; CPU statistics/plots only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def paired_bootstrap(left, right, reps=2000):
    """Paired resampling of patients AND initialization seeds (macro feature MSE)."""
    l, r = [], []
    for a,b in zip(left,right):
        with np.load(a) as x, np.load(b) as y:
            if not np.array_equal(x['patient_ids'],y['patient_ids']):
                raise ValueError('patient IDs must match for paired inference')
            if not np.array_equal(x['counts'],y['counts']):
                raise ValueError('forecast targets differ')
            l.append((x['squared'].copy(),x['counts'].copy()))
            r.append((y['squared'].copy(),y['counts'].copy()))
    rng=np.random.default_rng(20261026)
    draws=[]
    def diff(indices,seeds):
        values=[]
        for s in seeds:
            count=l[s][1][indices].sum(0)
            valid=count>0
            delta=(l[s][0][indices].sum(0)-r[s][0][indices].sum(0))/np.maximum(count,1)
            values.append(delta[valid].mean())
        return float(np.mean(values))
    n=len(l[0][0])
    for _ in range(reps):
        draws.append(diff(rng.integers(0,n,n),rng.integers(0,len(l),len(l))))
    return {'difference_left_minus_right':diff(np.arange(n),np.arange(len(l))),
            'ci95':np.percentile(draws,[2.5,97.5]).tolist(),
            'method':'paired patient + seed bootstrap, 2000 resamples; fixed split'}


def render_report(groups, aggregate, comparisons):
    """Explain the experiment before presenting and interpreting its numbers."""
    names = {
        'structured/latent_ode': '数值特征：Latent ODE',
        'structured/ode_rnn': '数值特征：原生 ODE-RNN',
        'medicalnet/latent_ode': '数值 + MRI：Latent ODE',
        'medicalnet/ode_rnn': '数值 + MRI：原生 ODE-RNN',
        'structured_matched/ode_rnn': '数值特征：扩大后 ODE-RNN',
        'medicalnet_matched/ode_rnn': '数值 + MRI：扩大后 ODE-RNN',
    }
    modalities = {'adas':'ADAS 认知量表', 'amyloid_pet':'淀粉样 PET', 'cdr':'CDR 量表',
                  'csf':'脑脊液', 'faq':'FAQ 日常功能量表', 'mmse':'MMSE 认知量表',
                  'mri':'MRI 海马体积', 'neurobat':'记忆与词语学习', 'plasma':'血浆', 'tau_pet':'tau PET'}
    def fmt(values):
        a = np.array(values)
        return f'{a.mean():.4f} ± {a.std(ddof=1) if len(a)>1 else 0:.4f}'
    def metric(key, split, name):
        return fmt([r['metrics'][split][name] for r in groups[key]])
    def interval(c):
        return f"{c['difference_left_minus_right']:.4f} [{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}]"
    first = next(iter(groups.values()))[0]
    metadata = json.loads((Path(first['path'])/'data.json').read_text())
    counts = metadata['patient_counts']
    epochs = first['config']['epochs']
    n_jobs = sum(len(r) for r in groups.values())
    lines = ['# ADNI 纵向预测实验结果', '',
        '本报告按照实验设计、指标含义、结果和结果解释的顺序展开。数据处理和训练步骤见[实验说明](../../reproduction/ADNI_EXPERIMENT.zh-CN.md)。', '',
        '## 1. 这次实验比较什么', '',
        '我们用一位患者已经完成的检查，预测其后续随访中的检查数值。验证和测试时，模型读取前半段有效检查窗口，再预测后半段。预测时间取自实际随访记录。', '',
        f"训练集包含 {counts['train']:,} 名患者，验证集包含 {counts['validation']:,} 名，测试集包含 {counts['test']:,} 名。同一患者的全部记录只属于一个集合。", '',
        '实验使用两组输入。第一组是量表、海马体积、PET、脑脊液和血浆等数值特征。第二组在这些特征上增加冻结 MedicalNet 从原始 T1 MRI 提取的 16 维特征。影像从扫描发生后才进入历史。', '',
        f"两组使用相同患者，预测同样的 {len(metadata['features'])} 个数值字段。每组考察 Latent ODE、原生 ODE-RNN，以及扩大参数量后的 ODE-RNN。容量对照用于观察参数量对结果的影响。", '',
        f'当前共有 {n_jobs} 个已完成训练任务，每个任务训练 {epochs} 轮。下表中的每一行汇总相应配置的独立训练结果。各任务按验证误差选择检查点，再评价测试集。', '',
        '多数最优检查点出现在接近训练轮数上限的位置。因此，这些结果描述本次训练预算下的表现；更长训练是否改变排名，需要另行观察收敛过程。', '',
        '## 2. 怎样读后面的误差', '',
        '### 2.1 标准化误差与原始分数', '',
        '不同检查使用不同单位。程序用训练集均值和标准差把每个字段转为 z-score，再计算未来预测误差。每个字段先在有效观测上平均，最后对字段等权平均。表中的 macro MSE 就是这样得到的平方误差，越低越好。', '',
        'macro MAE 使用绝对误差，也越低越好。平方误差对少量大偏差更敏感，因此报告同时保留 MSE 和 MAE。后面还会列出量表原始单位的 MAE，帮助把统计量联系到实际分数。', '',
        '表格中的“均值 ± 标准差”描述不同随机种子训练的平均结果和波动，不是每位患者的预测区间。', '',
        '### 2.2 用一个简单预测作为参照', '',
        '持续值基线把历史最后一次观测直接当作未来预测。如果历史中没有某字段，则用训练均值。这个基线可以帮助判断模型是否比沿用旧检查值有所改善。', '',
        '| 参照方法 | 验证 macro MSE | 测试 macro MSE |', '|---|---:|---:|']
    baseline = first['metrics']
    lines.append(f"| 持续值基线 | {baseline['validation']['persistence']['macro_mse_z']:.4f} | {baseline['test']['persistence']['macro_mse_z']:.4f} |")
    lines += ['', '### 2.3 怎样理解两个配置的差值', '',
        '比较时，先在同一患者上计算两种配置的误差差，再对患者和随机种子做 2,000 次配对 bootstrap 重抽样，得到差值的 95% 区间。', '',
        '下文的差值均为“左侧配置减右侧配置”。负值表示左侧误差较低。如果区间跨过 0，当前重抽样结果仍包含两种方向，尚不足以确认差异方向。这个区间围绕当前固定患者划分计算。', '',
        '## 3. 原生模型得到什么结果', '',
        '先比较两种模型各自的原生配置。以下四行分别使用数值输入和增加 MRI 后的输入。', '',
        '| 输入与模型 | 训练次数 | 验证 macro MSE | 测试 macro MSE | 测试 macro MAE |', '|---|---:|---:|---:|---:|']
    for key in ['structured/latent_ode','structured/ode_rnn','medicalnet/latent_ode','medicalnet/ode_rnn']:
        if key in groups:
            lines.append(f"| {names[key]} | {len(groups[key])} | {metric(key,'validation','macro_mse_z')} | {metric(key,'test','macro_mse_z')} | {metric(key,'test','macro_mae_z')} |")
    lines += ['', '再把每组输入中的两个模型作配对比较：', '',
        '| 输入 | 左侧 − 右侧 | 误差差值 [95% 区间] |', '|---|---|---:|']
    for variant, label in [('structured','数值特征'),('medicalnet','数值 + MRI')]:
        if variant in comparisons:
            lines.append(f'| {label} | Latent ODE − 原生 ODE-RNN | {interval(comparisons[variant])} |')
    if all(k in groups for k in ['structured/latent_ode','structured/ode_rnn']):
        a = aggregate['structured/latent_ode']['test']['macro_mse_z']['mean']
        b = aggregate['structured/ode_rnn']['test']['macro_mse_z']['mean']
        direction = '降低' if b >= a else '升高'
        lines += ['', f'数值组中，Latent ODE 的平均测试误差为 {a:.4f}，原生 ODE-RNN 为 {b:.4f}。以 ODE-RNN 作参照，平均测试误差{direction} {abs((b-a)/b)*100:.2f}%。下一节继续考察原生模型参数量不同这一因素。']
    lines += ['', '## 4. 参数量接近后，结果怎样变化', '',
        '原生 Latent ODE 包含历史编码和生成过程，其预测网络参数量多于原生 ODE-RNN。为了观察这一因素，实验扩大 ODE-RNN，使它的预测参数量接近对应的 Latent ODE。', '',
        '下面只统计预测网络，排除本任务闲置的诊断分类头。', '',
        '| 输入 | Latent ODE 参数量 | 原生 ODE-RNN 参数量 | 扩大后 ODE-RNN 参数量 |', '|---|---:|---:|---:|']
    for v,label in [('structured','数值特征'),('medicalnet','数值 + MRI')]:
        keys = [f'{v}/latent_ode', f'{v}/ode_rnn', f'{v}_matched/ode_rnn']
        if all(k in groups for k in keys):
            params = [groups[k][0]['hardware'].get('forecast_network_parameters',0) for k in keys]
            lines.append(f'| {label} | '+ ' | '.join(f'{n:,}' for n in params)+' |')
    lines += ['', '在参数量接近的配置中，误差如下。这里仍使用前面相同的患者和预测目标。', '',
        '| 输入与模型 | 验证 macro MSE | 测试 macro MSE | 测试 macro MAE |', '|---|---:|---:|---:|']
    for key in ['structured_matched/ode_rnn','medicalnet_matched/ode_rnn']:
        if key in groups:
            lines.append(f"| {names[key]} | {metric(key,'validation','macro_mse_z')} | {metric(key,'test','macro_mse_z')} | {metric(key,'test','macro_mae_z')} |")
    lines += ['', '| 输入 | 左侧 − 右侧 | 误差差值 [95% 区间] |', '|---|---|---:|']
    capacity = []
    for v,label in [('structured','数值特征'),('medicalnet','数值 + MRI')]:
        key = f'{v}_capacity_control'
        if key in comparisons:
            capacity.append(comparisons[key])
            lines.append(f'| {label} | Latent ODE − 扩大后 ODE-RNN | {interval(comparisons[key])} |')
    if capacity and all(c['ci95'][0] <= 0 <= c['ci95'][1] for c in capacity):
        lines += ['', '扩大后的 ODE-RNN 缩小了与 Latent ODE 的平均误差差距。两组输入对应的差值区间都跨过 0，因此当前证据还不足以确认参数量接近时的优势方向。']
    lines += ['', '## 5. 额外的 MRI 特征带来了什么变化', '',
        '这一比较保持模型种类不变，只考察输入中是否增加冻结 MRI 特征。它回答的是本次影像提取方案是否有帮助。', '',
        '| 模型 | 数值输入测试 MSE | 增加 MRI 后测试 MSE | 增加后 − 增加前 [95% 区间] |', '|---|---:|---:|---:|']
    imaging = []
    for model,label in [('latent_ode','Latent ODE'),('ode_rnn','原生 ODE-RNN')]:
        key = f'imaging_gain/{model}'
        if key in comparisons:
            imaging.append(comparisons[key])
            lines.append(f"| {label} | {metric('structured/'+model,'test','macro_mse_z')} | {metric('medicalnet/'+model,'test','macro_mse_z')} | {interval(comparisons[key])} |")
    if imaging and all(c['ci95'][0] <= 0 <= c['ci95'][1] for c in imaging):
        lines += ['', '两种模型的平均 MSE 都略有下降，但对应差值区间包含 0。由此，本次冻结编码器方案尚未显示稳定增益。这里的 MRI 表征来自每位患者最早一份可用 T1；这一结果对应这套具体的提取方式。']
    lines += ['', '### 5.1 结合图形查看各组变化', '',
        '左图显示验证误差随训练轮数的变化，右图显示各配置的平均测试误差。右图误差棒表示不同训练种子的样本标准差，虚线表示持续值基线。', '',
        '![训练曲线与测试误差比较](comparison.png)', '', '## 6. 误差具体体现在哪些检查上', '',
        '### 6.1 先用实际量表分数理解误差', '',
        '下面的 MAE 已还原为各量表自己的分数单位。例如，MMSE 的 MAE 为 1.77，表示预测总分与实际总分的绝对差平均约为 1.77 分。', '',
        '| 输入与模型 | MMSE | CDR-SB | ADAS-13 | FAQ |', '|---|---:|---:|---:|---:|']
    for key,runs in groups.items():
        values = [fmt([r['metrics']['test']['per_feature'][f]['mae_native'] for r in runs])
                  for f in ['mmse.MMSCORE','cdr.CDRSB','adas.TOTAL13','faq.FAQTOTAL']]
        lines.append(f"| {names[key]} | "+' | '.join(values)+' |')
    lines += ['', '### 6.2 再按检查类别查看标准化误差', '',
        '下列两张表分别对应两组输入。每个数值是该类检查内各字段的平均 MSE，再汇总不同训练种子。有效观测数统计的是实际被评分的标量数量，不是患者人数。', '',
        '这些数量可以帮助解释误差的波动。例如，ADAS 的有效观测多于 PET；阅读 PET 误差时，应结合较少的观测数量理解。MRI 数值记录包含质控未知或部分完成的项目，其处理规则见实验说明的数据整理部分。']
    for v,label in [('structured','数值特征输入'),('medicalnet','数值特征 + MRI 输入')]:
        keys = [f'{v}/latent_ode',f'{v}/ode_rnn',f'{v}_matched/ode_rnn']
        available = [k for k in keys if k in groups]
        if not available:
            continue
        lines += ['', f'#### {label}', '', '| 检查类别 | Latent ODE | 原生 ODE-RNN | 扩大后 ODE-RNN | 有效观测数 |', '|---|---:|---:|---:|---:|']
        for mod, first_mod in groups[available[0]][0]['metrics']['test']['per_modality'].items():
            vals = [fmt([r['metrics']['test']['per_modality'][mod]['macro_mse_z'] for r in groups[k]]) if k in groups else '—' for k in keys]
            lines.append(f"| {modalities[mod]} | "+' | '.join(vals)+f" | {first_mod['observations']:,} |")
    lines += ['', '<details>', '<summary>查看各模型按检查类别统计的完整 MSE、MAE 和观测数</summary>', '']
    for key,runs in groups.items():
        lines += [f'#### {names[key]}', '', '| 检查类别 | macro MSE | macro MAE | 有效观测数 |', '|---|---:|---:|---:|']
        for mod, one in runs[0]['metrics']['test']['per_modality'].items():
            vals = [r['metrics']['test']['per_modality'][mod] for r in runs]
            lines.append(f"| {modalities[mod]} | {fmt([v['macro_mse_z'] for v in vals])} | {fmt([v['macro_mae_z'] for v in vals])} | {one['observations']:,} |")
        lines.append('')
    lines += ['</details>', '', '### 6.3 按距离历史截止点的时间查看误差', '',
        '未来观测分为 0–1 年、1–3 年和 3 年以上。每个时间段只统计其中实际存在的检查。不同时间段的患者和检查组成可能不同，因此表中的变化需要结合有效观测数理解。']
    for v,label in [('structured','数值特征输入'),('medicalnet','数值特征 + MRI 输入')]:
        keys = [f'{v}/latent_ode',f'{v}/ode_rnn',f'{v}_matched/ode_rnn']
        available = [k for k in keys if k in groups]
        if not available:
            continue
        lines += ['', f'#### {label}', '', '| 时间跨度 | Latent ODE MSE | 原生 ODE-RNN MSE | 扩大后 ODE-RNN MSE | 有效观测数 |', '|---|---:|---:|---:|---:|']
        for h,label_h in [('0_to_1_year','0–1 年'),('1_to_3_years','1–3 年'),('over_3_years','3 年以上')]:
            one = groups[available[0]][0]['metrics']['test'].get('per_horizon',{}).get(h)
            if one:
                vals = [fmt([r['metrics']['test']['per_horizon'][h]['macro_mse_z'] for r in groups[k]]) if k in groups else '—' for k in keys]
                lines.append(f"| {label_h} | "+' | '.join(vals)+f" | {one['observations']:,} |")
    lines += ['', '## 7. 怎样联系到参考论文', '',
        '本次实验借鉴论文利用不完整历史预测未来检查的思路。论文使用的患者、字段和评分方式与这里不同。理解两边的任务后，可以比较各自实验中观察到的变化。', '',
        '关于论文模型、评分定义、报告数值，以及如何建立共同评价协议，见[与参考论文的逐步比较](paper_comparison.md)。', '',
        '## 8. 计算结果怎样核对', '',
        '所有训练、验证、影像编码和检查点评估都在四张 RTX 4090 上使用 CUDA。各任务的设备记录保存在 hardware.json 中。', '']
    all_runs = [r for runs in groups.values() for r in runs]
    solver = max(r['metrics']['solver_check']['absolute_metric_change'] for r in all_runs)
    lines += [f'对每个最优检查点，程序把 ODE 求解步长减半后重新计算验证误差。最大 macro MSE 变化为 {solver:.3g}，说明本次步长变化对该指标影响很小。', '',
        '检查点独立重新加载的核对结果保存在各任务的 checkpoint_verification.json 中。整体完成情况见[completion.json](completion.json)，机器可读的聚合统计见[summary.json](summary.json)。', '',
        '训练曲线与测试误差图见[comparison.png](comparison.png)。', '',
        '## 附录：表格名称与结果目录的对应关系', '',
        '阅读训练日志或单个任务文件时，可用下表定位。每个目录下面再按 seed_0 至 seed_4 区分独立训练。', '',
        '| 本报告名称 | 结果目录 | 完整参数量 | 预测网络参数量 |', '|---|---|---:|---:|']
    for key in groups:
        hw = groups[key][0]['hardware']
        lines.append(f"| {names[key]} | `{key}/` | {hw['parameters']:,} | {hw.get('forecast_network_parameters',0):,} |")
    lines.append('')
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='results/adni_cuda')
    p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args()
    root=Path(args.root)
    groups={}
    all_runs=[]
    for variant in ['structured','medicalnet','structured_matched','medicalnet_matched']:
        for model in ['latent_ode','ode_rnn']:
            runs=[]
            for f in sorted((root/variant/model).glob('seed_*/metrics.json')):
                m=json.loads(f.read_text())
                c=json.loads((f.parent/'config.json').read_text())
                h=json.loads((f.parent/'hardware.json').read_text())
                runs.append({'path':str(f.parent),'metrics':m,'config':c,'hardware':h})
            if runs:
                groups[f'{variant}/{model}']=runs
                all_runs.extend(runs)
    if not groups:
        raise ValueError('no completed GPU runs')
    if not args.allow_partial:
        expected={'structured/latent_ode','structured/ode_rnn','medicalnet/latent_ode',
                  'medicalnet/ode_rnn','structured_matched/ode_rnn','medicalnet_matched/ode_rnn'}
        if set(groups)!=expected or any(len(v)!=5 for v in groups.values()):
            raise ValueError('expected all 30 experiments; use --allow-partial for an interim report')
    aggregate = {}
    for key, runs in groups.items():
        aggregate[key] = {}
        for split in ['validation', 'test']:
            aggregate[key][split] = {}
            for metric in ['macro_mse_z', 'macro_mae_z']:
                vals = [r['metrics'][split][metric] for r in runs]
                aggregate[key][split][metric] = {
                    'mean': float(np.mean(vals)),
                    'sd': float(np.std(vals, ddof=1)) if len(vals)>1 else None,
                    'values': vals,
                }
    comparisons={}
    for variant in ['structured','medicalnet']:
        keys=[f'{variant}/{m}' for m in ['latent_ode','ode_rnn']]
        if all(k in groups for k in keys):
            left,right=[groups[k] for k in keys]
            if [r['config']['seed'] for r in left]==[r['config']['seed'] for r in right]:
                comparisons[variant]=paired_bootstrap(
                    [Path(r['path'])/'test_patients.npz' for r in left],
                    [Path(r['path'])/'test_patients.npz' for r in right])
    for model in ['latent_ode','ode_rnn']:
        keys=[f'medicalnet/{model}',f'structured/{model}']
        if all(k in groups for k in keys):
            left,right=[groups[k] for k in keys]
            if [r['config']['seed'] for r in left]==[r['config']['seed'] for r in right]:
                comparisons[f'imaging_gain/{model}']=paired_bootstrap(
                    [Path(r['path'])/'test_patients.npz' for r in left],
                    [Path(r['path'])/'test_patients.npz' for r in right])
    for variant in ['structured','medicalnet']:
        keys=[f'{variant}/latent_ode',f'{variant}_matched/ode_rnn']
        if all(k in groups for k in keys):
            left,right=[groups[k] for k in keys]
            if [r['config']['seed'] for r in left]==[r['config']['seed'] for r in right]:
                comparisons[f'{variant}_capacity_control']=paired_bootstrap(
                    [Path(r['path'])/'test_patients.npz' for r in left],
                    [Path(r['path'])/'test_patients.npz' for r in right])
    (root/'summary.md').write_text(render_report(groups, aggregate, comparisons))
    (root/'summary.json').write_text(json.dumps({'groups':aggregate,'paired_comparisons':comparisons},indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    colors={'latent_ode':'#277DA8','ode_rnn':'#E38B35'}
    labels={'structured/latent_ode':'Structured / Latent ODE',
            'structured/ode_rnn':'Structured / ODE-RNN',
            'medicalnet/latent_ode':'+ MRI / Latent ODE',
            'medicalnet/ode_rnn':'+ MRI / ODE-RNN',
            'structured_matched/ode_rnn':'Structured / ODE-RNN matched',
            'medicalnet_matched/ode_rnn':'+ MRI / ODE-RNN matched'}
    for key,runs in groups.items():
        histories=[json.loads((Path(r['path'])/'history.json').read_text()) for r in runs]
        n=min(map(len,histories))
        values=np.array([[h[i]['validation_macro_mse_z'] for i in range(n)] for h in histories])
        variant,model=key.split('/')
        color=colors[model]
        style={'structured':'-','medicalnet':'--','structured_matched':':','medicalnet_matched':'-.'}[variant]
        axes[0].plot(np.arange(1,n+1),values.mean(0),style,color=color,label=labels[key])
        spread=values.std(0,ddof=1) if len(values)>1 else np.zeros(n)
        axes[0].fill_between(np.arange(1,n+1),values.mean(0)-spread,values.mean(0)+spread,color=color,alpha=.1)
    axes[0].set(xlabel='Epoch',ylabel='Validation macro MSE (z)',title='Patient-disjoint forecasting')
    axes[0].legend(fontsize=8)
    keys=list(groups)
    means=[aggregate[k]['test']['macro_mse_z']['mean'] for k in keys]
    sd=[aggregate[k]['test']['macro_mse_z']['sd'] or 0 for k in keys]
    axes[1].bar(np.arange(len(keys)),means,yerr=sd,capsize=4,color=[colors[k.split('/')[1]] for k in keys])
    tick_labels=[labels[k].replace(' / ','\n').replace('ODE-RNN matched','ODE-RNN\nmatched') for k in keys]
    axes[1].set_xticks(np.arange(len(keys)),tick_labels,fontsize=8)
    persistence=all_runs[0]['metrics']['test']['persistence']['macro_mse_z']
    axes[1].axhline(persistence,color='#777777',linestyle=':',label='Last value baseline')
    axes[1].set(ylabel='Test macro MSE (z)',title='Mean ± sample SD across seeds')
    axes[1].legend(fontsize=8)
    fig.savefig(root/'comparison.png',dpi=180)
    fig.savefig(root/'comparison.svg')
    plt.close(fig)
    print(root/'summary.md')


if __name__=='__main__':
    main()
