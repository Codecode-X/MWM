"""Summarize saved results; numerical aggregation only, no model execution."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='paper_odeae/results/local_available')
    a=p.parse_args();root=Path(a.root)
    reference=json.loads((Path(__file__).parent/'reference/reported_results.json').read_text())
    groups={}
    for model in ['ode_ae','deterministic','lstm']:
        runs=[json.loads(p.read_text()) for p in sorted((root/model).glob('seed_*/metrics.json'))]
        if runs:groups[model]=runs
    lines=['# ODE-AE 复现目录的运行记录','','## 1. 本次运行使用什么设置','']
    configs=[json.loads(p.read_text()) for p in root.glob('*/seed_*/config.json')]
    smoke=bool(configs) and all(c['smoke'] for c in configs)
    if smoke:
        lines+=['本次是 CUDA 端到端小规模检查。每个任务只训练一轮，使用少量患者，目的是核对数据、梯度、保存和重新加载流程。','']
    else:
        lines+=['下面汇总已完成的独立训练任务。各模型按验证集整体重构误差选择检查点，再计算测试结果。','']
    data_files=list(root.glob('*/seed_*/data.json'))
    if data_files:
        data=json.loads(data_files[0].read_text())
        lines += [f"数据范围为 `{data['scope']}`，有训练观测的字段数为 {data['features_with_training_observations']}。患者划分为 {data['patient_counts']}。",'']
        if data['missing_training_features']:
            lines+=['与 46 项字段合同相比，当前没有训练观测的字段为：'+', '.join(data['missing_training_features'])+'。','']
    lines+=['## 2. 本地运行的误差','','每个字段先在其有效观测上平均，再将数值 MSE 和类别交叉熵相加。以下点预测使用后验均值。','','| 模型 | 完成任务数 | 测试插值误差 | 测试预测误差 | 测试整体误差 |','|---|---:|---:|---:|---:|']
    aggregate={}
    for model,runs in groups.items():
        aggregate[model]={}
        cells=[]
        for region in ['interpolation','forecast','overall']:
            vals=[r['test'][region]['reconstruction_error'] for r in runs]
            aggregate[model][region]={'mean':float(np.mean(vals)),'sd':float(np.std(vals,ddof=1)) if len(vals)>1 else None}
            cells.append(f'{np.mean(vals):.4f}'+(f' ± {np.std(vals,ddof=1):.4f}' if len(vals)>1 else ''))
        lines.append(f'| {model} | {len(runs)} | '+' | '.join(cells)+' |')
    lines+=['','## 3. 论文报告的参照数值','','| 论文模型 | 插值误差 | 预测误差 | 整体误差 |','|---|---:|---:|---:|']
    for name,values in reference['model_comparison_table']['rows'].items():
        lines.append(f'| {name} | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines+=['','## 4. 怎样理解这次运行','','先检查第一节的数据范围，再检查训练轮数与字段合同。小规模检查的通过说明运行链路可用。论文效果的复现则需要完整的数据和足够的正式训练。',
            '', '本目录当前实现了按原文重建的模型。未说明的网络细节、归一化和 LSTM 设置记录在 [实现对照](../../docs/IMPLEMENTATION.zh-CN.md) 中。论文原始患者划分尚未取得，因此当前不把本地误差标记为论文数值已复现。','']
    # This path is derived from the actual output directory, including custom roots.
    import os
    doc=os.path.relpath(Path(__file__).parent/'docs/IMPLEMENTATION.zh-CN.md',root)
    lines=[line.replace('../../docs/IMPLEMENTATION.zh-CN.md',doc) for line in lines]
    (root/'summary.md').write_text('\n'.join(lines))
    (root/'summary.json').write_text(json.dumps({'smoke':smoke,'published_numbers_reproduced':False,'groups':aggregate},indent=2))
    print(root/'summary.md')


if __name__=='__main__':main()
