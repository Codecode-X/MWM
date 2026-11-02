"""Saved metrics and curves for the standard-model INIT input comparison."""
import csv
from pathlib import Path

import numpy as np

from .prepare import atomic
from .report import read


def generate(out):
    out=Path(out);manifest=read(out/'manifest.json');queue=read(out/'queue.json',{});models=[]
    specifications=[('standard cutoff-age INIT',Path(manifest['reference']),{'status':'complete'},'cutoff_age')]
    specifications += [(j['id'],Path(j['config']['output']),queue.get('jobs',{}).get(j['id'],{}),j['config']['init_covariate']) for j in manifest['jobs']]
    for name,run,status,source in specifications:
        history=read(run/'history.json',[]);m=read(run/'metrics.json');best=min(history,key=lambda h:h['validation']['overall']['reconstruction_error']) if history else None
        models.append({'name':name,'path':str(run),'source':source,'status':'complete' if m else status.get('status','waiting'),
                       'phase':status.get('phase'),'history':history,'best':best,'metrics':m,'hardware':read(run/'hardware.json',{})})
    lines=['# MIMIC：标准latent2x both的INIT输入消融','',
           '## 1. 这次改变什么','',
           '标准模型从第一条历史记录开始吸收观测，但交给INIT的是最后一条已知历史记录处的年龄。本次分别将这个标量替换为首次观测年龄、性别，比较初始化信息对训练稳定性和预测精度的影响。','',
           '两组恢复标准latent2x both的完整训练流程：按时间顺序吸收历史，从历史截止点向过去重建历史、向未来预测。weak_prior为0.01，Jacobian为0.01，ODE参数L2为0.02。','',
           '## 2. 三组的INIT输入','',
           '| 项目 | 标准模型 | 消融1：首次观测年龄 | 消融2：性别 |','|---|---|---|---|',
           '| 实际进入INIT的标量 | (入ICU年龄＋最后历史时刻/8766)/100 | (入ICU年龄＋首次观测时刻/8766)/100 | F=0、M=1 |',
           '| 初始化状态的时间位置 | 首次历史观测处 | 相同 | 相同 |',
           '| INIT结构 / 参数 | 1→159→28 / 4,798 | 相同 | 相同 |',
           '| 总参数 / 状态维度 | 43,280 / 28 | 相同 | 相同 |',
           '| 历史读取 / 生成起点 | 按时间顺序 / 历史截止点 | 相同 | 相同 |',
           '| 训练数据项 | 历史重建＋未来预测 | 相同 | 相同 |',
           '| weak_prior / Jacobian / ODE L2 | 0.01 / 0.01 / 0.02 | 相同 | 相同 |',
           '| batch / 梯度累积 / seed | 128 / 无 / 0 | 相同 | 相同 |',
           '| 物理GPU | 3（已有结果） | 3 | 1 |','',
           'MIMIC的ODE时钟使用小时，年龄使用年，8766为一年小时数。主要历史窗口不超过24小时，首次记录与最后已知记录的年龄差最多约23/8766=0.002625年；除以100后，INIT输入差最多约0.00002625。实际输入另存于每组init_input_audit.json。','',
           '性别从首次已知历史记录取得，缓存中F编码为0、M为1。性别作为INIT静态输入，与原年龄INIT一样不施加观测dropout；原来的性别观测字段仍照常进入UPDATE并保留原dropout。原INIT接口会除以100，输入准备阶段将性别编码乘100，使进入网络的值恰为0或1。','',
           '## 3. 训练与测试设置','',
           '新增组从seed 0重新初始化，与标准模型保持相同初始权重、数据顺序、历史窗口采样、观测dropout和全部网络。Adam学习率0.003，按0.97^(step/2000)衰减，梯度裁剪10；最多300轮、至少50轮，验证整体误差连续40轮未改善时早停。','',
           '主要测试读取前24小时并预测至48小时，另外记录6小时和12小时历史条件下的测试。验证与测试沿用四个固定患者分片、每批32人的共同评价器。最佳检查点按验证历史重建＋未来预测的整体MSE和交叉熵选择。','',
           '两组均使用单卡RTX 4090、完整batch128，每批一次反向传播和参数更新，无梯度累积；训练接入SwanLab。实际INIT输入由CUDA前置hook记录，训练、验证和测试采用同一输入规则。','',
           '## 4. 进度与指标','',
           '| 组别 | 状态 | 已训练 / 最佳轮次 | 验证整体 | 测试历史 | 测试未来 | 测试整体 | 数值标准化MSE |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for model in models:
        h=model['history'];best=model['best'];m=model['metrics'];t=m['test'] if m else None
        values=[model['name'],model['status'],f"{len(h)} / {best['epoch'] if best else '—'}",f"{best['validation']['overall']['reconstruction_error']:.6f}" if best else '—']
        values += [f"{t[r]['reconstruction_error']:.6f}" if t else '—' for r in ['interpolation','forecast','overall']]
        values += [f"{t['mean_standardized_numeric_mse']:.6f}" if t else '—'];lines.append('| '+' | '.join(values)+' |')
    signature=[len(m['history']) for m in models]
    if any(signature) and signature!=read(out/'plot_signature.json'):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(17,9))
        specs=[('Training total loss',lambda h:h['training_loss']),('Validation overall',lambda h:h['validation']['overall']['reconstruction_error']),
               ('Validation forecast',lambda h:h['validation']['forecast']['reconstruction_error']),('Validation history',lambda h:h['validation']['interpolation']['reconstruction_error']),
               ('Training gradient norm before clipping',lambda h:h['training_components']['gradient_norm']),('Validation Jacobian squared norm',lambda h:h['validation']['dynamics']['jacobian_frobenius_squared'])]
        for ax,(title,extract) in zip(axes.flat,specs):
            for model in models:
                if not model['history']:continue
                ax.plot([h['epoch'] for h in model['history']],[extract(h) for h in model['history']],label=model['name'])
            ax.axvline(40,color='gray',ls='--',alpha=.6);ax.set(title=title,xlabel='Epoch');ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[1,1].set_yscale('log');axes[1,2].set_yscale('log');fig.tight_layout()
        for ext in ['png','svg','pdf']:
            path=out/f'training_curves.{ext}';tmp=out/f'training_curves.partial.{ext}';fig.savefig(tmp,dpi=160);tmp.replace(path)
        plt.close(fig);atomic(out/'plot_signature.json',signature)
    lines+=['','## 5. 训练与动力学曲线','',
            '虚线为第40轮，后两幅图采用对数纵轴。曲线直接使用每轮记录，不做平滑；梯度范数为裁剪前范数的轮均值。','',
            '![训练与验证曲线](training_curves.png)','', '[SVG](training_curves.svg) · [PDF](training_curves.pdf)','',
            '## 6. 轮次窗口统计','',
            '| 组别 | 轮次 | 已记录轮数 | 验证整体均值 | 验证未来均值 | 最大轮均梯度范数 | 最大验证Jacobian |',
            '|---|---|---:|---:|---:|---:|---:|']
    windows=[(1,30),(31,40),(41,50),(51,65),(66,81)]
    if max(signature)>81:windows.append((82,120))
    if max(signature)>120:windows.append((121,200))
    if max(signature)>200:windows.append((201,300))
    for model in models:
        for start,end in windows:
            h=[h for h in model['history'] if start<=h['epoch']<=end];values=[model['name'],f'{start}–{end}',str(len(h))]
            if h:
                values += [f"{np.mean([x['validation'][r]['reconstruction_error'] for x in h]):.6f}" for r in ['overall','forecast']]
                values += [f"{max(x['training_components']['gradient_norm'] for x in h):.6f}",f"{max(x['validation']['dynamics']['jacobian_frobenius_squared'] for x in h):.6f}"]
            else:values+=['—']*4
            lines.append('| '+' | '.join(values)+' |')
    lines+=['','## 7. 原始文件','',
            '[配置与来源](manifest.json) · [队列](queue.json) · [CUDA核验](cuda_checks.json) · [逐轮CSV](epoch_metrics.csv) · [完整比较](comparison.json)','',
            '[首诊年龄组](first_age/seed_0/) · [性别组](sex/seed_0/) · [标准正向组](../direction_full_batch_cuda/summary.md) · [此前损失消融](../weak_prior_ablation_cuda/summary.md)','']
    (out/'summary.md').write_text('\n'.join(lines))
    with (out/'epoch_metrics.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['group','epoch','optimizer_steps','training_loss','validation_overall','validation_history','validation_forecast','gradient_norm','training_jacobian','validation_jacobian','prior_weighted'])
        for model in models:
            for h in model['history']:
                c=h['training_components'];writer.writerow([model['name'],h['epoch'],h['optimizer_steps'],h['training_loss']]+[h['validation'][r]['reconstruction_error'] for r in ['overall','interpolation','forecast']]+[c['gradient_norm'],c['jacobian_frobenius_squared'],h['validation']['dynamics']['jacobian_frobenius_squared'],c['weighted_prior']])
    atomic(out/'comparison.json',{'models':[{k:v for k,v in m.items() if k not in ['history','best']} for m in models]})
