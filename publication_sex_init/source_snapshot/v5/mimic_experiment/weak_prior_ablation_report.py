"""Statistics and plots from recorded histories for the weak-prior ablation."""
import csv
import json
from pathlib import Path

import numpy as np

from .prepare import atomic
from .report import read


def rows(out,manifest):
    queue=read(out/'queue.json',{});results=[]
    specifications=[('weak_prior=0.01',Path(manifest['reference']),.01,queue,True),('weak_prior=0',Path(manifest['config']['output']),0.,queue,True)]
    history_manifest=read(out/'history_reconstruction_study/manifest.json')
    if history_manifest:
        specifications.append(('weak_prior=0, future-only',Path(history_manifest['config']['output']),0.,read(out/'history_reconstruction_study/queue.json',{}),False))
    future_prior=read(out/'future_prior_scaling_study/manifest.json')
    if future_prior:
        states=read(out/'future_prior_scaling_study/queue.json',{}).get('jobs',{})
        for job in future_prior['jobs']:
            weight=job['config']['prior_weight']
            specifications.append((f'weak_prior={weight:g}, future-only',Path(job['config']['output']),weight,states.get(job['id'],{}),False))
    supervision=read(out/'supervision_study/manifest.json')
    if supervision:
        states=read(out/'supervision_study/queue.json',{}).get('jobs',{})
        for job in supervision['jobs']:
            label='A: multi-cutoff forecast' if job['id']=='A' else 'B: masked retrospection'
            specifications.append((label,Path(job['config']['output']),0.,states.get(job['id'],{}),job['id']=='B'))
    for label,path,weight,state,train_history in specifications:
        history=read(path/'history.json',[]);metrics=read(path/'metrics.json');best=min(history,key=lambda r:r['validation']['overall']['reconstruction_error']) if history else None
        results.append({'label':label,'path':str(path),'weight':weight,'history':history,'metrics':metrics,'best':best,
                        'hardware':read(path/'hardware.json',{}),'status':'complete' if metrics else state.get('status','waiting'),
                        'train_history':train_history})
    return results


def generate(out):
    out=Path(out);manifest=read(out/'manifest.json');models=rows(out,manifest)
    history_ablation=read(out/'history_reconstruction_study/manifest.json')
    future_prior=read(out/'future_prior_scaling_study/manifest.json')
    supervision=read(out/'supervision_study/manifest.json')
    lines=['# MIMIC：weak_prior与历史重建的连续消融','',
           '## 1. 要回答的问题','',
           '原latent2x both正向ODE在第41轮取得最佳验证结果，第46轮附近开始出现明显误差和动力学波动。第一阶段将weak_prior权重从0.01改为0，检查这种波动是否仍出现。原曲线横轴是训练轮次，第40轮对应6,760次参数更新。','',
           '## 2. 实验设置','',
           '### 2.1 第一阶段：去掉weak_prior','',
           '| 项目 | 原组 | 新增消融组 |','|---|---|---|',
           '| weak_prior权重 | 0.01 | 0 |',
           '| Jacobian / ODE L2 | 0.01 / 0.02 | 0.01 / 0.02 |',
           '| 参数量 / 状态维度 | 43,280 / 28 | 43,280 / 28 |',
           '| batch / 梯度累积 | 128 / 无 | 128 / 无 |',
           '| 训练设备 / seed | 单卡RTX 4090 GPU 3 / 0 | 单卡RTX 4090 GPU 3 / 0 |',
           '| 优化器 | Adam，lr=0.003，梯度裁剪10 | 相同 |',
           '| 学习率衰减 | 每步按0.97^(step/2000)缩放 | 相同 |',
           '| 训练上限 / 下限 / 早停耐心 | 300 / 50 / 40轮 | 相同 |','',
           '新组从seed 0重新初始化，数据顺序、历史截止点采样、输入dropout、年龄初始化、共享ODE、历史重建和未来预测保持原实现。UPDATE前预测仍照常计算并记录，其损失系数设为0；其余网络与数值训练代码使用原实验冻结快照。训练接入SwanLab。','',
           '验证选模仍使用历史重建＋未来预测的整体MSE和交叉熵。按共同覆盖的轮次比较过程曲线，另外记录各自最佳检查点的测试结果。','',
           ]
    if history_ablation:
        lines+=['### 2.2 第二阶段：移除训练中的历史重建路径','',
                '第一阶段在weak_prior=0时仍出现验证误差和Jacobian突增。该组按要求停止在第42轮，最佳检查点为第28轮，结果与检查点保留。第二阶段沿用weak_prior=0，进一步移除训练时从截止点向过去生成历史的路径。','',
                '| 项目 | 原latent2x both | 去除weak_prior | 新增future-only |','|---|---|---|---|',
                '| weak_prior权重 | 0.01 | 0 | 0 |',
                '| 历史按时间顺序吸收 | 保留 | 保留 | 保留 |',
                '| 训练时向过去重建历史 | 保留 | 保留 | 移除 |',
                '| 训练数据监督 | 历史＋未来 | 历史＋未来 | 仅截止点之后的未来观测 |',
                '| 训练Jacobian采样状态 | 历史与未来生成状态 | 历史与未来生成状态 | 仅未来生成状态 |',
                '| Jacobian / ODE L2权重 | 0.01 / 0.02 | 相同 | 相同 |',
                '| 参数量 / batch / 梯度累积 | 43,280 / 128 / 无 | 相同 | 相同 |',
                '| 验证与测试历史回顾指标 | 保留 | 保留 | 保留 |',
                '| 选模与早停指标 | 验证整体误差 | 相同 | 相同 |','',
                '新组从seed 0重新训练。先按时间顺序吸收历史，得到截止点状态；随后只向未来积分，在未来实际检查时间计算MSE和交叉熵。每个字段按未来有效观测数归一化。历史观测用于编码，不再作为训练重建目标。','',
                '新组训练损失为：未来MSE＋未来交叉熵＋0.01×未来生成状态Jacobian＋0.02×ODE参数L2。移除历史生成后，Jacobian仍按同一公式计算，训练采样位置改为未来生成状态。模型结构、初始化、数据、优化器、学习率衰减、dropout和训练预算沿用第一阶段。','',
                '验证和测试仍计算原来的历史回顾、未来预测和整体指标；这一阶段不反向传播。选模与早停继续使用验证整体误差。分析新组训练稳定性时，结合未来误差、训练梯度与训练Jacobian观察，历史回顾误差单独列出。','',
                '[第二阶段配置](history_reconstruction_study/manifest.json) · [运行状态](history_reconstruction_study/queue.json) · [CUDA路径检查](history_reconstruction_study/cuda_checks.json)','']
    if future_prior:
        lines+=['### 2.3 第三阶段：在future-only版本上增加weak_prior','',
                '第二阶段的weak_prior=0、future-only已完成，直接作为本阶段参照。新增两组从seed 0重新训练，仅将weak_prior系数设为0.01和1，保留同样的future-only主生成路径。','',
                '| 项目 | 已有future-only | 新增组一 | 新增组二 |','|---|---|---|---|',
                '| weak_prior权重 | 0 | 0.01 | 1 |',
                '| 训练时向过去重建历史 | 无 | 无 | 无 |',
                '| 主数据项 | 未来MSE＋交叉熵 | 相同 | 相同 |',
                '| 历史UPDATE前预测监督 | 系数0 | 系数0.01 | 系数1 |',
                '| Jacobian采样位置 / 权重 | 未来生成状态 / 0.01 | 相同 | 相同 |',
                '| ODE L2 | 0.02 | 0.02 | 0.02 |',
                '| 参数量 / 状态维度 | 43,280 / 28 | 相同 | 相同 |',
                '| batch / 梯度累积 / seed | 128 / 无 / 0 | 相同 | 相同 |',
                '| 实际物理GPU | 3（已完成） | 3 | 1 |',
                '| 选模与早停 | 验证整体误差 | 相同 | 相同 |','',
                '这里future-only指主生成损失只监督未来。weak_prior>0时，还会监督历史读取过程中UPDATE前的预测：先沿ODE向未来到达下一条历史记录，读出指标并计算误差，再吸收该记录。整个训练过程保持向未来积分，历史回顾只在验证与测试中计算。','',
                '第三阶段训练目标：**未来MSE＋未来交叉熵＋λ×历史UPDATE前预测误差＋0.01×未来状态Jacobian＋0.02×ODE参数L2**。λ分别为0、0.01和1。未来项与prior项分别按各字段有效观测数归一化。','',
                '数据、网络、年龄初始化、dropout、优化器、学习率衰减、训练上限与早停保持一致。两个新增组仍接入SwanLab，并保存实际CUDA张量设备与GPU分配。CUDA检查将核对损失和参数梯度的增量都等于λ倍prior，同时核对λ=0时与已完成版本数值一致。','',
                '[第三阶段配置](future_prior_scaling_study/manifest.json) · [运行状态](future_prior_scaling_study/queue.json) · [损失及梯度CUDA检查](future_prior_scaling_study/cuda_checks.json)','']
    if supervision:
        lines+=Path(__file__).with_name('supervision_ablation_method.zh-CN.md').read_text().splitlines()
    lines+=[
           '## 3. 训练进度与最佳结果','',
           '| 组别 | 状态 | 已训练 / 最佳轮次 | 验证整体 | 测试历史 | 测试未来 | 测试整体 | 数值标准化MSE |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for model in models:
        best=model['best'];metric=model['metrics'];test=metric['test'] if metric else None
        values=[model['label'],model['status'],f"{len(model['history'])} / {best['epoch'] if best else '—'}",
                f"{best['validation']['overall']['reconstruction_error']:.6f}" if best else '—']
        values += [f"{test[k]['reconstruction_error']:.6f}" if test else '—' for k in ['interpolation','forecast','overall']]
        values += [f"{test['mean_standardized_numeric_mse']:.6f}" if test else '—'];lines.append('| '+' | '.join(values)+' |')
    signature=[len(model['history']) for model in models]
    if any(signature) and signature!=read(out/'plot_signature.json'):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(17,9))
        specifications=[('Training total loss',lambda h:h['training_loss']),
                        ('Validation overall',lambda h:h['validation']['overall']['reconstruction_error']),
                        ('Validation forecast',lambda h:h['validation']['forecast']['reconstruction_error']),
                        ('Validation history reconstruction',lambda h:h['validation']['interpolation']['reconstruction_error']),
                        ('Training gradient norm before clipping',lambda h:h['training_components']['gradient_norm']),
                        ('Validation squared Jacobian norm',lambda h:h['validation']['dynamics']['jacobian_frobenius_squared'])]
        for ax,(title,extract) in zip(axes.flat,specifications):
            for model in models:
                if not model['history']:continue
                ax.plot([h['epoch'] for h in model['history']],[extract(h) for h in model['history']],label=model['label'])
            ax.axvline(40,color='gray',ls='--',alpha=.6);ax.set(title=title,xlabel='Epoch');ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[1,1].set_yscale('log');axes[1,2].set_yscale('log');fig.tight_layout()
        for ext in ['png','svg','pdf']:
            target=out/f'training_curves.{ext}';tmp=out/f'training_curves.partial.{ext}';fig.savefig(tmp,dpi=160);tmp.replace(target)
        plt.close(fig);atomic(out/'plot_signature.json',signature)
        if history_ablation:
            fig,axes=plt.subplots(1,2,figsize=(13,4.5))
            for model in models:
                if not model['history']:continue
                epochs=[h['epoch'] for h in model['history']]
                axes[0].plot(epochs,[h['training_components']['jacobian_frobenius_squared'] for h in model['history']],label=model['label'])
                axes[1].plot(epochs,[h['training_components']['ode_l2'] for h in model['history']],label=model['label'])
            for ax,title in zip(axes,['Training Jacobian regularizer','Shared ODE parameter L2']):
                ax.axvline(40,color='gray',ls='--',alpha=.6);ax.set(title=title,xlabel='Epoch');ax.set_yscale('log');ax.grid(alpha=.2);ax.legend(fontsize=8)
            fig.tight_layout()
            for ext in ['png','svg','pdf']:
                target=out/f'training_regularizers.{ext}';tmp=out/f'training_regularizers.partial.{ext}';fig.savefig(tmp,dpi=160);tmp.replace(target)
            plt.close(fig)
    lines+=['','## 4. 训练稳定性曲线','',
            '虚线表示第40轮。梯度范数取裁剪前每个训练batch的范数，再按轮求均值；Jacobian为验证生成状态上平方Frobenius范数的均值。后两幅图采用对数纵轴，原始曲线不平滑。','',
            '![训练误差与动力学变化](training_curves.png)','',
            '[SVG](training_curves.svg) · [PDF](training_curves.pdf)','',
            ]
    if history_ablation:
        lines+=['future-only组的主数据项只监督未来；weak_prior>0时再加入历史UPDATE前预测误差，图中的训练总损失包含对应系数的prior。验证曲线继续使用共同评价器。下面补充训练正则曲线：前两组Jacobian覆盖历史与未来生成状态，所有future-only组均覆盖未来生成状态。','',
                '![训练正则变化](training_regularizers.png)','',
                '[SVG](training_regularizers.svg) · [PDF](training_regularizers.pdf)','']
    lines+=[
            '## 5. 第40轮前后的分段统计','',
            '| 组别 | 轮次窗口 | 已完成轮数 | 验证整体均值 | 验证未来均值 | 最大轮均梯度范数 | 最大验证Jacobian |',
            '|---|---|---:|---:|---:|---:|---:|']
    for model in models:
        windows=[(1,30),(31,40),(41,50),(51,65),(66,81)]
        if max(signature)>81:windows.append((82,120))
        if max(signature)>120:windows.append((121,200))
        if max(signature)>200:windows.append((201,300))
        for start,end in windows:
            selected=[h for h in model['history'] if start<=h['epoch']<=end]
            values=[model['label'],f'{start}–{end}',str(len(selected))]
            if selected:
                values += [f"{np.mean([h['validation'][r]['reconstruction_error'] for h in selected]):.6f}" for r in ['overall','forecast']]
                values += [f"{max(h['training_components']['gradient_norm'] for h in selected):.6f}",f"{max(h['validation']['dynamics']['jacobian_frobenius_squared'] for h in selected):.6f}"]
            else:values+=['—']*4
            lines.append('| '+' | '.join(values)+' |')
    common=min(signature)
    pairs=[(0,1)]+([(1,2)] if history_ablation else [])
    if future_prior or supervision:pairs.extend((2,index) for index in range(3,len(models)))
    for a,b in pairs:
        end=min(signature[a],signature[b])
        if end>=41:
            selected=[[h for h in models[j]['history'] if 41<=h['epoch']<=end] for j in [a,b]]
            means=[np.mean([h['validation']['forecast']['reconstruction_error'] for h in hs]) for hs in selected]
            lines+=['',f"共同的第41–{end}轮中，{models[a]['label']}与{models[b]['label']}的验证未来误差均值分别为{means[0]:.6f}、{means[1]:.6f}。",'']
    lines+=['','## 6. 数据和原始记录','',f"数据SHA256：`{manifest['config']['dataset_sha256']}`。",'',
            '[配置与来源](manifest.json) · [运行状态](queue.json) · [CUDA损失检查](cuda_loss_check.json) · [逐轮CSV](epoch_metrics.csv) · [完整比较](comparison.json)', '',
            '[原方向实验](../direction_full_batch_cuda/summary.md) · [新增组目录](no_weak_prior/seed_0/)', '']
    if (out.parent/'init_covariate_ablation_cuda/manifest.json').exists():
        lines+=['后续实验恢复标准latent2x both的完整损失，分别更改INIT输入为首次观测年龄、性别：[标准模型INIT消融与曲线](../init_covariate_ablation_cuda/summary.md)。','']
    mechanism=out/'mechanism_diagnosis/summary_section.zh-CN.md'
    if mechanism.exists():lines+=['','<!-- latent2x-mechanism-diagnosis -->','',mechanism.read_text()]
    causal=out/'targeted_causal_ablation/summary_section.zh-CN.md'
    if causal.exists():lines+=['','<!-- latent2x-targeted-causal -->','',causal.read_text()]
    (out/'summary.md').write_text('\n'.join(lines))
    with (out/'epoch_metrics.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['group','epoch','optimizer_steps','training_loss','prior_unweighted','prior_weighted','gradient_norm','validation_overall','validation_history','validation_forecast','validation_jacobian','training_jacobian','training_ode_l2','prior_weight','train_history'])
        for model in models:
            for h in model['history']:
                t=h['training_components'];prior=t['prior_mse']+t['prior_cce'] if t.get('prior_mse') is not None and t.get('prior_cce') is not None else ''
                writer.writerow([model['label'],h['epoch'],h['optimizer_steps'],h['training_loss'],prior,t['weighted_prior'],t['gradient_norm']]+[h['validation'][r]['reconstruction_error'] for r in ['overall','interpolation','forecast']]+[h['validation']['dynamics']['jacobian_frobenius_squared'],t['jacobian_frobenius_squared'],t['ode_l2'],model['weight'],model['train_history']])
    atomic(out/'comparison.json',{'models':[{k:v for k,v in model.items() if k not in ['history','best']} for model in models],'common_epochs_completed':common})
