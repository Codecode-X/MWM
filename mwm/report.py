"""Saved-statistic progress, batch-size comparisons and training curves."""
import json
import os
from pathlib import Path

from mimic_experiment.prepare import atomic


def read(path,default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def number(value,digits=6):
    return f'{value:.{digits}f}' if value is not None else '—'


def row_data(job,queue):
    cfg=job['config'];run=Path(cfg['output']);metrics=read(run/'metrics.json');history=read(run/'history.json',[])
    state=queue.get('jobs',{}).get(job['id'],{});hardware=read(run/'hardware.json',{})
    best=min(history,key=lambda r:r['validation']['overall']['reconstruction_error']) if history else None
    return {'id':job['id'],'config':cfg,'status':metrics['status'] if metrics else state.get('status','waiting'),
            'phase':state.get('phase'),'hardware':hardware,'gpu':hardware.get('physical_gpu',state.get('physical_gpu')),
            'epochs':len(history),'best_epoch':metrics['best_epoch'] if metrics else best['epoch'] if best else None,
            'metrics':metrics,'validation_best_causal':best['validation']['overall']['reconstruction_error'] if best else None,
            'history':history,'best_history':best}


def report(out,manifest):
    out=Path(out);queue=read(out/'queue.json',{});resources=read(out/'resource_allocation.json',{});all_rows=[]
    for dataset in ['adni','mimic']:
        jobs=[j for j in manifest['jobs'] if j['dataset']==dataset];folder=Path(jobs[0]['config']['comparison_output'])
        rows=[row_data(j,queue) for j in jobs];all_rows+=rows
        batch_comparison=any(r['config']['batch_size']==32 for r in rows)
        active=[f"GPU {r['gpu']}：MWM {r['config']['variant']}（{r['config']['dataset_name'].upper()}，batch={r['config']['batch_size']}）"
                for r in [row_data(j,queue) for j in manifest['jobs']] if r['status']=='running' and r['gpu'] is not None]
        lines=[f'# {dataset.upper()}：MWM训练与比较','',
               '当前运行：'+'；'.join(active)+'。' if active else '当前状态见下方进度表。','',
               '## 1. 训练设置','',
               '所有版本按检查时间顺序读取历史。首次观测年龄用于INIT，第一次检查经UPDATE建立初始估计；此后每次先用ODE预测当前检查，再吸收该次检查。历史截止点之后只做ODE预测。','',
               ('主比较的main、A、B使用batch=128；新增A和B的batch=32对照，分别从seed 0重新训练。' if batch_comparison else '本数据集各版本使用batch=128。')+
               '每组独占一张RTX 4090，每个batch执行一次反向传播和参数更新，无梯度累积。最多300轮、至少50轮，验证指标连续40轮未改善时早停。全部训练接入SwanLab。','',
               '三种方法统一按验证集未加权的逐次预测MSE＋交叉熵选择检查点。A的计数权重和B的NLL只作用于训练。验证与共同测试沿用固定32条查询批次。','',
               '## 2. 损失与参数量','',
               '| 版本 | batch | 数据监督 | INIT / UPDATE / Decoder / ODE隐藏宽度 | 参数量 |',
               '|---|---:|---|---|---:|']
        if queue.get('status')=='paused_by_user':
            lines[2:2]=['MWM相关训练已按要求暂停，已有结果、曲线和恢复检查点保留。当前主模型恢复为latent2x both，同参数量baseline继续训练。','']
        for row in rows:
            cfg=row['config'];v=cfg['variant'];desc={'main':'数值MSE＋类别交叉熵，所有观测同权','A':'逐次预测误差乘sigmoid(N−3)，N为已吸收观测数','B':'数值Gaussian NLL＋类别交叉熵，每个时刻输出一个σ'}[v]
            lines.append(f"| MWM {v} | {cfg['batch_size']} | {desc} | {'159 / 160 / 158 / 187' if v=='B' else '159 / 159 / 159 / 188'} | {cfg['target_parameters']:,} |")
        lines+=['','各组保留0.01 Jacobian与0.02共享ODE参数L2。B的σ头通过少量隐藏宽度调整匹配总参数量，σ=softplus(输出)＋0.01，初始为1。','',
                '## 3. 实验进度','',
                '| 版本 | batch | 状态 | 物理GPU | 已训练 / 最佳轮次 | 验证逐次预测整体误差 |',
                '|---|---:|---|---:|---:|---:|']
        for r in rows:
            lines.append('| '+' | '.join([f"MWM {r['config']['variant']}",str(r['config']['batch_size']),r['status']+' / '+str(r['phase']),
                                         str(r['gpu']) if r['gpu'] is not None else '—',f"{r['epochs']} / {r['best_epoch'] or '—'}",number(r['validation_best_causal'])])+' |')
        from .plotting import training_curves
        plots=training_curves(folder,rows,read(folder/'curve_signature.json'))
        lines+=['','## 4. 训练曲线','',
                '曲线随已完成轮次自动更新。第一幅图用统一的未加权MSE＋交叉熵展示训练数据误差、验证逐次预测整体误差与未来预测误差；右下图按优化器更新次数比较未来预测误差。','']
        if 'training_curves' in plots:
            lines+=['![训练与验证曲线](training_curves.png)','',
                    '[SVG](training_curves.svg) · [PDF](training_curves.pdf)','']
        else:lines+=['首轮完成后显示训练曲线。','']
        if batch_comparison:
            lines+=['第二幅图分别比较A、B在两个batch下的真实训练总损失、验证整体误差与按更新次数计的未来误差。B的训练总损失包含NLL；A、B的损失数值含义不同，各自在同一行比较batch差异。','']
            if 'batch_comparison_curves' in plots:
                lines+=['![A与B的batch对照曲线](batch_comparison_curves.png)','',
                        '[SVG](batch_comparison_curves.svg) · [PDF](batch_comparison_curves.pdf)','']
        lines+=['## 5. 共同测试结果','',
                '测试历史回顾从已知历史截止点向过去自由ODE积分；测试未来从同一截止点向未来自由ODE积分，两者均不吸收目标观测。“整体”汇总历史回顾和未来预测。历史逐次预测单列，首次检查为UPDATE后读出，其余历史检查均在UPDATE前预测。','',
                '| 模型 | batch | 参数量 | 最佳轮次 | 测试历史回顾 | 测试未来 | 测试整体 | 测试历史逐次预测 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
        ref=Path(jobs[0]['config']['reference_output']);refm=read(ref/'metrics.json');refh=read(ref/'hardware.json',{});refc=read(ref/'config.json',{})
        if refm:
            t=refm['test'];values=['latent2x both参考',str(refc.get('batch_size','—')),str(refh.get('parameters','—')),str(refm['best_epoch'])]
            lines.append('| '+' | '.join(values+[number(t[k]['reconstruction_error']) for k in ['interpolation','forecast','overall']]+['—'])+' |')
        for r in rows:
            m=r['metrics'];t=m['test'] if m else None;cfg=r['config']
            values=[f"MWM {cfg['variant']}",str(cfg['batch_size']),str(cfg['target_parameters']),str(r['best_epoch'] or '—')]
            values+=[number(t[k]['reconstruction_error']) if t else '—' for k in ['interpolation','forecast','overall']]
            values+=[number(m['causal_test']['interpolation']['reconstruction_error']) if m else '—']
            lines.append('| '+' | '.join(values)+' |')
        lines+=['','参考组按旧的历史回顾＋未来整体误差选检查点；MWM按逐次预测整体误差选择。MWM与参考组的变化包括首次年龄初始化、监督方式、空输入UPDATE处理和选模方式。','']
        if batch_comparison:
            lines+=['## 6. batch=32与128的配对比较','',
                    '每种方法仅改变训练batch，模型参数、seed 0、数据划分、输入dropout、学习率0.003、衰减规则、损失、早停和测试流程保持一致。新batch32组从初始权重重新训练。','',
                    'batch32每轮的参数更新次数约为batch128的四倍。学习率仍按更新次数衰减。下表同时记录最佳轮次、累计更新次数与达到最佳验证分数的时间，以解释batch变化带来的优化差异。','',
                    '| 方法 | batch | 最佳轮次 | 最佳时累计更新数 | 达到最佳用时/分 | 验证逐次整体 | 测试未来 | 测试整体 |',
                    '|---|---:|---:|---:|---:|---:|---:|---:|']
            for variant in ['A','B']:
                for batch_size in [32,128]:
                    r=next(r for r in rows if r['config']['variant']==variant and r['config']['batch_size']==batch_size)
                    h=r['best_history'];m=r['metrics']
                    vals=[f'MWM {variant}',str(batch_size),str(r['best_epoch'] or '—'),str(h['optimizer_steps']) if h else '—',
                          number(h['elapsed_seconds']/60,2) if h else '—',number(r['validation_best_causal'])]
                    vals += [number(m['test'][k]['reconstruction_error']) if m else '—' for k in ['forecast','overall']]
                    lines.append('| '+' | '.join(vals)+' |')
            lines+=['','| 方法 | batch32−batch128：测试未来误差 | batch32未来误差相对降低 |','|---|---:|---:|']
            for variant in ['A','B']:
                a=next(r for r in rows if r['config']['variant']==variant and r['config']['batch_size']==32)['metrics']
                b=next(r for r in rows if r['config']['variant']==variant and r['config']['batch_size']==128)['metrics']
                if a and b:
                    difference=a['test']['forecast']['reconstruction_error']-b['test']['forecast']['reconstruction_error']
                    relative=-100*difference/b['test']['forecast']['reconstruction_error']
                    lines.append(f'| MWM {variant} | {difference:.6f} | {relative:.2f}% |')
                else:lines.append(f'| MWM {variant} | 等待两组完成 | — |')
            lines+=['','差值为负表示batch32未来误差更小；相对降低为正表示batch32有改善。','']
        section=7 if batch_comparison else 6
        lines+=[f'## {section}. 数据与原始记录','',f"数据SHA256：`{jobs[0]['config']['dataset_sha256']}`。",'',
                '各组保存config.json、hardware.json、history.json、best.pt、last.pt、metrics.json和swanlab.json；设备记录包含实际物理GPU和张量设备。','',
                f"[方法说明]({os.path.relpath(Path(manifest['project_root'])/'paper_odeae/MWM_METHOD.zh-CN.md',folder)}) · [运行队列]({os.path.relpath(out/'queue.json',folder)}) · [结构化比较](comparison.json)",'']
        if dataset=='adni' and resources.get('archived_adni_batch32'):
            lines+=[f"此前中断的batch32结果保存在[归档目录]({os.path.relpath(Path(resources['archived_adni_batch32'])/'summary.md',folder)})；本轮新增的batch32对照保存在A_batch32、B_batch32目录。",'']
        (folder/'summary.md').write_text('\n'.join(lines))
        atomic(folder/'comparison.json',{'dataset':dataset,'models':[{k:v for k,v in r.items() if k not in ['history','best_history']} for r in rows]})
    atomic(out/'comparison.json',{'models':[{k:v for k,v in r.items() if k not in ['history','best_history']} for r in all_rows]})
    lines=['# MWM：ADNI与MIMIC实验','',
           '每组使用独立单卡，无梯度累积。ADNI主比较batch128，新增A/B的batch32对照；MIMIC主版本batch128，MIMIC A/B保持暂停。','',
           '| 作业 | batch | 状态 | 阶段 | GPU |','|---|---:|---|---|---:|']
    if queue.get('status')=='paused_by_user':
        lines[2:2]=['全部未完成MWM训练已暂停，不会自动恢复。主方法为latent2x both；GPU用于恢复同参数量baseline。','']
    for r in all_rows:lines.append(f"| {r['id']} | {r['config']['batch_size']} | {r['status']} | {r['phase']} | {r['gpu'] if r['gpu'] is not None else '—'} |")
    lines+=['']+[f"- [{d.upper()}结果]({os.path.relpath(Path(next(j['config']['comparison_output'] for j in manifest['jobs'] if j['dataset']==d))/'summary.md',out)})" for d in ['adni','mimic']]
    lines+=['','[队列](queue.json) · [CUDA检查](architecture_checks.json) · [原始固定配置](manifest.json) · [新增对照配置](additional_jobs.json)','']
    (out/'summary.md').write_text('\n'.join(lines))
