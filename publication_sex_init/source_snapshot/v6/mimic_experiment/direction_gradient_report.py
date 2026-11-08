"""CPU-only summaries of recorded CUDA measurements."""
import json
from pathlib import Path

import numpy as np

from .prepare import atomic


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def stats(rows,key):
    a=np.array([r[key] for r in rows],dtype=float)
    return {'median':float(np.median(a)),'q25':float(np.quantile(a,.25)),'q75':float(np.quantile(a,.75)),
            'min':float(a.min()),'max':float(a.max()),'negative_fraction':float((a<0).mean())}


def generate(out):
    out=Path(out);base=out.parent;index=read(out/'checkpoint_index.json',{'entries':[]});queue=read(out/'queue.json',{})
    entries=sorted(index['entries'],key=lambda x:(x['epoch'],'peak' in x['id']))
    gradients={e['id']:read(out/'gradients'/(e['id']+'.json')) for e in entries}
    sensitivity={e['id']:read(out/'sensitivity'/(e['id']+'.json')) for e in entries}
    replay=read(out/'replay/history.json',[]);original=read(base/'forward_shared/seed_0/history.json',[])
    lines=['## 10. 标准正向模型：历史与未来梯度、向过去积分的敏感性','',
           '### 10.1 从训练现象出发','',
           '本次分析使用原单卡latent2x both标准模型：28维状态、43,280参数、batch=128。原训练在第41轮取得最佳验证结果，第46轮开始出现明显突增，第50轮的平均梯度范数达到135.75；第64—65轮再次恶化。第81轮是后期状态。','',
           'A：multi-cutoff forecast已在第19轮停止，B：masked retrospection已在第46轮停止，检查点和日志保留。本次不再训练这两个消融组。','',
           '### 10.2 如何测量','',
           '将原主重建损失按目标时间拆成历史贡献和未来贡献。每个字段仍除以该batch全部历史与未来的有效观测数，因此两项相加恰好等于原主损失。分别对同一组共享ODE参数求梯度，梯度经过完整历史编码和生成计算图。','',
           r'\[g_H=\nabla_{\theta_f}L_H,\quad g_F=\nabla_{\theta_f}L_F,\qquad C=\frac{g_H^\top g_F}{\|g_H\|\|g_F\|}.\]','',
           '余弦为负，表示两项在这一批数据上倾向于把共享速度场推向相反方向；范数反映每项对速度场施加的梯度强度。所有范数均在梯度裁剪之前计算。固定比较使用8个训练batch、2个验证batch，每个128人；患者、截止点和输入遮蔽在各检查点之间保持一致。下表先列训练batch的统计量。','',
           '状态敏感性取截止点状态到历史时点的ODE映射，求其28×28 Jacobian的最大奇异值。数值为10，表示最敏感方向上的微小状态扰动可放大约10倍。使用固定64位验证患者；向未来对照从同一状态出发，依次使用相同的时间间隔绝对值。另用更严格容差和实际微扰验证结果。','',
           r'\[A_i^- = \left\|D_z\Phi_{t_1-t_c}(z_c)\right\|_2,\qquad A_i^+ = \left\|D_z\Phi_{t_c-t_1}(z_c)\right\|_2.\]','',
           '### 10.3 检查点与执行进度','',
           '原训练保留了第41轮最佳模型与第81轮末尾模型。本次从第41轮的模型、Adam和学习率状态重放第42—65轮，每轮核对原记录的历史、未来、整体验证误差、训练损失和平均梯度范数，差异阈值为1e−5。通过核对后保存关键轮次及该轮梯度最大的实际训练批次状态。','',
           '| 任务 | 物理GPU | 当前状态 |','|---|---:|---|']
    for name,title in [('replay','恢复关键轮次与实际训练梯度'),('gradients','固定患者梯度比较'),('sensitivity','ODE状态敏感性')]:
        j=queue.get('jobs',{}).get(name,{})
        status=read(out/name/'status.json',{}).get('status',j.get('status','等待启动'))
        if j.get('status') in ['failed','complete']:status=j['status']
        if name=='replay' and replay:status+=f"；已完成第{replay[-1]['epoch']}轮核对"
        for english,chinese in [('waiting_for_checkpoint','等待新增检查点'),('running','运行中'),('complete','完成'),('failed','异常退出')]:status=status.replace(english,chinese)
        lines.append(f"| {title} | {j.get('physical_gpu',{'replay':3,'gradients':1,'sensitivity':2}[name])} | {status} |")
    if replay:
        diff=max(max(r['reference_absolute_differences'].values()) for r in replay)
        lines+=['',f'已重放{len(replay)}轮。与原记录的最大绝对差为 **{diff:.3g}**。重放保持整批128人、一次反向传播、无梯度累积；诊断额外使用autograd.grad读取梯度，不执行参数更新。','']
    lines+=['','### 10.4 固定患者：梯度方向与强度','',
            '| 轮次与来源 | 历史梯度范数中位数 | 未来梯度范数中位数 | 历史/未来范数比中位数 | 梯度余弦中位数［四分位区间］ | 余弦为负的batch比例 |',
            '|---|---:|---:|---:|---:|---:|']
    aggregate=[]
    for e in entries:
        data=gradients[e['id']]
        if not data or 'peak' in e['id']:continue
        batch=[r for r in data['batches'] if r['split']=='train'];cos=stats(batch,'history_future_cosine')
        row={'id':e['id'],'epoch':e['epoch'],'history_norm':stats(batch,'history_gradient_norm'),
             'future_norm':stats(batch,'future_gradient_norm'),'ratio':stats(batch,'history_to_future_norm_ratio'),'cosine':cos}
        aggregate.append(row);tag='原检查点' if 'original' in e['id'] else '已核对重放'
        lines.append(f"| {e['epoch']}（{tag}） | {row['history_norm']['median']:.4g} | {row['future_norm']['median']:.4g} | {row['ratio']['median']:.3f} | {cos['median']:.3f}［{cos['q25']:.3f}, {cos['q75']:.3f}］ | {cos['negative_fraction']:.1%} |")
    if not aggregate:lines+=['| 等待CUDA测量 | — | — | — | — | — |']
    lines+=['','每个原始结果文件同时保存全部训练和验证batch、逐批余弦与范数，以及首批中weak prior、Jacobian＋L2、总梯度的分解核对。还计算固定截止点状态后的“仅生成路径梯度”，用于区分生成ODE与完整编码路径的作用。','',
            '### 10.5 实际训练：突增发生在哪些批次','',
            '| 轮次 | 全参数梯度均值 / 最大值 | 最大批次 / 人数 | 该批历史梯度范数 | 该批未来梯度范数 | 该批余弦 | 历史/未来范数比 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    batch_aggregates=[]
    for path in sorted((out/'replay').glob('epoch_*_batch_gradients.json')):
        rows=read(path,[])
        if not rows:continue
        peak=max(rows,key=lambda x:x['all_parameter_preclip_norm']);epoch=int(path.name.split('_')[1])
        mean=float(np.mean([r['all_parameter_preclip_norm'] for r in rows]));batch_aggregates.append({'epoch':epoch,'mean':mean,'peak':peak})
        ref=next(r for r in original if r['epoch']==epoch)
        size=peak.get('actual_batch_size',ref['minimum_batch_size'] if peak['batch']==ref['training_batches'] else 128)
        lines.append(f"| {epoch} | {mean:.4g} / {peak['all_parameter_preclip_norm']:.4g} | {peak['batch']} / {size} | {peak['history_gradient_norm']:.4g} | {peak['future_gradient_norm']:.4g} | {peak['history_future_cosine']:.3f} | {peak['history_to_future_norm_ratio']:.3f} |")
    if not batch_aggregates:lines+=['| 等待重放 | — | — | — | — | — | — |']
    lines+=['','“全参数梯度”包含INIT、UPDATE、Decoder和ODE，用于与原训练曲线核对；历史与未来两列只统计共享ODE参数。每轮169个batch，其中168个128人、末批13人。表中同时记录峰值批次人数。逐批文件还保存真实Adam更新与两项梯度的内积：正值表示这一次ODE参数更新在一阶近似下使该项损失增加。','',
            '### 10.6 向过去积分：状态扰动会放大多少','',
            '| 轮次 | 向过去放大倍数：中位数 / P95 / 最大值 | 向未来对照：中位数 / P95 / 最大值 | 向过去超过10倍的患者比例 |',
            '|---|---:|---:|---:|']
    flow_aggregate=[]
    for e in entries:
        data=sensitivity[e['id']]
        if not data:continue
        c={r['condition']:r for r in data['conditions']};past,future=c['history_tight'],c['matched_future_tight']
        if 'peak' not in e['id']:flow_aggregate.append({'epoch':e['epoch'],'id':e['id'],'past':past,'future':future})
        def fmt(x):return ' / '.join(f"{x['maximum_gain'][k]:.3g}" for k in ['median','p95','max'])
        label=str(e['epoch']) if 'peak' not in e['id'] else f"{e['epoch']}（批次{e['batch']}更新前）"
        lines.append(f"| {label} | {fmt(past)} | {fmt(future)} | {past['fraction_above_10']:.1%} |")
    if not flow_aggregate:lines+=['| 等待CUDA测量 | — | — | — |']
    lines+=['','上表使用rtol=1e−6、atol=1e−8，沿用历史回顾时的各次积分间隔。原训练容差的结果、固定±6/12/24小时结果、每位患者的奇异值，以及相对扰动1e−4和1e−3的实测放大倍数均保存于原始文件。','',
            '### 10.7 根据测量逐步判断','']
    if aggregate:
        first=next((r for r in aggregate if r['epoch']==41),aggregate[0])
        lines.append(f"第{first['epoch']}轮固定训练batch的梯度余弦中位数为{first['cosine']['median']:.3f}，历史/未来梯度范数比中位数为{first['ratio']['median']:.3f}。这给出了突增之前两项监督的关系。")
        lines.append('')
    event=[r for r in aggregate if r['epoch'] in [46,47,48,50,64,65]]
    if event:
        for r in event:
            lines.append(f"第{r['epoch']}轮的梯度余弦中位数为{r['cosine']['median']:.3f}，历史/未来范数比为{r['ratio']['median']:.3f}；历史梯度范数中位数为{r['history_norm']['median']:.4g}，未来为{r['future_norm']['median']:.4g}。")
        lines.append('')
    else:lines+=['突增阶段检查点正在恢复。该阶段的余弦、梯度范数与敏感性将随CUDA任务完成自动写入上表。','']
    if flow_aggregate:
        for r in flow_aggregate:
            if r['epoch'] in [41,46,47,50,64,65,81]:
                lines.append(f"第{r['epoch']}轮向过去积分的扰动放大倍数中位数为{r['past']['maximum_gain']['median']:.3g}，相同跨度向未来为{r['future']['maximum_gain']['median']:.3g}。")
        lines.append('')
    lines+=['将固定患者结果与实际突增批次合起来看：余弦衡量两项监督是否相互牵制，范数衡量哪一项占主导，积分敏感性衡量截止点状态误差沿时间传播的放大。三者共同用于判断本次训练失稳的具体路径。','',
            '### 10.8 图表与完整记录','',
            '[配置、固定患者抽样与源码指纹](gradient_sensitivity_analysis/manifest.json) · [任务与GPU](gradient_sensitivity_analysis/queue.json) · [检查点来源](gradient_sensitivity_analysis/checkpoint_index.json) · [逐轮重放核对](gradient_sensitivity_analysis/replay/history.json)','',
            '[固定患者梯度原始结果](gradient_sensitivity_analysis/gradients/) · [状态敏感性原始结果](gradient_sensitivity_analysis/sensitivity/) · [实际训练逐批梯度](gradient_sensitivity_analysis/replay/)','']
    if aggregate or flow_aggregate:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(13,9))
        axes[0,0].plot([r['epoch'] for r in original],[r['validation']['interpolation']['reconstruction_error'] for r in original],label='History')
        axes[0,0].plot([r['epoch'] for r in original],[r['validation']['forecast']['reconstruction_error'] for r in original],label='Future')
        axes[0,0].set(title='Original validation errors',xlabel='Epoch',ylabel='Error')
        if aggregate:
            x=[r['epoch'] for r in aggregate]
            for key,label in [('history_norm','History'),('future_norm','Future')]:axes[0,1].plot(x,[r[key]['median'] for r in aggregate],'o-',label=label)
            axes[0,1].set(title='Fixed training batches: median field gradient norm',xlabel='Epoch',ylabel='Pre-clip norm',yscale='log')
            axes[1,0].plot(x,[r['cosine']['median'] for r in aggregate],'o-',label='Median cosine')
            axes[1,0].fill_between(x,[r['cosine']['q25'] for r in aggregate],[r['cosine']['q75'] for r in aggregate],alpha=.2)
            axes[1,0].axhline(0,color='gray',lw=1);axes[1,0].set(title='History vs future: field-gradient cosine',xlabel='Epoch',ylabel='Cosine',ylim=(-1,1))
        if flow_aggregate:
            x=[r['epoch'] for r in flow_aggregate]
            for key,label in [('past','Past'),('future','Matched future')]:
                axes[1,1].plot(x,[r[key]['maximum_gain']['median'] for r in flow_aggregate],'o-',label=label)
            axes[1,1].set(title='State perturbation amplification: median',xlabel='Epoch',ylabel='Largest singular value',yscale='log')
        for ax in axes.flat:
            ax.axvspan(46,50,color='orange',alpha=.10);ax.axvspan(64,65,color='red',alpha=.10);ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]:ax.legend()
        fig.tight_layout();fig.savefig(out/'gradient_sensitivity.png',dpi=160);fig.savefig(out/'gradient_sensitivity.pdf');plt.close(fig)
        lines+=['![原训练曲线、固定患者梯度与状态敏感性](gradient_sensitivity_analysis/gradient_sensitivity.png)','']
    atomic(out/'aggregates.json',{'fixed_batch_gradients':aggregate,'actual_training_peak_batches':batch_aggregates,'flow_sensitivity':flow_aggregate})
    text='\n'.join(lines);(out/'summary_section.zh-CN.md').write_text(text)
    (out/'report.zh-CN.md').write_text(text.replace('gradient_sensitivity_analysis/',''))
    from .direction_full_batch_report import generate as main_report
    main_report(base)
