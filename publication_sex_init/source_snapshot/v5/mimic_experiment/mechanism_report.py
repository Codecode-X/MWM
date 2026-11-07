"""CPU-only reporting; consumes CUDA-produced JSON without loading a model."""
import json
from pathlib import Path

import numpy as np


MARKER='<!-- latent2x-mechanism-diagnosis -->'


def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default


def generate(out):
    import importlib
    from . import mechanism_plots,mechanism_interpretation
    importlib.reload(mechanism_plots);importlib.reload(mechanism_interpretation)
    out=Path(out);q=read(out/'queue.json',{});audit=read(out/'recovery_audit.json',{})
    trace=read(out/'experiment1/status.json',{});sens=read(out/'experiment1/sensitivity_status.json',{})
    records=[read(p) for p in sorted((out/'experiment1/steps').glob('e*.json'))]
    measured=[read(p) for p in sorted((out/'experiment1/sensitivity').glob('e*.json'))]
    precision=[read(p) for p in sorted((out/'experiment2').glob('*/results.json'))]
    patients=[read(p) for p in sorted((out/'experiment3').glob('*/results.json'))]
    thresholds=read(out/'experiment1/thresholds.json',{})
    last=records[-1] if records else trace
    phases={'P0_experiment1_update_trace':'逐更新重放与固定患者敏感性','P0_experiment2_solver_convergence':'求解精度对照','P1_experiment3_patient_decomposition':'患者级路径分解','finished':'三项主实验完成'}
    plot_links,statistics=mechanism_plots.graphics(out,records,measured,precision,patients)
    findings=mechanism_interpretation.interpret(out,records,measured,precision,patients,statistics)
    lines=['## 7. 历史反向积分失稳：逐更新、精度与患者级机制诊断','',
           '### 7.1 已有证据与这次要区分的问题','',
           '原方向实验第10节位于[方向比较文档](../direction_full_batch_cuda/summary.md)。第46轮第121个真实训练batch中，历史与未来共享ODE梯度范数分别为95.68和4.177，余弦为−0.693。该参数快照下，固定验证患者向过去的状态放大倍数中位数为88.9，向未来为3.89。稳定检查点的固定batch没有显示持续增强的整体梯度冲突。weak_prior=0时仍出现波动，移除历史回顾的future-only组则稳定。','',
           '本次依次检查：哪次参数更新改变了敏感性；数值积分精度是否足够；哪些患者和哪条梯度路径贡献了异常。模型仍为28维、43,280参数，原损失与权重全部保留。','',
           '### 7.2 恢复能力与执行边界','',
           '原训练保留了第41轮与第81轮检查点；第45轮与第46轮第121个batch的快照来自此前核对过逐轮记录的训练重放。第41轮和第45轮检查点含模型、Adam与调度器，没有保存RNG。异常batch快照含参数和训练输入，没有Adam。原代码每轮重置Python、NumPy、torch与CUDA种子；本次据此重建随机序列，逐batch核对此前重放记录，并单独核对原SwanLab实际记录的步骤。','',
           '**本次称为“按原逐轮种子重建、经过逐batch核对的训练重放”。原始RNG快照缺失，原运行全部未记录步骤的因果起点无法逐项审计；更新前后的因果比较对应这条重建序列。** 新重放每次更新前后保存完整参数、Adam、调度器和RNG。“真实训练batch”指训练重放实际使用的完整batch，和额外固定的验证患者诊断分开标注。全部模型计算使用ode环境和RTX 4090；只有标记为training_replay的进程执行optimizer.step。','',
           '原训练使用CUDA Tsit5、rtol=1e−4、atol=1e−6，直接autograd经过积分计算图，误差控制器的步长决策被detach。上一轮敏感性测量使用1e−6/1e−8。新精度实验分别保留这两档，并加入更严容差、float64和独立固定步长RK4。','',
           f"当前阶段：{phases.get(q.get('phase'),'准备')}；训练重放完成到第{last.get('epoch','—')}轮第{last.get('batch','—')}个batch；已测量{sens.get('measured_updates',0)}次更新的敏感性。",'',
           '### 7.3 实验一：更新前后的敏感性','',
           '使用先前固定64位验证患者抽样中的前8位，保持相同历史输入与时间间隔。A每次以当前参数重新编码；B固定第41轮截止点状态。连续测量每次更新前后的历史/未来敏感性、真实训练梯度、裁剪与Adam更新幅度。阈值预先取稳定阶段第41、42、45轮共192个固定患者测量的P99，另标记单次更新增加5倍。完整连续值同时保留。','']
    if thresholds:lines += [f"稳定参考P99阈值为 **{thresholds['stable_p99']:.6g}**。",'']
    if records:
        original=[d for r in records for d in r['original_sampled_log_absolute_differences'].values()]
        previous=[d for r in records for d in r['prior_replay_absolute_differences'].values()]
        lines += [f"已保存{len(records)}个真实训练步骤。原SwanLab已记录的对应标量共核对{len(original)}项，最大绝对差{max(original,default=0.):.3g}；已有逐batch重放记录最大差{max(previous,default=0.):.3g}。",'']
    reproduction=read(out/'experiment1/abnormal_batch_reproduction.json')
    if reproduction:lines += [f"第46轮第121个batch更新前，与已有异常快照的最大参数差为 **{reproduction['parameter_max_absolute_difference']:.3g}**，输入张量逐项相同：**{all(reproduction['batch_tensors_equal'].values())}**。",'']
    resumed=read(out/'experiment1/resume_validation.json')
    if resumed:
        lines += ['本次新保存的完整状态已通过一次恢复检查：从第45轮第40个batch后的RNG、Adam和调度器恢复，再执行第41个batch，参数、Adam、调度器和RNG均与中断前保存的同一步结果逐项相同。[恢复核对记录](mechanism_diagnosis/experiment1/resume_validation.json)。','']
    lines += ['| 测量方式 | 首次跨过稳定P99 | 首次单步增加5倍 | 最大单步增长 |','|---|---|---|---:|']
    for key,title in [('A_reencoded','A：重新编码历史'),('B_fixed_state','B：固定截止点状态')]:
        crossing=[r for r in measured if r['flags'][key]['crosses_stable_p99']];jump=[r for r in measured if r['flags'][key]['single_update_fivefold']]
        def location(xs):return f"第{xs[0]['step']['epoch']}轮 / batch {xs[0]['step']['batch']}" if xs else '当前已测范围内未出现' if measured else '待测'
        growth=max((r['flags'][key]['after_before_ratio'] for r in measured),default=0.)
        growth_text=f'{growth:.4g}' if measured else '—'
        lines.append(f"| {title} | {location(crossing)} | {location(jump)} | {growth_text} |")
    lines+=['']+findings.get('onset',[])
    lines += ['','### 7.4 实验二：积分与梯度收敛','',
              '固定患者、输入和目标。每个参数快照分别使用固定第41轮状态与该快照重新编码后冻结的状态；同一状态在全部求解设置中保持一致。float64由保存的float32权重数值精确转换，不训练新权重。独立RK4从0.2小时逐次减半；最近两档所有关键量的相对差小于0.001时判为步长收敛，必要时细化到0.0125小时。原Tsit5在512步内无法达到指定容差时记录失败。','',
              '| 参数快照 | 状态锚点 | RK4步长收敛 | 1e−6/1e−8状态相对差 | 反向最大奇异值相对差 | 历史状态梯度相对差 | 历史ODE直接梯度相对差 |','|---|---|---|---:|---:|---:|']
    for item in precision:
        for anchor in item['convergence']:
            row=next((r for r in item['settings'] if r['anchor']==anchor['anchor'] and r['setting']=='diagnostic32' and r['status']=='complete'),None)
            diff=row.get('relative_differences_to_reference',{}) if row else {}
            f=lambda k:f'{diff[k]:.3g}' if k in diff else '—'
            lines.append(f"| {item['snapshot']['id']} | {anchor['anchor']} | {anchor['step_halving_converged']} | {f('past_endpoint')} | {f('past_singular_value')} | {f('history_state_gradient')} | {f('history_field_gradient')} |")
    if not precision:lines.append('| 等待实验一完成后运行 | — | — | — | — | — | — |')
    lines+=['']+findings.get('precision',[])
    lines += ['','原始文件同时保存原训练容差、全部严格容差、float32/float64、终点状态、完整梯度向量、求解步数，以及6/12/24小时往返误差。往返误差衡量数值可逆精度。','',
              '### 7.5 实验三：患者贡献与梯度链式分解','',
              '对异常真实batch、稳定阶段真实batch，以及相同异常batch在稳定参数下的配对诊断分别计算。患者贡献使用原始完整batch的字段分母，患者独立均值另存；既报告患者梯度范数，也报告其沿batch总历史梯度方向的投影贡献。','',
              '将历史ODE参数梯度拆成“固定截止点状态的生成直接路径”与“经截止点状态返回历史编码ODE的路径”。单个最早历史目标另测Decoder端梯度、实际状态梯度放大和流Jacobian奇异值；多历史目标的Decoder梯度按堆叠状态统计。','',
              r'\[\nabla_{\theta_f}L_H=\left.\nabla_{\theta_f}L_H\right|_{z_c\,\mathrm{fixed}}+\left(\frac{\partial z_c}{\partial\theta_f}\right)^\top\nabla_{z_c}L_H.\]','',
              '第一项沿共享ODE向过去生成；第二项通过截止点状态，返回此前按时间顺序吸收历史的ODE＋UPDATE链。两条路径使用同一套ODE参数。','',
              '| Case | 历史ODE梯度范数 | 生成直接路径 | 编码路径 | 患者梯度求和相对差 |','|---|---:|---:|---:|---:|']
    for item in patients:
        labels={'abnormal_real_e46b121':'异常真实batch：46轮121批','stable_matched_same_patients':'同一输入＋45轮稳定参数','stable_real_e45b121':'稳定真实batch：45轮121批','first_clipped_e46_b082':'首次裁剪batch：46轮82批'}
        b=item['batch'];label=labels.get(item['case']['id'],item['case']['id']);lines.append(f"| {label} | {b['history_gradient_norm']:.5g} | {b['direct_generation_norm']:.5g} | {b['encoder_path_norm']:.5g} | {item['patient_gradient_sum_relative_error']:.3g} |")
    if not patients:lines.append('| 等待前两项完成后运行 | — | — | — | — |')
    early=read(out/'early_patient_queue.json')
    if early:lines+=['',f"另以原训练裁剪阈值10定位第46轮首次触发裁剪的真实batch，待P0完成后在GPU 0诊断。当前状态：`{early['status']}`。[该对照的选择与设备记录](mechanism_diagnosis/early_patient_queue.json)。",'']
    lines+=['']+findings.get('patients',[])
    lines+=findings.get('followup',[])
    lines += ['','### 7.6 机制判断表','',
              '| 假设 | 关键证据 | 支持程度 |','|---|---|---|',
              '| H1：真实ODE流敏感性增长 | '+findings['grades']['H1'][0]+' | '+findings['grades']['H1'][1]+' |',
              '| H2：ODE数值误差 | '+findings['grades']['H2'][0]+' | '+findings['grades']['H2'][1]+' |',
              '| H3：少数患者或时间跨度主导 | '+findings['grades']['H3'][0]+' | '+findings['grades']['H3'][1]+' |',
              '| H4：其他梯度传播因素 | '+findings['grades']['H4'][0]+' | '+findings['grades']['H4'][1]+' |','',
              '### 7.7 文件、设备和图表','',
              '[恢复能力审计](mechanism_diagnosis/recovery_audit.json) · [固定配置与代码指纹](mechanism_diagnosis/manifest.json) · [任务与GPU](mechanism_diagnosis/queue.json)','',
              '[逐更新记录及快照](mechanism_diagnosis/experiment1/) · [精度对照](mechanism_diagnosis/experiment2/) · [患者级结果](mechanism_diagnosis/experiment3/)','']
    lines+=['| 审阅对象 | 真实来源 |','|---|---|',
            '| 模型结构与共享关系 | [冻结模型代码](../direction_full_batch_cuda/source_snapshot/paper_odeae/models.py) |',
            '| 原始训练损失与Adam步骤 | [冻结训练代码](../direction_full_batch_cuda/source_snapshot/mimic_experiment/train_full_batch.py) |',
            '| 异常batch更新前完整模型、Adam、调度器、RNG | [e046_b121_pre.pt](mechanism_diagnosis/experiment1/snapshots/e046_b121_pre.pt) |',
            '| 对应更新后完整状态 | [e046_b121_post.pt](mechanism_diagnosis/experiment1/snapshots/e046_b121_post.pt) |',
            '| 异常真实训练输入与患者索引 | [batch文件](mechanism_diagnosis/experiment1/batches/e046_b121.pt) |',
            '| 固定8位验证患者及其时间 | [患者审计JSON](mechanism_diagnosis/experiment1/fixed_cohort.json) · [输入与固定潜在状态](mechanism_diagnosis/experiment1/fixed_inputs_and_states.pt) |',
            '| GPU、源码、核对结果 | [完成审计](mechanism_diagnosis/completion_audit.json) · [各项复现命令](mechanism_diagnosis/reproduction.json) |','']
    lines+=findings.get('conclusion',[])
    for label,path in plot_links:lines+=['!['+label+'](mechanism_diagnosis/'+path+')','']
    # Charts are statistical summaries of already recorded CUDA results.
    if records:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        x=[r['optimizer_step'] for r in records];fig,axes=plt.subplots(2,2,figsize=(13,8))
        for key,label in [('history_gradient_norm','History'),('future_gradient_norm','Future')]:axes[0,0].plot(x,[r['before'][key] for r in records],label=label)
        axes[0,0].set(yscale='log',title='Real training batch field gradients',ylabel='Pre-clip norm')
        axes[0,1].plot(x,[r['before']['history_future_cosine'] for r in records]);axes[0,1].axhline(0,color='gray');axes[0,1].set(title='History / future gradient cosine',ylim=(-1,1))
        if measured:
            sx=[(r['step']['epoch']-1)*169+r['step']['batch'] for r in measured]
            for key,title in [('A_reencoded','A: re-encoded'),('B_fixed_state','B: fixed state')]:
                axes[1,0].plot(sx,[r['after'][key]['past']['gain']['median'] for r in measured],label=title)
                axes[1,1].plot(sx,[r['after'][key]['past_future_ratio_median'] for r in measured],label=title)
            axes[1,0].axhline(thresholds['stable_p99'],color='gray',ls='--',label='Stable P99')
            axes[1,0].set(yscale='log',title='After-update past sensitivity: fixed patients')
            axes[1,1].set(yscale='log',title='Past / future sensitivity ratio')
        for ax in axes.flat:
            ax.set_xlabel('Global optimizer step');ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]:ax.legend()
        fig.tight_layout();fig.savefig(out/'update_trace.png',dpi=160);plt.close(fig)
        lines += ['![逐更新梯度与状态敏感性](mechanism_diagnosis/update_trace.png)','']
    section='\n'.join(lines);(out/'summary_section.zh-CN.md').write_text(section)
    (out/'report.zh-CN.md').write_text(section.replace('mechanism_diagnosis/',''))
    target=out.parent/'summary.md';text=target.read_text().split(MARKER)[0].rstrip()
    content=text+'\n\n'+MARKER+'\n\n'+section
    causal=out.parent/'targeted_causal_ablation/summary_section.zh-CN.md'
    if causal.exists():content+='\n\n<!-- latent2x-targeted-causal -->\n\n'+causal.read_text()
    target.write_text(content)
