"""CPU-only report for the two targeted interventions."""
import json
import fcntl
from pathlib import Path

import numpy as np

MARKER='<!-- latent2x-targeted-causal -->'


def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default


def generate(out):
    out=Path(out)
    with (out/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _generate(out)


def _generate(out):
    out=Path(out);queue=read(out/'queue.json',{});check=read(out/'checks.json');manifest=read(out/'manifest.json',{})
    histories={key:read(out/key/'history.json',[]) for key in ['control','history_stop']}
    metrics={key:read(out/key/'metrics.json') for key in histories}
    lines=['## 8. 针对性因果消融：历史梯度截断与模块参数交叉','',
           '### 8.1 从机制证据到这两个实验','',
           '上一节把异常定位到历史损失经过截止点状态返回编码器的高增益路径。本轮先改变这一条梯度边，检查训练是否恢复稳定；再固定观测、交叉ODE与UPDATE权重，区分权重变化与状态位置变化的作用。','',
           '本轮结果集中在[targeted_causal_ablation](targeted_causal_ablation/)。原模型、旧检查点、旧日志和第7节诊断结果全部保留。','',
           '### 8.2 实验一：只截断历史损失返回编码器的梯度','',
           r'\[L_H^{\mathrm{stop}}=\ell\left(g_\psi\left(\Phi_\theta(-T,\operatorname{sg}(z_c))\right),y_H\right).\]','',
           '历史仍完整地向过去自由积分；这部分损失仍更新共享ODE与Decoder。只有历史损失经过截止点返回INIT、UPDATE和编码ODE的路径被截断。未来预测保留完整梯度；0.01 weak prior、0.01 ODE Jacobian与0.02 ODE L2沿用原实现。UPDATE仍由未来损失与weak prior监督。','',
           '数值MSE和类别CE的字段掩码、字段分母、权重均不变。历史与未来仍共用原batch全部有效字段的分母。验证和测试照常从截止点向过去回顾、向未来预测。','',
           '| 项目 | 控制组 | history stop-gradient组 |','|---|---|---|',
           '| 起点 | 第45轮末完整模型、Adam、调度器与RNG | 同一份完整状态 |',
           '| 唯一改变 | 原历史梯度路径 | 历史生成起点使用z_c.detach() |',
           '| 状态 / 参数量 | 28维 / 43,280 | 相同 |',
           '| batch / 梯度累积 | 128 / 无，末批保留原13人 | 相同 |',
           '| 训练范围 | 第46—81轮，共36轮 | 相同 |',
           '| 早停 | 本轮采用固定36轮 | 相同 |',
           '| 选模 | 第46—81轮验证整体误差最低 | 相同 |',
           '| 物理GPU | 0 | 1 |','',
           '这是同一稳定起点后的成对续训。它覆盖原第46轮和第64—65轮的两次波动，直接检查训练路径的作用。两组训练曲线按相同全局轮次并排显示；原完整实验选中的第41轮另列为既有参照。','',
           '起点来自第7节已核验训练重放保存的第45轮末完整状态。本轮两组从该文件恢复模型、Adam、调度器与RNG；控制组还逐轮核对原日志，并核对第46轮第121个batch更新前的参数和输入。原训练最初未保存的RNG，仍沿用第7节说明的重建与核验方式。','',
           '#### 实现检查','']
    if check:
        lines += [f"在已保存的异常batch上，两组前向预测与损失数值逐位相同。stop-gradient后，历史损失对INIT和UPDATE的梯度范数为{check['history_init_update_gradient_norm']:.3g}；对共享ODE仍为{check['stopped_history_field_gradient_norm']:.5g}，对Decoder仍为{check['stopped_history_decoder_gradient_norm']:.5g}。",'',
                  f"未来项全部参数梯度相对差为{check['future_all_parameter_gradient_relative_error']:.3g}；prior、Jacobian、L2梯度相对差为{check['prior_jacobian_l2_gradient_relative_error']:.3g}。同一参数点的ODE总梯度范数由{check['original_total_field_gradient_norm']:.5g}变为{check['stopped_total_field_gradient_norm']:.5g}。",'',
                  '[CUDA梯度核验](targeted_causal_ablation/checks.json)','']
    else:lines += ['CUDA梯度一致性检查正在准备。','']
    lines += ['#### 续训进度与验证结果','',
              '| 模型 | 当前状态 | 最新轮次 / 最佳轮次 | 验证历史 | 验证未来 | 验证整体 | 最大训练batch梯度 |','|---|---|---:|---:|---:|---:|---:|']
    for key,label in [('control','原路径控制组'),('history_stop','历史梯度截断组')]:
        history=histories[key];best=min(history,key=lambda r:r['validation']['overall']['reconstruction_error']) if history else None
        status=queue.get('jobs',{}).get(key,{}).get('status','等待开始')
        if (out/key/'stop_request.json').exists():status='stopped_by_user'
        row=[label,status,f"{history[-1]['epoch'] if history else 45} / {best['epoch'] if best else '—'}"]
        row += [f"{best['validation'][r]['reconstruction_error']:.6f}" if best else '—' for r in ['interpolation','forecast','overall']]
        row += [f"{max(r['maximum_batch_gradient_norm'] for r in history):.5g}" if history else '—']
        lines.append('| '+' | '.join(row)+' |')
    lines+=['','验证列使用各组第46轮之后的最佳检查点。最大梯度是裁剪前、包含全部模型参数的最大真实训练batch范数。超过100用于统计梯度峰值；是否出现训练失稳，结合验证误差的变化判断。','',
            '#### 历史回顾与未来预测的测试比较','',
            '| 模型 | 选中轮次 | 测试历史 | 测试未来 | 测试整体 | 数值标准化MSE |','|---|---:|---:|---:|---:|---:|']
    reference=read(Path(manifest.get('original_metrics','/nonexistent')))
    for key,label,result in [('original','原完整实验最佳（既有参照）',reference)]+[(key,label,metrics[key]) for key,label in [('control','原路径续训控制组'),('history_stop','历史梯度截断续训')]]:
        t=result['test'] if result else None;vals=[label,str(result['best_epoch']) if result else '—']
        vals += [f"{t[r]['reconstruction_error']:.6f}" if t else '—' for r in ['interpolation','forecast','overall']]
        vals += [f"{t['mean_standardized_numeric_mse']:.6f}" if t else '—'];lines.append('| '+' | '.join(vals)+' |')
    if all(metrics.values()):
        control,stop=metrics['control'],metrics['history_stop'];h=stop['test']['interpolation']['reconstruction_error']-control['test']['interpolation']['reconstruction_error'];f=stop['test']['forecast']['reconstruction_error']-control['test']['forecast']['reconstruction_error']
        lines += ['',f"截断组相对同起点控制组，测试历史误差差值为{h:+.6f}，测试未来误差差值为{f:+.6f}。两项需分别评价。",'']
        lines+=['| 第46—81轮稳定性 | 原路径控制组 | 历史梯度截断组 |','|---|---:|---:|']
        for title,fn in [('最大裁剪前batch梯度',lambda hs:max(r['maximum_batch_gradient_norm'] for r in hs)),
                         ('梯度超过100的batch数',lambda hs:sum(r['batches_above_100'] for r in hs)),
                         ('验证整体误差最大单轮上升倍率',lambda hs:max(hs[i]['validation']['overall']['reconstruction_error']/hs[i-1]['validation']['overall']['reconstruction_error'] for i in range(1,len(hs))))]:
            lines.append(f"| {title} | {fn(histories['control']):.6g} | {fn(histories['history_stop']):.6g} |")
        if stop['maximum_training_batch_gradient']<100<=control['maximum_training_batch_gradient']:
            lines+=['','在这个同起点、同预算续训区间内，截断组没有复现控制组超过100的训练梯度峰值。历史与未来测试误差继续按上表分别判断，训练稳定性与预测性能分开回答。','']
        elif stop['maximum_training_batch_gradient']>=100:
            lines += ['',f"切断历史损失返回编码器后，续训中仍出现超过100的梯度峰值，最大为{stop['maximum_training_batch_gradient']:.6g}。因此，这次第45轮后的干预没有消除全部梯度异常。峰值大小、验证波动以及历史和未来测试结果按上表分别评价。",'']
    elif (out/'history_stop/stop_request.json').exists():
        lines += ['','历史梯度完全截断组已按用户要求停止，已有记录保留；原路径控制组继续。新版本改为历史编码梯度最大递归深度8，见8.5。被停止组不再继续生成测试指标。','']
    else:lines += ['','测试指标将在各组完成36轮续训并重新载入最佳检查点后填写。','']
    if any(histories.values()):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for key,hist in histories.items():
            x=[r['epoch'] for r in hist]
            for ax,region in zip(axes.flat,['interpolation','forecast','overall',None]):
                y=[r['validation'][region]['reconstruction_error'] if region else r['maximum_batch_gradient_norm'] for r in hist]
                ax.plot(x,y,'o-',ms=3,label=key)
        for ax,title in zip(axes.flat,['Validation history','Validation future','Validation overall','Maximum real batch gradient: pre-clip']):
            ax.set(xlabel='Global epoch',title=title);ax.grid(alpha=.2);ax.legend()
        axes[1,1].set_yscale('log');fig.tight_layout();fig.savefig(out/'continuation_curves.png',dpi=170);plt.close(fig)
        lines += ['![成对续训曲线：历史、未来、整体与真实训练梯度](targeted_causal_ablation/continuation_curves.png)','']
    fresh=read(out/'fresh_history_stop/history.json',[]);fresh_metrics=read(out/'fresh_history_stop/metrics.json');fresh_queue=read(out/'fresh_history_stop/queue.json')
    if fresh_queue:
        best=min(fresh,key=lambda h:h['validation']['overall']['reconstruction_error']) if fresh else None
        lines+=['#### 全程截断：从seed 0初始化开始','',
                '第45轮后的干预检验“在已训练模型上切断路径是否足够”。本组另外从seed 0初始化开始，全程使用同一stop-gradient版本，检验完整训练策略。已有原始seed 0训练作为对照，避免重复训练相同控制模型。','',
                '这组恢复原完整训练预算：最多300轮、至少50轮、验证整体误差40轮未改善时早停。batch=128、无梯度累积、Adam、学习率衰减、所有损失权重与原模型相同，使用GPU 2。构造时核对两个版本的初始参数逐项相同。','',
                '| 组别 | 当前状态 | 已训练 / 最佳轮次 | 验证历史 | 验证未来 | 验证整体 |','|---|---|---:|---:|---:|---:|']
        if (out/'fresh_history_stop/stop_request.json').exists():fresh_queue['status']='stopped_by_user'
        for label,result,history,status in [('原始从零训练',reference,[], 'complete'),('全程历史梯度截断',fresh_metrics,fresh,fresh_queue['status'])]:
            selected=result['validation'] if result else best['validation'] if best else None
            epoch_count=result['epochs_run'] if result else len(history);best_epoch=result['best_epoch'] if result else best['epoch'] if best else '—'
            vals=[label,status,f'{epoch_count} / {best_epoch}']+[f"{selected[r]['reconstruction_error']:.6f}" if selected else '—' for r in ['interpolation','forecast','overall']]
            lines.append('| '+' | '.join(vals)+' |')
        lines+=['','| 从零训练测试比较 | 最佳轮次 | 测试历史 | 测试未来 | 测试整体 | 数值标准化MSE |','|---|---:|---:|---:|---:|---:|']
        for label,result in [('原始模型',reference),('全程截断历史梯度',fresh_metrics)]:
            test=result['test'] if result else None;values=[label,str(result['best_epoch']) if result else '—']+[f"{test[r]['reconstruction_error']:.6f}" if test else '—' for r in ['interpolation','forecast','overall']]+[f"{test['mean_standardized_numeric_mse']:.6f}" if test else '—']
            lines.append('| '+' | '.join(values)+' |')
        lines+=['','[从零初始化一致性](targeted_causal_ablation/fresh_history_stop/initialization_check.json) · [从零训练配置](targeted_causal_ablation/fresh_history_stop/manifest.json) · [训练记录](targeted_causal_ablation/fresh_history_stop/)','']
        if (out/'fresh_history_stop/stop_request.json').exists():
            lines += ['本组已按用户要求停止，由8.5的历史梯度深度8组替代。从零训练的最后完整轮次、最佳验证指标和曲线保留在此。','']
        first_batch=read(out/'fresh_history_stop/first_batch_forward_audit.json')
        if first_batch:
            lines += [f"原训练与全程截断组在第一次Adam更新前的batch前向损失均为{first_batch['original_first_batch_forward_loss']:.12g}，差值为{first_batch['absolute_difference']:.3g}。初始参数核验与这一条原始SwanLab记录一致；首次更新后，梯度路径不同会使训练轨迹逐渐分开。",'',
                      '[首个训练batch前向损失核对](targeted_causal_ablation/fresh_history_stop/first_batch_forward_audit.json)','']
        if fresh:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            original=read(Path(manifest['original_history']),[]);fig,axes=plt.subplots(1,3,figsize=(15,4.5))
            for name,hs in [('original',original),('history_stop_from_seed0',fresh)]:
                for ax,key in zip(axes,['interpolation','forecast',None]):
                    ax.plot([h['epoch'] for h in hs],[h['validation'][key]['reconstruction_error'] if key else h['training_components']['gradient_norm'] for h in hs],label=name)
            for ax,title in zip(axes,['Validation history','Validation future','Mean pre-clip batch gradient']):ax.set(xlabel='Epoch',title=title);ax.grid(alpha=.2);ax.legend(fontsize=7)
            axes[2].set_yscale('log');fig.tight_layout();fig.savefig(out/'fresh_training_curves.png',dpi=170);plt.close(fig)
            lines+=['![从零训练：历史、未来与轮均梯度](targeted_causal_ablation/fresh_training_curves.png)','']
    lines += ['### 8.3 实验二：ODE与UPDATE权重的2×2交叉','',
              '观测与输入遮蔽固定为第46轮第121个实际训练batch，测量已定位的患者索引8。主对照固定第45轮INIT与其余模块，只交换ODE与UPDATE；附加对照将INIT统一固定为异常快照，再重复同样四组。Decoder始终固定为稳定版本，不参与编码Jacobian计算。','',
              '| 组合 | ODE权重 | UPDATE权重 |','|---|---|---|','| A | 稳定 | 稳定 |','| B | 异常 | 稳定 |','| C | 稳定 | 异常 |','| D | 异常 | 异常 |','',
              '每个组合分别记录截止状态对各次UPDATE后状态的敏感性，以及截止状态对ODE参数的Jacobian最大奇异值。参数值来自已有快照，计算采用CUDA float64、Tsit5、rtol=1e−8、atol=1e−10，不执行优化步骤。','']
    crossed={ctx:read(out/'module_cross'/ctx/'results.json') for ctx in ['stable','abnormal']}
    for ctx,title in [('stable','固定稳定INIT：主对照'),('abnormal','固定异常INIT：初始化背景对照')]:
        data=crossed[ctx];lines += ['#### '+title,'','| 组合 | 编码ODE参数Jacobian最大奇异值 | 各历史节点至截止点最大状态增益 | INIT至截止点增益 |','|---|---:|---:|---:|']
        if not data:lines+=['| 等待CUDA计算 | — | — | — |',''];continue
        for key,row in data['conditions'].items():
            lines.append(f"| {key} | {row['field_parameter_jacobian_max_singular_value']:.6g} | {row['maximum_postupdate_gain']:.6g} | {row['initial_state_gain']:.6g} |" if row['status']=='complete' else f"| {key}：求解失败 | — | — | — |")
        effect=data['factorial_effects'].get('field_parameter_jacobian_max_singular_value')
        if effect:lines += ['',f"ODE单独交换的比值B/A={effect['B_over_A']:.4g}；UPDATE单独交换的比值C/A={effect['C_over_A']:.4g}；共同交换D/A={effect['D_over_A']:.4g}。乘性交互比DA/(BC)={effect['multiplicative_interaction_DA_over_BC']:.4g}：1表示在该指标上等于两种单独倍率的乘积。",'']
        lines+=['**固定UPDATE输入，再交换UPDATE权重。** 从A、B、C、D各自轨迹保存校正前状态与相同观测，冻结这些张量，分别应用稳定和异常UPDATE。下表列该患者20次检查中的最大局部状态增益。','',
                '| 固定输入取自哪条轨迹 | 稳定UPDATE权重 | 异常UPDATE权重 |','|---|---:|---:|']
        for bank,r in data['fixed_input_update_comparison'].items():lines.append(f"| {bank}轨迹 | {r['stable']['maximum']:.5g} | {r['abnormal']['maximum']:.5g} |")
        lines.append('')
        fixed=data['fixed_input_update_comparison'];times=data['history_times_hours']
        if all(k in fixed for k in ['A','B','D']):
            lines+=['同一INIT背景下，四条轨迹在首次检查前的输入相同，表中部分最大值也出现在这一时点。因此，下面继续按检查时间比较局部增益，观察后续轨迹分开之后的变化。','',
                    '| 时间h | A输入＋稳定U | A输入＋异常U | B输入＋稳定U | B输入＋异常U | D输入＋稳定U | D输入＋异常U |','|---|---:|---:|---:|---:|---:|---:|']
            indices=sorted({i for i,t in enumerate(times) if t in [14,16,18,20,22,23,24]})
            for i in indices:
                vals=[fixed[bank][u]['per_visit_gain'][i] for bank in ['A','B','D'] for u in ['stable','abnormal']]
                lines.append('| '+f'{times[i]:g}'+' | '+' | '.join(f'{v:.4g}' for v in vals)+' |')
            lines.append('')
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,2,figsize=(13,4.5))
            for name,r in data['conditions'].items():
                if r['status']=='complete':axes[0].plot(times,r['postupdate_gains'],'o-',ms=3,label=name)
            for bank in ['A','B','D']:
                for u,style in [('stable','--'),('abnormal','-')]:axes[1].plot(times,fixed[bank][u]['per_visit_gain'],ls=style,label=bank+' inputs / '+u+' U')
            axes[0].set(yscale='log',title='Cutoff sensitivity to post-update state',xlabel='Hours',ylabel='Largest singular value')
            axes[1].set(title='UPDATE Jacobian on frozen inputs',xlabel='Hours',ylabel='Largest singular value')
            for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
            fig.tight_layout();fig.savefig(out/f'module_cross_{ctx}.png',dpi=180);fig.savefig(out/f'module_cross_{ctx}.pdf');plt.close(fig)
            lines+=['![模块交叉与固定UPDATE输入：'+title+'](targeted_causal_ablation/module_cross_'+ctx+'.png)','']
        if effect:
            lines += [f"在同一组A轨迹输入上换UPDATE权重，可直接比较权重的局部作用；在稳定UPDATE下比较A、B输入，可比较只改变ODE后落点与路径的作用。全递归D与B、C的组合关系由交互比记录，和局部UPDATE指标一起解释。",'']
    if all(crossed.values()):
        main=crossed['stable']['factorial_effects'].get('field_parameter_jacobian_max_singular_value')
        alt=crossed['abnormal']['factorial_effects'].get('field_parameter_jacobian_max_singular_value')
        if main and alt:
            d_main=crossed['stable']['conditions']['D']['field_parameter_jacobian_max_singular_value'];d_alt=crossed['abnormal']['conditions']['D']['field_parameter_jacobian_max_singular_value']
            lines += ['#### 如何理解这组交叉结果','',
                      f"固定稳定INIT时，单独换ODE仅为A的{main['B_over_A']:.2f}倍，单独换UPDATE为{main['C_over_A']:.2f}倍；同时换两者达到{main['D_over_A']:.2f}倍，超过两种单独倍率的乘积{main['multiplicative_interaction_DA_over_BC']:.2f}倍。完整递归的异常不能只按单项局部增益理解。",'',
                      f"将INIT背景统一固定为异常版本后，A、B、C仍处在相近量级，D由{d_main:.5g}升至{d_alt:.5g}，约增大{d_alt/d_main:.2f}倍。初始化背景对共同异常配置有很强的调节作用。",'',
                      '固定输入的逐点对照同时显示UPDATE权重的局部作用与状态位置的作用。结合A—D整体递归，当前证据支持ODE、UPDATE和轨迹位置的强非线性交互，不能只归结为UPDATE权重变化或ODE落点变化中的单一项。','']
    lines += ['混合权重可能形成训练中没有出现过的轨迹。本实验给出固定背景下的模块敏感性及交互，不将混合组合解释为自然训练轨迹，也不作为完整因果识别。','']
    import importlib
    from . import damped_update_sweep_report
    lines += importlib.reload(damped_update_sweep_report).section(out, manifest)
    from . import history_depth_report
    lines += importlib.reload(history_depth_report).section(out, manifest)
    from . import adni_fixed_update_report
    lines += importlib.reload(adni_fixed_update_report).section(out)
    from . import alpha_direction_report
    lines += importlib.reload(alpha_direction_report).section(out)
    lines += ['### 8.8 原始记录与复现','',
              '当前α=0.1主模型的完整等参数模块消融，统一记录于[Ablation文档](../../../ablation/Ablation.zh-CN.md)。','',
              '[实验配置与源码指纹](targeted_causal_ablation/manifest.json) · [运行队列及物理GPU](targeted_causal_ablation/queue.json) · [CUDA梯度检查](targeted_causal_ablation/checks.json)','',
              '[控制组完整记录](targeted_causal_ablation/control/) · [截断组完整记录](targeted_causal_ablation/history_stop/) · [固定稳定INIT交叉结果](targeted_causal_ablation/module_cross/stable/results.json) · [固定异常INIT交叉结果](targeted_causal_ablation/module_cross/abnormal/results.json)','',
              '```bash','bash scripts/run_mimic_targeted_causal_ablation.sh','bash scripts/run_mimic_history_stop_fresh.sh','```','',
              '两项命令可分别启动并行运行：第一项管理成对续训及模块交叉，第二项管理从零初始化的stop-gradient训练。','']
    section='\n'.join(lines);(out/'summary_section.zh-CN.md').write_text(section)
    (out/'summary.md').write_text(section.replace('targeted_causal_ablation/',''))
    target=out.parent/'summary.md';original=target.read_text().split(MARKER)[0].rstrip();target.write_text(original+'\n\n'+MARKER+'\n\n'+section)

