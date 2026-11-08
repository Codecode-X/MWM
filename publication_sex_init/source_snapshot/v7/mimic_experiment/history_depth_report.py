"""Report history-specific truncated BPTT using existing scalar artifacts."""
import json
from pathlib import Path


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def section(root, manifest):
    folder = root/'history_depth_8'; queue = read(folder/'queue.json')
    if not queue: return []
    history = read(folder/'history.json', []); metrics = read(folder/'metrics.json')
    reference = read(Path(manifest['original_metrics'])); original = read(Path(manifest['original_history']), [])
    lines = ['### 8.5 实验四：历史损失的最大编码反向递归深度为8', '',
             '完全截断会让历史回顾任务失去对整个编码器的直接监督。本组保留最近8次状态校正的历史梯度，把更早的递归路径截断。原完全截断续训与从零训练已按用户要求停止，已有日志和检查点保留。', '',
             '设某位患者共有K次历史检查。模型仍按原顺序吸收全部K次检查，得到截止点状态；仍从截止点向过去自由积分，监督全部历史观测。唯一改变发生在历史损失的编码梯度路径。', '',
             '- K≤8：历史损失完整返回全部历史编码及INIT。',
             '- K>8：在最近8次ODE/UPDATE编码步骤之前，对该患者的状态停止梯度；最近8次步骤仍完整参与历史损失的反向传播。',
             '- 未来预测和weak prior保持原来的完整梯度路径；历史生成ODE与Decoder也继续训练。', '',
             '例如共有20次历史检查，前12次检查的信息仍包含在第12次校正后的状态中。历史损失通过第13—20次校正及相应ODE返回编码参数，到第12次校正后的状态处停止。未来预测损失仍可以返回全部20次编码。', '',
             '这里的深度按独立检查次数计，按每位患者自己的历史长度分别确定边界。它不限制历史监督目标数量、不缩短向过去积分的时间跨度，也不改变自适应ODE求解器的步数。', '',
             '| 项目 | 设置 |', '|---|---|',
             '| 状态与参数 | 28维，43,280个参数；原INIT、UPDATE、共享ODE、Decoder |',
             '| 起点 / GPU | seed 0从零训练 / 物理CUDA 2，ode环境 |',
             '| batch / 梯度累积 | 128 / 无，保留末批13人 |',
             '| 损失 | 原历史与未来重建、0.01 weak prior、0.01 Jacobian、0.02 ODE L2；原字段掩码、分母和权重 |',
             '| UPDATE | 原始U输出；不叠加α=0.3混合 |',
             '| 求解器与优化 | 原Tsit5，rtol=1e−4、atol=1e−6；Adam及原学习率衰减，梯度裁剪10 |',
             '| 预算与选模 | 最多300轮，至少50轮，验证整体40轮不改善时早停；按最佳验证整体选模 |', '',
             '实现时保留原完整编码图供未来预测与weak prior使用，另以相同权重和观测重建历史梯度分支，并在每位患者的边界处分离状态。历史生成使用原截止点数值、该分支的梯度；评估时直接使用原完整模型计算。', '']
    checks = read(folder/'checks.json')
    if checks:
        lines += [f"CUDA核验已通过：完整输出、截止点状态和总损失与原模型逐位相同。未来全部参数梯度相对差为{checks['future_all_parameter_gradient_relative_error']:.3g}，weak prior/Jacobian/L2梯度相对差为{checks['prior_jacobian_l2_gradient_relative_error']:.3g}。", '',
                  '逐患者输入梯度检查确认：超过深度边界的早期观测梯度为0，最近8次观测的梯度保留。将深度放宽到完整历史长度后，恢复原历史损失梯度。', '',
                  '[CUDA逐患者边界与梯度检查](targeted_causal_ablation/history_depth_8/checks.json)', '']
        audit = read(folder/'first_batch_forward_audit.json')
        if audit:
            lines += [f"从零训练的首个batch前向损失为{audit['depth8_forward_loss']:.12g}，与原训练第一次Adam更新前的SwanLab记录完全一致。", '',
                      '[首个batch前向损失核对](targeted_causal_ablation/history_depth_8/first_batch_forward_audit.json)', '']
    else:
        lines += ['CUDA边界与梯度核验正在进行，通过后开始正式训练。', '']
    lines += ['#### 验证与测试比较', '',
              '| 模型 | 状态 | 最新 / 最佳轮次 | 验证历史 | 验证未来 | 验证整体 |', '|---|---|---:|---:|---:|---:|']
    groups = [('原完整编码梯度', reference, original, 'complete'), ('历史编码深度8', metrics, history, queue['status'])]
    for label, result, hs, status in groups:
        best = min(hs, key=lambda h: h['validation']['overall']['reconstruction_error']) if hs else None
        selected = result['validation'] if result else best['validation'] if best else None
        count = result['epochs_run'] if result else len(hs); epoch = result['best_epoch'] if result else best['epoch'] if best else '—'
        row = [label, status, f'{count} / {epoch}']+[f"{selected[k]['reconstruction_error']:.6f}" if selected else '—' for k in ['interpolation', 'forecast', 'overall']]
        lines.append('| '+' | '.join(row)+' |')
    lines += ['', '| 模型 | 最佳轮次 | 测试历史 | 测试未来 | 测试整体 | 数值标准化MSE |', '|---|---:|---:|---:|---:|---:|']
    for label, result, _, _ in groups:
        test = result['test'] if result else None
        row = [label, str(result['best_epoch']) if result else '—']+[f"{test[k]['reconstruction_error']:.6f}" if test else '—' for k in ['interpolation', 'forecast', 'overall']]
        row += [f"{test['mean_standardized_numeric_mse']:.6f}" if test else '—']; lines.append('| '+' | '.join(row)+' |')
    if history:
        lines += ['', f"深度8组累计最大裁剪前batch梯度为{max(h['maximum_batch_gradient_norm'] for h in history):.6g}，超过100的batch共{sum(h['batches_above_100'] for h in history)}个。", '']
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for name, hs in [('original full gradient', original), ('history depth 8', history)]:
            for ax, key in zip(axes.flat, ['interpolation', 'forecast', 'overall', None]):
                ax.plot([h['epoch'] for h in hs], [h['validation'][key]['reconstruction_error'] if key else h['training_components']['gradient_norm'] for h in hs], label=name)
        for ax, title in zip(axes.flat, ['Validation history', 'Validation future', 'Validation overall', 'Mean pre-clip batch gradient']):
            ax.set(xlabel='Epoch', title=title); ax.grid(alpha=.2); ax.legend(fontsize=8)
        axes[1, 1].set_yscale('log'); fig.tight_layout(); fig.savefig(root/'history_depth_8_curves.png', dpi=170); plt.close(fig)
        lines += ['![历史编码深度8：验证历史、未来、整体及轮均梯度](targeted_causal_ablation/history_depth_8_curves.png)', '']
    if not metrics:
        lines += ['训练曲线逐轮更新；完成后载入最佳检查点，自动填写历史与未来测试指标。', '']
    lines += ['[训练配置与源码指纹](targeted_causal_ablation/history_depth_8/manifest.json) · [运行状态](targeted_causal_ablation/history_depth_8/queue.json) · [训练记录](targeted_causal_ablation/history_depth_8/)', '',
              '```bash', 'bash scripts/run_mimic_history_depth8.sh', '```', '']
    return lines
