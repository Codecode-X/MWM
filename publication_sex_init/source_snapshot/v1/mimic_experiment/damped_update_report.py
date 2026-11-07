"""Existing scalar artifacts only; no model loading or evaluation."""
import json
from pathlib import Path


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def section(root, manifest):
    folder = root/'update_alpha_03'
    queue = read(folder/'queue.json')
    if not queue:
        return []
    history = read(folder/'history.json', [])
    metrics = read(folder/'metrics.json')
    reference = read(Path(manifest['original_metrics']))
    original_history = read(Path(manifest['original_history']), [])
    stop_history = read(root/'fresh_history_stop/history.json', [])
    stop_metrics = read(root/'fresh_history_stop/metrics.json')
    lines = ['### 8.4 实验三：固定α=0.3，减小每次UPDATE的修正幅度', '',
             '上一项交叉实验发现，ODE轨迹与UPDATE局部增益会共同影响编码递归的敏感性。本组保留原UPDATE网络，把它的输出视为一个状态修正建议，每次只采用其中30%，其余70%保留观测校正前的状态。', '',
             r'\[z_k^+=0.7z_k^-+0.3U_\phi(z_k^-,y_k).\]', '',
             'α固定为0.3，不参与学习。原Linear→Tanh→Linear的层、宽度和参数全部保留，总参数量仍为43,280，潜在状态仍为28维。第一次及后续每次历史UPDATE均使用这一公式，ODE积分方式保持原样。', '',
             '这组独立检验UPDATE混合的作用：历史回顾损失仍经过截止点返回INIT、UPDATE及编码ODE，未来预测也保持完整梯度。0.01 weak prior、0.01 ODE Jacobian、0.02 ODE L2及历史/未来重建损失的掩码、分母和权重，均沿用原标准模型。', '',
             '| 训练项目 | 设置 |', '|---|---|',
             '| 物理GPU / Conda环境 | CUDA 3 / ode |',
             '| 初始化 | seed 0，从零训练；与原模型初始权重一致 |',
             '| batch / 梯度累积 | 128 / 无；末批13人 |',
             '| 优化与求解器 | 原Adam与学习率衰减；梯度裁剪10；Tsit5，rtol=1e−4，atol=1e−6 |',
             '| 预算与选模 | 最多300轮，至少50轮，验证整体误差连续40轮无改善时停止；选择验证整体误差最低轮次 |',
             '| 测试 | 载入最佳检查点，分别评价向过去历史回顾、向未来自由预测 |',
             '| 监测 | SwanLab、逐轮损失分项、验证曲线、裁剪前梯度范数及超过100的batch数 |', '',
             '在相同输入状态与观测下，新的局部Jacobian为：', '',
             r'\[J_{\mathrm{UPDATE}}=0.7I+0.3J_U,\qquad \|J_{\mathrm{UPDATE}}\|_2\leq0.7+0.3\|J_U\|_2.\]', '',
             '这一形式减小网络修正项在单次更新中的占比。完整编码还包含多次ODE演化与UPDATE，其最终稳定性由下面的训练结果检验。', '']
    checks = read(folder/'checks.json')
    if checks:
        lines += [f"CUDA实现检查已通过：α=1时，完整模型的输出和原损失与原模型逐位相同；α=0时，UPDATE直接返回输入状态。α=0.3时，混合公式逐位一致，float64下自动微分Jacobian与0.7I+0.3J_U的最大绝对差为{checks['jacobian_formula_max_absolute_error_float64']:.3g}。", '',
                  '历史和未来损失对INIT、UPDATE、共享ODE、Decoder均有非零梯度。检查期间未执行参数更新。', '',
                  '[CUDA公式及梯度检查](targeted_causal_ablation/update_alpha_03/checks.json)', '']
    else:
        lines += ['CUDA实现检查与训练正在准备。', '']
    lines += ['#### 从零训练的验证比较', '',
              '| 模型 | 状态 | 最新 / 最佳轮次 | 验证历史 | 验证未来 | 验证整体 |',
              '|---|---|---:|---:|---:|---:|']
    groups = [('原始UPDATE（α=1）', reference, original_history, 'complete'),
              ('历史梯度截断（独立对照）', stop_metrics, stop_history, read(root/'fresh_history_stop/queue.json', {}).get('status', '等待')),
              ('固定α=0.3 UPDATE', metrics, history, queue['status'])]
    for label, result, hs, status in groups:
        best = min(hs, key=lambda h: h['validation']['overall']['reconstruction_error']) if hs else None
        selected = result['validation'] if result else best['validation'] if best else None
        epoch = result['best_epoch'] if result else best['epoch'] if best else '—'
        count = result['epochs_run'] if result else len(hs)
        row = [label, status, f'{count} / {epoch}']
        row += [f"{selected[key]['reconstruction_error']:.6f}" if selected else '—' for key in ['interpolation', 'forecast', 'overall']]
        lines.append('| '+' | '.join(row)+' |')
    lines += ['', '以上均为各组最佳验证整体误差检查点的指标。第45轮后的干预续训另见8.2，不混入从零训练表。', '',
              '#### 最佳检查点测试比较', '',
              '| 模型 | 最佳轮次 | 测试历史 | 测试未来 | 测试整体 | 数值标准化MSE |',
              '|---|---:|---:|---:|---:|---:|']
    for label, result, _, _ in groups:
        test = result['test'] if result else None
        row = [label, str(result['best_epoch']) if result else '—']
        row += [f"{test[key]['reconstruction_error']:.6f}" if test else '—' for key in ['interpolation', 'forecast', 'overall']]
        row += [f"{test['mean_standardized_numeric_mse']:.6f}" if test else '—']
        lines.append('| '+' | '.join(row)+' |')
    if history:
        lines += ['', f"α=0.3组当前已完成{history[-1]['epoch']}轮；累计最大裁剪前batch梯度为{max(h['maximum_batch_gradient_norm'] for h in history):.6g}，超过100的batch共{sum(h['batches_above_100'] for h in history)}个。", '']
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for label, _, hs, _ in groups:
            if not hs:
                continue
            name = {'原始UPDATE（α=1）': 'original alpha=1', '历史梯度截断（独立对照）': 'history stop-gradient', '固定α=0.3 UPDATE': 'damped alpha=0.3'}[label]
            for ax, key in zip(axes.flat, ['interpolation', 'forecast', 'overall', None]):
                ax.plot([h['epoch'] for h in hs], [h['validation'][key]['reconstruction_error'] if key else h['training_components']['gradient_norm'] for h in hs], label=name)
        for ax, title in zip(axes.flat, ['Validation history', 'Validation future', 'Validation overall', 'Mean pre-clip batch gradient']):
            ax.set(xlabel='Epoch', title=title); ax.grid(alpha=.2); ax.legend(fontsize=8)
        axes[1, 1].set_yscale('log'); fig.tight_layout(); fig.savefig(root/'update_alpha_03_curves.png', dpi=170); plt.close(fig)
        lines += ['![固定α=0.3消融：验证历史、未来、整体及轮均梯度](targeted_causal_ablation/update_alpha_03_curves.png)', '']
    if metrics:
        dh = metrics['test']['interpolation']['reconstruction_error']-reference['test']['interpolation']['reconstruction_error']
        df = metrics['test']['forecast']['reconstruction_error']-reference['test']['forecast']['reconstruction_error']
        lines += [f"相对原始从零训练，α=0.3的测试历史误差差值为{dh:+.6f}，测试未来误差差值为{df:+.6f}。完整训练的梯度曲线与两项测试指标共同用于判断收益。", '']
    else:
        lines += ['测试列在训练结束、重新载入最佳检查点后自动填写。', '']
    lines += ['[配置与源码指纹](targeted_causal_ablation/update_alpha_03/manifest.json) · [运行状态](targeted_causal_ablation/update_alpha_03/queue.json) · [训练记录](targeted_causal_ablation/update_alpha_03/) ', '',
              '本组检查点保留原网络的state_dict键。单独载入时，使用本组冻结源码damped_update_model.py中的make_damped_model构造模型，并保持alpha=0.3；这会同时恢复权重和本组的状态更新公式。', '',
              '```bash', 'bash scripts/run_mimic_damped_update.sh', '```', '']
    return lines
