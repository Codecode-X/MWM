"""Joint fixed-alpha report; CPU reads existing scalar artifacts only."""
import json
from pathlib import Path


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def section(root, manifest):
    reference = read(Path(manifest['original_metrics']))
    original = read(Path(manifest['original_history']), [])
    runs = []
    for alpha, tag, gpu in [(0., '00', 1), (.1, '01', 1), (.2, '02', 0), (.3, '03', 3), (.4, '04', 2), (.5, '05', 0)]:
        folder = root/f'update_alpha_{tag}'
        queue = read(folder/'queue.json', {'status': '待启动'})
        metrics = read(folder/'metrics_150.json') or read(folder/'metrics.json')
        if (folder/'stop_request.json').exists() and not metrics: queue['status']='stopped_by_user'
        if (folder/'epoch_cap_stop.json').exists(): queue['status']='complete_at_150' if metrics else 'stopped_at_150_pending_evaluation'
        runs.append({'alpha': alpha, 'tag': tag, 'gpu': gpu, 'folder': folder,
                     'queue': queue,
                     'history': read(folder/'history.json', []), 'metrics': metrics,
                     'checks': read(folder/'checks.json'), 'initial': read(folder/'initialization_check.json')})
    if not any(r['folder'].exists() for r in runs): return []
    lines = ['### 8.4 实验三：固定α=0—0.5，比较UPDATE的修正幅度', '',
             '模块交叉实验显示，ODE轨迹和UPDATE增益会共同影响编码递归的敏感性。本组保留原UPDATE网络，让它先给出状态修正建议，再按固定比例与观测校正前的状态混合。', '',
             r'\[z_k^+=(1-\alpha)z_k^-+\alpha U_\phi(z_k^-,y_k).\]', '',
             '| 系数 | 保留校正前状态 | 采用UPDATE输出 | 运行时物理GPU |', '|---|---:|---:|---|',
             '| α=0（新增：不执行UPDATE） | 100% | 0% | CUDA 1 |',
             '| α=0.1 | 90% | 10% | CUDA 1 |',
             '| α=0.2（新增） | 80% | 20% | CUDA 0 |',
             '| α=0.3（原消融） | 70% | 30% | CUDA 3 |',
             '| α=0.4（新增） | 60% | 40% | CUDA 2 |',
             '| α=0.5 | 50% | 50% | CUDA 0（该组已完成） |',
             '| α=1（原标准模型） | 0% | 100% | 已完成的seed 0实验 |', '',
             'α在各组中固定，不增加可学习参数。原Linear→Tanh→Linear的UPDATE层和宽度保留，每次历史UPDATE都使用上述公式。各组仍为28维潜在状态、43,280个参数。', '',
             'α>0的组保留完整历史梯度，经过INIT、UPDATE和编码ODE。α=0直接令校正后状态等于ODE演化后的状态，观测值不再改变状态；INIT、共享ODE和Decoder仍训练，UPDATE网络不执行。所有组保留原重建损失、0.01 weak prior、0.01 ODE Jacobian、0.02 ODE L2及字段掩码和归一化。深度8消融单独记录在8.5。', '',
             'α=0仍保留原网络的参数键，文件中总参数为43,280；其中16,087个UPDATE参数不参与计算，实际参与任务的参数为27,193。该组用于检验吸收观测的作用。', '',
             '| 共同训练项目 | 设置 |', '|---|---|',
             '| 初始化与环境 | ode环境；seed 0从零训练，与原模型初始权重一致 |',
             '| batch / 梯度累积 | 128 / 无；末批13人 |',
             '| 优化与求解器 | 原Adam与学习率衰减；梯度裁剪10；Tsit5，rtol=1e−4，atol=1e−6 |',
             '| 当前预算与选模 | 新实验最多150轮，至少50轮，验证整体误差连续40轮无改善时停止；选择验证整体误差最低轮次 |',
             '| 测试 | 载入最佳检查点，分别评价向过去历史回顾和向未来自由预测；同时记录6h/12h历史上下文测试 |',
             '| 监测 | SwanLab、损失分项、验证曲线、裁剪前梯度范数、超过100的batch数 |', '',
             '对于固定输入状态和观测，局部Jacobian满足：', '',
             r'\[J_{\mathrm{UPDATE}}=(1-\alpha)I+\alpha J_U,\qquad \|J_{\mathrm{UPDATE}}\|_2\leq(1-\alpha)+\alpha\|J_U\|_2.\]', '',
             '较小的α让单次校正更接近保留原状态。各组仍会经历多次ODE演化与UPDATE，因此通过完整训练比较递归稳定性和预测效果。', '',
             '150轮上限在本次新增实验时开始执行。此前的α=0.1、α=0.3已按要求中断，实际训练轮数保留；α=0.2、α=0.4将在150轮停止并测试该区间的最佳检查点。历史结果不改写为150轮实验。', '',
             '[本轮轮次政策](targeted_causal_ablation/epoch_policy.json) · [150轮停止与测试队列](targeted_causal_ablation/epoch_cap_watch.json)', '',
             '#### CUDA公式与梯度核验', '',
             '| α | 状态 | Jacobian公式最大绝对差（float64） | 原网络参数/键一致 | 历史与未来保留编码器梯度 |',
             '|---|---|---:|---|---|']
    for run in runs:
        checks = run['checks']
        row = [f"{run['alpha']:g}", checks['status'] if checks else '待核验',
               f"{checks['jacobian_formula_max_absolute_error_float64']:.3g}" if checks else '—',
               '是' if checks and checks['same_parameter_hash_and_state_dict_keys'] else '—',
               ('是（INIT/ODE；无UPDATE）' if run['alpha']==0 else '是') if checks and not checks['history_encoder_gradient_stopped'] else '—']
        lines.append('| '+' | '.join(row)+' |')
    lines += ['', '核验同时检查α=1时完整输出与原损失逐位一致、α=0时返回原输入状态，以及各系数下的混合公式和梯度。核验期间不执行参数更新，通过后开始正式训练。', '',
              '#### 从零训练的验证比较', '',
              '| 模型 | 状态 | 最新 / 最佳轮次 | 验证历史 | 验证未来 | 验证整体 |', '|---|---|---:|---:|---:|---:|']
    groups = [('原标准模型 α=1', reference, original, 'complete')]
    groups += [(f"固定α={r['alpha']:g}", r['metrics'], r['history'], r['queue']['status']) for r in runs]
    for label, result, hs, status in groups:
        best = min(hs, key=lambda h: h['validation']['overall']['reconstruction_error']) if hs else None
        selected = result['validation'] if result else best['validation'] if best else None
        count = result['epochs_run'] if result else len(hs)
        epoch = result['best_epoch'] if result else best['epoch'] if best else '—'
        row = [label, status, f'{count} / {epoch}']
        row += [f"{selected[k]['reconstruction_error']:.6f}" if selected else '—' for k in ['interpolation', 'forecast', 'overall']]
        lines.append('| '+' | '.join(row)+' |')
    lines += ['', '验证列对应各组已完成轮次内验证整体误差最低的检查点。α=0使用新150轮上限；既有长预算结果保留实际轮次，不与150轮结果混称同预算。第45轮后的干预续训保留在8.2。', '',
              '#### 最佳检查点测试比较', '',
              '| 模型 | 最佳轮次 | 测试历史 | 测试未来 | 测试整体 | 数值标准化MSE |', '|---|---:|---:|---:|---:|---:|']
    for label, result, _, _ in groups:
        test = result['test'] if result else None
        row = [label, str(result['best_epoch']) if result else '—']
        row += [f"{test[k]['reconstruction_error']:.6f}" if test else '—' for k in ['interpolation', 'forecast', 'overall']]
        row += [f"{test['mean_standardized_numeric_mse']:.6f}" if test else '—']
        lines.append('| '+' | '.join(row)+' |')
    lines += ['', '继续运行和新增的组在结束后载入最佳检查点并自动填写测试列；用户中断的旧组保留已有验证记录。', '',
              '#### 训练梯度与验证波动', '',
              '| α | 已完成轮次 | 累计最大裁剪前batch梯度 | 梯度超过100的batch数 | 验证整体误差最大单轮上升倍率 |', '|---|---:|---:|---:|---:|']
    for run in runs:
        hs = run['history']
        maximum = max(h['maximum_batch_gradient_norm'] for h in hs) if hs else None
        count = sum(h['batches_above_100'] for h in hs) if hs else None
        jump = max(hs[i]['validation']['overall']['reconstruction_error']/hs[i-1]['validation']['overall']['reconstruction_error'] for i in range(1, len(hs))) if len(hs) > 1 else None
        row = [f"{run['alpha']:g}", str(len(hs)), f'{maximum:.6g}' if maximum is not None else '—',
               str(count) if count is not None else '—', f'{jump:.6g}' if jump is not None else '—']
        lines.append('| '+' | '.join(row)+' |')
    lines += ['', '梯度计数包含每个真实训练batch。验证波动与历史、未来测试误差分别记录，便于同时判断稳定性和预测性能。', '']
    if any(r['history'] for r in runs):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for name, hs in [('original alpha=1', original)]+[(f"alpha={r['alpha']:g}", r['history']) for r in runs]:
            if not hs: continue
            for ax, key in zip(axes.flat, ['interpolation', 'forecast', 'overall', None]):
                ax.plot([h['epoch'] for h in hs], [h['validation'][key]['reconstruction_error'] if key else h['training_components']['gradient_norm'] for h in hs], label=name)
        for ax, title in zip(axes.flat, ['Validation history', 'Validation future', 'Validation overall', 'Mean pre-clip batch gradient']):
            ax.set(xlabel='Epoch', title=title); ax.grid(alpha=.2); ax.legend(fontsize=8)
        axes[1, 1].set_yscale('log'); fig.tight_layout()
        fig.savefig(root/'update_alpha_sweep_curves.png', dpi=170); plt.close(fig)
        lines += ['![α=0—0.5与原模型：验证误差及轮均梯度](targeted_causal_ablation/update_alpha_sweep_curves.png)', '']
    lines += ['#### 每组原始记录', '']
    for run in runs:
        prefix = f"targeted_causal_ablation/update_alpha_{run['tag']}"
        lines += [f"- α={run['alpha']:g}：[配置与源码指纹]({prefix}/manifest.json) · [运行状态]({prefix}/queue.json) · [CUDA核验]({prefix}/checks.json) · [初始化核对]({prefix}/initialization_check.json) · [训练记录]({prefix}/)"]
        if (run['folder']/'comparison_to_alpha03.json').exists():
            lines[-1] += f" · [与α=0.3配置逐项比较]({prefix}/comparison_to_alpha03.json)"
    audited = [r for r in runs if (r['folder']/'comparison_to_alpha03.json').exists()]
    if audited:
        labels = '、'.join(f"α={r['alpha']:g}" for r in audited)
        lines += ['', f'{labels}的配置逐项核对已通过：与α=0.3组相比，仅输出目录、物理GPU和update_alpha不同，其余训练配置一致。']
    lines += ['', 'α>0的旧组使用各组冻结damped_update_model.py构造；新增α=0使用fixed_update_model.py，并在零系数时直接跳过UPDATE网络。检查点配置中的update_alpha记录实际值。', '',
              '```bash', 'bash scripts/run_mimic_damped_update_sweep.sh --alpha 0.5 --gpu 0',
              'bash scripts/run_mimic_damped_update_sweep.sh --alpha 0.1 --gpu 1',
              'bash scripts/run_mimic_damped_update_sweep.sh --alpha 0.2 --gpu 0',
              'bash scripts/run_mimic_damped_update_sweep.sh --alpha 0.4 --gpu 2',
              '# 原α=0.3组', 'bash scripts/run_mimic_damped_update.sh', '```', '']
    return lines
