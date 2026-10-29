"""Compare paired CUDA capacity experiments, without selecting on test outcomes."""
import argparse
import json
from pathlib import Path
import numpy as np


def aggregate(values):
    return {'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else None,
            'n': len(values)}


def formatted(value):
    return f"{value['mean']:.4f} ± {value['std']:.4f}" if value['std'] is not None else f"{value['mean']:.4f}"


def summarize(root):
    root = Path(root)
    report = {}
    lines = ['# GPU 参数量匹配实验', '',
             '固定患者划分 seed=42；训练种子默认 0–4；均值 ± 样本标准差（ddof=1）。',
             '所有训练、验证和测试均在 CUDA 上完成。验证 AUROC 选模，验证 F1 选阈值。', '',
             '| 损失 | 模型 | 参数量 | Test AUROC | Test AP | Test Brier |',
             '|---|---|---:|---:|---:|---:|']
    for objective in ('native', 'bce'):
        base = json.loads((root / f'{objective}_base/summary.json').read_text())
        matched = json.loads((root / f'{objective}_matched/summary.json').read_text())
        configs = [json.loads((root / f'{objective}_{c}/protocol.json').read_text()) for c in ('base', 'matched')]
        common = [{k: v for k, v in c.items() if k not in ('capacity', 'models', 'output')} for c in configs]
        assert common[0] == common[1], 'Unequal training/evaluation protocol'
        assert json.loads((root / f'{objective}_base/data.json').read_text()) == json.loads((root / f'{objective}_matched/data.json').read_text())
        by_variant = {}
        for label, name, source in [('旧模型', 'legacy_transformer', base),
                ('Latent ODE 小', 'latent_ode', base), ('Latent ODE 扩大', 'latent_ode', matched),
                ('ODE-RNN 小', 'ode_rnn', base), ('ODE-RNN 扩大', 'ode_rnn', matched)]:
            entry = source['summary'][name]
            rows = [r for r in source['runs'] if r['model'] == name]
            assert all(r['runtime']['tensor_device'].startswith('cuda') for r in rows)
            by_variant[label] = entry
            fields = ' | '.join(formatted(entry['test'][k]) for k in ('auroc', 'ap', 'brier'))
            lines.append(f"| {objective} | {label} | {entry['parameters']:,} | {fields} |")
        pairs = {}
        for name in ('latent_ode', 'ode_rnn'):
            large = {r['seed']: r for r in matched['runs'] if r['model'] == name}
            for reference in (name, 'legacy_transformer'):
                small = {r['seed']: r for r in base['runs'] if r['model'] == reference}
                assert large.keys() == small.keys()
                pairs[f'{name} matched minus {reference} base'] = {
                    metric: aggregate([large[s]['test'][metric] - small[s]['test'][metric] for s in sorted(large)])
                    for metric in ('auroc', 'ap', 'brier')}
        report[objective] = {'variants': by_variant, 'paired_differences': pairs}
    lines += ['', '## 同种子配对差值', '', '| 损失 | 对比 | ΔAUROC | ΔAP | ΔBrier |', '|---|---|---:|---:|---:|']
    for objective, group in report.items():
        for label, values in group['paired_differences'].items():
            lines.append(f"| {objective} | {label} | " + ' | '.join(formatted(values[k]) for k in ('auroc', 'ap', 'brier')) + ' |')
    lines += ['', '差值为扩大模型减参照模型。AUROC/AP 越高越好，Brier 越低越好。',
              '这里只估计固定划分上的训练随机性；15 名测试患者不能支持稳定的总体泛化结论。',
              '总参数量接近不代表结构、有效容量或优化难度相同。BCE 下 Latent ODE 的生成 ODE 与解码器不参与分类梯度。',
              '20 epoch 是相同训练预算，并不保证扩大模型已经收敛。扩容结果不能单独证明性能差距的唯一原因。']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    variants = [('Legacy', 'legacy_transformer', 'base'),
                ('Latent small', 'latent_ode', 'base'),
                ('Latent matched', 'latent_ode', 'matched'),
                ('ODE-RNN small', 'ode_rnn', 'base'),
                ('ODE-RNN matched', 'ode_rnn', 'matched')]
    for ax, objective in zip(axes, ('native', 'bce')):
        for x, (label, model, capacity) in enumerate(variants):
            rows = json.loads((root / f'{objective}_{capacity}/summary.json').read_text())['runs']
            scores = [r['test']['auroc'] for r in sorted(rows, key=lambda r: r['seed']) if r['model'] == model]
            value = aggregate(scores)
            ax.errorbar(x, value['mean'], yerr=value['std'] or 0., fmt='o', capsize=5, color='tab:blue')
            ax.scatter(x + np.linspace(-.10, .10, len(scores)), scores, s=15, alpha=.6, color='tab:orange')
        ax.set_xticks(range(len(variants)), [v[0] for v in variants], rotation=25, ha='right')
        ax.set_title(objective.upper() + ': mean ± sample SD; dots = training seeds')
        ax.axhline(.5, color='gray', linestyle='--', linewidth=.7)
        ax.set_ylim(0, 1.07)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Test AUROC')
    fig.tight_layout()
    fig.savefig(root / 'capacity_auroc.png', dpi=180)
    fig.savefig(root / 'capacity_auroc.pdf')
    plt.close(fig)
    (root / 'capacity_summary.json').write_text(json.dumps(report, indent=2))
    (root / 'capacity_summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output')
    summarize(p.parse_args().output)
