"""Plot individual training seeds and mean ± sample SD for benchmark protocols."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots', nargs='+')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    fig, axes = plt.subplots(len(args.roots), 3, figsize=(12, 3.6 * len(args.roots)), squeeze=False)
    names = ['legacy_transformer', 'latent_ode', 'ode_rnn']
    labels = ['Transformer\n(corrected)', 'Latent ODE', 'ODE-RNN']
    for row, root in enumerate(args.roots):
        root = Path(root)
        report = json.loads((root / 'summary.json').read_text())
        protocol = json.loads((root / 'protocol.json').read_text())
        expected = {(m, s) for m in protocol['models'] for s in protocol['seeds']}
        assert {(r['model'], r['seed']) for r in report['runs']} == expected, 'incomplete comparison'
        for ax, metric in zip(axes[row], ['auroc', 'ap', 'brier']):
            for index, name in enumerate(names):
                values = np.array([r['test'][metric] for r in report['runs'] if r['model'] == name])
                if not len(values):
                    continue
                ax.scatter(index + np.linspace(-.10, .10, len(values)), values, alpha=.7, s=28)
                ax.errorbar(index, values.mean(), yerr=values.std(ddof=1) if len(values) > 1 else 0.,
                            fmt='D', color='black', capsize=5, markersize=5)
            ax.set_xticks(range(3), labels)
            ax.set_xlim(-.5, 2.5)
            if metric != 'brier':
                ax.set_ylim(0, 1.12)
            ax.set_ylabel({'auroc': 'AUROC (higher is better)', 'ap': 'AP (higher is better)',
                           'brier': 'Brier (lower is better)'}[metric])
            ax.set_title(f"{protocol['objective']}: {len(protocol['seeds'])} training seeds, {protocol['epochs']} epochs")
            ax.grid(axis='y', alpha=.25)
    fig.suptitle('Same patient split; points = independent training runs; bars = mean ± sample SD')
    fig.tight_layout(rect=(0, 0, 1, .96))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
