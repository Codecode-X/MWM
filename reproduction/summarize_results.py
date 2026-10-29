"""Collect completed experiments without selecting or tuning on test results."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    records = []
    for root in args.roots:
        for path in sorted(Path(root).glob('*/metrics.json')):
            metrics = json.loads(path.read_text())
            config = json.loads((path.parent / 'config.json').read_text())
            records.append({'experiment': str(path.parent), 'model': config['model'],
                            'task': config['task'], 'classification_weight': config['classification_weight'],
                            'epochs': config['epochs'], 'seed': config['seed'], **metrics})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(records, indent=2))
    lines = ['# Synthetic ICU experiments', '',
             'Validation MSE selects the checkpoint. MSE is in training-set min-max units.', '',
             '| Experiment | Best epoch | Initial val MSE | Best val MSE | Test MSE | Test AUROC | Test AUPRC |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in records:
        test = row['test']
        auc = f"{test['auroc']:.4f}" if 'auroc' in test else '—'
        ap = f"{test['auprc']:.4f}" if 'auprc' in test else '—'
        lines.append(f"| {row['experiment']} | {row['best_epoch']} | {row['baseline_validation']['mse']:.6f} | "
                     f"{row['validation']['mse']:.6f} | {test['mse']:.6f} | "
                     f"{auc} | {ap} |")
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if records:
        fig, axes = plt.subplots(len(records), 1, figsize=(8, max(3, len(records) * 2.4)), squeeze=False)
        for ax, row in zip(axes[:, 0], records):
            history = json.loads((Path(row['experiment']) / 'history.json').read_text())
            ax.plot([h['epoch'] for h in history], [h['validation']['mse'] for h in history])
            ax.axvline(row['best_epoch'], color='gray', linestyle='--', linewidth=1)
            ax.set(title=row['experiment'], xlabel='Epoch', ylabel='Validation MSE')
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(out / 'validation_curves.png', dpi=150)
        plt.close(fig)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
