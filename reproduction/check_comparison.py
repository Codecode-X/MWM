"""Audit the full paired-seed comparison without training or tuning on test data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.comparison import classification_metrics, choose_threshold


def audit(root):
    root = Path(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    data = json.loads((root / 'data.json').read_text())
    assert data['data_sha256'] == hashlib.sha256(Path(protocol['data']).read_bytes()).hexdigest()
    groups = data['patient_ids']
    combined = [pid for ids in groups.values() for pid in ids]
    assert len(set(combined)) == len(combined), 'split overlap'
    summary = json.loads((root / 'summary.json').read_text())
    by_pair = {(r['model'], r['seed']): r for r in summary['runs']}
    expected = {(m, s) for m in protocol['models'] for s in protocol['seeds']}
    assert set(by_pair) == expected and len(summary['runs']) == len(expected), 'incomplete or duplicate runs'
    for model, seed in sorted(expected):
        path = root / model / f'seed_{seed}'
        checkpoint = torch.load(path / 'best.pt', map_location='cpu', weights_only=False)
        metrics = json.loads((path / 'metrics.json').read_text())
        predictions = json.loads((path / 'predictions.json').read_text())
        history = json.loads((path / 'history.json').read_text())
        assert checkpoint['metadata'] == data
        assert checkpoint['config'] == {**protocol, 'model': model, 'training_seed': seed}
        assert len(history) == protocol['epochs']
        assert [h['epoch'] for h in history] == list(range(1, protocol['epochs'] + 1))
        chosen = max(history, key=lambda h: (h['validation']['auroc'], h['validation']['ap']))
        assert chosen['epoch'] == checkpoint['epoch'] == metrics['best_epoch']
        assert chosen['validation'] == checkpoint['validation']
        for split, metric_name in [('val', 'validation'), ('test', 'test')]:
            rows = predictions[metric_name]
            assert [r['patient_id'] for r in rows] == groups[split]
            assert classification_metrics(rows, metrics['threshold']) == metrics[metric_name]
        assert classification_metrics(predictions['test'], .5) == metrics['test_fixed_05']
        assert choose_threshold(predictions['validation']) == metrics['threshold']
        assert metrics == by_pair[(model, seed)]
        expected_device = protocol['device'].split(':')[0]
        assert metrics['runtime']['tensor_device'].split(':')[0] == expected_device
        if expected_device == 'cuda':
            assert metrics['runtime']['gpu_name'] is not None
        print(f'OK {model} seed={seed}: shared split, validation-only selection, cached predictions')
    for model in protocol['models']:
        rows = [by_pair[(model, seed)] for seed in protocol['seeds']]
        assert summary['summary'][model]['n_training_seeds'] == len(rows)
        for key in rows[0]['test']:
            values = np.array([r['test'][key] for r in rows])
            aggregate = summary['summary'][model]['test'][key]
            assert np.isclose(aggregate['mean'], values.mean())
            if len(values) > 1:
                assert np.isclose(aggregate['std'], values.std(ddof=1))
            else:
                assert aggregate['std'] is None
    for name, fields in summary['paired_differences'].items():
        a, b = name.split(' minus ')
        for metric, aggregate in fields.items():
            differences = [by_pair[(a, seed)]['test'][metric] - by_pair[(b, seed)]['test'][metric]
                           for seed in protocol['seeds']]
            assert aggregate['n_pairs'] == len(differences)
            assert np.isclose(aggregate['mean'], np.mean(differences))
            if len(differences) > 1:
                assert np.isclose(aggregate['std'], np.std(differences, ddof=1))
    if protocol['device'].startswith('cuda'):
        print('Physical GPUs recorded:', sorted({r['runtime']['cuda_visible_devices'] for r in by_pair.values()}, key=str))
    print(f'Validated {len(expected)} runs and all mean/sample-standard-deviation summaries')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output')
    audit(p.parse_args().output)
