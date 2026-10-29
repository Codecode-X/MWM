"""Check experiment artifacts, validation selection and independent evaluation."""
import argparse
import json
import math
from pathlib import Path
import torch


def check(directory):
    def read(name):
        return json.loads((directory / name).read_text())
    config, metrics, history = read('config.json'), read('metrics.json'), read('history.json')
    assert len(history) == config['epochs'], f'{directory}: incomplete training history'
    assert [row['epoch'] for row in history] == list(range(1, config['epochs'] + 1))
    chosen = min(history, key=lambda row: row['validation']['mse'])
    assert metrics['best_epoch'] == chosen['epoch'], f'{directory}: validation selection mismatch'
    assert chosen['validation'] == metrics['validation'], f'{directory}: validation replay differs'
    assert metrics['test'] == read('test_reloaded.json'), f'{directory}: test replay differs'
    checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == chosen['epoch']
    assert checkpoint['validation'] == metrics['validation']
    assert checkpoint['config'] == config
    metadata = read('data.json')
    assert checkpoint['metadata'] == metadata
    ids = list(metadata['patient_ids'].values())
    assert sum(len(x) for x in ids) == len(set(sum(ids, []))), 'patient split overlap'
    for split in ('baseline_validation', 'validation', 'test'):
        assert all(math.isfinite(value) for value in metrics[split].values())
    print(f'OK {directory}: {len(history)} epochs, best={chosen["epoch"]}, replay identical')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+')
    args = parser.parse_args()
    count = 0
    for root in args.roots:
        for path in sorted(Path(root).glob('*/metrics.json')):
            check(path.parent)
            count += 1
    if not count:
        raise SystemExit('No completed experiments found')
    print(f'Validated {count} experiments')


if __name__ == '__main__':
    main()
