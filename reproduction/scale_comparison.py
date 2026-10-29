"""Run base and parameter-matched controls in one shared CUDA worker pool."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import numpy as np
import torch
from src.comparison import parser, configuration, initialize_worker, run_pair, summarize, build_model
from src.preprocessing import prepare_data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='results/capacity_cuda')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--resume', action='store_true')
    options = p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise RuntimeError('Four CUDA GPUs are required; CPU fallback disabled')
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        raise ValueError('Unset CUDA_VISIBLE_DEVICES; workers assign physical GPUs')
    if options.jobs < 4 or options.epochs < 1 or len(set(options.seeds)) != len(options.seeds):
        raise ValueError('Require >=4 jobs, positive epochs and distinct seeds')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    tasks, groups, variants = [], {}, {}
    root = Path(options.output)
    root.mkdir(parents=True, exist_ok=True)
    for objective in ('native', 'bce'):
        for capacity in ('base', 'matched'):
            args = parser().parse_args([])
            args.output = str(Path(options.output) / f'{objective}_{capacity}')
            args.objective, args.capacity = objective, capacity
            args.models = list(args.models) if capacity == 'base' else ['latent_ode', 'ode_rnn']
            args.epochs, args.seeds, args.jobs, args.resume = options.epochs, options.seeds, options.jobs, options.resume
            out = Path(args.output)
            out.mkdir(parents=True, exist_ok=True)
            _, metadata = prepare_data(args.data, args.split_seed)
            metadata['data_sha256'] = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
            for filename, value in [('protocol.json', configuration(args)), ('data.json', metadata)]:
                target = out / filename
                if target.exists() and json.loads(target.read_text()) != value:
                    raise ValueError(f'Conflicting protocol/data: {target}')
                target.write_text(json.dumps(value, indent=2))
            (out / 'environment.json').write_text(json.dumps({'python': platform.python_version(),
                'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cuda',
                'requested_gpus': [0, 1, 2, 3], 'threads': 1, 'deterministic_algorithms': True}, indent=2))
            groups[args.output] = []
            if objective == 'native':
                for name in args.models:
                    model = build_model(name, args)
                    variants[f'{name}_{capacity}'] = {
                        'parameters': sum(p.numel() for p in model.parameters()),
                        'modules': {key: sum(p.numel() for p in module.parameters())
                                    for key, module in model.named_children()},
                        'latent_or_state': 140 if capacity == 'matched' and name == 'ode_rnn' else args.latent,
                        'recognition_hidden': 143 if capacity == 'matched' and name == 'latent_ode' else args.hidden,
                        'width': args.width, 'layers': args.layers}
                    if name == 'legacy_transformer':
                        variants[f'{name}_{capacity}'] = {k: v for k, v in variants[f'{name}_{capacity}'].items()
                                                          if k in ('parameters', 'modules')}
                    del model
            for seed in args.seeds:
                for name in args.models:
                    tasks.append((name, seed, args))
    (root / 'capacity_manifest.json').write_text(json.dumps({
        'baseline_commit': 'e3752ce', 'baseline_tag': 'baseline-before-capacity-scaling-20261020',
        'epochs': options.epochs, 'training_seeds': options.seeds,
        'physical_gpus': [0, 1, 2, 3], 'jobs': options.jobs, 'device': 'cuda',
        'variants': variants}, indent=2))
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    for i in range(options.jobs):
        queue.put(i % 4)
    with ProcessPoolExecutor(max_workers=options.jobs, mp_context=context,
                             initializer=initialize_worker, initargs=(queue,)) as pool:
        futures = {pool.submit(run_pair, *task): task[2].output for task in tasks}
        for future in as_completed(futures):
            root = futures[future]
            groups[root].append(future.result())
            groups[root].sort(key=lambda r: (r['seed'], r['model']))
            summarize(groups[root], root)
    # Audits are invoked by the bash wrapper, which propagates any failure.


if __name__ == '__main__':
    main()
