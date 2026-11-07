"""Evaluate a saved paper checkpoint on its original patient split."""
import argparse
from functools import partial
import json
import torch
from torch.utils.data import DataLoader
from .model import MODELS
from .preprocessing import prepare_data, collate_fn
from .train import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint')
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('Project policy requires CUDA evaluation; CPU fallback is disabled')
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = argparse.Namespace(**checkpoint['config'])
    config.device = args.device
    torch.set_num_threads(config.threads)
    splits, metadata = prepare_data(config.data, config.seed)
    if metadata != checkpoint['metadata']:
        raise ValueError('dataset or patient split differs from checkpoint')
    model = MODELS[config.model](latent=config.latent, hidden=config.hidden,
                                width=config.width, layers=config.layers).to(args.device)
    model.load_state_dict(checkpoint['state_dict'])
    loader = DataLoader(splits[args.split], batch_size=config.batch_size, collate_fn=partial(collate_fn, task=config.task))
    print(json.dumps(evaluate(model, loader, config), indent=2))


if __name__ == '__main__':
    main()
