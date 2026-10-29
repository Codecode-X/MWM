"""Run the upstream pipeline with fixed random seeds from the repository root."""
import os
import random
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch

stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
if stage not in ('preprocessing', 'train', 'evaluate', 'all'):
    raise SystemExit('Choose preprocessing, train, evaluate, or all')
for name in ('preprocessing', 'train', 'evaluate'):
    if stage not in (name, 'all'):
        continue
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f'Running {name}; seed=42; torch={torch.__version__}', flush=True)
    runpy.run_path(str(ROOT / 'src' / f'{name}.py'), run_name='__main__')
