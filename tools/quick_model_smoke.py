"""Forward/backward smoke check for both paper models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.model import MODELS, objective

torch.set_num_threads(1)
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is required; CPU fallback is disabled')
for name, factory in MODELS.items():
    model = factory().cuda()
    values = torch.randn(2, 5, 6, device='cuda')
    times = torch.tensor([0., .1, .3, .7, 1.], device='cuda')
    mask = torch.ones_like(values)
    output = model(values, times, mask)
    loss = objective(output, values, mask)['loss']
    loss.backward()
    print(name, tuple(output['prediction'].shape), float(loss.detach()))
