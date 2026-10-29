"""CUDA checks of score-unit conversion and exact decoder directional rates."""
import unittest
import torch
from torch import nn
from paper_odeae.diagnostics.clinical_velocity import clinical_values_and_rates


class ClinicalVelocityChecks(unittest.TestCase):
    def test_chain_rule_all_heads_and_units(self):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(1);torch.manual_seed(3)
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder=nn.Sequential(nn.Linear(6,11),nn.Tanh(),nn.Linear(11,4))
                self.forward_ode=nn.Sequential(nn.Linear(6,9),nn.Tanh(),nn.Linear(9,6))
        model=Model().double().cuda();z=torch.randn(5,6,device='cuda',dtype=torch.float64,requires_grad=True)
        indices=torch.tensor([0,2,3],device='cuda');scales=torch.tensor([30.,18.,8.],device='cuda',dtype=torch.float64);offsets=torch.tensor([0.,-3.,2.],device='cuda',dtype=torch.float64)
        values,rates,field=clinical_values_and_rates(model,z,indices,scales,offsets)
        raw=model.decoder(z)[:,indices]*scales+offsets
        torch.testing.assert_close(values,raw,rtol=0,atol=0)
        for i in range(3):
            grad=torch.autograd.grad(raw[:,i].sum(),z,retain_graph=True)[0]
            torch.testing.assert_close((grad*field).sum(1),rates[:,i],rtol=1e-10,atol=1e-10)
        eps=1e-5
        plus=model.decoder(z+eps*field)[:,indices]*scales+offsets
        minus=model.decoder(z-eps*field)[:,indices]*scales+offsets
        torch.testing.assert_close((plus-minus)/(2*eps),rates,rtol=1e-6,atol=1e-8)


if __name__=='__main__':unittest.main()
