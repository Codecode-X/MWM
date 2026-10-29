"""CUDA check: identical latent speed, different clinical directions."""
import unittest
import torch
from torch import nn
from paper_odeae.diagnostics.mmse_rate_mechanism import mechanism


class MechanismChecks(unittest.TestCase):
    def test_parallel_opposite_and_tangent_directions(self):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        class Model(nn.Module):
            def __init__(self):
                super().__init__();self.decoder=nn.Sequential(nn.Linear(6,6),nn.Tanh(),nn.Linear(6,1));self.forward_ode=nn.Linear(6,6)
        model=Model().double().cuda();z=torch.zeros(3,6,device='cuda',dtype=torch.float64)
        with torch.no_grad():
            model.decoder[0].weight.copy_(torch.eye(6,device='cuda',dtype=torch.float64));model.decoder[0].bias.zero_()
            model.decoder[2].weight.zero_();model.decoder[2].weight[0,0]=1;model.decoder[2].bias.zero_();model.forward_ode.weight.zero_()
            for direction,expected_cosine,rate in [([2,0,0,0,0,0],1,60),([-2,0,0,0,0,0],-1,-60),([0,2,0,0,0,0],0,0)]:
                model.forward_ode.bias.copy_(torch.tensor(direction,device='cuda',dtype=torch.float64))
                r=mechanism(model,z,0,30.)
                torch.testing.assert_close(r['speed'],torch.full((3,),2.,device='cuda',dtype=torch.float64))
                torch.testing.assert_close(r['gradient_norm'],torch.full((3,),30.,device='cuda',dtype=torch.float64))
                torch.testing.assert_close(r['cosine'],torch.full((3,),float(expected_cosine),device='cuda',dtype=torch.float64))
                torch.testing.assert_close(r['rate'],torch.full((3,),float(rate),device='cuda',dtype=torch.float64))
                torch.testing.assert_close(r['speed']*r['gradient_norm']*r['cosine'],r['rate'])


if __name__=='__main__':unittest.main()
