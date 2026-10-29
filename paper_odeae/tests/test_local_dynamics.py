"""CUDA tests of patient-excluded neighbor selection and local metrics."""
import unittest
import torch
from paper_odeae.diagnostics.local_dynamics import distinct_patient_knn,pair_metrics,region_assignment


class LocalDynamicsChecks(unittest.TestCase):
    def test_distinct_neighbors_and_known_sensitivity(self):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(1)
        reference=torch.zeros(6,6,device='cuda');reference[:,0]=torch.tensor([0.,.2,1.1,1.9,2.,10.],device='cuda')
        query=reference[torch.tensor([0,4],device='cuda')];pids=torch.tensor([0,0,1,1,2,3],device='cuda')
        indices,distance=distinct_patient_knn(query,reference,torch.tensor([0,2],device='cuda'),pids,2,4)
        torch.testing.assert_close(indices,torch.tensor([[2,4],[3,1]],device='cuda'),rtol=0,atol=0)
        # With v(z)=2z, every nonzero pair difference has sensitivity exactly 2.
        z=reference+1;neighbors=torch.tensor([[2,4],[2,4],[1,4],[1,4],[1,3],[2,4]],device='cuda')
        C,D,_=pair_metrics(z,2*z,neighbors)
        torch.testing.assert_close(D,torch.full_like(D,2),rtol=1e-6,atol=1e-6)
        constant=torch.ones_like(z);C,D,_=pair_metrics(z,constant,neighbors)
        torch.testing.assert_close(C,torch.ones_like(C),rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(D,torch.zeros_like(D),rtol=0,atol=0)

    def test_fixed_region_rules(self):
        labels=torch.tensor([[0]*20,[0]*10+[1]*10,[1]*20,[1]*10+[2]*10,[2]*20,[0]*10+[2]*10,[-1]*20],device='cuda')
        regions,_,_=region_assignment(labels)
        torch.testing.assert_close(regions,torch.arange(7,device='cuda'),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
