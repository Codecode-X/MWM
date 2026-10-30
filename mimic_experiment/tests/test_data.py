"""Small data and CUDA causality checks; no model experiment on CPU."""
import unittest
import numpy as np
import pandas as pd
import torch
from mimic_experiment.prepare import schema,split,chronology_support
from mimic_experiment.data import ExactBatchSampler,ICUContexts,collate
from mimic_experiment.train_ddp import carry_forward
from paper_odeae.models import paper_model_from_config
from paper_odeae.train import device_batch


class ICUChecks(unittest.TestCase):
    def test_complete_living_hospital_window(self):
        frame=pd.DataFrame({'intime':['2020-01-01 01:00']*5,
                            'admittime':['2020-01-01 00:00']*4+['2020-01-01 02:00'],
                            'dischtime':['2020-01-04 00:00']*3+['2020-01-02 00:00']*2,
                            'deathtime':[None,'2020-01-03 01:00','2020-01-02 00:00',None,None]})
        self.assertEqual(chronology_support(frame).tolist(),[True,True,False,False,False])

    def sequence(self,subject=123):
        fields=schema();values=np.full((48,len(fields)),.5,np.float32);mask=np.ones_like(values)
        for j,s in enumerate(fields):
            if s['kind']=='categorical':values[:,j]=0
        return {'subject_id':str(subject),'admission_age':73.,'ages':np.arange(1,49,dtype=np.float32),'values':values,'mask':mask,'dates':[str(i) for i in range(48)]}

    def test_no_duplicate_or_dropped_ddp_patients(self):
        for length in [4,127,128,129,130,131,132,257,1000]:
            samplers=[ExactBatchSampler(length,128,rank,4) for rank in range(4)]
            local=[list(s) for s in samplers];self.assertTrue(all(len(s)==len(local[0]) for s in local))
            flat=[i for worker in local for batch in worker for i in batch]
            self.assertEqual(sorted(flat),list(range(length)))
            self.assertTrue(all(len(batch)>0 for worker in local for batch in worker))

    def test_prefixes_and_separate_age(self):
        seq=self.sequence();ds=ICUContexts([seq],horizon=24);item=ds[0];self.assertEqual(item[1],24)
        batch=collate([item]);self.assertAlmostEqual(float(batch['initialization_age'][0]),73+24/8766,places=5)
        self.assertEqual(float(batch['ages'][0,23]),24.)
        train=ICUContexts([seq],training=True,seed=0);train.set_epoch(9)
        self.assertEqual(train[0][1],train[0][1]);self.assertIn(train[0][1],[6,12,18,24])
        self.assertEqual(split(123),split(123))

    def test_cuda_age_and_no_future_update(self):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(1);torch.manual_seed(0);fields=schema();seq=self.sequence();batch=device_batch(collate([(seq,12)]))
        cfg={'model':'deterministic','latent':28,'mlp_width':159,'ode_width':188,'rtol':1e-4,'atol':1e-6,'encoder_direction':'forward','encoder_initialization':'age',
             'shared_vector_field':True,'collect_prior_predictions':True,'collect_jacobian_states':True,'prior_weight':.01,'jacobian_weight':.01}
        model=paper_model_from_config(fields,cfg).cuda().eval();seen=[]
        hook=model.init.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
        with torch.no_grad():original=model(batch)
        hook.remove();torch.testing.assert_close(seen[0],batch['initialization_age'][:,None]/100,rtol=0,atol=0)
        changed={k:v.clone() if torch.is_tensor(v) else v for k,v in batch.items()}
        for j,s in enumerate(fields):
            if s['kind']=='numerical':changed['values'][:,12:,j]+=5
            else:changed['values'][:,12:,j]=(changed['values'][:,12:,j]+1)%s['classes']
        changed['mask'][:,12:]=0
        with torch.no_grad():other=model(changed)
        torch.testing.assert_close(original['prediction'],other['prediction'],rtol=0,atol=0)
        baseline=carry_forward(batch,fields,[0]*len(fields),[.5]*len(fields))
        torch.testing.assert_close(baseline,carry_forward(changed,fields,[0]*len(fields),[.5]*len(fields)),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
