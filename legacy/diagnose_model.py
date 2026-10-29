"""Read-only diagnostics; does not modify checkpoints or official metrics."""
import json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from model import LatentODEModel
from preprocessing import create_dataloaders

torch.set_num_threads(4)
train, val, test = create_dataloaders(32)
model = LatentODEModel().cuda().eval()
model.load_state_dict(torch.load('models/best_model.pt', map_location='cuda', weights_only=False)['model_state_dict'])
batches = {name: next(iter(loader)) for name, loader in [('val',val),('test',test)]}
def predict(b):
    with torch.no_grad():
        o = model(b['vitals'].cuda(), b['times'].cuda(), b['masks'].cuda())
    return (o['alpha']/(o['alpha']+o['beta'])).cpu().numpy().ravel()
result = {}
for name,b in batches.items():
    scores=[]
    for seed in range(20):
        torch.manual_seed(seed)
        p=predict(b)
        scores.append(float(roc_auc_score(b['labels'].numpy().ravel(),p)))
    result[name+'_raw_auroc_20_seeds']={'values':scores,'mean':float(np.mean(scores)),'std':float(np.std(scores)),'min':min(scores),'max':max(scores)}
model.reparameterize=lambda mu,logvar: mu
for name,b in batches.items():
    p=predict(b)
    result[name+'_mean_latent_raw_auroc']=float(roc_auc_score(b['labels'].numpy().ravel(),p))
b=batches['test']
p_batch=predict(b)
p_single=np.array([predict({k:v[i:i+1] for k,v in b.items()})[0] for i in range(len(b['labels']))])
result['mean_latent_single_vs_batch_max_prob_diff']=float(np.max(np.abs(p_batch-p_single)))
result['test_global_end_hours']=float(b['times'].max())
result['test_first_patient_end_hours']=float(b['times'][0].max())
result['reconstruction_valid_patients_per_training_batch']=[int(x['masks'][:,-1].sum()) for x in train]
Path('reproduction/model-diagnostics.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
