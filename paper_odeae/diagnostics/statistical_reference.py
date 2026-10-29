"""Aggregate cached observations and saved metrics; no model execution."""
import json
from pathlib import Path
import pickle

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    payload = pickle.loads((ROOT/'paper_odeae/cache/local_available.pkl').read_bytes())
    schema, sequences = payload['schema'], payload['sequences']
    train = [s for s in sequences if s['split']=='train']
    values = np.concatenate([s['values'] for s in train])
    mask = np.concatenate([s['mask'] for s in train])
    weights = np.concatenate([np.full(len(s['ages']), len(s['ages'])) for s in train])
    means, probabilities, variances = [], [], []
    for j, field in enumerate(schema):
        observed = mask[:,j]>0
        x, w = values[observed,j], weights[observed]
        means.append(np.average(x, weights=w)); variances.append(np.var(x))
        probabilities.append(np.bincount(x.astype(int), weights=w, minlength=field['classes'])/w.sum()
                             if field['kind']=='categorical' else None)
    reference, counts = {}, {}
    for split in ['train', 'validation', 'test']:
        selected = [s for s in sequences if s['split']==split]
        lengths = np.array([len(s['ages']) for s in selected])
        counts[split] = {'patients': len(selected), 'quarter_visits': int(lengths.sum()),
                         'prefixes': int(lengths.sum()), 'prefix_weighted_target_visits': int((lengths**2).sum())}
        if split=='train':
            continue
        totals = np.zeros((2,len(schema)))
        for sequence in selected:
            for j, field in enumerate(schema):
                x = sequence['values'][sequence['mask'][:,j]>0,j]
                loss = (-np.log(np.maximum(probabilities[j][x.astype(int)],1e-12))
                        if field['kind']=='categorical' else (x-means[j])**2)
                totals[0,j] += loss.sum()*len(sequence['ages'])
                totals[1,j] += len(x)*len(sequence['ages'])
        error = totals[0]/np.maximum(totals[1],1)
        reference[split] = {'mse': sum(float(e) for e,f in zip(error,schema) if f['kind']=='numerical'),
                            'cce': sum(float(e) for e,f in zip(error,schema) if f['kind']=='categorical'),
                            'overall': float(error.sum()),
                            'per_field': {f['name']:float(e) for f,e in zip(schema,error)}}
    transformations = {}
    runs = ROOT/'paper_odeae/results/local_available'
    for path in sorted(runs.glob('*/seed_*/metrics.json')):
        results = json.loads(path.read_text()); fields = results['test']['overall']['per_field']
        transformations[str(path.parent.relative_to(runs))] = {
            'reported_minmax_mse': results['test']['overall']['sum_feature_mse'],
            'same_prediction_mse_in_train_sd_units': sum(fields[f['name']]['loss']/variances[j]
                for j,f in enumerate(schema) if f['kind']=='numerical')}
    result = {'dataset_counts': counts, 'population_constant_reference': reference,
              'fixed_prediction_unit_conversion': transformations,
              'note': 'Statistical reference from training-set prefix-weighted means/frequencies; '
                      'unit conversion does not retrain and does not identify the original normalization.'}
    output = ROOT/'paper_odeae/results/discrepancy_audit'
    output.mkdir(parents=True,exist_ok=True)
    (output/'statistical_reference.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
