#!/usr/bin/env python3
"""Audit prepared data and extraction completeness without model execution."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np


def main():
    cache=Path('data/adni_cache')
    output=Path('results/adni_cuda')
    data=pickle.load((cache/'prepared.pkl').open('rb'))
    sequences=data['sequences']
    splits={k:{s['patient_id'] for s in sequences if s['split']==k} for k in ['train','validation','test']}
    assert not splits['train'] & splits['validation']
    assert not splits['train'] & splits['test']
    assert not splits['validation'] & splits['test']
    for s in sequences:
        assert len(s['times'])>=3
        assert np.all(np.diff(s['times'])>0)
        assert not np.isinf(s['values']).any()
    manifest=json.loads((cache/'mri_manifest.json').read_text())
    eligible=set.union(*splits.values())
    selected={s['subject_id'] for s in manifest['scans']} & eligible
    embeddings={p.stem for p in (cache/'embeddings').glob('*.npz')}
    failures=[]
    for shard in range(4):
        failures.extend(json.loads((cache/f'encoder_failures_{shard}.json').read_text()))
        hardware=json.loads((cache/f'encoder_hardware_{shard}.json').read_text())
        assert hardware['physical_gpu']==str(shard)
        assert hardware['input_device'].startswith('cuda')
        assert hardware['output_device'].startswith('cuda')
        assert hardware['weights_sha256']==json.loads((output/'research/encoder_provenance.json').read_text())['weight_sha256']
        (output/f'encoder_hardware_{shard}.json').write_text(json.dumps(hardware,indent=2))
    failed={f['subject_id'] for f in failures}
    missing=selected-embeddings-failed
    if missing:
        raise ValueError(f'extraction incomplete: {len(missing)} eligible selected patients unaccounted for')
    for p in (cache/'embeddings').glob('*.npz'):
        with np.load(p) as a:
            assert a['embedding'].shape==(512,)
            assert np.isfinite(a['embedding']).all()
    summary={'patient_counts':{k:len(v) for k,v in splits.items()},
             'patient_disjoint':True,'prepared_sha256':hashlib.sha256((cache/'prepared.pkl').read_bytes()).hexdigest(),
             'feature_count':len(data['metadata']['features']),
             'selected_eligible_T1':len(selected),'successfully_encoded_eligible':len(selected & embeddings),
             'failed_eligible':len((selected-embeddings)&failed),
             'no_selected_T1_eligible':len(eligible-selected),
             'failure_reasons':dict(Counter(f['error'] for f in failures)),
             'normalization_fit_split':'train','PCA_fit_split':'train',
             'feature_extraction':'one earliest recognized T1 per eligible patient, frozen MedicalNet on 4 CUDA GPUs'}
    (output/'data_audit.json').write_text(json.dumps(summary,indent=2))
    (output/'data_preparation.json').write_text(json.dumps(data['metadata'],indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
