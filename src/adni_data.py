"""Auditable, patient-disjoint ADNI numerical features on actual-date timelines.

No diagnosis, outcomes, treatment text, future imputation or undated events enter
the forecasting inputs. Quarterly bins end AFTER every contributing acquisition.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

DEFAULT_ROOT = '/data/SAMSUNG_backup/ADNI_data/ADNI_multimodal_timeline'
CACHE = 'data/adni_cache'

# Source-specific fields prevent accidental pooling of incompatible assays.
SPEC = {
    'mmse': [('MMSCORE', 0, 30)],
    'cdr': [(k, 0, 18 if k == 'CDRSB' else 3) for k in
            ['CDRSB', 'CDGLOBAL', 'CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE']],
    'adas': [(k, 0, 85 if k == 'TOTAL13' else 70 if k == 'TOTSCORE' else 30)
             for k in ['TOTSCORE', 'TOTAL13'] + [f'Q{i}SCORE' for i in range(1, 14)]],
    'faq': [(k, 0, 30 if k == 'FAQTOTAL' else 3) for k in
            ['FAQTOTAL', 'FAQFINAN', 'FAQFORM', 'FAQSHOP', 'FAQGAME', 'FAQBEVG',
             'FAQMEAL', 'FAQEVENT', 'FAQTV', 'FAQREM', 'FAQTRAVL']],
    'neurobat': [(k, 0, 100) for k in ['LIMMTOTAL', 'LDELTOTAL', 'AVTOT1', 'AVTOT2',
                                      'AVTOT3', 'AVTOT4', 'AVTOT5', 'AVDEL30MIN']],
    'plasma': [(k, 0, 1e6) for k in ['pT217_F', 'AB42_F', 'AB40_F', 'AB42_AB40_F',
                                    'NfL_Q', 'GFAP_Q']],
    'csf': [(k, 0, 1e5) for k in ['ABETA40', 'ABETA42', 'TAU', 'PTAU']],
    'amyloid_pet': [('CENTILOIDS', -100, 400)],
    'tau_pet': [('META_TEMPORAL_SUVR', 0, 10), ('CTX_ENTORHINAL_SUVR', 0, 10)],
    'mri': [('ST29SV', 0, 20000), ('ST88SV', 0, 20000)],
}


def numeric(value, lower, upper):
    # Censored assay strings (<, >) are deliberately missing, never exact values.
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) and lower <= x <= upper else None


def read_patient(args):
    root, subject = args
    p = root / 'patients' / subject
    timeline = json.loads((p / 'timeline.json').read_text())
    rows, audit = defaultdict(lambda: defaultdict(list)), Counter()
    for item in timeline['timepoints']:
        record = json.loads((p / item['record_file']).read_text())
        exams = record['multimodal_examinations']
        groups = {**exams['neuropsychological'], **exams['biomarkers'],
                  **{k: v['measurements'] + v['metadata'] for k, v in exams['imaging'].items()}}
        # The source timeline stores some quantitative tables as metadata.
        # Locate the explicit assay source rather than assuming its container.
        groups['csf'] = list(groups['csf']) + [e for entries in groups.values()
            for e in entries if 'ROCHE_ELECSYS' in e['source']['file']
            and e not in groups['csf']]
        for group, fields in SPEC.items():
            for event in groups.get(group, []):
                source = event['source']['file']
                v = event['values']
                if group == 'csf' and 'ROCHE_ELECSYS' not in source:
                    continue
                if group in ('tau_pet', 'amyloid_pet'):
                    # Only full visual QC pass; tau uses non-PVC throughout.
                    if 'TAUPVC' in source or numeric(v.get('qc_flag'), 2, 2) is None:
                        audit['pet_excluded_qc_or_pvc'] += 1
                        continue
                if group == 'mri':
                    if 'UCSFFSX7' not in source:
                        continue
                    # Explicit failures are excluded. Missing/partial QC is
                    # retained and counted separately, not represented as a pass.
                    qc = str(v.get('HIPPOQC', v.get('OVERALLQC'))).lower()
                    if any('fail' in str(v.get(k)).lower() for k in ('HIPPOQC','OVERALLQC')):
                        audit['mri_failed_qc'] += 1
                        continue
                    audit['mri_qc_' + qc] += 1
                event_date = event.get('event_date')
                if (not event_date or event_date != item['date']
                        or event.get('time_resolution') == 'provisional'):
                    audit['undated_or_misaligned'] += 1
                    continue
                audit['time_resolution_' + str(event.get('time_resolution'))] += 1
                for key, low, high in fields:
                    x = numeric(v.get(key), low, high)
                    if x is not None:
                        rows[event_date][f'{group}.{key}'].append(x)
    return subject, {d: {k: float(np.mean(v)) for k, v in f.items()}
                     for d, f in rows.items() if f}, audit


def build(root=DEFAULT_ROOT, cache=CACHE):
    root, cache = Path(root), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    valid = json.loads((root / 'reports/validation_summary.json').read_text())
    if not valid['valid']:
        raise ValueError('source timeline validation did not pass')
    split_file = root / 'indexes/patient_splits.csv'
    split = pd.read_csv(split_file)
    if split.subject_id.duplicated().any():
        raise ValueError('duplicate patient assignment')
    patients, audit = [], Counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (subject, rows, counts) in enumerate(pool.map(read_patient,
                [(root, s) for s in split.subject_id])):
            audit.update(counts)
            if rows:
                patients.append({'patient_id': subject, 'rows': rows})
            if (i + 1) % 300 == 0:
                print(f'parsed {i+1}/{len(split)} patients', flush=True)
    with (cache / 'raw.pkl').open('wb') as f:
        pickle.dump(patients, f)
    features = [f'{g}.{k}' for g, fields in SPEC.items() for k, _, _ in fields]
    assignments = dict(zip(split.subject_id, split.split))
    sequences, excluded = [], Counter()
    for p in patients:
        days = sorted(p['rows'])
        start = date.fromisoformat(days[0]).toordinal()
        # Fixed 3-month duration, relative to the first usable examination.
        quarters = defaultdict(list)
        for d in days:
            q = int((date.fromisoformat(d).toordinal() - start) // 91.3125)
            quarters[q].append(d)
        if len(quarters) < 3:
            excluded[assignments[p['patient_id']]] += 1
            continue
        x = np.full((len(quarters), len(features)), np.nan, np.float32)
        times, bin_dates = [], []
        for i, (q, ds) in enumerate(sorted(quarters.items())):
            times.append((q + 1) / 4.)
            bin_dates.append(max(ds))
            for j, name in enumerate(features):
                vals = [p['rows'][d][name] for d in ds if name in p['rows'][d]]
                if vals:
                    x[i, j] = np.mean(vals)
        sequences.append({'patient_id': p['patient_id'], 'split': assignments[p['patient_id']],
                          'times': np.array(times, np.float32), 'values': x,
                          'start_ordinal': start, 'latest_dates': bin_dates})
    # Feature eligibility, normalization and all dimensionality reduction use
    # TRAIN patients alone. No validation/test data inform these parameters.
    train = np.concatenate([s['values'] for s in sequences if s['split'] == 'train'])
    counts = np.isfinite(train).sum(0)
    keep = counts >= 100
    selected = [f for f, k in zip(features, keep) if k]
    train = train[:, keep]
    mean, scale = np.nanmean(train, 0), np.nanstd(train, 0)
    scale = np.maximum(scale, 1e-5)
    for s in sequences:
        s['values'] = s['values'][:, keep]
    metadata = {'root': str(root.resolve()), 'features': selected,
                'excluded_features_train_count_below_100': [f for f,k in zip(features,keep) if not k],
                'mean': mean.tolist(), 'scale': scale.tolist(),
                'normalization': 'train-only mean/std, no clipping or imputation',
                'binning': '91.3125-day bins, timestamp at right edge, mean within bin',
                'split_sha256': hashlib.sha256(split_file.read_bytes()).hexdigest(),
                'patient_counts': dict(Counter(s['split'] for s in sequences)),
                'excluded_fewer_than_3_quarters': dict(excluded),
                'training_feature_counts': dict(zip(selected, counts[keep].tolist())),
                'audit': dict(audit)}
    with (cache / 'prepared.pkl').open('wb') as f:
        pickle.dump({'sequences': sequences, 'metadata': metadata}, f)
    (cache / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=DEFAULT_ROOT)
    parser.add_argument('--cache', default=CACHE)
    args = parser.parse_args()
    build(args.root, args.cache)
