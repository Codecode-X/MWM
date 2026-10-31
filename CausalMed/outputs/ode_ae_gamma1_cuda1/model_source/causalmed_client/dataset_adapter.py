"""Public-only loading. This module never imports or opens the generator / oracle.

Use a filesystem mount containing ONLY public/ for actual model training/inference.
Path separation is not an OS security sandbox for arbitrary caller code.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ['Y_obs', 'X1_obs', 'X2_obs']
COLUMNS = {
    'patients': {'patient_id', 'c'},
    'visits': {'patient_id', 'visit_index', 'time', *FEATURES},
    'treatments': {'patient_id', 't_start', 't_end', 'treatment'},
}


class PublicDataset:
    def __init__(self, public_dir, split='train'):
        self.root = Path(public_dir).resolve()
        if 'oracle_private' in self.root.parts or split not in ('train', 'val', 'test'):
            raise ValueError('a public directory and valid split are required')
        self.split = split
        visit_file = 'test_histories' if split == 'test' else f'{split}_visits'
        treatment_file = 'test_history_treatments' if split == 'test' else f'{split}_treatments'
        self.patients = self._read(f'{split}_patients', 'patients')
        visits = self._read(visit_file, 'visits')
        treatments = self._read(treatment_file, 'treatments')
        self.visits = {pid: g.sort_values('time') for pid, g in visits.groupby('patient_id', sort=False)}
        self.treatments = {pid: g.sort_values('t_start') for pid, g in treatments.groupby('patient_id', sort=False)}
        self.ids = self.patients.patient_id.tolist()
        self.c = dict(zip(self.patients.patient_id, self.patients.c))
        self.policy_specs = json.loads((self.root / 'policy_specs.json').read_text())

    def _read(self, filename, kind):
        path = self.root / f'{filename}.parquet'
        if path.resolve().parent != self.root:
            raise ValueError('public file may not resolve outside the public directory')
        df = pd.read_parquet(path)
        if set(df.columns) != COLUMNS[kind]:
            raise ValueError(f'public schema violation in {filename}: {list(df.columns)}')
        return df

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        pid = self.ids[index]
        v, tr = self.visits[pid], self.treatments.get(pid)
        return {'patient_id': pid, 'c': float(self.c[pid]), 'times': v.time.tolist(),
                'observations': v[FEATURES].values.tolist(),
                'observation_mask': np.ones((len(v), 3), dtype=int).tolist(),
                'treatment_intervals': [] if tr is None else [
                    {'start': float(r.t_start), 'end': float(r.t_end), 'treatment': int(r.treatment)}
                    for r in tr.itertuples()]}

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def queries(self):
        if self.split != 'test':
            raise ValueError('counterfactual queries exist only for test')
        path = self.root / 'test_queries.parquet'
        if path.resolve().parent != self.root:
            raise ValueError('query path escapes public directory')
        df = pd.read_parquet(path)
        if set(df.columns) != {'patient_id', 'fork_time', 'policy_id', 'prediction_times',
                               'trajectory_times', 'future_visit_times'}:
            raise ValueError('query schema violation')
        return df

    def shared_matched_sequences(self, normalize=True):
        """Same sequence keys as paper_odeae.data; treatment/c remain explicit extras.

        ages is elapsed years, NOT chronological patient age. Feature schema has 3
        continuous outputs; existing fixed ADNI heads are not directly applicable.
        """
        train = self._read('train_visits', 'visits')[FEATURES].values
        offset, scale = train.mean(0), np.maximum(train.std(0), 1e-6)
        for s in self:
            values = np.asarray(s['observations'], dtype=np.float32)
            yield {'subject_id': s['patient_id'], 'split': self.split,
                   'ages': np.array(s['times'], dtype=np.float32),
                   'values': ((values - offset) / scale).astype(np.float32) if normalize else values,
                   'mask': np.asarray(s['observation_mask'], dtype=np.float32),
                   'c': s['c'], 'treatment_intervals': s['treatment_intervals']}
