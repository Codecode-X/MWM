"""Patient-disjoint raw CSV splits and feature masks on exact union timestamps."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

FEATURES = ['heart_rate', 'sbp', 'dbp', 'spo2', 'temperature', 'respiratory_rate']


class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


def prepare_data(path='data/raw/synthetic_icu_data.csv', seed=42):
    df = pd.read_csv(path)
    if not np.isfinite(df.time_hours).all() or (df.time_hours < 0).any() or (df.time_hours > 72).any():
        raise ValueError('expected finite timestamps in the synthetic ICU window [0,72] hours')
    grouped = list(df.groupby('patient_id', sort=True))
    ids = np.arange(len(grouped))
    labels = [int(group.adverse_event.iloc[0]) for _, group in grouped]
    train, rest = train_test_split(ids, train_size=.7, random_state=seed, stratify=labels)
    val, test = train_test_split(rest, test_size=.5, random_state=seed,
                                 stratify=np.array(labels)[rest])
    train_values = pd.concat([grouped[i][1] for i in train])[FEATURES].to_numpy(float)
    minimum, maximum = np.nanmin(train_values, 0), np.nanmax(train_values, 0)
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise ValueError('each feature needs an observed training value')
    scale = np.where(maximum > minimum, maximum - minimum, 1.)
    splits = {}
    for name, indices in [('train', train), ('val', val), ('test', test)]:
        sequences = []
        for i in indices:
            pid, frame = grouped[i]
            if frame.adverse_event.nunique() != 1:
                raise ValueError('patient label must be constant')
            # Merge repeated timestamps feature-wise; no rounding/binning.
            observations = frame.groupby('time_hours')[FEATURES].mean().sort_index()
            raw = observations.to_numpy(float)
            mask = np.isfinite(raw)
            sequences.append({'patient_id': int(pid),
                              'times': torch.tensor(observations.index.to_numpy() / 72., dtype=torch.float32),
                              'values': torch.tensor(np.where(mask, (raw - minimum) / scale, 0.), dtype=torch.float32),
                              'mask': torch.tensor(mask, dtype=torch.float32),
                              'label': float(labels[i])})
        splits[name] = SequenceDataset(sequences)
    metadata = {'seed': seed, 'features': FEATURES, 'minimum': minimum.tolist(),
                'scale': scale.tolist(), 'time_scale_hours': 72.,
                'patient_ids': {k: [s['patient_id'] for s in v.sequences] for k, v in splits.items()}}
    return splits, metadata


def collate_fn(sequences, task='interpolation'):
    if task == 'extrapolation':
        # Autonomous dynamics are invariant to translation. Align each patient's
        # last conditioning time at zero, retaining a common physical time scale.
        sequences = [{**s, 'times': s['times'] - s['times'][max(1, len(s['times']) // 2) - 1]}
                     for s in sequences]
    times = torch.unique(torch.cat([s['times'] for s in sequences]), sorted=True)
    shape = (len(sequences), len(times), len(FEATURES))
    values, mask = torch.zeros(shape), torch.zeros(shape)
    for i, seq in enumerate(sequences):
        indices = torch.searchsorted(times, seq['times'])
        values[i, indices], mask[i, indices] = seq['values'], seq['mask']
    return {'times': times, 'values': values, 'mask': mask,
            'labels': torch.tensor([s['label'] for s in sequences]),
            'patient_ids': [s['patient_id'] for s in sequences]}


def make_context(batch, task, fraction=.5, generator=None):
    """Return encoder inputs and scoring targets without sharing hidden values."""
    if not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0,1]')
    times, values, mask = batch['times'], batch['values'], batch['mask']
    if task == 'interpolation':
        context_mask = mask.clone()
        # Subsample actual measurement times independently for each patient.
        for i in range(len(values)):
            observed = torch.where(mask[i].bool().any(-1))[0]
            count = max(1, int(len(observed) * fraction))
            chosen = observed[torch.randperm(len(observed), generator=generator)[:count]]
            keep = torch.zeros(len(times), dtype=torch.bool)
            keep[chosen] = True
            context_mask[i, ~keep] = 0
        present = context_mask.bool().any((0, 2))
        # Keep the initial prior anchor even if that timestamp was subsampled.
        present[0] = True
        return {'values': torch.where(context_mask[:, present].bool(), values[:, present], 0.),
                'times': times[present], 'mask': context_mask[:, present],
                'query_times': times, 'target': values, 'target_mask': mask}
    if task != 'extrapolation':
        raise ValueError('unknown task')
    # collate_fn aligns each patient at its last first-half observation.
    context, future = times <= 0, times > 0
    if not context.any() or not future.any():
        raise ValueError('extrapolation needs observations in both halves of [0,72]h')
    if not mask[:, context].bool().any((1, 2)).all():
        raise ValueError('each patient needs history at or before the aligned cutoff')
    return {'values': values[:, context], 'times': times[context], 'mask': mask[:, context],
            'query_times': times[future], 'target': values[:, future], 'target_mask': mask[:, future]}
