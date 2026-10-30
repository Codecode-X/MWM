"""Offline scoring; oracle access belongs only to this separate evaluation process."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import dump_json

KEYS = ['patient_id', 'policy_id', 'relative_time']


def canonicalize(df, value):
    df = df.copy()
    if not set(KEYS + [value]) <= set(df.columns):
        raise ValueError(f'required columns: {KEYS + [value]}')
    if df[KEYS + [value]].isna().any().any():
        raise ValueError('null keys or predictions')
    if not np.isfinite(df[['relative_time', value]].values).all():
        raise ValueError('nonfinite times or predictions')
    if not np.all(df.policy_id == df.policy_id.astype(int)):
        raise ValueError('policy_id must be integral')
    df['policy_id'] = df.policy_id.astype(int)
    df['relative_time'] = df.relative_time.round(9)
    if df.duplicated(KEYS).any():
        raise ValueError('duplicate prediction keys (times compared to 1e-9 years)')
    return df


def predicted_durations(intervals, reference):
    needed = {'patient_id', 'policy_id', 't_start', 't_end', 'treatment'}
    if not needed <= set(intervals.columns) or intervals[list(needed)].isna().any().any():
        raise ValueError('invalid predicted treatment interval schema')
    if not np.isfinite(intervals[['t_start', 't_end', 'treatment']]).all().all():
        raise ValueError('nonfinite predicted treatment interval')
    if not intervals.treatment.isin([0, 1]).all() or (intervals.t_end <= intervals.t_start).any():
        raise ValueError('predicted treatments must be binary, positive-length intervals')
    groups = dict(tuple(intervals.groupby(['patient_id', 'policy_id'])))
    expected = set(map(tuple, reference[['patient_id', 'policy_id']].drop_duplicates().values))
    if set(groups) != expected:
        raise ValueError('predicted treatment patient/policy coverage must be exact')
    answer = []
    for (pid, policy), rows in reference.groupby(['patient_id', 'policy_id']):
        g = groups[(pid, policy)].sort_values('t_start')
        if abs(g.t_start.iloc[0]) > 1e-8 or not np.allclose(g.t_start.values[1:], g.t_end.values[:-1], atol=1e-8, rtol=0):
            raise ValueError('predicted intervals must continuously cover time from zero')
        if g.t_end.iloc[-1] < rows.relative_time.max() - 1e-8:
            raise ValueError('insufficient predicted treatment follow-up')
        for t in rows.relative_time:
            duration = np.sum(np.maximum(0, np.minimum(g.t_end, t) - np.maximum(g.t_start, 0)) * g.treatment)
            answer.append((pid, policy, t, duration))
    return pd.DataFrame(answer, columns=KEYS + ['treatment_duration_pred'])


def score(predictions, oracle_dir, predicted_treatments=None, mode='trajectory'):
    import json
    root = Path(oracle_dir)
    spec = json.loads((root / 'evaluation_spec.json').read_text())
    truth_file = 'cf_trajectories.parquet' if mode == 'trajectory' else 'cf_outcomes.parquet'
    truth = canonicalize(pd.read_parquet(root / truth_file), 'Y_cf_true')
    pred = canonicalize(predictions, 'Y_pred')
    joined = truth[KEYS + ['Y_cf_true']].merge(pred, on=KEYS, how='outer', indicator=True, validate='one_to_one')
    if not joined['_merge'].eq('both').all():
        counts = joined['_merge'].value_counts().to_dict()
        raise ValueError(f'prediction coverage must match every required oracle key exactly: {counts}')
    joined = joined.drop(columns='_merge')
    horizon_truth = canonicalize(pd.read_parquet(root / 'cf_outcomes.parquet'), 'Y_cf_true')
    h = horizon_truth.merge(pred, on=KEYS, how='left', validate='one_to_one')
    if predicted_treatments is not None:
        if 'treatment_duration_pred' in h:
            raise ValueError('provide predicted durations OR predicted intervals, not both')
        h = h.merge(predicted_durations(predicted_treatments, horizon_truth), on=KEYS, validate='one_to_one')
    duration_available = 'treatment_duration_pred' in h.columns
    if duration_available:
        dur = h.treatment_duration_pred.values
        if not np.isfinite(dur).all() or np.any(dur < -1e-8) or np.any(dur > h.relative_time + 1e-8):
            raise ValueError('predicted duration must be finite and between zero and horizon for all queries')
    metrics = {'gamma': spec['gamma'], 'mode': mode, 'n_patients': int(h.patient_id.nunique()),
               'prediction_rows': len(pred), 'treatment_cost': spec['treatment_cost'],
               'cf_rmse_trajectory': float(np.sqrt(np.mean((joined.loc[joined.relative_time > 0, 'Y_pred'] -
                                                          joined.loc[joined.relative_time > 0, 'Y_cf_true']) ** 2)))
                    if mode == 'trajectory' else None,
               'policy_regret_status': 'available' if duration_available else
                   'unavailable: supply treatment_duration_pred at every horizon or --predicted-treatments; oracle durations are never substituted',
               'horizons': {}}
    for t in spec['horizons']:
        g = h[h.relative_time == round(t, 9)]
        true_y = g.pivot(index='patient_id', columns='policy_id', values='Y_cf_true').sort_index(axis=1)
        pred_y = g.pivot(index='patient_id', columns='policy_id', values='Y_pred').sort_index(axis=1)
        result = {'cf_rmse': float(np.sqrt(np.mean((g.Y_pred - g.Y_cf_true) ** 2))),
                  'by_policy_rmse': {str(p): float(np.sqrt(np.mean((v.Y_pred - v.Y_cf_true) ** 2)))
                                     for p, v in g.groupby('policy_id')}, 'effects_vs_never': {}}
        for p in [1, 2, 3, 4]:
            tau, estimate = true_y[p] - true_y[0], pred_y[p] - pred_y[0]
            result['effects_vs_never'][str(p)] = {
                'sqrt_pehe': float(np.sqrt(np.mean((estimate - tau) ** 2))),
                'ate_error': float(abs(estimate.mean() - tau.mean())),
                'true_ate': float(tau.mean()), 'predicted_ate': float(estimate.mean())}
        true_u = g.pivot(index='patient_id', columns='policy_id', values='utility').reindex_like(true_y)
        true_best = true_u.values.argmax(axis=1)
        result['true_optimal_policy_counts'] = {str(p): int((true_best == p).sum()) for p in range(5)}
        if duration_available:
            pred_d = g.pivot(index='patient_id', columns='policy_id', values='treatment_duration_pred').reindex_like(true_y)
            chosen = (pred_y - spec['treatment_cost'] * pred_d).values.argmax(axis=1)
            regrets = true_u.values.max(axis=1) - true_u.values[np.arange(len(chosen)), chosen]
            result['policy_regret'] = float(regrets.mean())
            result['optimal_policy_accuracy'] = float(np.mean(chosen == true_best))
            result['selected_policy_counts'] = {str(p): int((chosen == p).sum()) for p in range(5)}
        else:
            result['policy_regret'] = None
        metrics['horizons'][str(t)] = result
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions')
    parser.add_argument('--oracle-dir')
    parser.add_argument('--predicted-treatments', help='intervals in RELATIVE years, generated by model own rollout')
    parser.add_argument('--mode', choices=['trajectory', 'horizons'], default='trajectory')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--suite', nargs='+', help='four evaluation directories, each containing metrics.json')
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.suite:
        from .plotting import evaluation_suite_plot
        evaluation_suite_plot(args.suite, out)
        return
    if not args.predictions or not args.oracle_dir:
        parser.error('--predictions and --oracle-dir are required unless --suite is used')
    pred = pd.read_parquet(args.predictions)
    intervals = pd.read_parquet(args.predicted_treatments) if args.predicted_treatments else None
    metrics = score(pred, args.oracle_dir, intervals, args.mode)
    dump_json(out / 'metrics.json', metrics)
    print(__import__('json').dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
