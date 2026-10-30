import numpy as np
import pandas as pd
import pytest

from synthetic_cf_level3.evaluation import score


def oracle_predictions(root):
    private = root / 'oracle_private'
    pred = pd.read_parquet(private / 'cf_trajectories.parquet')[['patient_id', 'policy_id', 'relative_time', 'Y_cf_true']].rename(columns={'Y_cf_true': 'Y_pred'})
    duration = pd.read_parquet(private / 'cf_outcomes.parquet')[['patient_id', 'policy_id', 'relative_time', 'treatment_duration']].rename(columns={'treatment_duration': 'treatment_duration_pred'})
    return pred.merge(duration, on=['patient_id', 'policy_id', 'relative_time'], how='left')


def test_exact_and_known_error_metrics(small_dataset):
    root, _ = small_dataset
    pred = oracle_predictions(root)
    result = score(pred, root / 'oracle_private')
    assert result['cf_rmse_trajectory'] == 0
    for h in result['horizons'].values():
        assert h['effects_vs_never']['1']['sqrt_pehe'] == 0 and h['policy_regret'] == 0
    pred.Y_pred += np.where(pred.policy_id == 1, 2., 0.)
    changed = score(pred, root / 'oracle_private')
    assert abs(changed['cf_rmse_trajectory'] - np.sqrt(4 / 5)) < 1e-12
    for h in changed['horizons'].values():
        assert abs(h['effects_vs_never']['1']['sqrt_pehe'] - 2.) < 1e-12
        assert abs(h['effects_vs_never']['1']['ate_error'] - 2.) < 1e-12


def test_missing_duplicate_extra_and_nonfinite_rejected(small_dataset):
    root, _ = small_dataset
    pred = oracle_predictions(root)
    variants = [pred.iloc[:-1], pd.concat([pred, pred.iloc[:1]])]
    extra = pred.iloc[:1].copy()
    extra['patient_id'] = 'unknown'
    variants.append(pd.concat([pred, extra]))
    nonfinite = pred.copy()
    nonfinite.loc[0, 'Y_pred'] = np.inf
    variants.append(nonfinite)
    for v in variants:
        with pytest.raises(ValueError):
            score(v, root / 'oracle_private')


def test_regret_requires_model_own_treatment_duration(small_dataset):
    root, _ = small_dataset
    pred = oracle_predictions(root).drop(columns='treatment_duration_pred')
    result = score(pred, root / 'oracle_private')
    assert 'unavailable' in result['policy_regret_status']
    assert all(r['policy_regret'] is None for r in result['horizons'].values())
    pred['treatment_duration_pred'] = 99.
    with pytest.raises(ValueError, match='duration'):
        score(pred, root / 'oracle_private')


def test_interval_duration_scoring_and_coverage(small_dataset):
    root, _ = small_dataset
    pred = oracle_predictions(root).drop(columns='treatment_duration_pred')
    tr = pd.read_parquet(root / 'oracle_private/cf_treatments.parquet')
    result = score(pred, root / 'oracle_private', tr)
    assert all(r['policy_regret'] == 0 for r in result['horizons'].values())
    with pytest.raises(ValueError, match='coverage'):
        score(pred, root / 'oracle_private', tr[tr.policy_id != 4])
    tr.loc[0, 't_start'] = .01
    with pytest.raises(ValueError, match='cover'):
        score(pred, root / 'oracle_private', tr)
