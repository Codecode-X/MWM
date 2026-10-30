"""Numerical acceptance, causal interventions, leakage checks and descriptive audit."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .behavior_policy import ObservedHistory, propensity
from .common import dump_json, read_config, sha256
from .counterfactual_policies import decide
from .observation import clean_observations
from .simulator import Patient, PiecewiseTrajectory, integrate, rhs


def _replay_job(args):
    cfg, record = args
    p = record['patient']
    trajectory = PiecewiseTrajectory(p, p.initial_state, 0.0, record['intervals'], cfg)
    replay = clean_observations(trajectory.state(p.times), p.c, cfg)[0]
    original = clean_observations(record['states'].T, p.c, cfg)[0]
    return float(np.max(np.abs(replay - original)))


def paired_mechanisms(records, cfg):
    effects, feedback, risk_effects, propensity_effects, ablations = [], [], [], [], []
    for r in records:
        p = r['patient']
        j = np.searchsorted(p.times, cfg['simulation']['fork_after'])
        t, x = p.times[j], r['states'][j]
        end = t + max(cfg['simulation']['horizons'])
        no = integrate(p, x, t, end, 0, cfg).y[:, -1]
        yes = integrate(p, x, t, end, 1, cfg).y[:, -1]
        effect = clean_observations(yes, p.c, cfg)[0] - clean_observations(no, p.c, cfg)[0]
        effects.append((p.patient_id, p.c, effect))
        no_short = integrate(p, x, t, t + .5, 0, cfg).y[:, -1]
        yes_short = integrate(p, x, t, t + .5, 1, cfg).y[:, -1]
        feedback.append(float(yes_short[1] - no_short[1]))
        higher_L = x.copy()
        higher_L[1] += .2
        shifted = integrate(p, higher_L, t, t + .5, 0, cfg).y[:, -1]
        risk_effects.append(float(shifted[0] - no_short[0]))
        base = ObservedHistory(t + .5, 75., no_short[1])
        high = ObservedHistory(t + .5, 75., no_short[1] + .2)
        propensity_effects.append(propensity(high, cfg) - propensity(base, cfg))
        # Surgical ablations isolate the causal links rather than mere correlation.
        ab = deepcopy(cfg)
        ab['dynamics']['risk_weight'] = 0.0
        a = integrate(p, x, t, t + .5, 0, ab).y[:, -1]
        b = integrate(p, higher_L, t, t + .5, 0, ab).y[:, -1]
        ab['behavior']['risk_weight'] = 0.0
        risk_ab = propensity(high, ab) - propensity(base, ab)
        ablations.append((float(b[0] - a[0]), float(risk_ab)))
    values = np.array([e[2] for e in effects])
    result = {'paired_n': len(records), 'ate_3y': float(values.mean()), 'ite_sd': float(values.std()),
              'ite_quantiles': dict(zip(['p05', 'p50', 'p95'], np.quantile(values, [.05, .5, .95]).tolist())),
              'A_to_next_L_mean': float(np.mean(feedback)), 'L_to_next_D_mean': float(np.mean(risk_effects)),
              'controlled_L_to_propensity_mean': float(np.mean(propensity_effects)),
              'ablated_L_to_D_max': float(np.max(np.abs(np.array(ablations)[:, 0]))),
              'ablated_L_to_propensity_max': float(np.max(np.abs(np.array(ablations)[:, 1])))}
    if not (values.min() > 0 and max(feedback) < 0 and min(risk_effects) > 0):
        raise AssertionError(f'causal mechanism failed: {result}')
    if cfg['behavior']['gamma'] > 0 and min(propensity_effects) <= 0:
        raise AssertionError('current observable risk must increase treatment propensity')
    if result['ablated_L_to_D_max'] > 1e-7 or result['ablated_L_to_propensity_max'] > 1e-12:
        raise AssertionError('causal ablation failed')
    return result, pd.DataFrame(effects, columns=['patient_id', 'c', 'ite'])


def confounding_diagnostic(cfg):
    from .generate import factual
    rows = []
    for gamma in cfg['diagnostics']['gammas']:
        setting = deepcopy(cfg)
        setting['behavior']['gamma'] = gamma
        selected = []
        for i in range(cfg['diagnostics']['confounding_patients']):
            # Dedicated diagnostic population, isolated from benchmark splits.
            setting['seed'] = cfg['seed'] + 7777777
            r = factual(setting, 'train', i)
            p = r['patient']
            j = np.searchsorted(p.times, cfg['simulation']['fork_after'])
            x, t = r['states'][j], p.times[j]
            end, A = r['intervals'][j][1:]
            # Both potential next-visit changes use exactly this patient's dt.
            no = integrate(p, x, t, end, 0, setting).y[:, -1]
            yes = integrate(p, x, t, end, 1, setting).y[:, -1]
            y0, y1 = (clean_observations(v, p.c, setting)[0] for v in (no, yes))
            dt = end - t
            current = clean_observations(x, p.c, setting)[0]
            selected.append((x[1], x[0], current, r['propensities'][j], A,
                             ((y1 if A else y0) - current) / dt, (y1 - y0) / dt,
                             y1 if A else y0))
        a = np.array(selected)
        treated, untreated = a[:, 4] == 1, a[:, 4] == 0
        if not treated.any() or not untreated.any():
            raise AssertionError('diagnostic sample lacks treatment overlap')
        naive = a[treated, 5].mean() - a[untreated, 5].mean()
        true = a[:, 6].mean()
        rows.append({'gamma': gamma, 'n': len(a), 'risk_treatment_corr': float(np.corrcoef(a[:, 0], a[:, 4])[0, 1]),
                     'risk_propensity_spearman': float(spearmanr(a[:, 0], a[:, 3]).statistic) if gamma else 0.,
                     'naive_effect_rate': float(naive), 'paired_true_effect_rate': float(true),
                     'naive_bias': float(naive - true),
                     'treated_risk': float(a[treated, 0].mean()), 'untreated_risk': float(a[untreated, 0].mean()),
                     'treated_D': float(a[treated, 1].mean()), 'untreated_D': float(a[untreated, 1].mean()),
                     'treated_Y': float(a[treated, 2].mean()), 'untreated_Y': float(a[untreated, 2].mean()),
                     'treated_future_Y': float(a[treated, 7].mean()), 'untreated_future_Y': float(a[untreated, 7].mean())})
    return rows


def validate_dataset(data_dir, records=None, plots=True):
    from .dataset_adapter import COLUMNS, PublicDataset
    from .generate import factual, run_jobs
    root = Path(data_dir)
    cfg = read_config(root / 'config.yaml')
    manifest = json.loads((root / 'manifest.json').read_text())
    for filename, digest in manifest['files'].items():
        if sha256(root / filename) != digest:
            raise AssertionError(f'checksum mismatch: {filename}')
    private = root / 'oracle_private'
    hidden = pd.read_parquet(private / 'hidden_states.parquet')
    parameters = pd.read_parquet(private / 'patient_parameters.parquet')
    factual_visits = pd.read_parquet(private / 'factual_visits.parquet')
    factual_tr = pd.read_parquet(private / 'factual_treatments.parquet')
    cf = pd.read_parquet(private / 'cf_trajectories.parquet')
    cfhidden = pd.read_parquet(private / 'cf_hidden_states.parquet')
    cfvisits = pd.read_parquet(private / 'cf_visits.parquet')
    cftr = pd.read_parquet(private / 'cf_treatments.parquet')
    outcomes = pd.read_parquet(private / 'cf_outcomes.parquet')
    queries = pd.read_parquet(root / 'public/test_queries.parquet')
    specs = json.loads((root / 'public/policy_specs.json').read_text())
    numeric = pd.concat([hidden[['D_true', 'L_true', 'E_true']], cfhidden[['D_true', 'L_true', 'E_true']]])
    assert np.isfinite(numeric).all().all()
    assert numeric.D_true.min() >= -1e-7 and numeric.E_true.min() >= -1e-7 and numeric.E_true.max() <= 1 + 1e-7
    for df in (factual_visits, hidden, cf, cfhidden, parameters, outcomes):
        assert np.isfinite(df.select_dtypes(include='number')).all().all()
    lo, hi = cfg['behavior']['probability_bounds']
    assert hidden.propensity.between(lo, hi).all()
    if cfg['behavior']['gamma'] == 0:
        assert (hidden.propensity == .5).all()
    adapters = {s: PublicDataset(root / 'public', s) for s in cfg['splits']}
    patient_sets = [set(a.ids) for a in adapters.values()]
    assert all(not (patient_sets[i] & patient_sets[j]) for i in range(3) for j in range(i))
    for split, adapter in adapters.items():
        assert len(adapter) == cfg['splits'][split]
    fork_by_pid = queries.groupby('patient_id').fork_time.first().to_dict()
    for seq in adapters['test']:
        fork = fork_by_pid[seq['patient_id']]
        assert seq['times'][-1] == fork
        assert all(a['start'] < fork and a['end'] <= fork for a in seq['treatment_intervals'])
    assert cf.groupby(['patient_id', 'relative_time']).policy_id.nunique().eq(5).all()
    assert cf[cf.relative_time == 0].groupby('patient_id').Y_cf_true.nunique().eq(1).all()
    assert cfhidden[cfhidden.relative_time == 0].groupby('patient_id')[['D_true', 'L_true', 'E_true']].nunique().eq(1).all().all()
    assert (queries.fork_time >= cfg['simulation']['fork_after']).all()
    # Exact fork matching against private factual state.
    atfork = hidden[hidden.patient_id.isin(fork_by_pid)].copy()
    atfork = atfork[atfork.time == atfork.patient_id.map(fork_by_pid)]
    origin = cfhidden[cfhidden.relative_time == 0].merge(atfork, on='patient_id', suffixes=('_cf', '_fact'))
    for col in ('D_true', 'L_true', 'E_true'):
        assert np.array_equal(origin[col + '_cf'], origin[col + '_fact'])
    for r in cfvisits.itertuples():
        fork = fork_by_pid[r.patient_id]
        assert r.treatment == decide(r.policy_id, r.relative_time + fork, fork, r.X2_cf_true, specs)
    branch_divergence = cfvisits.pivot(index=['patient_id', 'relative_time'], columns='policy_id', values='X2_cf_true')
    unequal = (branch_divergence[0] - branch_divergence[1]).abs() > 1e-6
    assert unequal.any()
    train_X2 = pd.read_parquet(root / 'public/train_visits.parquet').X2_obs
    for p, q in zip(('3', '4'), cfg['policies']['quantiles']):
        assert specs['thresholds'][p] == float(train_X2.quantile(q))
    # All visit intervals satisfy the clinical schedule; terminal residual is separate.
    ordered = factual_visits.sort_values(['patient_id', 'time'])
    gaps = ordered.groupby('patient_id').time.diff().dropna()
    lower, upper = cfg['simulation']['visit_gap']
    assert gaps.between(lower - 1e-12, upper + 1e-12).all()
    assert factual_tr.treatment.isin([0, 1]).all()
    assert (factual_tr.t_end > factual_tr.t_start).all()
    if records is None:
        # Rebuild Patient objects from persisted initial state, parameters and treatment intervals.
        lookup_v = dict(tuple(factual_visits.groupby('patient_id')))
        lookup_t = dict(tuple(factual_tr.groupby('patient_id')))
        lookup_x = dict(tuple(hidden.groupby('patient_id')))
        records = {s: [] for s in cfg['splits']}
        for row in parameters.itertuples():
            p = Patient(row.patient_id, row.c, row.alpha_true, row.beta_true,
                        np.array([row.D0, row.L0, row.E0]), lookup_v[row.patient_id].time.values)
            tr = lookup_t[row.patient_id]
            records[row.split].append({'patient': p, 'states': lookup_x[row.patient_id][['D_true', 'L_true', 'E_true']].values,
                                       'intervals': list(zip(tr.t_start, tr.t_end, tr.treatment))})
    all_records = [r for group in records.values() for r in group]
    print(f'validate replay: all {len(all_records)} patients', flush=True)
    errors = run_jobs(_replay_job, ((cfg, r) for r in all_records), cfg['workers'])
    max_error = max(errors)
    assert max_error < 1e-6
    sample = records['train'][:cfg['diagnostics']['paired_patients']]
    paired, heterogeneity = paired_mechanisms(sample, cfg)
    # Tolerance agreement uses independent higher-accuracy solve.
    tight = deepcopy(cfg)
    tight['simulation']['solver'].update(rtol=1e-10, atol=1e-12)
    tol_errors = []
    for r in sample:
        p = r['patient']
        a = PiecewiseTrajectory(p, p.initial_state, 0., r['intervals'], cfg).state(p.times)
        b = PiecewiseTrajectory(p, p.initial_state, 0., r['intervals'], tight).state(p.times)
        tol_errors.append(float(np.max(np.abs(a - b))))
    assert max(tol_errors) < 1e-6
    # Verify serialized generation against independent per-purpose replay, no global RNG.
    for split in cfg['splits']:
        actual = factual(cfg, split, 0)
        pid = actual['patient'].patient_id
        persisted = factual_visits[factual_visits.patient_id == pid]
        assert np.array_equal(actual['observations'], persisted[['Y_obs', 'X1_obs', 'X2_obs']].values)
        assert np.array_equal(actual['states'], hidden[hidden.patient_id == pid][['D_true', 'L_true', 'E_true']].values)
    print('validate causal selection / feedback interventions', flush=True)
    confounding = confounding_diagnostic(cfg)
    # Selection uses actual observed treatment samples as well as known probabilities.
    joined = hidden.merge(factual_tr[['patient_id', 't_start', 'treatment']], left_on=['patient_id', 'time'],
                          right_on=['patient_id', 't_start'], validate='one_to_one')
    joined['risk_bin'] = pd.qcut(joined.L_true, 5, labels=False, duplicates='drop')
    overlap = joined.groupby('risk_bin').agg(n=('treatment', 'size'),
                propensity_mean=('propensity', 'mean'), treatment_fraction=('treatment', 'mean'),
                propensity_min=('propensity', 'min'), propensity_max=('propensity', 'max'), risk_mean=('L_true', 'mean'))
    risk_corr = float(np.corrcoef(joined.L_true, joined.treatment)[0, 1])
    prop_corr = float(spearmanr(joined.L_true, joined.propensity).statistic) if cfg['behavior']['gamma'] else 0.
    if cfg['behavior']['gamma'] > 0:
        assert prop_corr > 0, 'higher severity should have higher treatment probability'
        assert risk_corr > 0, 'observed risk-treatment association absent'
    diversity = {}
    for t, g in outcomes.groupby('relative_time'):
        pivot = g.pivot(index='patient_id', columns='policy_id', values='utility').sort_index(axis=1)
        best = pivot.values.argmax(axis=1)
        diversity[str(t)] = {str(p): int((best == p).sum()) for p in range(5)}
    counts = ordered.groupby('patient_id').size()
    switches = factual_tr.groupby('patient_id').treatment.apply(lambda x: int(np.count_nonzero(np.diff(x))))
    dynamic_switches = cftr[cftr.policy_id.isin([3, 4])].groupby(['patient_id', 'policy_id']).treatment.apply(lambda x: int(np.count_nonzero(np.diff(x))))
    stats = {'status': 'PASS', 'patient_counts': cfg['splits'], 'gamma': cfg['behavior']['gamma'],
             'checked_factual_patients_replay': len(errors), 'replay_max_Y_error': max_error,
             'tolerance_max_state_error': max(tol_errors), 'paired_causal_checks': paired,
             'propensity_quantiles': np.quantile(hidden.propensity, [0, .05, .5, .95, 1]).tolist(),
             'risk_treatment_correlation': risk_corr, 'risk_propensity_spearman': prop_corr,
             'overlap_by_risk_quintile': overlap.reset_index().to_dict('records'),
             'visit_counts': {'min': int(counts.min()), 'max': int(counts.max()), 'mean': float(counts.mean())},
             'visit_gaps': {'min': float(gaps.min()), 'max': float(gaps.max()), 'mean': float(gaps.mean()), 'variance': float(gaps.var())},
             'treatment_switches': {'mean': float(switches.mean()), 'max': int(switches.max())},
             'dynamic_branch_switches': {'mean': float(dynamic_switches.mean()), 'max': int(dynamic_switches.max())},
             'branch_risk_divergence_count': int(unequal.sum()), 'optimal_policy_counts': diversity,
             'confounding_diagnostics': confounding,
             'warnings': ['finite-sample naive bias need not increase monotonically with gamma; randomized error is sampling error',
                          'policy ties use the smallest policy_id; threshold policies may coincide for some patients'],
             'checks': ['finite states/observations', 'D/E invariants', 'solver tolerance', 'causal treatment effect',
                        'paired A->L and L->D, L->propensity with ablations', 'selection and overlap',
                        'all-patient factual replay', 'fork exact consistency', 'branch-local dynamic decisions',
                        'train-only thresholds', 'strictly irregular schedule', 'deterministic replay', 'public schema / cutoff isolation']}
    dump_json(root / 'diagnostics/statistics.json', stats)
    heterogeneity.to_parquet(root / 'diagnostics/paired_effects.parquet', index=False)
    pd.DataFrame(confounding).to_parquet(root / 'diagnostics/confounding.parquet', index=False)
    lines = ['# Level 3 合成数据验收报告', '', f'状态：**PASS**；gamma={stats["gamma"]}；患者数={cfg["splits"]}。',
             '', '这些检查是 SciPy 合成数据准备、数学验证和统计汇总，未训练或评估任何模型。',
             '', f'全部 {len(errors)} 位患者事实治疗重放的最大无噪声 Y 误差：{max_error:.3g}；',
             f'更严格容差的最大隐藏状态差异：{max(tol_errors):.3g}。没有裁剪任何隐藏状态。',
             '', '## 成对因果干预与消融', '', '```json', json.dumps(paired, indent=2), '```',
             '', '## 治疗选择偏差（固定下一检查终点，按每年结局变化率比较）', '',
             '| gamma | 风险–治疗相关 | 朴素效应 | 成对真实效应 | 偏差 |', '|---|---|---|---|---|']
    lines += [f'| {r["gamma"]} | {r["risk_treatment_corr"]:.4f} | {r["naive_effect_rate"]:.4f} | {r["paired_true_effect_rate"]:.4f} | {r["naive_bias"]:.4f} |' for r in confounding]
    lines += ['', '同一诊断总体在四种 gamma 下重用外生随机因素，用于控制比较；四套正式基准患者总体使用独立种子。',
              '朴素估计采用当次实际治疗的下一次检查变化率，两种潜在结局用同一检查间隔配对。',
              'gamma=0 的有限样本偏差不要求精确为零，偏差不要求随 gamma 单调变化。',
              '', '## 真实最优策略数量（含治疗成本）', '', '```json', json.dumps(diversity, indent=2), '```',
              '', '## 自动验收检查', ''] + [f'- PASS: {s}' for s in stats['checks']]
    lines += ['', '## 数据隔离与使用范围', '',
              '模型只挂载 public/；oracle_private/ 权限 0700，适配器不读 manifest/config/private。',
              '生成代码和私有审计文件对数据管理员可见，目录分离本身不隔离任意用户代码。',
              '8 年是积分边界而非额外检查，临床访问间隔均在 [0.25,0.75] 年。',
              '反事实积分到 8 年；标准评估保存分叉后至8年边界的逐月真值及精确 1/2/3 年终点。',
              '全部临床观测在本次治疗前生成；分叉历史不包含 fork 时刻的事实治疗。',
              '', '图表见 plots/；完整统计、风险分层重叠、策略切换和最优策略计数见 statistics.json。']
    (root / 'diagnostics/validation_report.md').write_text('\n'.join(lines) + '\n')
    if plots:
        from .plotting import plot_diagnostics
        plot_diagnostics(root)
    # Exact oracle prediction self-check is arithmetic evaluator validation, not a fitted model experiment.
    from .evaluation import score
    predictions = cf[['patient_id', 'policy_id', 'relative_time', 'Y_cf_true']].rename(columns={'Y_cf_true': 'Y_pred'})
    exact = outcomes[['patient_id', 'policy_id', 'relative_time', 'treatment_duration']].rename(columns={'treatment_duration': 'treatment_duration_pred'})
    predictions = predictions.merge(exact, on=['patient_id', 'policy_id', 'relative_time'], how='left')
    sanity = score(predictions, private)
    assert sanity['cf_rmse_trajectory'] == 0
    assert all(r['policy_regret'] == 0 for r in sanity['horizons'].values())
    dump_json(root / 'diagnostics/evaluator_selfcheck.json', sanity)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    result = validate_dataset(args.data_dir, plots=not args.no_plots)
    print(json.dumps({'status': result['status'], 'replay_max_Y_error': result['replay_max_Y_error']}, indent=2))


if __name__ == '__main__':
    main()
