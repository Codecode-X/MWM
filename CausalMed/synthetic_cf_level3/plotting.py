"""Standalone diagnostic figures; no model loading or inference."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .counterfactual_policies import POLICY_NAMES

COLORS = ['#777777', '#247ba0', '#e58f27', '#39a275', '#a45aaf']


def save(fig, root, name):
    fig.tight_layout()
    fig.savefig(root / f'{name}.png', dpi=160)
    fig.savefig(root / f'{name}.svg')
    plt.close(fig)


def causal_graph(root):
    fig, ax = plt.subplots(figsize=(11, 5))
    positions = {'L(k)': (0, 2), 'A(k)': (1.5, 2), 'E(k+1)': (3, 2),
                 'L(k+1)': (4.5, 2), 'A(k+1)': (6.2, 2), 'D(k+1)': (3, .3),
                 'Y(k+1)': (4.5, .3), 'Y(k+2)': (6.2, .3)}
    for label, pos in positions.items():
        ax.text(*pos, label, ha='center', va='center', bbox={'boxstyle': 'round', 'fc': '#e4f1f8', 'pad': .5}, fontsize=12)
    edges = [('L(k)', 'A(k)'), ('A(k)', 'E(k+1)'), ('E(k+1)', 'L(k+1)'),
             ('L(k+1)', 'A(k+1)'), ('L(k)', 'D(k+1)'), ('E(k+1)', 'D(k+1)'),
             ('D(k+1)', 'Y(k+1)'), ('L(k+1)', 'Y(k+1)'), ('A(k+1)', 'Y(k+2)'),
             ('L(k+1)', 'Y(k+2)')]
    for source, target in edges:
        ax.annotate('', xy=positions[target], xytext=positions[source],
                    arrowprops={'arrowstyle': '->', 'color': '#415b6e', 'shrinkA': 30, 'shrinkB': 34, 'lw': 1.6})
    ax.set(xlim=(-.7, 6.9), ylim=(-.4, 2.7), title='Treatment / time-varying confounder feedback (arrows may include mediated paths)')
    ax.axis('off')
    save(fig, root, 'figure1_causal_graph')


def plot_diagnostics(data_dir):
    data = Path(data_dir)
    out, private = data / 'diagnostics/plots', data / 'oracle_private'
    stats = json.loads((data / 'diagnostics/statistics.json').read_text())
    specs = json.loads((data / 'public/policy_specs.json').read_text())
    visits = pd.read_parquet(private / 'factual_visits.parquet')
    tr = pd.read_parquet(private / 'factual_treatments.parquet')
    cf = pd.read_parquet(private / 'cf_trajectories.parquet')
    cftr = pd.read_parquet(private / 'cf_treatments.parquet')
    cfvisits = pd.read_parquet(private / 'cf_visits.parquet')
    causal_graph(out)
    # Seeded random patient selection, independent from the generation RNG.
    ids = visits.patient_id.unique()
    selected = np.random.default_rng(915).choice(ids, min(5, len(ids)), replace=False)
    fig, axes = plt.subplots(len(selected), 2, figsize=(12, 2.5 * len(selected)), squeeze=False)
    for row, pid in enumerate(selected):
        v, intervals = visits[visits.patient_id == pid], tr[tr.patient_id == pid]
        for ax in axes[row]:
            for r in intervals[intervals.treatment == 1].itertuples():
                ax.axvspan(r.t_start, r.t_end, color='#39a275', alpha=.16)
        axes[row, 0].plot(v.time, v.Y_obs, 'o-', ms=3)
        axes[row, 0].set(ylabel='Observed Y', title=f'{pid} (green: treatment)')
        axes[row, 1].plot(v.time, v.X2_obs, 'o-', ms=3, color=COLORS[2])
        axes[row, 1].set(ylabel='Observable risk X2', title=pid)
    for ax in axes[-1]:
        ax.set_xlabel('Years')
    save(fig, out, 'figure2_factual_patients')
    # Select an informative example: maximum dynamic policy switches (private diagnostic only).
    switches = cftr[cftr.policy_id.isin([3, 4])].groupby('patient_id').treatment.apply(lambda v: int(np.count_nonzero(np.diff(v))))
    pid = switches.idxmax()
    example = cf[cf.patient_id == pid]
    fig, ax = plt.subplots(figsize=(9, 5))
    for p, g in example.groupby('policy_id'):
        ax.plot(g.relative_time, g.Y_cf_true, label=POLICY_NAMES[p], color=COLORS[p])
    ax.set(xlabel='Years after fork', ylabel='Clean counterfactual Y', title=f'Paired counterfactual trajectories: {pid}')
    ax.legend()
    save(fig, out, 'figure3_counterfactuals')
    overlap = pd.DataFrame(stats['overlap_by_risk_quintile'])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(overlap.risk_mean, overlap.propensity_mean, 'o-', label='True propensity')
    ax.plot(overlap.risk_mean, overlap.treatment_fraction, 's--', label='Observed treatment fraction')
    ax.set(xlabel='Risk X2 (quintile mean)', ylabel='Probability / treatment fraction', ylim=(0, 1), title='Risk stratification and treatment overlap')
    ax.legend()
    save(fig, out, 'figure4_confounding_overlap')
    effects = pd.read_parquet(data / 'diagnostics/paired_effects.parquet')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(effects.ite, bins=min(25, max(5, len(effects) // 5)), color=COLORS[1], alpha=.8)
    axes[0].set(xlabel='3-year individual effect (Always - Never)', ylabel='Patients')
    axes[1].scatter(effects.c, effects.ite, alpha=.6, s=18)
    axes[1].set(xlabel='Static feature c', ylabel='3-year individual treatment effect')
    save(fig, out, 'figure5_effect_heterogeneity')
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for p, g in example.groupby('policy_id'):
        axes[0].plot(g.relative_time, g.X2_cf_true, color=COLORS[p], label=POLICY_NAMES[p])
        axes[2].plot(g.relative_time, g.Y_cf_true, color=COLORS[p])
        v = cfvisits[(cfvisits.patient_id == pid) & (cfvisits.policy_id == p)]
        v = v[v.relative_time <= 3.0]
        axes[0].scatter(v.relative_time, v.X2_cf_true, s=12, color=COLORS[p])
        intervals = cftr[(cftr.patient_id == pid) & (cftr.policy_id == p)]
        # Separate vertical lanes so coinciding treatment values remain visible.
        xs = list(intervals.t_start) + [intervals.t_end.iloc[-1]]
        ys = list(intervals.treatment) + [intervals.treatment.iloc[-1]]
        axes[1].step(xs, np.array(ys) * .65 + p, where='post', color=COLORS[p])
    for p in ('3', '4'):
        axes[0].axhline(specs['thresholds'][p], ls=':', color=COLORS[int(p)], alpha=.7)
    axes[0].set(ylabel='Branch-specific X2', title=f'Closed-loop decisions at branch-specific visits: {pid}')
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].set(ylabel='Treatment lanes (height = on)', yticks=range(5), yticklabels=POLICY_NAMES)
    axes[2].set(xlabel='Years after fork', ylabel='Clean Y', xlim=(0, 3))
    save(fig, out, 'figure6_dynamic_policy_timing')
    confounding_plot(pd.DataFrame(stats['confounding_diagnostics']), out, 'figure7_confounding_strength')


def confounding_plot(df, out, name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df.gamma, df.risk_treatment_corr, 'o-')
    axes[0].axhline(0, color='gray', lw=.8)
    axes[0].set(xlabel='Confounding strength gamma', ylabel='Risk / treatment correlation')
    axes[1].plot(df.gamma, df.naive_effect_rate, 'o-', label='Naive observational effect')
    axes[1].plot(df.gamma, df.paired_true_effect_rate, 's-', label='Paired causal effect')
    axes[1].plot(df.gamma, df.naive_bias, 'x--', label='Naive bias')
    axes[1].set(xlabel='Confounding strength gamma', ylabel='Next-visit Y change per year')
    axes[1].legend(fontsize=8)
    save(fig, out, name)


def suite_plot(paths, out):
    rows = []
    for p in paths:
        s = json.loads((p / 'diagnostics/statistics.json').read_text())
        rows.append({'gamma': s['gamma'], 'risk_treatment_corr': s['risk_treatment_correlation'],
                     'risk_propensity_spearman': s['risk_propensity_spearman'], 'path': str(p),
                     'status': s['status'], 'patients': sum(s['patient_counts'].values())})
    dump = {'datasets': rows, 'independent_populations': True,
            'model_evaluation': 'not performed; supply model predictions to evaluation.py then --suite'}
    (out / 'suite_manifest.json').write_text(json.dumps(dump, indent=2) + '\n')
    pd.DataFrame(rows).to_csv(out / 'suite_statistics.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([r['gamma'] for r in rows], [r['risk_treatment_corr'] for r in rows], 'o-', label='Observed risk / treatment correlation')
    ax.set(xlabel='Gamma', ylabel='Correlation', title='Four independent benchmark populations')
    ax.legend()
    save(fig, out, 'confounding_strength')


def evaluation_suite_plot(directories, out):
    rows = []
    for directory in directories:
        metrics = json.loads((Path(directory) / 'metrics.json').read_text())
        for t, result in metrics['horizons'].items():
            effect = result['effects_vs_never']['1']
            rows.append({'gamma': metrics['gamma'], 'horizon': float(t),
                         'cf_rmse': result['cf_rmse'], 'sqrt_pehe': effect['sqrt_pehe'],
                         'ate_error': effect['ate_error'], 'policy_regret': result['policy_regret']})
    df = pd.DataFrame(rows).sort_values(['horizon', 'gamma'])
    if set(df.gamma) != {0., .5, 1., 2.} or df.duplicated(['gamma', 'horizon']).any():
        raise ValueError('suite must contain exactly one evaluation per gamma=0,0.5,1,2')
    df.to_csv(out / 'comparison.csv', index=False)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, metric in zip(axes, ['cf_rmse', 'sqrt_pehe', 'ate_error', 'policy_regret']):
        for t, g in df.groupby('horizon'):
            ax.plot(g.gamma, g[metric], 'o-', label=f'{t:g} yr')
        ax.set(xlabel='Gamma', ylabel=metric)
        ax.legend()
    save(fig, out, 'prediction_error_vs_confounding')
