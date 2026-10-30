"""Reproducible factual-only public data and isolated paired counterfactual oracle."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit
import scipy
import yaml

from .behavior_policy import ObservedHistory, propensity
from .common import DEFAULT_CONFIG, dump_json, prediction_grid, read_config, rng_for, sha256
from .counterfactual_policies import POLICY_NAMES, decide, treatment_duration
from .observation import clean_observations, observe
from .simulator import Patient, PiecewiseTrajectory, integrate


def make_patient(cfg, split, index):
    init_rng, schedule_rng = (rng_for(cfg, split, index, purpose) for purpose in (1, 2))
    p, d, s = cfg['initial'], cfg['dynamics'], cfg['simulation']
    c = float(init_rng.normal(p['c_mean'], p['c_sd']))
    D = float(init_rng.uniform(*p['D_range']))
    x = np.array([D, p['L_D'] * D + init_rng.normal(0, p['L_sd']), p['E']])
    times = [0.0]
    while True:
        next_time = times[-1] + schedule_rng.uniform(*s['visit_gap'])
        if next_time >= s['years']:
            break
        times.append(float(next_time))
    # 8 years is a simulation boundary, NOT an extra short-gap clinical visit.
    return Patient(f'{split}_{index:06d}', c, d['alpha_base'] + d['alpha_scale'] * expit(c),
                   d['beta_base'] + d['beta_scale'] * expit(-c), x, np.array(times))


def factual(cfg, split, index):
    patient = make_patient(cfg, split, index)
    noise_rng, decision_rng = (rng_for(cfg, split, index, purpose) for purpose in (3, 4))
    x, prev_Y, prev_t, prev_A = patient.initial_state.copy(), None, None, 0
    observations, states, probs, actions, intervals = [], [], [], [], []
    for k, t in enumerate(patient.times):
        states.append(x.copy())
        obs = observe(x, patient.c, noise_rng, cfg)
        p = propensity(ObservedHistory(t, obs[0], obs[2], prev_t, prev_Y, prev_A), cfg)
        A = int(decision_rng.random() < p)
        end = patient.times[k + 1] if k + 1 < len(patient.times) else cfg['simulation']['years']
        intervals.append((float(t), float(end), A))
        observations.append(obs)
        probs.append(p)
        actions.append(A)
        x = integrate(patient, x, t, end, A, cfg).y[:, -1]
        prev_t, prev_Y, prev_A = t, obs[0], A
    return {'patient': patient, 'observations': np.array(observations), 'states': np.array(states),
            'propensities': np.array(probs), 'actions': np.array(actions), 'intervals': intervals,
            'boundary_state': x}


def branch(record, policy_id, specs, cfg):
    patient = record['patient']
    j = int(np.searchsorted(patient.times, cfg['simulation']['fork_after']))
    fork = patient.times[j]
    x, intervals, states, actions = record['states'][j].copy(), [], [], []
    future = patient.times[j:]
    # Each decision uses this branch's current observable L. No future factual observations.
    for k, t in enumerate(future):
        states.append(x.copy())
        A = decide(policy_id, t, fork, float(x[1]), specs)
        end = future[k + 1] if k + 1 < len(future) else cfg['simulation']['years']
        intervals.append((float(t), float(end), A))
        actions.append(A)
        x = integrate(patient, x, t, end, A, cfg).y[:, -1]
    # Reintegrating intervals provides an independently reusable dense trajectory.
    trajectory = PiecewiseTrajectory(patient, record['states'][j], fork, intervals, cfg)
    grid = prediction_grid(cfg, cfg['simulation']['years'] - fork)
    hidden = trajectory.state(fork + grid)
    clean = clean_observations(hidden, patient.c, cfg)
    return {'fork': float(fork), 'grid': grid, 'hidden': hidden, 'clean': clean,
            'intervals': intervals, 'treatment': trajectory.treatment(fork + grid),
            'visit_times': future, 'visit_states': np.array(states), 'visit_actions': actions}


def _fact_job(args):
    return factual(*args)


def _branch_job(args):
    cfg, record, specs = args
    return record['patient'].patient_id, [branch(record, p, specs, cfg) for p in cfg['policies']['ids']]


def run_jobs(func, args, workers):
    if workers == 1:
        return list(map(func, args))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(func, args, chunksize=16))


def frames(records, split, cfg):
    patients, visits, treatments, audit, params = [], [], [], [], []
    for r in records:
        p = r['patient']
        patients.append((p.patient_id, p.c))
        params.append((p.patient_id, split, p.c, p.alpha, p.beta, *p.initial_state))
        for k, (t, obs, x, prop) in enumerate(zip(p.times, r['observations'], r['states'], r['propensities'])):
            visits.append((p.patient_id, k, t, *obs))
            audit.append((p.patient_id, split, k, t, *x,
                          clean_observations(x, p.c, cfg)[0], prop))
        treatments.extend((p.patient_id, a, b, u) for a, b, u in r['intervals'])
    return (pd.DataFrame(patients, columns=['patient_id', 'c']),
            pd.DataFrame(visits, columns=['patient_id', 'visit_index', 'time', 'Y_obs', 'X1_obs', 'X2_obs']),
            pd.DataFrame(treatments, columns=['patient_id', 't_start', 't_end', 'treatment']),
            pd.DataFrame(audit, columns=['patient_id', 'split', 'visit_index', 'time', 'D_true', 'L_true',
                                         'E_true', 'Y_clean', 'propensity']),
            pd.DataFrame(params, columns=['patient_id', 'split', 'c', 'alpha_true', 'beta_true', 'D0', 'L0', 'E0']))


def write_frame(path, frame):
    frame.to_parquet(path, index=False, compression='zstd')


def generate(cfg, out_dir, validate=True):
    cfg = deepcopy(cfg)
    out = Path(out_dir)
    cfg['out_dir'] = str(out)
    if (out / 'manifest.json').exists() or (out / 'public').exists():
        raise FileExistsError(f'refusing to overwrite existing dataset: {out}')
    for name in ('public', 'oracle_private', 'diagnostics/plots'):
        (out / name).mkdir(parents=True, exist_ok=True)
    # Permission boundary supplements the loader boundary. For deployment mount ONLY public/.
    os.chmod(out / 'oracle_private', 0o700)
    start = time.time()
    all_records, audits, parameters, private_visits, private_treatments = {}, [], [], [], []
    workers = cfg['workers']
    for split, count in cfg['splits'].items():
        print(f'gamma={cfg["behavior"]["gamma"]}: simulate {split} n={count}', flush=True)
        records = run_jobs(_fact_job, ((cfg, split, i) for i in range(count)), workers)
        all_records[split] = records
        patients, visits, treatments, audit, params = frames(records, split, cfg)
        audits.append(audit)
        parameters.append(params)
        private_visits.append(visits.assign(split=split))
        private_treatments.append(treatments.assign(split=split))
        write_frame(out / 'public' / f'{split}_patients.parquet', patients)
        if split != 'test':
            write_frame(out / 'public' / f'{split}_visits.parquet', visits)
            write_frame(out / 'public' / f'{split}_treatments.parquet', treatments)
        else:
            forks = {r['patient'].patient_id: r['patient'].times[
                np.searchsorted(r['patient'].times, cfg['simulation']['fork_after'])] for r in records}
            write_frame(out / 'public/test_histories.parquet',
                        visits[visits.time <= visits.patient_id.map(forks)])
            write_frame(out / 'public/test_history_treatments.parquet',
                        treatments[treatments.t_start < treatments.patient_id.map(forks)])
    training_X2 = np.concatenate([r['observations'][:, 2] for r in all_records['train']])
    q = np.quantile(training_X2, cfg['policies']['quantiles'])
    specs = {'policies': {str(i): n for i, n in enumerate(POLICY_NAMES)},
             'thresholds': {'3': float(q[0]), '4': float(q[1])},
             'threshold_source': 'all training visits only', 'delay_years': cfg['policies']['delay_years'],
             'treatment_cost': cfg['policies']['treatment_cost'],
             'decision_rule': 'visit observation first; use current branch X2; strict > threshold',
             'treatment_semantics': '[t_start,t_end); no decision at terminal boundary'}
    if 'cost_calibration' in cfg['policies']:
        provenance = cfg['policies']['cost_calibration']
        specs['cost_calibration'] = {k: provenance[k] for k in (
            'source', 'n', 'selected_cost', 'common_cost_frozen_for_all_gammas')}
    if cfg['policies']['tune_cost']:
        # Optional calibration uses only training patients, before any test branch is computed.
        calibration = all_records['train'][:cfg['diagnostics']['paired_patients']]
        horizon = max(cfg['simulation']['horizons'])
        clean_y, durations = [], []
        for r in calibration:
            bs = [branch(r, p, specs, cfg) for p in cfg['policies']['ids']]
            clean_y.append([b['clean'][0, np.flatnonzero(b['grid'] == horizon)[0]] for b in bs])
            durations.append([treatment_duration(b['intervals'], b['fork'], b['fork'] + horizon) for b in bs])
        scores = []
        for cost in cfg['policies']['cost_candidates']:
            choices = np.argmax(np.array(clean_y) - cost * np.array(durations), axis=1)
            proportions = np.bincount(choices, minlength=5) / len(choices)
            entropy = -np.sum(proportions[proportions > 0] * np.log(proportions[proportions > 0]))
            scores.append((float(entropy), -abs(cost - 2.0), cost))
        specs['treatment_cost'] = max(scores)[2]
        specs['cost_calibration'] = {'source': 'training only', 'n': len(calibration), 'scores': scores}
        cfg['policies']['treatment_cost'] = specs['treatment_cost']
    dump_json(out / 'public/policy_specs.json', specs)
    # Patient-level JSONL retains ragged sequences, with no pickle deserialization.
    from .dataset_adapter import PublicDataset
    for split in cfg['splits']:
        adapter = PublicDataset(out / 'public', split)
        with (out / 'public' / f'{split}_sequences.jsonl').open('w') as f:
            for seq in adapter:
                f.write(json.dumps(seq, allow_nan=False) + '\n')
    print('generate paired test counterfactuals', flush=True)
    results = run_jobs(_branch_job, ((cfg, r, specs) for r in all_records['test']), workers)
    trajectories, outcomes, intervals, hidden, cf_visits, queries, schedules = [], [], [], [], [], [], []
    for pid, branches in results:
        fork = branches[0]['fork']
        schedules.extend((pid, float(t)) for t in branches[0]['visit_times'])
        for policy_id, b in zip(cfg['policies']['ids'], branches):
            queries.append((pid, fork, policy_id, cfg['simulation']['horizons'],
                            b['grid'].tolist(), b['visit_times'].tolist()))
            for j, t in enumerate(b['grid']):
                trajectories.append((pid, policy_id, t, *b['clean'][:, j], int(b['treatment'][j])))
                hidden.append((pid, policy_id, t, *b['hidden'][:, j]))
            for t in cfg['simulation']['horizons']:
                j = int(np.flatnonzero(b['grid'] == t)[0])
                duration = treatment_duration(b['intervals'], fork, fork + t)
                outcomes.append((pid, policy_id, t, b['clean'][0, j], duration,
                                 b['clean'][0, j] - specs['treatment_cost'] * duration))
            intervals.extend((pid, policy_id, a - fork, z - fork, u) for a, z, u in b['intervals'])
            cf_visits.extend((pid, policy_id, float(t - fork), float(x[1]), A)
                             for t, x, A in zip(b['visit_times'], b['visit_states'], b['visit_actions']))
    tables = {
        'cf_trajectories': pd.DataFrame(trajectories, columns=['patient_id', 'policy_id', 'relative_time',
                                      'Y_cf_true', 'X1_cf_true', 'X2_cf_true', 'treatment']),
        'cf_outcomes': pd.DataFrame(outcomes, columns=['patient_id', 'policy_id', 'relative_time',
                                  'Y_cf_true', 'treatment_duration', 'utility']),
        'cf_treatments': pd.DataFrame(intervals, columns=['patient_id', 'policy_id', 't_start', 't_end', 'treatment']),
        'cf_hidden_states': pd.DataFrame(hidden, columns=['patient_id', 'policy_id', 'relative_time', 'D_true', 'L_true', 'E_true']),
        'cf_visits': pd.DataFrame(cf_visits, columns=['patient_id', 'policy_id', 'relative_time', 'X2_cf_true', 'treatment']),
        'hidden_states': pd.concat(audits, ignore_index=True),
        'patient_parameters': pd.concat(parameters, ignore_index=True),
        'factual_visits': pd.concat(private_visits, ignore_index=True),
        'factual_treatments': pd.concat(private_treatments, ignore_index=True),
    }
    for name, table in tables.items():
        write_frame(out / 'oracle_private' / f'{name}.parquet', table)
    write_frame(out / 'public/test_queries.parquet', pd.DataFrame(queries, columns=[
        'patient_id', 'fork_time', 'policy_id', 'prediction_times', 'trajectory_times', 'future_visit_times']))
    write_frame(out / 'public/test_future_visits.parquet', pd.DataFrame(schedules, columns=['patient_id', 'time']))
    dump_json(out / 'oracle_private/evaluation_spec.json', {
        'horizons': cfg['simulation']['horizons'], 'treatment_cost': specs['treatment_cost'],
        'gamma': cfg['behavior']['gamma'], 'tie_break': 'smallest policy_id', 'policies': cfg['policies']['ids']})
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    source_hashes = {str(p.relative_to(Path(__file__).parent)): sha256(p)
                     for p in sorted(Path(__file__).parent.rglob('*.py'))}
    manifest = {'version': cfg['version'], 'seed': cfg['seed'], 'gamma': cfg['behavior']['gamma'],
                'splits': cfg['splits'], 'split_streams': cfg['split_streams'], 'policy_specs': specs,
                'simulation_boundary': cfg['simulation']['years'], 'time_unit': 'year',
                'terminal_boundary_is_visit': False, 'oracle_sampling': 'monthly fork..8 years, plus exact 1/2/3-year horizons and terminal boundary',
                'numerics': cfg['simulation']['solver'], 'initialization': cfg['initial'],
                'final_dynamics': cfg['dynamics'], 'treatment_cost': specs['treatment_cost'],
                'rng': 'SeedSequence([seed, split_stream, patient_index, purpose]); 1=init 2=schedule 3=noise 4=treatment',
                'environment': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                                'pandas': pd.__version__, 'platform': platform.platform(), 'conda_env': os.environ.get('CONDA_DEFAULT_ENV')},
                'execution': {'kind': 'SciPy synthetic data preparation; no model experiment',
                              'device': 'CPU data preparation', 'tensor_devices': [], 'physical_gpu_assignments': [],
                              'workers': workers, 'elapsed_generation_seconds': time.time() - start},
                'source_sha256': source_hashes,
                'files': {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    dump_json(out / 'manifest.json', manifest)
    if validate:
        from .validation import validate_dataset
        validate_dataset(out, records=all_records)
    print(f'complete: {out} ({time.time() - start:.1f}s)', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--out-dir')
    parser.add_argument('--smoke', action='store_true', help='100 total patients: 60/20/20')
    parser.add_argument('--all-gammas', action='store_true')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--gamma', type=float)
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.workers:
        cfg['workers'] = args.workers
    if args.gamma is not None:
        cfg['behavior']['gamma'] = args.gamma
    if args.smoke:
        cfg['splits'] = {'train': 60, 'val': 20, 'test': 20}
        cfg['diagnostics']['paired_patients'] = 20
        cfg['diagnostics']['confounding_patients'] = 100
    out = Path(args.out_dir or cfg['out_dir'])
    if args.all_gammas:
        paths = []
        for j, gamma in enumerate(cfg['diagnostics']['gammas']):
            setting = deepcopy(cfg)
            setting['behavior']['gamma'] = gamma
            setting['seed'] += j * 1000000  # independent benchmark populations
            paths.append(generate(setting, out / f'gamma_{gamma:g}'))
        from .plotting import suite_plot
        suite_plot(paths, out)
    else:
        generate(cfg, out)


if __name__ == '__main__':
    main()
