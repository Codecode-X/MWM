"""Optional common treatment-cost calibration using training patients only."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from .common import DEFAULT_CONFIG, dump_json, read_config
from .counterfactual_policies import treatment_duration
from .generate import _branch_job, _fact_job, run_jobs


def calibrate(cfg, out_config, report_path, count=500):
    cfg = deepcopy(cfg)
    # The primary gamma=1 dataset in the four-setting suite uses seed+2,000,000.
    pilot = deepcopy(cfg)
    pilot['seed'] += 2000000
    pilot['behavior']['gamma'] = 1.0
    records = run_jobs(_fact_job, ((pilot, 'train', i) for i in range(count)), cfg['workers'])
    quantiles = np.quantile(np.concatenate([r['observations'][:, 2] for r in records]), cfg['policies']['quantiles'])
    specs = {'thresholds': {'3': float(quantiles[0]), '4': float(quantiles[1])},
             'delay_years': cfg['policies']['delay_years']}
    results = run_jobs(_branch_job, ((pilot, r, specs) for r in records), cfg['workers'])
    horizon = max(cfg['simulation']['horizons'])
    ys, durations = [], []
    for pid, branches in results:
        ys.append([b['clean'][0, np.flatnonzero(b['grid'] == horizon)[0]] for b in branches])
        durations.append([treatment_duration(b['intervals'], b['fork'], b['fork'] + horizon) for b in branches])
    candidates = []
    for cost in cfg['policies']['cost_candidates']:
        best = np.argmax(np.array(ys) - cost * np.array(durations), axis=1)
        proportions = np.bincount(best, minlength=5) / count
        entropy = -np.sum(proportions[proportions > 0] * np.log(proportions[proportions > 0]))
        candidates.append({'cost': cost, 'entropy': float(entropy),
                           'counts': {str(p): int((best == p).sum()) for p in range(5)}})
    chosen = max(candidates, key=lambda r: (r['entropy'], -abs(r['cost'] - 2.0)))['cost']
    provenance = {'source': 'training patients only; primary gamma=1 population', 'seed': pilot['seed'],
                  'n': count, 'horizon': horizon, 'selected_cost': chosen,
                  'selection': 'maximum 3-year optimal-policy entropy; ties closest to 2',
                  'thresholds_from_training_pilot': specs['thresholds'], 'candidates': candidates,
                  'common_cost_frozen_for_all_gammas': True, 'validation_or_test_used': False}
    cfg['policies']['treatment_cost'] = chosen
    cfg['policies']['tune_cost'] = False
    cfg['policies']['cost_calibration'] = provenance
    Path(out_config).write_text(yaml.safe_dump(cfg, sort_keys=False))
    dump_json(report_path, provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--out-config', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--patients', type=int, default=500)
    args = parser.parse_args()
    print(calibrate(read_config(args.config), args.out_config, args.report, args.patients))


if __name__ == '__main__':
    main()
