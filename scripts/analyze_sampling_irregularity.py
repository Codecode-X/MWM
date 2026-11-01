"""Compare sampling-interval irregularity on the exact ADNI and MIMIC training timelines."""
import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADNI = ROOT / "paper_odeae/cache/local_available.pkl"
DEFAULT_MIMIC = Path("/data/MIMIC/derived/latentode_iv31/dataset.pkl")
QUANTILES = (0, .1, .25, .5, .75, .9, .95, 1)
STATIC_FIELDS = {
    "ADNI": {"sex", "age", "marital_status", "education_years", "handedness", "apoe4_count"},
    "MIMIC": {"sex"},
}


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {str(p): float(np.quantile(values, p)) for p in QUANTILES}


def gini(values):
    values = np.sort(np.asarray(values, dtype=np.float64))
    n = len(values)
    if not n or values.sum() == 0:
        return 0.0
    weights = 2 * np.arange(1, n + 1) - n - 1
    return float(weights @ values / (n * values.sum()))


def timeline(sequence, dataset):
    times = np.asarray(sequence["ages"], dtype=np.float64)
    if dataset == "ADNI":
        dates = [date.fromisoformat(value) for value in sequence["dates"]]
        physical = np.asarray(
            [(right - left).days / 365.25 for left, right in zip(dates[:-1], dates[1:])]
        )
        # All retained dates are right edges of calendar-quarter bins.
        grid_positions = np.asarray(
            [value.year * 4 + (value.month - 1) // 3 for value in dates], dtype=np.float64
        )
    else:
        dates = None
        physical = np.diff(times)
        # MIMIC ages are integer endpoints of hourly bins.
        grid_positions = times
    return times, dates, physical, grid_positions


def feature_intervals(indices, times, dates, dataset):
    if dataset == "MIMIC":
        return np.diff(times[indices])
    return np.asarray(
        [(dates[right] - dates[left]).days / 365.25 for left, right in zip(indices[:-1], indices[1:])]
    )


def summarize(dataset, path):
    payload = pickle.loads(path.read_bytes())
    sequences = payload["sequences"]
    schema = payload["schema"]
    pooled_intervals = []
    pooled_grid_gaps = []
    visits = []
    durations = []
    patient_cv = []
    patient_gini = []
    adjacent_change = []
    max_min_ratio = []
    equal_gaps = []
    consecutive = []
    any_skipped = []
    field_intervals = []
    field_pair_cv = []
    field_pair_gini = []
    field_pair_equal = []
    field_pairs_two = 0
    field_pairs_three = 0

    for sequence in sequences:
        times, dates, intervals, grid_positions = timeline(sequence, dataset)
        grid_gaps = np.diff(grid_positions)
        if not (len(intervals) == len(grid_gaps) and np.all(grid_gaps >= 1)):
            raise ValueError("invalid increasing base-grid timeline")
        pooled_intervals.extend(intervals)
        pooled_grid_gaps.extend(grid_gaps)
        visits.append(len(times))
        durations.append(times[-1] - times[0])

        # At least two gaps are needed to measure within-patient variability.
        if len(grid_gaps) >= 2:
            patient_cv.append(float(np.std(intervals) / np.mean(intervals)))
            patient_gini.append(gini(intervals))
            adjacent_change.append(
                float(np.mean(np.abs(np.diff(intervals)) / (intervals[:-1] + intervals[1:])))
            )
            max_min_ratio.append(float(np.max(intervals) / np.min(intervals)))
            equal_gaps.append(bool(np.all(grid_gaps == grid_gaps[0])))
            consecutive.append(bool(np.all(grid_gaps == 1)))
            any_skipped.append(bool(np.any(grid_gaps > 1)))

        mask = np.asarray(sequence["mask"], dtype=bool)
        for feature in range(mask.shape[1]):
            if schema[feature]["name"] in STATIC_FIELDS[dataset]:
                continue
            indices = np.flatnonzero(mask[:, feature])
            if len(indices) < 2:
                continue
            intervals_ = feature_intervals(indices, times, dates, dataset)
            field_intervals.extend(intervals_)
            field_pairs_two += 1
            if len(intervals_) < 2:
                continue
            field_pair_cv.append(float(np.std(intervals_) / np.mean(intervals_)))
            field_pair_gini.append(gini(intervals_))
            feature_grid_gaps = np.diff(grid_positions[indices])
            field_pair_equal.append(bool(np.all(feature_grid_gaps == feature_grid_gaps[0])))
            field_pairs_three += 1

    pooled_intervals = np.asarray(pooled_intervals)
    pooled_grid_gaps = np.asarray(pooled_grid_gaps)
    mode = Counter(pooled_grid_gaps).most_common(1)[0][0]
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "patients": len(sequences),
        "timestamps": int(sum(visits)),
        "intervals": len(pooled_intervals),
        "features": int(np.asarray(sequences[0]["mask"]).shape[1]),
        "dynamic_features_in_field_analysis": len(schema) - len(STATIC_FIELDS[dataset]),
        "time_unit": "years" if dataset == "ADNI" else "hours",
        "base_grid": "calendar quarter" if dataset == "ADNI" else "hour",
        "splits": dict(Counter(sequence["split"] for sequence in sequences)),
        "visits_per_patient_quantiles": quantiles(visits),
        "followup_duration_quantiles": quantiles(durations),
        "pooled_interval_mean": float(pooled_intervals.mean()),
        "pooled_interval_sd": float(pooled_intervals.std()),
        "pooled_interval_cv": float(pooled_intervals.std() / pooled_intervals.mean()),
        "pooled_interval_quantiles": quantiles(pooled_intervals),
        "base_grid_gap_quantiles": quantiles(pooled_grid_gaps),
        "global_modal_base_gap": float(mode),
        "fraction_intervals_modal_base_gap": float(np.mean(pooled_grid_gaps == mode)),
        "fraction_intervals_one_base_step": float(np.mean(pooled_grid_gaps == 1)),
        "fraction_intervals_gt_one_base_step": float(np.mean(pooled_grid_gaps > 1)),
        "patient_variability_eligible_n": len(patient_cv),
        "patient_cv_mean": float(np.mean(patient_cv)),
        "patient_cv_quantiles": quantiles(patient_cv),
        "patient_gini_quantiles": quantiles(patient_gini),
        "adjacent_gap_change_quantiles": quantiles(adjacent_change),
        "max_min_gap_ratio_quantiles": quantiles(max_min_ratio),
        "fraction_patients_exactly_equal_gaps": float(np.mean(equal_gaps)),
        "fraction_patients_all_consecutive_base_steps": float(np.mean(consecutive)),
        "fraction_patients_with_any_skipped_base_step": float(np.mean(any_skipped)),
        "feature_patient_pairs_ge2_obs": field_pairs_two,
        "feature_intervals": len(field_intervals),
        "feature_interval_quantiles": quantiles(field_intervals),
        "feature_pairs_ge3_obs": field_pairs_three,
        "feature_pair_cv_mean": float(np.mean(field_pair_cv)),
        "feature_pair_cv_quantiles": quantiles(field_pair_cv),
        "feature_pair_gini_quantiles": quantiles(field_pair_gini),
        "fraction_feature_pairs_exactly_equal_gaps": float(np.mean(field_pair_equal)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adni", type=Path, default=DEFAULT_ADNI)
    parser.add_argument("--mimic", type=Path, default=DEFAULT_MIMIC)
    args = parser.parse_args()
    result = {
        "definition": {
            "cv": "population SD / mean of intervals; unitless; zero means equal gaps",
            "gini": "within-series interval inequality; unitless; zero means equal gaps",
            "adjacent_gap_change": "mean |d[i+1]-d[i]|/(d[i+1]+d[i]); bounded [0,1]",
            "eligibility": "patient or patient-feature series needs at least three timestamps for variability metrics",
        },
        "ADNI": summarize("ADNI", args.adni),
        "MIMIC": summarize("MIMIC", args.mimic),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
