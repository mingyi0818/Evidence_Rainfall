"""
Aggregate 5-seed experiment results from fixed_results_temporal_all.json.
Produces:
  - results/main_results_v2.csv  (per-method mean/std/95% CI over 5 seeds)
  - results/main_results_v2.json (full per-seed + aggregated data)

All numbers are computed from REAL experimental outputs (no fabrication).
Temporal split (S1): train 2007-2014 / val 2015 / test 2016-2017.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
INPUT_FILE = os.path.join(RESULTS_DIR, 'fixed_results_temporal_all.json')

# Metrics to aggregate
METRICS = ['accuracy', 'precision', 'recall', 'f1_macro', 'auc',
           'brier', 'ece', 'uncertainty_auroc']

# Methods in display order
METHODS = ['Climatology', 'LR', 'RF', 'XGB',
           'LSTM', 'GRU', 'MCDropout', 'BNN',
           'EDL-C1', 'EDL-Fixed']


def aggregate():
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    per_seed = data['per_seed']
    seeds = sorted([int(s) for s in per_seed.keys()])
    print(f"Loaded {len(seeds)} seeds: {seeds}")
    print(f"Methods found: {list(per_seed[str(seeds[0])].keys())}")

    # Build per-method metric arrays
    agg = {}
    for method in METHODS:
        if method not in per_seed[str(seeds[0])]:
            print(f"  [WARN] method {method} not in results, skipping")
            continue
        method_data = {'per_seed': {}, 'stats': {}}
        for metric in METRICS:
            values = []
            for s in seeds:
                v = per_seed[str(s)].get(method, {}).get(metric, None)
                if v is not None:
                    values.append(float(v))
            if not values:
                continue
            arr = np.array(values)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            # 95% CI (t-distribution, seed variability only - NOT sample-level CI)
            if len(arr) > 1:
                t_crit = float(stats.t.ppf(0.975, df=len(arr) - 1))
                ci_half = t_crit * std / np.sqrt(len(arr))
                ci_lower = mean - ci_half
                ci_upper = mean + ci_half
            else:
                ci_lower = ci_upper = mean
            method_data['per_seed'][metric] = values
            method_data['stats'][metric] = {
                'mean': mean,
                'std': std,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_seeds': len(arr),
            }
        agg[method] = method_data

    # Save JSON
    out_json = os.path.join(RESULTS_DIR, 'main_results_v2.json')
    with open(out_json, 'w') as f:
        json.dump({'seeds': seeds, 'split': 'temporal_S1', 'aggregated': agg,
                   'note': 'All numbers from real 5-seed experiments on temporal split (S1: 2007-2014 train / 2015 val / 2016-2017 test). CI is t-distribution over 5 seeds (seed variability only).'},
                  f, indent=2)
    print(f"Saved aggregated JSON: {out_json}")

    # Build CSV table: one row per method, columns: mean/std/ci_lower/ci_upper per metric
    csv_rows = []
    for method in METHODS:
        if method not in agg:
            continue
        row = {'method': method}
        for metric in METRICS:
            s = agg[method]['stats'].get(metric, {})
            row[f'{metric}_mean'] = s.get('mean', None)
            row[f'{metric}_std'] = s.get('std', None)
            row[f'{metric}_ci_lower'] = s.get('ci_lower', None)
            row[f'{metric}_ci_upper'] = s.get('ci_upper', None)
        csv_rows.append(row)
    df = pd.DataFrame(csv_rows)
    out_csv = os.path.join(RESULTS_DIR, 'main_results_v2.csv')
    df.to_csv(out_csv, index=False)
    print(f"Saved aggregated CSV: {out_csv}")

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Method':<14}{'Accuracy':>14}{'F1-Macro':>14}{'AUC':>14}{'ECE':>14}{'Brier':>14}{'Unc-AUROC':>14}")
    print("-" * 100)
    for method in METHODS:
        if method not in agg:
            continue
        s = agg[method]['stats']
        acc = s.get('accuracy', {}).get('mean', float('nan'))
        acc_std = s.get('accuracy', {}).get('std', 0)
        f1 = s.get('f1_macro', {}).get('mean', float('nan'))
        f1_std = s.get('f1_macro', {}).get('std', 0)
        auc = s.get('auc', {}).get('mean', float('nan'))
        auc_std = s.get('auc', {}).get('std', 0)
        ece = s.get('ece', {}).get('mean', float('nan'))
        ece_std = s.get('ece', {}).get('std', 0)
        br = s.get('brier', {}).get('mean', float('nan'))
        br_std = s.get('brier', {}).get('std', 0)
        uar = s.get('uncertainty_auroc', {}).get('mean', float('nan'))
        uar_std = s.get('uncertainty_auroc', {}).get('std', 0)
        print(f"{method:<14}{acc:>10.4f}±{acc_std:.3f}{f1:>10.4f}±{f1_std:.3f}{auc:>10.4f}±{auc_std:.3f}{ece:>10.4f}±{ece_std:.3f}{br:>10.4f}±{br_std:.3f}{uar:>10.4f}±{uar_std:.3f}")

    # Also compute climatology baseline accuracy
    # Test rain rate (already printed in data_loader): 0.2289
    # Climatology = always predict "no rain" => accuracy = 0.7711
    # Already in data
    print("\nNotes:")
    print(f"  - Temporal split S1: train 2007-2014 / val 2015 / test 2016-2017")
    print(f"  - Train: 98988 samples, Val: 17231, Test: 25974 (test rain rate = 0.2289)")
    print(f"  - Climatology baseline: always predict 'no rain', accuracy = 0.7711")
    print(f"  - EDL-Fixed uses masked KL (Sensoy correct implementation)")
    print(f"  - EDL-C1 uses climatology-anchored prior at inference")
    print(f"  - All baselines use class_weight=False (uniform) - fixes F4")

    return agg


if __name__ == '__main__':
    aggregate()
