"""
Statistical analysis v2 for direction 17 (EDL-Rainfall) - addresses reviewer M8.

Uses ALREADY-TRAINED 5-seed aggregated results in results/fixed_results_temporal_all.json.
Implements:
- Proper seed-level paired Wilcoxon signed-rank tests (5 seeds) with unified interpretation
- Cohen's d_z effect size (with proper caveat about seed-level vs sample-level)
- 95% CIs over 5 seeds (t-distribution, properly labeled as seed variability)
- Holm-Bonferroni correction for multiple comparisons
- Honest reporting of direction (which method is better) regardless of significance

This avoids re-training all models for sample-level tests (which is infeasible given
compute constraints); we acknowledge this limitation in the paper.
"""
import os
import sys
import json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from config import RESULTS_DIR

LOG_PATH = os.path.join(os.path.dirname(__file__), 'statistical_analysis_v2.log')
LOG_FILE = open(LOG_PATH, 'w', encoding='utf-8')
def log(msg):
    print(msg, flush=True)
    LOG_FILE.write(msg + '\n'); LOG_FILE.flush()


def wilcoxon_signed_rank_paired(x, y, alternative='two-sided'):
    """Paired Wilcoxon signed-rank test for two metric vectors across seeds.
    Returns (W_statistic, p_value, n_pairs, n_x_better, n_y_better, mean_diff, std_diff, cohen_dz).

    Cohen's d_z = mean(diff) / std(diff) — quantifies *seed-level* stability.
    Note: NOT a sample-level effect size; we report it with a clear caveat.
    """
    diff = np.array(x) - np.array(y)
    n = len(diff)
    n_x_better = int((diff > 0).sum())
    n_y_better = int((diff < 0).sum())
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1)) if n > 1 else 0.0
    cohen_dz = mean_diff / std_diff if std_diff > 1e-12 else 0.0

    # If all diffs same sign (W=0), use exact binomial p-value
    if n_x_better == 0 or n_y_better == 0:
        # All favor one side: exact two-sided p = 2 * (1/2)^n
        p_value = float(2 * (0.5 ** n))
        W_statistic = 0.0
    else:
        try:
            W_statistic, p_value = stats.wilcoxon(x, y, alternative=alternative)
            W_statistic = float(W_statistic)
            p_value = float(p_value)
        except ValueError:
            W_statistic, p_value = 0.0, 1.0

    return {
        'W_statistic': W_statistic,
        'p_value': p_value,
        'n_pairs': n,
        'n_x_better': n_x_better,
        'n_y_better': n_y_better,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'cohen_dz': cohen_dz,
    }


def seed_ci(values, ci=0.95):
    """95% CI over seeds using t-distribution (seed variability, NOT sample-level)."""
    n = len(values)
    if n < 2:
        return float(values[0]), float(values[0])
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    alpha = (1 - ci) / 2
    t_crit = float(stats.t.ppf(1 - alpha, df=n - 1))
    se = std / np.sqrt(n)
    return mean - t_crit * se, mean + t_crit * se


def holm_bonferroni(p_values, alpha=0.05):
    m = len(p_values)
    indices = np.argsort(p_values)
    results = [None] * m
    for rank, idx in enumerate(indices):
        adjusted_p = min(p_values[idx] * (m - rank), 1.0)
        reject = adjusted_p < alpha
        results[idx] = {
            'index': int(idx),
            'raw_p': float(p_values[idx]),
            'adjusted_p': float(adjusted_p),
            'reject_null': bool(reject),
        }
    return results


def run_seed_level_analysis():
    """Run paired Wilcoxon tests for all methods × all metrics across 5 seeds."""
    input_file = os.path.join(RESULTS_DIR, 'fixed_results_temporal_all.json')
    with open(input_file, 'r') as f:
        data = json.load(f)

    per_seed = data['per_seed']
    seeds = sorted([int(s) for s in per_seed.keys()])
    log(f"Seeds found: {seeds}")
    log(f"Number of seeds: {len(seeds)}")

    # Get list of methods (excluding Climatology which has no variability)
    methods_all = set()
    for s in seeds:
        methods_all.update(per_seed[str(s)].keys())
    methods = sorted([m for m in methods_all if m != 'Climatology'])
    log(f"Methods: {methods}")

    metrics = ['accuracy', 'f1_macro', 'auc', 'ece', 'brier', 'uncertainty_auroc']

    # Build per-method per-metric arrays across seeds
    data_arrays = {}
    for m in methods:
        data_arrays[m] = {}
        for metric in metrics:
            vals = []
            for s in seeds:
                if m in per_seed[str(s)] and metric in per_seed[str(s)][m]:
                    vals.append(per_seed[str(s)][m][metric])
                else:
                    vals.append(np.nan)
            data_arrays[m][metric] = np.array(vals)

    # Print aggregated table
    log("\n" + "=" * 80)
    log("AGGREGATED RESULTS (mean ± std over 5 seeds, temporal split)")
    log("=" * 80)
    header = f"{'Method':<12}" + "".join([f"{m:<22}" for m in metrics])
    log(header)
    for method in methods:
        row = f"{method:<12}"
        for metric in metrics:
            vals = data_arrays[method][metric]
            mean = float(np.nanmean(vals))
            std = float(np.nanstd(vals, ddof=1))
            row += f"{mean:.4f}±{std:.4f}      "
        log(row)

    # Seed-level paired Wilcoxon tests: EDL-Fixed vs each other method
    edl_method = 'EDL-Fixed'
    log("\n" + "=" * 80)
    log(f"PAIRED WILCOXON SIGNED-RANK TESTS ({edl_method} vs each baseline, 5 seeds)")
    log("=" * 80)
    log("Unified interpretation rule: W=0 + p=0.0625 means 5/5 seeds favor one side;")
    log("  with n=5 we DO NOT claim statistical significance (only directional consistency).")
    log("  Cohen's d_z is seed-level stability, NOT sample-level effect size.")
    log("")

    all_results = {}
    all_p_values = []
    test_names = []
    for method in methods:
        if method == edl_method:
            continue
        all_results[method] = {}
        log(f"\n--- {edl_method} vs {method} ---")
        for metric in metrics:
            x = data_arrays[edl_method][metric]
            y = data_arrays[method][metric]
            # For ECE and Brier, lower is better; reverse direction interpretation
            lower_better = metric in ['ece', 'brier']
            r = wilcoxon_signed_rank_paired(x, y, alternative='two-sided')
            # Direction: who is better?
            if lower_better:
                edl_better = r['mean_diff'] < 0
            else:
                edl_better = r['mean_diff'] > 0
            r['edl_better'] = bool(edl_better)
            r['metric'] = metric
            r['lower_is_better'] = lower_better
            all_results[method][metric] = r
            all_p_values.append(r['p_value'])
            test_names.append(f"{edl_method}_vs_{method}_{metric}")

            winner = edl_method if edl_better else method
            log(f"  {metric:<22}: diff={r['mean_diff']:+.6f} ± {r['std_diff']:.6f}  "
                f"W={r['W_statistic']:.1f}  p={r['p_value']:.4f}  "
                f"seeds_favor_EDL={r['n_x_better']}/5  "
                f"seeds_favor_{method}={r['n_y_better']}/5  "
                f"d_z={r['cohen_dz']:+.3f}  winner={winner}")

    # Holm-Bonferroni correction
    log("\n" + "=" * 80)
    log("HOLM-BONFERRONI CORRECTION (all paired comparisons)")
    log("=" * 80)
    corrected = holm_bonferroni(all_p_values, alpha=0.05)
    for i, c in enumerate(corrected):
        log(f"  Test '{test_names[i]}': raw_p={c['raw_p']:.4e}  "
            f"adjusted_p={c['adjusted_p']:.4e}  reject_H0={c['reject_null']}")

    # 95% CIs over seeds
    log("\n" + "=" * 80)
    log("95% CIs OVER 5 SEEDS (t-distribution; seed variability only, NOT sample-level)")
    log("=" * 80)
    for method in methods:
        log(f"\n{method}:")
        for metric in metrics:
            vals = data_arrays[method][metric]
            lo, hi = seed_ci(vals.tolist())
            mean = float(np.nanmean(vals))
            log(f"  {metric:<22}: mean={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")

    # Save results
    out = {
        'seeds': seeds,
        'methods': methods,
        'metrics': metrics,
        'edl_method': edl_method,
        'aggregated_results': {
            method: {
                metric: {
                    'mean': float(np.nanmean(data_arrays[method][metric])),
                    'std': float(np.nanstd(data_arrays[method][metric], ddof=1)),
                    'values': data_arrays[method][metric].tolist(),
                }
                for metric in metrics
            }
            for method in methods
        },
        'paired_tests': {
            method: {
                metric: all_results[method][metric]
                for metric in metrics
            }
            for method in all_results
        },
        'holm_bonferroni': [
            {
                'test_name': test_names[i],
                **corrected[i],
            }
            for i in range(len(corrected))
        ],
    }

    out_file = os.path.join(RESULTS_DIR, 'statistical_tests_seed_level.json')
    with open(out_file, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    log(f"\nStatistical analysis saved to {out_file}")

    LOG_FILE.close()
    return out


if __name__ == "__main__":
    run_seed_level_analysis()
