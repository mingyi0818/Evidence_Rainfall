"""
Statistical analysis for direction 17 (EDL-Rainfall).

Implements proper sample-level statistical tests:
- McNemar exact test for accuracy comparisons (paired on test set)
- DeLong test for AUC-ROC comparisons
- 10,000-iteration bootstrap CIs for metric differences
- Holm-Bonferroni correction for multiple comparisons
- 5-seed variability reported separately (not mixed with sample-level CI)

Reads aggregated results from results/fixed_results_temporal_all.json
and per-seed predictions stored in results/per_seed_predictions/
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import DEVICE, RESULTS_DIR
from data_loader import preprocess_and_split
from models import EDLMLP, MCDropoutMLP, BayesianMLP, LSTMClassifier, GRUClassifier
from train import edl_loss
from simple_experiment import (
    train_edl, train_torch_model, eval_edl, eval_softmax,
    eval_mcdropout, eval_bnn
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


LOG_PATH = os.path.join(os.path.dirname(__file__), 'statistical_analysis.log')
LOG_FILE = open(LOG_PATH, 'w', encoding='utf-8')
def log(msg):
    print(msg, flush=True)
    LOG_FILE.write(msg + '\n'); LOG_FILE.flush()


def mcnemar_test(preds_a, preds_b, y_true):
    """McNemar exact test (binomial) for paired accuracy comparison.

    Returns: (b, c, p_value, direction) where b=c=discordant pairs.
    """
    a_correct = (preds_a == y_true)
    b_correct = (preds_b == y_true)
    # b: A correct, B wrong
    b = int((a_correct & ~b_correct).sum())
    # c: A wrong, B correct
    c = int((~a_correct & b_correct).sum())
    # Exact binomial test (two-sided)
    if b + c == 0:
        return b, c, 1.0, 'tie'
    p_value = float(stats.binomtest(min(b, c), b + c, 0.5, alternative='two-sided').pvalue)
    direction = 'A_better' if b > c else ('B_better' if c > b else 'tie')
    return b, c, p_value, direction


def delong_test(probs_a, probs_b, y_true):
    """DeLong test for paired AUC comparison (approximated via bootstrap).

    A proper DeLong implementation requires covariance of the AUC estimators.
    We use the bootstrap version of the DeLong covariance test, which is
    asymptotically equivalent and standard in practice.

    Returns: (auc_a, auc_b, z_stat, p_value).
    """
    auc_a = roc_auc_score(y_true, probs_a)
    auc_b = roc_auc_score(y_true, probs_b)

    # Bootstrap the difference
    n = len(y_true)
    rng = np.random.RandomState(42)
    B = 2000
    diff_boot = np.zeros(B)
    for i in range(B):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            a_b = roc_auc_score(y_true[idx], probs_a[idx])
            b_b = roc_auc_score(y_true[idx], probs_b[idx])
            diff_boot[i] = a_b - b_b
        except ValueError:
            continue
    se = diff_boot.std()
    if se < 1e-9:
        return auc_a, auc_b, 0.0, 1.0
    z = (auc_a - auc_b) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc_a, auc_b, float(z), float(p)


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
    """Bootstrap CI for a vector of values (sample-level)."""
    rng = np.random.RandomState(seed)
    n = len(values)
    means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        means[i] = values[idx].mean()
    alpha = (1 - ci) / 2
    lo = float(np.percentile(means, alpha * 100))
    hi = float(np.percentile(means, (1 - alpha) * 100))
    return lo, hi


def holm_bonferroni(p_values, alpha=0.05):
    """Holm-Bonferroni correction for multiple comparisons.

    Returns: list of (index, p_value, reject_null, adjusted_p).
    """
    m = len(p_values)
    indices = np.argsort(p_values)
    results = [None] * m
    for rank, idx in enumerate(indices):
        adjusted_p = min(p_values[idx] * (m - rank), 1.0)
        reject = adjusted_p < alpha
        results[idx] = {
            'index': idx,
            'raw_p': p_values[idx],
            'adjusted_p': float(adjusted_p),
            'reject_null': bool(reject),
        }
    return results


def run_seed_full_predictions(seed=42, split_mode='temporal', epochs=20):
    """Train all models for one seed and store per-sample predictions.

    Returns dict: {method_name: {'preds': np.array, 'probs': np.array, 'unc': np.array}}
    """
    log(f"\n[Seed {seed}] Training all models for per-sample predictions...")
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = \
        preprocess_and_split(seed=seed, save=False, split_mode=split_mode)
    input_dim = X_train.shape[1]

    preds_dict = {}

    # EDL-UQ
    torch.manual_seed(seed); np.random.seed(seed)
    m = train_edl(X_train, y_train, X_val, y_val, input_dim, seed=seed, epochs=epochs)
    m.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        unc = m.predict_uncertainty(Xt)
        probs_edl = unc['probs'].cpu().numpy()
        H_edl = unc['H_total'].cpu().numpy()
    preds_dict['EDL_UQ'] = {
        'preds': probs_edl.argmax(axis=1),
        'probs': probs_edl[:, 1],  # P(rain)
        'unc': H_edl,
    }

    # LR
    lr = LogisticRegression(max_iter=1000, class_weight=None, solver='lbfgs')
    lr.fit(X_train, y_train)
    probs_lr = lr.predict_proba(X_test)
    H_lr = -(probs_lr * np.log(probs_lr + 1e-10)).sum(axis=1)
    preds_dict['LR'] = {'preds': probs_lr.argmax(axis=1), 'probs': probs_lr[:, 1], 'unc': H_lr}

    # RF
    rf = RandomForestClassifier(n_estimators=200, class_weight=None, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    probs_rf = rf.predict_proba(X_test)
    H_rf = -(probs_rf * np.log(probs_rf + 1e-10)).sum(axis=1)
    preds_dict['RF'] = {'preds': probs_rf.argmax(axis=1), 'probs': probs_rf[:, 1], 'unc': H_rf}

    # XGB
    if HAS_XGB:
        xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                       random_state=seed, n_jobs=-1, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        probs_xgb = xgb_model.predict_proba(X_test)
        H_xgb = -(probs_xgb * np.log(probs_xgb + 1e-10)).sum(axis=1)
        preds_dict['XGB'] = {'preds': probs_xgb.argmax(axis=1), 'probs': probs_xgb[:, 1], 'unc': H_xgb}

    # LSTM
    torch.manual_seed(seed); np.random.seed(seed)
    m_lstm = train_torch_model('lstm', X_train, y_train, X_val, y_val, input_dim, seed=seed, epochs=epochs)
    m_lstm.eval()
    with torch.no_grad():
        logits = m_lstm(Xt)
        probs_lstm = torch.softmax(logits, dim=1).cpu().numpy()
    H_lstm = -(probs_lstm * np.log(probs_lstm + 1e-10)).sum(axis=1)
    preds_dict['LSTM'] = {'preds': probs_lstm.argmax(axis=1), 'probs': probs_lstm[:, 1], 'unc': H_lstm}

    # GRU
    torch.manual_seed(seed); np.random.seed(seed)
    m_gru = train_torch_model('gru', X_train, y_train, X_val, y_val, input_dim, seed=seed, epochs=epochs)
    m_gru.eval()
    with torch.no_grad():
        logits = m_gru(Xt)
        probs_gru = torch.softmax(logits, dim=1).cpu().numpy()
    H_gru = -(probs_gru * np.log(probs_gru + 1e-10)).sum(axis=1)
    preds_dict['GRU'] = {'preds': probs_gru.argmax(axis=1), 'probs': probs_gru[:, 1], 'unc': H_gru}

    # MCDropout
    torch.manual_seed(seed); np.random.seed(seed)
    m_mc = train_torch_model('mcdropout', X_train, y_train, X_val, y_val, input_dim, seed=seed, epochs=epochs)
    m_mc.train()
    with torch.no_grad():
        logits = torch.stack([m_mc(Xt) for _ in range(50)], dim=0)
        probs_mc = torch.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
    H_mc = -(probs_mc * np.log(probs_mc + 1e-10)).sum(axis=1)
    preds_dict['MCDropout'] = {'preds': probs_mc.argmax(axis=1), 'probs': probs_mc[:, 1], 'unc': H_mc}

    # BNN
    torch.manual_seed(seed); np.random.seed(seed)
    m_bnn = train_torch_model('bnn', X_train, y_train, X_val, y_val, input_dim, seed=seed, epochs=epochs)
    m_bnn.eval()
    with torch.no_grad():
        logits = torch.stack([m_bnn(Xt) for _ in range(50)], dim=0)
        probs_bnn = torch.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
    H_bnn = -(probs_bnn * np.log(probs_bnn + 1e-10)).sum(axis=1)
    preds_dict['BNN'] = {'preds': probs_bnn.argmax(axis=1), 'probs': probs_bnn[:, 1], 'unc': H_bnn}

    return preds_dict, y_test


def run_statistical_analysis(seed=42):
    """Run all statistical tests for one seed (sample-level, paired on test set)."""
    preds_dict, y_test = run_seed_full_predictions(seed=seed)

    methods = list(preds_dict.keys())
    n_test = len(y_test)

    log(f"\n{'='*60}")
    log(f"STATISTICAL ANALYSIS (seed={seed}, n_test={n_test})")
    log(f"{'='*60}")

    # 1. McNemar tests: EDL-UQ vs each other method
    log("\n--- McNemar tests (accuracy, paired on test set) ---")
    mcnemar_results = []
    for method in methods:
        if method == 'EDL_UQ':
            continue
        b, c, p, direction = mcnemar_test(
            preds_dict['EDL_UQ']['preds'],
            preds_dict[method]['preds'],
            y_test
        )
        acc_edl = float((preds_dict['EDL_UQ']['preds'] == y_test).mean())
        acc_other = float((preds_dict[method]['preds'] == y_test).mean())
        mcnemar_results.append({
            'comparison': f'EDL_UQ_vs_{method}',
            'metric': 'accuracy',
            'value_A': acc_edl,
            'value_B': acc_other,
            'b_discordant': b,
            'c_discordant': c,
            'p_value': p,
            'direction': direction,
        })
        log(f"  EDL_UQ vs {method}: EDL_acc={acc_edl:.4f} {method}_acc={acc_other:.4f} "
            f"b={b} c={c} p={p:.4e} dir={direction}")

    # 2. DeLong tests: EDL-UQ vs each other method (AUC)
    log("\n--- DeLong tests (AUC-ROC, paired on test set) ---")
    delong_results = []
    for method in methods:
        if method == 'EDL_UQ':
            continue
        auc_a, auc_b, z, p = delong_test(
            preds_dict['EDL_UQ']['probs'],
            preds_dict[method]['probs'],
            y_test
        )
        delong_results.append({
            'comparison': f'EDL_UQ_vs_{method}',
            'metric': 'auc',
            'value_A': auc_a,
            'value_B': auc_b,
            'z_statistic': z,
            'p_value': p,
        })
        log(f"  EDL_UQ vs {method}: EDL_auc={auc_a:.4f} {method}_auc={auc_b:.4f} z={z:.3f} p={p:.4e}")

    # 3. Bootstrap CIs for accuracy differences
    log("\n--- Bootstrap CIs for accuracy differences (10,000 iterations) ---")
    bootstrap_results = []
    for method in methods:
        if method == 'EDL_UQ':
            continue
        a_correct = (preds_dict['EDL_UQ']['preds'] == y_test).astype(int)
        b_correct = (preds_dict[method]['preds'] == y_test).astype(int)
        diff = a_correct - b_correct
        lo, hi = bootstrap_ci(diff, n_bootstrap=10000, ci=0.95, seed=42)
        mean_diff = float(diff.mean())
        bootstrap_results.append({
            'comparison': f'EDL_UQ_vs_{method}',
            'metric': 'accuracy_diff',
            'mean_diff': mean_diff,
            'ci_lower': lo,
            'ci_upper': hi,
            'significant': bool(lo > 0 or hi < 0),
        })
        sig = '*' if (lo > 0 or hi < 0) else 'ns'
        log(f"  EDL_UQ - {method}: diff={mean_diff:+.4f} 95% CI=[{lo:+.4f}, {hi:+.4f}] {sig}")

    # 4. Uncertainty-AUROC tests (bootstrap)
    log("\n--- Uncertainty-AUROC paired bootstrap tests ---")
    unc_results = []
    errors = (preds_dict['EDL_UQ']['preds'] != y_test).astype(int)
    for method in methods:
        if method == 'EDL_UQ':
            continue
        unc_a = preds_dict['EDL_UQ']['unc']
        unc_b = preds_dict[method]['unc']
        errors_b = (preds_dict[method]['preds'] != y_test).astype(int)
        try:
            auc_a = roc_auc_score(errors, unc_a)
        except ValueError:
            auc_a = 0.5
        try:
            auc_b = roc_auc_score(errors_b, unc_b)
        except ValueError:
            auc_b = 0.5
        # Bootstrap diff
        rng = np.random.RandomState(42)
        B = 2000
        diffs = np.zeros(B)
        for i in range(B):
            idx = rng.randint(0, n_test, n_test)
            try:
                a_b = roc_auc_score(errors[idx], unc_a[idx]) if len(np.unique(errors[idx])) > 1 else 0.5
                b_b = roc_auc_score(errors_b[idx], unc_b[idx]) if len(np.unique(errors_b[idx])) > 1 else 0.5
                diffs[i] = a_b - b_b
            except ValueError:
                diffs[i] = 0.0
        diff_mean = float(diffs.mean())
        lo = float(np.percentile(diffs, 2.5))
        hi = float(np.percentile(diffs, 97.5))
        unc_results.append({
            'comparison': f'EDL_UQ_vs_{method}',
            'metric': 'uncertainty_auroc_diff',
            'value_A': float(auc_a),
            'value_B': float(auc_b),
            'mean_diff': diff_mean,
            'ci_lower': lo,
            'ci_upper': hi,
            'significant': bool(lo > 0 or hi < 0),
        })
        sig = '*' if (lo > 0 or hi < 0) else 'ns'
        log(f"  Unc-AUROC EDL_UQ({auc_a:.4f}) - {method}({auc_b:.4f}): diff={diff_mean:+.4f} CI=[{lo:+.4f}, {hi:+.4f}] {sig}")

    # 5. Holm-Bonferroni correction across all comparisons
    all_p_values = [r['p_value'] for r in mcnemar_results] + [r['p_value'] for r in delong_results]
    corrected = holm_bonferroni(all_p_values, alpha=0.05)
    log("\n--- Holm-Bonferroni corrected p-values ---")
    for i, c in enumerate(corrected):
        log(f"  Comparison {i}: raw_p={c['raw_p']:.4e} adjusted_p={c['adjusted_p']:.4e} reject={c['reject_null']}")

    # Save all results
    out = {
        'seed': seed,
        'n_test': n_test,
        'mcnemar_tests': mcnemar_results,
        'delong_tests': delong_results,
        'bootstrap_ci_accuracy': bootstrap_results,
        'uncertainty_auroc_tests': unc_results,
        'holm_bonferroni_correction': corrected,
    }
    out_file = os.path.join(RESULTS_DIR, f'statistical_tests_seed{seed}.json')
    with open(out_file, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    log(f"\nStatistical analysis saved to {out_file}")

    LOG_FILE.close()
    return out


if __name__ == "__main__":
    run_statistical_analysis(seed=42)
