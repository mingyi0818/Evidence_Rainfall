"""
Comprehensive evaluation module.
Computes classification metrics, calibration metrics, uncertainty metrics,
statistical tests, and confidence intervals.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss
)
from scipy import stats

from config import DEVICE, OUTPUT, RANDOM_SEEDS, ECE_N_BINS, STATS, CHECKPOINT_DIR
from data_loader import preprocess_and_split, WeatherDataset
from models import build_model, SklearnWrapper, EDLMLP, MCDropoutMLP, BayesianMLP
from torch.utils.data import DataLoader


# ------------------------------------------------------------------------------
# Metric Computations
# ------------------------------------------------------------------------------

def compute_ece(y_true, y_prob, n_bins=15):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []
    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if i == 0:
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_counts.append(0)
    return ece, bin_accs, bin_confs, bin_counts


def compute_nll_dirichlet(y_true, alpha, eps=1e-10):
    """Negative log-likelihood under Dirichlet predictive."""
    alpha = np.asarray(alpha, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    alpha0 = alpha.sum(axis=1, keepdims=True)
    probs = alpha / alpha0
    # NLL = -log p(y|alpha)
    # p(y=k|alpha) = alpha_k / alpha_0
    nll = -np.log(probs[np.arange(len(y_true)), y_true] + eps)
    return float(nll.mean())


def compute_sharpness(y_prob):
    """Average predictive variance (sharpness)."""
    return float(np.var(y_prob))


def compute_uncertainty_auroc(y_true, y_prob, uncertainty):
    """Use uncertainty to detect errors; return AUROC and AUPR."""
    errors = (y_prob.argmax(axis=1) != y_true).astype(int)
    if len(np.unique(errors)) < 2:
        return 0.5, 0.0
    try:
        auroc = roc_auc_score(errors, uncertainty)
        aupr = average_precision_score(errors, uncertainty)
    except Exception:
        auroc = 0.5
        aupr = 0.0
    return auroc, aupr


def compute_classification_metrics(y_true, y_pred, y_prob):
    """Compute standard classification metrics."""
    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['f1_micro'] = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
    if y_prob.shape[1] == 2:
        metrics['auc'] = float(roc_auc_score(y_true, y_prob[:, 1]))
        metrics['average_precision'] = float(average_precision_score(y_true, y_prob[:, 1]))
        metrics['brier_score'] = float(brier_score_loss(y_true, y_prob[:, 1]))
        ece, _, _, _ = compute_ece(y_true, y_prob[:, 1], n_bins=ECE_N_BINS)
        metrics['ece'] = float(ece)
    else:
        metrics['auc'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
        metrics['average_precision'] = 0.0
        metrics['brier_score'] = 0.0
        metrics['ece'] = 0.0
    return metrics


def compute_uncertainty_metrics(y_true, y_pred, y_prob, uncertainty_dict):
    """Compute uncertainty-specific metrics."""
    metrics = {}
    metrics['sharpness'] = compute_sharpness(y_prob[:, 1] if y_prob.shape[1]==2 else y_prob.max(axis=1))
    # Uncertainty AUROC (detecting errors)
    u = uncertainty_dict.get('H_total', np.zeros(len(y_true)))
    if isinstance(u, torch.Tensor):
        u = u.cpu().numpy()
    auroc, aupr = compute_uncertainty_auroc(y_true, y_prob, u)
    metrics['uncertainty_auroc'] = float(auroc)
    metrics['uncertainty_aupr'] = float(aupr)
    # NLL-Dirichlet if alpha available
    if 'alpha' in uncertainty_dict:
        alpha = uncertainty_dict['alpha']
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.cpu().numpy()
        metrics['nll_dirichlet'] = compute_nll_dirichlet(y_true, alpha)
    else:
        metrics['nll_dirichlet'] = float(log_loss(y_true, y_prob))
    # Decomposition stats
    for key in ['H_total', 'H_alea', 'H_epi', 'precision']:
        if key in uncertainty_dict:
            val = uncertainty_dict[key]
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            metrics[f'{key}_mean'] = float(np.mean(val))
            metrics[f'{key}_std'] = float(np.std(val))
    return metrics


# ------------------------------------------------------------------------------
# Model Inference Helpers
# ------------------------------------------------------------------------------

def predict_sklearn(model_wrapper, X):
    """Predict with sklearn wrapper."""
    probs = model_wrapper.predict_proba(X)
    preds = probs.argmax(axis=1)
    unc = model_wrapper.predict_uncertainty(X)
    return preds, probs, unc


def predict_torch_model(model, X, batch_size=512):
    """Predict with torch model."""
    model.eval()
    dataset = WeatherDataset(X, np.zeros(len(X)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_probs = []
    all_unc = {k: [] for k in ['H_total', 'H_alea', 'H_epi', 'precision', 'alpha']}

    with torch.no_grad():
        for Xb, _ in loader:
            Xb = Xb.to(DEVICE)
            if isinstance(model, EDLMLP):
                unc = model.predict_uncertainty(Xb)
                probs = unc['probs'].cpu().numpy()
                preds = probs.argmax(axis=1)
                for k in ['H_total', 'H_alea', 'H_epi', 'precision', 'alpha']:
                    if k in unc:
                        all_unc[k].append(unc[k].cpu().numpy())
            elif isinstance(model, (MCDropoutMLP, BayesianMLP)):
                unc = model.predict_with_uncertainty(Xb, n_samples=50)
                probs = unc['probs'].cpu().numpy()
                preds = probs.argmax(axis=1)
                for k in ['H_total', 'H_epi']:
                    if k in unc:
                        all_unc[k].append(unc[k].cpu().numpy())
            else:
                logits = model(Xb)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)
                # Fix F1: compute softmax-based uncertainty for plain classifiers
                # (predictive entropy, MSP, margin — standard UQ baselines per Hendrycks & Gimpel 2017)
                probs_t = torch.from_numpy(probs)
                H_total = -(probs_t * torch.log(probs_t + 1e-10)).sum(dim=1).numpy()
                all_unc['H_total'].append(H_total)
                # epistemic approx: 0 for single-pass softmax (no variance estimate)
                all_unc['H_epi'].append(np.zeros_like(H_total))
            all_preds.append(preds)
            all_probs.append(probs)

    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    unc_dict = {}
    for k, vlist in all_unc.items():
        if vlist:
            unc_dict[k] = np.concatenate(vlist)
    return preds, probs, unc_dict


# ------------------------------------------------------------------------------
# Statistical Testing
# ------------------------------------------------------------------------------

def paired_wilcoxon(scores_a, scores_b):
    """Paired Wilcoxon signed-rank test."""
    if len(scores_a) < 3 or np.allclose(scores_a, scores_b):
        return {'statistic': 0.0, 'pvalue': 1.0}
    try:
        stat, p = stats.wilcoxon(scores_a, scores_b, alternative='two-sided')
    except Exception:
        stat, p = 0.0, 1.0
    return {'statistic': float(stat), 'pvalue': float(p)}


def cohens_d(scores_a, scores_b):
    """Cohen's d effect size for paired differences."""
    diff = np.array(scores_a) - np.array(scores_b)
    if len(diff) < 2 or diff.std(ddof=1) == 0:
        return 0.0
    d = diff.mean() / diff.std(ddof=1)
    return float(d)


def bootstrap_ci(scores, confidence=0.95, n_bootstrap=1000):
    """Bootstrap confidence interval."""
    scores = np.array(scores)
    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=len(scores), replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    alpha = 1 - confidence
    lower = np.percentile(boot_means, alpha / 2 * 100)
    upper = np.percentile(boot_means, (1 - alpha / 2) * 100)
    return float(lower), float(upper)


def percentile_ci(scores, confidence=0.95):
    """Percentile-based CI assuming normal approximation."""
    scores = np.array(scores)
    mean = scores.mean()
    sem = stats.sem(scores)
    if sem == 0 or len(scores) < 2:
        return mean, mean
    h = sem * stats.t.ppf((1 + confidence) / 2., len(scores) - 1)
    return float(mean - h), float(mean + h)


# ------------------------------------------------------------------------------
# Main Evaluation Orchestrator
# ------------------------------------------------------------------------------

def evaluate_all_models(seed=42):
    """Evaluate all trained models for a given seed."""
    # Load data for this seed
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = preprocess_and_split(seed=seed, save=False)

    results = {}

    # Evaluate EDL-UQ
    edl_path = os.path.join(CHECKPOINT_DIR, f"edl_seed{seed}.pth")
    if os.path.exists(edl_path):
        print(f"[Eval] EDL-UQ (seed={seed})")
        model = build_model('edl', X_test.shape[1], num_classes=2,
                            hidden_dims=[128,64,32], dropout_rate=0.3).to(DEVICE)
        model.load_state_dict(torch.load(edl_path, map_location=DEVICE))
        preds, probs, unc = predict_torch_model(model, X_test)
        metrics = compute_classification_metrics(y_test, preds, probs)
        metrics.update(compute_uncertainty_metrics(y_test, preds, probs, unc))
        results['EDL-UQ'] = metrics
        print(f"  Accuracy={metrics['accuracy']:.4f} F1-macro={metrics['f1_macro']:.4f} AUC={metrics['auc']:.4f} ECE={metrics['ece']:.4f}")

    # Evaluate baselines
    baseline_names = ['LogisticRegression', 'RandomForest', 'XGBoost',
                      'LSTM', 'GRU', 'BNN', 'MCDropout']
    for bname in baseline_names:
        pkl_path = os.path.join(CHECKPOINT_DIR, f"{bname.lower()}_seed{seed}.pkl")
        pth_path = os.path.join(CHECKPOINT_DIR, f"{bname.lower()}_seed{seed}.pth")
        print(f"[Eval] {bname} (seed={seed})")
        try:
            if os.path.exists(pkl_path):
                wrapper = pickle.load(open(pkl_path, 'rb'))
                preds, probs, unc = predict_sklearn(wrapper, X_test)
            elif os.path.exists(pth_path):
                bname_lower = bname.lower()
                if bname_lower == 'lstm':
                    model = build_model('lstm', X_test.shape[1], num_classes=2,
                                        hidden_size=64, num_layers=2, dropout=0.3).to(DEVICE)
                elif bname_lower == 'gru':
                    model = build_model('gru', X_test.shape[1], num_classes=2,
                                        hidden_size=64, num_layers=2, dropout=0.3).to(DEVICE)
                elif bname_lower == 'bnn':
                    model = build_model('bnn', X_test.shape[1], num_classes=2,
                                        hidden_dims=[128,64], prior_sigma=1.0).to(DEVICE)
                elif bname_lower == 'mcdropout':
                    model = build_model('mcdropout', X_test.shape[1], num_classes=2,
                                        hidden_dims=[128,64,32], dropout_rate=0.3).to(DEVICE)
                else:
                    continue
                model.load_state_dict(torch.load(pth_path, map_location=DEVICE))
                preds, probs, unc = predict_torch_model(model, X_test)
            else:
                print(f"  [Skip] No checkpoint found for {bname}")
                continue
            metrics = compute_classification_metrics(y_test, preds, probs)
            metrics.update(compute_uncertainty_metrics(y_test, preds, probs, unc))
            results[bname] = metrics
            print(f"  Accuracy={metrics['accuracy']:.4f} F1-macro={metrics['f1_macro']:.4f} AUC={metrics['auc']:.4f} ECE={metrics['ece']:.4f}")
        except Exception as e:
            print(f"  [ERROR] {bname}: {e}")

    return results


def aggregate_results(all_seed_results):
    """Aggregate results across seeds: mean, std, CI."""
    aggregated = {}
    model_names = set()
    for sr in all_seed_results.values():
        model_names.update(sr.keys())

    for mname in sorted(model_names):
        metrics_dict = {}
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_macro', 'f1_micro',
                            'auc', 'average_precision', 'brier_score', 'ece',
                            'uncertainty_auroc', 'uncertainty_aupr', 'nll_dirichlet',
                            'sharpness']:
            values = [all_seed_results[s][mname][metric_name]
                      for s in all_seed_results if mname in all_seed_results[s]
                      and metric_name in all_seed_results[s][mname]]
            if not values:
                continue
            arr = np.array(values)
            lower, upper = percentile_ci(arr)
            metrics_dict[metric_name] = {
                'mean': float(arr.mean()),
                'std': float(arr.std(ddof=1)),
                'ci_lower': lower,
                'ci_upper': upper,
                'values': [float(v) for v in values],
            }
        aggregated[mname] = metrics_dict
    return aggregated


def statistical_tests(all_seed_results):
    """Run Wilcoxon and Cohen's d comparing EDL-UQ to each baseline."""
    tests = {}
    edl_values = {m: [all_seed_results[s]['EDL-UQ'][m]
                      for s in all_seed_results if 'EDL-UQ' in all_seed_results[s]]
                  for m in ['accuracy', 'f1_macro', 'auc', 'ece']}

    for bname in ['LogisticRegression', 'RandomForest', 'XGBoost',
                  'LSTM', 'GRU', 'BNN', 'MCDropout']:
        tests[bname] = {}
        for metric in ['accuracy', 'f1_macro', 'auc', 'ece']:
            bvals = [all_seed_results[s][bname][metric]
                     for s in all_seed_results if bname in all_seed_results[s]]
            if len(edl_values[metric]) == len(bvals) and len(bvals) > 0:
                tests[bname][metric] = {
                    'wilcoxon': paired_wilcoxon(edl_values[metric], bvals),
                    'cohens_d': cohens_d(edl_values[metric], bvals),
                }
    return tests


def run_full_evaluation():
    """Run evaluation across all seeds, aggregate, and save."""
    all_seed_results = {}
    for seed in RANDOM_SEEDS:
        print("\n" + "=" * 60)
        print(f"EVALUATION SEED = {seed}")
        print("=" * 60)
        all_seed_results[seed] = evaluate_all_models(seed=seed)

    aggregated = aggregate_results(all_seed_results)
    tests = statistical_tests(all_seed_results)

    final = {
        'per_seed': all_seed_results,
        'aggregated': aggregated,
        'statistical_tests': tests,
    }

    with open(OUTPUT['results_json'], 'w') as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\n[Done] Evaluation results saved to {OUTPUT['results_json']}")
    return final


if __name__ == "__main__":
    run_full_evaluation()
