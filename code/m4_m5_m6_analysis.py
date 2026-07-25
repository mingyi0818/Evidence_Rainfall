"""
M4/M5/M6: Meteorological skill scores, cost-loss analysis, and selective
prediction comparison across all baselines (5 seeds).

M4: POD/FAR/CSI/HSS/ETS/BSS/Murphy decomposition + reliability diagrams
M5: Cost-loss curves with varying cost/loss ratios
M6: AURC/E-AURC and selective prediction comparison across all baselines

All results are computed from existing checkpoints (no retraining needed).
Author: GLM-5.2
Date: 2026-07-25
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DEVICE, RANDOM_SEEDS, CHECKPOINT_DIR, RESULTS_DIR, PLOTS_DIR,
    ECE_N_BINS, PREPROCESS
)
# Disable rare category grouping to match checkpoint dimensions (123 features)
PREPROCESS['min_category_frequency'] = 0
from data_loader import preprocess_and_split
from models import build_model, EDLMLP, MCDropoutMLP, BayesianMLP, SklearnWrapper
from evaluate import compute_ece

SEED = 42  # primary seed for single-seed analyses (M4/M5)
SEEDS = RANDOM_SEEDS  # all 5 seeds for M6 multi-seed analysis


# ============================================================================
# Helper: load all models for a given seed
# ============================================================================

def load_all_models(seed, input_dim):
    """Load all trained models for a given seed."""
    models = {}

    # EDL-Fixed
    edl_path = os.path.join(CHECKPOINT_DIR, f"edl_seed{seed}.pth")
    if os.path.exists(edl_path):
        m = build_model('edl', input_dim, num_classes=2,
                        hidden_dims=[128, 64, 32], dropout_rate=0.3).to(DEVICE)
        m.load_state_dict(torch.load(edl_path, map_location=DEVICE))
        m.eval()
        models['EDL-Fixed'] = m

    # LSTM
    lstm_path = os.path.join(CHECKPOINT_DIR, f"lstm_seed{seed}.pth")
    if os.path.exists(lstm_path):
        m = build_model('lstm', input_dim, num_classes=2,
                        hidden_size=64, num_layers=2, dropout=0.3).to(DEVICE)
        m.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
        m.eval()
        models['LSTM'] = m

    # GRU
    gru_path = os.path.join(CHECKPOINT_DIR, f"gru_seed{seed}.pth")
    if os.path.exists(gru_path):
        m = build_model('gru', input_dim, num_classes=2,
                        hidden_size=64, num_layers=2, dropout=0.3).to(DEVICE)
        m.load_state_dict(torch.load(gru_path, map_location=DEVICE))
        m.eval()
        models['GRU'] = m

    # MCDropout
    mcd_path = os.path.join(CHECKPOINT_DIR, f"mcdropout_seed{seed}.pth")
    if os.path.exists(mcd_path):
        m = build_model('mcdropout', input_dim, num_classes=2,
                        hidden_dims=[128, 64, 32], dropout_rate=0.3).to(DEVICE)
        m.load_state_dict(torch.load(mcd_path, map_location=DEVICE))
        models['MCDropout'] = m

    # BNN
    bnn_path = os.path.join(CHECKPOINT_DIR, f"bnn_seed{seed}.pth")
    if os.path.exists(bnn_path):
        m = build_model('bnn', input_dim, num_classes=2,
                        hidden_dims=[128, 64], prior_sigma=1.0).to(DEVICE)
        m.load_state_dict(torch.load(bnn_path, map_location=DEVICE))
        m.eval()
        models['BNN'] = m

    # Sklearn models
    for name_lower, name_display in [('logisticregression', 'LR'),
                                      ('randomforest', 'RF'),
                                      ('xgboost', 'XGB')]:
        pkl_path = os.path.join(CHECKPOINT_DIR, f"{name_lower}_seed{seed}.pkl")
        if os.path.exists(pkl_path):
            wrapper = pickle.load(open(pkl_path, 'rb'))
            models[name_display] = wrapper

    return models


def get_predictions(model, X_test, model_name):
    """Get predictions and uncertainties from a model."""
    Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        if isinstance(model, EDLMLP):
            unc = model.predict_uncertainty(Xt)
            probs = unc['probs'].cpu().numpy()
            H_T = unc['H_total'].cpu().numpy()
        elif isinstance(model, (MCDropoutMLP, BayesianMLP)):
            # For MC models, use multiple forward passes
            if isinstance(model, MCDropoutMLP):
                model.train()  # keep dropout active
            else:
                model.eval()
            n_samples = 50
            logits_list = []
            for _ in range(n_samples):
                logits_list.append(model(Xt))
            logits_stack = torch.stack(logits_list, dim=0)
            probs_stack = F.softmax(logits_stack, dim=-1)
            probs = probs_stack.mean(dim=0).cpu().numpy()
            H_T = -(probs * np.log(probs + 1e-10)).sum(axis=1)
        elif isinstance(model, SklearnWrapper):
            probs = model.predict_proba(X_test)
            H_T = -(probs * np.log(probs + 1e-10)).sum(axis=1)
        else:
            # LSTM/GRU - plain softmax
            logits = model(Xt)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            H_T = -(probs * np.log(probs + 1e-10)).sum(axis=1)

    preds = probs.argmax(axis=1)
    return preds, probs, H_T


# ============================================================================
# M4: Meteorological Skill Scores
# ============================================================================

def compute_skill_scores(y_true, y_pred, y_prob_rain):
    """
    Compute meteorological skill scores for binary rainfall prediction.
    All scores use rain (class=1) as the "event" class.

    POD = TP / (TP + FN)
    FAR = FP / (TP + FP)
    CSI = TP / (TP + FP + FN)
    HSS = 2*(TP*TN - FP*FN) / [(TP+FN)(FN+TN) + (TP+FP)(FP+TN)]
    ETS = (TP - R) / (TP + FP + FN - R),  R = (TP+FP)(TP+FN)/N
    Bias = (TP + FP) / (TP + FN)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    N = TP + FP + FN + TN

    eps = 1e-10
    pod = TP / (TP + FN + eps)
    far = FP / (TP + FP + eps)
    csi = TP / (TP + FP + FN + eps)
    bias = (TP + FP) / (TP + FN + eps)

    denom_hss = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
    hss = 2 * (TP * TN - FP * FN) / (denom_hss + eps)

    R = (TP + FP) * (TP + FN) / (N + eps)
    ets = (TP - R) / (TP + FP + FN - R + eps)

    # Brier Score and Brier Skill Score
    bs = float(brier_score_loss(y_true, y_prob_rain))
    # Reference: climatology forecast (always predict base rate)
    pi = y_true.mean()
    bs_ref = float(brier_score_loss(y_true, np.full_like(y_true, pi)))
    bss = 1.0 - bs / (bs_ref + eps)

    return {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'POD': float(pod),
        'FAR': float(far),
        'CSI': float(csi),
        'BIAS': float(bias),
        'HSS': float(hss),
        'ETS': float(ets),
        'Brier': float(bs),
        'BSS': float(bss),
        'climatology_rate': float(pi),
    }


def compute_murphy_decomposition(y_true, y_prob_rain, n_bins=15):
    """
    Murphy decomposition of Brier score:
    BS = REL - RES + UNC
    REL: reliability (calibration)
    RES: resolution
    UNC: uncertainty (climatological)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob_rain)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    rel = 0.0
    res = 0.0
    pi = y_true.mean()
    unc = pi * (1 - pi)

    for i in range(n_bins):
        if i == 0:
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        else:
            in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        n_k = in_bin.sum()
        if n_k > 0:
            obs_freq = y_true[in_bin].mean()
            fore_mean = y_prob[in_bin].mean()
            rel += n_k * (obs_freq - fore_mean) ** 2
            res += n_k * (obs_freq - pi) ** 2

    rel /= len(y_true)
    res /= len(y_true)

    return {
        'REL': float(rel),
        'RES': float(res),
        'UNC': float(unc),
        'BS_check': float(rel - res + unc),
    }


def run_m4_analysis(models, X_test, y_test, seed=42):
    """Run M4: meteorological skill scores for all models."""
    print("\n" + "=" * 60)
    print("M4: METEOROLOGICAL SKILL SCORES (seed={})".format(seed))
    print("=" * 60)

    results = {}
    for name, model in models.items():
        print(f"\n  [{name}]")
        preds, probs, _ = get_predictions(model, X_test, name)
        y_prob_rain = probs[:, 1]

        skill = compute_skill_scores(y_test, preds, y_prob_rain)
        murphy = compute_murphy_decomposition(y_test, y_prob_rain, n_bins=ECE_N_BINS)

        results[name] = {**skill, **murphy}
        print(f"    POD={skill['POD']:.4f} FAR={skill['FAR']:.4f} CSI={skill['CSI']:.4f}")
        print(f"    HSS={skill['HSS']:.4f} ETS={skill['ETS']:.4f} BSS={skill['BSS']:.4f}")
        print(f"    Murphy: REL={murphy['REL']:.6f} RES={murphy['RES']:.6f} UNC={murphy['UNC']:.6f}")

    # Save
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(RESULTS_DIR, 'm4_skill_scores.csv'))

    # Also save as JSON
    with open(os.path.join(RESULTS_DIR, 'm4_skill_scores.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved to results/m4_skill_scores.csv and m4_skill_scores.json")
    return results


def plot_reliability_diagram(models, X_test, y_test, seed=42):
    """Plot reliability diagrams for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')

    n_bins = ECE_N_BINS
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(models)))
    for (name, model), color in zip(models.items(), colors):
        preds, probs, _ = get_predictions(model, X_test, name)
        y_prob = probs[:, 1]

        bin_accs = []
        bin_confs = []
        for i in range(n_bins):
            if i == 0:
                in_bin = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
            else:
                in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                bin_accs.append(y_test[in_bin].mean())
                bin_confs.append(y_prob[in_bin].mean())
            else:
                bin_accs.append(0)
                bin_confs = bin_centers[i] if len(bin_confs) < i + 1 else None
                bin_accs[-1] = np.nan

        # Filter out NaN
        valid = ~np.isnan(bin_accs)
        bc = np.array(bin_centers[:len(bin_accs)])[valid]
        ba = np.array(bin_accs)[valid]
        ax.plot(bc, ba, 'o-', color=color, label=name, markersize=4, alpha=0.8)

    ax.set_xlabel('Forecast Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_title('Reliability Diagram')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Sharpness diagram (histogram of forecast probabilities)
    ax = axes[1]
    for (name, model), color in zip(models.items(), colors):
        preds, probs, _ = get_predictions(model, X_test, name)
        y_prob = probs[:, 1]
        ax.hist(y_prob, bins=20, alpha=0.3, color=color, label=name, density=True)

    ax.set_xlabel('Forecast Probability')
    ax.set_ylabel('Density')
    ax.set_title('Sharpness (Forecast Distribution)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'fig7_reliability_murphy.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Reliability diagram saved to {path}")


# ============================================================================
# M5: Cost-Loss Analysis
# ============================================================================

def compute_cost_loss(y_true, y_prob_rain, cost_loss_ratios=None):
    """
    Compute expected cost-loss for different cost/loss ratios.

    In the cost-loss model:
    - If rain occurs and we predicted rain: cost = C (protect cost)
    - If rain occurs and we predicted no-rain: cost = L (loss from unprepared)
    - If no rain and we predicted rain: cost = C (unnecessary protect)
    - If no rain and we predicted no-rain: cost = 0

    Expected cost = P(pred_rain) * [C * P(no_rain) + C * P(rain)] + P(pred_no_rain) * L * P(rain)
                  = P(pred_rain) * C + (1 - P(pred_rain)) * L * P(rain)

    For each threshold t: pred_rain if prob > t
    Optimal threshold: t* = C/L (when C < L)

    We compare:
    1. Model-based prediction (optimal threshold for each C/L)
    2. Climatology (always predict rain at rate pi, or always predict rain)
    3. Perfect forecast (cost = pi * C)
    """
    if cost_loss_ratios is None:
        cost_loss_ratios = np.linspace(0.01, 0.99, 50)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob_rain)
    n = len(y_true)
    pi = y_true.mean()  # climatology rain rate

    results = []
    for cl_ratio in cost_loss_ratios:
        C = cl_ratio  # cost of protection
        L = 1.0       # loss (normalized to 1)

        # Optimal threshold = C/L = cl_ratio
        threshold = cl_ratio
        pred_rain = (y_prob > threshold).astype(int)

        # Model expected cost
        # TP: protect when rain -> cost C
        # FP: protect when no rain -> cost C
        # FN: no protect when rain -> cost L
        # TN: no protect when no rain -> cost 0
        tp = np.sum((pred_rain == 1) & (y_true == 1))
        fp = np.sum((pred_rain == 1) & (y_true == 0))
        fn = np.sum((pred_rain == 0) & (y_true == 1))
        tn = np.sum((pred_rain == 0) & (y_true == 0))

        model_cost = (C * (tp + fp) + L * fn) / n

        # Climatology baseline: always predict rain
        clim_cost_always_rain = C  # always protect
        # Or never predict rain
        clim_cost_never_rain = L * pi  # never protect
        clim_cost = min(clim_cost_always_rain, clim_cost_never_rain)

        # Perfect forecast cost
        perfect_cost = C * pi  # only protect when rain

        # Skill score: 1 - model_cost / clim_cost
        skill = 1 - model_cost / (clim_cost + 1e-10)

        results.append({
            'cost_loss_ratio': float(cl_ratio),
            'model_cost': float(model_cost),
            'climatology_cost': float(clim_cost),
            'perfect_cost': float(perfect_cost),
            'skill_score': float(skill),
            'threshold': float(threshold),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        })

    return results


def run_m5_analysis(models, X_test, y_test, seed=42):
    """Run M5: cost-loss analysis for all models."""
    print("\n" + "=" * 60)
    print("M5: COST-LOSS ANALYSIS (seed={})".format(seed))
    print("=" * 60)

    all_results = {}
    for name, model in models.items():
        print(f"\n  [{name}]")
        preds, probs, _ = get_predictions(model, X_test, name)
        y_prob_rain = probs[:, 1]

        cl_results = compute_cost_loss(y_test, y_prob_rain)
        all_results[name] = cl_results

        # Report at key C/L ratios
        for cl in [0.1, 0.3, 0.5, 0.7]:
            idx = min(range(len(cl_results)), key=lambda i: abs(cl_results[i]['cost_loss_ratio'] - cl))
            r = cl_results[idx]
            print(f"    C/L={r['cost_loss_ratio']:.2f}: model_cost={r['model_cost']:.4f} "
                  f"clim_cost={r['climatology_cost']:.4f} skill={r['skill_score']:.4f}")

    # Save
    with open(os.path.join(RESULTS_DIR, 'm5_cost_loss.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary table: average skill across C/L ratios
    summary_rows = []
    for name, cl_list in all_results.items():
        avg_skill = np.mean([r['skill_score'] for r in cl_list])
        max_skill = max(r['skill_score'] for r in cl_list)
        best_cl = cl_list[np.argmax([r['skill_score'] for r in cl_list])]['cost_loss_ratio']
        summary_rows.append({
            'model': name,
            'avg_skill': float(avg_skill),
            'max_skill': float(max_skill),
            'best_cl_ratio': float(best_cl),
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 'm5_cost_loss_summary.csv'), index=False)

    print(f"\n  Saved to results/m5_cost_loss.json and m5_cost_loss_summary.csv")

    # Plot cost-loss curves
    plot_cost_loss_curves(all_results)
    return all_results


def plot_cost_loss_curves(all_results):
    """Plot cost-loss curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Expected cost vs C/L ratio
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(all_results)))
    for (name, cl_list), color in zip(all_results.items(), colors):
        cl = [r['cost_loss_ratio'] for r in cl_list]
        cost = [r['model_cost'] for r in cl_list]
        ax.plot(cl, cost, '-', color=color, label=name, linewidth=1.5)

    # Add climatology and perfect
    cl = [r['cost_loss_ratio'] for r in list(all_results.values())[0]]
    clim_cost = [r['climatology_cost'] for r in list(all_results.values())[0]]
    perfect_cost = [r['perfect_cost'] for r in list(all_results.values())[0]]
    ax.plot(cl, clim_cost, 'k--', label='Climatology', linewidth=2)
    ax.plot(cl, perfect_cost, 'k:', label='Perfect', linewidth=2)

    ax.set_xlabel('Cost/Loss Ratio (C/L)')
    ax.set_ylabel('Expected Cost')
    ax.set_title('Cost-Loss Curve')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: Skill score vs C/L ratio
    ax = axes[1]
    for (name, cl_list), color in zip(all_results.items(), colors):
        cl = [r['cost_loss_ratio'] for r in cl_list]
        skill = [r['skill_score'] for r in cl_list]
        ax.plot(cl, skill, '-', color=color, label=name, linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Cost/Loss Ratio (C/L)')
    ax.set_ylabel('Skill Score')
    ax.set_title('Economic Skill Score')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'fig8_cost_loss.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Cost-loss plot saved to {path}")


# ============================================================================
# M6: Selective Prediction Comparison (AURC / E-AURC, 5 seeds)
# ============================================================================

def compute_aurc(errors, uncertainties):
    """
    Compute Area Under Risk-Coverage curve (AURC).
    AURC = integral of risk over coverage [0, 1].

    Risk at coverage c: average error rate of the top-c fraction
    of samples ranked by confidence (lowest uncertainty first).

    Also compute E-AURC = AURC - AURC_oracle, where AURC_oracle
    is the AURC of a model that always abstains on errors first.
    """
    errors = np.asarray(errors, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)
    n = len(errors)

    # Sort by uncertainty (ascending: most confident first)
    order = np.argsort(uncertainties)
    errors_sorted = errors[order]

    # Risk at each coverage level
    # coverage = k/n for k=1..n
    # risk(k) = sum(errors_sorted[:k]) / k
    cumsum_errors = np.cumsum(errors_sorted)
    coverage = np.arange(1, n + 1) / n
    risk = cumsum_errors / np.arange(1, n + 1)

    # AURC = mean risk over all coverage levels (trapezoidal)
    aurc = np.trapezoid(risk, coverage)

    # Oracle: sort by error (0s first, 1s last) — always abstain on errors
    errors_oracle_sorted = np.sort(errors)
    cumsum_oracle = np.cumsum(errors_oracle_sorted)
    risk_oracle = cumsum_oracle / np.arange(1, n + 1)
    aurc_oracle = np.trapezoid(risk_oracle, coverage)

    # E-AURC = AURC - AURC_oracle (excess risk over oracle)
    e_aurc = aurc - aurc_oracle

    return {
        'aurc': float(aurc),
        'aurc_oracle': float(aurc_oracle),
        'e_aurc': float(e_aurc),
        'risk_at_0.8': float(risk[int(0.8 * n) - 1]),
        'risk_at_0.9': float(risk[int(0.9 * n) - 1]),
        'risk_at_0.95': float(risk[int(0.95 * n) - 1]),
    }


def run_m6_analysis(X_test, y_test, input_dim):
    """Run M6: selective prediction comparison across all models and seeds."""
    print("\n" + "=" * 60)
    print("M6: SELECTIVE PREDICTION COMPARISON (5 seeds)")
    print("=" * 60)

    all_results = {}

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        models = load_all_models(seed, input_dim)
        all_results[seed] = {}

        for name, model in models.items():
            preds, probs, H_T = get_predictions(model, X_test, name)
            errors = (preds != y_test).astype(int)

            aurc_result = compute_aurc(errors, H_T)
            all_results[seed][name] = aurc_result
            print(f"    {name}: AURC={aurc_result['aurc']:.6f} E-AURC={aurc_result['e_aurc']:.6f}")

    # Aggregate across seeds
    print("\n  Aggregating across seeds...")
    model_names = set()
    for s in all_results:
        model_names.update(all_results[s].keys())

    aggregated = {}
    for name in sorted(model_names):
        values = {metric: [] for metric in ['aurc', 'aurc_oracle', 'e_aurc',
                                             'risk_at_0.8', 'risk_at_0.9', 'risk_at_0.95']}
        for s in all_results:
            if name in all_results[s]:
                for metric in values:
                    values[metric].append(all_results[s][name][metric])

        aggregated[name] = {}
        for metric, vals in values.items():
            if vals:
                arr = np.array(vals)
                aggregated[name][metric] = {
                    'mean': float(arr.mean()),
                    'std': float(arr.std(ddof=1)),
                    'values': [float(v) for v in vals],
                }

    # Save
    with open(os.path.join(RESULTS_DIR, 'm6_selective_prediction.json'), 'w') as f:
        json.dump({'per_seed': all_results, 'aggregated': aggregated}, f, indent=2)

    # Summary table
    rows = []
    for name in sorted(model_names):
        if name in aggregated and 'aurc' in aggregated[name]:
            r = aggregated[name]
            rows.append({
                'model': name,
                'aurc_mean': r['aurc']['mean'],
                'aurc_std': r['aurc']['std'],
                'e_aurc_mean': r['e_aurc']['mean'],
                'e_aurc_std': r['e_aurc']['std'],
                'risk_0.9_mean': r['risk_at_0.9']['mean'],
                'risk_0.9_std': r['risk_at_0.9']['std'],
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'm6_selective_prediction_summary.csv'), index=False)
    print(f"\n  Saved to results/m6_selective_prediction.json and m6_selective_prediction_summary.csv")

    # Plot selective prediction curves (seed 42)
    plot_selective_prediction_curves()

    return aggregated


def plot_selective_prediction_curves():
    """Plot risk-coverage curves for all models (seed 42)."""
    # Load seed 42 data
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = preprocess_and_split(seed=42, save=False)
    input_dim = X_test.shape[1]
    models = load_all_models(42, input_dim)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(models)))

    for (name, model), color in zip(models.items(), colors):
        preds, probs, H_T = get_predictions(model, X_test, name)
        errors = (preds != y_test).astype(int)

        n = len(errors)
        order = np.argsort(H_T)
        errors_sorted = errors[order]
        cumsum = np.cumsum(errors_sorted)
        coverage = np.arange(1, n + 1) / n
        risk = cumsum / np.arange(1, n + 1)

        ax.plot(coverage, risk, '-', color=color, label=name, linewidth=1.5)

    # Oracle
    errors_oracle = np.sort(errors)
    cumsum_o = np.cumsum(errors_oracle)
    risk_o = cumsum_o / np.arange(1, n + 1)
    ax.plot(coverage, risk_o, 'k--', label='Oracle', linewidth=2)

    ax.set_xlabel('Coverage')
    ax.set_ylabel('Risk (Error Rate)')
    ax.set_title('Risk-Coverage Curves (Seed 42)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, 0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'fig9_risk_coverage.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Risk-coverage plot saved to {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("M4/M5/M6 COMPREHENSIVE ANALYSIS")
    print("=" * 60)

    # Load data (temporal split, seed 42)
    print("\nLoading data (temporal split, seed=42)...")
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = preprocess_and_split(
        seed=42, save=False, split_mode='temporal')
    input_dim = X_test.shape[1]
    print(f"  Train={len(X_train)} Val={len(X_val)} Test={len(X_test)} dim={input_dim}")
    print(f"  Test rain rate: {y_test.mean():.4f}")

    # Load all models (seed 42)
    print("\nLoading all models (seed=42)...")
    models = load_all_models(42, input_dim)
    print(f"  Loaded {len(models)} models: {list(models.keys())}")

    # M4: Skill scores
    m4_results = run_m4_analysis(models, X_test, y_test, seed=42)
    plot_reliability_diagram(models, X_test, y_test, seed=42)

    # M5: Cost-loss analysis
    m5_results = run_m5_analysis(models, X_test, y_test, seed=42)

    # M6: Selective prediction (5 seeds)
    m6_results = run_m6_analysis(X_test, y_test, input_dim)

    print("\n" + "=" * 60)
    print("ALL M4/M5/M6 ANALYSES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
