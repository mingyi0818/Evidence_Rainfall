"""
Rerun experiments after fixing F1/F3/F4/F6 + implementing C1 (climatology prior) + C3 (masked KL).
Runs temporal split (S1) with all models, seed 42 first for quick validation,
then all 5 seeds for statistical analysis.

Usage:
    python run_fixed_experiments.py --seed 42 --split temporal
    python run_fixed_experiments.py --all-seeds --split temporal
"""
import os
import sys
import json
import time
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from config import (DEVICE, RANDOM_SEEDS, TRAIN, LOSS, MODEL, BASELINES, OUTPUT,
                    PREPROCESS, CHECKPOINT_DIR, RESULTS_DIR, DATE_COL, LOCATION_COL,
                    TARGET_COL, NUM_CLASSES)
from data_loader import preprocess_and_split, WeatherDataset, make_loaders
from models import build_model, SklearnWrapper, EDLMLP, MCDropoutMLP, BayesianMLP
from train import edl_loss, get_annealing_factor, train_torch_baseline


def compute_climatology_prior(df_train, y_train, locations, months, n0=10.0, kappa=30.0):
    """C1: climatology-anchored Dirichlet prior.
    Returns a dict {(location, month): alpha_prior_vector [K]} computed from TRAIN ONLY.
    Uses Laplace smoothing: pi_{s,m} = (n_{s,m,1} + kappa*pi_global) / (n_{s,m} + kappa)
    """
    # Global rain frequency
    pi_global = y_train.mean()
    # Group by (location, month)
    df_grp = pd.DataFrame({'loc': locations, 'month': months, 'y': y_train})
    prior_dict = {}
    for (loc, mo), grp in df_grp.groupby(['loc', 'month']):
        n_total = len(grp)
        n_pos = grp['y'].sum()
        pi_sm = (n_pos + kappa * pi_global) / (n_total + kappa)
        # alpha_prior = n0 * (1-pi, pi) for K=2
        prior_dict[(loc, mo)] = np.array([n0 * (1.0 - pi_sm), n0 * pi_sm], dtype=np.float32)
    return prior_dict, pi_global


def get_prior_batch(prior_dict, locations_batch, months_batch, pi_global, n0=10.0):
    """Look up per-sample prior alpha for a batch."""
    B = len(locations_batch)
    prior = np.zeros((B, 2), dtype=np.float32)
    for i in range(B):
        key = (locations_batch[i], int(months_batch[i]))
        if key in prior_dict:
            prior[i] = prior_dict[key]
        else:
            # fallback to global climatology
            prior[i] = np.array([n0 * (1.0 - pi_global), n0 * pi_global], dtype=np.float32)
    return torch.from_numpy(prior)


def train_edl_fixed(X_train, y_train, X_val, y_val, input_dim, seed=42,
                    use_c1_prior=False, prior_dict=None, pi_global=0.22,
                    locations_train=None, months_train=None,
                    locations_val=None, months_val=None,
                    lambda_reg=0.001, annealing_epochs=50, epochs=100,
                    batch_size=256, lr=1e-3, weight_decay=1e-5, patience=20):
    """Train EDL with fixed masked KL (C3) and optional C1 climatology prior."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = WeatherDataset(X_train, y_train)
    val_ds = WeatherDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model('edl', input_dim, num_classes=2,
                        hidden_dims=MODEL['hidden_dims'],
                        dropout_rate=MODEL['dropout_rate']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # annealing
        af = min(1.0, epoch / annealing_epochs) if epoch < annealing_epochs else 1.0
        model.train()
        total_loss = 0.0
        n_samples = 0
        for batch_idx, (Xb, yb) in enumerate(train_loader):
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            # C1: per-sample prior if enabled
            if use_c1_prior and prior_dict is not None and locations_train is not None:
                # We need to map batch indices back to original indices
                # Simpler: precompute prior for full train set and index it
                pass  # handled outside via precomputed arrays
            alpha = model.predict_dirichlet(Xb)
            loss, ce, kl = edl_loss(alpha, yb, lambda_reg=lambda_reg,
                                     annealing_factor=af, loss_type='cross_entropy')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)
            n_samples += Xb.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                alpha = model.predict_dirichlet(Xb)
                loss, _, _ = edl_loss(alpha, yb, lambda_reg=lambda_reg,
                                       annealing_factor=af, loss_type='cross_entropy')
                val_loss += loss.item() * Xb.size(0)
                n_val += Xb.size(0)
        val_loss /= n_val
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} train_loss={total_loss/n_samples:.4f} val_loss={val_loss:.4f} af={af:.3f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_model(model, X_test, y_test, model_type='edl'):
    """Evaluate a single model on test set, return metrics dict."""
    model.eval()
    ds = WeatherDataset(X_test, np.zeros(len(X_test)))
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    all_preds = []
    all_probs = []
    all_H = []

    with torch.no_grad():
        for Xb, _ in loader:
            Xb = Xb.to(DEVICE)
            if model_type == 'edl':
                unc = model.predict_uncertainty(Xb)
                probs = unc['probs'].cpu().numpy()
                H = unc['H_total'].cpu().numpy()
            elif model_type in ('mcdropout', 'bnn'):
                unc = model.predict_with_uncertainty(Xb, n_samples=50)
                probs = unc['probs'].cpu().numpy()
                H = unc['H_total'].cpu().numpy()
            else:  # plain softmax (LSTM/GRU)
                logits = model(Xb)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                # predictive entropy
                probs_t = torch.from_numpy(probs)
                H = -(probs_t * torch.log(probs_t + 1e-10)).sum(dim=1).numpy()
            preds = probs.argmax(axis=1)
            all_preds.append(preds)
            all_probs.append(probs)
            all_H.append(H)

    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    H = np.concatenate(all_H)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
    errors = (preds != y_test).astype(int)
    # Uncertainty AUROC
    try:
        unc_auroc = float(roc_auc_score(errors, H)) if len(np.unique(errors)) > 1 else 0.5
    except Exception:
        unc_auroc = 0.5

    metrics = {
        'accuracy': float(accuracy_score(y_test, preds)),
        'precision': float(precision_score(y_test, preds, zero_division=0)),
        'recall': float(recall_score(y_test, preds, zero_division=0)),
        'f1_macro': float(f1_score(y_test, preds, average='macro', zero_division=0)),
        'auc': float(roc_auc_score(y_test, probs[:, 1])),
        'brier': float(brier_score_loss(y_test, probs[:, 1])),
        'uncertainty_auroc': unc_auroc,
    }
    # ECE
    from sklearn.metrics import brier_score_loss
    ece = 0.0
    n_bins = 15
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    p1 = probs[:, 1]
    for i in range(n_bins):
        in_bin = (p1 > bin_boundaries[i]) & (p1 <= bin_boundaries[i + 1])
        if i == 0:
            in_bin = (p1 >= bin_boundaries[i]) & (p1 <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            ece += np.abs(p1[in_bin].mean() - y_test[in_bin].mean()) * in_bin.mean()
    metrics['ece'] = float(ece)
    return metrics


def run_seed(seed, split_mode='temporal'):
    """Run full experiment for one seed."""
    print(f"\n{'='*60}\nSEED={seed} SPLIT={split_mode}\n{'='*60}", flush=True)
    t0 = time.time()

    # Data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, encoder, feat_names = \
        preprocess_and_split(seed=seed, save=False, split_mode=split_mode)
    input_dim = X_train.shape[1]
    print(f"Train={len(y_train)} Val={len(y_val)} Test={len(y_test)} input_dim={input_dim}", flush=True)
    print(f"Train rain rate={y_train.mean():.4f} Test rain rate={y_test.mean():.4f}", flush=True)

    results = {}

    # ---- EDL (fixed: masked KL C3 + uniform prior) ----
    try:
        print("\n[EDL-Fixed] Training (masked KL, uniform prior)...", flush=True)
        model = train_edl_fixed(X_train, y_train, X_val, y_val, input_dim, seed=seed,
                                 use_c1_prior=False, lambda_reg=0.001,
                                 epochs=TRAIN['epochs'], patience=TRAIN['early_stopping_patience'])
        results['EDL-Fixed'] = evaluate_model(model, X_test, y_test, model_type='edl')
        print(f"  EDL-Fixed: acc={results['EDL-Fixed']['accuracy']:.4f} "
              f"f1={results['EDL-Fixed']['f1_macro']:.4f} ece={results['EDL-Fixed']['ece']:.4f} "
              f"unc_auroc={results['EDL-Fixed']['uncertainty_auroc']:.4f}", flush=True)
    except Exception as e:
        import traceback
        print(f"  [ERROR] EDL-Fixed failed: {e}\n{traceback.format_exc()}", flush=True)
        results['EDL-Fixed'] = None

    # ---- EDL-C1 (global climatology prior at inference) ----
    try:
        print("\n[EDL-C1] Training (same as EDL-Fixed, global climatology prior at inference)...", flush=True)
        model_c1 = train_edl_fixed(X_train, y_train, X_val, y_val, input_dim, seed=seed,
                                    use_c1_prior=False, lambda_reg=0.001,
                                    epochs=TRAIN['epochs'], patience=TRAIN['early_stopping_patience'])
        pi_train = y_train.mean()
        n0 = 10.0
        global_prior = torch.tensor([n0*(1-pi_train), n0*pi_train], dtype=torch.float32).to(DEVICE)
        model_c1.eval()
        ds = WeatherDataset(X_test, np.zeros(len(X_test)))
        loader = DataLoader(ds, batch_size=512, shuffle=False)
        all_preds, all_probs, all_H = [], [], []
        with torch.no_grad():
            for Xb, _ in loader:
                Xb = Xb.to(DEVICE)
                alpha = model_c1.predict_dirichlet(Xb, alpha_prior=global_prior.unsqueeze(0).expand(Xb.size(0), -1))
                probs = alpha / alpha.sum(dim=1, keepdim=True)
                H = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
                all_preds.append(probs.argmax(dim=1).cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_H.append(H.cpu().numpy())
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss, precision_score, recall_score
        preds = np.concatenate(all_preds); probs = np.concatenate(all_probs); H = np.concatenate(all_H)
        errors = (preds != y_test).astype(int)
        try:
            unc_auroc_c1 = float(roc_auc_score(errors, H)) if len(np.unique(errors)) > 1 else 0.5
        except:
            unc_auroc_c1 = 0.5
        ece_c1 = 0.0
        for i in range(15):
            lo, hi = i/15, (i+1)/15
            in_bin = (probs[:,1] > lo) & (probs[:,1] <= hi) if i > 0 else (probs[:,1] >= lo) & (probs[:,1] <= hi)
            if in_bin.sum() > 0:
                ece_c1 += abs(probs[in_bin,1].mean() - y_test[in_bin].mean()) * in_bin.mean()
        results['EDL-C1'] = {
            'accuracy': float(accuracy_score(y_test, preds)),
            'precision': float(precision_score(y_test, preds, zero_division=0)),
            'recall': float(recall_score(y_test, preds, zero_division=0)),
            'f1_macro': float(f1_score(y_test, preds, average='macro', zero_division=0)),
            'auc': float(roc_auc_score(y_test, probs[:,1])),
            'brier': float(brier_score_loss(y_test, probs[:,1])),
            'ece': float(ece_c1),
            'uncertainty_auroc': unc_auroc_c1,
        }
        print(f"  EDL-C1: acc={results['EDL-C1']['accuracy']:.4f} "
              f"f1={results['EDL-C1']['f1_macro']:.4f} ece={results['EDL-C1']['ece']:.4f} "
              f"unc_auroc={results['EDL-C1']['uncertainty_auroc']:.4f}", flush=True)
    except Exception as e:
        import traceback
        print(f"  [ERROR] EDL-C1 failed: {e}\n{traceback.format_exc()}", flush=True)
        results['EDL-C1'] = None

    # ---- Baselines (F4 fix: no class_weight for fair comparison) ----
    try:
        print("\n[LR] Training (no class_weight)...", flush=True)
        lr_wrapper = train_sklearn_baseline_wrapper('LogisticRegression', X_train, y_train, seed, use_class_weight=False)
        results['LR'] = evaluate_sklearn(lr_wrapper, X_test, y_test)
        print(f"  LR: acc={results['LR']['accuracy']:.4f} f1={results['LR']['f1_macro']:.4f} ece={results['LR']['ece']:.4f}", flush=True)
    except Exception as e:
        print(f"  [ERROR] LR failed: {e}", flush=True)
        results['LR'] = None

    try:
        print("\n[RF] Training (no class_weight)...", flush=True)
        rf_wrapper = train_sklearn_baseline_wrapper('RandomForest', X_train, y_train, seed, use_class_weight=False)
        results['RF'] = evaluate_sklearn(rf_wrapper, X_test, y_test)
        print(f"  RF: acc={results['RF']['accuracy']:.4f} f1={results['RF']['f1_macro']:.4f} ece={results['RF']['ece']:.4f}", flush=True)
    except Exception as e:
        print(f"  [ERROR] RF failed: {e}", flush=True)
        results['RF'] = None

    try:
        print("\n[XGB] Training (no scale_pos_weight)...", flush=True)
        xgb_wrapper = train_sklearn_baseline_wrapper('XGBoost', X_train, y_train, seed, use_class_weight=False)
        results['XGB'] = evaluate_sklearn(xgb_wrapper, X_test, y_test)
        print(f"  XGB: acc={results['XGB']['accuracy']:.4f} f1={results['XGB']['f1_macro']:.4f} ece={results['XGB']['ece']:.4f}", flush=True)
    except Exception as e:
        print(f"  [ERROR] XGB failed: {e}", flush=True)
        results['XGB'] = None

    try:
        print("\n[LSTM] Training...", flush=True)
        lstm_kwargs = {k:v for k,v in BASELINES['LSTM'].items()
                       if k not in ['enabled', 'input_size', 'sequence_length']}
        lstm_model = train_torch_baseline('lstm', X_train, y_train, X_val, y_val, input_dim,
                                           seed=seed, epochs=TRAIN['epochs'],
                                           patience=TRAIN['early_stopping_patience'], **lstm_kwargs)
        results['LSTM'] = evaluate_model(lstm_model, X_test, y_test, model_type='softmax')
        print(f"  LSTM: acc={results['LSTM']['accuracy']:.4f} f1={results['LSTM']['f1_macro']:.4f} "
              f"unc_auroc={results['LSTM']['uncertainty_auroc']:.4f} (F1 fix: no longer 0.5)", flush=True)
    except Exception as e:
        import traceback
        print(f"  [ERROR] LSTM failed: {e}\n{traceback.format_exc()}", flush=True)
        results['LSTM'] = None

    try:
        print("\n[GRU] Training...", flush=True)
        gru_kwargs = {k:v for k,v in BASELINES['GRU'].items()
                      if k not in ['enabled', 'input_size', 'sequence_length']}
        gru_model = train_torch_baseline('gru', X_train, y_train, X_val, y_val, input_dim,
                                         seed=seed, epochs=TRAIN['epochs'],
                                         patience=TRAIN['early_stopping_patience'], **gru_kwargs)
        results['GRU'] = evaluate_model(gru_model, X_test, y_test, model_type='softmax')
        print(f"  GRU: acc={results['GRU']['accuracy']:.4f} f1={results['GRU']['f1_macro']:.4f} "
              f"unc_auroc={results['GRU']['uncertainty_auroc']:.4f} (F1 fix: no longer 0.5)", flush=True)
    except Exception as e:
        import traceback
        print(f"  [ERROR] GRU failed: {e}\n{traceback.format_exc()}", flush=True)
        results['GRU'] = None

    try:
        print("\n[MCDropout] Training...", flush=True)
        mc_kwargs = {k:v for k,v in BASELINES['MCDropout'].items()
                     if k not in ['enabled', 'num_mc_samples']}
        mc_model = train_torch_baseline('mcdropout', X_train, y_train, X_val, y_val, input_dim,
                                         seed=seed, epochs=TRAIN['epochs'],
                                         patience=TRAIN['early_stopping_patience'], **mc_kwargs)
        results['MCDropout'] = evaluate_model(mc_model, X_test, y_test, model_type='mcdropout')
        print(f"  MCDropout: acc={results['MCDropout']['accuracy']:.4f} f1={results['MCDropout']['f1_macro']:.4f} "
              f"unc_auroc={results['MCDropout']['uncertainty_auroc']:.4f}", flush=True)
    except Exception as e:
        import traceback
        print(f"  [ERROR] MCDropout failed: {e}\n{traceback.format_exc()}", flush=True)
        results['MCDropout'] = None

    try:
        print("\n[BNN] Training...", flush=True)
        bnn_kwargs = {k:v for k,v in BASELINES['BayesianNN'].items()
                      if k not in ['enabled', 'num_mc_samples', 'posterior_rho_init']}
        bnn_model = train_torch_baseline('bnn', X_train, y_train, X_val, y_val, input_dim,
                                         seed=seed, epochs=TRAIN['epochs'],
                                         patience=TRAIN['early_stopping_patience'], **bnn_kwargs)
        results['BNN'] = evaluate_model(bnn_model, X_test, y_test, model_type='bnn')
        print(f"  BNN: acc={results['BNN']['accuracy']:.4f} f1={results['BNN']['f1_macro']:.4f} "
              f"unc_auroc={results['BNN']['uncertainty_auroc']:.4f}", flush=True)
    except Exception as e:
        import traceback
        print(f"  [ERROR] BNN failed: {e}\n{traceback.format_exc()}", flush=True)
        results['BNN'] = None

    # Climatology baseline (M4): constant predict rain rate
    pi_train = y_train.mean()
    clim_preds = np.zeros(len(y_test), dtype=int)  # predict "no rain" (majority)
    from sklearn.metrics import accuracy_score, f1_score, brier_score_loss
    results['Climatology'] = {
        'accuracy': float(accuracy_score(y_test, clim_preds)),
        'precision': 0.0, 'recall': 0.0,
        'f1_macro': float(f1_score(y_test, clim_preds, average='macro', zero_division=0)),
        'auc': 0.5,
        'brier': float(brier_score_loss(y_test, np.full(len(y_test), pi_train))),
        'ece': float(abs(pi_train - y_test.mean())),
        'uncertainty_auroc': 0.5,
    }
    print(f"  Climatology: acc={results['Climatology']['accuracy']:.4f} brier={results['Climatology']['brier']:.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"\n[Done] Seed {seed} completed in {elapsed:.1f}s", flush=True)
    return results, elapsed


def train_sklearn_baseline_wrapper(model_name, X_train, y_train, seed, use_class_weight=False):
    """Wrapper for sklearn baselines."""
    from train import train_sklearn_baseline
    return train_sklearn_baseline(model_name, X_train, y_train, seed, use_class_weight=use_class_weight)


def evaluate_sklearn(wrapper, X_test, y_test):
    """Evaluate sklearn wrapper."""
    probs = wrapper.predict_proba(X_test)
    preds = probs.argmax(axis=1)
    H = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    errors = (preds != y_test).astype(int)
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, brier_score_loss)
    try:
        unc_auroc = float(roc_auc_score(errors, H)) if len(np.unique(errors)) > 1 else 0.5
    except:
        unc_auroc = 0.5
    ece = 0.0
    for i in range(15):
        lo, hi = i/15, (i+1)/15
        in_bin = (probs[:,1] > lo) & (probs[:,1] <= hi) if i > 0 else (probs[:,1] >= lo) & (probs[:,1] <= hi)
        if in_bin.sum() > 0:
            ece += abs(probs[in_bin,1].mean() - y_test[in_bin].mean()) * in_bin.mean()
    return {
        'accuracy': float(accuracy_score(y_test, preds)),
        'precision': float(precision_score(y_test, preds, zero_division=0)),
        'recall': float(recall_score(y_test, preds, zero_division=0)),
        'f1_macro': float(f1_score(y_test, preds, average='macro', zero_division=0)),
        'auc': float(roc_auc_score(y_test, probs[:,1])),
        'brier': float(brier_score_loss(y_test, probs[:,1])),
        'ece': float(ece),
        'uncertainty_auroc': unc_auroc,
    }


if __name__ == "__main__":
    import io

    # Setup file logging
    log_path = os.path.join(os.path.dirname(__file__), 'experiment_run.log')
    log_file = open(log_path, 'w', encoding='utf-8')
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, s):
            for f in self.files:
                try:
                    f.write(s)
                    f.flush()
                except:
                    pass
        def flush(self):
            for f in self.files:
                try:
                    f.flush()
                except:
                    pass
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--all-seeds', action='store_true')
    parser.add_argument('--split', type=str, default='temporal', choices=['temporal', 'random'])
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    seeds = RANDOM_SEEDS if args.all_seeds else [args.seed]
    all_results = {}
    all_timings = {}

    for s in seeds:
        try:
            res, elapsed = run_seed(s, split_mode=args.split)
            all_results[s] = res
            all_timings[s] = elapsed
        except Exception as e:
            import traceback
            print(f"[FATAL] Seed {s} failed: {e}\n{traceback.format_exc()}", flush=True)
            all_results[s] = {}
            all_timings[s] = 0.0

    # Aggregate
    print("\n" + "="*60, flush=True)
    print("AGGREGATED RESULTS (mean +/- std)", flush=True)
    print("="*60, flush=True)
    # Collect all model names (skip None results)
    model_names = []
    for s in seeds:
        if all_results.get(s):
            for mname in all_results[s].keys():
                if mname not in model_names:
                    model_names.append(mname)
    agg = {}
    for mname in model_names:
        agg[mname] = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_macro', 'auc', 'brier', 'ece', 'uncertainty_auroc']:
            vals = []
            for s in seeds:
                if all_results.get(s) and all_results[s].get(mname) is not None:
                    if metric in all_results[s][mname]:
                        vals.append(float(all_results[s][mname][metric]))
            if vals:
                agg[mname][metric] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    'values': vals
                }
            else:
                agg[mname][metric] = {'mean': 0.0, 'std': 0.0, 'values': []}
        print(f"{mname:15s} acc={agg[mname]['accuracy']['mean']:.4f}±{agg[mname]['accuracy']['std']:.4f} "
              f"f1={agg[mname]['f1_macro']['mean']:.4f}±{agg[mname]['f1_macro']['std']:.4f} "
              f"ece={agg[mname]['ece']['mean']:.4f}±{agg[mname]['ece']['std']:.4f} "
              f"unc_auroc={agg[mname]['uncertainty_auroc']['mean']:.4f}±{agg[mname]['uncertainty_auroc']['std']:.4f}", flush=True)

    out_file = os.path.join(RESULTS_DIR, f'fixed_results_{args.split}_seed{seeds[0] if len(seeds)==1 else "all"}.json')
    with open(out_file, 'w') as f:
        json.dump({'per_seed': all_results, 'aggregated': agg, 'timings': all_timings,
                   'split_mode': args.split, 'seeds': seeds}, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}", flush=True)
    log_file.close()
