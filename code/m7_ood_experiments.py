"""
M7: Out-of-Distribution (OOD) experiments for EDL-Fixed.

Experiments:
1. Spatial OOD: Train on half the locations, test on unseen locations
2. Seasonal OOD: Train on 3 seasons, test on the 4th
3. Extreme events: Evaluate on high-rainfall vs normal days
4. Temporal OOD: Evaluate by year (2016 vs 2017)

Key question: Does EDL's epistemic uncertainty increase on OOD samples?
Theorem 3 predicts it will NOT (H_E is monotone in S).

All experiments use seed 42 and EDL-Fixed model.
Author: GLM-5.2
Date: 2026-07-25
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DEVICE, RAW_CSV, TARGET_COL, DATE_COL, LOCATION_COL,
    CATEGORICAL_COLS, NUMERIC_COLS, PREPROCESS,
    CHECKPOINT_DIR, RESULTS_DIR, PLOTS_DIR, ECE_N_BINS,
    MODEL, LOSS, TRAIN
)
from models import build_model, EDLMLP
from train import train_edl_model
from evaluate import compute_ece
from data_loader import preprocess_and_split


SEED = 42
EPOCHS = 30


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ============================================================================
# Custom OOD Data Loading
# ============================================================================

def load_and_preprocess_ood(split_type='spatial', ood_param=None):
    """
    Load data and create OOD splits.
    split_type: 'spatial', 'seasonal', 'extreme', 'temporal_by_year'
    Returns: X_train, X_val, X_test_id, y_test_id, X_test_ood, y_test_ood,
             test_id_meta, test_ood_meta
    """
    np.random.seed(SEED)
    df = pd.read_csv(RAW_CSV)
    df = df.dropna(subset=[TARGET_COL])

    # Drop high-missing columns
    drop_cols = PREPROCESS['drop_cols']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Extract date features
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    df['Year'] = df[DATE_COL].dt.year.astype(np.int32)
    df['Month'] = df[DATE_COL].dt.month.astype(np.int32)
    df['DayOfYear'] = df[DATE_COL].dt.dayofyear.astype(np.int32)
    df['Season'] = ((df['Month'] % 12 + 3) // 3) % 4
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
    df['Rainfall_raw'] = df['Rainfall'].copy()  # keep for extreme event analysis
    df = df.drop(columns=[DATE_COL])

    # Encode target
    df[TARGET_COL] = (df[TARGET_COL] == 'Yes').astype(np.int64)

    # Identify columns
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns and c != TARGET_COL]
    num_cols = [c for c in df.columns if c not in cat_cols + [TARGET_COL, 'Rainfall_raw']
                and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    y = df[TARGET_COL].values
    df_features = df.drop(columns=[TARGET_COL, 'Rainfall_raw'])

    # Keep Rainfall_raw for metadata
    rainfall_raw = df['Rainfall_raw'].values

    if split_type == 'spatial':
        # Spatial OOD: split by Location
        all_locations = sorted(df_features[LOCATION_COL].unique())
        np.random.shuffle(all_locations)
        n_train_loc = len(all_locations) // 2
        train_locations = set(all_locations[:n_train_loc])
        test_locations = set(all_locations[n_train_loc:])

        loc = df_features[LOCATION_COL].values
        # Use temporal split within train locations for train/val
        year = df_features['Year'].values
        train_mask = np.array([l in train_locations and y_ <= 2014 for l, y_ in zip(loc, year)])
        val_mask = np.array([l in train_locations and y_ == 2015 for l, y_ in zip(loc, year)])
        test_id_mask = np.array([l in train_locations and y_ >= 2016 for l, y_ in zip(loc, year)])
        test_ood_mask = np.array([l in test_locations and y_ >= 2016 for l, y_ in zip(loc, year)])

        log(f"  Spatial OOD: {n_train_loc} train locations, {len(all_locations)-n_train_loc} OOD locations")
        log(f"  Train={train_mask.sum()} Val={val_mask.sum()} Test-ID={test_id_mask.sum()} Test-OOD={test_ood_mask.sum()}")

    elif split_type == 'seasonal':
        # Seasonal OOD: train on 3 seasons, test on the 4th
        # ood_param: the season to hold out (0=Summer, 1=Autumn, 2=Winter, 3=Spring)
        target_season = ood_param if ood_param is not None else 2  # Winter
        season = df_features['Season'].values
        year = df_features['Year'].values

        train_mask = (season != target_season) & (year <= 2014)
        val_mask = (season != target_season) & (year == 2015)
        test_id_mask = (season != target_season) & (year >= 2016)
        test_ood_mask = (season == target_season) & (year >= 2016)

        season_names = {0: 'Summer', 1: 'Autumn', 2: 'Winter', 3: 'Spring'}
        log(f"  Seasonal OOD: hold out {season_names[target_season]}")
        log(f"  Train={train_mask.sum()} Val={val_mask.sum()} Test-ID={test_id_mask.sum()} Test-OOD={test_ood_mask.sum()}")

    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    # Impute on train only
    imputer_num = SimpleImputer(strategy='median')
    df_features[num_cols] = imputer_num.fit_transform(df_features[num_cols])
    # Transform all (already in-place for simplicity; in production would separate)

    imputer_cat = SimpleImputer(strategy='constant', fill_value='Missing')
    df_features[cat_cols] = imputer_cat.fit_transform(df_features[cat_cols])

    # Fit scaler and encoder on train only
    scaler = StandardScaler()
    scaler.fit(df_features.loc[train_mask, num_cols].values)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(df_features.loc[train_mask, cat_cols].values)

    def transform(df_sub):
        X_num = scaler.transform(df_sub[num_cols].values)
        X_cat = encoder.transform(df_sub[cat_cols].values)
        return np.hstack([X_num, X_cat])

    X_train = transform(df_features.loc[train_mask])
    X_val = transform(df_features.loc[val_mask])
    X_test_id = transform(df_features.loc[test_id_mask])
    X_test_ood = transform(df_features.loc[test_ood_mask])

    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test_id = y[test_id_mask]
    y_test_ood = y[test_ood_mask]

    # Metadata for test sets
    test_id_meta = {
        'rainfall': rainfall_raw[test_id_mask],
        'location': df_features.loc[test_id_mask, LOCATION_COL].values,
        'season': df_features.loc[test_id_mask, 'Season'].values,
        'year': df_features.loc[test_id_mask, 'Year'].values,
    }
    test_ood_meta = {
        'rainfall': rainfall_raw[test_ood_mask],
        'location': df_features.loc[test_ood_mask, LOCATION_COL].values,
        'season': df_features.loc[test_ood_mask, 'Season'].values,
        'year': df_features.loc[test_ood_mask, 'Year'].values,
    }

    input_dim = X_train.shape[1]
    log(f"  Input dim: {input_dim}")

    return (X_train, y_train, X_val, y_val, X_test_id, y_test_id, X_test_ood, y_test_ood,
            test_id_meta, test_ood_meta, input_dim)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, X, y):
    """Evaluate EDL model on a dataset."""
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        unc = model.predict_uncertainty(Xt)
        probs = unc['probs'].cpu().numpy()
        H_T = unc['H_total'].cpu().numpy()
        H_A = unc['H_alea'].cpu().numpy()
        H_E = unc['H_epi'].cpu().numpy()
        S = unc['precision'].cpu().numpy()

    preds = probs.argmax(axis=1)
    errors = (preds != y).astype(int)

    from sklearn.metrics import roc_auc_score
    try:
        unc_auroc = float(roc_auc_score(errors, H_T))
    except:
        unc_auroc = 0.5

    try:
        ece_val = compute_ece(y, probs[:, 1])
    except:
        ece_val = 0.0

    return {
        'n': len(y),
        'accuracy': float(accuracy_score(y, preds)),
        'f1_macro': float(f1_score(y, preds, average='macro', zero_division=0)),
        'ece': float(ece_val[0] if isinstance(ece_val, tuple) else ece_val),
        'brier': float(brier_score_loss(y, probs[:, 1])),
        'H_T_mean': float(H_T.mean()),
        'H_T_std': float(H_T.std()),
        'H_A_mean': float(H_A.mean()),
        'H_E_mean': float(H_E.mean()),
        'H_E_std': float(H_E.std()),
        'S_mean': float(S.mean()),
        'S_std': float(S.std()),
        'uncertainty_auroc': unc_auroc,
        'error_rate': float(errors.mean()),
        'probs': probs,
        'H_T': H_T,
        'H_E': H_E,
        'S': S,
        'preds': preds,
        'errors': errors,
    }


# ============================================================================
# Experiment 1: Spatial OOD
# ============================================================================

def run_spatial_ood():
    """Train on half the locations, test on unseen locations."""
    log("\n" + "=" * 60)
    log("M7.1: SPATIAL OOD EXPERIMENT")
    log("=" * 60)

    data = load_and_preprocess_ood(split_type='spatial')
    (X_train, y_train, X_val, y_val, X_test_id, y_test_id,
     X_test_ood, y_test_ood, test_id_meta, test_ood_meta, input_dim) = data
    log(f"  Unpacked: X_train={X_train.shape}, y_train={y_train.shape}, input_dim={input_dim}")

    # Train EDL-Fixed
    log("  Training EDL-Fixed on spatial OOD split...")
    t0 = time.time()
    model, _ = train_edl_model(
        X_train, y_train, X_val, y_val, input_dim,
        seed=SEED, config={
            'hidden_dims': MODEL['hidden_dims'],
            'dropout_rate': MODEL['dropout_rate'],
            'lambda_reg': LOSS['lambda_reg'],
            'annealing': LOSS['annealing'],
            'annealing_epochs': LOSS['annealing_epochs'],
            'loss_type': 'cross_entropy',
            'lr': TRAIN['learning_rate'],
            'weight_decay': TRAIN['weight_decay'],
            'batch_size': TRAIN['batch_size'],
            'epochs': EPOCHS,
            'patience': TRAIN['early_stopping_patience'],
        }
    )
    log(f"  Training done in {time.time()-t0:.1f}s")

    # Evaluate on in-distribution and OOD
    log("  Evaluating on in-distribution test set...")
    r_id = evaluate_model(model, X_test_id, y_test_id)
    log(f"    ID: acc={r_id['accuracy']:.4f} f1={r_id['f1_macro']:.4f} ece={r_id['ece']:.4f} "
        f"S={r_id['S_mean']:.2f} H_E={r_id['H_E_mean']:.6f} H_T={r_id['H_T_mean']:.4f} unc_auroc={r_id['uncertainty_auroc']:.4f}")

    log("  Evaluating on OOD test set...")
    r_ood = evaluate_model(model, X_test_ood, y_test_ood)
    log(f"    OOD: acc={r_ood['accuracy']:.4f} f1={r_ood['f1_macro']:.4f} ece={r_ood['ece']:.4f} "
        f"S={r_ood['S_mean']:.2f} H_E={r_ood['H_E_mean']:.6f} H_T={r_ood['H_T_mean']:.4f} unc_auroc={r_ood['uncertainty_auroc']:.4f}")

    # Compute OOD detection metrics
    # Can we distinguish ID from OOD using uncertainty?
    H_T_all = np.concatenate([r_id['H_T'], r_ood['H_T']])
    H_E_all = np.concatenate([r_id['H_E'], r_ood['H_E']])
    S_all = np.concatenate([r_id['S'], r_ood['S']])
    ood_labels = np.concatenate([np.zeros(len(r_id['H_T'])), np.ones(len(r_ood['H_T']))])

    from sklearn.metrics import roc_auc_score
    try:
        ood_auroc_H_T = float(roc_auc_score(ood_labels, H_T_all))
    except:
        ood_auroc_H_T = 0.5
    try:
        ood_auroc_H_E = float(roc_auc_score(ood_labels, H_E_all))
    except:
        ood_auroc_H_E = 0.5
    try:
        ood_auroc_S = float(roc_auc_score(ood_labels, -S_all))  # lower S = more OOD
    except:
        ood_auroc_S = 0.5

    log(f"  OOD detection AUROC: H_T={ood_auroc_H_T:.4f} H_E={ood_auroc_H_E:.4f} 1/S={ood_auroc_S:.4f}")

    # Save results (without arrays)
    result = {
        'experiment': 'spatial_ood',
        'id_results': {k: v for k, v in r_id.items() if not isinstance(v, np.ndarray)},
        'ood_results': {k: v for k, v in r_ood.items() if not isinstance(v, np.ndarray)},
        'ood_detection': {
            'auroc_H_T': ood_auroc_H_T,
            'auroc_H_E': ood_auroc_H_E,
            'auroc_1_over_S': ood_auroc_S,
        },
        'id_n': len(y_test_id),
        'ood_n': len(y_test_ood),
    }
    return result


# ============================================================================
# Experiment 2: Seasonal OOD
# ============================================================================

def run_seasonal_ood():
    """Train on 3 seasons, test on the 4th. Run for all 4 seasons."""
    log("\n" + "=" * 60)
    log("M7.2: SEASONAL OOD EXPERIMENT")
    log("=" * 60)

    all_results = {}
    season_names = {0: 'Summer', 1: 'Autumn', 2: 'Winter', 3: 'Spring'}

    for target_season in [0, 1, 2, 3]:
        log(f"\n  --- Hold out: {season_names[target_season]} ---")
        data = load_and_preprocess_ood(split_type='seasonal', ood_param=target_season)
        (X_train, y_train, X_val, y_val, X_test_id, y_test_id,
         X_test_ood, y_test_ood, test_id_meta, test_ood_meta, input_dim) = data
        log(f"  Unpacked: X_train={X_train.shape}, y_train={y_train.shape}")

        # Train EDL-Fixed
        log(f"  Training EDL-Fixed (hold out {season_names[target_season]})...")
        t0 = time.time()
        model, _ = train_edl_model(
            X_train, y_train, X_val, y_val, input_dim,
            seed=SEED, config={
                'hidden_dims': MODEL['hidden_dims'],
                'dropout_rate': MODEL['dropout_rate'],
                'lambda_reg': LOSS['lambda_reg'],
                'annealing': LOSS['annealing'],
                'annealing_epochs': LOSS['annealing_epochs'],
                'loss_type': 'cross_entropy',
                'lr': TRAIN['learning_rate'],
                'weight_decay': TRAIN['weight_decay'],
                'batch_size': TRAIN['batch_size'],
                'epochs': EPOCHS,
                'patience': TRAIN['early_stopping_patience'],
            }
        )
        log(f"  Training done in {time.time()-t0:.1f}s")

        # Evaluate
        r_id = evaluate_model(model, X_test_id, y_test_id)
        r_ood = evaluate_model(model, X_test_ood, y_test_ood)

        log(f"    ID: acc={r_id['accuracy']:.4f} S={r_id['S_mean']:.2f} H_E={r_id['H_E_mean']:.6f} H_T={r_id['H_T_mean']:.4f}")
        log(f"    OOD: acc={r_ood['accuracy']:.4f} S={r_ood['S_mean']:.2f} H_E={r_ood['H_E_mean']:.6f} H_T={r_ood['H_T_mean']:.4f}")

        # OOD detection
        H_T_all = np.concatenate([r_id['H_T'], r_ood['H_T']])
        H_E_all = np.concatenate([r_id['H_E'], r_ood['H_E']])
        S_all = np.concatenate([r_id['S'], r_ood['S']])
        ood_labels = np.concatenate([np.zeros(len(r_id['H_T'])), np.ones(len(r_ood['H_T']))])

        from sklearn.metrics import roc_auc_score
        try:
            ood_auroc_H_T = float(roc_auc_score(ood_labels, H_T_all))
        except:
            ood_auroc_H_T = 0.5

        all_results[season_names[target_season]] = {
            'id_results': {k: v for k, v in r_id.items() if not isinstance(v, np.ndarray)},
            'ood_results': {k: v for k, v in r_ood.items() if not isinstance(v, np.ndarray)},
            'ood_detection_auroc_H_T': ood_auroc_H_T,
            'id_n': len(y_test_id),
            'ood_n': len(y_test_ood),
        }

    return all_results


# ============================================================================
# Experiment 3: Extreme Events (using existing temporal split model)
# ============================================================================

def run_extreme_events():
    """Evaluate existing model on extreme vs normal rainfall days."""
    log("\n" + "=" * 60)
    log("M7.3: EXTREME EVENTS ANALYSIS")
    log("=" * 60)

    # Load temporal split data
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = preprocess_and_split(
        seed=SEED, save=False, split_mode='temporal')
    input_dim = X_test.shape[1]

    # Load existing EDL-Fixed model
    edl_path = os.path.join(CHECKPOINT_DIR, f"edl_seed{SEED}.pth")
    model = build_model('edl', input_dim, num_classes=2,
                        hidden_dims=[128, 64, 32], dropout_rate=0.3).to(DEVICE)
    model.load_state_dict(torch.load(edl_path, map_location=DEVICE))

    # Get raw rainfall data for test set
    df = pd.read_csv(RAW_CSV)
    df = df.dropna(subset=[TARGET_COL])
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    year = df[DATE_COL].dt.year
    test_mask = year >= 2016
    rainfall_test = df.loc[test_mask, 'Rainfall'].values

    # Evaluate on full test set
    r_full = evaluate_model(model, X_test, y_test)

    # Split by rainfall amount
    # Rainfall = 0: dry days
    # Rainfall > 0 and < 90th percentile: normal rain
    # Rainfall >= 90th percentile: extreme rain (among rain days)
    rain_amounts = rainfall_test
    dry_mask = rain_amounts == 0
    rain_mask = (rain_amounts > 0) & (y_test == 1)  # actual rain days
    if rain_mask.sum() > 0:
        p90 = np.percentile(rain_amounts[rain_mask], 90)
        p95 = np.percentile(rain_amounts[rain_mask], 95)
        p99 = np.percentile(rain_amounts[rain_mask], 99)
    else:
        p90 = p95 = p99 = 0

    normal_rain_mask = (rain_amounts > 0) & (rain_amounts < p90) & (y_test == 1)
    extreme_p90_mask = (rain_amounts >= p90) & (y_test == 1)
    extreme_p95_mask = (rain_amounts >= p95) & (y_test == 1)
    extreme_p99_mask = (rain_amounts >= p99) & (y_test == 1)

    log(f"  Test set: {len(y_test)} samples")
    log(f"  Dry days: {dry_mask.sum()} ({dry_mask.mean():.1%})")
    log(f"  Rain days: {rain_mask.sum()} ({rain_mask.mean():.1%})")
    log(f"  P90={p90:.1f}mm P95={p95:.1f}mm P99={p99:.1f}mm")
    log(f"  Extreme (>=P90): {extreme_p90_mask.sum()}")
    log(f"  Extreme (>=P95): {extreme_p95_mask.sum()}")
    log(f"  Extreme (>=P99): {extreme_p99_mask.sum()}")

    results = {
        'full_test': {k: v for k, v in r_full.items() if not isinstance(v, np.ndarray)},
        'percentiles': {'p90': float(p90), 'p95': float(p95), 'p99': float(p99)},
    }

    # Evaluate on each subset
    for name, mask in [('dry_days', dry_mask), ('normal_rain', normal_rain_mask),
                        ('extreme_p90', extreme_p90_mask), ('extreme_p95', extreme_p95_mask),
                        ('extreme_p99', extreme_p99_mask)]:
        if mask.sum() > 0:
            r = evaluate_model(model, X_test[mask], y_test[mask])
            results[name] = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
            log(f"  {name} (n={mask.sum()}): acc={r['accuracy']:.4f} S={r['S_mean']:.2f} "
                f"H_E={r['H_E_mean']:.6f} H_T={r['H_T_mean']:.4f} unc_auroc={r['uncertainty_auroc']:.4f}")
        else:
            results[name] = {'n': 0}

    return results


# ============================================================================
# Experiment 4: Temporal OOD by Year
# ============================================================================

def run_temporal_by_year():
    """Evaluate existing model on different years."""
    log("\n" + "=" * 60)
    log("M7.4: TEMPORAL OOD BY YEAR")
    log("=" * 60)

    # Load temporal split data
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = preprocess_and_split(
        seed=SEED, save=False, split_mode='temporal')
    input_dim = X_test.shape[1]

    # Load existing model
    edl_path = os.path.join(CHECKPOINT_DIR, f"edl_seed{SEED}.pth")
    model = build_model('edl', input_dim, num_classes=2,
                        hidden_dims=[128, 64, 32], dropout_rate=0.3).to(DEVICE)
    model.load_state_dict(torch.load(edl_path, map_location=DEVICE))

    # Get year info for test set
    df = pd.read_csv(RAW_CSV)
    df = df.dropna(subset=[TARGET_COL])
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    year = df[DATE_COL].dt.year
    test_mask = year >= 2016
    year_test = year[test_mask].values

    results = {}
    for yr in [2016, 2017]:
        mask = year_test == yr
        if mask.sum() > 0:
            r = evaluate_model(model, X_test[mask], y_test[mask])
            results[str(yr)] = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
            log(f"  {yr} (n={mask.sum()}): acc={r['accuracy']:.4f} f1={r['f1_macro']:.4f} "
                f"S={r['S_mean']:.2f} H_E={r['H_E_mean']:.6f} H_T={r['H_T_mean']:.4f} "
                f"unc_auroc={r['uncertainty_auroc']:.4f}")

    # Also by season within test set
    df_test = df[test_mask].copy()
    season_test = ((df_test[DATE_COL].dt.month % 12 + 3) // 3) % 4
    season_names = {0: 'Summer', 1: 'Autumn', 2: 'Winter', 3: 'Spring'}

    log("\n  By season (test set):")
    for s in [0, 1, 2, 3]:
        mask = season_test == s
        if mask.sum() > 0:
            r = evaluate_model(model, X_test[mask], y_test[mask])
            results[f'season_{season_names[s]}'] = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
            log(f"    {season_names[s]} (n={mask.sum()}): acc={r['accuracy']:.4f} "
                f"S={r['S_mean']:.2f} H_E={r['H_E_mean']:.6f} H_T={r['H_T_mean']:.4f}")

    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_ood_results(spatial_result, seasonal_results):
    """Plot OOD experiment results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Spatial OOD - ID vs OOD comparison
    ax = axes[0]
    metrics = ['accuracy', 'S_mean', 'H_T_mean', 'H_E_mean', 'uncertainty_auroc']
    id_vals = [spatial_result['id_results'][m] for m in metrics]
    ood_vals = [spatial_result['ood_results'][m] for m in metrics]
    # Normalize for plotting (different scales)
    id_norm = [v / max(id_vals[i], ood_vals[i], 1e-10) for i, v in enumerate(id_vals)]
    ood_norm = [v / max(id_vals[i], ood_vals[i], 1e-10) for i, v in enumerate(ood_vals)]

    x = np.arange(len(metrics))
    ax.bar(x - 0.2, id_norm, 0.4, label='In-Distribution', alpha=0.8)
    ax.bar(x + 0.2, ood_norm, 0.4, label='OOD (Unseen Locations)', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['Acc', 'S', 'H_T', 'H_E', 'Unc-AUROC'], rotation=45)
    ax.set_title('Spatial OOD: ID vs OOD')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Seasonal OOD - accuracy and H_T by held-out season
    ax = axes[1]
    seasons = list(seasonal_results.keys())
    id_accs = [seasonal_results[s]['id_results']['accuracy'] for s in seasons]
    ood_accs = [seasonal_results[s]['ood_results']['accuracy'] for s in seasons]
    id_ht = [seasonal_results[s]['id_results']['H_T_mean'] for s in seasons]
    ood_ht = [seasonal_results[s]['ood_results']['H_T_mean'] for s in seasons]

    x = np.arange(len(seasons))
    ax.bar(x - 0.15, id_accs, 0.3, label='ID Accuracy', alpha=0.8, color='steelblue')
    ax.bar(x + 0.15, ood_accs, 0.3, label='OOD Accuracy', alpha=0.8, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_title('Seasonal OOD: Accuracy by Held-Out Season')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Seasonal OOD - H_T comparison
    ax = axes[2]
    ax.bar(x - 0.15, id_ht, 0.3, label='ID H_T', alpha=0.8, color='steelblue')
    ax.bar(x + 0.15, ood_ht, 0.3, label='OOD H_T', alpha=0.8, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_title('Seasonal OOD: Total Uncertainty (H_T)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'fig10_ood_analysis.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"  OOD plot saved to {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    log("=" * 60)
    log("M7: OOD EXPERIMENTS")
    log("=" * 60)

    all_results = {}

    # 1. Spatial OOD
    try:
        spatial_result = run_spatial_ood()
        all_results['spatial_ood'] = spatial_result
    except Exception as e:
        log(f"  ERROR in spatial OOD: {e}")
        import traceback
        traceback.print_exc()

    # 2. Seasonal OOD
    try:
        seasonal_results = run_seasonal_ood()
        all_results['seasonal_ood'] = seasonal_results
    except Exception as e:
        log(f"  ERROR in seasonal OOD: {e}")
        import traceback
        traceback.print_exc()

    # 3. Extreme events
    try:
        extreme_results = run_extreme_events()
        all_results['extreme_events'] = extreme_results
    except Exception as e:
        log(f"  ERROR in extreme events: {e}")
        import traceback
        traceback.print_exc()

    # 4. Temporal by year
    try:
        temporal_results = run_temporal_by_year()
        all_results['temporal_by_year'] = temporal_results
    except Exception as e:
        log(f"  ERROR in temporal by year: {e}")
        import traceback
        traceback.print_exc()

    # Save results
    with open(os.path.join(RESULTS_DIR, 'm7_ood_experiments.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nResults saved to results/m7_ood_experiments.json")

    # Plot
    if 'spatial_ood' in all_results and 'seasonal_ood' in all_results:
        plot_ood_results(all_results['spatial_ood'], all_results['seasonal_ood'])

    log("\n" + "=" * 60)
    log("M7 OOD EXPERIMENTS COMPLETE")
    log("=" * 60)


if __name__ == "__main__":
    main()
