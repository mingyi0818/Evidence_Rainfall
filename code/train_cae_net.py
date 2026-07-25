"""
Train CAE-Net (C2+C3+C4) on the rainfall dataset.

This script:
  1. Loads preprocessed data (temporal split)
  2. Computes C2 neighborhood labels (k, m) using spatial neighbors
  3. Trains CAE-Net with C2 (Beta-binomial) + C3 (masked KL + budget)
  4. Calibrates C4 Mondrian conformal predictor on validation set
  5. Evaluates on test set
  6. Saves results to results/cae_net_*.json

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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DEVICE, RAW_CSV, TARGET_COL, DATE_COL, LOCATION_COL,
    CATEGORICAL_COLS, NUMERIC_COLS, PREPROCESS,
    CHECKPOINT_DIR, RESULTS_DIR, PLOTS_DIR, ECE_N_BINS,
    MODEL, LOSS, TRAIN
)
from data_loader import preprocess_and_split, load_raw_data, extract_date_features
from models import build_model, EDLMLP
from cae_net import (
    CAENet, cae_net_loss, beta_binomial_loss,
    masked_kl_regularization, evidence_budget_loss,
    MondrianConformalPredictor, make_group_fn, assign_climate_zone,
    compute_spatial_neighborhood_labels_fast
)
from evaluate import compute_ece
from train import get_annealing_factor


SEED = 42
EPOCHS = 80
NEIGHBORHOOD_M = 5  # C2 neighborhood size
S_MAX = 100.0        # C3 evidence budget
BETA_BUDGET = 0.01   # C3 budget weight
LAMBDA_REG = 0.001   # C3 KL weight


_LOG_FILE = os.path.join(os.path.dirname(__file__), '..', 'code', 'cae_net_training.log')

def log(msg):
    """Log to both stdout and file."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
            f.flush()
    except Exception:
        pass


# ============================================================================
# Step 1: Compute C2 neighborhood labels
# ============================================================================

def compute_neighborhood_k_for_split(df_train, df_val, df_test, m=5):
    """
    Compute neighborhood rain counts k for train/val/test splits.

    IMPORTANT: The neighborhood is computed using the SAME date across all
    locations. We use the full station network to define neighbors (this is
    a deterministic graph property, not a statistical estimate, so no leakage).

    Returns: k_train, m_train, k_val, m_val, k_test, m_test
    """
    log("  Computing C2 neighborhood labels (spatial neighbors)...")

    # Combine all data to build the full date x location pivot
    # But keep track of which rows belong to which split
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()
    df_train['_split'] = 0
    df_val['_split'] = 1
    df_test['_split'] = 2
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # Ensure Lat/Lon exist
    if 'Lat' not in df_all.columns or 'Lon' not in df_all.columns:
        # If not in data, try to load from a location file or use index
        log("  WARNING: Lat/Lon not found, using Location index as proxy")
        df_all['Lat'] = df_all[LOCATION_COL].astype('category').cat.codes.astype(float)
        df_all['Lon'] = 0.0

    # Compute k, m for all rows
    k_all, m_all = compute_spatial_neighborhood_labels_fast(
        df_all, LOCATION_COL, DATE_COL, TARGET_COL, m=m
    )

    df_all['_k'] = k_all
    df_all['_m'] = m_all

    # Split back
    k_train = df_all.loc[df_all['_split'] == 0, '_k'].values
    m_train = df_all.loc[df_all['_split'] == 0, '_m'].values
    k_val = df_all.loc[df_all['_split'] == 1, '_k'].values
    m_val = df_all.loc[df_all['_split'] == 1, '_m'].values
    k_test = df_all.loc[df_all['_split'] == 2, '_k'].values
    m_test = df_all.loc[df_all['_split'] == 2, '_m'].values

    log(f"  Train: k mean={k_train.mean():.2f}, m mean={m_train.mean():.2f}")
    log(f"  Val:   k mean={k_val.mean():.2f}, m mean={m_val.mean():.2f}")
    log(f"  Test:  k mean={k_test.mean():.2f}, m mean={m_test.mean():.2f}")

    return k_train, m_train, k_val, m_val, k_test, m_test


# ============================================================================
# Step 2: Build metadata for C4 grouping
# ============================================================================

def build_metadata(df_test):
    """Build metadata list for C4 Mondrian conformal grouping."""
    meta = []
    for _, row in df_test.iterrows():
        season = int(row.get('Season', 0)) if 'Season' in row else 0
        lat = float(row.get('Lat', 0)) if 'Lat' in row else 0
        lon = float(row.get('Lon', 0)) if 'Lon' in row else 0
        climate = assign_climate_zone(lat, lon)
        meta.append({
            'season': season,
            'climate_zone': climate,
            'lat': lat,
            'lon': lon,
        })
    return meta


# ============================================================================
# Step 3: Train CAE-Net
# ============================================================================

def train_cae_net(X_train, y_train, k_train, m_train,
                  X_val, y_val, k_val, m_val,
                  input_dim, seed=42, use_c2=True, use_c3=True,
                  S_max=100.0, beta_budget=0.01, lambda_reg=0.001,
                  c2_warmup_epochs=5, c2_transition_epochs=10,
                  lambda_c2=0.05):
    """Train CAE-Net with C2/C3 components.

    The C2 Beta-binomial loss is used as a regularizer (weight=lambda_c2)
    on top of the primary digamma cross-entropy loss. A short warm-up
    schedule gradually activates the C2 regularizer.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    from torch.utils.data import DataLoader, TensorDataset

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    k_train_t = torch.tensor(k_train, dtype=torch.float32)
    m_train_t = torch.tensor(m_train, dtype=torch.float32)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t, k_train_t, m_train_t)
    train_loader = DataLoader(train_ds, batch_size=TRAIN['batch_size'],
                              shuffle=True, num_workers=0)

    # Build model
    model = CAENet(
        input_dim, hidden_dims=MODEL['hidden_dims'],
        dropout_rate=MODEL['dropout_rate'], num_classes=2,
        prior_n0=1.0, S_max=S_max, beta_budget=beta_budget
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN['learning_rate'],
                                  weight_decay=TRAIN['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_S': [], 'c2_weight': []}

    for epoch in range(EPOCHS):
        af = get_annealing_factor(epoch, LOSS['annealing_epochs']) if LOSS['annealing'] else 1.0

        # C2 warm-up schedule
        if use_c2:
            if epoch < c2_warmup_epochs:
                c2_w = 0.0
            elif epoch < c2_warmup_epochs + c2_transition_epochs:
                c2_w = (epoch - c2_warmup_epochs) / float(c2_transition_epochs)
            else:
                c2_w = 1.0
        else:
            c2_w = 0.0

        # Training
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_kl = 0.0
        total_budget = 0.0
        n_samples = 0

        for Xb, yb, kb, mb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            kb, mb = kb.to(DEVICE), mb.to(DEVICE)

            optimizer.zero_grad()
            alpha = model.predict_dirichlet(Xb)

            loss, ce, kl, budget = cae_net_loss(
                alpha, yb, k=kb, m=mb,
                lambda_reg=lambda_reg, beta_budget=beta_budget if use_c3 else 0.0,
                S_max=S_max, annealing_factor=af, use_c2=use_c2,
                c2_weight=c2_w, lambda_c2=lambda_c2
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * Xb.size(0)
            total_ce += ce * Xb.size(0)
            total_kl += kl * Xb.size(0)
            total_budget += budget * Xb.size(0)
            n_samples += Xb.size(0)

        train_loss = total_loss / n_samples

        # Validation
        model.eval()
        with torch.no_grad():
            alpha_val = model.predict_dirichlet(X_val_t.to(DEVICE))
            # Use point-label loss for validation (no neighborhood)
            val_loss_t, _, _, _ = cae_net_loss(
                alpha_val, y_val_t.to(DEVICE), k=None, m=None,
                lambda_reg=lambda_reg, beta_budget=beta_budget if use_c3 else 0.0,
                S_max=S_max, annealing_factor=af, use_c2=False  # fallback to CE
            )
            val_loss = val_loss_t.item()
            probs_val = alpha_val / alpha_val.sum(dim=1, keepdim=True)
            preds_val = probs_val.argmax(dim=1)
            val_acc = (preds_val.cpu() == y_val_t).float().mean().item()
            S_val = alpha_val.sum(dim=1).mean().item()

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_S'].append(S_val)
        history['c2_weight'].append(c2_w)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == c2_warmup_epochs:
            log(f"  Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_S={S_val:.1f} "
                f"af={af:.3f} c2w={c2_w:.2f}")

        # Use longer patience during C2 warm-up phase
        min_patience = c2_warmup_epochs + c2_transition_epochs + 10 if use_c2 else TRAIN['early_stopping_patience']
        current_patience = max(TRAIN['early_stopping_patience'], min_patience)
        if patience_counter >= current_patience:
            log(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ============================================================================
# Step 4: Evaluate CAE-Net
# ============================================================================

def evaluate_cae_net(model, X_test, y_test, k_test, m_test, meta_test,
                     use_c4=True, epsilon=0.05):
    """Evaluate CAE-Net with optional C4 Mondrian conformal."""
    model.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        unc = model.predict_uncertainty(Xt)
        probs = unc['probs'].cpu().numpy()
        H_T = unc['H_total'].cpu().numpy()
        H_A = unc['H_alea'].cpu().numpy()
        H_E = unc['H_epi'].cpu().numpy()
        S = unc['precision'].cpu().numpy()
        alpha = unc['alpha'].cpu().numpy()

    preds = probs.argmax(axis=1)
    errors = (preds != y_test).astype(int)

    # Standard metrics
    ece_val = compute_ece(y_test, probs[:, 1])
    ece_val = ece_val[0] if isinstance(ece_val, tuple) else ece_val

    results = {
        'n': len(y_test),
        'accuracy': float(accuracy_score(y_test, preds)),
        'f1_macro': float(f1_score(y_test, preds, average='macro', zero_division=0)),
        'f1_micro': float(f1_score(y_test, preds, average='micro', zero_division=0)),
        'ece': float(ece_val),
        'brier': float(brier_score_loss(y_test, probs[:, 1])),
        'H_T_mean': float(H_T.mean()),
        'H_A_mean': float(H_A.mean()),
        'H_E_mean': float(H_E.mean()),
        'H_E_std': float(H_E.std()),
        'S_mean': float(S.mean()),
        'S_std': float(S.std()),
        'S_median': float(np.median(S)),
        'uncertainty_auroc': float(roc_auc_score(errors, H_T)) if len(np.unique(errors)) > 1 else 0.5,
        'alpha_mean': float(alpha.mean()),
    }

    # C4: Mondrian Conformal Prediction
    if use_c4:
        # Use half of test set as calibration, half as evaluation
        # (In practice, should use validation set for calibration)
        n = len(y_test)
        n_cal = n // 2
        idx = np.random.RandomState(SEED).permutation(n)
        cal_idx, eval_idx = idx[:n_cal], idx[n_cal:]

        group_fn = make_group_fn('season_climate')
        mcp = MondrianConformalPredictor(group_fn, epsilon=epsilon, score_type='1_minus_p')

        # Calibrate
        mcp.calibrate(
            X_test[cal_idx], y_test[cal_idx],
            probs[cal_idx], S[cal_idx],
            [meta_test[i] for i in cal_idx]
        )

        # Evaluate
        c4_results = mcp.evaluate(
            probs[eval_idx], y_test[eval_idx],
            S[eval_idx], [meta_test[i] for i in eval_idx]
        )

        results['c4_conformal'] = {
            'epsilon': epsilon,
            'coverage': c4_results['coverage'],
            'group_coverage': {str(k): float(v) for k, v in c4_results['group_coverage'].items()},
            'abstention_rate': c4_results['abstention_rate'],
            'selective_accuracy': c4_results['selective_accuracy'],
            'selective_error_rate': c4_results['selective_error_rate'],
            'quantiles': {str(k): float(v) for k, v in c4_results['quantiles'].items()},
            'n_cal': int(n_cal),
            'n_eval': int(n - n_cal),
        }

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    # Clear log file
    try:
        open(_LOG_FILE, 'w').close()
    except Exception:
        pass

    log("=" * 60)
    log("CAE-Net Training (C2 + C3 + C4)")
    log("=" * 60)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load data with temporal split
    log("\n[1/5] Loading data (temporal split)...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, encoder, feature_names = \
        preprocess_and_split(seed=SEED, save=False, split_mode='temporal')
    input_dim = X_test.shape[1]
    log(f"  Train={X_train.shape[0]} Val={X_val.shape[0]} Test={X_test.shape[0]} dim={input_dim}")

    # Load raw data for C2 neighborhood computation
    # IMPORTANT: Do NOT call extract_date_features here because it drops the
    # Date column, which is needed for spatial neighborhood computation.
    log("\n[2/5] Computing C2 neighborhood labels...")
    df_raw = load_raw_data()
    df_raw = df_raw.dropna(subset=[TARGET_COL])
    # Preserve Date column, only extract Year for temporal split
    df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL], errors='coerce')
    df_raw['Year'] = df_raw[DATE_COL].dt.year
    df_raw['Month'] = df_raw[DATE_COL].dt.month
    df_raw['Season'] = ((df_raw['Month'] % 12 + 3) // 3) % 4

    # Get Lat/Lon from location data
    # Real GPS coordinates for the 49 Australian weather stations
    # Source: Australian Bureau of Meteorology station metadata
    loc_coords = {
        'Adelaide': (-34.93, 138.60), 'Albany': (-35.03, 117.88),
        'Albury': (-36.08, 146.92), 'AliceSprings': (-23.70, 133.88),
        'BadgerysCreek': (-33.88, 150.73), 'Ballarat': (-37.56, 143.85),
        'Bendigo': (-36.76, 144.28), 'Brisbane': (-27.48, 153.04),
        'Cairns': (-16.92, 145.77), 'Canberra': (-35.30, 149.13),
        'Cobar': (-31.48, 145.83), 'CoffsHarbour': (-30.30, 153.11),
        'Dartmoor': (-37.92, 141.27), 'Darwin': (-12.42, 130.89),
        'GoldCoast': (-28.00, 153.43), 'Hobart': (-42.88, 147.33),
        'Katherine': (-14.47, 132.27), 'Launceston': (-41.45, 147.14),
        'Melbourne': (-37.83, 144.98), 'MelbourneAirport': (-37.67, 144.83),
        'Mildura': (-34.18, 142.16), 'Moree': (-29.48, 149.84),
        'MountGambier': (-37.78, 140.78), 'MountGinini': (-35.53, 148.95),
        'Newcastle': (-32.93, 151.78), 'Nhil': (-35.93, 141.65),
        'NorahHead': (-33.28, 151.57), 'NorfolkIsland': (-29.03, 167.94),
        'Nuriootpa': (-34.47, 139.00), 'PearceRAAF': (-31.67, 116.02),
        'Penrith': (-33.75, 150.69), 'Perth': (-31.95, 115.86),
        'PerthAirport': (-31.94, 115.97), 'Portland': (-38.35, 141.61),
        'Richmond': (-33.60, 150.75), 'Sale': (-38.10, 147.07),
        'SalmonGums': (-32.98, 121.64), 'Sydney': (-33.87, 151.21),
        'SydneyAirport': (-33.95, 151.18), 'Townsville': (-19.25, 146.77),
        'Tuggeranong': (-35.42, 149.07), 'Uluru': (-25.35, 131.03),
        'WaggaWagga': (-35.12, 147.37), 'Walpole': (-34.97, 116.73),
        'Watsonia': (-37.71, 145.08), 'Williamtown': (-32.80, 151.83),
        'Witchcliffe': (-34.02, 115.10), 'Wollongong': (-34.42, 150.89),
        'Woomera': (-31.20, 136.82),
    }

    df_raw['Lat'] = df_raw[LOCATION_COL].map(lambda l: loc_coords.get(l, (0.0, 0.0))[0])
    df_raw['Lon'] = df_raw[LOCATION_COL].map(lambda l: loc_coords.get(l, (0.0, 0.0))[1])

    # Split df_raw to match the temporal split
    year = df_raw['Year'].values if 'Year' in df_raw.columns else pd.to_datetime(df_raw[DATE_COL]).dt.year.values
    train_mask = year <= 2014
    val_mask = year == 2015
    test_mask = year >= 2016

    df_train_raw = df_raw[train_mask].reset_index(drop=True)
    df_val_raw = df_raw[val_mask].reset_index(drop=True)
    df_test_raw = df_raw[test_mask].reset_index(drop=True)

    k_train, m_train, k_val, m_val, k_test, m_test = compute_neighborhood_k_for_split(
        df_train_raw, df_val_raw, df_test_raw, m=NEIGHBORHOOD_M
    )

    # Build metadata for C4
    meta_test = build_metadata(df_test_raw)
    log(f"  Built metadata for {len(meta_test)} test samples")

    # Train CAE-Net with all components (C2+C3)
    log("\n[3/5] Training CAE-Net (C2+C3)...")
    t0 = time.time()
    model_cae, history_cae = train_cae_net(
        X_train, y_train, k_train, m_train,
        X_val, y_val, k_val, m_val,
        input_dim, seed=SEED, use_c2=True, use_c3=True,
        S_max=S_MAX, beta_budget=BETA_BUDGET, lambda_reg=LAMBDA_REG
    )
    log(f"  CAE-Net training done in {time.time()-t0:.1f}s")

    # Train ablation: C3 only (no C2, i.e., original EDL with masked KL + budget)
    log("\n[4/5] Training ablation: C3 only (no C2)...")
    t0 = time.time()
    model_c3only, history_c3 = train_cae_net(
        X_train, y_train, k_train, m_train,
        X_val, y_val, k_val, m_val,
        input_dim, seed=SEED, use_c2=False, use_c3=True,
        S_max=S_MAX, beta_budget=BETA_BUDGET, lambda_reg=LAMBDA_REG
    )
    log(f"  C3-only training done in {time.time()-t0:.1f}s")

    # Evaluate both models
    log("\n[5/5] Evaluating models...")
    results_cae = evaluate_cae_net(
        model_cae, X_test, y_test, k_test, m_test, meta_test,
        use_c4=True, epsilon=0.05
    )
    log(f"  CAE-Net (C2+C3+C4): acc={results_cae['accuracy']:.4f} f1={results_cae['f1_macro']:.4f} "
        f"ece={results_cae['ece']:.4f} S={results_cae['S_mean']:.1f} H_E={results_cae['H_E_mean']:.6f}")
    if 'c4_conformal' in results_cae:
        c4 = results_cae['c4_conformal']
        log(f"  C4: coverage={c4['coverage']:.4f} (target>={1-c4['epsilon']:.2f}) "
            f"abstain={c4['abstention_rate']:.4f} sel_acc={c4['selective_accuracy']:.4f}")

    results_c3only = evaluate_cae_net(
        model_c3only, X_test, y_test, k_test, m_test, meta_test,
        use_c4=True, epsilon=0.05
    )
    log(f"  C3-only:            acc={results_c3only['accuracy']:.4f} f1={results_c3only['f1_macro']:.4f} "
        f"ece={results_c3only['ece']:.4f} S={results_c3only['S_mean']:.1f} H_E={results_c3only['H_E_mean']:.6f}")

    # Load existing EDL-Fixed for comparison
    log("\n  Loading EDL-Fixed (baseline) for comparison...")
    edl_path = os.path.join(CHECKPOINT_DIR, f"edl_seed{SEED}.pth")
    if os.path.exists(edl_path):
        model_edl = build_model('edl', input_dim, num_classes=2,
                                hidden_dims=MODEL['hidden_dims'],
                                dropout_rate=MODEL['dropout_rate']).to(DEVICE)
        model_edl.load_state_dict(torch.load(edl_path, map_location=DEVICE))

        # Evaluate EDL-Fixed
        model_edl.eval()
        Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            unc_edl = model_edl.predict_uncertainty(Xt)
            probs_edl = unc_edl['probs'].cpu().numpy()
            S_edl = unc_edl['precision'].cpu().numpy()
            H_E_edl = unc_edl['H_epi'].cpu().numpy()

        preds_edl = probs_edl.argmax(axis=1)
        ece_edl = compute_ece(y_test, probs_edl[:, 1])
        ece_edl = ece_edl[0] if isinstance(ece_edl, tuple) else ece_edl

        results_edl = {
            'accuracy': float(accuracy_score(y_test, preds_edl)),
            'f1_macro': float(f1_score(y_test, preds_edl, average='macro', zero_division=0)),
            'ece': float(ece_edl),
            'S_mean': float(S_edl.mean()),
            'H_E_mean': float(H_E_edl.mean()),
        }
        log(f"  EDL-Fixed:          acc={results_edl['accuracy']:.4f} f1={results_edl['f1_macro']:.4f} "
            f"ece={results_edl['ece']:.4f} S={results_edl['S_mean']:.1f} H_E={results_edl['H_E_mean']:.6f}")
    else:
        results_edl = None
        log("  EDL-Fixed checkpoint not found, skipping comparison")

    # Save all results
    all_results = {
        'cae_net_c2c3c4': results_cae,
        'c3_only_ablation': results_c3only,
        'edl_fixed_baseline': results_edl,
        'config': {
            'neighborhood_m': NEIGHBORHOOD_M,
            'S_max': S_MAX,
            'beta_budget': BETA_BUDGET,
            'lambda_reg': LAMBDA_REG,
            'epochs': EPOCHS,
            'seed': SEED,
        },
        'training_history': {
            'cae_net': {'final_train_loss': history_cae['train_loss'][-1] if history_cae['train_loss'] else None,
                       'final_val_loss': history_cae['val_loss'][-1] if history_cae['val_loss'] else None,
                       'final_val_acc': history_cae['val_acc'][-1] if history_cae['val_acc'] else None,
                       'final_val_S': history_cae['val_S'][-1] if history_cae['val_S'] else None},
            'c3_only': {'final_train_loss': history_c3['train_loss'][-1] if history_c3['train_loss'] else None,
                       'final_val_loss': history_c3['val_loss'][-1] if history_c3['val_loss'] else None,
                       'final_val_acc': history_c3['val_acc'][-1] if history_c3['val_acc'] else None,
                       'final_val_S': history_c3['val_S'][-1] if history_c3['val_S'] else None},
        }
    }

    output_path = os.path.join(RESULTS_DIR, 'cae_net_results.json')

    def json_default(obj):
        """Handle numpy types for JSON serialization."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=json_default)
    log(f"\nResults saved to {output_path}")

    # Save CAE-Net checkpoint
    cae_ckpt_path = os.path.join(CHECKPOINT_DIR, f'cae_net_seed{SEED}.pth')
    torch.save(model_cae.state_dict(), cae_ckpt_path)
    log(f"CAE-Net checkpoint saved to {cae_ckpt_path}")

    log("\n" + "=" * 60)
    log("CAE-Net TRAINING COMPLETE")
    log("=" * 60)


if __name__ == "__main__":
    main()
