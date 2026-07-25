"""
Ablation / Sensitivity / Robustness experiments for EDL-UQ (direction 17).
All numbers are produced on the temporal split (S1: 2007-2014 train / 2015 val / 2016-2017 test).
Seed 42 only for ablation/sensitivity/robustness (single-seed diagnostic experiments).
Results saved as CSV/JSON in results/ directory for full traceability.
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

sys.path.insert(0, os.path.dirname(__file__))
from config import DEVICE, MODEL, TRAIN, RESULTS_DIR
from data_loader import preprocess_and_split
from models import EDLMLP, MCDropoutMLP, BayesianMLP
from train import edl_loss
from simple_experiment import (
    load_to_gpu, get_batches_gpu, compute_ece, evaluate_predictions,
    train_edl, train_torch_model, eval_edl, eval_softmax, eval_mcdropout, eval_bnn
)

LOG_PATH = os.path.join(os.path.dirname(__file__), 'ablation_sens_robust.log')
LOG_FILE = open(LOG_PATH, 'w', encoding='utf-8')
def log(msg):
    print(msg, flush=True)
    LOG_FILE.write(msg + '\n'); LOG_FILE.flush()

SEED = 42
EPOCHS = 20


def run_ablation(X_train, y_train, X_val, y_val, X_test, y_test, input_dim):
    """Component-level ablation: Full / w/o KL / w/o Annealing / Softmax / MSE.

    All variants share the SAME backbone ([128,64,32] + BN + ReLU + Dropout 0.3)
    for controlled comparison.
    """
    log("\n" + "="*60 + "\nABLATION STUDY (seed=42, temporal split)\n" + "="*60)
    rows = []

    # Full EDL-UQ (already trained, re-train for self-containedness)
    log("\n[A1] Full EDL-UQ (KL + annealing + cross-entropy evidence)")
    torch.manual_seed(SEED); np.random.seed(SEED)
    m_full = train_edl(X_train, y_train, X_val, y_val, input_dim, seed=SEED, epochs=EPOCHS)
    r_full = eval_edl(m_full, X_test, y_test)
    rows.append({'variant': 'Full_EDL_UQ', **r_full})
    log(f"  acc={r_full['accuracy']:.4f} f1={r_full['f1_macro']:.4f} ece={r_full['ece']:.4f} unc_auroc={r_full['uncertainty_auroc']:.4f}")

    # A2: w/o KL Regularization (lambda_reg = 0)
    log("\n[A2] w/o KL Regularization (lambda_reg=0)")
    torch.manual_seed(SEED); np.random.seed(SEED)
    m_no_kl = _train_edl_custom(X_train, y_train, X_val, y_val, input_dim, lambda_reg=0.0, epochs=EPOCHS)
    r_no_kl = eval_edl(m_no_kl, X_test, y_test)
    rows.append({'variant': 'wo_KL_Regularization', **r_no_kl})
    log(f"  acc={r_no_kl['accuracy']:.4f} f1={r_no_kl['f1_macro']:.4f} ece={r_no_kl['ece']:.4f}")

    # A3: w/o Annealing (annealing factor fixed at 1.0)
    log("\n[A3] w/o Annealing (annealing_factor=1.0 from start)")
    torch.manual_seed(SEED); np.random.seed(SEED)
    m_no_ann = _train_edl_custom(X_train, y_train, X_val, y_val, input_dim, lambda_reg=0.001,
                                 annealing_epochs=0, epochs=EPOCHS)
    r_no_ann = eval_edl(m_no_ann, X_test, y_test)
    rows.append({'variant': 'wo_Annealing', **r_no_ann})
    log(f"  acc={r_no_ann['accuracy']:.4f} f1={r_no_ann['f1_macro']:.4f} ece={r_no_ann['ece']:.4f}")

    # A4: Softmax Baseline (same backbone, no evidential head)
    log("\n[A4] Softmax Baseline (same backbone, no evidential head)")
    torch.manual_seed(SEED); np.random.seed(SEED)
    m_sm = _train_softmax_same_backbone(X_train, y_train, X_val, y_val, input_dim, epochs=EPOCHS)
    r_sm = eval_softmax(m_sm, X_test, y_test)
    rows.append({'variant': 'Softmax_Baseline', **r_sm})
    log(f"  acc={r_sm['accuracy']:.4f} f1={r_sm['f1_macro']:.4f} ece={r_sm['ece']:.4f}")

    # A5: Softmax + Temperature Scaling (calibration baseline, same backbone)
    log("\n[A5] Softmax + Temperature Scaling (Guo et al. 2017)")
    torch.manual_seed(SEED); np.random.seed(SEED)
    T_opt = _fit_temperature(m_sm, X_val, y_val)
    r_sm_ts = _eval_softmax_ts(m_sm, X_test, y_test, T_opt)
    rows.append({'variant': 'Softmax_TempScaling', **r_sm_ts, 'T_optimal': T_opt})
    log(f"  acc={r_sm_ts['accuracy']:.4f} f1={r_sm_ts['f1_macro']:.4f} ece={r_sm_ts['ece']:.4f} T={T_opt:.3f}")

    # A6: MSE Evidence Loss
    log("\n[A6] MSE Evidence Loss (instead of cross-entropy)")
    torch.manual_seed(SEED); np.random.seed(SEED)
    m_mse = _train_edl_custom(X_train, y_train, X_val, y_val, input_dim, lambda_reg=0.001,
                              loss_type='mse', epochs=EPOCHS)
    r_mse = eval_edl(m_mse, X_test, y_test)
    rows.append({'variant': 'MSE_Evidence_Loss', **r_mse})
    log(f"  acc={r_mse['accuracy']:.4f} f1={r_mse['f1_macro']:.4f} ece={r_mse['ece']:.4f}")

    # A7: EDL-C1 (climatology-anchored prior at inference)
    log("\n[A7] EDL-C1 (climatology-anchored prior at inference)")
    pi = y_train.mean(); n0 = 10.0
    prior = torch.tensor([n0*(1-pi), n0*pi], dtype=torch.float32).to(DEVICE)
    m_c1 = m_full  # same trained model, different inference prior
    m_c1.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        alpha = m_c1.predict_dirichlet(Xt, alpha_prior=prior.unsqueeze(0).expand(len(Xt), -1))
        probs = (alpha / alpha.sum(dim=1, keepdim=True)).cpu().numpy()
    H = -(probs * np.log(probs + 1e-10)).sum(axis=1)
    preds = probs.argmax(axis=1)
    r_c1 = evaluate_predictions(preds, probs, H, y_test)
    rows.append({'variant': 'EDL_C1_Climatology_Prior', **r_c1})
    log(f"  acc={r_c1['accuracy']:.4f} f1={r_c1['f1_macro']:.4f} ece={r_c1['ece']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'ablation_results_v2.csv'), index=False)
    log(f"\nAblation results saved to results/ablation_results_v2.csv")
    return df


def _train_edl_custom(X_train, y_train, X_val, y_val, input_dim, lambda_reg=0.001,
                      annealing_epochs=50, loss_type='cross_entropy', epochs=20, lr=1e-3):
    """Train EDL with custom hyperparameters (for ablation)."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = EDLMLP(input_dim, hidden_dims=MODEL['hidden_dims'], dropout_rate=MODEL['dropout_rate']).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val = float('inf'); best_state = None
    bs = 256
    Xtr, ytr = load_to_gpu(X_train, y_train)
    Xva, yva = load_to_gpu(X_val, y_val)
    for epoch in range(epochs):
        af = min(1.0, epoch / annealing_epochs) if annealing_epochs > 0 else 1.0
        model.train()
        for Xb, yb in get_batches_gpu(Xtr, ytr, bs):
            opt.zero_grad()
            alpha = model.predict_dirichlet(Xb)
            loss, ce, kl = edl_loss(alpha, yb, lambda_reg=lambda_reg,
                                     annealing_factor=af, loss_type=loss_type)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vtotal = 0; vn = 0
        with torch.no_grad():
            for Xb, yb in get_batches_gpu(Xva, yva, bs, shuffle=False):
                alpha = model.predict_dirichlet(Xb)
                loss, _, _ = edl_loss(alpha, yb, lambda_reg=lambda_reg,
                                       annealing_factor=af, loss_type=loss_type)
                vtotal += loss.item() * Xb.size(0); vn += Xb.size(0)
        vloss = vtotal / vn
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"  EDL({loss_type},lam={lambda_reg},ann={annealing_epochs}) epoch {epoch+1}/{epochs} val={vloss:.4f} af={af:.3f}")
    if best_state:
        model.load_state_dict(best_state)
    del Xtr, ytr, Xva, yva
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return model


class SoftmaxSameBackbone(nn.Module):
    """Softmax MLP with the SAME backbone as EDLMLP (satisfies M14)."""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _train_softmax_same_backbone(X_train, y_train, X_val, y_val, input_dim, epochs=20, lr=1e-3):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = SoftmaxSameBackbone(input_dim, hidden_dims=MODEL['hidden_dims'],
                                 dropout_rate=MODEL['dropout_rate']).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss()
    best_val = float('inf'); best_state = None
    bs = 256
    Xtr, ytr = load_to_gpu(X_train, y_train)
    Xva, yva = load_to_gpu(X_val, y_val)
    for epoch in range(epochs):
        model.train()
        for Xb, yb in get_batches_gpu(Xtr, ytr, bs):
            opt.zero_grad()
            logits = model(Xb)
            loss = crit(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vtotal = 0; vn = 0
        with torch.no_grad():
            for Xb, yb in get_batches_gpu(Xva, yva, bs, shuffle=False):
                logits = model(Xb)
                loss = crit(logits, yb)
                vtotal += loss.item() * Xb.size(0); vn += Xb.size(0)
        vloss = vtotal / vn
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"  Softmax epoch {epoch+1}/{epochs} val={vloss:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    del Xtr, ytr, Xva, yva
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return model


def _fit_temperature(model, X_val, y_val):
    """Fit temperature T for temperature scaling (Guo et al. 2017) on validation set."""
    model.eval()
    Xv = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(Xv)
    T = torch.ones(1, requires_grad=True, device=DEVICE)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=100)
    yv = torch.tensor(y_val, dtype=torch.long).to(DEVICE)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T, yv)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.item())


def _eval_softmax_ts(model, X_test, y_test, T):
    model.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(Xt) / T
        probs = F.softmax(logits, dim=1).cpu().numpy()
    H = -(probs * np.log(probs + 1e-10)).sum(axis=1)
    preds = probs.argmax(axis=1)
    return evaluate_predictions(preds, probs, H, y_test)


def run_sensitivity(X_train, y_train, X_val, y_val, X_test, y_test, input_dim):
    """Parameter sensitivity analysis with elasticity coefficient.

    Elasticity = |Δy/y| / |Δx/x|, computed as percentage change in F1 over percentage change in parameter.
    Levels: high (>0.5), medium (0.2-0.5), low (<0.2).
    All tuning is on VALIDATION set (fixes M15 - test-set tuning leakage).
    """
    log("\n" + "="*60 + "\nSENSITIVITY ANALYSIS (seed=42, validation-set tuning)\n" + "="*60)
    rows = []

    # 1. lambda_reg in {0.0, 0.001, 0.01, 0.1}
    log("\n[S1] lambda_reg sensitivity")
    for lam in [0.0, 0.001, 0.01, 0.1]:
        torch.manual_seed(SEED); np.random.seed(SEED)
        m = _train_edl_custom(X_train, y_train, X_val, y_val, input_dim,
                              lambda_reg=lam, epochs=EPOCHS)
        # Tune on val, report on test
        r_val = eval_edl(m, X_val, y_val)
        r_test = eval_edl(m, X_test, y_test)
        rows.append({
            'parameter': 'lambda_reg', 'value': lam,
            'val_f1_macro': r_val['f1_macro'], 'test_f1_macro': r_test['f1_macro'],
            'test_accuracy': r_test['accuracy'], 'test_ece': r_test['ece'],
            'test_unc_auroc': r_test['uncertainty_auroc'],
        })
        log(f"  lam={lam}: val_f1={r_val['f1_macro']:.4f} test_f1={r_test['f1_macro']:.4f}")

    # 2. dropout in {0.0, 0.2, 0.3, 0.4, 0.5}
    log("\n[S2] dropout sensitivity")
    for dr in [0.0, 0.2, 0.3, 0.4, 0.5]:
        torch.manual_seed(SEED); np.random.seed(SEED)
        m = _train_edl_custom_dropout(X_train, y_train, X_val, y_val, input_dim,
                                       dropout_rate=dr, epochs=EPOCHS)
        r_val = eval_edl(m, X_val, y_val)
        r_test = eval_edl(m, X_test, y_test)
        rows.append({
            'parameter': 'dropout', 'value': dr,
            'val_f1_macro': r_val['f1_macro'], 'test_f1_macro': r_test['f1_macro'],
            'test_accuracy': r_test['accuracy'], 'test_ece': r_test['ece'],
            'test_unc_auroc': r_test['uncertainty_auroc'],
        })
        log(f"  dropout={dr}: val_f1={r_val['f1_macro']:.4f} test_f1={r_test['f1_macro']:.4f}")

    # 3. learning rate in {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}
    log("\n[S3] learning rate sensitivity")
    for lr in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
        torch.manual_seed(SEED); np.random.seed(SEED)
        m = _train_edl_custom(X_train, y_train, X_val, y_val, input_dim,
                              lambda_reg=0.001, epochs=EPOCHS, lr=lr)
        r_val = eval_edl(m, X_val, y_val)
        r_test = eval_edl(m, X_test, y_test)
        rows.append({
            'parameter': 'learning_rate', 'value': lr,
            'val_f1_macro': r_val['f1_macro'], 'test_f1_macro': r_test['f1_macro'],
            'test_accuracy': r_test['accuracy'], 'test_ece': r_test['ece'],
            'test_unc_auroc': r_test['uncertainty_auroc'],
        })
        log(f"  lr={lr}: val_f1={r_val['f1_macro']:.4f} test_f1={r_test['f1_macro']:.4f}")

    df = pd.DataFrame(rows)

    # Compute elasticity per parameter
    log("\n[Elasticity Summary]")
    summary_rows = []
    for p in df['parameter'].unique():
        sub = df[df['parameter'] == p].copy()
        x = sub['value'].values.astype(float)
        y = sub['test_f1_macro'].values.astype(float)
        # Elasticity = (max(y) - min(y)) / mean(y) / ((max(x) - min(x)) / mean(x))
        y_range = (y.max() - y.min()) / max(y.mean(), 1e-9)
        x_range = (x.max() - x.min()) / max(x.mean(), 1e-9)
        elast = y_range / x_range if x_range > 0 else 0.0
        level = 'High' if elast > 0.5 else ('Medium' if elast > 0.2 else 'Low')
        best_idx = sub['val_f1_macro'].idxmax()
        best_val = sub.loc[best_idx, 'value']
        best_f1 = sub.loc[best_idx, 'test_f1_macro']
        summary_rows.append({
            'parameter': p,
            'range': f"[{x.min()}, {x.max()}]",
            'best_value_val': best_val,
            'best_test_f1_macro': best_f1,
            'elasticity': round(elast, 4),
            'sensitivity_level': level,
        })
        log(f"  {p}: range=[{x.min()},{x.max()}] best_val={best_val} best_test_f1={best_f1:.4f} elasticity={elast:.4f} ({level})")

    df_summary = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'sensitivity_results_v2.csv'), index=False)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 'sensitivity_summary_v2.csv'), index=False)
    log(f"\nSensitivity results saved to results/sensitivity_results_v2.csv and sensitivity_summary_v2.csv")
    return df, df_summary


def _train_edl_custom_dropout(X_train, y_train, X_val, y_val, input_dim,
                               dropout_rate=0.3, epochs=20, lr=1e-3):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = EDLMLP(input_dim, hidden_dims=MODEL['hidden_dims'], dropout_rate=dropout_rate).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val = float('inf'); best_state = None
    bs = 256
    Xtr, ytr = load_to_gpu(X_train, y_train)
    Xva, yva = load_to_gpu(X_val, y_val)
    for epoch in range(epochs):
        af = min(1.0, epoch / 50)
        model.train()
        for Xb, yb in get_batches_gpu(Xtr, ytr, bs):
            opt.zero_grad()
            alpha = model.predict_dirichlet(Xb)
            loss, _, _ = edl_loss(alpha, yb, lambda_reg=0.001, annealing_factor=af)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vtotal = 0; vn = 0
        with torch.no_grad():
            for Xb, yb in get_batches_gpu(Xva, yva, bs, shuffle=False):
                alpha = model.predict_dirichlet(Xb)
                loss, _, _ = edl_loss(alpha, yb, lambda_reg=0.001, annealing_factor=af)
                vtotal += loss.item() * Xb.size(0); vn += Xb.size(0)
        vloss = vtotal / vn
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    del Xtr, ytr, Xva, yva
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return model


def run_robustness(X_train, y_train, X_val, y_val, X_test, y_test, input_dim):
    """Robustness analysis: Gaussian noise + feature missing.

    Also reports S (total evidence), H_epi (epistemic uncertainty), Unc-AUROC
    to show uncertainty degradation under shift.
    """
    log("\n" + "="*60 + "\nROBUSTNESS ANALYSIS (seed=42)\n" + "="*60)
    rows = []

    # Train model once on clean data
    torch.manual_seed(SEED); np.random.seed(SEED)
    m = train_edl(X_train, y_train, X_val, y_val, input_dim, seed=SEED, epochs=EPOCHS)

    rng = np.random.RandomState(SEED)

    # Clean baseline
    r_clean = eval_edl(m, X_test, y_test)
    rows.append({'perturbation': 'Clean', 'level': 0.0, **r_clean})
    log(f"\n  Clean: acc={r_clean['accuracy']:.4f} ece={r_clean['ece']:.4f} unc_auroc={r_clean['uncertainty_auroc']:.4f}")

    # Gaussian noise (additive, scale relative to feature std)
    for noise_pct in [0.01, 0.05, 0.10, 0.15]:
        np.random.seed(SEED)
        # noise scaled per-feature by std of training data
        X_test_noisy = X_test + rng.normal(0, noise_pct * X_train.std(axis=0), X_test.shape).astype(np.float32)
        r = eval_edl(m, X_test_noisy, y_test)
        rows.append({'perturbation': 'Gaussian_Noise', 'level': noise_pct, **r})
        log(f"  Noise {noise_pct:.0%}: acc={r['accuracy']:.4f} ece={r['ece']:.4f} unc_auroc={r['uncertainty_auroc']:.4f}")

    # Feature missing (randomly zero out features)
    for missing_pct in [0.05, 0.10, 0.20, 0.30]:
        np.random.seed(SEED)
        mask = rng.rand(*X_test.shape) < missing_pct
        X_test_missing = X_test.copy()
        X_test_missing[mask] = 0.0
        r = eval_edl(m, X_test_missing, y_test)
        rows.append({'perturbation': 'Feature_Missing', 'level': missing_pct, **r})
        log(f"  Missing {missing_pct:.0%}: acc={r['accuracy']:.4f} ece={r['ece']:.4f} unc_auroc={r['uncertainty_auroc']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'robustness_results_v2.csv'), index=False)
    log(f"\nRobustness results saved to results/robustness_results_v2.csv")
    return df


def run_selective_prediction(X_train, y_train, X_val, y_val, X_test, y_test, input_dim):
    """Selective prediction analysis with proper baselines (addresses M6).

    Reports AURC (Area Under Risk-Coverage Curve) and Excess-AURC (vs oracle).
    Compares EDL-UQ with MCDropout, LSTM, GRU, Softmax-MSP.
    """
    log("\n" + "="*60 + "\nSELECTIVE PREDICTION ANALYSIS (seed=42)\n" + "="*60)
    rows = []
    summary = {}

    # Train all required models
    torch.manual_seed(SEED); np.random.seed(SEED)
    edl_model = train_edl(X_train, y_train, X_val, y_val, input_dim, seed=SEED, epochs=EPOCHS)
    torch.manual_seed(SEED); np.random.seed(SEED)
    lstm_model = train_torch_model('lstm', X_train, y_train, X_val, y_val, input_dim, seed=SEED, epochs=EPOCHS)
    torch.manual_seed(SEED); np.random.seed(SEED)
    gru_model = train_torch_model('gru', X_train, y_train, X_val, y_val, input_dim, seed=SEED, epochs=EPOCHS)
    torch.manual_seed(SEED); np.random.seed(SEED)
    mc_model = train_torch_model('mcdropout', X_train, y_train, X_val, y_val, input_dim, seed=SEED, epochs=EPOCHS)
    torch.manual_seed(SEED); np.random.seed(SEED)
    sm_model = _train_softmax_same_backbone(X_train, y_train, X_val, y_val, input_dim, epochs=EPOCHS)

    # Get uncertainties and errors for each method
    methods = {}
    # EDL-UQ: total predictive entropy H_T
    methods['EDL_UQ'] = _get_edl_uncertainty(edl_model, X_test, y_test)
    # MCDropout: predictive entropy of mean probs
    methods['MCDropout'] = _get_softmax_uncertainty(mc_model, X_test, y_test, mc_dropout=True)
    # LSTM: predictive entropy
    methods['LSTM'] = _get_softmax_uncertainty(lstm_model, X_test, y_test)
    # GRU: predictive entropy
    methods['GRU'] = _get_softmax_uncertainty(gru_model, X_test, y_test)
    # Softmax-MSP: 1 - max_prob (Maximum Softmax Probability, Hendrycks & Gimpel 2017)
    methods['Softmax_MSP'] = _get_softmax_uncertainty(sm_model, X_test, y_test, use_msp=True)
    # Random baseline (lower bound)
    methods['Random'] = {'uncertainties': rng_uniform(len(y_test), seed=SEED), 'errors': methods['EDL_UQ']['errors']}

    n = len(y_test)
    oracle_acc = 1.0  # best possible: reject all errors first
    for method_name, info in methods.items():
        uncertainties = info['uncertainties']
        errors = info['errors']
        preds = info.get('preds', None)
        if preds is None:
            # Use EDL-UQ's predictions for fair comparison of uncertainty quality
            preds = methods['EDL_UQ']['preds']
        # Sort by uncertainty (descending = most uncertain first)
        order = np.argsort(-uncertainties)
        sorted_errors = errors[order]
        sorted_correct = 1 - sorted_errors
        # Coverage from 0 to 1, risk = error rate among retained
        n_test = len(y_test)
        coverages = np.linspace(0.05, 1.0, 20)
        risks = []
        for cov in coverages:
            n_retain = int(cov * n_test)
            # Retain the n_retain most CONFIDENT (least uncertain) samples
            retained_correct = sorted_correct[-n_retain:].sum()
            risk = 1.0 - retained_correct / n_retain
            risks.append(risk)
        risks = np.array(risks)
        aurc = np.trapz(risks, coverages)
        # Oracle AURC (perfect uncertainty: all errors first)
        oracle_risks = []
        n_errors = errors.sum()
        for cov in coverages:
            n_retain = int(cov * n_test)
            # If we can reject all errors: retain only correct samples
            if n_retain <= (n_test - n_errors):
                oracle_risk = 0.0
            else:
                oracle_risk = (n_retain - (n_test - n_errors)) / n_retain
            oracle_risks.append(oracle_risk)
        oracle_risks = np.array(oracle_risks)
        oracle_aurc = np.trapz(oracle_risks, coverages)
        excess_aurc = aurc - oracle_aurc
        # Random AURC (uncertainty independent of error)
        random_aurc = (1 - methods['EDL_UQ']['base_acc']) * 1.0  # average risk = base error rate over [0,1]

        summary[method_name] = {
            'AURC': float(aurc),
            'oracle_AURC': float(oracle_aurc),
            'excess_AURC': float(excess_aurc),
            'random_AURC': float(random_aurc),
            'accuracy_at_20pct_reject': float(1.0 - risks[20//5 - 1]) if len(risks) >= 4 else 0.0,
            'accuracy_at_50pct_reject': float(1.0 - risks[10]) if len(risks) >= 11 else 0.0,
        }

        for cov, risk in zip(coverages, risks):
            rows.append({
                'method': method_name,
                'coverage': float(cov),
                'risk': float(risk),
                'retained_accuracy': float(1.0 - risk),
            })

        log(f"  {method_name}: AURC={aurc:.4f} E-AURC={excess_aurc:.4f} "
            f"acc@20%reject={1.0-risks[3]:.4f} acc@50%reject={1.0-risks[10]:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'selective_prediction_v2.csv'), index=False)
    with open(os.path.join(RESULTS_DIR, 'selective_prediction_summary_v2.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"\nSelective prediction results saved to results/selective_prediction_v2.csv and summary_v2.json")
    return df, summary


def _get_edl_uncertainty(model, X_test, y_test):
    model.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        unc = model.predict_uncertainty(Xt)
        probs = unc['probs'].cpu().numpy()
        H = unc['H_total'].cpu().numpy()
    preds = probs.argmax(axis=1)
    errors = (preds != y_test).astype(int)
    return {'uncertainties': H, 'errors': errors, 'preds': preds,
            'base_acc': float((preds == y_test).mean())}


def _get_softmax_uncertainty(model, X_test, y_test, mc_dropout=False, use_msp=False, n_samples=50):
    if mc_dropout:
        model.train()
    else:
        model.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        if mc_dropout:
            logits = torch.stack([model(Xt) for _ in range(n_samples)], dim=0)
            probs = F.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
        else:
            logits = model(Xt)
            probs = F.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    if use_msp:
        # Maximum Softmax Probability (lower = more uncertain)
        unc = 1.0 - probs.max(axis=1)
    else:
        # Predictive entropy
        unc = -(probs * np.log(probs + 1e-10)).sum(axis=1)
    errors = (preds != y_test).astype(int)
    return {'uncertainties': unc, 'errors': errors, 'preds': preds,
            'base_acc': float((preds == y_test).mean())}


def rng_uniform(n, seed=42):
    rng = np.random.RandomState(seed)
    return rng.rand(n)


def main():
    log(f"Device: {DEVICE}")
    log(f"PyTorch: {torch.__version__}")

    # Load data once with temporal split
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = \
        preprocess_and_split(seed=SEED, save=False, split_mode='temporal')
    input_dim = X_train.shape[1]
    log(f"Train={len(y_train)} Val={len(y_val)} Test={len(y_test)} dim={input_dim}")
    log(f"Train rain={y_train.mean():.4f} Test rain={y_test.mean():.4f}")

    # Run all three diagnostic experiments
    df_abl = run_ablation(X_train, y_train, X_val, y_val, X_test, y_test, input_dim)
    df_sens, df_sens_sum = run_sensitivity(X_train, y_train, X_val, y_val, X_test, y_test, input_dim)
    df_rob = run_robustness(X_train, y_train, X_val, y_val, X_test, y_test, input_dim)
    df_sel, summary_sel = run_selective_prediction(X_train, y_train, X_val, y_val, X_test, y_test, input_dim)

    log("\n" + "="*60 + "\nALL DIAGNOSTIC EXPERIMENTS COMPLETE\n" + "="*60)
    LOG_FILE.close()


if __name__ == "__main__":
    main()
