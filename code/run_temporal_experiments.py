"""
Complete temporal-split experiments for Tables 3, 4, 5, 6.
Runs: sensitivity, robustness, uncertainty analysis, selective prediction.
All on temporal split (S1: 2007-2014 train / 2015 val / 2016-2017 test).
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
from models import EDLMLP
from train import edl_loss
from simple_experiment import (
    load_to_gpu, get_batches_gpu, compute_ece, evaluate_predictions,
    train_edl, eval_edl
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss

LOG_PATH = os.path.join(os.path.dirname(__file__), 'run_temporal_experiments.log')
LOG_FILE = open(LOG_PATH, 'w', encoding='utf-8')
def log(msg):
    print(msg, flush=True)
    LOG_FILE.write(msg + '\n'); LOG_FILE.flush()

SEED = 42
EPOCHS = 20


def _train_edl_custom(X_train, y_train, X_val, y_val, input_dim, lambda_reg=0.001,
                      annealing_epochs=50, loss_type='cross_entropy', epochs=20, lr=1e-3,
                      dropout_rate=None):
    torch.manual_seed(SEED); np.random.seed(SEED)
    hdr = MODEL['hidden_dims']
    dr = dropout_rate if dropout_rate is not None else MODEL['dropout_rate']
    model = EDLMLP(input_dim, hidden_dims=hdr, dropout_rate=dr).to(DEVICE)
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
    if best_state:
        model.load_state_dict(best_state)
    del Xtr, ytr, Xva, yva
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return model


def run_sensitivity(X_train, y_train, X_val, y_val, X_test, y_test, input_dim):
    log("\n" + "="*60 + "\nSENSITIVITY ANALYSIS (seed=42, temporal split, val tuning)\n" + "="*60)
    rows = []
    
    # 1. lambda_reg
    log("\n[S1] lambda_reg sensitivity")
    for lam in [0.0, 0.001, 0.01, 0.1]:
        m = _train_edl_custom(X_train, y_train, X_val, y_val, input_dim, lambda_reg=lam, epochs=EPOCHS)
        r_val = eval_edl(m, X_val, y_val)
        r_test = eval_edl(m, X_test, y_test)
        rows.append({'parameter': 'lambda_reg', 'value': lam,
                     'val_f1_macro': r_val['f1_macro'], 'test_f1_macro': r_test['f1_macro'],
                     'test_accuracy': r_test['accuracy'], 'test_ece': r_test['ece'],
                     'test_unc_auroc': r_test['uncertainty_auroc']})
        log(f"  lam={lam}: val_f1={r_val['f1_macro']:.4f} test_f1={r_test['f1_macro']:.4f}")
    
    # 2. dropout
    log("\n[S2] dropout sensitivity")
    for dr in [0.0, 0.2, 0.3, 0.4, 0.5]:
        m = _train_edl_custom(X_train, y_train, X_val, y_val, input_dim, dropout_rate=dr, epochs=EPOCHS)
        r_val = eval_edl(m, X_val, y_val)
        r_test = eval_edl(m, X_test, y_test)
        rows.append({'parameter': 'dropout', 'value': dr,
                     'val_f1_macro': r_val['f1_macro'], 'test_f1_macro': r_test['f1_macro'],
                     'test_accuracy': r_test['accuracy'], 'test_ece': r_test['ece'],
                     'test_unc_auroc': r_test['uncertainty_auroc']})
        log(f"  dropout={dr}: val_f1={r_val['f1_macro']:.4f} test_f1={r_test['f1_macro']:.4f}")
    
    # 3. learning rate
    log("\n[S3] learning rate sensitivity")
    for lr in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
        m = _train_edl_custom(X_train, y_train, X_val, y_val, input_dim, lr=lr, epochs=EPOCHS)
        r_val = eval_edl(m, X_val, y_val)
        r_test = eval_edl(m, X_test, y_test)
        rows.append({'parameter': 'learning_rate', 'value': lr,
                     'val_f1_macro': r_val['f1_macro'], 'test_f1_macro': r_test['f1_macro'],
                     'test_accuracy': r_test['accuracy'], 'test_ece': r_test['ece'],
                     'test_unc_auroc': r_test['uncertainty_auroc']})
        log(f"  lr={lr}: val_f1={r_val['f1_macro']:.4f} test_f1={r_test['f1_macro']:.4f}")
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'sensitivity_results_v2.csv'), index=False)
    
    # Summary with elasticity
    log("\n[Elasticity Summary]")
    summary_rows = []
    for p in df['parameter'].unique():
        sub = df[df['parameter'] == p].copy()
        x = sub['value'].values.astype(float)
        y = sub['test_f1_macro'].values.astype(float)
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
            'elasticity': round(elast, 6),
            'sensitivity_level': level,
        })
        log(f"  {p}: range=[{x.min()},{x.max()}] best_val={best_val} best_test_f1={best_f1:.4f} elasticity={elast:.6f} ({level})")
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 'sensitivity_summary_v2.csv'), index=False)
    log(f"\nSensitivity results saved to results/sensitivity_results_v2.csv and sensitivity_summary_v2.csv")
    return df, df_summary


def run_robustness(X_train, y_train, X_val, y_val, X_test, y_test, input_dim):
    log("\n" + "="*60 + "\nROBUSTNESS ANALYSIS (seed=42, temporal split)\n" + "="*60)
    rows = []
    
    torch.manual_seed(SEED); np.random.seed(SEED)
    m = train_edl(X_train, y_train, X_val, y_val, input_dim, seed=SEED, epochs=EPOCHS)
    
    rng = np.random.RandomState(SEED)
    
    # Clean baseline - get full uncertainty info
    m.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        unc = m.predict_uncertainty(Xt)
        probs = unc['probs'].cpu().numpy()
        H_T = unc['H_total'].cpu().numpy()
        H_A = unc['H_alea'].cpu().numpy()
        H_E = unc['H_epi'].cpu().numpy()
        S = unc['precision'].cpu().numpy()
    preds = probs.argmax(axis=1)
    
    def _full_eval(probs, H_T, H_A, H_E, S, y_test):
        errors = (preds != y_test).astype(int)
        try:
            unc_auroc = float(roc_auc_score(errors, H_T))
        except:
            unc_auroc = 0.5
        return {
            'accuracy': float(accuracy_score(y_test, preds)),
            'f1_macro': float(f1_score(y_test, preds, average='macro', zero_division=0)),
            'ece': compute_ece(probs, y_test),
            'brier': float(brier_score_loss(y_test, probs[:, 1])),
            'uncertainty_auroc': unc_auroc,
            'S_mean': float(S.mean()),
            'H_E_mean': float(H_E.mean()),
            'H_T_mean': float(H_T.mean()),
            'H_A_mean': float(H_A.mean()),
        }
    
    r_clean = _full_eval(probs, H_T, H_A, H_E, S, y_test)
    rows.append({'perturbation': 'Clean', 'level': 0.0, **r_clean})
    log(f"\n  Clean: acc={r_clean['accuracy']:.4f} f1={r_clean['f1_macro']:.4f} ece={r_clean['ece']:.4f} S={r_clean['S_mean']:.2f} H_E={r_clean['H_E_mean']:.6f} unc_auroc={r_clean['uncertainty_auroc']:.4f}")
    
    # Gaussian noise
    for noise_pct in [0.01, 0.05, 0.10, 0.15]:
        np.random.seed(SEED)
        X_test_noisy = X_test + rng.normal(0, noise_pct * X_train.std(axis=0), X_test.shape).astype(np.float32)
        Xt_noisy = torch.tensor(X_test_noisy, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            unc = m.predict_uncertainty(Xt_noisy)
            probs_n = unc['probs'].cpu().numpy()
            H_T_n = unc['H_total'].cpu().numpy()
            H_A_n = unc['H_alea'].cpu().numpy()
            H_E_n = unc['H_epi'].cpu().numpy()
            S_n = unc['precision'].cpu().numpy()
        preds_n = probs_n.argmax(axis=1)
        r = _full_eval(probs_n, H_T_n, H_A_n, H_E_n, S_n, y_test)
        rows.append({'perturbation': 'Gaussian_Noise', 'level': noise_pct, **r})
        log(f"  Noise {noise_pct:.0%}: acc={r['accuracy']:.4f} f1={r['f1_macro']:.4f} ece={r['ece']:.4f} S={r['S_mean']:.2f} H_E={r['H_E_mean']:.6f} unc_auroc={r['uncertainty_auroc']:.4f}")
    
    # Feature missing
    for missing_pct in [0.05, 0.10, 0.20, 0.30]:
        np.random.seed(SEED)
        mask = rng.rand(*X_test.shape) < missing_pct
        X_test_missing = X_test.copy()
        X_test_missing[mask] = 0.0
        Xt_m = torch.tensor(X_test_missing, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            unc = m.predict_uncertainty(Xt_m)
            probs_m = unc['probs'].cpu().numpy()
            H_T_m = unc['H_total'].cpu().numpy()
            H_A_m = unc['H_alea'].cpu().numpy()
            H_E_m = unc['H_epi'].cpu().numpy()
            S_m = unc['precision'].cpu().numpy()
        preds_m = probs_m.argmax(axis=1)
        r = _full_eval(probs_m, H_T_m, H_A_m, H_E_m, S_m, y_test)
        rows.append({'perturbation': 'Feature_Missing', 'level': missing_pct, **r})
        log(f"  Missing {missing_pct:.0%}: acc={r['accuracy']:.4f} f1={r['f1_macro']:.4f} ece={r['ece']:.4f} S={r['S_mean']:.2f} H_E={r['H_E_mean']:.6f} unc_auroc={r['uncertainty_auroc']:.4f}")
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'robustness_results_v2.csv'), index=False)
    log(f"\nRobustness results saved to results/robustness_results_v2.csv")
    return df


def run_uncertainty_analysis(X_train, y_train, X_val, y_val, X_test, y_test, input_dim):
    log("\n" + "="*60 + "\nUNCERTAINTY ANALYSIS (seed=42, temporal split)\n" + "="*60)
    
    torch.manual_seed(SEED); np.random.seed(SEED)
    m = train_edl(X_train, y_train, X_val, y_val, input_dim, seed=SEED, epochs=EPOCHS)
    m.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        unc = m.predict_uncertainty(Xt)
        probs = unc['probs'].cpu().numpy()
        H_T = unc['H_total'].cpu().numpy()
        H_A = unc['H_alea'].cpu().numpy()
        H_E = unc['H_epi'].cpu().numpy()
        S = unc['precision'].cpu().numpy()
    preds = probs.argmax(axis=1)
    errors = (preds != y_test).astype(int)
    
    n_test = len(y_test)
    n_correct = int((errors == 0).sum())
    n_errors = int((errors == 1).sum())
    
    correct_mask = errors == 0
    error_mask = errors == 1
    
    result = {
        'n_test': n_test,
        'n_correct': n_correct,
        'n_errors': n_errors,
        'H_total': {
            'mean': float(H_T.mean()),
            'std': float(H_T.std()),
            'median': float(np.median(H_T)),
            'correct_mean': float(H_T[correct_mask].mean()),
            'error_mean': float(H_T[error_mask].mean()),
        },
        'H_alea': {
            'mean': float(H_A.mean()),
            'std': float(H_A.std()),
            'median': float(np.median(H_A)),
            'correct_mean': float(H_A[correct_mask].mean()),
            'error_mean': float(H_A[error_mask].mean()),
        },
        'H_epi': {
            'mean': float(H_E.mean()),
            'std': float(H_E.std()),
            'median': float(np.median(H_E)),
            'correct_mean': float(H_E[correct_mask].mean()),
            'error_mean': float(H_E[error_mask].mean()),
        },
        'precision': {
            'mean': float(S.mean()),
            'std': float(S.std()),
            'median': float(np.median(S)),
            'correct_mean': float(S[correct_mask].mean()),
            'error_mean': float(S[error_mask].mean()),
        },
        'error_detection': {
            'uncertainty_auroc': float(roc_auc_score(errors, H_T)),
        },
        'base_accuracy': float(accuracy_score(y_test, preds)),
    }
    
    # Selective prediction curve
    order = np.argsort(-H_T)  # most uncertain first
    sorted_errors = errors[order]
    rejection_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    rejection_analysis = []
    for rr in rejection_rates:
        n_reject = int(rr * n_test)
        n_retain = n_test - n_reject
        retained_errors = sorted_errors[n_reject:]
        retained_acc = 1.0 - retained_errors.sum() / n_retain if n_retain > 0 else 0.0
        # Random rejection baseline
        random_acc = result['base_accuracy']
        # Oracle: reject errors first
        n_errors_total = n_errors
        if n_reject >= n_errors_total:
            oracle_acc = 1.0
        else:
            oracle_acc = (n_test - n_errors_total) / n_retain
        rejection_analysis.append({
            'rejection_rate': rr,
            'n_retained': n_retain,
            'accuracy_retained': retained_acc,
            'random_baseline': random_acc,
            'oracle_baseline': oracle_acc,
        })
        log(f"  Reject {rr:.0%}: retain={n_retain} acc={retained_acc:.4f} random={random_acc:.4f} oracle={oracle_acc:.4f}")
    
    result['rejection_rate_analysis'] = rejection_analysis
    
    with open(os.path.join(RESULTS_DIR, 'uncertainty_analysis_v2.json'), 'w') as f:
        json.dump(result, f, indent=2)
    
    log(f"\nUncertainty analysis saved to results/uncertainty_analysis_v2.json")
    log(f"  n_test={n_test} n_correct={n_correct} n_errors={n_errors}")
    log(f"  H_T: correct={H_T[correct_mask].mean():.4f} error={H_T[error_mask].mean():.4f}")
    log(f"  H_A: correct={H_A[correct_mask].mean():.4f} error={H_A[error_mask].mean():.4f}")
    log(f"  H_E: correct={H_E[correct_mask].mean():.6f} error={H_E[error_mask].mean():.6f}")
    log(f"  S:   correct={S[correct_mask].mean():.2f} error={S[error_mask].mean():.2f}")
    log(f"  Unc-AUROC={result['error_detection']['uncertainty_auroc']:.4f}")
    
    return result


def main():
    t0 = time.time()
    log(f"Device: {DEVICE}")
    log(f"PyTorch: {torch.__version__}")
    
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = \
        preprocess_and_split(seed=SEED, save=False, split_mode='temporal')
    input_dim = X_train.shape[1]
    log(f"Train={len(y_train)} Val={len(y_val)} Test={len(y_test)} dim={input_dim}")
    log(f"Train rain={y_train.mean():.4f} Test rain={y_test.mean():.4f}")
    
    # 1. Sensitivity
    log("\n>>> Starting sensitivity analysis...")
    t1 = time.time()
    run_sensitivity(X_train, y_train, X_val, y_val, X_test, y_test, input_dim)
    log(f">>> Sensitivity done in {time.time()-t1:.1f}s")
    
    # 2. Robustness
    log("\n>>> Starting robustness analysis...")
    t2 = time.time()
    run_robustness(X_train, y_train, X_val, y_val, X_test, y_test, input_dim)
    log(f">>> Robustness done in {time.time()-t2:.1f}s")
    
    # 3. Uncertainty analysis + selective prediction
    log("\n>>> Starting uncertainty analysis + selective prediction...")
    t3 = time.time()
    run_uncertainty_analysis(X_train, y_train, X_val, y_val, X_test, y_test, input_dim)
    log(f">>> Uncertainty analysis done in {time.time()-t3:.1f}s")
    
    log(f"\n{'='*60}")
    log(f"ALL TEMPORAL-SPLIT EXPERIMENTS COMPLETE")
    log(f"Total time: {time.time()-t0:.1f}s")
    log(f"{'='*60}")
    LOG_FILE.close()


if __name__ == "__main__":
    main()
