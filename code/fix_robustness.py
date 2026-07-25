"""Fix robustness_results_v2.csv: re-compute accuracy/f1 from perturbed probs."""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from config import DEVICE, MODEL
from data_loader import preprocess_and_split
from simple_experiment import train_edl, compute_ece

SEED = 42
EPOCHS = 20

def main():
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = \
        preprocess_and_split(seed=SEED, save=False, split_mode='temporal')
    input_dim = X_train.shape[1]

    torch.manual_seed(SEED); np.random.seed(SEED)
    m = train_edl(X_train, y_train, X_val, y_val, input_dim, seed=SEED, epochs=EPOCHS)
    m.eval()
    rng = np.random.RandomState(SEED)

    rows = []
    # Clean
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
    rows.append({
        'perturbation': 'Clean', 'level': 0.0,
        'accuracy': float(accuracy_score(y_test, preds)),
        'f1_macro': float(f1_score(y_test, preds, average='macro', zero_division=0)),
        'ece': compute_ece(probs, y_test),
        'brier': float(brier_score_loss(y_test, probs[:, 1])),
        'uncertainty_auroc': float(roc_auc_score(errors, H_T)),
        'S_mean': float(S.mean()), 'H_E_mean': float(H_E.mean()),
        'H_T_mean': float(H_T.mean()), 'H_A_mean': float(H_A.mean()),
    })
    print(f"Clean: acc={rows[-1]['accuracy']:.4f} f1={rows[-1]['f1_macro']:.4f}")

    # Gaussian noise
    for noise_pct in [0.01, 0.05, 0.10, 0.15]:
        np.random.seed(SEED)
        X_test_noisy = X_test + rng.normal(0, noise_pct * X_train.std(axis=0), X_test.shape).astype(np.float32)
        Xt_n = torch.tensor(X_test_noisy, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            unc = m.predict_uncertainty(Xt_n)
            p = unc['probs'].cpu().numpy()
            H_T_n = unc['H_total'].cpu().numpy()
            H_A_n = unc['H_alea'].cpu().numpy()
            H_E_n = unc['H_epi'].cpu().numpy()
            S_n = unc['precision'].cpu().numpy()
        pr = p.argmax(axis=1)
        er = (pr != y_test).astype(int)
        rows.append({
            'perturbation': 'Gaussian_Noise', 'level': noise_pct,
            'accuracy': float(accuracy_score(y_test, pr)),
            'f1_macro': float(f1_score(y_test, pr, average='macro', zero_division=0)),
            'ece': compute_ece(p, y_test),
            'brier': float(brier_score_loss(y_test, p[:, 1])),
            'uncertainty_auroc': float(roc_auc_score(er, H_T_n)),
            'S_mean': float(S_n.mean()), 'H_E_mean': float(H_E_n.mean()),
            'H_T_mean': float(H_T_n.mean()), 'H_A_mean': float(H_A_n.mean()),
        })
        print(f"Noise {noise_pct:.0%}: acc={rows[-1]['accuracy']:.4f} f1={rows[-1]['f1_macro']:.4f}")

    # Feature missing
    for missing_pct in [0.05, 0.10, 0.20, 0.30]:
        np.random.seed(SEED)
        mask = rng.rand(*X_test.shape) < missing_pct
        X_test_missing = X_test.copy()
        X_test_missing[mask] = 0.0
        Xt_m = torch.tensor(X_test_missing, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            unc = m.predict_uncertainty(Xt_m)
            p = unc['probs'].cpu().numpy()
            H_T_m = unc['H_total'].cpu().numpy()
            H_A_m = unc['H_alea'].cpu().numpy()
            H_E_m = unc['H_epi'].cpu().numpy()
            S_m = unc['precision'].cpu().numpy()
        pr = p.argmax(axis=1)
        er = (pr != y_test).astype(int)
        rows.append({
            'perturbation': 'Feature_Missing', 'level': missing_pct,
            'accuracy': float(accuracy_score(y_test, pr)),
            'f1_macro': float(f1_score(y_test, pr, average='macro', zero_division=0)),
            'ece': compute_ece(p, y_test),
            'brier': float(brier_score_loss(y_test, p[:, 1])),
            'uncertainty_auroc': float(roc_auc_score(er, H_T_m)),
            'S_mean': float(S_m.mean()), 'H_E_mean': float(H_E_m.mean()),
            'H_T_mean': float(H_T_m.mean()), 'H_A_mean': float(H_A_m.mean()),
        })
        print(f"Missing {missing_pct:.0%}: acc={rows[-1]['accuracy']:.4f} f1={rows[-1]['f1_macro']:.4f}")

    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'robustness_results_v2.csv')
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")
    print(df.to_string())

if __name__ == "__main__":
    main()
