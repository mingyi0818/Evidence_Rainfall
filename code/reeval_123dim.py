"""
Re-evaluate all 123-dim checkpoints on the current 123-dim test set.

This script loads existing checkpoints (which are already 123-dim) and
evaluates them on the current temporal split test set. No re-training.

Output: results/main_results_v3.json (123-dim consistent results)
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, brier_score_loss)

sys.path.insert(0, os.path.dirname(__file__))
from config import (DEVICE, RANDOM_SEEDS, CHECKPOINT_DIR, RESULTS_DIR,
                    TARGET_COL, DATE_COL, LOCATION_COL)
from data_loader import preprocess_and_split
from models import build_model, SklearnWrapper
from evaluate import compute_ece


SEEDS = [42, 123, 456, 789, 2024]


def compute_unc_auroc(y_true, preds, uncertainty):
    errors = (preds != y_true).astype(int)
    if len(np.unique(errors)) < 2:
        return 0.5
    return float(roc_auc_score(errors, uncertainty))


def evaluate_edl(model, X_test, y_test):
    model.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        unc = model.predict_uncertainty(Xt)
        probs = unc['probs'].cpu().numpy()
        H_T = unc['H_total'].cpu().numpy()
        S = unc['precision'].cpu().numpy()
    preds = probs.argmax(axis=1)
    ece_val = compute_ece(y_test, probs[:, 1])
    ece_val = ece_val[0] if isinstance(ece_val, tuple) else ece_val
    return {
        'accuracy': float(accuracy_score(y_test, preds)),
        'precision': float(precision_score(y_test, preds, zero_division=0)),
        'recall': float(recall_score(y_test, preds, zero_division=0)),
        'f1_macro': float(f1_score(y_test, preds, average='macro', zero_division=0)),
        'auc': float(roc_auc_score(y_test, probs[:, 1])),
        'brier': float(brier_score_loss(y_test, probs[:, 1])),
        'ece': float(ece_val),
        'uncertainty_auroc': compute_unc_auroc(y_test, preds, H_T),
        'S_mean': float(S.mean()),
    }


def evaluate_softmax(model, X_test, y_test, model_type='lstm'):
    model.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        if model_type == 'mcdropout':
            # MC Dropout: average over n_samples forward passes
            n_samples = 50
            all_probs = []
            for _ in range(n_samples):
                logits = model(Xt)
                all_probs.append(F.softmax(logits, dim=1))
            probs = torch.stack(all_probs).mean(dim=0).cpu().numpy()
        else:
            logits = model(Xt)
            probs = F.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    H_T = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    ece_val = compute_ece(y_test, probs[:, 1])
    ece_val = ece_val[0] if isinstance(ece_val, tuple) else ece_val
    return {
        'accuracy': float(accuracy_score(y_test, preds)),
        'precision': float(precision_score(y_test, preds, zero_division=0)),
        'recall': float(recall_score(y_test, preds, zero_division=0)),
        'f1_macro': float(f1_score(y_test, preds, average='macro', zero_division=0)),
        'auc': float(roc_auc_score(y_test, probs[:, 1])),
        'brier': float(brier_score_loss(y_test, probs[:, 1])),
        'ece': float(ece_val),
        'uncertainty_auroc': compute_unc_auroc(y_test, preds, H_T),
    }


def evaluate_sklearn(wrapper, X_test, y_test):
    probs = wrapper.predict_proba(X_test)
    preds = probs.argmax(axis=1)
    H_T = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    ece_val = compute_ece(y_test, probs[:, 1])
    ece_val = ece_val[0] if isinstance(ece_val, tuple) else ece_val
    return {
        'accuracy': float(accuracy_score(y_test, preds)),
        'precision': float(precision_score(y_test, preds, zero_division=0)),
        'recall': float(recall_score(y_test, preds, zero_division=0)),
        'f1_macro': float(f1_score(y_test, preds, average='macro', zero_division=0)),
        'auc': float(roc_auc_score(y_test, probs[:, 1])),
        'brier': float(brier_score_loss(y_test, probs[:, 1])),
        'ece': float(ece_val),
        'uncertainty_auroc': compute_unc_auroc(y_test, preds, H_T),
    }


def main():
    print("=" * 60)
    print("Re-evaluation of 123-dim checkpoints")
    print("=" * 60)

    all_results = {}
    all_timings = {}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---", flush=True)
        t0 = time.time()

        # Load data with current 123-dim data_loader
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, encoder, feat_names = \
            preprocess_and_split(seed=seed, save=False, split_mode='temporal')
        input_dim = X_train.shape[1]
        print(f"  Train={len(y_train)} Val={len(y_val)} Test={len(y_test)} dim={input_dim}", flush=True)

        if input_dim != 123:
            print(f"  WARNING: input_dim={input_dim}, expected 123", flush=True)

        results = {}

        # EDL-Fixed
        try:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"edl_seed{seed}.pth")
            if os.path.exists(ckpt_path):
                model = build_model('edl', input_dim, num_classes=2,
                                    hidden_dims=[128, 64, 32], dropout_rate=0.3).to(DEVICE)
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                results['EDL-Fixed'] = evaluate_edl(model, X_test, y_test)
                print(f"  EDL-Fixed: acc={results['EDL-Fixed']['accuracy']:.4f} "
                      f"f1={results['EDL-Fixed']['f1_macro']:.4f} "
                      f"ece={results['EDL-Fixed']['ece']:.4f} "
                      f"S={results['EDL-Fixed']['S_mean']:.1f}", flush=True)
        except Exception as e:
            print(f"  EDL-Fixed ERROR: {e}", flush=True)

        # LSTM
        try:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"lstm_seed{seed}.pth")
            if os.path.exists(ckpt_path):
                model = build_model('lstm', input_dim, num_classes=2,
                                    hidden_size=64, num_layers=2).to(DEVICE)
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                results['LSTM'] = evaluate_softmax(model, X_test, y_test, model_type='lstm')
                print(f"  LSTM: acc={results['LSTM']['accuracy']:.4f} "
                      f"f1={results['LSTM']['f1_macro']:.4f}", flush=True)
        except Exception as e:
            print(f"  LSTM ERROR: {e}", flush=True)

        # GRU
        try:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"gru_seed{seed}.pth")
            if os.path.exists(ckpt_path):
                model = build_model('gru', input_dim, num_classes=2,
                                    hidden_size=64, num_layers=2).to(DEVICE)
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                results['GRU'] = evaluate_softmax(model, X_test, y_test, model_type='gru')
                print(f"  GRU: acc={results['GRU']['accuracy']:.4f} "
                      f"f1={results['GRU']['f1_macro']:.4f}", flush=True)
        except Exception as e:
            print(f"  GRU ERROR: {e}", flush=True)

        # MCDropout
        try:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"mcdropout_seed{seed}.pth")
            if os.path.exists(ckpt_path):
                model = build_model('mcdropout', input_dim, num_classes=2,
                                    hidden_dims=[128, 64, 32], dropout_rate=0.3).to(DEVICE)
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                results['MCDropout'] = evaluate_softmax(model, X_test, y_test, model_type='mcdropout')
                print(f"  MCDropout: acc={results['MCDropout']['accuracy']:.4f} "
                      f"f1={results['MCDropout']['f1_macro']:.4f}", flush=True)
        except Exception as e:
            print(f"  MCDropout ERROR: {e}", flush=True)

        # BNN
        try:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"bnn_seed{seed}.pth")
            if os.path.exists(ckpt_path):
                model = build_model('bnn', input_dim, num_classes=2,
                                    hidden_dims=[128, 64]).to(DEVICE)
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                results['BNN'] = evaluate_softmax(model, X_test, y_test, model_type='bnn')
                print(f"  BNN: acc={results['BNN']['accuracy']:.4f} "
                      f"f1={results['BNN']['f1_macro']:.4f}", flush=True)
        except Exception as e:
            print(f"  BNN ERROR: {e}", flush=True)

        # Sklearn baselines (load from pkl)
        for name, fname in [('LR', 'logisticregression'), ('RF', 'randomforest'), ('XGB', 'xgboost')]:
            try:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"{fname}_seed{seed}.pkl")
                if os.path.exists(ckpt_path):
                    import pickle
                    with open(ckpt_path, 'rb') as f:
                        wrapper = pickle.load(f)
                    results[name] = evaluate_sklearn(wrapper, X_test, y_test)
                    print(f"  {name}: acc={results[name]['accuracy']:.4f} "
                          f"f1={results[name]['f1_macro']:.4f}", flush=True)
            except Exception as e:
                print(f"  {name} ERROR: {e}", flush=True)

        # Climatology baseline
        pi_train = y_train.mean()
        clim_preds = np.zeros(len(y_test), dtype=int)
        results['Climatology'] = {
            'accuracy': float(accuracy_score(y_test, clim_preds)),
            'precision': 0.0, 'recall': 0.0,
            'f1_macro': float(f1_score(y_test, clim_preds, average='macro', zero_division=0)),
            'auc': 0.5,
            'brier': float(brier_score_loss(y_test, np.full(len(y_test), pi_train))),
            'ece': float(abs(pi_train - y_test.mean())),
            'uncertainty_auroc': 0.5,
        }

        elapsed = time.time() - t0
        all_results[str(seed)] = results
        all_timings[str(seed)] = elapsed
        print(f"  Seed {seed} done in {elapsed:.1f}s", flush=True)

    # Aggregate
    print("\n" + "=" * 60)
    print("AGGREGATED (mean +/- std)")
    print("=" * 60)
    model_names = []
    for s in all_results:
        for m in all_results[s]:
            if m not in model_names:
                model_names.append(m)

    aggregated = {}
    for mname in model_names:
        per_metric = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_macro', 'auc',
                       'brier', 'ece', 'uncertainty_auroc']:
            vals = []
            for s in all_results:
                if mname in all_results[s] and all_results[s][mname] and metric in all_results[s][mname]:
                    vals.append(all_results[s][mname][metric])
            if vals:
                per_metric[metric] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'values': vals,
                }
        aggregated[mname] = per_metric
        if 'accuracy' in per_metric:
            print(f"  {mname}: acc={per_metric['accuracy']['mean']:.4f} "
                  f"f1={per_metric['f1_macro']['mean']:.4f}", flush=True)

    # Save
    output = {
        'seeds': SEEDS,
        'split': 'temporal_S1',
        'feature_dim': 123,
        'per_seed': all_results,
        'aggregated': aggregated,
        'timings': all_timings,
        'note': 'Re-evaluated with 123-dim features (Year/Month/DayOfYear/Season included)',
    }

    out_path = os.path.join(RESULTS_DIR, 'main_results_v3.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
