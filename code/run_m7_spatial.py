"""Quick test: run only spatial OOD experiment with reduced epochs."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import json
import time
import numpy as np
import torch

from config import DEVICE, MODEL, LOSS, TRAIN
from m7_ood_experiments import load_and_preprocess_ood, evaluate_model
from train import train_edl_model

# Override epochs
EPOCHS = 3

def run_spatial_ood_short():
    print("\n" + "=" * 60, flush=True)
    print("M7.1: SPATIAL OOD EXPERIMENT (15 epochs)", flush=True)
    print("=" * 60, flush=True)

    data = load_and_preprocess_ood(split_type='spatial')
    (X_train, y_train, X_val, y_val, X_test_id, y_test_id,
     X_test_ood, y_test_ood, test_id_meta, test_ood_meta, input_dim) = data
    print(f"  Unpacked: X_train={X_train.shape}, y_train={y_train.shape}, input_dim={input_dim}", flush=True)

    print("  Training EDL-Fixed on spatial OOD split...", flush=True)
    t0 = time.time()
    model, _ = train_edl_model(
        X_train, y_train, X_val, y_val, input_dim,
        seed=42, config={
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
            'patience': 10,
        }
    )
    print(f"  Training done in {time.time()-t0:.1f}s", flush=True)

    print("  Evaluating on in-distribution test set...", flush=True)
    r_id = evaluate_model(model, X_test_id, y_test_id)
    print(f"    ID: acc={r_id['accuracy']:.4f} f1={r_id['f1_macro']:.4f} ece={r_id['ece']:.4f} "
          f"S={r_id['S_mean']:.2f} H_E={r_id['H_E_mean']:.6f} H_T={r_id['H_T_mean']:.4f}", flush=True)

    print("  Evaluating on OOD test set...", flush=True)
    r_ood = evaluate_model(model, X_test_ood, y_test_ood)
    print(f"    OOD: acc={r_ood['accuracy']:.4f} f1={r_ood['f1_macro']:.4f} ece={r_ood['ece']:.4f} "
          f"S={r_ood['S_mean']:.2f} H_E={r_ood['H_E_mean']:.6f} H_T={r_ood['H_T_mean']:.4f}", flush=True)

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

    print(f"  OOD detection AUROC: H_T={ood_auroc_H_T:.4f} H_E={ood_auroc_H_E:.4f}", flush=True)

    result = {
        'experiment': 'spatial_ood',
        'id_results': {k: v for k, v in r_id.items() if not isinstance(v, np.ndarray)},
        'ood_results': {k: v for k, v in r_ood.items() if not isinstance(v, np.ndarray)},
        'ood_detection': {
            'auroc_H_T': ood_auroc_H_T,
            'auroc_H_E': ood_auroc_H_E,
        },
        'id_n': len(y_test_id),
        'ood_n': len(y_test_ood),
    }
    return result

result = run_spatial_ood_short()
print("\n=== RESULT ===", flush=True)
print(json.dumps(result, indent=2, default=str), flush=True)

with open(os.path.join('..', 'results', 'm7_spatial_ood.json'), 'w') as f:
    json.dump(result, f, indent=2, default=str)
print("\nSaved to results/m7_spatial_ood.json", flush=True)
