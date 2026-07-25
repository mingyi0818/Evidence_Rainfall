"""Minimal test: replicate test_m7_quick.py exactly but with run_m7_spatial.py's structure."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.path.insert(0, os.path.dirname(__file__))
import time
import numpy as np
import torch

from config import DEVICE, MODEL, LOSS, TRAIN
from m7_ood_experiments import load_and_preprocess_ood, evaluate_model
from train import train_edl_model

print(f"DEVICE: {DEVICE}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

# Load spatial OOD data
print("\nLoading spatial OOD data...", flush=True)
data = load_and_preprocess_ood(split_type='spatial')
(X_train, y_train, X_val, y_val, X_test_id, y_test_id,
 X_test_ood, y_test_ood, test_id_meta, test_ood_meta, input_dim) = data
print(f"  X_train={X_train.shape}, y_train={y_train.shape}, dim={input_dim}", flush=True)

# Train for 3 epochs
print("\nTraining EDL for 3 epochs...", flush=True)
t0 = time.time()
model, history = train_edl_model(
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
        'epochs': 3,
        'patience': 10,
    }
)
print(f"\nTraining done in {time.time()-t0:.1f}s", flush=True)

# Now evaluate
print("\nEvaluating on ID test set...", flush=True)
r_id = evaluate_model(model, X_test_id, y_test_id)
print(f"  ID: acc={r_id['accuracy']:.4f} S={r_id['S_mean']:.2f} H_E={r_id['H_E_mean']:.6f}", flush=True)

print("\nEvaluating on OOD test set...", flush=True)
r_ood = evaluate_model(model, X_test_ood, y_test_ood)
print(f"  OOD: acc={r_ood['accuracy']:.4f} S={r_ood['S_mean']:.2f} H_E={r_ood['H_E_mean']:.6f}", flush=True)

print("\nTest PASSED.", flush=True)
