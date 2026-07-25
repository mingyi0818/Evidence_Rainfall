"""Quick test: train EDL for 3 epochs to verify the pipeline works."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
import sys
import time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

from config import DEVICE, MODEL, LOSS, TRAIN
from m7_ood_experiments import load_and_preprocess_ood
from train import train_edl_model

print(f"DEVICE: {DEVICE}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)

# Load spatial OOD data
print("\nLoading spatial OOD data...", flush=True)
data = load_and_preprocess_ood(split_type='spatial')
(X_train, y_train, X_val, y_val, X_test_id, y_test_id,
 X_test_ood, y_test_ood, test_id_meta, test_ood_meta, input_dim) = data
print(f"  X_train={X_train.shape}, y_train={y_train.shape}, dim={input_dim}", flush=True)

# Train for 3 epochs only
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
print(f"History: train_loss={history['train_loss']}, val_loss={history['val_loss']}", flush=True)
print("\nTest PASSED: training pipeline works correctly.", flush=True)
