"""Quick test with reduced epochs to verify the pipeline works."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Setup file logging
log_file = open(os.path.join(os.path.dirname(__file__), 'quick_test.log'), 'w', encoding='utf-8')
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, s):
        for f in self.files:
            f.write(s)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

import numpy as np
import torch
from config import DEVICE, MODEL, TRAIN
from data_loader import preprocess_and_split, WeatherDataset
from models import build_model
from train import edl_loss, get_annealing_factor, train_sklearn_baseline, train_torch_baseline
from torch.utils.data import DataLoader
import time

print(f"Device: {DEVICE}", flush=True)

# Quick test: 5 epochs only
print("\n=== Quick Test: 5 epochs, seed 42, temporal split ===", flush=True)
t0 = time.time()

X_train, X_val, X_test, y_train, y_val, y_test, scaler, encoder, feat_names = \
    preprocess_and_split(seed=42, save=False, split_mode='temporal')
input_dim = X_train.shape[1]
print(f"Train={len(y_train)} Val={len(y_val)} Test={len(y_test)} input_dim={input_dim}", flush=True)
print(f"Train rain rate={y_train.mean():.4f} Test rain rate={y_test.mean():.4f}", flush=True)

# Quick EDL training (5 epochs)
print("\n[EDL-Fixed] Quick training (5 epochs)...", flush=True)
torch.manual_seed(42)
np.random.seed(42)

train_ds = WeatherDataset(X_train, y_train)
val_ds = WeatherDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

model = build_model('edl', input_dim, num_classes=2,
                    hidden_dims=MODEL['hidden_dims'],
                    dropout_rate=MODEL['dropout_rate']).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

for epoch in range(5):
    af = min(1.0, epoch / 50)
    model.train()
    total_loss = 0.0
    n_samples = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        alpha = model.predict_dirichlet(Xb)
        loss, ce, kl = edl_loss(alpha, yb, lambda_reg=0.001,
                                 annealing_factor=af, loss_type='cross_entropy')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
        n_samples += Xb.size(0)
    print(f"  Epoch {epoch+1}/5 train_loss={total_loss/n_samples:.4f} af={af:.3f}", flush=True)

# Evaluate
print("\n[EDL-Fixed] Evaluating...", flush=True)
model.eval()
test_ds = WeatherDataset(X_test, np.zeros(len(X_test)))
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
all_preds, all_probs, all_H = [], [], []
with torch.no_grad():
    for Xb, _ in test_loader:
        Xb = Xb.to(DEVICE)
        unc = model.predict_uncertainty(Xb)
        probs = unc['probs'].cpu().numpy()
        H = unc['H_total'].cpu().numpy()
        preds = probs.argmax(axis=1)
        all_preds.append(preds)
        all_probs.append(probs)
        all_H.append(H)

preds = np.concatenate(all_preds)
probs = np.concatenate(all_probs)
H = np.concatenate(all_H)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss, precision_score, recall_score
errors = (preds != y_test).astype(int)
unc_auroc = float(roc_auc_score(errors, H)) if len(np.unique(errors)) > 1 else 0.5
ece = 0.0
for i in range(15):
    lo, hi = i/15, (i+1)/15
    in_bin = (probs[:,1] > lo) & (probs[:,1] <= hi) if i > 0 else (probs[:,1] >= lo) & (probs[:,1] <= hi)
    if in_bin.sum() > 0:
        ece += abs(probs[in_bin,1].mean() - y_test[in_bin].mean()) * in_bin.mean()

print(f"\n=== EDL-Fixed Results (5 epochs) ===", flush=True)
print(f"  accuracy={accuracy_score(y_test, preds):.4f}", flush=True)
print(f"  f1_macro={f1_score(y_test, preds, average='macro'):.4f}", flush=True)
print(f"  precision={precision_score(y_test, preds, zero_division=0):.4f}", flush=True)
print(f"  recall={recall_score(y_test, preds, zero_division=0):.4f}", flush=True)
print(f"  auc={roc_auc_score(y_test, probs[:,1]):.4f}", flush=True)
print(f"  brier={brier_score_loss(y_test, probs[:,1]):.4f}", flush=True)
print(f"  ece={ece:.4f}", flush=True)
print(f"  unc_auroc={unc_auroc:.4f}", flush=True)

# Quick LR baseline
print("\n[LR] Training (no class_weight)...", flush=True)
lr_wrapper = train_sklearn_baseline('LogisticRegression', X_train, y_train, 42, use_class_weight=False)
lr_probs = lr_wrapper.predict_proba(X_test)
lr_preds = lr_probs.argmax(axis=1)
lr_H = -np.sum(lr_probs * np.log(lr_probs + 1e-10), axis=1)
lr_errors = (lr_preds != y_test).astype(int)
lr_unc_auroc = float(roc_auc_score(lr_errors, lr_H)) if len(np.unique(lr_errors)) > 1 else 0.5
print(f"  accuracy={accuracy_score(y_test, lr_preds):.4f}", flush=True)
print(f"  f1_macro={f1_score(y_test, lr_preds, average='macro'):.4f}", flush=True)
print(f"  unc_auroc={lr_unc_auroc:.4f}", flush=True)

# Quick LSTM baseline
print("\n[LSTM] Training (10 epochs)...", flush=True)
from config import BASELINES
lstm_kwargs = {k:v for k,v in BASELINES['LSTM'].items() if k != 'enabled'}
# Remove sequence_length to avoid TypeError
lstm_kwargs.pop('sequence_length', None)
lstm_model = train_torch_baseline('lstm', X_train, y_train, X_val, y_val, input_dim,
                                   seed=42, **lstm_kwargs)
# Evaluate LSTM with F1 fix (softmax entropy)
lstm_model.eval()
all_preds, all_probs, all_H = [], [], []
with torch.no_grad():
    for Xb, _ in test_loader:
        Xb = Xb.to(DEVICE)
        logits = lstm_model(Xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        H = -(probs * np.log(probs + 1e-10)).sum(axis=1)
        preds = probs.argmax(axis=1)
        all_preds.append(preds)
        all_probs.append(probs)
        all_H.append(H)
preds = np.concatenate(all_preds)
probs = np.concatenate(all_probs)
H = np.concatenate(all_H)
errors = (preds != y_test).astype(int)
lstm_unc_auroc = float(roc_auc_score(errors, H)) if len(np.unique(errors)) > 1 else 0.5
print(f"  accuracy={accuracy_score(y_test, preds):.4f}", flush=True)
print(f"  f1_macro={f1_score(y_test, preds, average='macro'):.4f}", flush=True)
print(f"  unc_auroc={lstm_unc_auroc:.4f} (F1 fix: no longer 0.5)", flush=True)

elapsed = time.time() - t0
print(f"\n[Done] Quick test completed in {elapsed:.1f}s", flush=True)
log_file.close()
