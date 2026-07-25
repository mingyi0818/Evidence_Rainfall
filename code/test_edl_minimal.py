"""Test EDL model and loss specifically."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn.functional as F
from config import DEVICE, MODEL
from models import build_model
from train import edl_loss
from torch.utils.data import DataLoader, TensorDataset

print(f"Device: {DEVICE}", flush=True)

# Small synthetic data
N = 500
dim = 119
X = torch.randn(N, dim).to(DEVICE)
y = torch.randint(0, 2, (N,)).to(DEVICE)
ds = TensorDataset(X, y)
loader = DataLoader(ds, batch_size=256, shuffle=True)

# Build EDL model
print("Building EDL model...", flush=True)
model = build_model('edl', dim, num_classes=2,
                    hidden_dims=MODEL['hidden_dims'],
                    dropout_rate=MODEL['dropout_rate']).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

print(f"Model: {model}", flush=True)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}", flush=True)

# Train 3 epochs
print("\nStarting EDL training...", flush=True)
for epoch in range(3):
    af = min(1.0, epoch / 50)
    model.train()
    total_loss = 0
    n = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        alpha = model.predict_dirichlet(Xb)
        print(f"  Epoch {epoch+1} batch: alpha.shape={alpha.shape}, alpha[:2]={alpha[:2].tolist()}", flush=True)
        loss, ce, kl = edl_loss(alpha, yb, lambda_reg=0.001,
                                 annealing_factor=af, loss_type='cross_entropy')
        print(f"    loss={loss.item():.4f} ce={ce:.4f} kl={kl:.4f}", flush=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
        n += Xb.size(0)
    print(f"  Epoch {epoch+1}/3 avg_loss={total_loss/n:.4f}", flush=True)

# Test predict_uncertainty
print("\nTesting predict_uncertainty...", flush=True)
model.eval()
with torch.no_grad():
    unc = model.predict_uncertainty(X[:5])
    print(f"  probs={unc['probs'][:2].tolist()}", flush=True)
    print(f"  H_total={unc['H_total'][:2].tolist()}", flush=True)
    print(f"  H_epi={unc['H_epi'][:2].tolist()}", flush=True)

print("\nDone!", flush=True)
