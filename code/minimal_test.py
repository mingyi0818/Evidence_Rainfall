"""Minimal test: just check if CUDA training works for 2 epochs."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print(f"PyTorch: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

# Simple model
class SimpleModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 64)
        self.fc2 = nn.Linear(64, 2)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleModel(119).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Dummy data
N = 1000
X = torch.randn(N, 119).to(device)
y = torch.randint(0, 2, (N,)).to(device)
bs = 256

print("\nStarting training...", flush=True)
for epoch in range(3):
    model.train()
    total_loss = 0
    for i in range(0, N, bs):
        xb = X[i:i+bs]
        yb = y[i:i+bs]
        optimizer.zero_grad()
        out = model(xb)
        loss = F.cross_entropy(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}/3 loss={total_loss:.4f}", flush=True)

print("\nDone!", flush=True)
