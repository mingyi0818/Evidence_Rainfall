"""
Simplified experiment runner that avoids DataLoader.
Uses manual batching for robustness.
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
from config import DEVICE, MODEL, TRAIN, BASELINES, RESULTS_DIR
from data_loader import preprocess_and_split
from models import build_model, EDLMLP, MCDropoutMLP, BayesianMLP, LSTMClassifier, GRUClassifier, SklearnWrapper
from train import edl_loss, train_sklearn_baseline

# File logging
log_path = os.path.join(os.path.dirname(__file__), 'simple_experiment.log')
log_file = open(log_path, 'w', encoding='utf-8')
def log(msg):
    print(msg, flush=True)
    log_file.write(msg + '\n')
    log_file.flush()

log(f"Device: {DEVICE}")
log(f"PyTorch: {torch.__version__}")

# Manual batching with pre-loaded GPU tensors
def load_to_gpu(X, y):
    """Load entire dataset to GPU once."""
    return torch.tensor(X, dtype=torch.float32).to(DEVICE), torch.tensor(y, dtype=torch.long).to(DEVICE)

def get_batches_gpu(X_gpu, y_gpu, batch_size, shuffle=True):
    n = len(X_gpu)
    idx = torch.randperm(n, device=X_gpu.device) if shuffle else torch.arange(n, device=X_gpu.device)
    for i in range(0, n, batch_size):
        bidx = idx[i:i+batch_size]
        yield X_gpu[bidx], y_gpu[bidx]

def compute_ece(probs, y, n_bins=15):
    ece = 0.0
    p1 = probs[:, 1]
    for i in range(n_bins):
        lo, hi = i/n_bins, (i+1)/n_bins
        in_bin = (p1 > lo) & (p1 <= hi) if i > 0 else (p1 >= lo) & (p1 <= hi)
        if in_bin.sum() > 0:
            ece += abs(p1[in_bin].mean() - y[in_bin].mean()) * in_bin.mean()
    return float(ece)

def evaluate_predictions(preds, probs, H, y_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
    errors = (preds != y_test).astype(int)
    try:
        unc_auroc = float(roc_auc_score(errors, H)) if len(np.unique(errors)) > 1 else 0.5
    except:
        unc_auroc = 0.5
    return {
        'accuracy': float(accuracy_score(y_test, preds)),
        'precision': float(precision_score(y_test, preds, zero_division=0)),
        'recall': float(recall_score(y_test, preds, zero_division=0)),
        'f1_macro': float(f1_score(y_test, preds, average='macro', zero_division=0)),
        'auc': float(roc_auc_score(y_test, probs[:, 1])),
        'brier': float(brier_score_loss(y_test, probs[:, 1])),
        'ece': compute_ece(probs, y_test),
        'uncertainty_auroc': unc_auroc,
    }

def train_edl(X_train, y_train, X_val, y_val, input_dim, seed=42, epochs=20, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EDLMLP(input_dim, hidden_dims=MODEL['hidden_dims'], dropout_rate=MODEL['dropout_rate']).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val = float('inf')
    best_state = None
    bs = 256
    # Pre-load to GPU
    Xtr, ytr = load_to_gpu(X_train, y_train)
    Xva, yva = load_to_gpu(X_val, y_val)
    for epoch in range(epochs):
        af = min(1.0, epoch / 50)
        model.train()
        total = 0; n = 0
        for Xb, yb in get_batches_gpu(Xtr, ytr, bs):
            opt.zero_grad()
            alpha = model.predict_dirichlet(Xb)
            loss, ce, kl = edl_loss(alpha, yb, lambda_reg=0.001, annealing_factor=af, loss_type='cross_entropy')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * Xb.size(0); n += Xb.size(0)
        # Val
        model.eval()
        vtotal = 0; vn = 0
        with torch.no_grad():
            for Xb, yb in get_batches_gpu(Xva, yva, bs, shuffle=False):
                alpha = model.predict_dirichlet(Xb)
                loss, _, _ = edl_loss(alpha, yb, lambda_reg=0.001, annealing_factor=af, loss_type='cross_entropy')
                vtotal += loss.item() * Xb.size(0); vn += Xb.size(0)
        vloss = vtotal / vn
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"  EDL epoch {epoch+1}/{epochs} train={total/n:.4f} val={vloss:.4f} af={af:.3f}")
    if best_state:
        model.load_state_dict(best_state)
    # Free GPU memory
    del Xtr, ytr, Xva, yva
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return model

def eval_edl(model, X_test, y_test):
    model.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        unc = model.predict_uncertainty(Xt)
        probs = unc['probs'].cpu().numpy()
        H = unc['H_total'].cpu().numpy()
    preds = probs.argmax(axis=1)
    return evaluate_predictions(preds, probs, H, y_test)

def train_torch_model(model_name, X_train, y_train, X_val, y_val, input_dim, seed=42, epochs=20, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if model_name == 'lstm':
        model = LSTMClassifier(input_dim, hidden_size=64, num_layers=2, dropout=0.3).to(DEVICE)
    elif model_name == 'gru':
        model = GRUClassifier(input_dim, hidden_size=64, num_layers=2, dropout=0.3).to(DEVICE)
    elif model_name == 'mcdropout':
        model = MCDropoutMLP(input_dim, hidden_dims=[128,64,32], dropout_rate=0.3).to(DEVICE)
    elif model_name == 'bnn':
        model = BayesianMLP(input_dim, hidden_dims=[128,64]).to(DEVICE)
    else:
        raise ValueError(model_name)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss()
    best_val = float('inf')
    best_state = None
    bs = 256
    # Pre-load to GPU
    Xtr, ytr = load_to_gpu(X_train, y_train)
    Xva, yva = load_to_gpu(X_val, y_val)
    for epoch in range(epochs):
        model.train()
        total = 0; n = 0
        for Xb, yb in get_batches_gpu(Xtr, ytr, bs):
            opt.zero_grad()
            logits = model(Xb)
            loss = crit(logits, yb)
            if hasattr(model, 'kl_divergence'):
                loss = loss + model.kl_divergence() / len(X_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * Xb.size(0); n += Xb.size(0)
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
            log(f"  {model_name.upper()} epoch {epoch+1}/{epochs} train={total/n:.4f} val={vloss:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    # Free GPU memory
    del Xtr, ytr, Xva, yva
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return model

def eval_softmax(model, X_test, y_test):
    model.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(Xt)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    H = -(probs * np.log(probs + 1e-10)).sum(axis=1)
    preds = probs.argmax(axis=1)
    return evaluate_predictions(preds, probs, H, y_test)

def eval_mcdropout(model, X_test, y_test, n_samples=50):
    model.train()  # keep dropout active
    Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = torch.stack([model(Xt) for _ in range(n_samples)], dim=0)
        probs = F.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
    H = -(probs * np.log(probs + 1e-10)).sum(axis=1)
    preds = probs.argmax(axis=1)
    return evaluate_predictions(preds, probs, H, y_test)

def eval_bnn(model, X_test, y_test, n_samples=50):
    model.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = torch.stack([model(Xt) for _ in range(n_samples)], dim=0)
        probs = F.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
    H = -(probs * np.log(probs + 1e-10)).sum(axis=1)
    preds = probs.argmax(axis=1)
    return evaluate_predictions(preds, probs, H, y_test)


def run_seed(seed, split_mode='temporal', epochs=20):
    log(f"\n{'='*60}\nSEED={seed} SPLIT={split_mode} EPOCHS={epochs}\n{'='*60}")
    t0 = time.time()
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = \
        preprocess_and_split(seed=seed, save=False, split_mode=split_mode)
    input_dim = X_train.shape[1]
    log(f"Train={len(y_train)} Val={len(y_val)} Test={len(y_test)} dim={input_dim}")
    log(f"Train rain={y_train.mean():.4f} Test rain={y_test.mean():.4f}")
    results = {}

    # EDL-Fixed
    try:
        log("\n[EDL-Fixed] Training...")
        m = train_edl(X_train, y_train, X_val, y_val, input_dim, seed=seed, epochs=epochs)
        results['EDL-Fixed'] = eval_edl(m, X_test, y_test)
        log(f"  EDL-Fixed: acc={results['EDL-Fixed']['accuracy']:.4f} f1={results['EDL-Fixed']['f1_macro']:.4f} "
            f"ece={results['EDL-Fixed']['ece']:.4f} unc={results['EDL-Fixed']['uncertainty_auroc']:.4f}")
    except Exception as e:
        import traceback; log(f"  [ERROR] EDL-Fixed: {e}\n{traceback.format_exc()}")
        results['EDL-Fixed'] = None

    # EDL-C1 (global climatology prior at inference)
    try:
        log("\n[EDL-C1] Training (same as EDL-Fixed, climatology prior at inference)...")
        m_c1 = train_edl(X_train, y_train, X_val, y_val, input_dim, seed=seed, epochs=epochs)
        pi = y_train.mean(); n0 = 10.0
        prior = torch.tensor([n0*(1-pi), n0*pi], dtype=torch.float32).to(DEVICE)
        m_c1.eval()
        Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            alpha = m_c1.predict_dirichlet(Xt, alpha_prior=prior.unsqueeze(0).expand(len(Xt), -1))
            probs = (alpha / alpha.sum(dim=1, keepdim=True)).cpu().numpy()
        H = -(probs * np.log(probs + 1e-10)).sum(axis=1)
        preds = probs.argmax(axis=1)
        results['EDL-C1'] = evaluate_predictions(preds, probs, H, y_test)
        log(f"  EDL-C1: acc={results['EDL-C1']['accuracy']:.4f} f1={results['EDL-C1']['f1_macro']:.4f} "
            f"ece={results['EDL-C1']['ece']:.4f} unc={results['EDL-C1']['uncertainty_auroc']:.4f}")
    except Exception as e:
        import traceback; log(f"  [ERROR] EDL-C1: {e}\n{traceback.format_exc()}")
        results['EDL-C1'] = None

    # LR
    try:
        log("\n[LR] Training (no class_weight)...")
        w = train_sklearn_baseline('LogisticRegression', X_train, y_train, seed, use_class_weight=False)
        probs = w.predict_proba(X_test); preds = probs.argmax(axis=1)
        H = -(probs * np.log(probs + 1e-10)).sum(axis=1)
        results['LR'] = evaluate_predictions(preds, probs, H, y_test)
        log(f"  LR: acc={results['LR']['accuracy']:.4f} f1={results['LR']['f1_macro']:.4f} unc={results['LR']['uncertainty_auroc']:.4f}")
    except Exception as e:
        log(f"  [ERROR] LR: {e}"); results['LR'] = None

    # RF
    try:
        log("\n[RF] Training (no class_weight)...")
        w = train_sklearn_baseline('RandomForest', X_train, y_train, seed, use_class_weight=False)
        probs = w.predict_proba(X_test); preds = probs.argmax(axis=1)
        H = -(probs * np.log(probs + 1e-10)).sum(axis=1)
        results['RF'] = evaluate_predictions(preds, probs, H, y_test)
        log(f"  RF: acc={results['RF']['accuracy']:.4f} f1={results['RF']['f1_macro']:.4f} unc={results['RF']['uncertainty_auroc']:.4f}")
    except Exception as e:
        log(f"  [ERROR] RF: {e}"); results['RF'] = None

    # XGB
    try:
        log("\n[XGB] Training (no scale_pos_weight)...")
        w = train_sklearn_baseline('XGBoost', X_train, y_train, seed, use_class_weight=False)
        probs = w.predict_proba(X_test); preds = probs.argmax(axis=1)
        H = -(probs * np.log(probs + 1e-10)).sum(axis=1)
        results['XGB'] = evaluate_predictions(preds, probs, H, y_test)
        log(f"  XGB: acc={results['XGB']['accuracy']:.4f} f1={results['XGB']['f1_macro']:.4f} unc={results['XGB']['uncertainty_auroc']:.4f}")
    except Exception as e:
        log(f"  [ERROR] XGB: {e}"); results['XGB'] = None

    # LSTM
    try:
        log("\n[LSTM] Training...")
        m = train_torch_model('lstm', X_train, y_train, X_val, y_val, input_dim, seed=seed, epochs=epochs)
        results['LSTM'] = eval_softmax(m, X_test, y_test)
        log(f"  LSTM: acc={results['LSTM']['accuracy']:.4f} f1={results['LSTM']['f1_macro']:.4f} unc={results['LSTM']['uncertainty_auroc']:.4f}")
    except Exception as e:
        import traceback; log(f"  [ERROR] LSTM: {e}\n{traceback.format_exc()}")
        results['LSTM'] = None

    # GRU
    try:
        log("\n[GRU] Training...")
        m = train_torch_model('gru', X_train, y_train, X_val, y_val, input_dim, seed=seed, epochs=epochs)
        results['GRU'] = eval_softmax(m, X_test, y_test)
        log(f"  GRU: acc={results['GRU']['accuracy']:.4f} f1={results['GRU']['f1_macro']:.4f} unc={results['GRU']['uncertainty_auroc']:.4f}")
    except Exception as e:
        import traceback; log(f"  [ERROR] GRU: {e}\n{traceback.format_exc()}")
        results['GRU'] = None

    # MCDropout
    try:
        log("\n[MCDropout] Training...")
        m = train_torch_model('mcdropout', X_train, y_train, X_val, y_val, input_dim, seed=seed, epochs=epochs)
        results['MCDropout'] = eval_mcdropout(m, X_test, y_test, n_samples=50)
        log(f"  MCDropout: acc={results['MCDropout']['accuracy']:.4f} f1={results['MCDropout']['f1_macro']:.4f} unc={results['MCDropout']['uncertainty_auroc']:.4f}")
    except Exception as e:
        import traceback; log(f"  [ERROR] MCDropout: {e}\n{traceback.format_exc()}")
        results['MCDropout'] = None

    # BNN
    try:
        log("\n[BNN] Training...")
        m = train_torch_model('bnn', X_train, y_train, X_val, y_val, input_dim, seed=seed, epochs=epochs)
        results['BNN'] = eval_bnn(m, X_test, y_test, n_samples=50)
        log(f"  BNN: acc={results['BNN']['accuracy']:.4f} f1={results['BNN']['f1_macro']:.4f} unc={results['BNN']['uncertainty_auroc']:.4f}")
    except Exception as e:
        import traceback; log(f"  [ERROR] BNN: {e}\n{traceback.format_exc()}")
        results['BNN'] = None

    # Climatology
    pi = y_train.mean()
    clim_preds = np.zeros(len(y_test), dtype=int)
    clim_probs = np.zeros((len(y_test), 2))
    clim_probs[:, 0] = 1 - pi; clim_probs[:, 1] = pi
    from sklearn.metrics import accuracy_score, f1_score, brier_score_loss
    results['Climatology'] = {
        'accuracy': float(accuracy_score(y_test, clim_preds)),
        'precision': 0.0, 'recall': 0.0,
        'f1_macro': float(f1_score(y_test, clim_preds, average='macro', zero_division=0)),
        'auc': 0.5,
        'brier': float(brier_score_loss(y_test, np.full(len(y_test), pi))),
        'ece': float(abs(pi - y_test.mean())),
        'uncertainty_auroc': 0.5,
    }
    log(f"  Climatology: acc={results['Climatology']['accuracy']:.4f} brier={results['Climatology']['brier']:.4f}")

    elapsed = time.time() - t0
    log(f"\n[Done] Seed {seed} completed in {elapsed:.1f}s")
    return results, elapsed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--all-seeds', dest='all_seeds', action='store_true')
    parser.add_argument('--split', type=str, default='temporal', choices=['temporal', 'random'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--resume', action='store_true', help='Resume from existing partial results')
    args = parser.parse_args()

    from config import RANDOM_SEEDS
    os.makedirs(RESULTS_DIR, exist_ok=True)
    seeds = list(RANDOM_SEEDS) if args.all_seeds else [args.seed]
    log(f"Args: all_seeds={args.all_seeds}, seeds_to_run={seeds}, resume={args.resume}")

    # Load existing partial results if resuming
    partial_file = os.path.join(RESULTS_DIR, f'fixed_results_{args.split}_partial.json')
    all_results = {}
    all_timings = {}
    if args.resume and os.path.exists(partial_file):
        try:
            with open(partial_file, 'r') as f:
                existing = json.load(f)
            all_results = existing.get('per_seed', {})
            all_timings = existing.get('timings', {})
            # Convert string keys back to int
            all_results = {int(k): v for k, v in all_results.items()}
            all_timings = {int(k): v for k, v in all_timings.items()}
            log(f"Loaded existing partial results: seeds_done={list(all_results.keys())}")
        except Exception as e:
            log(f"Warning: failed to load partial results: {e}")

    # Filter out already-completed seeds
    remaining_seeds = [s for s in seeds if s not in all_results]
    log(f"Seeds to run (after skipping completed): {remaining_seeds}")

    for s in remaining_seeds:
        try:
            res, elapsed = run_seed(s, split_mode=args.split, epochs=args.epochs)
            all_results[s] = res
            all_timings[s] = elapsed
            # Save after each seed
            with open(partial_file, 'w') as f:
                json.dump({'per_seed': {str(k): v for k, v in all_results.items()},
                           'timings': {str(k): v for k, v in all_timings.items()},
                           'split_mode': args.split,
                           'seeds_done': list(all_results.keys())},
                          f, indent=2, default=str)
            log(f"Saved partial results to {partial_file} (seeds done: {list(all_results.keys())})")
        except Exception as e:
            import traceback
            log(f"[FATAL] Seed {s}: {e}\n{traceback.format_exc()}")
            all_results[s] = {}
            all_timings[s] = 0.0

    # Aggregate over ALL seeds (including those loaded from partial)
    all_seeds = sorted(all_results.keys())
    log("\n" + "="*60)
    log(f"AGGREGATED RESULTS (mean +/- std) over seeds {all_seeds}")
    log("="*60)
    model_names = []
    for s in all_seeds:
        if all_results.get(s):
            for mname in all_results[s].keys():
                if mname not in model_names:
                    model_names.append(mname)
    agg = {}
    for mname in model_names:
        agg[mname] = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_macro', 'auc', 'brier', 'ece', 'uncertainty_auroc']:
            vals = []
            for s in all_seeds:
                if all_results.get(s) and all_results[s].get(mname) is not None:
                    if metric in all_results[s][mname]:
                        vals.append(float(all_results[s][mname][metric]))
            if vals:
                agg[mname][metric] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    'values': vals
                }
            else:
                agg[mname][metric] = {'mean': 0.0, 'std': 0.0, 'values': []}
        log(f"{mname:15s} acc={agg[mname]['accuracy']['mean']:.4f}+/-{agg[mname]['accuracy']['std']:.4f} "
            f"f1={agg[mname]['f1_macro']['mean']:.4f}+/-{agg[mname]['f1_macro']['std']:.4f} "
            f"ece={agg[mname]['ece']['mean']:.4f}+/-{agg[mname]['ece']['std']:.4f} "
            f"unc={agg[mname]['uncertainty_auroc']['mean']:.4f}+/-{agg[mname]['uncertainty_auroc']['std']:.4f}")

    out_file = os.path.join(RESULTS_DIR, f'fixed_results_{args.split}_all.json')
    with open(out_file, 'w') as f:
        json.dump({'per_seed': {str(k): v for k, v in all_results.items()},
                   'aggregated': agg,
                   'timings': {str(k): v for k, v in all_timings.items()},
                   'split_mode': args.split, 'seeds': all_seeds, 'epochs': args.epochs},
                  f, indent=2, default=str)
    log(f"\nResults saved to {out_file}")
    log_file.close()
