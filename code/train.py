"""
Training script for EDL-UQ and baseline models.
Supports multi-seed experiments, early stopping, and checkpointing.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    DEVICE, RANDOM_SEEDS, TRAIN, LOSS, MODEL, BASELINES, OUTPUT,
    PREPROCESS, CHECKPOINT_DIR
)
from data_loader import preprocess_and_split, load_preprocessed_data, make_loaders
from models import build_model, SklearnWrapper

# ------------------------------------------------------------------------------
# Loss Functions
# ------------------------------------------------------------------------------

def edl_loss(alpha, y, lambda_reg=0.001, annealing_factor=1.0,
             loss_type='cross_entropy', alpha_prior=None):
    """
    EDL loss: digamma CE + masked KL.
    alpha: (batch, K) Dirichlet params
    y: (batch,) labels
    alpha_prior: (batch, K) prior for C1; None => uniform
    """
    K = alpha.size(1)
    alpha0 = alpha.sum(dim=1, keepdim=True)
    probs = alpha / alpha0

    if loss_type == 'cross_entropy':
        # digamma form (Sensoy Eq.5)
        y_onehot = F.one_hot(y, num_classes=K).float()
        digamma_alpha0 = torch.digamma(alpha0)
        digamma_alpha = torch.digamma(alpha)
        ce_loss = torch.sum(y_onehot * (digamma_alpha0 - digamma_alpha), dim=1)
        ce_loss = ce_loss.mean()
    elif loss_type == 'mse':
        y_onehot = F.one_hot(y, num_classes=K).float()
        error = (y_onehot - probs) ** 2
        var = probs * (1 - probs) / (alpha0 + 1)
        ce_loss = (error + var).sum(dim=1).mean()
    else:
        raise ValueError(loss_type)

    # masked KL: alpha_tilde = y * alpha_prior + (1-y) * alpha
    # true-class evidence gets no KL gradient
    y_onehot = F.one_hot(y, num_classes=K).float()
    if alpha_prior is None:
        alpha_prior = torch.ones_like(alpha)
    alpha_tilde = y_onehot * alpha_prior + (1.0 - y_onehot) * alpha
    alpha0_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    alpha0_prior = alpha_prior.sum(dim=1, keepdim=True)

    kl = (torch.lgamma(alpha0_tilde) - torch.lgamma(alpha0_prior)
          - torch.sum(torch.lgamma(alpha_tilde) - torch.lgamma(alpha_prior), dim=1, keepdim=True)
          + torch.sum((alpha_tilde - alpha_prior) * (torch.digamma(alpha_tilde) - torch.digamma(alpha0_tilde)), dim=1, keepdim=True))
    kl = kl.mean()

    loss = ce_loss + annealing_factor * lambda_reg * kl
    return loss, ce_loss.item(), kl.item()


def get_annealing_factor(epoch, annealing_epochs=50, max_factor=1.0):
    """Linear annealing from 0 to max_factor."""
    if epoch < annealing_epochs:
        return min(1.0, epoch / annealing_epochs) * max_factor
    return max_factor


# ------------------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, lambda_reg, annealing_factor,
                loss_type='cross_entropy', grad_clip=1.0):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        alpha = model.predict_dirichlet(Xb)
        loss, ce, kl = edl_loss(alpha, yb, lambda_reg=lambda_reg,
                                 annealing_factor=annealing_factor,
                                 loss_type=loss_type)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
        total_ce += ce * Xb.size(0)
        total_kl += kl * Xb.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_ce / n, total_kl / n


@torch.no_grad()
def evaluate_epoch(model, loader, lambda_reg=0.0, annealing_factor=0.0,
                   loss_type='cross_entropy'):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        alpha = model.predict_dirichlet(Xb)
        loss, _, _ = edl_loss(alpha, yb, lambda_reg=lambda_reg,
                               annealing_factor=annealing_factor,
                               loss_type=loss_type)
        total_loss += loss.item() * Xb.size(0)
        probs = alpha / alpha.sum(dim=1, keepdim=True)
        preds = probs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(yb.cpu().numpy())
    n = len(loader.dataset)
    acc = (np.concatenate(all_preds) == np.concatenate(all_labels)).mean()
    return total_loss / n, acc


def train_edl_model(X_train, y_train, X_val, y_val, input_dim,
                    seed=42, config=None, ablation_name='full_model'):
    """Train EDL-UQ model with given configuration."""
    if config is None:
        config = {
            'hidden_dims': MODEL['hidden_dims'],
            'dropout_rate': MODEL['dropout_rate'],
            'lambda_reg': LOSS['lambda_reg'],
            'annealing': LOSS['annealing'],
            'annealing_epochs': LOSS['annealing_epochs'],
            'loss_type': 'cross_entropy',
            'lr': TRAIN['learning_rate'],
            'weight_decay': TRAIN['weight_decay'],
            'batch_size': TRAIN['batch_size'],
            'epochs': TRAIN['epochs'],
            'patience': TRAIN['early_stopping_patience'],
        }

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, _ = make_loaders(
        X_train, y_train, X_val, y_val, X_val, y_val,
        batch_size=config['batch_size'], num_workers=0
    )

    model = build_model('edl', input_dim, num_classes=2,
                        hidden_dims=config['hidden_dims'],
                        dropout_rate=config['dropout_rate']).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(config['epochs']):
        af = get_annealing_factor(epoch, config['annealing_epochs']) if config['annealing'] else 1.0
        if ablation_name == 'no_annealing':
            af = 1.0
        if ablation_name == 'no_kl_regularization':
            lr = 0.0
        else:
            lr = config['lambda_reg']

        train_loss, train_ce, train_kl = train_epoch(
            model, train_loader, optimizer, lr, af,
            loss_type=config['loss_type'], grad_clip=TRAIN.get('gradient_clip_val', 1.0)
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, lambda_reg=lr, annealing_factor=af,
            loss_type=config['loss_type']
        )
        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']} | train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} af={af:.3f}", flush=True)

        if patience_counter >= config['patience']:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_torch_baseline(model_name, X_train, y_train, X_val, y_val, input_dim,
                         seed=42, epochs=100, batch_size=256, lr=1e-3,
                         weight_decay=1e-5, patience=20, **model_kwargs):
    """Train LSTM, GRU, MC Dropout, or BNN baseline."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, _ = make_loaders(
        X_train, y_train, X_val, y_val, X_val, y_val,
        batch_size=batch_size, num_workers=0
    )

    model = build_model(model_name, input_dim, num_classes=2, **model_kwargs).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            if hasattr(model, 'kl_divergence'):
                loss += model.kl_divergence() / len(X_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits = model(Xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)
        train_loss /= len(train_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [{model_name.upper()}] Epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if patience_counter >= patience:
            print(f"  [{model_name.upper()}] Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_sklearn_baseline(model_name, X_train, y_train, seed=42, use_class_weight=False):
    """Train sklearn-based baselines: LR, RF, XGBoost.
    class_weight is optional (default False for fair comparison with EDL-UQ
    which has no class weighting). Set use_class_weight=True to reproduce the original
    'handcuffed' baseline behavior."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    try:
        import xgboost as xgb
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

    np.random.seed(seed)
    cw = 'balanced' if use_class_weight else None
    if model_name.lower() == 'logisticregression':
        model = LogisticRegression(max_iter=1000, class_weight=cw, solver='lbfgs')
    elif model_name.lower() == 'randomforest':
        model = RandomForestClassifier(n_estimators=200, max_depth=20,
                                       min_samples_split=5,
                                       class_weight=('balanced_subsample' if use_class_weight else None),
                                       n_jobs=-1, random_state=seed)
    elif model_name.lower() == 'xgboost':
        if not HAS_XGB:
            print("[Warning] XGBoost not installed, using RandomForest instead.")
            model = RandomForestClassifier(n_estimators=200, max_depth=20,
                                           class_weight=('balanced_subsample' if use_class_weight else None),
                                           n_jobs=-1, random_state=seed)
        else:
            model = xgb.XGBClassifier(n_estimators=200, max_depth=6,
                                       learning_rate=0.1, subsample=0.8,
                                       colsample_bytree=0.8,
                                       scale_pos_weight=(3.46 if use_class_weight else 1.0),
                                       eval_metric='logloss',
                                       use_label_encoder=False,
                                       random_state=seed, n_jobs=-1)
    else:
        raise ValueError(model_name)

    print(f"[Sklearn] Training {model_name} (class_weight={use_class_weight}) ...")
    wrapper = SklearnWrapper(model)
    wrapper.fit(X_train, y_train)
    return wrapper


# ------------------------------------------------------------------------------
# Main Multi-Seed Training Orchestrator
# ------------------------------------------------------------------------------

def run_all_experiments(force_reprocess=False):
    """Run full experimental pipeline across all seeds and models."""
    # Preprocess once with first seed (splits will vary per seed)
    if force_reprocess or not os.path.exists(OUTPUT['split_indices']):
        print("=" * 60)
        print("PREPROCESSING (seed=42)")
        print("=" * 60)
        preprocess_and_split(seed=42, save=True)

    # Load base processed data (we'll re-split per seed)
    results = {
        'edl': {},
        'baselines': {},
        'ablation': {},
        'sensitivity': {},
    }

    for seed in RANDOM_SEEDS:
        print("\n" + "=" * 60)
        print(f"SEED = {seed}")
        print("=" * 60)

        # Re-split with current seed
        X_train, X_val, X_test, y_train, y_val, y_test, _, _, feature_names = preprocess_and_split(seed=seed, save=False)
        input_dim = X_train.shape[1]
        print(f"Input dimension: {input_dim}")

        # ---- EDL-UQ ----
        print("\n[Train] EDL-UQ")
        edl_model, edl_hist = train_edl_model(
            X_train, y_train, X_val, y_val, input_dim, seed=seed
        )
        edl_path = os.path.join(CHECKPOINT_DIR, f"edl_seed{seed}.pth")
        torch.save(edl_model.state_dict(), edl_path)
        results['edl'][seed] = {'model_path': edl_path, 'history': edl_hist}

        # ---- Baselines ----
        baselines_to_train = {
            'LogisticRegression': lambda: train_sklearn_baseline('LogisticRegression', X_train, y_train, seed),
            'RandomForest': lambda: train_sklearn_baseline('RandomForest', X_train, y_train, seed),
            'XGBoost': lambda: train_sklearn_baseline('XGBoost', X_train, y_train, seed),
            'LSTM': lambda: train_torch_baseline('lstm', X_train, y_train, X_val, y_val, input_dim,
                                                  seed=seed, **BASELINES['LSTM']),
            'GRU': lambda: train_torch_baseline('gru', X_train, y_train, X_val, y_val, input_dim,
                                                 seed=seed, **BASELINES['GRU']),
            'BNN': lambda: train_torch_baseline('bnn', X_train, y_train, X_val, y_val, input_dim,
                                                 seed=seed, **BASELINES['BayesianNN']),
            'MCDropout': lambda: train_torch_baseline('mcdropout', X_train, y_train, X_val, y_val, input_dim,
                                                       seed=seed, **BASELINES['MCDropout']),
        }

        for bname, btrain in baselines_to_train.items():
            print(f"\n[Train] Baseline: {bname}")
            try:
                bmodel = btrain()
                bpath = os.path.join(CHECKPOINT_DIR, f"{bname.lower()}_seed{seed}.pth")
                if isinstance(bmodel, SklearnWrapper):
                    pickle.dump(bmodel, open(bpath.replace('.pth', '.pkl'), 'wb'))
                else:
                    torch.save(bmodel.state_dict(), bpath)
                if seed not in results['baselines']:
                    results['baselines'][seed] = {}
                results['baselines'][seed][bname] = {'model_path': bpath}
            except Exception as e:
                print(f"  [ERROR] {bname} failed: {e}")

    # Save results metadata
    os.makedirs(os.path.dirname(OUTPUT['results_json']), exist_ok=True)
    with open(OUTPUT['results_json'], 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\n[Done] All training completed. Results saved.")
    return results


if __name__ == "__main__":
    run_all_experiments(force_reprocess=True)
