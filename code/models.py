"""
Neural network models and baselines for rainfall prediction.
Includes EDL-UQ MLP, Bayesian NN, MC Dropout, LSTM, GRU, and wrappers for sklearn models.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# EDL-UQ MLP with Dirichlet-based Uncertainty Quantification
# ------------------------------------------------------------------------------

class EDLMLP(nn.Module):
    """
    Evidence Deep Learning MLP (with C1 climatology-anchored prior support).
    Outputs evidence vector e >= 0 (via softplus) such that Dirichlet parameters
    alpha = e + alpha_prior, where alpha_prior defaults to ones (Sensoy 2018) or
    can be set per-sample to n0 * climatology_frequency (C1, Theorem 1).
    """
    def __init__(self, input_dim, hidden_dims=[128,64,32], dropout_rate=0.3,
                 activation='relu', use_batch_norm=True,
                 evidence_activation='softplus', evidence_min=1e-6,
                 num_classes=2, prior_n0=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.evidence_min = evidence_min
        self.use_batch_norm = use_batch_norm
        self.prior_n0 = prior_n0  # C1: prior effective sample size

        act = {'relu': nn.ReLU, 'elu': nn.ELU, 'leaky_relu': nn.LeakyReLU}[activation]

        layers = []
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev = h

        self.backbone = nn.Sequential(*layers)
        self.evidence_layer = nn.Linear(prev, num_classes)

        # Evidence activation
        if evidence_activation == 'softplus':
            self.evidence_act = nn.Softplus()
        elif evidence_activation == 'relu':
            self.evidence_act = nn.ReLU()
        else:
            raise ValueError(evidence_activation)

    def forward(self, x):
        h = self.backbone(x)
        e = self.evidence_act(self.evidence_layer(h)) + self.evidence_min
        return e

    def predict_dirichlet(self, x, alpha_prior=None):
        """Return alpha parameters. If alpha_prior given (B,K), use C1 anchoring;
        otherwise fall back to uniform prior ones (Sensoy 2018)."""
        e = self.forward(x)
        if alpha_prior is None:
            alpha = e + 1.0
        else:
            alpha = e + alpha_prior
        return alpha

    def predict_probs(self, x, alpha_prior=None):
        """Expected probability under Dirichlet predictive."""
        alpha = self.predict_dirichlet(x, alpha_prior)
        return alpha / alpha.sum(dim=1, keepdim=True)

    def predict_uncertainty(self, x, alpha_prior=None):
        """
        Decompose uncertainty into:
        - Total uncertainty (predictive entropy)
        - Aleatoric uncertainty (expected data uncertainty)
        - Epistemic uncertainty (knowledge uncertainty / distributional uncertainty)
        Returns dict with tensors.
        """
        alpha = self.predict_dirichlet(x, alpha_prior)
        alpha0 = alpha.sum(dim=1, keepdim=True)
        probs = alpha / alpha0

        # Total uncertainty: entropy of predictive distribution
        H_total = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

        # Aleatoric: expected entropy of categorical under Dirichlet
        # E_{p~Dir}[H(p)] = sum_k (alpha_k/alpha_0) * (psi(alpha_0+1) - psi(alpha_k+1))
        digamma_alpha0 = torch.digamma(alpha0 + 1.0)
        digamma_alpha = torch.digamma(alpha + 1.0)
        H_alea = torch.sum(probs * (digamma_alpha0 - digamma_alpha), dim=1)

        # Epistemic = Total - Aleatoric
        H_epi = H_total - H_alea

        # Also return precision (confidence) = alpha_0
        precision = alpha0.squeeze(1)

        return {
            'probs': probs,
            'alpha': alpha,
            'H_total': H_total,
            'H_alea': H_alea,
            'H_epi': H_epi,
            'precision': precision,
        }


# ------------------------------------------------------------------------------
# MC Dropout MLP
# ------------------------------------------------------------------------------

class MCDropoutMLP(nn.Module):
    """Standard MLP with dropout at test time for MC Dropout UQ."""
    def __init__(self, input_dim, hidden_dims=[128,64,32], dropout_rate=0.3,
                 num_classes=2, activation='relu'):
        super().__init__()
        act = {'relu': nn.ReLU, 'elu': nn.ELU}[activation]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act())
            layers.append(nn.Dropout(dropout_rate))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict_with_uncertainty(self, x, n_samples=100):
        self.train()  # keep dropout active
        logits = torch.stack([self.forward(x) for _ in range(n_samples)], dim=0)
        probs = F.softmax(logits, dim=-1)
        mean_probs = probs.mean(dim=0)
        # Predictive entropy as uncertainty
        H = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
        # Epistemic approximated by variance of predicted probabilities
        var = probs.var(dim=0).sum(dim=1)
        return {'probs': mean_probs, 'H_total': H, 'H_epi': var, 'logits': logits}


# ------------------------------------------------------------------------------
# Bayesian Neural Network (Bayes-by-Backprop style)
# ------------------------------------------------------------------------------

class BayesianLinear(nn.Module):
    """Linear layer with Gaussian variational posterior over weights."""
    def __init__(self, in_features, out_features, prior_sigma=1.0, rho_init=-3.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).fill_(rho_init))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).fill_(rho_init))

    def forward(self, x):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        kl_weight = self._kl_gaussian(self.weight_mu, weight_sigma, 0, self.prior_sigma)
        kl_bias = self._kl_gaussian(self.bias_mu, bias_sigma, 0, self.prior_sigma)
        return kl_weight + kl_bias

    @staticmethod
    def _kl_gaussian(mu_q, sigma_q, mu_p, sigma_p):
        return torch.sum(torch.log(sigma_p / sigma_q) +
                         (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) - 0.5)


class BayesianMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128,64], num_classes=2,
                 prior_sigma=1.0, rho_init=-3.0):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = input_dim
        for h in hidden_dims:
            self.layers.append(BayesianLinear(prev, h, prior_sigma, rho_init))
            prev = h
        self.out_layer = BayesianLinear(prev, num_classes, prior_sigma, rho_init)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out_layer(x)

    def kl_divergence(self):
        kl = 0.0
        for layer in self.layers:
            kl += layer.kl_divergence()
        kl += self.out_layer.kl_divergence()
        return kl

    def predict_with_uncertainty(self, x, n_samples=100):
        self.eval()
        logits = torch.stack([self.forward(x) for _ in range(n_samples)], dim=0)
        probs = F.softmax(logits, dim=-1)
        mean_probs = probs.mean(dim=0)
        H = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
        var = probs.var(dim=0).sum(dim=1)
        return {'probs': mean_probs, 'H_total': H, 'H_epi': var, 'logits': logits}


# ------------------------------------------------------------------------------
# LSTM / GRU for tabular sequence (treat each sample as single-step sequence)
# ------------------------------------------------------------------------------

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 num_classes=2, dropout=0.3, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional)
        direction = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, input_dim) -> add sequence dim
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, (hn, cn) = self.lstm(x)
        h = hn[-1] if not self.lstm.bidirectional else torch.cat([hn[-2], hn[-1]], dim=1)
        h = self.dropout(h)
        return self.fc(h)


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 num_classes=2, dropout=0.3, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0,
                          bidirectional=bidirectional)
        direction = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, hn = self.gru(x)
        h = hn[-1] if not self.gru.bidirectional else torch.cat([hn[-2], hn[-1]], dim=1)
        h = self.dropout(h)
        return self.fc(h)


# ------------------------------------------------------------------------------
# Sklearn Model Wrappers for Unified Interface
# ------------------------------------------------------------------------------

class SklearnWrapper:
    """Wrapper for sklearn models to provide predict_proba interface."""
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_uncertainty(self, X):
        """Return heuristic uncertainty: entropy of predictive distribution."""
        probs = self.predict_proba(X)
        H = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        return {'probs': probs, 'H_total': H, 'H_epi': np.zeros_like(H)}


# ------------------------------------------------------------------------------
# Model Factory
# ------------------------------------------------------------------------------

def build_model(model_name, input_dim, num_classes=2, **kwargs):
    """Factory function to build any model by name."""
    model_name = model_name.lower()
    if model_name in ['edl', 'edl-uq', 'edl_mlp']:
        return EDLMLP(input_dim, num_classes=num_classes, **kwargs)
    elif model_name in ['mcdropout', 'mc_dropout']:
        return MCDropoutMLP(input_dim, num_classes=num_classes, **kwargs)
    elif model_name in ['bnn', 'bayesian']:
        return BayesianMLP(input_dim, num_classes=num_classes, **kwargs)
    elif model_name == 'lstm':
        return LSTMClassifier(input_dim, num_classes=num_classes, **kwargs)
    elif model_name == 'gru':
        return GRUClassifier(input_dim, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Quick sanity check
    x = torch.randn(4, 10)
    model = EDLMLP(10, hidden_dims=[8,4], num_classes=2)
    out = model.predict_uncertainty(x)
    print("EDL output keys:", out.keys())
    print("Probs shape:", out['probs'].shape)
