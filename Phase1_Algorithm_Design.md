# Phase 1: Algorithm Design Document
## Direction 3 -- 17_Evidence_Rainfall

**Research Topic**: Evidence Deep Learning with Uncertainty Quantification (EDL-UQ) for Rainfall Prediction
**Target Journal**: Environmental Modelling & Software
**Dataset**: Australian Weather (weatherAUS.csv)

---

## 1. Dataset Analysis

### 1.1 Basic Statistics

| Property | Value |
|----------|-------|
| Total Samples | 145,460 |
| Features (raw) | 23 (16 numeric + 7 categorical/object) |
| Target | RainTomorrow (binary: Yes/No) |
| Date Range | 2007-11-01 to 2017-06-25 (3,524 days) |
| Locations | 49 unique weather stations |
| Valid Samples (after dropping target-NA) | 142,193 |
| Class Distribution (No / Yes) | 110,316 / 31,877 |
| Imbalance Ratio (No:Yes) | 3.46 : 1 |

### 1.2 Feature Inventory

**Numeric Features (16)**

| Feature | Description | Missing Rate | Correlation with Target |
|---------|-------------|--------------|------------------------|
| MinTemp | Minimum temperature (C) | 1.02% | +0.084 |
| MaxTemp | Maximum temperature (C) | 0.87% | -0.159 |
| Rainfall | Daily rainfall (mm) | 2.24% | +0.239 |
| Evaporation | Daily evaporation (mm) | 43.17% | -0.119 |
| Sunshine | Bright sunshine hours | 48.01% | -0.451 |
| WindGustSpeed | Strongest wind gust (km/h) | 7.06% | +0.234 |
| WindSpeed9am | Wind speed at 9am (km/h) | 1.21% | +0.091 |
| WindSpeed3pm | Wind speed at 3pm (km/h) | 2.11% | +0.088 |
| Humidity9am | Humidity at 9am (%) | 1.82% | +0.257 |
| Humidity3pm | Humidity at 3pm (%) | 3.10% | +0.446 |
| Pressure9am | Atmospheric pressure at 9am (hPa) | 10.36% | -0.246 |
| Pressure3pm | Atmospheric pressure at 3pm (hPa) | 10.33% | -0.226 |
| Cloud9am | Cloud cover at 9am (oktas) | 38.42% | +0.317 |
| Cloud3pm | Cloud cover at 3pm (oktas) | 40.81% | +0.382 |
| Temp9am | Temperature at 9am (C) | 1.21% | -0.026 |
| Temp3pm | Temperature at 3pm (C) | 2.48% | -0.192 |

**Categorical Features (5)**

| Feature | Categories | Missing Rate | Encoding Strategy |
|---------|-----------|--------------|-------------------|
| Location | 49 cities | 0% | One-Hot |
| WindGustDir | 16 cardinal directions | 7.10% | One-Hot + NA indicator |
| WindDir9am | 16 cardinal directions | 7.26% | One-Hot + NA indicator |
| WindDir3pm | 16 cardinal directions | 2.91% | One-Hot + NA indicator |
| RainToday | Yes/No | 2.24% | Binary (0/1) + NA indicator |

### 1.3 Data Quality Assessment

1. **Severe Missing Values**: Four features have >38% missing data (Sunshine 48.0%, Evaporation 43.2%, Cloud3pm 40.8%, Cloud9am 38.4%). These will be dropped rather than imputed to avoid injecting excessive noise.
2. **Class Imbalance**: Positive class (Rain) accounts for only 22.4% of valid samples, necessitating class-weighting or resampling.
3. **Temporal Structure**: Date field enables seasonality extraction (month, day-of-year, season).
4. **Multi-Station Data**: 49 locations with varying climates; location encoding captures geographic heterogeneity.

---

## 2. EDL-UQ Algorithm Framework

### 2.1 Core Idea

Traditional neural networks output point-estimate probabilities via softmax, which conflate two distinct sources of uncertainty:
- **Epistemic (cognitive) uncertainty**: uncertainty due to lack of knowledge / insufficient training data.
- **Aleatoric (aleatory) uncertainty**: uncertainty inherent in the data-generating process (irreducible noise).

Evidence Deep Learning (EDL) treats the network output as parameters of a Dirichlet distribution over class probabilities, enabling principled Bayesian reasoning and explicit uncertainty decomposition.

### 2.2 Subjective Opinion Model (Dirichlet Parameterization)

For a K-class classification problem (here K=2), the neural network outputs a vector of non-negative evidence:

```
e = f(x; theta) in R^K_+,   e_k >= 0
```

where `f` is the network with softplus/ReLU output activation. The evidence parameterizes a Dirichlet distribution:

```
Dir(p | alpha),   alpha_k = e_k + 1
```

The prior `alpha_0 = [1, 1, ..., 1]` corresponds to zero evidence (uniform prior). The strength of evidence is `S = sum_k alpha_k = sum_k e_k + K`.

**Predictive probability** (expected probability under the Dirichlet):

```
p_hat_k = E[p_k] = alpha_k / S = (e_k + 1) / (sum_j e_j + K)
```

**Variance of the probability estimate**:

```
Var(p_k) = alpha_k (S - alpha_k) / (S^2 (S + 1))
```

Large S => low variance => high confidence. Small S => high variance => high uncertainty.

### 2.3 Evidence Theory Loss Function

The total loss combines a classification term and an uncertainty regularization term with annealing:

```
L_total(t) = L_cls + lambda(t) * L_KL
```

**Classification Loss** (expected cross-entropy under Dirichlet, approximated):

```
L_cls = sum_{k=1}^K y_k * (log(S) - log(alpha_k))
```

This is derived from `E_{p~Dir}[ -sum_k y_k log(p_k) ] ≈ sum_k y_k (psi(S) - psi(alpha_k))`, where `psi` is the digamma function. For computational efficiency and stability, we use the log approximation which is tight for alpha_k >= 1.

**KL Regularization** (drives the posterior toward the uniform prior when evidence is weak):

```
L_KL = KL( Dir(alpha) || Dir(alpha_0) )
     = log( Gamma(S_0) / Gamma(S) )
       + sum_k log( Gamma(alpha_k) / Gamma(alpha_0k) )
       + sum_k (alpha_k - alpha_0k) * (psi(alpha_k) - psi(S))
```

where `alpha_0k = 1`, `S_0 = K`. In PyTorch: `torch.lgamma` and `torch.digamma`.

**Annealing Schedule** (critical for training stability):

```
lambda(t) = min(1.0, t / T_anneal) * lambda_max
```

Without annealing, the KL term dominates early training and prevents the model from accumulating evidence. `T_anneal` is typically 50 epochs.

**Alternative: EDL-MSE Loss**

For comparison, an MSE variant can be derived:

```
L_MSE = sum_k ( (y_k - p_hat_k)^2 + p_hat_k (1 - p_hat_k) / (S + 1) )
```

The second term is the Dirichlet variance, serving as a natural regularizer. This variant will be included in ablation studies.

### 2.4 Uncertainty Decomposition

Given Dirichlet parameters `alpha`, we decompose uncertainty into epistemic and aleatoric components using the mutual-information decomposition:

**Total Uncertainty** (Predictive Entropy):

```
H_total = H( p_hat ) = - sum_k p_hat_k * log(p_hat_k)
```

**Expected Entropy** (Aleatoric Uncertainty):

```
H_alea = E_{p~Dir}[ H(p) ]
       = - sum_k p_hat_k * ( psi(alpha_k + 1) - psi(S + 1) )
```

**Epistemic Uncertainty** (Mutual Information):

```
H_epi = H_total - H_alea = I(y, theta | x)
```

**Interpretation**:
- `H_alea` is high when the expected probability is near 0.5 (inherently ambiguous data, e.g., transitional weather).
- `H_epi` is high when the Dirichlet is broad (S is small), indicating the model has not seen enough similar training examples.
- For out-of-distribution inputs, `H_epi` will be high while `H_alea` may remain moderate.

**Simplified Uncertainty Metrics (for fast inference)**:

- **Vacuity** (cognitive uncertainty): `V = K / S` in [0, 1]. V=1 when no evidence; V->0 when evidence is overwhelming.
- **Dissonance** (aleatoric-style conflict): based on Josang's subjective logic, measuring conflict among belief masses.

For this study, we report both the full MI decomposition and vacuity for interpretability.

### 2.5 Essential Difference from Traditional Softmax

| Aspect | Softmax Output | EDL Dirichlet Output |
|--------|---------------|----------------------|
| Output type | Point probability p in Delta_K | Distribution over probabilities Dir(alpha) |
| Uncertainty | Derived ad-hoc (entropy of p) | Intrinsic to the model (width of Dirichlet) |
| Epistemic/Aleatoric separation | Not possible | Natural decomposition via MI |
| Overconfidence | Prone (overfits to training labels) | Regularized by KL term; high uncertainty on OOD |
| Calibration | Requires post-hoc (temperature scaling) | Naturally better calibrated due to Bayesian treatment |
| Decision threshold | Fixed at 0.5 | Can incorporate uncertainty for rejection |

### 2.6 Network Architecture

```
Input (D_dim)
    |
    v
Linear(D_dim, 128) -> BatchNorm -> ReLU -> Dropout(0.3)
    |
    v
Linear(128, 64)    -> BatchNorm -> ReLU -> Dropout(0.3)
    |
    v
Linear(64, 32)     -> BatchNorm -> ReLU -> Dropout(0.3)
    |
    v
Linear(32, 2)      -> Softplus(evidence_min=1e-6)
    |
    v
Evidence e = [e_0, e_1]
    |
    v
alpha = e + 1,  p_hat = alpha / sum(alpha)
```

Expected input dimension after preprocessing: ~80 (16 numeric + ~64 OHE categorical).

---

## 3. Data Preprocessing Pipeline

### 3.1 Step-by-Step Flow

```
Raw CSV (145,460 x 23)
    |
    v
[1] Drop rows with missing target (RainTomorrow)
    => 142,193 rows
    |
    v
[2] Drop high-missing columns (>40%)
    Drop: Evaporation, Sunshine, Cloud9am, Cloud3pm
    Remaining: 19 columns
    |
    v
[3] Feature Engineering
    - Parse Date -> month (1-12), dayofyear (1-365), season (4 categories)
    - Cyclical encoding: sin(2*pi*month/12), cos(2*pi*month/12)
    |
    v
[4] Missing Value Imputation
    Numeric: median imputation (robust to outliers)
    Categorical: mode imputation
    Add binary NA-indicator columns for imputed values
    |
    v
[5] Categorical Encoding
    One-Hot Encoding for: Location (49), WindGustDir (16), WindDir9am (16), WindDir3pm (16), season (4)
    Binary encoding for: RainToday (Yes/No)
    Group rare categories if frequency < 10
    |
    v
[6] Numeric Scaling
    StandardScaler (zero mean, unit variance)
    Fit on training set only; transform val/test
    |
    v
[7] Class Balancing
    Compute class_weight = {No: 1.0, Yes: 3.46}
    Passed to loss function (no resampling to preserve temporal structure)
    |
    v
[8] Train/Val/Test Split
    Stratified split: 70% train / 15% val / 15% test
    Random seeds: [42, 123, 456, 789, 2024] for statistical robustness
    |
    v
Processed Tensors -> DataLoader
```

### 3.2 Preprocessing Rationale

- **Drop high-missing columns**: Imputing >40% missing data introduces more noise than signal. Sunshine and Evaporation are also indirectly captured by other features (Humidity, Temp, Rainfall).
- **Median imputation**: More robust than mean for meteorological data which often contains skewed distributions.
- **NA indicators**: Preserve information about which values were missing; missingness itself may be predictive.
- **No temporal leakage**: Scaling and encoding fitted strictly on training data.
- **Stratified split**: Preserves class distribution across splits.

---

## 4. Evaluation Protocol

### 4.1 Classification Performance Metrics

| Metric | Symbol | Formula / Description |
|--------|--------|----------------------|
| Accuracy | ACC | (TP + TN) / (TP + TN + FP + FN) |
| Precision | P | TP / (TP + FP) |
| Recall | R | TP / (TP + FN) |
| F1-Macro | F1_mac | Harmonic mean of P and R, averaged over classes |
| F1-Micro | F1_mic | F1 computed from global TP/FP/FN |
| AUC-ROC | AUC | Area under ROC curve |
| Average Precision | AP | Area under precision-recall curve |

### 4.2 Calibration Metrics

| Metric | Symbol | Description |
|--------|--------|-------------|
| Expected Calibration Error | ECE | Average gap between predicted confidence and observed accuracy, across M bins (M=15) |
| Brier Score | BS | Mean squared error between predicted probability and true outcome |
| Negative Log-Likelihood | NLL | -sum log(p_hat_y), measures probabilistic quality |

```
ECE = sum_{m=1}^M ( |B_m| / N ) * | acc(B_m) - conf(B_m) |
```

### 4.3 Uncertainty Quality Metrics

| Metric | Description |
|--------|-------------|
| Uncertainty AUROC | Use uncertainty score to discriminate correct vs incorrect predictions. High AUROC = uncertainty is informative. |
| Uncertainty AUPR | PR-curve variant for error detection via uncertainty. |
| Sharpness | Average variance of predicted probabilities; measures calibration sharpness. |
| NLL-Dirichlet | Negative log-likelihood under the Dirichlet predictive distribution. |
| Reliability Diagram | Visual comparison of predicted confidence vs observed accuracy. |

**Error Detection Protocol**:
1. Compute `H_total` for each test sample.
2. Sort samples by descending uncertainty.
3. Label top-X% uncertain samples as "rejected".
4. Measure accuracy on retained samples vs rejected samples.
5. Compute AUROC using uncertainty as score and correctness as label.

### 4.4 Statistical Testing

- **Multi-seed evaluation**: 5 random seeds; report mean +/- std.
- **Paired test**: Wilcoxon signed-rank test for comparing EDL-UQ vs each baseline.
- **Effect size**: Cohen's d for magnitude of improvement.
- **95% Confidence Intervals**: Bootstrap (n=1000) for all reported metrics.

---

## 5. Baseline Methods

| Method | Type | Uncertainty Mechanism | Implementation Notes |
|--------|------|----------------------|----------------------|
| Logistic Regression | Classical linear | None (baseline discriminant) | sklearn, class_weight='balanced' |
| Random Forest | Ensemble | Entropy of tree votes | sklearn, 200 estimators |
| XGBoost | Gradient boosting | None (point estimate) | xgboost, scale_pos_weight=3.46 |
| LSTM | Deep sequential | None | PyTorch, 7-day input window |
| GRU | Deep sequential | None | PyTorch, 7-day input window |
| Bayesian NN | Deep Bayesian | Variational inference (Bayes-by-Backprop) | PyTorch, 100 MC samples |
| MC Dropout | Deep approximate Bayesian | Dropout at test time | PyTorch, 100 forward passes |
| EDL-UQ (Ours) | Evidence Deep Learning | Dirichlet parameterization | PyTorch, full MI decomposition |

**Methodological Differences for Paper**:
- LSTM/GRU use temporal sequences (7-day windows) while others use static features; this comparison is fair because the task admits both formulations.
- Bayesian NN and MC Dropout provide uncertainty estimates but cannot naturally decompose epistemic vs aleatoric uncertainty.
- EDL-UQ provides the only framework that natively parameterizes a distribution over class probabilities with explicit uncertainty decomposition.

---

## 6. Ablation Study Design

| Ablation Variant | Modification | Purpose |
|-----------------|-------------|---------|
| Full Model | EDL + KL + annealing | Main proposed method |
| No KL Regularization | lambda = 0 | Validate necessity of KL term |
| No Annealing | lambda fixed at epoch 1 | Validate necessity of annealing |
| Softmax Baseline | Replace Dirichlet with softmax | Show benefit of distributional output |
| EDL-MSE | Use MSE loss instead of cross-entropy | Compare loss formulations |

**Hyperparameter Ablation**:
- lambda_reg in [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
- hidden_dims in [[64], [128,64], [128,64,32], [256,128,64]]
- dropout in [0.0, 0.1, 0.3, 0.5]
- learning_rate in [1e-4, 5e-4, 1e-3, 5e-3]

---

## 7. Sensitivity Analysis Design

For each parameter, compute **elasticity** of F1-macro with respect to parameter perturbation:

```
E(p) = (Delta Metric / Metric) / (Delta p / p)
```

| Sensitivity Level | Elasticity Range |
|-------------------|------------------|
| High | |E| > 0.5 |
| Medium | 0.2 <= |E| <= 0.5 |
| Low | |E| < 0.2 |

Parameters analyzed: lambda_reg, dropout_rate, learning_rate, hidden_dim.

---

## 8. Robustness Analysis Design

| Corruption Type | Levels | Description |
|-----------------|--------|-------------|
| Gaussian Noise | std = [0.0, 0.01, 0.05, 0.10, 0.15] * feature_std | Add noise to numeric features |
| Random Missing | rates = [0.0, 0.05, 0.10, 0.20, 0.30] | Randomly zero-out features |

Measure degradation curves for Accuracy, F1, ECE, and uncertainty quality.

---

## 9. File Outputs (Planned)

```
17_Evidence_Rainfall/
├── code/
│   ├── config.py              # [DONE] All hyperparameters
│   ├── data_loader.py         # Preprocessing pipeline
│   ├── models.py              # EDL-UQ + baseline architectures
│   ├── train.py               # Training loops
│   ├── evaluate.py            # Metric computation
│   ├── visualize.py           # Plot generation
│   └── requirements.txt       # Dependencies
├── data/
│   └── processed/
├── results/
│   ├── main_results.json
│   ├── baseline_results.json
│   ├── ablation_results.json
│   ├── sensitivity_results.json
│   ├── robustness_results.json
│   ├── uncertainty_analysis.json
│   └── plots/
│       ├── fig1_architecture.png
│       ├── fig2_method_comparison.png
│       ├── fig3_ablation.png
│       ├── fig4_sensitivity.png
│       ├── fig5_uncertainty_decomposition.png
│       ├── fig6_reliability_diagram.png
│       └── fig7_feature_tsne.png
├── checkpoints/
│   └── best_model.pth
└── paper/
    ├── paper_draft.md
    ├── cover_letter.md
    └── highlights.md
```

---

## 10. Innovation Points

1. **First application of full EDL-UQ with MI-based uncertainty decomposition** to operational rainfall prediction on the Australian Weather dataset, providing both epistemic and aleatoric uncertainty estimates for each forecast.
2. **Systematic comparison with 7 baselines** including Bayesian NN and MC Dropout, demonstrating superior calibration (lower ECE) and uncertainty-aware error detection (higher uncertainty-AUROC).
3. **Comprehensive ablation and sensitivity analysis** with elasticity-based parameter importance ranking, providing actionable guidance for practitioners deploying EDL-UQ in environmental modelling software.
4. **Annealing-schedule analysis** showing its necessity for stable EDL training on imbalanced environmental datasets.

---

*Document generated for Phase 1 algorithm design. All parameters and designs are aligned with config.py.*
