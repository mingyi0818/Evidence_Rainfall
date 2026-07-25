"""
Configuration file for EDL-UQ rainfall prediction experiments.
Evidence Deep Learning with Uncertainty Quantification for binary classification
on the Australian Weather dataset (RainTomorrow prediction).

Author: Zeng Jingyuan, Guo Jianghong, Jiang Chuanxian, Feng Yafen
Affiliation: Jiaying University, Meizhou, Guangdong, China
"""

import os
import torch

# ------------------------------------------------------------------------------
# Hardware & Reproducibility
# ------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Force CPU for debugging
DEVICE = torch.device("cpu")
RANDOM_SEEDS = [42, 123, 456, 789, 2024]  # multi-seed for statistical analysis
NUM_WORKERS = 4

# ------------------------------------------------------------------------------
# Data Paths
# ------------------------------------------------------------------------------
DATA_DIR = r"D:\datasets\timeseries\Rain_Australia"
RAW_CSV = os.path.join(DATA_DIR, "weatherAUS.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "plots")

for d in [PROCESSED_DIR, RESULTS_DIR, CHECKPOINT_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------------------------
# Dataset Properties (auto-detected from weatherAUS.csv)
# ------------------------------------------------------------------------------
DATASET_NAME = "AustralianWeather"
NUM_SAMPLES = 145460
NUM_FEATURES_RAW = 23
TARGET_COL = "RainTomorrow"
DATE_COL = "Date"
LOCATION_COL = "Location"

# Class distribution after dropping target-missing rows:
# No: 110316, Yes: 31877  =>  imbalance ratio ~3.46:1
NUM_CLASSES = 2
CLASS_NAMES = ["NoRain", "Rain"]
POSITIVE_CLASS = "Yes"

# Columns by type (derived from data profiling)
NUMERIC_COLS = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm",
    "Temp9am", "Temp3pm",
]

CATEGORICAL_COLS = [
    "Location",          # 49 unique locations
    "WindGustDir",       # 16 cardinal directions + NaN
    "WindDir9am",        # 16 cardinal directions + NaN
    "WindDir3pm",        # 16 cardinal directions + NaN
    "RainToday",         # Yes/No
]

# Missing-value rates (%) observed in raw data:
# Sunshine 48.0, Evaporation 43.2, Cloud3pm 40.8, Cloud9am 38.4,
# Pressure9am 10.4, Pressure3pm 10.3, WindGustDir 7.1, WindGustSpeed 7.1,
# WindDir9am 7.3, Humidity3pm 3.1, WindDir3pm 2.9, Temp3pm 2.5,
# RainTomorrow 2.2, Rainfall 2.2, RainToday 2.2, WindSpeed3pm 2.1,
# Humidity9am 1.8, WindSpeed9am 1.2, Temp9am 1.2, MinTemp 1.0, MaxTemp 0.9

# ------------------------------------------------------------------------------
# Preprocessing Strategy
# ------------------------------------------------------------------------------
PREPROCESS = {
    # Missing-value handling
    "missing_strategy_numeric": "median",      # options: "median", "mean", "knn"
    "missing_strategy_categorical": "mode",    # options: "mode", "constant"
    "missing_constant": "Missing",

    # High-missing columns: drop vs impute
    # Columns with >40% missing are dropped by default to reduce noise.
    "drop_high_missing": True,
    "missing_drop_threshold": 0.40,
    "drop_cols": ["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"],

    # Feature engineering
    "extract_date_features": True,
    "date_features": ["month", "dayofyear", "season"],

    # Encoding
    "categorical_encoder": "onehot",           # "onehot" or "target"
    "min_category_frequency": 10,              # group rare categories

    # Scaling
    "numeric_scaler": "standard",              # "standard" or "minmax"

    # Class imbalance
    "balance_strategy": "class_weight",        # "class_weight", "oversample", "none"
    "pos_class_weight": 3.46,                  # approx ratio of negative/positive

    # Train/val/test split
    "test_size": 0.15,
    "val_size": 0.15,
    "stratify": True,
    "shuffle": True,
}

# ------------------------------------------------------------------------------
# EDL-UQ Model Architecture
# ------------------------------------------------------------------------------
MODEL = {
    "name": "EDL_UQ_MLP",
    "input_dim": None,          # set dynamically after preprocessing (~80 after OHE)
    "hidden_dims": [128, 64, 32],
    "dropout_rate": 0.3,
    "activation": "relu",
    "use_batch_norm": True,

    # Evidence output layer
    "evidence_activation": "softplus",   # "softplus" or "relu"; ensures non-negative evidence
    "evidence_min": 1e-6,                # floor for numerical stability

    # Prior parameters for Dirichlet
    "prior_alpha": 1.0,                  # uniform prior => alpha_0 = [1,1]
    "num_classes": NUM_CLASSES,
}

# ------------------------------------------------------------------------------
# EDL Loss Function
# ------------------------------------------------------------------------------
LOSS = {
    "classification_loss": "edl_cross_entropy",   # E[p~Dir][CE(p,y)] approx
    "regularization": "kl_divergence",            # KL(Dir(alpha) || Dir(alpha_0))
    "lambda_reg": 0.001,                          # weight of KL term
    "annealing": True,
    "annealing_epochs": 50,
    "annealing_max": 1.0,
    "use_mse_loss": False,                        # if True, use EDL-MSE instead
}

# ------------------------------------------------------------------------------
# Training Hyperparameters
# ------------------------------------------------------------------------------
TRAIN = {
    "batch_size": 256,
    "epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "optimizer": "adam",
    "scheduler": "reduce_on_plateau",    # "reduce_on_plateau", "cosine", "none"
    "scheduler_patience": 10,
    "scheduler_factor": 0.5,
    "early_stopping_patience": 8,
    "early_stopping_metric": "val_loss",
    "gradient_clip_val": 1.0,
}

# ------------------------------------------------------------------------------
# Baseline Model Configs
# ------------------------------------------------------------------------------
BASELINES = {
    "LogisticRegression": {
        "enabled": True,
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "lbfgs",
    },
    "RandomForest": {
        "enabled": True,
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
    },
    "XGBoost": {
        "enabled": True,
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 3.46,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    },
    "LSTM": {
        "enabled": True,
        "input_size": None,       # set dynamically
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": False,
        "sequence_length": 7,     # use past 7 days as sequence
    },
    "GRU": {
        "enabled": True,
        "input_size": None,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": False,
        "sequence_length": 7,
    },
    "BayesianNN": {
        "enabled": True,
        "hidden_dims": [128, 64],
        "prior_sigma": 1.0,
        "posterior_rho_init": -3.0,
        "num_mc_samples": 100,    # Monte-Carlo samples for prediction
    },
    "MCDropout": {
        "enabled": True,
        "hidden_dims": [128, 64, 32],
        "dropout_rate": 0.3,
        "num_mc_samples": 100,
    },
}

# ------------------------------------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------------------------------------
METRICS = {
    # Classification performance
    "accuracy": True,
    "precision": True,
    "recall": True,
    "f1_macro": True,
    "f1_micro": True,
    "auc_roc": True,
    "average_precision": True,

    # Calibration metrics
    "ece": True,              # Expected Calibration Error
    "mce": False,             # Maximum Calibration Error
    "brier_score": True,
    "nll": True,              # Negative Log-Likelihood

    # Uncertainty quality metrics
    "uncertainty_auroc": True,        # AUROC using uncertainty to detect errors
    "uncertainty_aupr": True,         # AUPR using uncertainty to detect errors
    "reliability_diagram": True,      # generate reliability plot
    "sharpness": True,                # average predicted probability variance
    "nll_dirichlet": True,            # NLL under Dirichlet predictive

    # Uncertainty decomposition thresholds
    "high_uncertainty_threshold": 0.5,   # for error detection analysis
}

# ECE binning
ECE_N_BINS = 15

# ------------------------------------------------------------------------------
# Ablation Study Configuration
# ------------------------------------------------------------------------------
ABLATION = {
    # Component ablation
    "components": [
        "full_model",                       # EDL with KL regularization + annealing
        "no_kl_regularization",             # remove KL term (lambda=0)
        "no_annealing",                     # fixed lambda from epoch 1
        "softmax_baseline",                 # replace Dirichlet with softmax
        "mse_evidence",                     # EDL-MSE loss instead of cross-entropy
    ],

    # Hyperparameter ablation
    "lambda_reg_values": [0.0, 1e-4, 1e-3, 1e-2, 1e-1],
    "hidden_dims_values": [[64], [128,64], [128,64,32], [256,128,64]],
    "dropout_values": [0.0, 0.1, 0.3, 0.5],
    "learning_rate_values": [1e-4, 5e-4, 1e-3, 5e-3],
}

# ------------------------------------------------------------------------------
# Sensitivity Analysis
# ------------------------------------------------------------------------------
SENSITIVITY = {
    "parameters": [
        {"name": "lambda_reg", "values": [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]},
        {"name": "dropout_rate", "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        {"name": "learning_rate", "values": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]},
        {"name": "hidden_dim", "values": [32, 64, 128, 256]},
    ],
    "metric_for_best": "val_f1_macro",
    "elasticity_delta": 0.1,   # relative perturbation for elasticity computation
}

# ------------------------------------------------------------------------------
# Robustness Analysis
# ------------------------------------------------------------------------------
ROBUSTNESS = {
    "noise_levels": [0.0, 0.01, 0.05, 0.10, 0.15],   # Gaussian noise std relative to feature std
    "missing_rates": [0.0, 0.05, 0.10, 0.20, 0.30],   # random feature deletion
    "feature_corruption": ["none", "Gaussian", "missing"],
}

# ------------------------------------------------------------------------------
# Logging & Reproducibility
# ------------------------------------------------------------------------------
LOG = {
    "log_interval_steps": 50,
    "save_best_only": True,
    "verbose": 1,
    "tensorboard": False,
    "log_file": os.path.join(RESULTS_DIR, "training.log"),
}

# ------------------------------------------------------------------------------
# Output Filenames
# ------------------------------------------------------------------------------
OUTPUT = {
    "processed_data": os.path.join(PROCESSED_DIR, "weather_processed.csv"),
    "split_indices": os.path.join(PROCESSED_DIR, "split_indices.npz"),
    "scaler": os.path.join(PROCESSED_DIR, "scaler.pkl"),
    "encoder": os.path.join(PROCESSED_DIR, "encoder.pkl"),

    "results_json": os.path.join(RESULTS_DIR, "main_results.json"),
    "baseline_results": os.path.join(RESULTS_DIR, "baseline_results.json"),
    "ablation_results": os.path.join(RESULTS_DIR, "ablation_results.json"),
    "sensitivity_results": os.path.join(RESULTS_DIR, "sensitivity_results.json"),
    "robustness_results": os.path.join(RESULTS_DIR, "robustness_results.json"),
    "uncertainty_analysis": os.path.join(RESULTS_DIR, "uncertainty_analysis.json"),

    "best_checkpoint": os.path.join(CHECKPOINT_DIR, "best_model.pth"),
    "final_checkpoint": os.path.join(CHECKPOINT_DIR, "final_model.pth"),

    "plot_architecture": os.path.join(PLOTS_DIR, "fig1_architecture.png"),
    "plot_comparison": os.path.join(PLOTS_DIR, "fig2_method_comparison.png"),
    "plot_ablation": os.path.join(PLOTS_DIR, "fig3_ablation.png"),
    "plot_sensitivity": os.path.join(PLOTS_DIR, "fig4_sensitivity.png"),
    "plot_uncertainty": os.path.join(PLOTS_DIR, "fig5_uncertainty_decomposition.png"),
    "plot_reliability": os.path.join(PLOTS_DIR, "fig6_reliability_diagram.png"),
    "plot_tsne": os.path.join(PLOTS_DIR, "fig7_feature_tsne.png"),
}

# ------------------------------------------------------------------------------
# Statistical Testing
# ------------------------------------------------------------------------------
STATS = {
    "n_bootstrap": 1000,
    "confidence_level": 0.95,
    "paired_test": "wilcoxon",      # "wilcoxon" or "t_test"
    "effect_size": "cohens_d",      # "cohens_d" or "cliffs_delta"
}

# ------------------------------------------------------------------------------
# Target Journal Metadata (for paper generation)
# ------------------------------------------------------------------------------
JOURNAL = {
    "name": "Environmental Modelling & Software",
    "publisher": "Elsevier",
    "scope": "Environmental modelling, software, and decision support systems",
    "open_access_option": True,
    "apc_estimate_usd": 0,          # hybrid journal; OA optional; check current rate
}


def get_config():
    """Return a deep copy of the full configuration dict."""
    import copy
    return copy.deepcopy({
        "device": str(DEVICE),
        "random_seeds": RANDOM_SEEDS,
        "preprocess": PREPROCESS,
        "model": MODEL,
        "loss": LOSS,
        "train": TRAIN,
        "baselines": BASELINES,
        "metrics": METRICS,
        "ablation": ABLATION,
        "sensitivity": SENSITIVITY,
        "robustness": ROBUSTNESS,
        "log": LOG,
        "output": OUTPUT,
        "stats": STATS,
        "journal": JOURNAL,
    })


if __name__ == "__main__":
    import json
    cfg = get_config()
    print(json.dumps(cfg, indent=2, default=str))
