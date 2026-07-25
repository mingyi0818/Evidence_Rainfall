"""
Focused runner: only sensitivity + robustness on temporal split.
Ablation already done in ablation_results_v2.csv.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import DEVICE, MODEL, TRAIN, RESULTS_DIR
from data_loader import preprocess_and_split
from ablation_sens_robust import run_sensitivity, run_robustness, SEED, EPOCHS, log, LOG_FILE

def main():
    log(f"Device: {DEVICE}")
    log(f"PyTorch: {torch.__version__}")

    # Load data with temporal split
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = \
        preprocess_and_split(seed=SEED, save=False, split_mode='temporal')
    input_dim = X_train.shape[1]
    log(f"Train={len(y_train)} Val={len(y_val)} Test={len(y_test)} dim={input_dim}")
    log(f"Train rain={y_train.mean():.4f} Test rain={y_test.mean():.4f}")

    # Run only sensitivity + robustness
    df_sens, df_sens_sum = run_sensitivity(X_train, y_train, X_val, y_val, X_test, y_test, input_dim)
    df_rob = run_robustness(X_train, y_train, X_val, y_val, X_test, y_test, input_dim)

    log("\n" + "="*60 + "\nSENSITIVITY + ROBUSTNESS COMPLETE\n" + "="*60)
    LOG_FILE.close()

    # Print summary
    print("\n=== SENSITIVITY SUMMARY ===")
    print(df_sens_sum.to_string(index=False))
    print("\n=== ROBUSTNESS SUMMARY (key columns) ===")
    print(df_rob[['perturbation', 'level', 'accuracy', 'f1_macro', 'ece',
                   'precision_mean', 'H_epi_mean', 'uncertainty_auroc']].to_string(index=False))


if __name__ == "__main__":
    main()
