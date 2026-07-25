# Diagnosing and Mitigating Epistemic Uncertainty Degeneracy in Binary Evidence Deep Learning

This repository contains the full source code, experimental data, and pre-trained checkpoints for the paper:

> **Diagnosing and Mitigating Epistemic Uncertainty Degeneracy in Binary Evidence Deep Learning: A Case Study on Rainfall Occurrence Prediction**
>
> Jingyuan Zeng, Ming Zeng, Jianghong Guo, Chuanxian Jiang, Yafen Feng

## Quick Start for Reviewers

If you are a reviewer wanting to verify the experimental results, please follow these steps:

### 1. Environment Setup

```bash
# Python 3.10+ required
pip install -r code/requirements.txt
```

Hardware requirements (reference): Windows 11, RTX Pro 2000 16GB, Xeon W7-2595X 24-core, 48GB RAM. The code also runs on CPU (slower).

### 2. Download the Dataset

The dataset is the publicly available "Rain in Australia" dataset from Kaggle:

```
https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package
```

Download `weatherAUS.csv` and place it in `D:\datasets\rain_in_australia\weatherAUS.csv` (or update `DATASET_PATH` in `code/config.py`).

### 3. Verify Pre-computed Results

All experimental results reported in the paper are saved in the `results/` directory as JSON/CSV files. To verify that the paper's numbers match the result files, run:

```bash
python code/verify_results.py
```

This script checks every number reported in the paper against the result files and reports any discrepancies.

### 4. Reproduce the Experiments

To reproduce all experiments from scratch (this takes approximately 12-24 hours on the reference hardware):

```bash
# Step 1: Train all models with 5 random seeds (main results, Table 1)
python code/run_temporal_experiments.py

# Step 2: Aggregate 5-seed results
python code/aggregate_results.py

# Step 3: Ablation, sensitivity, and robustness experiments (Tables 2-4)
python code/run_sens_robust_v2.py

# Step 4: Statistical analysis (paired Wilcoxon tests)
python code/statistical_analysis_v2.py

# Step 5: Meteorological skill scores, cost-loss, selective prediction (Tables 7-9)
python code/m4_m5_m6_analysis.py

# Step 6: OOD experiments (Tables 10-13)
python code/m7_ood_experiments.py

# Step 7: CAE-Net training and evaluation (Tables 14-15)
python code/train_cae_net.py
```

### 5. Re-evaluate Pre-trained Checkpoints

If you want to re-evaluate the pre-trained models without retraining, the checkpoints are in `checkpoints/`:

```bash
# Re-evaluate all models with consistent 123-dim features
python code/reeval_123dim.py
```

> **Note on Random Forest checkpoints**: The 5 Random Forest model files (`randomforest_seed{42,123,456,789,2024}.pkl`, ~180MB each) are excluded from this repository because they exceed GitHub's 100MB file size limit. To regenerate them, run the main experiment script (`python code/run_temporal_experiments.py`), which trains all baselines including Random Forest. The `results/` directory already contains all evaluation metrics from these models, so verification of paper-reported numbers does not require the RF checkpoint files.

## Repository Structure

```
17_Evidence_Rainfall/
├── code/                           # Source code
│   ├── config.py                   # Configuration (paths, hyperparameters)
│   ├── data_loader.py              # Data loading and temporal split
│   ├── models.py                   # Model implementations (EDL, LSTM, GRU, etc.)
│   ├── train.py                    # Training script for all baselines
│   ├── evaluate.py                 # Evaluation script
│   ├── train_cae_net.py            # CAE-Net training (C2/C3/C4)
│   ├── cae_net.py                  # CAE-Net model implementation
│   ├── ablation_sens_robust.py     # Ablation, sensitivity, robustness
│   ├── run_sens_robust_v2.py       # Updated ablation/sensitivity/robustness
│   ├── statistical_analysis_v2.py  # Paired Wilcoxon tests + Holm-Bonferroni
│   ├── m4_m5_m6_analysis.py        # Meteorological skill, cost-loss, selective prediction
│   ├── m7_ood_experiments.py       # OOD experiments (spatial/seasonal/extreme/temporal)
│   ├── run_temporal_experiments.py # Main experiments (5 seeds)
│   ├── aggregate_results.py        # 5-seed aggregation
│   ├── reeval_123dim.py            # Re-evaluation with 123-dim features
│   ├── visualize.py                # Figure generation
│   ├── verify_results.py           # Result verification script
│   └── requirements.txt            # Python dependencies
├── data/
│   └── processed/
│       └── neighborhood_labels.npz # Spatial neighborhood labels for C4 Mondrian
├── results/                        # All experimental results
│   ├── main_results_v3.json        # Table 1 (main results, 5-seed)
│   ├── ablation_results_v2.csv     # Table 2 (ablation)
│   ├── sensitivity_results.json    # Table 3 (sensitivity)
│   ├── robustness_results.json     # Table 4 (robustness)
│   ├── uncertainty_analysis_v2.json# Tables 5-6 (uncertainty decomposition)
│   ├── m4_skill_scores.json        # Table 7 (meteorological skill)
│   ├── m5_cost_loss.json           # Table 8 (cost-loss)
│   ├── m6_selective_prediction.json# Table 9 (selective prediction)
│   ├── m7_ood_experiments.json     # Tables 10-13 (OOD)
│   ├── cae_net_results.json        # Tables 14-15 (CAE-Net)
│   ├── statistical_tests_seed_level.json  # Statistical tests
│   ├── fixed_results_temporal_all.json    # Per-seed results
│   ├── plots/                      # Generated figures (PNG/SVG)
│   └── tables/
│       └── data_verification_report.json  # Data-Verifier report
├── checkpoints/                    # Pre-trained model weights
│   ├── edl_seed{42,123,456,789,2024}.pth
│   ├── lstm_seed{42,123,456,789,2024}.pth
│   ├── gru_seed{42,123,456,789,2024}.pth
│   ├── bnn_seed{42,123,456,789,2024}.pth
│   ├── mcdropout_seed{42,123,456,789,2024}.pth
│   ├── xgboost_seed{42,123,456,789,2024}.pkl
│   ├── logisticregression_seed{42,123,456,789,2024}.pkl
│   ├── cae_net_seed42.pth
│   └── (randomforest_seed*.pkl excluded — see note above, regenerate via run_temporal_experiments.py)
├── paper/
│   ├── paper_draft.md              # Manuscript (Markdown)
│   ├── cover_letter.md             # Cover letter
│   ├── highlights.md               # Highlights
│   └── figures/                    # Paper figures
├── reproduce.md                    # Detailed reproduction guide
└── README.md                       # This file
```

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Hidden layers | [128, 64, 32] | MLP architecture |
| Dropout rate | 0.3 | Dropout after each hidden layer |
| Learning rate | 0.01 | Adam optimizer |
| Batch size | 256 | Mini-batch size |
| Epochs | 80 | Training epochs |
| λ_reg | 0.001 | KL regularization weight |
| Annealing epochs | 50 | KL annealing period |
| Prior n_0 | 10 | Climatology prior strength |
| S_max | 100 | Evidence budget (C3) |
| β_budget | 0.01 | Evidence budget weight |
| ε (conformal) | 0.05 | Miscoverage rate |
| Random seeds | {42, 123, 456, 789, 2024} | 5-seed experiments |

## Data Split

- **Training**: 2007-2014 (113,642 samples)
- **Validation**: 2015 (14,851 samples)
- **Test**: 2016-2017 (25,974 samples)

The split is strictly temporal (no leakage). See `code/data_loader.py` for implementation details.

## Result Verification

Every number reported in the paper is traceable to a result file in `results/`. The verification report is in `results/tables/data_verification_report.json`. To re-verify:

```bash
python code/verify_results.py
```

Expected output: `Data Authenticity Score: 100/100` (all 350+ numbers traceable).

## Citation

If you use this code, please cite:

```bibtex
@article{zeng2026diagnosing,
  title={Diagnosing and Mitigating Epistemic Uncertainty Degeneracy in Binary Evidence Deep Learning: A Case Study on Rainfall Occurrence Prediction},
  author={Zeng, Jingyuan and Zeng, Ming and Guo, Jianghong and Jiang, Chuanxian and Feng, Yafen},
  journal={Applied Intelligence},
  year={2026},
  note={Under review}
}
```

## Contact

- **Corresponding author**: Yafen Feng (fyf81@163.com)
- **First author**: Jingyuan Zeng (zjy@jyu.edu.cn)

## License

This project is licensed under the MIT License - see the code files for details. The dataset is subject to Kaggle's terms of use.
