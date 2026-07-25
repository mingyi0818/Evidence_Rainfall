# Experiment Reproduction Guide

## Environment Setup

```bash
pip install -r code/requirements.txt
```

## Data Preparation

This project uses evidence-based rainfall prediction datasets with meteorological observations and evidence features. Place the raw data in the `data/raw/` directory. The data loader handles feature engineering, normalization, and train/test splitting.

## Running Experiments

```bash
cd code
python train.py
```

For multi-seed experiments, modify the seed parameter in `config.py`.

## Expected Results

Results are saved in `results/tables/` (CSV/JSON format) and `results/plots/` (PNG format).

## Hardware Requirements

- GPU: NVIDIA RTX Pro 2000 (16GB VRAM)
- CPU: Intel Xeon W7-2595X (24 cores)
- RAM: 48GB DDR5
