# EPMO — Empirical Performance Modeling and Optimization

> Hardware-aware ML system that predicts the optimal PyTorch DataLoader configuration (`num_workers`, `batch_size`) for any hardware environment — eliminating manual tuning and reducing data loading bottlenecks.

## Overview

PyTorch DataLoader performance varies significantly across hardware. EPMO benchmarks real environments, trains two domain-specific ML models (CPU/GPU), and recommends the best config in milliseconds — replacing guesswork with data.

**Two-model architecture:**
- **Model A** — CPU-only environments (`gpu_available=0`) → Ridge Regression · 75% Config Accuracy
- **Model B** — GPU-present environments (`gpu_available=1`) → Random Forest · 88% Config Accuracy · 100% Top-3 Accuracy · 5.70% mean Regret

## How It Works

```
Hardware Benchmarking (20 real environments — 12 CPU + 8 GPU)
        ↓
Feature Engineering (11 features: 6 base + 5 interaction terms)
        ↓
5 Algorithms: Random Forest, Gradient Boosting, XGBoost, Decision Tree, Ridge Regression
        ↓
Leave-One-Environment-Out Cross Validation (LOOCV)
        ↓
Best model selected by ConfigAcc (not MAE)
        ↓
predict.py → Auto-detects hardware → Returns optimal config instantly
```

## Key Results

| Domain | Model | Config Acc | Top-3 Acc | Avg Saving% | Mean Regret% |
|---|---|---|---|---|---|
| CPU (Model A) | Ridge Regression | 75% | 83% | +25.7% | 14.72% |
| GPU (Model B) | Random Forest | 88% | 100% | +41.4% | 5.70% |

**Case study highlights (held-out unseen machines):**

| Environment | Workload | Default | EPMO | Saving% |
|---|---|---|---|---|
| CPU workstation | CNN | 0.2019s | 0.1161s | +42.5% |
| CPU workstation | Transformer | 0.1592s | 0.0748s | +53.0% |
| GPU system | MatMul | 0.2631s | 0.0958s | +63.6% |
| GPU system | CNN | 0.3391s | 0.2236s | +34.0% |

## Evaluation Metrics

| Metric | Description |
|---|---|
| Config Accuracy | % of folds where predicted config == oracle config |
| Regret% | `(epmo_time - oracle_time) / oracle_time × 100` — lower is better |
| Saving% | Time saved vs static default (`workers=0, batch=32`) |
| PctOracle | % of oracle savings that EPMO achieves |

**Regret thresholds:** <2% = Excellent · <5% = Success · <15% = Acceptable

## Interaction Features

| Feature | Physical Meaning |
|---|---|
| `workers_x_cpu` | Parallel pressure: workers × CPU cores |
| `batch_x_ram` | Memory intensity: batch / RAM (dominant predictor — 0.294 CPU, 0.569 GPU) |
| `cpu_per_worker` | Core-to-worker ratio: cores / (workers + 1) |
| `workers_x_gpu` | GPU gate: workers × gpu_available |
| `batch_x_gpu` | GPU batch gate: batch × gpu_available |

## Repository Structure

```
EPMO/
├── Model_training.py          # training, LOOCV, figures, paper numbers
├── DATA/                      # epmo_data_cpu.csv, epmo_data_gpu.csv (20 environments)
├── TESTING/                   # held-out test evaluation scripts
├── EPMO_Benchmark_files/      # benchmark scripts (epmo_cpu_benchmark.py, epmo_gpu_benchmark.py)
├── Model_Training_output/     # generated figures, tables, pkl models, predict.py
└── README.md
```

## How to Run

```bash
git clone https://github.com/princehooda9/EPMO
cd EPMO
pip install numpy pandas scikit-learn matplotlib xgboost psutil

# Train models + generate all figures and paper numbers
python Model_training.py

# Run inference on your machine (auto-detects hardware)
python Model_Training_output/predict.py
```

## Inference Example

```
Hardware: {'cpu_logical': 8, 'ram_gb': 16.0, 'gpu_available': 1, 'gpu_vram_gb': 8.0}

EPMO Recommended Config:
  num_workers     = 4
  batch_size      = 128
  predicted_time  = 0.0312s
  model_used      = Model_B (GPU)
```

## Outputs

After training, `Model_Training_output/` contains:

| File | Description |
|---|---|
| `model_cpu.pkl`, `model_gpu.pkl` | Deployable trained models |
| `fig_1_heatmaps_*.png` | Per-environment config heatmaps |
| `fig_2_model_comparison_*.png` | MAE + Config Accuracy across algorithms |
| `fig_3_speedup_*.png` | EPMO vs Oracle vs Static Default |
| `fig_4_amdahl.png` | Empirical vs Amdahl's Law speedup curves |
| `fig_5_feature_importance_*.png` | Feature importance |
| `fig_6_decision_tree_cpu.png` | Interpretable decision rules |
| `fig_7_pred_vs_actual_*.png` | Predicted vs actual load times |
| `paper_numbers.txt` | All key metrics for reporting |
| `predict.py` | Standalone inference script |

## Tech Stack

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

`scikit-learn` · `XGBoost` · `pandas` · `NumPy` · `matplotlib` · `psutil` · `PyTorch`
