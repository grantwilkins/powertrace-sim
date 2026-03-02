# Results Directory

This directory contains training outputs, model checkpoints, evaluation metrics, and experiment artifacts for the PowerTrace-Sim project.

## Directory Structure

```
results/
├── stage0/                       # Preprocessed training data
├── continuous_v1_gmm_bigru/      # Primary GMM-BiGRU experiments
│   ├── k10_f2/                   # K=10 components, f2 features (main)
│   ├── k10_f2_ar1/               # With AR(1) smoothing
│   ├── k10_f2_ar1_thresh/        # With AR(1) + idle thresholding
│   ├── k10_f3/                   # K=10, f3 features
│   ├── k8_f2/                    # K=8 ablation
│   └── k12_f2/                   # K=12 ablation
├── continuous_v1/                # Legacy GRU-only experiments
├── continuous_v1_stateless/      # Stateless baseline experiments
├── continuous_v1_lambda_mu01/    # Lambda regularization experiments
├── experimental_continuous_v1/   # Experimental preprocessing
├── azure_facility/               # Azure facility-level results
│   ├── node_traces/              # Per-node generated traces
│   └── aggregated/               # Aggregated power traces
└── eval_paper/                   # Paper evaluation outputs
```

## Primary Results: GMM-BiGRU

### `continuous_v1_gmm_bigru/k10_f2/` (Recommended)

Main experiment with K=10 GMM components and f2 feature set.

```
k10_f2/
├── checkpoints/                  # Model weights
│   ├── llama-3-8b_H100_tp1.pt
│   ├── llama-3-70b_H100_tp4.pt
│   └── ...
├── gmms/                         # GMM parameters
│   ├── llama-3-8b_H100_tp1.json
│   └── ...
├── norm_params/                  # Normalization parameters
│   ├── llama-3-8b_H100_tp1.json
│   └── ...
├── eval_metrics/                 # Evaluation results
│   ├── eval_metrics.csv
│   ├── eval_metrics_summary.csv
│   └── plots/
└── training_curves/              # Training loss plots
```

### Checkpoint Format (.pt)

```python
{
    "model_state_dict": {...},    # GRU classifier weights
    "optimizer_state_dict": {...}, # Optimizer state
    "epoch": int,                  # Training epoch
    "loss": float,                 # Final training loss
    "config": {                    # Configuration metadata
        "input_size": 2,
        "hidden_size": 64,
        "num_layers": 2,
        "num_classes": 10,
        "bidirectional": True,
    }
}
```

### GMM Parameters Format (.json)

```json
{
    "k": 10,
    "means": [150.2, 245.7, 312.4, ...],
    "stds": [12.3, 18.5, 22.1, ...],
    "weights": [0.05, 0.12, 0.18, ...],
    "label_map": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "order": [0, 3, 1, 5, 2, 7, 4, 8, 6, 9]
}
```

### Normalization Parameters Format (.json)

```json
{
    "active_mean": 1.52,
    "active_std": 0.87,
    "delta_mean": 0.003,
    "delta_std": 0.245,
    "power_mean": 425.3,
    "power_std": 85.2
}
```

## Stage 0 Data

### `stage0/`

Preprocessed training data indexed by manifest.

```
stage0/
├── manifest.json                 # Data index
├── throughput.json               # Throughput statistics
└── {config_id}/
    ├── run_0.json                # Request data
    ├── power_0.csv               # Power trace
    ├── run_1.json
    └── power_1.csv
```

## Evaluation Results

### `eval_paper/`

Paper-ready evaluation outputs.

```
eval_paper/
├── {config_id}_metrics.csv       # Per-config metrics
├── {config_id}_summary.csv       # Summary statistics
├── all_metrics.csv               # Aggregated results
└── figures/                      # Generated figures
```

### Metrics CSV Format

```csv
config_id,method,seed,ks_stat,acf_r2,nrmse,p95_error_pct,p99_error_pct,delta_energy_pct
llama-3-8b_H100_tp1,ours,0,0.023,0.987,0.045,2.3,4.1,1.2
llama-3-8b_H100_tp1,ours,1,0.025,0.985,0.048,2.5,4.3,1.4
llama-3-8b_H100_tp1,tdp,0,0.312,0.000,0.234,15.2,18.7,12.3
```

## Azure Facility Results

### `azure_facility/`

Results from Azure facility-level evaluation.

```
azure_facility/
├── node_traces/                  # Per-node power traces
│   ├── node_0_power.csv
│   └── ...
├── aggregated/                   # Aggregated traces
│   ├── rack_0_power.csv
│   ├── row_0_power.csv
│   └── facility_power.csv
└── metrics.csv                   # Facility-level metrics
```

## Using Results

### Loading a Trained Model

```python
import torch
from model.classifiers.gru import GRUClassifier
from model.classifiers.gmm_bigru import load_gmm_params_json_dict
import json

# Load checkpoint
ckpt = torch.load("results/continuous_v1_gmm_bigru/k10_f2/checkpoints/llama-3-8b_H100_tp1.pt")
model = GRUClassifier(**ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])

# Load GMM parameters
gmm_params = load_gmm_params_json_dict(
    "results/continuous_v1_gmm_bigru/k10_f2/gmms/llama-3-8b_H100_tp1.json"
)

# Load normalization parameters
with open("results/continuous_v1_gmm_bigru/k10_f2/norm_params/llama-3-8b_H100_tp1.json") as f:
    norm_params = json.load(f)
```

### Loading Evaluation Results

```python
import pandas as pd

# Load metrics
df = pd.read_csv("results/eval_paper/all_metrics.csv")

# Filter by method
ours = df[df["method"] == "ours"]
print(f"Mean KS stat: {ours['ks_stat'].mean():.4f}")
print(f"Mean ACF R²: {ours['acf_r2'].mean():.4f}")
```

## Experiment Naming Convention

```
{experiment_type}_{variant}/
```

- `continuous_v1` - Original continuous power modeling
- `continuous_v1_gmm_bigru` - GMM + BiGRU pipeline
- `continuous_v1_stateless` - Stateless baseline
- `experimental_*` - Experimental/in-progress work

### Variant Suffixes

- `k{N}` - Number of GMM components
- `f{N}` - Feature set (f2, f3)
- `ar1` - AR(1) smoothing enabled
- `thresh` - Idle thresholding enabled
- `lambda_mu{X}` - Regularization experiments
