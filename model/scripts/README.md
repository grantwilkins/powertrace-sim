# Scripts

This directory contains the main command-line scripts for training, evaluating, and running inference with the GMM-BiGRU pipeline.

## Primary Scripts

### `train_gmm_bigru.py` - Model Training

Train GMM and BiGRU classifier for power state prediction.

```bash
python -m model.scripts.train_gmm_bigru \
    --stage0-manifest results/stage0/manifest.json \
    --out-dir results/continuous_v1_gmm_bigru/k10_f2 \
    --num-components 10 \
    --feature-set f2 \
    --config-ids llama-3-8b_H100_tp1 llama-3-70b_H100_tp4
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--stage0-manifest` | Path to data manifest JSON | Required |
| `--out-dir` | Output directory for artifacts | Required |
| `--num-components` | Number of GMM components (K) | 10 |
| `--feature-set` | Feature set: f2 or f3 | f2 |
| `--config-ids` | Specific configs to train (space-separated) | All |
| `--hidden-size` | GRU hidden dimension | 64 |
| `--num-layers` | Number of GRU layers | 2 |
| `--epochs` | Training epochs | 100 |
| `--lr` | Learning rate | 1e-3 |
| `--device` | Device: cuda, mps, or cpu | Auto-detect |

**Output Structure:**
```
out-dir/
├── checkpoints/          # Model weights (.pt files)
│   └── {config_id}.pt
├── gmms/                 # GMM parameters (.json files)
│   └── {config_id}.json
├── norm_params/          # Normalization parameters
│   └── {config_id}.json
└── training_curves/      # Loss plots (optional)
```

### `eval_gmm_bigru.py` - Model Evaluation

Evaluate trained models on held-out test data.

```bash
python -m model.scripts.eval_gmm_bigru \
    --stage0-manifest results/stage0/manifest.json \
    --checkpoint-dir results/continuous_v1_gmm_bigru/k10_f2/checkpoints \
    --norm-dir results/continuous_v1_gmm_bigru/k10_f2/norm_params \
    --gmm-dir results/continuous_v1_gmm_bigru/k10_f2/gmms \
    --out-dir results/continuous_v1_gmm_bigru/k10_f2/eval_metrics \
    --num-seeds 5
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--stage0-manifest` | Path to data manifest JSON | Required |
| `--checkpoint-dir` | Directory with model checkpoints | Required |
| `--norm-dir` | Directory with normalization params | Required |
| `--gmm-dir` | Directory with GMM parameters | Required |
| `--out-dir` | Output directory for metrics | Required |
| `--num-seeds` | Number of random seeds for stochastic methods | 5 |
| `--use-ar1` | Enable AR(1) smoothing | False |
| `--use-ar1-thresh` | Enable AR(1) with idle thresholding | False |

**Output:**
- `eval_metrics.csv` - Per-configuration metrics
- `eval_metrics_summary.csv` - Aggregated statistics
- `plots/` - Visualization of generated vs ground truth traces

### `infer_gmm_bigru.py` - Power Trace Generation

Generate power traces from request timelines using trained models.

```bash
python -m model.scripts.infer_gmm_bigru \
    --requests input_requests.csv \
    --checkpoint results/.../checkpoints/llama-3-8b_H100_tp1.pt \
    --norm results/.../norm_params/llama-3-8b_H100_tp1.json \
    --gmm results/.../gmms/llama-3-8b_H100_tp1.json \
    --out-csv generated_power.csv \
    --run-manifest results/stage0/manifest.json \
    --throughput-db data/perf_model.csv
```

**Input CSV Format:**
```csv
request_id,arrival_time,input_tokens,output_tokens
0,0.0,512,128
1,0.5,256,64
...
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--requests` | Input CSV with request timeline |
| `--checkpoint` | Model checkpoint path |
| `--norm` | Normalization parameters JSON |
| `--gmm` | GMM parameters JSON |
| `--out-csv` | Output power trace CSV |
| `--run-manifest` | Data manifest for config resolution |
| `--throughput-db` | Performance database for TTFT/TPOT |
| `--config-id` | Override config ID |
| `--dt` | Output time resolution (seconds) |
| `--seed` | Random seed for generation |

**Output CSV Format:**
```csv
timestamp_s,watts
0.0,450.2
0.1,487.6
...
```

### `compare_gmm_bigru.py` - Results Comparison

Compare evaluation results across experiments or configurations.

```bash
python -m model.scripts.compare_gmm_bigru \
    --results-dirs results/exp1 results/exp2 \
    --labels "Baseline" "New Method" \
    --out-dir results/comparison
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--results-dirs` | Directories with eval_metrics.csv |
| `--labels` | Labels for each experiment |
| `--out-dir` | Output directory for comparison |
| `--metrics` | Metrics to compare (default: all) |

## Supporting Scripts

### `generate_methods_figures.py`

Generate paper-quality figures for methods section.

```bash
python -m model.scripts.generate_methods_figures \
    --out-dir figures/methods
```

### `figure_d1_conditional_entropy.py`

Generate Figure D1 (single-panel grouped bars) for conditional entropy / NMI feature sufficiency versus regime labels.

```bash
python -m model.scripts.figure_d1_conditional_entropy \
    --experimental-manifest results/experimental_continuous_v1/manifest.json \
    --run-manifest results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json \
    --pair-manifest-csv results/stage0/pair_manifest.csv \
    --throughput-db model/config/throughput_database.json
```

Estimator policy:
- `A_t`, `ΔA_t`, `F2`: equal-frequency binned plugin MI.
- `Full-6D`: leave-one-trace-out kNN posterior MI (`I(X;z)=H(z)-CE_kNN(z|X)`).

Notes:
- Uses the C1 BIC config set by default, with optional `--config-ids` override.
- `--require-recorded-timestamps=true` is the default strict policy; configs can be dropped if valid traces are filtered out.
- Writes a per-config CSV and manifest JSON with selected/skipped config reasons for transparency.

### `simulate_server_power.py`

Standalone server power simulation script.

```bash
python -m model.scripts.simulate_server_power \
    --model llama-3-8b \
    --hardware H100 \
    --tp 1 \
    --duration 600 \
    --rate 1.0 \
    --out-csv server_power.csv
```

### `power_regression_analysis.py`

Analyze power consumption vs. workload characteristics.

```bash
python -m model.scripts.power_regression_analysis \
    --data-dir data/sharegpt-benchmark-llama-3-8b-h100 \
    --out-dir results/regression
```

## Typical Workflow

```bash
# 1. Prepare training data
python -m model.training_data.utils.prepare_training_data \
    --input-dir data/sharegpt-benchmark-* \
    --output results/stage0

# 2. Train models
python -m model.scripts.train_gmm_bigru \
    --stage0-manifest results/stage0/manifest.json \
    --out-dir results/my_experiment

# 3. Evaluate
python -m model.scripts.eval_gmm_bigru \
    --stage0-manifest results/stage0/manifest.json \
    --checkpoint-dir results/my_experiment/checkpoints \
    --norm-dir results/my_experiment/norm_params \
    --gmm-dir results/my_experiment/gmms \
    --out-dir results/my_experiment/eval_metrics

# 4. Generate traces for new workloads
python -m model.scripts.infer_gmm_bigru \
    --requests my_workload.csv \
    --checkpoint results/my_experiment/checkpoints/llama-3-8b_H100_tp1.pt \
    --norm results/my_experiment/norm_params/llama-3-8b_H100_tp1.json \
    --gmm results/my_experiment/gmms/llama-3-8b_H100_tp1.json \
    --out-csv generated_power.csv
```

## Configuration ID Format

Configuration IDs follow the pattern: `{model}_{hardware}_{tp}`

**Examples:**
- `llama-3-8b_H100_tp1`
- `llama-3-70b_A100_tp4`
- `deepseek-r1-distill-70b_H100_tp8`
