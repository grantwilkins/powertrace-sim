# Scripts

This directory contains the main command-line scripts for training, evaluating, and running inference with the GMM-BiGRU pipeline.

Default data policy for evaluation/figure scripts is strict ShareGPT recorded timestamps:
- `--request-timestamp-policy recorded_only`
- `--allowed-json-prefix data/sharegpt-benchmark`
Use `--request-timestamp-policy allow_synthesized` only when you explicitly want synthetic timestamp fallback.

## Primary Scripts

### `train_gmm_bigru.py` - Model Training

Train GMM and BiGRU classifier for power state prediction.

```bash
python -m model.scripts.train_gmm_bigru \
    --manifest results/experimental_continuous_v1_gru_all/manifest.json \
    --out-root results/continuous_v1_gmm_bigru_sharegpt_all \
    --auto-k \
    --max-k 12 \
    --feature-set f2
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--manifest` | Path to experimental manifest JSON | `results/experimental_continuous_v1_gru_all/manifest.json` |
| `--out-root` | Output root for artifacts | `results/continuous_v1_gmm_bigru_sharegpt_all` |
| `--k` | Fixed number of GMM components (when `--auto-k` is off) | 10 |
| `--auto-k` | Enable BIC-based K selection | False |
| `--max-k` | Upper bound for auto-K search | 20 |
| `--feature-set` | Feature set: f2 or f3 | f2 |
| `--config-id` | Specific config IDs (repeat flag) | All written configs |
| `--hidden-dim` | GRU hidden dimension | 64 |
| `--num-layers` | Number of GRU layers | 1 |
| `--epochs` | Training epochs | 500 |
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
    --run-manifest results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2/run_manifest.json \
    --experimental-manifest results/experimental_continuous_v1_gru_all/manifest.json \
    --pair-manifest-csv results/stage0/pair_manifest.csv \
    --request-timestamp-policy recorded_only \
    --allowed-json-prefix data/sharegpt-benchmark \
    --out-dir results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2_ar1_thresh/eval_metrics \
    --num-seeds 5
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--run-manifest` | Trained artifact manifest | `results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2/run_manifest.json` |
| `--experimental-manifest` | Experimental dataset manifest | `results/experimental_continuous_v1_gru_all/manifest.json` |
| `--pair-manifest-csv` | Stage0 pair manifest | `results/stage0/pair_manifest.csv` |
| `--request-timestamp-policy` | `recorded_only` or `allow_synthesized` | `recorded_only` |
| `--allowed-json-prefix` | Allowed request JSON root | `data/sharegpt-benchmark` |
| `--out-dir` | Output directory for metrics/artifacts | `results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2_ar1_thresh/eval_metrics` |
| `--num-seeds` | Number of random seeds for stochastic methods | 5 |
| `--decode-mode` | `stochastic` or `argmax` | `stochastic` |
| `--no-plots` | Disable plot generation | False |

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
    --run-manifest results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2/run_manifest.json \
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

Generate Figure D1 as a held-out BiGRU sequence-classification comparison for `F2` vs `Full-6D` features.

```bash
python -m model.scripts.figure_d1_conditional_entropy \
    --experimental-manifest results/experimental_continuous_v1_gru_all/manifest.json \
    --run-manifest results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2/run_manifest.json \
    --pair-manifest-csv results/stage0/pair_manifest.csv \
    --throughput-db model/config/throughput_database.json \
    --request-timestamp-policy recorded_only \
    --allowed-json-prefix data/sharegpt-benchmark \
    --use-legacy-f6-init true
```

Model policy:
- Trains separate BiGRU classifiers on `F2` and `Full-6D`.
- Reports held-out test cross-entropy and test accuracy.
- Optionally initializes the `Full-6D` model from legacy 6D checkpoints in `model/new_weights`, `model/best_weights`, and `model/gru_classifier_weights`.

Notes:
- Uses the C1 BIC config set by default, with optional `--config-ids` override.
- `--require-recorded-timestamps=true` is the default strict policy; configs can be dropped if valid traces are filtered out.
- Pass criteria are reported as: `Accuracy(F2)/Accuracy(F6)` and CE-retention relative to `Full-6D`.
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
    --manifest results/experimental_continuous_v1_gru_all/manifest.json \
    --out-root results/continuous_v1_gmm_bigru_sharegpt_all \
    --auto-k \
    --max-k 12

# 3. Evaluate
python -m model.scripts.eval_gmm_bigru \
    --run-manifest results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2/run_manifest.json \
    --experimental-manifest results/experimental_continuous_v1_gru_all/manifest.json \
    --pair-manifest-csv results/stage0/pair_manifest.csv \
    --request-timestamp-policy recorded_only \
    --allowed-json-prefix data/sharegpt-benchmark \
    --out-dir results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2_ar1_thresh/eval_metrics

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
