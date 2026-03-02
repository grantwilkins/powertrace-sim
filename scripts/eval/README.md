# Evaluation Scripts

This directory contains scripts for evaluating power trace generation methods, comparing against baselines, processing Azure traces, and generating paper figures and tables.

## Script Categories

### 1. Baseline Evaluation

#### `run_baselines_node.py` - Node-Level Baseline Comparison

Compare GMM-BiGRU against baseline methods at the server/node level.

```bash
python -m scripts.eval.run_baselines_node \
    --config-id llama-3-8b_H100_tp1 \
    --num-seeds 5 \
    --out-dir results/eval_paper
```

**Baselines Compared:**
- **TDP**: Constant thermal design power
- **Mean**: Average power from training data
- **Splitwise LUT**: Phase-aware lookup table
- **Ours**: GMM-BiGRU (with AR(1) variants)

**Output:**
- `{out-dir}/{config_id}_metrics.csv` - Per-seed metrics
- `{out-dir}/{config_id}_summary.csv` - Aggregated statistics

#### `run_baselines_node_groundtruth.py` - Ground Truth Comparison

Evaluate against actual measured power traces.

```bash
python -m scripts.eval.run_baselines_node_groundtruth \
    --config-id llama-3-8b_H100_tp1 \
    --ground-truth-dir data/sharegpt-benchmark-llama-3-8b-h100 \
    --out-dir results/eval_paper/groundtruth
```

#### `run_baselines_facility.py` - Facility-Level Evaluation

Evaluate at datacenter/facility scale using Azure facility traces.

```bash
python -m scripts.eval.run_baselines_facility \
    --facility-data results/azure_facility \
    --out-dir results/eval_paper/facility
```

#### `baselines.py` - Baseline Method Implementations

Contains implementations of all baseline generation methods:

```python
from scripts.eval.baselines import (
    generate_tdp,           # Constant TDP
    generate_mean,          # Training mean
    generate_splitwise_lut, # Splitwise lookup table
    generate_ours,          # GMM-BiGRU pipeline
    build_splitwise_lut_params,  # LUT parameter estimation
)
```

### 2. Azure Trace Processing

#### `parse_azure_trace.py`

Parse raw Azure conversation traces into usable format.

```bash
python -m scripts.eval.parse_azure_trace \
    --input data/azure_trace/raw/conversations.csv \
    --output data/azure_trace/parsed
```

#### `split_azure_week_to_days.py`

Split week-long traces into daily segments.

```bash
python -m scripts.eval.split_azure_week_to_days \
    --input data/azure_trace/parsed/week.csv \
    --output data/azure_trace/days
```

#### `azure_to_node_streams.py`

Convert parsed traces to per-node request streams.

```bash
python -m scripts.eval.azure_to_node_streams \
    --input data/azure_trace/parsed \
    --output data/azure_trace/node_requests \
    --num-nodes 32
```

#### `azure_aggregate.py`

Aggregate node-level traces to rack/row/facility levels.

```bash
python -m scripts.eval.azure_aggregate \
    --input results/azure_facility/node_traces \
    --output results/azure_facility/aggregated
```

#### `azure_generate_traces.py`

Generate power traces from Azure request data.

```bash
python -m scripts.eval.azure_generate_traces \
    --requests data/azure_trace/node_requests \
    --checkpoint-dir results/continuous_v1_gmm_bigru/k10_f2/checkpoints \
    --out-dir results/azure_facility/node_traces
```

#### `azure_metrics.py`

Compute metrics on Azure facility traces.

```bash
python -m scripts.eval.azure_metrics \
    --generated results/azure_facility/aggregated \
    --ground-truth data/azure_facility \
    --out-csv results/azure_facility/metrics.csv
```

#### `azure_trace_utils.py`

Utility functions for Azure trace handling:

```python
from scripts.eval.azure_trace_utils import (
    load_azure_requests,
    compute_arrival_rate,
    bin_requests_to_intervals,
    compute_token_statistics,
)
```

### 3. Result Collection and Tables

#### `collect_results.py`

Aggregate evaluation results across configurations.

```bash
python -m scripts.eval.collect_results \
    --results-dir results/eval_paper \
    --out-csv results/eval_paper/all_metrics.csv
```

#### `generate_baselines_node_table.py`

Generate LaTeX table for node-level baseline comparison.

```bash
python -m scripts.eval.generate_baselines_node_table \
    --metrics-csv results/eval_paper/all_metrics.csv \
    --out-tex figures/tables/baselines_node.tex
```

#### `generate_trace_fidelity_table.py`

Generate trace fidelity metrics table.

```bash
python -m scripts.eval.generate_trace_fidelity_table \
    --metrics-dir results/eval_paper \
    --out-tex figures/tables/trace_fidelity.tex
```

#### `generate_azure_facility_sizing_table.py`

Generate facility sizing analysis table.

```bash
python -m scripts.eval.generate_azure_facility_sizing_table \
    --facility-metrics results/azure_facility/metrics.csv \
    --out-tex figures/tables/facility_sizing.tex
```

### 4. Figure Generation

#### `generate_power_cdf_comparison.py`

Generate power CDF comparison figures.

```bash
python -m scripts.eval.generate_power_cdf_comparison \
    --config-id llama-3-8b_H100_tp1 \
    --ground-truth data/sharegpt-benchmark-llama-3-8b-h100 \
    --generated results/eval_paper \
    --out-dir figures/trace_power_cdf_comparison
```

#### `azure_figures.py`

Generate Azure-specific evaluation figures.

```bash
python -m scripts.eval.azure_figures \
    --data-dir results/azure_facility \
    --out-dir figures/azure
```

#### `hierarchy_figure.py`

Generate datacenter hierarchy visualization.

```bash
python -m scripts.eval.hierarchy_figure \
    --data-dir results/azure_facility/aggregated \
    --out-file figures/hierarchy.pdf
```

#### `oversubscription_figure.py`

Generate power oversubscription analysis figure.

```bash
python -m scripts.eval.oversubscription_figure \
    --data-dir results/azure_facility \
    --out-file figures/oversubscription.pdf
```

#### `aggregation_variance.py`

Generate aggregation variance analysis figure.

```bash
python -m scripts.eval.aggregation_variance \
    --data-dir results/azure_facility \
    --out-file figures/aggregation_variance.pdf
```

#### `aggregation_resolution.py`

Generate time resolution analysis figure.

```bash
python -m scripts.eval.aggregation_resolution \
    --data-dir results/azure_facility \
    --out-file figures/aggregation_resolution.pdf
```

### 5. Utility Module

#### `pipeline_utils.py`

Re-exports commonly used functions for evaluation scripts:

```python
from scripts.eval.pipeline_utils import (
    # From model.classifiers.gmm_bigru
    build_rollout_features_from_requests,
    generate_gmm_bigru_trace,
    load_gmm_params_json_dict,

    # From model.scripts.eval_gmm_bigru
    predict_sorted_gmm_labels_from_params,
    estimate_ar1_params,
    generate_gmm_bigru_trace_ar1_thresholded,
)
```

## Typical Evaluation Workflow

```bash
# 1. Run node-level baselines
python -m scripts.eval.run_baselines_node \
    --num-seeds 10 \
    --out-dir results/eval_paper

# 2. Collect results
python -m scripts.eval.collect_results \
    --results-dir results/eval_paper \
    --out-csv results/eval_paper/all_metrics.csv

# 3. Generate tables
python -m scripts.eval.generate_baselines_node_table \
    --metrics-csv results/eval_paper/all_metrics.csv \
    --out-tex figures/tables/baselines_node.tex

# 4. Generate figures
python -m scripts.eval.generate_power_cdf_comparison \
    --out-dir figures/trace_power_cdf_comparison
```

## Metrics Computed

| Metric | Description | Better |
|--------|-------------|--------|
| `ks_stat` | Kolmogorov-Smirnov statistic | Lower |
| `acf_r2` | Autocorrelation R² fit | Higher |
| `nrmse` | Normalized RMSE | Lower |
| `p95_error_pct` | 95th percentile error (%) | Lower |
| `p99_error_pct` | 99th percentile error (%) | Lower |
| `delta_energy_pct` | Total energy error (%) | Lower |

## Output Directory Structure

```
results/eval_paper/
├── {config_id}_metrics.csv       # Per-config, per-seed metrics
├── {config_id}_summary.csv       # Aggregated statistics
├── all_metrics.csv               # Combined results
├── groundtruth/                  # Ground truth comparisons
└── facility/                     # Facility-level results
```
