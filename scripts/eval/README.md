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
- **Splitwise Strict Emulation**: Request-centric LUT scheduler surrogate
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

#### `appendix_surrogate_validity.py` - Appendix A1 Surrogate Validity

Generate Appendix A1 sanity-check evidence for measured vs surrogate `A_t`
without queue reconstruction. The script writes per-config `A_t` plots
(time-series + histogram) and `\lambda` vs mean `A_t` saturation diagnostics.

```bash
python -m scripts.eval.appendix_surrogate_validity \
    --config-pool all_trained \
    --num-representative-configs 3 \
    --min-eligible-traces 2 \
    --stable-corr-threshold 0.80
```

**Output:**
- `results/eval_paper/appendix_a1_trace_metrics.csv` - Per-trace metrics and skip reasons
- `results/eval_paper/appendix_a1_config_summary.csv` - Per-config medians and saturation slopes
- `figures/appendix_a1_at_{config_slug}_trace{idx}_at_timeseries.pdf` - Per-config `A_t` time-series overlay
- `figures/appendix_a1_at_{config_slug}_trace{idx}_at_histogram.pdf` - Per-config `A_t` histogram overlay
- `figures/appendix_a1_lambda_vs_mean_at.pdf` - `\lambda` vs mean `A_t` scatter panels
- `results/eval_paper/appendix_a1_manifest.json` - Reproducibility manifest and selection notes
- Manifest `surrogate_quality` section and stdout summary - Aggregate mean/median correlation, MAE, RMSE, and mean-`A_t` error

#### `baselines.py` - Baseline Method Implementations

Contains implementations of all baseline generation methods:

```python
from scripts.eval.baselines import (
    generate_tdp,           # Constant TDP
    generate_mean,          # Training mean
    generate_splitwise_strict_emulation, # Splitwise strict emulation
    generate_ours,          # GMM-BiGRU pipeline
    build_splitwise_lut_params,  # strict emulation LUT parameter estimation
)
```

### 2. Azure Facility Pipeline

The Azure facility pipeline is now baseline-inclusive by default at the top level. The maintained workflow is:
- `ours`
- `splitwise_strict`
- plus facility baselines in downstream artifacts: `tdp_baseline`, `mean_baseline`

Default Azure configuration:
- `config_id`: `llama-3-70b_A100_tp4`
- Splitwise source curves: `llama-3-70b` on `a100-80gb` with `tp=4`

Default output roots:
- `results/azure_facility/node_traces/{method}`
- `results/azure_facility/aggregated/{method}`
- `results/eval_paper/azure_facility_metrics.csv`
- `results/eval_paper/azure_facility_ldc_15min.csv`
- `results/eval_paper/azure_facility_site_traces_15min.csv`
- `results/eval_paper/azure_oversubscription_capacity.{csv,json}`
- `results/eval_paper/azure_facility_sizing_table.{csv,json,tex}`
- `figures/azure_figure_*.pdf`

#### `run_azure_pipeline.py`

Run the full Azure facility workflow end to end.

```bash
python -m scripts.eval.run_azure_pipeline
```

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
    --output data/azure_facility/node_streams \
    --num-nodes 240
```

#### `azure_generate_traces.py`

Generate per-node Azure power traces for `ours` and `splitwise_strict`.

```bash
python -m scripts.eval.azure_generate_traces \
    --node-stream-dir data/azure_facility/node_streams \
    --output-root results/azure_facility/node_traces
```

#### `azure_aggregate.py`

Aggregate node traces into rack, row, and site traces under `aggregated/{method}`.

```bash
python -m scripts.eval.azure_aggregate \
    --node-traces-root results/azure_facility/node_traces \
    --output-root results/azure_facility/aggregated
```

#### `azure_metrics.py`

Write normalized multi-method facility metrics, 15-minute site traces, and load-duration rows.

```bash
python -m scripts.eval.azure_metrics \
    --aggregated-root results/azure_facility/aggregated \
    --node-traces-root results/azure_facility/node_traces \
    --metrics-csv results/eval_paper/azure_facility_metrics.csv \
    --ldc-csv results/eval_paper/azure_facility_ldc_15min.csv \
    --site-traces-15min-csv results/eval_paper/azure_facility_site_traces_15min.csv
```

#### `oversubscription_figure.py`

Compute the max feasible rack count per method under the configured row-power risk rule.

```bash
python -m scripts.eval.oversubscription_figure \
    --aggregated-root results/azure_facility/aggregated \
    --metrics-csv results/eval_paper/azure_facility_metrics.csv
```

#### `azure_figures.py`

Generate Azure facility figures. Figure 1 is `ours` only; comparison figures include TDP, Mean, Splitwise, and Ours.

```bash
python -m scripts.eval.azure_figures \
    --aggregated-root results/azure_facility/aggregated \
    --metrics-csv results/eval_paper/azure_facility_metrics.csv \
    --ldc-csv results/eval_paper/azure_facility_ldc_15min.csv \
    --site-traces-15min-csv results/eval_paper/azure_facility_site_traces_15min.csv \
    --out-dir figures
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

Generate the facility sizing comparison table for TDP, Mean, Splitwise, and Ours.

```bash
python -m scripts.eval.generate_azure_facility_sizing_table \
    --metrics-csv results/eval_paper/azure_facility_metrics.csv \
    --out-tex results/eval_paper/azure_facility_sizing_table.tex
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
    --aggregated-root results/azure_facility/aggregated \
    --metrics-csv results/eval_paper/azure_facility_metrics.csv \
    --ldc-csv results/eval_paper/azure_facility_ldc_15min.csv \
    --site-traces-15min-csv results/eval_paper/azure_facility_site_traces_15min.csv \
    --out-dir figures
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
    --aggregated-root results/azure_facility/aggregated \
    --metrics-csv results/eval_paper/azure_facility_metrics.csv
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

#### `feature_sufficiency_figure.py`

Generate the A2 feature sufficiency curve using held-out predictive
information retention for `A`, `ΔA`, `F2`, `F3`, and `F6` versus regime
label `z_t`.

Metric definitions:
- `IR_abs = 1 - CE_subset / CE_null`
- `IR_vs_F6 = (CE_null - CE_subset) / (CE_null - CE_F6)` (primary line)

```bash
python -m scripts.eval.feature_sufficiency_figure \
    --run-manifest results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json \
    --training-data-dir model/training_data \
    --gmm-dir results/continuous_v1_gmm_bigru/k10_f2/gmms \
    --epochs 8 \
    --hidden-dim 32 \
    --lr 1e-3 \
    --out-figure figures/feature_sufficiency_curve.pdf \
    --out-per-config-csv results/eval_paper/feature_sufficiency_per_config.csv \
    --out-summary-csv results/eval_paper/feature_sufficiency_summary.csv \
    --out-json results/eval_paper/feature_sufficiency_manifest.json
```

Outputs:
- Per-config scores: `config_id, subset, ce_subset, ce_null, ce_f6, ir_abs, ir_vs_f6, info_retained_abs_pct, info_retained_vs_f6_pct, n_test_samples, n_classes, input_dim, n_train_traces, n_test_traces`
- Summary rows: `subset, median_ir_vs_f6, ci95_low_ir_vs_f6, ci95_high_ir_vs_f6, median_ir_abs, ci95_low_ir_abs, ci95_high_ir_abs, median_info_retained_vs_f6_pct, median_info_retained_abs_pct, n_configs`
- Figure PDF and manifest JSON with selected/skipped configs, CE retention formulas, and sequence-model settings

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
| `delta_energy_pct` | Absolute total-trace energy error from `sum(power) * dt` (%) | Lower |

## Output Directory Structure

```
results/eval_paper/
├── {config_id}_metrics.csv       # Per-config, per-seed metrics
├── {config_id}_summary.csv       # Aggregated statistics
├── all_metrics.csv               # Combined results
├── groundtruth/                  # Ground truth comparisons
└── facility/                     # Facility-level results
```
