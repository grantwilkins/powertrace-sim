# Evaluation Scripts

This directory contains the maintained evaluation CLI entrypoints for node-level
baselines, Azure facility processing, paper tables, and figures. The old
`aggregation_variance.py` and `aggregation_resolution.py` references are no
longer relevant because those scripts are not present.

## Baseline Evaluation

### `run_baselines_node.py`

Node-level baseline comparison for TDP, Mean, Splitwise strict emulation, and Ours.

```bash
uv run -m scripts.eval.run_baselines_node \
  --config-ids llama-3-8b_H100_tp1 llama-3-8b_H100_tp2 \
  --num-seeds 5 \
  --decode-mode stochastic \
  --out-csv results/eval_paper/baselines_node_level.csv
```

Key flags: `--run-manifest`, `--experimental-manifest`, `--throughput-db`,
`--pair-manifest-csv`, `--ar1-params-dir`, `--out-csv`, `--config-ids`,
`--num-seeds`, `--base-seed`, `--device`, `--acf-max-lag`, `--decode-mode`,
`--median-filter-window`, `--ours-std-scale`, `--ours-logit-temperature`,
`--splitwise-perf-model-csv`, `--splitwise-source-model`,
`--splitwise-source-hardware`, `--splitwise-source-tp`.

### `run_baselines_node_groundtruth.py`

Held-out replay against measured GPU power traces.

```bash
uv run -m scripts.eval.run_baselines_node_groundtruth \
  --config-id deepseek-r1-distill-70b_H100_tp4 \
  --target-rate 0.25 \
  --out-csv results/eval_paper/baselines_node_groundtruth_metrics.csv \
  --out-plot-pdf figures/baselines_node_groundtruth_trace.pdf
```

Key flags: `--config-id`, `--target-rate`, `--test-trace-index`, `--tp-gpus`,
`--n-gpus-for-gpu-power`, `--gpu-tdp-w`, `--non-gpu-overhead-w`, `--num-seeds`,
`--base-seed`, `--device`, `--decode-mode`, `--median-filter-window`,
`--ours-std-scale`, `--ours-logit-temperature`, `--splitwise-perf-model-csv`,
`--splitwise-source-model`, `--splitwise-source-hardware`,
`--splitwise-source-tp`, `--splitwise-style-lut-mode`, `--splitwise-mode`.

### `run_baselines_facility.py`

Synthetic facility-level baseline comparison.

```bash
uv run -m scripts.eval.run_baselines_facility \
  --config-id deepseek-r1-distill-70b_H100_tp4 \
  --n-nodes 60 \
  --duration-s 3600 \
  --out-csv results/eval_paper/baselines_facility_metrics.csv \
  --traces-pdf figures/baselines_facility_traces.pdf \
  --ldc-pdf figures/baselines_load_duration.pdf
```

Key flags: `--config-id`, `--n-nodes`, `--duration-s`, `--dt`,
`--lambda-req-per-s-per-node`, `--tp-gpus`, `--n-gpus-for-gpu-power`,
`--gpu-tdp-w`, `--pue`, `--non-gpu-overhead-w`, `--facility-power-mode`,
`--traffic-model`, `--burst-rate-per-min`, `--burst-mean-duration-s`,
`--burst-peak-scale`, `--burst-background-sigma`, `--burst-node-scale-sigma`,
`--out-csv`, `--traces-pdf`, `--ldc-pdf`.

### `appendix_surrogate_validity.py`

Appendix A1 sanity checks for measured vs surrogate `A_t`.

```bash
uv run -m scripts.eval.appendix_surrogate_validity \
  --config-pool all_trained \
  --num-representative-configs 3 \
  --min-eligible-traces 2 \
  --stable-corr-threshold 0.80
```

Key flags: `--run-manifest`, `--experimental-manifest`, `--pair-manifest-csv`,
`--throughput-db`, `--config-pool`, `--num-representative-configs`,
`--min-eligible-traces`, `--stable-corr-threshold`, `--time-window-s`,
`--out-csv`, `--out-summary-csv`, `--out-figure-overlays`,
`--out-figure-scatter`, `--out-manifest-json`, `--dry-run`.

## Azure Facility Pipeline

### `run_azure_pipeline.py`

End-to-end Azure workflow: node streams, node traces, aggregation, metrics,
oversubscription, figures, and sizing table.

```bash
uv run -m scripts.eval.run_azure_pipeline
```

### `split_azure_week_to_days.py`

Split the week-long Azure trace into per-day CSV files.

```bash
uv run -m scripts.eval.split_azure_week_to_days \
  --input-csv data/azure_trace/raw/AzureLLMInferenceTrace_code_1week.csv \
  --output-dir data/azure_trace/days
```

### `parse_azure_trace.py`

Normalize a day CSV into request tuples plus metadata.

```bash
uv run -m scripts.eval.parse_azure_trace \
  --input-csv data/azure_trace/days/2024-05-16.csv \
  --output-dir data/azure_trace/parsed
```

### `azure_to_node_streams.py`

Convert parsed requests into per-node streams.

```bash
uv run -m scripts.eval.azure_to_node_streams \
  --input-csv data/azure_trace/parsed/day_2024-05-16_requests.csv \
  --output-dir data/azure_facility/node_streams \
  --rows 10 \
  --racks-per-row 6 \
  --nodes-per-rack 4 \
  --seed 42
```

### `azure_generate_traces.py`

Generate per-node power traces for `ours` and `splitwise_strict`.

```bash
uv run -m scripts.eval.azure_generate_traces \
  --node-stream-dir data/azure_facility/node_streams \
  --output-root results/azure_facility/node_traces
```

### `azure_aggregate.py`

Aggregate node traces to rack, row, and site traces.

```bash
uv run -m scripts.eval.azure_aggregate \
  --node-traces-root results/azure_facility/node_traces \
  --output-root results/azure_facility/aggregated
```

### `azure_metrics.py`

Write the Azure facility metrics CSVs used by the paper figures.

```bash
uv run -m scripts.eval.azure_metrics \
  --aggregated-root results/azure_facility/aggregated \
  --node-traces-root results/azure_facility/node_traces \
  --metrics-csv results/eval_paper/azure_facility_metrics.csv \
  --ldc-csv results/eval_paper/azure_facility_ldc_15min.csv \
  --site-traces-15min-csv results/eval_paper/azure_facility_site_traces_15min.csv
```

### `oversubscription_figure.py`

Compute the oversubscription capacity analysis and supporting plots.

```bash
uv run -m scripts.eval.oversubscription_figure \
  --aggregated-root results/azure_facility/aggregated \
  --metrics-csv results/eval_paper/azure_facility_metrics.csv
```

### `azure_figures.py`

Generate the Azure facility figures.

```bash
uv run -m scripts.eval.azure_figures \
  --aggregated-root results/azure_facility/aggregated \
  --metrics-csv results/eval_paper/azure_facility_metrics.csv \
  --ldc-csv results/eval_paper/azure_facility_ldc_15min.csv \
  --site-traces-15min-csv results/eval_paper/azure_facility_site_traces_15min.csv \
  --out-dir figures
```

### `hierarchy_figure.py`

Generate the server/rack/row/site hierarchy figure outputs.

```bash
uv run -m scripts.eval.hierarchy_figure \
  --node-trace-dir results/azure_facility/node_traces/ours \
  --aggregated-dir results/azure_facility/aggregated/ours \
  --output-mode separate
```

## Tables and Figures

### `generate_baselines_node_table.py`

```bash
uv run -m scripts.eval.generate_baselines_node_table \
  --metrics-csv results/eval_paper/baselines_node_level.csv \
  --out-tex figures/tables/baselines_node.tex
```

### `generate_trace_fidelity_table.py`

```bash
uv run -m scripts.eval.generate_trace_fidelity_table \
  --iid-dir results/continuous_v1_gmm_bigru/k10_f2/eval_metrics_fullheldout \
  --ar1-dir results/continuous_v1_gmm_bigru/k10_f2_ar1/eval_metrics_fullheldout \
  --ar1-thresh-dir results/continuous_v1_gmm_bigru/k10_f2_ar1_thresh/eval_metrics_fullheldout \
  --output figures/tables/trace_fidelity.tex \
  --format latex
```

### `generate_azure_facility_sizing_table.py`

```bash
uv run -m scripts.eval.generate_azure_facility_sizing_table \
  --metrics-csv results/eval_paper/azure_facility_metrics.csv \
  --out-tex results/eval_paper/azure_facility_sizing_table.tex
```

### `generate_power_cdf_comparison.py`

```bash
uv run -m scripts.eval.generate_power_cdf_comparison \
  --config-ids llama-3-8b_H100_tp1 \
  --num-seeds 5 \
  --out-plot-dir figures/trace_power_cdf_comparison
```

### `feature_sufficiency_figure.py`

```bash
uv run -m scripts.eval.feature_sufficiency_figure \
  --run-manifest results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json \
  --training-data-dir model/training_data \
  --gmm-dir results/continuous_v1_gmm_bigru/k10_f2/gmms \
  --out-figure figures/feature_sufficiency_curve.pdf
```

Key flags: `--bootstrap-samples`, `--seed`, `--out-per-config-csv`,
`--out-summary-csv`, `--out-json`, `--include-configs`, `--device`, `--epochs`,
`--hidden-dim`, `--lr`.

## Support Modules

- `baselines.py` - Shared baseline generation helpers
- `pipeline_utils.py` - Common loading, rollout, and checkpoint resolution helpers
- `azure_defaults.py` - Default paths and constants for the Azure workflow
- `azure_trace_utils.py` - Request loading and trace binning helpers
- `facility.py` - Facility layout and aggregation utilities

## Testing

```bash
uv run -m pytest -x
```
