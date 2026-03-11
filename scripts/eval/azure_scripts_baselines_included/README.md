# Azure Baselines-Included Pipeline (Isolated)

This package is an isolated Azure facility pipeline that includes:
- `ours`
- `splitwise_lut` (tuned)
- `splitwise_strict`
- plus constant facility baselines in metrics/tables: `tdp_baseline`, `mean_baseline`

It does not modify or depend on outputs from the existing `scripts/eval/azure_*.py` pipeline.

## Default output roots

- `results/azure_facility_baselines_included/node_traces/{method}`
- `results/azure_facility_baselines_included/aggregated/{method}`
- `results/eval_paper_baselines_included/azure_facility_metrics.csv`
- `results/eval_paper_baselines_included/azure_facility_ldc_15min.csv`
- `results/eval_paper_baselines_included/azure_facility_site_traces_15min.csv`
- `results/eval_paper_baselines_included/azure_facility_sizing_table.{csv,json,tex}`
- `figures/azure_baselines_included/*`

## One-command run

```bash
python -m scripts.eval.azure_scripts_baselines_included.run_pipeline
```

## Step-by-step

```bash
python -m scripts.eval.azure_scripts_baselines_included.azure_generate_traces
python -m scripts.eval.azure_scripts_baselines_included.azure_aggregate
python -m scripts.eval.azure_scripts_baselines_included.azure_metrics
python -m scripts.eval.azure_scripts_baselines_included.azure_figures
python -m scripts.eval.azure_scripts_baselines_included.generate_azure_facility_sizing_table
```

## TP scaling default

Splitwise defaults to TP-matched accounting:
- `tp_gpus`: from config id suffix (`*_tpX`) unless overridden
- `n_gpus_per_node`: defaults to `tp_gpus` unless overridden

You can override with CLI flags where needed for ablations.
