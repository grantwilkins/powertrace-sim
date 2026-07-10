# Eval Paper Artifact Map

> **Stale after the 2026-07-09 data-path audit.** Existing generated outputs
> predate disjoint splits, train-only throughput calibration, strict artifact
> identities, corrected diversity semantics, and required IID provenance. Do
> not cite them as current results. Regenerate datasets and models first, then
> rebuild this directory with the producers below.

`results/eval_paper/` contains checked-in paper support artifacts: CSV/JSON
inputs for plots and LaTeX tables consumed by the manuscript. Treat these files
as generated outputs from maintained scripts; update them by rerunning the
producer and reviewing the diff, not by editing values by hand. Related PDFs and
PNGs live under `figures/`.

Artifact classes:

- measured: derived from recorded GPU power traces or production request traces
- fitted: derived from trained model outputs or evaluation summaries
- generated: simulated or replayed power traces
- aggregated: rack, row, site, or summary-level reductions
- rendered: figure/table outputs derived from upstream artifacts

## Main paper artifacts

| Artifact family | Checked-in outputs | Producer | Upstream | Class | Seed and runtime notes |
| --- | --- | --- | --- | --- | --- |
| Trace fidelity table | `trace_fidelity_table.tex`, `trace_fidelity_table_gptoss_a100.tex` | `uv run -m scripts.eval.generate_trace_fidelity_table --output <path> --format latex` | `results/continuous_v1_gmm_bigru/*/eval_metrics_fullheldout/` | fitted, rendered | No new randomness; seconds once eval summaries exist. |
| Node baseline metrics and table | `baselines_node_level.csv`, `baselines_node_table.csv`, `baselines_node_table.json`, `baselines_node_table.tex` | `uv run -m scripts.eval.run_baselines_node`, then `uv run -m scripts.eval.generate_baselines_node_table` | run manifest, experimental manifest, pair manifest, throughput data, Splitwise perf model | generated, rendered | Defaults use `--num-seeds 5 --base-seed 42`; table generation is seconds after metrics exist. |
| Held-out node replay | `baselines_node_groundtruth_metrics.csv`; figure: `figures/baselines_node_groundtruth_trace.pdf` | `uv run -m scripts.eval.run_baselines_node_groundtruth` | held-out measured trace, manifests, throughput data, trained generator, Splitwise LUT | measured, generated, rendered | Defaults are deterministic for fixed `--base-seed`; expected runtime is one held-out replay. |
| Azure facility profile | `azure_facility_metrics.csv`, `azure_facility_ldc_15min.csv`, `azure_facility_site_traces_15min.csv`; figures: `figures/azure_figure_*.pdf`, `figures/azure_figure_manifest.json` | `uv run -m scripts.eval.run_azure_pipeline`, or `azure_metrics` plus `azure_figures` for partial regeneration | `data/azure_trace/`, `data/azure_facility/node_streams/`, `results/azure_facility/node_traces/`, `results/azure_facility/aggregated/` | measured, generated, aggregated, rendered | Node stream defaults use seed 42; full pipeline is the expensive path, figures are quick once aggregation exists. |
| Facility sizing table | `azure_facility_sizing_table.csv`, `azure_facility_sizing_table.json`, `azure_facility_sizing_table.tex` | `uv run -m scripts.eval.generate_azure_facility_sizing_table` | `azure_facility_metrics.csv` | aggregated, rendered | No new randomness; seconds after facility metrics exist. |
| Hierarchy smoothing figure | `azure_hierarchy_figure.csv`, `azure_hierarchy_figure.json`; figures: `figures/azure_hierarchy_figure.pdf` or `figures/azure_hierarchy_{server,rack,row,site}.pdf` | `uv run -m scripts.eval.hierarchy_figure` | `results/azure_facility/node_traces/ours/`, `results/azure_facility/aggregated/ours/` | aggregated, rendered | No new randomness; quick once Azure traces exist. |
| Oversubscription sensitivity | `azure_oversubscription_capacity.csv`, `azure_oversubscription_capacity.json`; figures: `figures/azure_oversubscription_capacity.pdf`, `figures/azure_oversubscription_lines.pdf` | `uv run -m scripts.eval.oversubscription_figure` | Azure aggregated rack traces and `azure_facility_metrics.csv` | aggregated, rendered | Defaults use `--seed 42`; runtime depends on sampling settings. |

## Appendix and support artifacts

| Artifact family | Checked-in outputs | Producer | Upstream | Class | Seed and runtime notes |
| --- | --- | --- | --- | --- | --- |
| Synthetic facility baselines | `baselines_facility_metrics.csv` | `uv run -m scripts.eval.run_baselines_facility` | generator manifests, throughput data, baseline settings | generated, aggregated | Uses explicit seed controls in the script; runtime scales with node count and duration. |
| Node-level summary rollup | `node_level_summary.csv` | `uv run -m scripts.eval.collect_results` | `results/continuous_v1_gmm_bigru/k10_f2*/eval_metrics/config_summary.csv` | fitted, aggregated | No new randomness; seconds. |
| Power CDF comparison | `trace_power_cdf_comparison_points.csv`, `trace_power_cdf_comparison.csv`, `trace_power_cdf_comparison.json`; figures: `figures/trace_power_cdf_comparison/*_power_cdf.{pdf,png}` | `uv run -m scripts.eval.generate_power_cdf_comparison` | run manifest, experimental manifest, throughput data, pair manifest | measured, generated, rendered | Defaults use `--num-seeds 5 --base-seed 42`; runtime scales with selected configs. |
| Feature sufficiency | `feature_sufficiency_per_config.csv`, `feature_sufficiency_summary.csv`, `feature_sufficiency_manifest.json`; figure: `figures/feature_sufficiency_curve.pdf` | `uv run -m scripts.eval.feature_sufficiency_figure` | run manifest, training data, GMM params | fitted, rendered | Defaults use `--seed 42`; retrains reduced-feature models. |
| Appendix A1 surrogate validity | `appendix_a1_trace_metrics.csv`, `appendix_a1_config_summary.csv`, `appendix_a1_manifest.json`; figures: `figures/appendix_a1_*` | `uv run -m scripts.eval.appendix_surrogate_validity` | run manifest, experimental manifest, pair manifest | measured, generated, rendered | Deterministic for fixed config selection and thresholds. |
| Splitwise arrival alignment | `splitwise_arrival_alignment_summary.csv` | No current maintained producer was found in `scripts/eval/` during this audit. | Existing checked-in support artifact | rendered | Keep only while referenced by the manuscript or replace with a maintained producer. |

## Regeneration policy

Prefer the entry points documented in `scripts/eval/README.md`. Preserve the
large upstream artifacts under `data/`, `results/azure_facility/`,
`results/continuous_v1_gmm_bigru/`, `figures/`, and `feature-test/results/`
unless the task explicitly asks to regenerate them.
