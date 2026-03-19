# Scripts

This directory contains evaluation scripts, baseline implementations, and paper reproduction utilities.

## Directory Structure

```
scripts/
└── eval/                         # Evaluation and paper scripts
    ├── run_baselines_*.py        # Baseline comparison runners
    ├── azure_*.py                # Azure trace processing
    ├── generate_*_table.py       # LaTeX table generation
    ├── *_figure.py               # Paper figure generation
    ├── baselines.py              # Baseline method implementations
    └── pipeline_utils.py         # Shared utilities
```

## Quick Reference

### Running Baselines

```bash
# Node-level evaluation
python -m scripts.eval.run_baselines_node \
    --config-id llama-3-8b_H100_tp1 \
    --num-seeds 5

# Facility-level evaluation
python -m scripts.eval.run_baselines_facility \
    --facility-data results/azure_facility
```

### Processing Azure Traces

```bash
# Parse raw traces
python -m scripts.eval.parse_azure_trace --input raw.csv --output parsed/

# Generate power traces
python -m scripts.eval.azure_generate_traces \
    --requests data/azure_trace/node_requests \
    --out-dir results/azure_facility/node_traces

# Aggregate to facility level
python -m scripts.eval.azure_aggregate \
    --input results/azure_facility/node_traces \
    --output results/azure_facility/aggregated
```

### Generating Tables and Figures

```bash
# Collect results
python -m scripts.eval.collect_results \
    --results-dir results/eval_paper \
    --out-csv results/eval_paper/all_metrics.csv

# Generate LaTeX tables
python -m scripts.eval.generate_baselines_node_table \
    --metrics-csv results/eval_paper/all_metrics.csv

# Generate figures
python -m scripts.eval.generate_power_cdf_comparison \
    --out-dir figures/trace_power_cdf_comparison
```

## See Also

- [scripts/eval/README.md](eval/README.md) - Detailed documentation for all evaluation scripts
- [model/scripts/README.md](../model/scripts/README.md) - Training and inference scripts
