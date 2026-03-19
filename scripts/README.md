# Scripts

This directory contains the evaluation entrypoints and a few shell wrappers.
The current command-line surface lives in `scripts/eval/`; training entrypoints
live under `model/scripts/`.

## Layout

```text
scripts/
├── eval/                 # Evaluation, Azure pipeline, tables, figures
├── collect_random_weights.sh
├── run_ablation_study.sh
└── run_training.sh
```

## Common Entry Points

```bash
uv run -m scripts.eval.run_azure_pipeline
uv run -m scripts.eval.run_baselines_node
uv run -m scripts.eval.run_baselines_node_groundtruth
uv run -m scripts.eval.run_baselines_facility
uv run -m scripts.eval.generate_power_cdf_comparison
uv run -m scripts.eval.generate_baselines_node_table
uv run -m scripts.eval.generate_trace_fidelity_table
uv run -m scripts.eval.generate_azure_facility_sizing_table
```

## See Also

- [scripts/eval/README.md](eval/README.md) - Current evaluation CLI reference
- [model/scripts/README.md](../model/scripts/README.md) - Training and inference entrypoints

## Testing

```bash
uv run -m pytest -x
```
