# Scripts

This directory is for CLI entry points only. Core logic lives in `model.pipeline` and `model.training_data`.

## CLI Entry Points

| Module | Purpose |
|---|---|
| `uv run -m model.scripts.stage0_inventory` | Build Stage0 inventory, pair manifest, and throughput database |
| `uv run -m model.scripts.prepare_manifest` | Build `experimental_continuous_v1` datasets, splits, and norm params |
| `uv run -m model.scripts.train_gmm_bigru` | Train GMM-BiGRU artifacts from an experimental manifest |
| `uv run -m model.scripts.eval_gmm_bigru` | Evaluate trained artifacts on held-out traces |
| `uv run -m model.scripts.infer_gmm_bigru` | Generate a power trace from request JSON |
| `uv run -m model.scripts.compare_gmm_bigru` | Compare config-summary CSVs across runs |
| `uv run -m model.scripts.generate_methods_figures` | Generate the methods-figure set used by the paper |

## Typical Workflow

```bash
# 1) Discover Stage0 pairs + throughput model
uv run -m model.scripts.stage0_inventory --data_root_dir data

# 2) Build experimental manifest artifacts
uv run -m model.scripts.prepare_manifest \
    --pair-manifest-csv results/stage0/pair_manifest.csv \
    --out-dir results/experimental_continuous_v1

# 3) Train
uv run -m model.scripts.train_gmm_bigru \
    --manifest results/experimental_continuous_v1/manifest.json \
    --out-root results/continuous_v1_gmm_bigru \
    --k 10

# 4) Evaluate
uv run -m model.scripts.eval_gmm_bigru \
    --run-manifest results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json \
    --experimental-manifest results/experimental_continuous_v1/manifest.json
```

Use `--help` on each module for full argument details, for example:

```bash
uv run -m model.scripts.train_gmm_bigru --help
```
