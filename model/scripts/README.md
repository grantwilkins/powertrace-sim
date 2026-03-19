# Scripts

This directory is for CLI entry points only. Core logic lives in `model.pipeline` and `model.training_data`.

## Thin Wrapper Scripts

| Script | Delegates to | Purpose |
|---|---|---|
| `train_gmm_bigru.py` | `model.pipeline.training.run_training_from_manifest` | Train GMM-BiGRU artifacts from experimental manifest data |
| `eval_gmm_bigru.py` | `model.pipeline.evaluation.evaluate_from_artifacts` | Evaluate trained artifacts on held-out traces |
| `infer_gmm_bigru.py` | `model.pipeline.inference.run_inference_from_artifacts` | Generate a power trace from request JSON |
| `stage0_inventory.py` | `model.training_data.stage0_inventory_and_throughput.main` | Build Stage0 inventory + pair manifest + throughput DB |
| `prepare_manifest.py` | `model.training_data.manifest.main` | Build `experimental_continuous_v1` datasets/splits/norm params |

## Other Scripts

- `compare_gmm_bigru.py`: experiment comparison utility.
- `generate_methods_figures.py`: methods-paper figure generation.
- `power_regression_analysis.py`: regression and exploratory analysis.

## Typical Workflow

```bash
# 1) Discover Stage0 pairs + throughput model
python -m model.scripts.stage0_inventory --data_root_dir data

# 2) Build experimental manifest artifacts
python -m model.scripts.prepare_manifest \
    --pair-manifest-csv results/stage0/pair_manifest.csv \
    --out-dir results/experimental_continuous_v1

# 3) Train
python -m model.scripts.train_gmm_bigru \
    --manifest results/experimental_continuous_v1/manifest.json \
    --out-root results/continuous_v1_gmm_bigru \
    --k 10

# 4) Evaluate
python -m model.scripts.eval_gmm_bigru \
    --run-manifest results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json \
    --experimental-manifest results/experimental_continuous_v1/manifest.json
```

Use `--help` on each script for full argument details.
