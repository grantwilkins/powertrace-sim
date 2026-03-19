# PowerTrace-Sim

PowerTrace-Sim trains and evaluates GMM-BiGRU models that generate realistic GPU power traces for LLM inference workloads.

## Project Structure

```text
powertrace-sim/
├── model/
│   ├── classifiers/      # GMM/BiGRU helpers, features, metrics, trace generation
│   ├── config/           # Throughput JSON databases
│   ├── pipeline/         # Reusable train/eval/infer logic
│   ├── scripts/          # Thin CLI wrappers
│   ├── training_data/    # Inventory, throughput, manifest preparation modules
│   ├── tests/            # Consolidated unit/integration tests
│   └── utils/            # Shared helpers
├── profiling/            # Data collection client/server/jobs
│   ├── client/
│   ├── jobs/
│   └── server/
├── scripts/
│   └── eval/             # Evaluation and baseline scripts
├── data/
├── results/
├── pyproject.toml
└── uv.lock
```

## Setup

### `uv` only

```bash
# Install uv
brew install uv
# or: curl -LsSf https://astral.sh/uv/install.sh | sh

# Create/sync the project environment from pyproject.toml + uv.lock
uv sync

# Optional: activate the venv directly
source .venv/bin/activate
```

## Quick Start

```bash
# 1) Stage0 inventory + throughput extraction
uv run -m model.scripts.stage0_inventory --data_root_dir data

# 2) Build experimental manifest datasets/splits
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

# 5) Inference
uv run -m model.scripts.infer_gmm_bigru \
    --config-id llama-3-8b_H100_tp1 \
    --requests-json input_requests.json \
    --out-csv generated_power.csv
```

Other available entry points:

- `uv run -m model.scripts.compare_gmm_bigru`
- `uv run -m model.scripts.generate_methods_figures`

## Splitwise-Style LUT Baseline Notes

The evaluation baseline API uses the Splitwise-style LUT entry points in `scripts/eval/baselines.py`:
- `build_splitwise_style_lut_params(...)`
- `generate_splitwise_style_lut_trace(...)`

Returned LUT params are explicitly namespaced by layer:
- `timing_support_*`
- `power_support_*`
- `scheduler_defaults_*`

## Testing

```bash
uv run -m pytest -x
```
