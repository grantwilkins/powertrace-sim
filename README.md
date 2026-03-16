# PowerTrace-Sim

PowerTrace-Sim trains and evaluates GMM-BiGRU models that generate realistic GPU power traces for LLM inference workloads.

## Project Structure

```text
powertrace-sim/
├── model/
│   ├── classifiers/      # GMM/GRU implementations and metrics
│   ├── config/           # Throughput and performance JSON databases
│   ├── pipeline/         # Reusable train/eval/infer logic
│   ├── scripts/          # Thin CLI wrappers
│   ├── training_data/    # Inventory, throughput, manifest preparation modules
│   ├── tests/            # Consolidated unit/integration tests
│   └── utils/            # Shared helpers
├── profiling/            # Data collection client/server/jobs
│   ├── client/
│   ├── server/
│   └── jobs/
├── scripts/eval/         # Evaluation and baseline scripts
├── data/
├── results/
├── figures/
└── training_results/
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
# 1) Stage0 inventory + throughput extraction
python -m model.scripts.stage0_inventory --data_root_dir data

# 2) Build experimental manifest datasets/splits
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

# 5) Inference
python -m model.scripts.infer_gmm_bigru \
    --config-id llama-3-8b_H100_tp1 \
    --requests-json input_requests.json \
    --out-csv generated_power.csv
```

## Testing

```bash
pytest
```
