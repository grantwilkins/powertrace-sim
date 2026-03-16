# Model Package

`model/` contains the reusable training/evaluation/inference library code plus thin CLI wrappers and tests.

## Package Structure

```text
model/
├── classifiers/        # GRU/GMM model code and metrics
├── config/             # Throughput/performance JSON databases
├── pipeline/           # Reusable train/eval/infer modules
├── scripts/            # Thin CLI wrappers over pipeline/training_data modules
├── training_data/      # Data discovery, alignment, manifest preparation
├── tests/              # Consolidated test suite
├── utils/              # Shared I/O, config, decode-time and stats helpers
└── summary_json.py     # Summary writer
```

## End-to-End Flow

1. Stage0 discovery and throughput extraction:

```bash
python -m model.scripts.stage0_inventory --data_root_dir data
```

2. Build experimental manifest artifacts:

```bash
python -m model.scripts.prepare_manifest \
    --pair-manifest-csv results/stage0/pair_manifest.csv \
    --out-dir results/experimental_continuous_v1
```

3. Train GMM-BiGRU models:

```bash
python -m model.scripts.train_gmm_bigru \
    --manifest results/experimental_continuous_v1/manifest.json \
    --out-root results/continuous_v1_gmm_bigru \
    --k 10
```

4. Evaluate trained artifacts:

```bash
python -m model.scripts.eval_gmm_bigru \
    --run-manifest results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json \
    --experimental-manifest results/experimental_continuous_v1/manifest.json
```

5. Generate traces for a request stream:

```bash
python -m model.scripts.infer_gmm_bigru \
    --config-id llama-3-8b_H100_tp1 \
    --requests-json input_requests.json \
    --out-csv generated_power.csv
```

## Testing

```bash
pytest
```
