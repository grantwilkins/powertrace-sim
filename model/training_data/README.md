# Training Data Modules

This package contains data discovery, parsing, alignment, and manifest preparation for the GMM-BiGRU pipeline.

## Module Layout

```text
training_data/
├── inventory.py                     # Dataset and file-pair discovery
├── throughput.py                    # Request metric extraction and throughput models
├── stage0_inventory_and_throughput.py
│                                   # Stage0 orchestrator (inventory + pair manifest + throughput DB)
├── manifest.py                      # Experimental manifest/dataset/split generation
├── power_parsing.py                 # Power CSV + request JSON parsing
├── alignment.py                     # Time-grid alignment and active-request series
├── normalization.py                 # Feature normalization and train/val/test split helpers
├── request_timestamps.py            # Timestamp utilities
├── data_file_renamer.py             # Data filename normalization helper
├── prepare_training_data.py         # Legacy pandas-oriented preparation path
├── prepare_experimental_manifest.py # Backward-compatible wrapper to manifest.py
└── losses/plot_loss.py              # Loss plotting utility
```

## Standard Pipeline

1. Build Stage0 inventory and throughput database:

```bash
python -m model.scripts.stage0_inventory --data_root_dir data
```

2. Build experimental manifest artifacts:

```bash
python -m model.scripts.prepare_manifest \
    --pair-manifest-csv results/stage0/pair_manifest.csv \
    --out-dir results/experimental_continuous_v1
```

3. Train/evaluate with `model.scripts.train_gmm_bigru` and `model.scripts.eval_gmm_bigru`.

## Notes

- Tests for these modules live under `model/tests/`.
- The old `training_data/utils/` package has been removed; imports should reference the top-level modules above.
