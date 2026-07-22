# Training Data Modules

This package contains shared data discovery, parsing, alignment, and ledger
preparation. The historical experimental-manifest builder remains for evidence
equivalence; GMM-BiGRU training and inference live only in the archive.

## Module Layout

```text
training_data/
├── inventory.py                # Dataset and file-pair discovery
├── throughput.py               # Request metric extraction and throughput models
├── stage0_inventory_and_throughput.py
│                               # Stage0 orchestrator for inventory + pair manifest + throughput DB
├── manifest.py                 # Experimental manifest/dataset/split generation
├── power_parsing.py            # Power CSV and request JSON parsing
├── alignment.py                # Time-grid alignment and active-request series
├── arch.py                     # Shared legacy architecture registry
├── run_record.py               # Common legacy/bundle ingestion contract
├── ledger_view.py              # First-principles work reconstruction view
├── normalization.py            # Feature normalization and train/val/test split helpers
└── losses/                     # Saved training-loss arrays used by analysis scripts
```

## Standard Pipeline

1. Build Stage0 inventory and throughput database:

```bash
uv run -m model.scripts.stage0_inventory --data_root_dir data
```

2. Build experimental manifest artifacts:

```bash
uv run -m model.scripts.prepare_manifest \
    --pair-manifest-csv results/stage0/pair_manifest.csv \
    --out-dir results/experimental_continuous_v1
```

`--bundle-dir data/runs/<campaign_id>/<run_id>` is repeatable and explicit when
canonical bundles exist; preparation never scans for bundles and legacy-only
preparation needs no bundle flag.
Each written dataset has a sibling `*.lineage.json` with source paths and
SHA-256 identities plus request projection/drop counts and retained row indices.
Normalization and clamp bounds use only `train_indices`. Preparation requires three traces and one
per-config median timestep: raw trace cadences must agree within 1%, then every
trace is projected to the exact median grid. Individual power-log gaps of at
most four cadence intervals are linearly interpolated; longer gaps fail.
Preparation binds throughput calibrated only on training
request timing. Token counts, TTFT, raw ITL sequences, and decode duration
remain in `RunRecord`; the exact ledger uses cumulative ITLs for post-first
decode completion times and records any chunk/token-count exclusions.
Trained GMM evaluation and inference then use the bound throughput and
hash-checked lineage; they do not read the mutable Stage0 throughput database
or pair manifest.

3. Historical GMM-BiGRU training and evaluation are run from
   `archive/gmm_bigru_v1/`; selected-model preparation and fitting use
   `model.scripts.prepare_data` and `model.scripts.train`.

## Notes

- Tests for shared modules live under `model/tests/`; historical model tests
  live under `archive/gmm_bigru_v1/tests/`.
- Canonical live bundles use `data/runs/<campaign_id>/<run_id>/`; legacy pairs
  remain supported through the same `RunRecord` contract.
- Canonical power rows require GPU index/UUID. The parser groups by timestamp and
  identity, derives the observed device set, and rejects manifest topology drift.
- `RunRecord` keeps raw device, request-array, and engine tables plus hashes;
  model views are explicit narrow projections. Bundle power wall time is corrected
  with `manifest.clock.local_utc_offset_s`, without the legacy timestamp fold.
- Research audits may request `keep_power_gaps=True` from the ledger view. It
  returns the full feature grid, NaN at unobserved power targets, and a
  `power_valid` mask. The default path still drops invalid targets, preserving
  existing cache behavior exactly.
- GRU preparation explicitly omits the architecture descriptor it does not
  consume, so unknown legacy model architectures can still be projected.
  Ledger/physics ingestion keeps architecture lookup mandatory.
- The old `training_data/utils/` package has been removed; imports should reference the top-level modules above.
- Legacy helpers such as `prepare_experimental_manifest.py` and `prepare_training_data.py` are not present in the current tree.
