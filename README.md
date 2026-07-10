# PowerTrace-Sim

PowerTrace-Sim trains and evaluates GMM-BiGRU models that generate realistic GPU power traces for LLM inference workloads.

## Planning Docs

- `EENERGY_PLAN.md`: audited, stage-gated execution plan for the e-Energy paper.
- `THEMES.md`: e-Energy paper framing, current conference themes, and result priorities.
- `cleaning-plan.md`: staged cleanup plan for trimming the repo around grid-facing evaluations.
- `data-path.md`: audited common ingestion, feature, metric, and identity contract.

## Project Structure

```text
powertrace-sim/
├── model/
│   ├── classifiers/      # GMM/BiGRU helpers, features, metrics, trace generation
│   ├── pipeline/         # Reusable train/eval/infer logic
│   ├── scripts/          # Thin CLI wrappers
│   ├── training_data/    # Inventory, throughput, manifest preparation modules
│   ├── tests/            # Consolidated unit/integration tests
│   ├── utils/            # Shared helpers
│   └── throughput_database.json
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

The canonical throughput database is `model/throughput_database.json` for Stage0
and ledger workflows. Trained GMM evaluation and inference use the throughput
bound in their run manifest and request JSONs resolved through hash-checked
lineage; their legacy `--throughput-db` and evaluation `--pair-manifest-csv`
flags are accepted for CLI compatibility but are not read. GMM input paths can
be repo-relative when loading manifests and request JSONs.
Add a repeatable `--bundle-dir data/runs/<campaign_id>/<run_id>` only when a
canonical bundle exists; it is not required for the legacy-data quick start.

Other available entry points:

- `uv run -m model.scripts.compare_gmm_bigru`
- `uv run -m model.scripts.generate_methods_figures`

## Common Data and Physics Path

Legacy ShareGPT pairs and canonical `data/runs/<campaign_id>/<run_id>/` bundles
ingest through `model.training_data.run_record.RunRecord`. The record keeps
native per-GPU power/utilization/memory, validated request timing, architecture,
clock basis, source paths, and SHA-256 provenance. GRU and first-principles
ledger views are derived from that record; the legacy rebuild equivalence gate is:

```bash
uv run -m scripts.gates.phase_b_equivalence --mode new
```

`--mode new` accepts a pre-existing difference only when its normalized
reference and rebuilt-content fingerprints match the recorded baseline. Record
a fresh baseline with `--mode baseline --record` before relying on new-mode
results; legacy filename-only baseline records are rejected.

Canonical bundle `power.csv` rows carry both GPU index and UUID. Ingestion requires
their stable one-to-one mapping, groups a 4 Hz sample across up to 50 ms of
per-GPU capture skew, and rejects any mismatch with the visible GPU count recorded
in the manifest. It retains every power, request-array, and engine column in raw
normalized tables while model views select only required fields.
Bundle power wall time is converted with `manifest.clock.local_utc_offset_s`; no
timestamp folding is used for bundles.
The default bundle-ledger scan (`data/runs/*/*`) ignores support directories that
lack `manifest.json` (such as campaign `logs/`); an explicitly supplied glob is
fail-fast for every directory it selects.

GRU preparation accepts canonical bundles only through repeatable explicit
`--bundle-dir` arguments; it does not auto-discover runs. Each projected dataset
has a compact lineage JSON with source paths, SHA-256 identities, request
retention/drop counts and retained row indices, and the fields not copied from
`RunRecord`. Training binds its lineage identity and train-only throughput into
the run manifest; evaluation rejects a missing or tampered binding.
Per-config cadence is the median of raw trace cadences that agree within 1%;
all traces are projected to that exact grid. Individual power-log gaps of at
most four cadence intervals are linearly interpolated, while longer gaps fail.
Normalization and power clamp bounds are fit from training indices only.
Preparation requires at least three traces so train, validation, and test sets
are disjoint. Prefill/decode rates are calibrated from the training indices and
bound into the trained run manifest; held-out timing never calibrates itself.

The deployable first-principles mean-power kernel is
`model.classifiers.physics`. Rebuild the legacy ledger and its source index,
then refit and export the versioned artifact with explicit inputs:

```bash
uv run python feature-test/build_ledger_cache.py \
  --throughput-db model/throughput_database.json \
  --out feature-test/ledger_cache.npz
uv run python feature-test/peak_and_holdout.py \
  --ledger-cache feature-test/ledger_cache.npz \
  --run-index feature-test/ledger_cache.runs.json
```

This writes `feature-test/results/physics_artifact_v1.json`. The artifact is
deterministic mean power by default; its optional residual layer is disabled.
Both sources use the one reconstruction implementation in
`model.training_data.ledger_view`; the legacy builder no longer carries a
second copy. Ledger construction fails when measured throughput is missing—there is no
synthetic rate fallback. The run index and artifact record source pair IDs,
source hashes, the selected ledger hash, Git revision, and Git dirty status.
Ledger artifacts also record their timestep and per-model architecture
descriptors, allowing full Hugging Face IDs and time-based lag conversion.

## Occupancy Roofline Profiling

Gemma-4-26B-A4B occupancy roofline collection is a composite profiling campaign:
it runs cache-off A100 TP2 capacity probes, long-context holds, a mixed prefill /
decode grid, and a synthetic long-agentic workload, then analyzes the collected
bundles with a reconstruction-based power-vs-occupancy roofline analyzer.

```bash
# Dry run: print the launch plan and write a sample bundle under data/dry-runs/.
bash profiling/jobs/run_campaign.sh profiling/campaigns/roofline_gemma-4-26b-a4b_a100.json

# Live GPU run.
bash profiling/jobs/run_campaign.sh profiling/campaigns/roofline_gemma-4-26b-a4b_a100.json --execute
```

Live campaign bundles are written under `data/runs/<campaign_id>/<run_id>/` by
default. The analyzer writes CSVs to `results/occupancy_roofline/` and figures
to `figures/occupancy_roofline/`. The occupancy coordinate is `ell = f / F + g / G`,
where `F` and `G` are p99.5 sustained 5-second rates reconstructed from the
dedicated prefill and decode probes. Treat roofline and agentic claims as
reconstruction-based until an `engine.csv` state parser is implemented and
validated. It uses the manifest's `clock.local_utc_offset_s` to compare power,
requests, and probe windows on exact UTC epochs; bundle analysis never applies
the legacy 30-minute timestamp fold.

## Splitwise-Style LUT Baseline Notes

The evaluation baseline API uses the Splitwise-style LUT entry points in `scripts/eval/baselines.py`:
- `build_splitwise_style_lut_params(...)`
- `generate_splitwise_style_lut_trace(...)`

Returned LUT params are explicitly namespaced by layer:
- `timing_support_*`
- `power_support_*`
- `scheduler_defaults_*`

## Azure Trace Splitter

`scripts/eval/split_azure_week_to_days.py` defaults to
`data/azure_trace/raw/AzureLLMInferenceTrace_code_1week.csv` and writes
`day_manifest.csv` with source week CSV and emitted day CSV provenance columns.

## Evaluation Metadata Semantics

Main GMM-BiGRU evaluation and inference use IID GMM sampling and record
`generation_mode=iid`. Evaluation uses recorded arrivals with modeled request
durations and records `request_alignment_mode=none`; a measured-power-derived
offset is retained only as the diagnostic `oracle_alignment_offset_s` and is
never applied. Evaluation features begin at `t=dt`, matching training features
and measured targets at `power[1:]`; standalone inference uses the same
next-step alignment and labels its first prediction at `t=dt`. Recorded
checkpoint, normalization, GMM, dataset, split, and lineage hashes are enforced.
Standalone request rows must contain finite, nonnegative
`arrival_time`, `input_tokens`, and `output_tokens`; zero- and one-token
completions remain valid. The retrospective CDF analysis still contains an explicit AR(1)
ablation, but it is not the main evaluation path. Trace accounting
separates `num_skipped_traces`, `num_failed_traces`, and the compatibility total
`num_skipped_or_failed_traces`.

## Testing

```bash
uv run -m pytest -x
```
