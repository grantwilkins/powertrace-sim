# Handoff Checklist (Bench → Train → Infer)

Date: 2026-01-29

This repo’s top priority workflows are:
1) bench collection
2) training dataset generation + training
3) inference (power trace generation)
Server/datacenter sims are secondary; plotting can break.

## Current “source of truth” direction
- Per-request measurements should come from `<results>.requests.jsonl` (preferred) and *not* be re-derived later.
- Training data prep should prefer recorded request timestamps, with explicit fallback behavior.
- Perf stats extraction should use the same per-request records for TTFT/TPOT sampling.

## Quick orientation (what already exists)
Docs:
- `docs/data_flow.md`
- `docs/workflows.md`

Bench changes:
- `client/backend_request_func.py` now sets `request_timestamp` at send time for all backends.
- `client/benchmark_serving.py` writes `e2els`, `tpots`, and aligned `request_timestamps` in the results JSON.
- `client/benchmark_serving.py` supports `--save-requests-jsonl` to write `<result>.requests.jsonl`.

Training prep changes:
- `model/training_data/utils/prepare_training_data.py` supports:
  - `--prefer_requests_jsonl`
  - `--timestamp_source {poisson,recorded,recorded_or_poisson,recorded_scaled_or_poisson}`
  - `--decode_time_source {legacy_itl_sum,e2e_minus_ttft}`
- Timestamp alignment core is in `model/training_data/utils/request_timestamps.py`
- Unittest exists: `model/training_data/utils/test_timestamp_alignment_unittest.py`

Perf DB changes:
- `model/utils/extract_performance_stats.py` prefers `<benchmark>.requests.jsonl` when present.

Inference entrypoint:
- `model/scripts/simulate_server_power.py` (supports `--use-fast-workload` to avoid ServeGen dependency)

## Step 0 — Repo hygiene (strongly recommended before more refactors)
Goal: avoid accidental diffs and huge noisy commits.
- Decide what to do with `ServeGen/` modifications (revert or intentionally vendor).
- Remove tracked `*.pyc` artifacts from git (they currently exist in history).
- Keep `.gitignore` updated to ignore `__pycache__/` and `*.pyc`.

Suggested commands (review before running):
- `git status`
- `git diff`
- `git checkout -- ServeGen` (if ServeGen diffs are accidental)
- `git rm -r --cached **/__pycache__ **/*.pyc` (if pycs are tracked; adjust globbing as needed)
- ensure `.gitignore` contains `__pycache__/` and `*.pyc`

## Step 1 — Introduce Run Bundles (reduce filename inference)
Goal: stop inferring model/tp/hw/qps/date from filenames; record metadata once.

Create a “run bundle” directory layout:
- `runs/<run_id>/run.json`
- `runs/<run_id>/requests.jsonl`  (copy from `<result>.requests.jsonl`)
- `runs/<run_id>/power_raw.csv`   (original nvidia-smi)
- `runs/<run_id>/power_agg.csv`   (TP-GPU aggregated power with epoch seconds)

Implement:
- `model/scripts/make_run_bundle.py` (new)
  - Inputs: `--result-json`, `--power-csv`, `--out-dir`, optional overrides (`--hardware`, `--model-name`, etc)
  - Writes `run.json` (minimal schema from `docs/data_flow.md`)
  - Copies/renames the sidecar `requests.jsonl`
  - Generates `power_agg.csv` using the same aggregation logic currently in `parse_power_csv()`

Acceptance:
- One run bundle can be created from a single benchmark JSON + power CSV without manual renaming.

## Step 2 — Update data prep to consume Run Bundles (not raw directories)
Goal: consistent training dataset generation using recorded timestamps.

Implement:
- Add `--run_bundle_dir` (or `--runs_dir`) mode to `model/training_data/utils/prepare_training_data.py`
- Prefer `requests.jsonl` as input; stop relying on `extract_results_info()` + filename regex.
- Keep legacy directory-scanning mode for backward compatibility, but mark it “legacy”.

Defaults to recommend for new runs:
- `--prefer_requests_jsonl`
- `--timestamp_source recorded_scaled_or_poisson`
- `--decode_time_source e2e_minus_ttft`

Acceptance:
- Training dataset generation for a run bundle produces the same NPZ keys as before.
- Timestamp alignment behavior is explicit and recorded in a small manifest JSON next to the NPZ.

## Step 3 — Update perf stats builder to consume Run Bundles
Goal: TTFT/TPOT models derived from the same per-request schema.

Implement:
- Extend `model/utils/extract_performance_stats.py` to optionally accept `--runs_dir`
- Read model/tp/hw/qps from `run.json` instead of path heuristics
- Read TTFT/TPOT samples from `requests.jsonl` (ignore legacy ITL formats when JSONL present)

Acceptance:
- Perf DB generation runs without needing filename patterns.
- Mixed legacy+new runs still work.

## Step 4 — Add Model Profiles (single inference source of truth)
Goal: inference should not guess mappings; it should load one profile.

Implement:
- New `profiles/<profile_id>/profile.json` schema (see `docs/data_flow.md`)
- New `model/scripts/build_profile.py` that:
  - selects weights from `model/best_weights/`
  - pulls TTFT/TPOT model params from perf DB
  - pulls GMM mu/sigma from NPZ-derived artifacts (or explicitly computes/stores)
- Update `model/scripts/simulate_server_power.py` to accept `--profile-json` and bypass implicit mapping.

Acceptance:
- Running inference requires only `profile.json` + weights file, no hidden heuristics.

## Step 5 — Add “golden” smoke tests (no heavy deps)
Goal: detect accidental behavior drift early.

Add stdlib `unittest` tests that avoid pandas/pyarrow:
- request timestamp alignment (already present)
- decode time derivation correctness (e2e-ttft and legacy fallback)
- per-request arrays alignment (same length across successes/ttfts/e2els/tpots/request_timestamps)

Optionally add a tiny fixture `tests/fixtures/requests.jsonl` (handwritten 3–5 lines).

Acceptance:
- `python -m unittest` passes in minimal env.

## Step 6 — (Defer) datacenter_24h_10mw cleanup
Not a priority, but note known inconsistencies for later:
- duration says 7 days but downsampling assumes 24h.
- hard-coded absolute trace path.

## Recommended “happy path” commands (new runs)
Bench:
- `python client/benchmark_serving.py ... --save-result --save-detailed --save-requests-jsonl`

Run bundle:
- `python model/scripts/make_run_bundle.py --result-json ... --power-csv ... --out-dir runs/<run_id>`

Dataset:
- `python model/training_data/utils/prepare_training_data.py --run_bundle_dir runs/<run_id> --timestamp_source recorded_scaled_or_poisson --decode_time_source e2e_minus_ttft`

Perf DB:
- `python model/utils/extract_performance_stats.py --runs_dir runs --output_file model/config/performance_database.json`

Inference:
- `python model/scripts/simulate_server_power.py ... --use-fast-workload --out-csv ...`
