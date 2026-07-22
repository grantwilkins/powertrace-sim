# Clean Model Migration

This document is the implementation contract for making the selected
architecture-aware timing and clean v4 power model the PowerTrace-Sim default.
It deliberately separates the maintained model from the historical
GMM-BiGRU example and from rejected research candidates.

## Target pipeline

```text
request schedule
  -> continuous-batching scheduler and KV admission
  -> architecture-derived iteration work and timing
  -> native 250 ms work ledger
  -> clean dense or support-bounded MoE power surface
  -> per-request timing and node GPU power
```

The selected equations are frozen. Training may refit their coefficients but
must not rerun a candidate ladder or use sealed targets for selection. The
default artifact remains `pre_sealed` until the external validation gate passes.

## Phase checklist

### 0. Freeze the reference

- [x] Record the current worktree and selected prototype paths.
- [x] Run the pre-migration suite (`394 passed`).
- [x] Add semantic tests for power-domain scaling, utilization coordinates,
  cached prefixes, seeded workload realization, and support enforcement.
- [ ] Create the named baseline commit/tag after separating scientific evidence
  from unrelated generated outputs.

### 1. Preserve GMM-BiGRU

- [x] Create a self-contained `archive/gmm_bigru_v1/` snapshot.
- [x] Move the complete tracked prepared datasets, checkpoints,
  GMM/normalization/AR artifacts, manifests, curves, and metric trees into the
  runnable archive before pruning.
- [ ] Remove duplicate reruns after documenting why their metrics differ.
- [x] Remove regenerable evaluation plot copies while retaining training curves.
- [x] Give the archive its own README, dependency extra, and focused tests.
- [x] Remove BiGRU commands from the default package without compatibility shims.
- [x] Move the BiGRU-specific node/facility comparisons, retrospective figures,
  trace-fidelity table, request-rate sweep, and their tests into the archive.
- [x] Retire B2 fitting and relative gates from the maintained feature evaluator
  while preserving the frozen B2 implementation and test in the archive.
- [x] Remove the BiGRU implementation modules and remaining training/sweep
  artifacts from the maintained package and root results tree.

### 2. Promote the selected implementation

- [x] Promote iteration work and roofline timing into `model/timing/`.
- [x] Promote the continuous-batching scheduler and 250 ms ledger projection.
- [x] Promote the clean dense and bounded-MoE power equations into `model/power/`.
- [x] Remove Torch and disk intermediates from the selected inference path.
- [ ] Replace remaining maintained imports from `timing-test/`, `power-test/`,
  and `feature-test/` with the promoted modules.

### 3. Consolidate artifacts and data preparation

- [x] Create one compact `powertrace-release-v1` inference bundle containing
  timing calibration, power surfaces, architectures, support, and presets.
- [x] Add a prepared-dataset schema and hash-complete source/split manifest.
- [x] Make `prepare_data` validate and bind the canonical prepared payloads.
- [x] Make `train` refit only the frozen timing and clean power equations.
- [x] Verify deterministic artifact regeneration within documented numerical
  tolerance. Large NPZ payloads remain outside Git.

### 4. Public inference

- [x] Add strict fixed or categorical output-length request semantics.
- [x] Add explicit cached-prefix semantics.
- [x] Add named deployment presets and recorded overrides.
- [x] Enforce support by default with explicit unsupported extrapolation.
- [x] Emit `power.csv`, `requests.csv`, and `manifest.json`.
- [x] Convert facility consumers to a genuinely incremental bin iterator whose
  retained memory does not grow with emitted bins.

### 5. Evaluation and paper regeneration

- [x] Add the default `evaluate` command using the common inference contract.
- [x] Migrate the maintained Azure facility generator to the selected model,
  including calibrated idle traces for empty nodes.
- [ ] Retain timing parity, representative dense/MoE traces,
  compatibility/coverage tables, sealed scores, and facility figures.
- [ ] Retain Qwen, BurstGPT, and OpenHands transfer panels as labeled appendix
  evidence; archive rejected candidates and diagnostic-only renderers.

### 6. Default repository cleanup

- [x] Replace the README quick start and package descriptions.
- [ ] Reduce `model/scripts/` and `scripts/` to maintained lifecycle and paper
  entry points.
- [x] Split minimal runtime, training/evaluation, profiling, plotting, and
  `archive-bigru` dependencies.
- [ ] Remove obsolete helpers and prototype compatibility branches.
- [x] Run `uv run -m pytest -x` and each primary CLI.

### 7. Refit and seal

- [x] Refit from the canonical prepared dataset after equivalence passes; every
  released timing and power coefficient regenerated exactly.
- [ ] Regenerate the final paper allowlist after the archived evidence is curated.
- [ ] Score sealed data once without refitting or threshold changes.

The sealing gate is median absolute end-to-end timing error at most 10%, total
energy error at most 6%, ACF-MAE at most 0.05, ACF R2 at least 0.90, and
range-normalized RMSE at most 0.20, with every supported run reported.
Normalized Soft-DTW remains report-only under the preregistered protocol.

### 8. Disaggregated extension

After colocated v1 is clean, add separate prefill, decode, and KV-transfer
ledgers and role-aware power composition. Two role GPUs must never be labeled
TP2. The existing Sherlock campaign is preserved as active evidence, not as a
default-model claim.

## Maintained interface

The intended command surface is:

```bash
uv run -m model.scripts.prepare_data
uv run -m model.scripts.train
uv run -m model.scripts.evaluate
uv run -m model.scripts.infer \
  --requests requests.json \
  --deployment llama-3-70b-a100-tp4 \
  --out-dir outputs/run
```

Inference is deterministic after any categorical output lengths are realized.
`--seed` controls only that realization. The model runs exclusively at its
validated native 250 ms grid; coarser grids are aggregations. `power.csv`
reports both TP-summed node GPU watts and mean active-GPU watts and never
pretends to predict distinct device traces.

## Release status

The clean v4 model is already the selected winner used by the `power_trace_*`
and compatibility evaluations. Cleanup does not reopen that decision. Until
the final campaign is complete, the bundled artifact is the repository default
with an explicit `pre_sealed` status and its known timing/transfer limitations.
