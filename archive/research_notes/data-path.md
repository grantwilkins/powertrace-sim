# data-path: the common data and metric path

Written 2026-07-09 from a four-way code trace (profiling cradle, GRU stream,
Azure stream + metric census, feature-test ledger stream). Every claim below
was verified against code or opened artifacts, with file:line citations kept
in the divergence registry. This document defines the unification target for
EENERGY_PLAN W2/W3 and extends cleaning-plan.md Phases 2-3.

Vocabulary used throughout: **run record** (the proposed common ingestion
contract), **the GRU stream** (GMM+BiGRU train/generate/eval), **the ledger
stream** (feature-test first-principles), **the Azure stream** (facility
generation and metrics).

Sections 1-2 preserve the pre-repair audit snapshot. Section 6 is the current
implementation record; where they differ, Section 6 governs.

## 1. The map: cradle to grave, as the code actually is

### 1a. Measurement cradle

```text
LEGACY (100% of real data)                      BUNDLE (0 runs on disk)
data/sharegpt-benchmark-<model>-<hw>/           data/runs/<run_id>/   [flat, not
  <model>_tp<N>_p<rate>_d<date>.csv               the documented 2-level layout]
    per-GPU power, 4 cols, 250 ms                 power.csv    8 cols per-GPU
    (timestamp, power.draw, util.gpu,             engine.csv   vLLM /metrics 4 Hz
     memory.used; local-time strings)                          (WRITE-ONLY today)
  vllm-<rate>qps-tp<N>-<served>-<date>.json       requests.json 5 arrays
    input_lens, output_lens, ttfts,               manifest.json identity + arch
    itls (mean scalar/req),                                     + server + clock
    request_timestamps (epoch)
IDENTITY: parsed from dir/file NAMES            IDENTITY: recorded in-band
ARCH: hardcoded dict (feature-test)             ARCH: arch_extract from config.json
```

Facts that shape everything downstream:
- The documented bundle has **never been produced**; `data/runs/` is empty.
  All real data is legacy format. The bundle emitter and its only consumer
  already disagree (manifest key path, directory nesting) — fixable now at
  zero cost because no data exists yet.
- `engine.csv` is collected by the scraper and read by nothing but a plotter;
  the measured-state parser (`bins_from_engine_csv`) is a NotImplementedError
  stub. All "work" features everywhere are reconstructions from client-side
  request timing.
- Per-GPU power exists in every raw CSV but is **collapsed to a single
  node/TP-sum scalar at first parse** (`power_parsing.parse_power_csv` sums
  the first TP rows; utilization and memory columns are dropped). Nothing
  per-device survives into any downstream artifact.

### 1b. Stage 0 (shared root of both model streams)

```text
data/<layouts> --inventory.py regexes--> results/stage0/pair_manifest.csv (800 matched pairs, 29 configs)
                                         results/stage0/data_inventory.json
                                         model/throughput_database.json (29 configs; THE canonical path —
                                         the model/config/ variant exists only inside stale stored artifacts)
```

### 1c. The GRU stream

```text
pair_manifest.csv --manifest.py re-parses raw--> experimental_continuous_v1/
                                                   datasets/<cfg>.npz  (ragged: power, active_requests,
                                                   t_arrive_log [DEAD — stored, never loaded], dt=0.25)
                                                   splits/ (seed 42)   norm_params/
   --training.py--> continuous_v1_gmm_bigru/k10_f2/ (GMM k=10 + BiGRU on [A, dA]; checkpoints, gmms,
                                                   norm_params, run_manifest.json)
   --inference.py--> IID ONLY; A_t MODELED from throughput medians (arrival + n_in/lambda_pre + n_out/lambda_dec)
   --evaluation.py--> IID; recorded arrivals + modeled durations; no measured-power alignment
                     (the oracle offset is retained as a diagnostic and never applied)
```

Train/generate asymmetry, stated plainly: at training, A_t comes from
measured per-request timing; at generation and evaluation, A_t comes from
modeled durations using throughput medians. Evaluation additionally aligns
the modeled schedule against measured power (oracle). These are the exact
asymmetries EENERGY_PLAN W3 requires removing from the main comparison.

### 1d. The ledger stream

```text
pair_manifest.csv --build_ledger_cache.py--> ledger_cache.npz (273,544 one-second bins;
    17 work/identity arrays; arch from HARDCODED dict; its own 30-min timestamp fold)
  --fit scripts--> final_coefficients.json / map_coefficients.json  [OLD 14-term feature set]
  peak_and_holdout.py / scalability_moe.py                          [BETTER 11-term set + fp8 + cap;
                                                                     exports NOTHING]
CONSUMERS OF THE EXPORTED JSONS: none in the repo.
PURE, REUSABLE CORE: _bin_work_rates (build_ledger_bundle.py) — the arrivals->work engine.
```

Minimal input contract for a future arrivals-only generation mode (what the
work-rate formulas actually consume): per request `arrival_time, n_in, n_out,
ttft, decode_time`; per config `lambda_prefill, tp, hardware constants, arch
descriptor`. TTFT and decode time must be modeled to run from arrivals alone.

### 1e. The Azure stream

```text
raw week CSV -> split_azure_week_to_days -> parse_azure_trace -> azure_to_node_streams (partition, seed 42)
  -> azure_generate_traces  [resolves: run_manifest, checkpoint/norm/gmm, throughput DB,
                             experimental manifest, splitwise CSV; IID GMM sampling;
                             TP resolved = 8 from config_id]
  -> azure_aggregate        [node overhead added at rack; PUE applied at site only]
  -> azure_metrics          [TP resolved from config_id unless explicitly overridden]
  -> sizing table / figures [same canonical TP accessor]
```

### 1f. Metrics

A canonical library exists (`model/metrics.py`) and the historical GRU
evaluation uses it. Facility-side scripts partially reimplement: one
byte-identical KS clone; two ramp definitions (diff at native resolution vs
always-downsample-to-1s); NRMSE normalized per-trace vs over pooled min/max;
two load-duration-curve estimators (percentile vs exceedance-rank); CoV
implemented once, in a figure script.

## 2. Divergence registry (verified; fix or explicitly keep each)

| # | Divergence | Where | Class |
|---|---|---|---|
| D1 | Bundle format never produced; emitter/consumer key mismatch (`probe.params.rate` read from a path that never exists -> silently 0.0); flat vs 2-level layout; schema doc drift (8 named field mismatches) | probe_runner/run_manifest vs build_ledger_bundle:282; BUNDLE_SCHEMA.md | fix before first campaign |
| D2 | engine.csv write-only; `bins_from_engine_csv` = NotImplementedError | build_ledger_bundle.py:189-215 | gated Phase-2 item (unchanged) |
| D3 | Identity from names: hardware/model from dir name, TP/rate/date from filename, arch from hardcoded dict | inventory.py:8-16,96-111; build_ledger_cache.py:31-67 | unify via registry (Layer 4) |
| D4 | Per-GPU power collapsed at first parse; util/memory dropped | power_parsing.py:38-45,83-118 | run record keeps per-GPU |
| D5 | Three timestamp-alignment schemes: 30-min fold (ledger), rebase (alignment.py), modulo fold (request_timestamps.py) | build_ledger_cache.py:82; alignment.py:113; request_timestamps.py:96 | one function, one policy |
| D6 | Two request-JSON extractors with different validity rules; a third reader in request_builder ignores ttft/itls | power_parsing.py:131 vs throughput.py:145 vs request_builder.py:55 | one extractor, explicit filters |
| D7 | Timestamp-fabrication policy differs by caller: raise vs silently synthesize vs never synthesize | run_baselines_node.py:375; generate_power_cdf_comparison.py:362; appendix_surrogate_validity.py:267 | one builder + recorded `timestamp_source` |
| D8 | Train A_t measured-timing vs inference/eval A_t modeled-timing; eval oracle alignment | features.py:32 vs alignment.py:8; evaluation.py:188 | W3 contract (label or unify) |
| D9 | AR(1) fitted at eval time, stored under training dir, unusable by inference (IID-only, no flag) | evaluation.py:470-492; inference.py:26 | move fit to training or expose mode in inference |
| D10 | TP8/TP4: generation tp=8 from config_id; TDP/sizing default tp_gpus=4; pipeline `else 4` | azure_metrics.py:208,425; generate_azure_facility_sizing_table.py:235,362; run_azure_pipeline.py:124 | bug; single accessor + hand-worked test (W0) |
| D11 | Metric duplicates: KS clone; two ramp definitions; NRMSE pooled vs per-trace; two LDC estimators | generate_power_cdf_comparison.py:430; azure_metrics.py:108 vs run_baselines_facility.py:471; metrics.py:212 vs 276 | one metrics module, parameterized |
| D12 | Ledger exports old 14-term model; best 11-term (+fp8, +cap) never exported; zero consumers of either JSON | final_model.py:37 vs peak_and_holdout.py:32 | W2b consolidation target |
| D13 | Dead data: npz `t_arrive_log` + its norm stats stored but never loaded; stale `results/training/*` from deleted `model/train_entry` | data_loading.py:61; features.py:188 | drop at next dataset version; archive orphans |
| D14 | Cross-script private import: azure_generate_traces imports `_is_moe_config` from run_baselines_node | azure_generate_traces.py:61 | move to model/utils/config (canonical exists) |
| D15 | Stored eval artifacts predate current code (old throughput path, missing columns) | k10_f2/eval_metrics/* | regenerate under W0 provenance freeze |

## 3. The unification contract

### Layer 1 — one run record (ingestion)

One reader per raw layout, all producing the same normalized record. Home:
`model/training_data/` (already the shared parsing home — the ledger stream
imports `power_parsing` from it today).

```text
RunRecord
  identity:   config_id, model, hardware, tp, gpus_per_node,
              source_layout (sharegpt | extraneous | bundle), provenance
              (paths, hashes), clock_basis
  device:     per-GPU power table at native 250 ms (+ util/mem where present,
              clocks when bundles arrive); node/TP-sum provided as an
              accessor, never as the stored truth   [fixes D4]
  requests:   arrival_time, n_in, n_out, ttft, decode_time (one
              derive_decode_time), timestamp_source = recorded | synthesized
              [fixes D6, D7]
  arch:       from manifest when present, else from ONE shared registry that
              merges the hardcoded ARCH dict and arch_extract   [fixes D3]
  alignment:  exactly one alignment function with a stated policy  [fixes D5]
```

### Layer 2 — two feature views over the same record

- GRU view: A_t / dA_t from the requests table (existing
  `compute_active_requests`).
- Ledger view: per-bin work rates via the pure `_bin_work_rates` core.
- The future arrivals-only mode fills the same requests table from the
  surrogate instead of measurement — the W2c measured_timing / arrival_only
  split becomes "who filled the requests table," recorded in the manifest.

### Layer 3 — one metrics module

`model/metrics.py` becomes the only implementation. Add the
facility metrics (peak, ramps with an explicit `resolution_s` parameter
recorded in every output row, one LDC estimator, load factor, CoV). Delete
the KS clone; scripts import instead of reimplementing. Every consumer
(GRU eval, baselines, Azure metrics, tables, future Monte Carlo) reads from
here — this is the W3 "same metric code for both paths" requirement.

### Layer 4 — identity and settings accessors

`model/utils/config.py` (already canonical for parse_config_id /
is_moe_config / resolve_device) gains `tp_gpus_from_config_id`; every
`tp_gpus=4` default and the `else 4` fallback are replaced by it, with a
hand-worked TP8 nameplate test. [fixes D10, D14]

## 4. Implementation phases (each additive, each with an equivalence gate)

| Phase | Work | Gate |
|---|---|---|
| A | Metrics unification (Layer 3) | Regenerated metric CSVs byte-match current outputs where definitions are intentionally unchanged; intentional changes (ramp resolution, LDC estimator) listed with before/after values |
| B | Run record over the legacy layout (Layer 1) + both feature views (Layer 2) | Rebuilt npz datasets and ledger_cache match current artifacts exactly (or every diff explained); stage0 outputs unchanged |
| C | Identity/arch registry + TP accessor (Layer 4) | Hand-worked TP8 test; facility numbers regenerated and the corrected TDP column reported as a changed paper number |
| D | Bundle emitter/consumer reconciliation (D1) | Round-trip test: emit a synthetic bundle, ingest to RunRecord, field-for-field match; decide flat vs 2-level layout once |
| E | Physics artifact export (D12, = EENERGY_PLAN W2b) consuming the run record | The exported artifact reproduces the printed holdout numbers from the research scripts |

Test gates for every phase: `uv run -m pytest -x` and
`uv run -m pytest -x model/tests profiling feature-test`. Cleanup ordering
rules from cleaning-plan.md apply; nothing in the maintained surface is
deleted until its replacement's gate passes.

## 5. Decisions the team must make (blocking, in order)

1. Bundle layout: flat `data/runs/<run_id>/` (what code does) or
   `data/runs/<campaign>/<run_id>/` (what the doc and EENERGY_PLAN assume).
2. Ramp definition: keep both native-resolution and 1-s variants as
   explicit parameters, or standardize on one for the paper.
3. LDC estimator: percentile or exceedance-rank interpolation.
4. Dead `t_arrive_log` channel: drop at next dataset version or keep for a
   future feature set.
5. A_t asymmetry (D8): unify on modeled-timing everywhere for the main
   comparison, or keep measured-timing training with the asymmetry stated —
   this is a W3 contract decision, not an implementation detail.

## 6. Audited implementation record (2026-07-09)

The repair deliberately keeps two feature views, but gives them one ingestion,
identity, and failure contract:

1. Canonical bundles use `data/runs/<campaign_id>/<run_id>/`. Power rows carry
   GPU index and UUID; the reader validates the manifest topology, corrects the
   recorded local-wall-time offset, preserves every device/request/engine
   column, and rejects empty engine streams.
2. Legacy pairs and bundles both enter `RunRecord`. GRU and ledger projections
   are explicit; source paths, SHA-256 hashes, every list-valued request-column
   length, retained source row indices, and every alignment/filter drop are
   recorded.
3. GRU preparation requires three traces, disjoint train/validation/test
   indices, and raw per-trace timesteps within 1% of the per-config median.
   It projects every accepted trace onto that exact median grid; an individual
   power-log gap of at most four cadence intervals is linearly interpolated,
   while a longer discontinuity fails explicitly.
   Normalization, clamp bounds, and modeled-duration throughput are fitted on
   training traces only. Token/timing arrays remain in the dataset so the
   calibration is reproducible.
4. GRU training binds that throughput into the trained run manifest. Evaluation
   and standalone inference use the bound calibration, the same half-open
   request lifetime convention, and features at `t=dt` corresponding to
   `power[1:]`. Bundle test traces resolve request JSON through hash-checked
   lineage rather than the legacy pair manifest.
5. Checkpoint, trained normalization, GMM, dataset, split, and lineage identities
   are verified before use. Explicit standalone overrides must provide all three
   model files together. Stochastic inference requires an explicit seed.
6. Legacy and bundle ledgers share `model.training_data.ledger_view`; the final
   full bin is retained, failed parses do not consume quotas, unknown hardware
   fails, and no synthetic prefill-rate fallback exists. Ledger sidecars map
   each run to stable source IDs, paths, and hashes.
7. The physics exporter records its ledger and run-index hashes, dirty Git
   state, actual `dt_s`, source index, and per-model architecture descriptors.
   Deployment therefore supports full Hugging Face IDs without alias guessing.
   Lag parameters are converted by elapsed time when inference uses a different
   grid. Azure can run the physics method without loading GRU artifacts.
8. Result aggregation requires recorded generation mode and rejects invalid
   config IDs. Facility diversity is `sum(node IT peaks) / coincident site IT
   peak`; ramp resolution and the exceedance-rank LDC definition remain explicit.

The maintained semantic tests cover exact-boundary and off-grid timing,
work/token conservation, disjoint splits, train-only calibration, bundle
lineage, topology/clock/device preservation, artifact tampering, full-HF-ID
deployment, physics timestep conversion, generation-mode conservation, and
facility metric definitions.

The regenerated paper outputs under `results/eval_paper/` are intentionally
retained. Checked-in trained models, ledger caches, and physics JSON produced
before these contract changes remain **stale**. Rebuild stale dependencies in
this order:
Stage0/bundles -> experimental manifest -> GRU training -> evaluation, and
ledger + run index -> physics fit -> Azure/paper analysis. Current consumers
will reject many stale artifacts because their required identities or bound
calibration are absent.

One planned boundary remains intentionally unimplemented: `engine.csv` is
preserved and required, but measured-engine-state ledger reconstruction remains
`NotImplementedError` until it can be validated against live bundles. All
current ledger/physics claims therefore remain request-timing reconstructions.
