# PowerTrace-Sim

PowerTrace-Sim predicts LLM request timing and deterministic GPU power from a
request schedule. Its default is the selected architecture-aware scheduler,
250 ms work ledger, and clean v4 dense/support-bounded-MoE power model. The
superseded GMM-BiGRU system is being preserved as a runnable historical
artifact rather than used by the default pipeline.

## Planning Docs

- `DATA_INVENTORY_CAMPAIGN_PLAN.md`: executable final sealed campaign plus the
  evidence inventory and YAGNI rationale for arbitrary-arrival, real-agent, and
  unseen-model transfer.
- `FEATURE_TEST_PLAN.md`: exact baseline, feature-ablation, transfer-split,
  metric, and pass/fail specification for selecting the smallest
  architecture-aware node-power model.
- `FEATURE_TEST_LEARNINGS.md`: current model/evidence map, paper-safe findings,
  OpenHands hardware-calibration result, historical failure analysis, and the
  minimum measurement needed to continue.
- `profiling/MODEL_READINESS_RUNBOOK.md`: authoritative, unconfounded campaign
  order, launch prerequisites, acceptance criteria, and explicit post-collection
  blockers before model fitting or sealed scoring.
- `EENERGY_PLAN.md`: post-data-repair paper specification with claims, Monte
  Carlo/LDC semantics, token-aware model features, the exact profiling campaign,
  critical path, and parallel work lanes.
- `MOE_PIPELINE_PLAN.md`: frozen-dense, probabilistic MoE redesign using
  empirical route assignments, exact-backend inclusive timing, MoE ledger
  channels, one dynamic-energy surface, experiments, and transfer gates.
- `EENERGY_PLAN.html`: plain-language browser view of the paper plan with a
  review checklist and profiling schedule. Open it directly in any browser.
- `THEMES.md`: e-Energy paper framing, current conference themes, and result priorities.
- `power-test/power_pipeline_methods.tex`: paper-ready methods section for the
  request-timing, dense-power, and support-limited MoE-power pipeline, including
  the fitted objectives, routing assumptions, split policy, and metric equations.
- `cleaning-plan.md`: staged cleanup plan for trimming the repo around grid-facing evaluations.
- `data-path.md`: audited common ingestion, feature, metric, and identity contract.
- `docs/MODEL_PIPELINE.md`: paper-facing end-to-end training, inference, and
  evaluation flow for the selected model.

## Project Structure

```text
powertrace-sim/
├── archive/gmm_bigru_v1/ # Runnable historical GMM-BiGRU snapshot
├── docs/                 # Paper-facing model and experiment documentation
├── model/
│   ├── artifacts/        # Compact selected release artifact
│   ├── power/            # Dense and support-bounded MoE power equations
│   ├── timing/           # Scheduler, iteration timing, and 250 ms ledger
│   ├── training/         # Frozen-equation coefficient fitting
│   ├── scripts/          # prepare_data, train, evaluate, and infer CLIs
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

# Full research/development environment (default dev group included).
uv sync

# Minimal selected-model inference environment (NumPy only).
uv sync --no-dev

# Add fitting, paper, profiling, or historical-model support as needed.
uv sync --extra train
uv sync --extra paper
uv sync --extra profiling
uv sync --extra archive-bigru

# Optional: activate the venv directly
source .venv/bin/activate
```

## Quick Start

```bash
uv run -m model.scripts.infer \
    --requests examples/requests.json \
    --deployment llama-3-70b-a100-tp4 \
    --seed 42 \
    --out-dir outputs/example
```

The output directory contains `power.csv`, `requests.csv`, and
`manifest.json`. The release artifact is currently marked `pre_sealed`:
it is the selected model used by the repository's clean power-trace and
compatibility evaluations, but the final external validation campaign is not
complete. Native inference is fixed at the calibrated 250 ms cadence.
The CLI and Azure facility consumer project ledger and power bins through a
single-pass iterator. Their retained output memory is bounded by the short
meter-response window rather than growing with trace duration; the public
`simulate()` function still materializes arrays for analysis compatibility.

Requests may instead provide an explicit categorical
`output_tokens_distribution` with `values` and `probabilities`; `--seed`
controls only that workload realization. Optional `cached_prefix_tokens` is
subtracted from total `input_tokens` for executed prefill work while remaining
part of the initial KV context. Unsupported deployment overrides fail unless
`--allow-unsupported` is supplied, in which case every output is labeled as an
unsupported extrapolation.

See `CLEAN_MODEL.md` for the phased archive, training, evaluation, and
disaggregated-extension migration.

Regenerate the selected release without reopening model selection:

```bash
uv run --extra train -m model.scripts.prepare_data
uv run --extra train -m model.scripts.train \
    --prepared-manifest results/clean_model/prepared_dataset.json \
    --out-artifact results/clean_model/powertrace_v1.json
```

`prepare_data` validates and hash-binds the external 450-run timing and power
payloads; it does not copy their roughly 400 MB of arrays into Git. `train`
refits only the frozen timing calibration and clean v4 dense/bounded-MoE
equations. On the current 354,125-request dataset it regenerates every released
timing and power coefficient exactly.

Score two already aligned standard power CSVs with:

```bash
uv run -m model.scripts.evaluate \
    --measured measured_power.csv \
    --predicted outputs/example/power.csv \
    --out results/clean_model/evaluation.json
```

## Historical GMM-BiGRU workflow

The first PowerTrace-Sim model is available only through its self-contained
archive. Run these commands from the archive directory so Python cannot resolve
the maintained root package instead:

```bash
cd archive/gmm_bigru_v1
uv run --project ../.. --extra archive-bigru python \
    -m model.scripts.train_gmm_bigru --help
```

The canonical throughput database is `model/throughput_database.json` for Stage0
and ledger workflows. Trained GMM evaluation and inference use the throughput
bound in their run manifest and request JSONs resolved through hash-checked
lineage; their legacy `--throughput-db` and evaluation `--pair-manifest-csv`
flags are accepted for CLI compatibility but are not read. GMM input paths can
be repo-relative when loading manifests and request JSONs.
Add a repeatable `--bundle-dir data/runs/<campaign_id>/<run_id>` only when a
canonical bundle exists; it is not required for the legacy-data quick start.

The archive README documents its remaining preparation, inference, evaluation,
comparison, figure, trace-fidelity, and request-rate-sweep entry points. The
maintained `scripts/eval/` package no longer imports the archived learned model.
Historical per-configuration checkpoints and sweep outputs are also stored
under the archive's `results/` tree.

Profile same-node disaggregated GPT-OSS-20B serving on two Sherlock A100-80GB
GPUs with the pinned vLLM 0.22 Queue-Haul image using one TP1 prefiller and one
TP1 decoder:

```bash
# Ten-minute 2 requests/s integration gate.
POWERTRACE_DISAGG_MODE=smoke sbatch profiling/jobs/disaggregated_gpt_oss_20b.sbatch

# 0.25, 2, and 4 requests/s; three ten-minute repetitions per rate.
sbatch profiling/jobs/disaggregated_gpt_oss_20b.sbatch
```

The job requires the staged `openai/gpt-oss-20b` checkpoint and the same
ShareGPT file used by the legacy GPT-OSS campaign. An alternate
`POWERTRACE_DISAGG_IMAGE` must preserve the vLLM 0.22 NIXL request contract;
disaggregated serving is experimental and the script does not claim generic
compatibility with every vLLM 0.20+ release. Each measured request carries one
ID through the client, proxy, prefiller, and decoder, so the benchmark's extra
preflight request is excluded and concurrent stage timelines remain joinable.

Each checkpointed cell contains the detailed request result, ordered proxy
events, canonical extended 250 ms power telemetry for both GPU UUIDs,
role-separated `engine_prefill.csv` and `engine_decode.csv` streams, NIXL
transfer time/byte/failure counters, and capture start/end epochs. The run-level
`run_metadata.json` binds each GPU UUID to its prefill or decode role. A nominal
ten-minute cell means ten minutes of offered Poisson arrivals; capture continues
until every admitted request drains. Resubmitting with the same `RUN_ROOT` skips
only cells that passed the complete-request and stage-timeline gates.

These files deliberately remain role-aware. Do not label the two GPUs as GPT-OSS
TP2 when evaluating the existing model: each engine is TP1 and the GPUs perform
different phases. The current TP1 GPT-OSS timing calibration is the appropriate
starting point for each role, but a small role-aware ingestion/projection step is
still required before the four-file dense bundle pipeline can score the combined
deployment.

Run the frozen 250 ms conditional-timing feature ladder and transfer matrix with:

```bash
uv run python feature-test/evaluate_candidates.py \
  --ledger-cache feature-test/ledger_cache_250ms.npz \
  --run-index feature-test/ledger_cache_250ms.runs.json \
  --out-dir results/feature_test_v1
```

The evaluator emits per-run and aggregate energy, 60-second ACF, NRMSE, and
normalized soft-DTW divergence metrics plus split, complexity, support, gate,
and selected-model manifests. Soft-DTW is report-only and does not alter the
frozen selection or pass/fail gates.
Candidate and hyperparameter selection uses source development runs only;
target results only grade the already-frozen choice.

Run the request-only timing-to-power path with:

```bash
uv run python timing-test/simulated_ledger.py --dt 0.25 --roles all
uv run python power-test/join_power.py
uv run python power-test/evaluate_arrival_only.py
```

This reproduction path consumes the frozen fitted surface; it does not refit it
in place.

Generate the standalone paper timing-parity panels from the frozen timing
evaluation with:

```bash
uv run python power-test/plot_timing_parity.py
```

This writes `power-test/timing_prefill_parity.pdf` and
`power-test/timing_decode_parity.pdf`. Prefill uses both the calibration
observations and a deterministic model/hardware sample of individual training
requests whose measured lifetimes do not overlap another request. This retains
the intrinsic queue-free phase contract rather than raw loaded-run TTFT, which
also contains queueing and admission delay. Decode evaluates one repetition
from every frozen non-training cell, then deterministically samples each
model/hardware stratum. Both panels use log-scaled
observed-versus-predicted milliseconds, model color, A100/H100 marker shape,
and enlarged paper-scale typography with no title; only the decode panel
includes the combined legend so the PDFs can be placed in adjacent LaTeX
subfigures.

Generate four standalone 10-minute measured-versus-predicted power traces for
Llama-3-70B on A100 TP4 with:

```bash
uv run python power-test/plot_best_rate_traces.py
```

The model, hardware, and TP are fixed; the selector uses the metric-medoid
repetition under energy error, ACF R², and soft-DTW at 0.125, 0.5, 1, and 2
requests/s and writes one PDF per rate for a 2x2 LaTeX `figure*`. These are four
source rates with at least 600 seconds of aligned data; rate 4 remains a
retrospective stress condition. Measured per-GPU power is black and the
PowerTrace-Sim prediction is Stanford red; every panel is cropped to exactly
600 seconds, reduced to matched nonoverlapping one-second means, and displayed
with common axes. The generator consumes the corrected uniform-routing ledger
and `clean_power_surfaces.json`, not the legacy frozen dense artifact. Selection
evidence, per-panel metrics, and output paths are recorded in
`power-test/best_rate_traces_1s.json`; the exact plotted samples are in
`power-test/best_rate_traces_1s.csv`. The rate-1 and rate-2 panels retain the
observed late-trace pointwise alignment miss; low energy error must not be read
as good ACF-profile agreement.

To rebuild the source-only controlled-probe candidate and apply the
pre-registered source-development selection rule, run:

```bash
uv run python power-test/build_probe_power_calibration.py
uv run python power-test/fit_power_surface.py
```

The calibration builder admits only the Llama-70B A100 prefill and decode
staircases. It preserves raw 250 ms power targets, conserves bursty engine
counters over each request-active level, and assigns equal total regression
weight to every level after per-GPU scaling. Cached-context, mixed-grid, and
transient probes are excluded because their counters do not isolate an
instantaneous power component. The candidate replaces the baseline only on a
strict source-development Pareto improvement in energy error, ACF-MAE, ACF R²,
and range NRMSE; Qwen and every holdout remain unseen during that choice.
The current candidate is rejected: source dense development energy improves
from 10.08% to 9.35% and ACF-MAE from 0.01371 to 0.01354, while ACF R² falls
from 0.96687 to 0.96672, soft-DTW rises from 0.01894 to 0.01943, and range
NRMSE rises from 0.1362 to 0.1477. The generated surface therefore retains the
baseline and the fit report preserves the candidate as diagnostic evidence.

Audit the frozen dense surface over every one of the 450 legacy runs with:

```bash
uv run python power-test/plot_power_metric_audit.py
```

The audit covers all seven models, both hardware types, TP1--TP8, every rate
from 0.125 through 4 requests/s, and every split role. It writes six standalone
rate-sweep PDFs for energy error, soft-DTW, range NRMSE, per-GPU RMSE, ACF R²,
and KS agreement; metric/feature correlation heatmaps; a signed-bias cell
heatmap; per-run and per-cell CSVs; and `power-test/power_metric_audit.json`.
Every metric uses the same nonoverlapping one-second window. GPT-OSS cells are
shown only as dotted, explicitly unsupported dense-surface comparators rather
than being mixed with the separate MoE-v3 artifact.

The audit records why `fitted_surface.json` is a diagnostic legacy baseline.
The synchronized replacement contract, complete fit/evaluation population,
and support boundary are in `power-test/POWER_MODEL_REDESIGN.md` and
`power-test/power_pipeline_methods.tex`.

Fit and score the clean dense/MoE replacement candidate with:

```bash
uv run python timing-test/simulated_ledger.py \
  --dt 0.25 --roles all --moe-routing uniform \
  --out feature-test/ledger_cache_sim_uniform_current_250ms.npz
uv run python power-test/join_power.py \
  --cache feature-test/ledger_cache_sim_uniform_current_250ms.npz \
  --out power-test/sim_ledger_power_uniform_current_250ms.npz \
  --provenance-out power-test/sim_ledger_power_uniform_current_250ms.provenance.json
uv run python power-test/fit_clean_power_pipelines.py
uv run python power-test/plot_best_rate_traces.py
uv run python timing-test/evaluate_expansion.py \
  --timing-fit timing-test/fitted_efficiencies.json \
  --power-fit power-test/clean_power_surfaces.json \
  --out power-test/clean_expansion_report.json
```

The four standalone trace PDFs use Llama-3 8B on H100 TP1 at 0.125, 0.5,
1, and 2 requests/s. Each shows exactly ten minutes of matched one-second
means, with measured power in black and the clean prediction in Stanford red.
Both lines use a subtle left-to-right opacity gradient, and the shared legend
sits above the axes so it cannot obscure trace data.
TP1 is fixed because it has the best joint energy/ACF/Soft-DTW summary among
the measured Llama-3 8B H100 TP configurations; repetitions are chosen by the
three-metric medoid rather than minimum error.

The dense path is one four-coordinate law per hardware: a fixed hardware idle
floor, busy-gated local weight fraction, exact timing-roofline compute
utilization, and duty-aware square-root HBM utilization. The last coordinate is
`sqrt(busy * memory_util)`, which averages the instantaneous nonlinear response
instead of taking a square root after idle and active time have already been
mixed. GEMM FLOPs, causal-attention FLOPs, attention bytes, and their exact
prefill/decode decomposition are conserved from every simulated engine
iteration. The fitted law deliberately combines the phase work: a phase-split
ablation did not improve source, twin, and rate-4 trace-shape and
autocorrelation-profile metrics together.
Dense fitting uses matched nonoverlapping one-second means, gives equal total
mass to each run, and splits that mass equally between its ordinary and upper
power-decile seconds. It selects response delay under the same objective
(A100 0 s; H100 0.25 s). The law never uses model name, arrival rate, or trace
time. The MoE path remains separate, now uses exact timing work for its compute
coordinate, and retains per-architecture coefficients with explicit TP1/2
versus TP4/8 support. Neither path clips predictions without an explicit
run-level operating limit.

The legacy repetitions replay nearly identical workloads and therefore cannot
be split honestly between fitting and development. All rate-through-2 source
repetitions fit 225 dense and 60 MoE runs; all 57 rate-4 runs remain stress
tests, and 108 related-model twins remain transfer comparators. The collected
405B/GPT-OSS-120B/Qwen request traces are retrospective development evidence,
not part of this fit and not sealed transfer evidence.

Dense source medians are 1.68% energy error, 13.10 W/GPU RMSE, 0.9941 ACF R²,
and 0.548 KS agreement. Dense twin comparators score 1.97%, 12.30 W/GPU,
0.9952, and 0.623. Dense rate-4 stress scores 2.99% energy error,
14.09 W/GPU RMSE, and 0.730 median ACF R². MoE source/stress energy medians
are 1.27%/2.08%, with ACF R² 0.968/0.916. The fixed measured-trace horizon
retains idle bins when simulation drains early and truncates events beyond the
common evaluation window. The A100 Llama-70B rate-2 showcase still has
negative ACF R²: its measured power rises while exact total compute, memory,
raw GPU utilization, and memory allocation remain nearly fixed. Phase-split
regression improves some energy summaries but lowers rate-4 ACF, so it remains
a diagnostic channel rather than an extra fitted coefficient. New clock,
P-state, and power-limit telemetry is required to distinguish phase response
from an unobserved device-state transition. This candidate is not a sealed
transfer model until the external traces are scored.

The script writes `clean_power_surfaces.json`, `clean_power_report.json`,
per-run and per-cell CSVs, six standalone metric PDFs, and the signed-bias
heatmap under `power-test/`.

The alternate coverage-basis experiment trains on 117 traces from 39 complete
configuration/rate cells and holds out 333 traces as new rates, new TP setups,
or new model setups. All three repetitions of a cell stay together. Training
covers A100/H100, TP1/2/4/8, every measured request rate including rate 4,
dense/MoE, and BF16/MXFP4/FP8. It is an explicitly versioned development
experiment and does not replace the frozen default split:

```bash
uv run python timing-test/build_coverage_split.py
uv run python timing-test/fit_efficiencies.py \
  --split-manifest timing-test/coverage_split_manifest.json \
  --out timing-test/coverage_fitted_efficiencies_base.json
uv run python timing-test/fit_fp8_bandwidth.py \
  --manifest-in timing-test/coverage_split_manifest.json \
  --fitted-in timing-test/coverage_fitted_efficiencies_base.json \
  --fitted-out timing-test/coverage_fitted_efficiencies.json \
  --manifest-out timing-test/coverage_split_manifest_fp8.json
uv run python timing-test/evaluate_timing.py \
  --manifest coverage_split_manifest_fp8.json \
  --fitted timing-test/coverage_fitted_efficiencies.json \
  --out-dir results/timing_test_coverage
uv run python timing-test/simulated_ledger.py \
  --dt 0.25 --roles all \
  --manifest timing-test/coverage_split_manifest_fp8.json \
  --fitted timing-test/coverage_fitted_efficiencies.json \
  --out feature-test/ledger_cache_coverage_250ms.npz
uv run python power-test/join_power.py \
  --cache feature-test/ledger_cache_coverage_250ms.npz \
  --out power-test/sim_ledger_power_coverage_250ms.npz \
  --provenance-out power-test/sim_ledger_power_coverage_250ms.provenance.json
uv run python power-test/evaluate_coverage_split.py
```

On the 333 held-out traces, this fit obtains 3.26% median end-to-end timing
error, 1.97% energy error, 13.6 W/GPU power RMSE, and 0.989 temporal
similarity. The reproducible confidence intervals and simple paper table are
written to `power-test/coverage_power_report.json` and
`power-test/coverage_trace_fidelity_table.tex`.

Generate the model-wise held-out simulator fidelity table with:

```bash
uv run python power-test/coverage_model_fidelity_table.py
```

This writes `power-test/coverage_model_fidelity_table.tex`, `.csv`, and
`.json`. The table reports power error, distribution agreement, energy error,
and temporal error by model. Temporal error is `100 * sqrt(Soft-DTW)`, reported
as a percentage of the measured trace power range. Repeated runs are first
collapsed within each model-local hardware/TP/rate point, then each model's
median and 95% confidence interval are bootstrapped over those points with
1000 deterministic resamples.

Plot the 108 held-out-model traces from this split with:

```bash
uv run python power-test/plot_coverage_heldout_models.py
```

This writes a six-rate metric sweep and a measured-versus-predicted rate-4
trace panel to `power-test/coverage_heldout_models_rate_sweep.pdf` and
`power-test/coverage_heldout_models_rate4_traces.pdf`, with matching PNGs and
the representative-run selections in
`power-test/coverage_heldout_models_plots.json`. Every rate-4 cell is retained;
the trace panels select the central repetition across energy error, power RMSE,
and Soft-DTW trace-shape error rather than the best repetition.

The paper-style four-trace showcase selects one fixed held-out configuration
using median energy error, power RMSE, and Soft-DTW trace-shape error across all
four requested rates, then plots the central repetition at each rate:

```bash
uv run python power-test/plot_best_coverage_holdout_traces.py
```

It writes separate `power_trace_*_coverage_1s.pdf` and PNG files for 0.125, 1,
2, and 4 requests/s, using an 11x4 talk-context layout with the legend below
the axes and the prediction labeled "Our Simulator". The plotted one-second
values and selection provenance are in
`power-test/best_coverage_holdout_traces_1s.csv` and `.json`.

Soft-DTW is report-only. It is a symmetric divergence on nonoverlapping
one-second means, normalized once by the measured power range, with
`gamma=0.01` and a 10-second Sakoe-Chiba band. The CSVs also report the
zero-band divergence and the signed band effect. The zero-band value is
exactly squared range-NRMSE; the band effect can be negative after the two
entropic self-costs are subtracted, so it is not labeled a timing benefit and
does not select coefficients. External temporal metrics require at least 62
seconds, at least 95% observed one-second bins, and no telemetry gap longer
than two seconds. Energy remains reported when this temporal coverage gate
fails, together with the missing-bin and maximum-gap diagnostics.

The clean external development report exposes the remaining boundaries rather
than pooling them into a rate correction. The three A100 Qwen rate/shape runs
score 3.06--3.55% energy and 5.73--6.48% median E2E error, while the H100 TP4
same-marks control scores 5.27% energy, 2.15% E2E, and 0.943 ACF R². The two
current GPT-OSS-120B runs underpredict E2E by 30--33% and energy by about 24%; a
2x timing slowdown improves alignment but not energy, so timing and active
power both fail across the legacy-to-current campaign boundary. The current
405B runs underpredict E2E by 20--25% while overpredicting energy by 10--14%;
their 17--20% missing one-second power coverage makes ACF and Soft-DTW
unscorable. BurstGPT and both TraceLab legs still fail at least one timing or
power gate, and the TraceLab cache pair remains identity-invalid. Gemma MoE
fails closed because no architecture-specific MoE surface exists. These
results rule out a universal rate, elapsed-time, or MoE multiplier.

### Known flawed development campaigns

Several collected bundles are intentionally retained but must not be described
as clean transfer evidence:

- Legacy ShareGPT runs have deterministic request ordering, protocol-locked
  midpoint residual changes, and no scheduler-policy, clock, P-state,
  temperature, or power-limit identity. They support within-protocol fitting,
  not an elapsed-time or generic rate effect.
- Current GPT-OSS-120B hard cells cross from legacy asynchronous serving to a
  synchronous current stack, use different ShareGPT marks, and lack a measured
  120B routing law. The rate-1 power log also has a seven-second gap, so its
  temporal metrics are rejected.
- Current H100 405B hard cells observe only 36-38% of native power bins and miss
  16.7-20.4% of one-second bins. Their timing and cadence-qualified mean energy
  remain diagnostic; ACF, NRMSE, and Soft-DTW are invalid.
- The TraceLab cache pair has all 136 keyed rows but differs in 10 prompt hashes
  and 34 output hashes. Its legs may be scored separately but never subtracted
  as a cache treatment.
- The 52/56 second Qwen rate-4 runs fail the 62-second temporal minimum. The
  80-83 second A100 rate/shape runs are technically eligible but leave only
  20-23 pairs at lag 60, so their long-lag metrics remain fragile development
  diagnostics.
- Two batch-256 calibration levels contain 58 and 66 preemptions, and an H100
  mixed grid contains further preempted windows. Only zero-preemption levels
  may calibrate the non-preemptive simulator.
- Probe directories without a canonical `manifest.json` and MoE routing rows
  without persisted source labels are quarantined from fitted or transfer
  claims that require those identities.

Unsupported Gemma MoE power is a model-support boundary, not a defective
bundle. The exact allowed use and minimum useful recollection for every item
above are recorded in `DATA_INVENTORY_CAMPAIGN_PLAN.md`, Section 2.4.

Normalize the captured GPT-OSS expert IDs and run the routing-aware
counterfactual without changing the default uniform-routing artifacts with:

```bash
uv run python profiling/moe_routing/build_routing_law.py
uv run python timing-test/evaluate_timing.py \
  --manifest split_manifest_fp8.json \
  --moe-routing measured \
  --out-dir results/timing_test_routing_v1
uv run python timing-test/simulated_ledger.py \
  --dt 0.25 --roles all \
  --manifest timing-test/split_manifest_fp8.json \
  --moe-routing measured \
  --out feature-test/ledger_cache_sim_routing_250ms.npz
uv run python power-test/join_power.py \
  --cache feature-test/ledger_cache_sim_routing_250ms.npz \
  --out power-test/sim_ledger_power_routing_250ms.npz \
  --provenance-out power-test/sim_ledger_power_routing_250ms.provenance.json
uv run python power-test/evaluate_arrival_only.py \
  --cache power-test/sim_ledger_power_routing_250ms.npz \
  --out-json power-test/moe_routing_report.json \
  --out-csv power-test/moe_routing_per_run.csv
```

The current routing law uses per-layer expert-touch probabilities and separate
prefill/decode correlation exponents, then unions both phases to estimate a
logical distinct-expert footprint. That footprint is a work proxy with no
ordering guarantee relative to physical HBM traffic. The law is bound to
the ShareGPT half of the capture; the current artifacts infer the documented
64-ShareGPT/64-SWE-smith ordering because the v2 capture did not persist source
labels. The reported entropy is selected-expert load entropy, not router-logit
or gate-confidence entropy. `MOE_PIPELINE_PLAN.md` replaces this compressed law
for future timing work with empirical per-layer expert-count distributions and
an expectation over an exact-runtime inclusive MoE timing lookup.

The like-for-like 2026-07-18 ablation rejects measured routing as a power-model
improvement. Against a uniform cache rebuilt from the same code, measured
routing worsens GPT-OSS-20B development energy/ACF-MAE/ACF-R2/range-NRMSE from
15.52%/0.0130/0.950/0.140 to 17.75%/0.0141/0.942/0.151, and its rate-4 holdout
from 17.23%/0.0274/0.913/0.233 to 22.41%/0.0439/0.756/0.303. GPT-OSS-120B also
worsens from 3.92%/0.0164/0.959/0.084 to
4.89%/0.0169/0.934/0.092. ACF-R2 is a goodness metric and should increase;
ACF-MAE is the corresponding error minimized here. Scheduling evidence is
mixed: 120B median decode error improves from 12.8% to 10.8% at TP4 and
6.6% to 6.2% at TP8, while 20B timing regresses. Measured routing therefore
remains an honest work-coordinate ablation, not the deployed default; a direct
entropy-to-watts term is not identified by these captures.

Fit and score the separate, support-limited GPT-OSS-20B MoE power surface with:

```bash
uv run python power-test/moe_surface.py fit
uv run python power-test/moe_surface.py score
```

This never modifies `power-test/fitted_surface.json`. It fits five nonnegative
node-power terms—TP floor, TP link floor, logical memory utilization, engine
iteration rate, and log batch—on the 20 GPT-OSS-20B A100 TP1/2 training runs
with equal weight per run. Logical memory is aligned one 250 ms ledger bin
later within each run, using a first-value hold at each run boundary; the
iteration and batch coordinates remain instantaneous, and there is no global
A100 meter delay. These are offline-ledger coordinates: online use still
requires the scheduler's output-length estimate. The surface applies only to
GPT-OSS-20B A100 TP1/2 under the current request-only timing cache with
independent-uniform top-k routing.
Measured-routing caches fail closed. GPT-OSS-120B and unsupported TP cells are
reported only as labeled retrospective baseline comparators; dense evaluation
remains on its separate frozen path.

On the ten in-domain development runs, median energy error falls from 15.52%
to 0.92%, ACF R2 rises from 0.950 to 0.961, and range NRMSE falls from 0.140
to 0.065. On the six retrospective rate-4 stress runs, energy error falls
from 17.23% to 1.16%, ACF R2 rises from 0.913 to 0.928, and NRMSE falls from
0.233 to 0.052. These rate-4 targets were inspected during model iteration
and are not a sealed holdout claim. For context, the frozen A100 dense-only
development median is 3.36% energy error, 0.974 ACF R2, and 0.089 NRMSE: the
MoE surface is better on energy and NRMSE and within 0.013 ACF R2. GPT-OSS-120B
retains the already strong
frozen comparator result: 3.92% energy error, 0.959 ACF R2, and 0.084 NRMSE.
Artifacts bind the exact training/evaluation design and target arrays, split,
joined-cache provenance, and model/evaluator code hashes. Complete results are
`power-test/fitted_moe_surface_v3.json`,
`power-test/moe_surface_selection_v3.json`,
`power-test/moe_surface_dev_v3.json`, and
`power-test/moe_surface_stress_v3.json`.

Plot matched rate-4 GPT-OSS traces against ground truth with and without the
measured-routing timing path using:

```bash
uv run python power-test/plot_moe_timing_comparison.py
```

This writes `power-test/gpt_oss_timing_comparison.png`. Every panel uses the
same run ID, joined measured-power samples, frozen dense-trained power surface,
and 5-second display smoothing. Routing-aware timing and its derived logical
work/weight channels change together.

Plot the best validated power path for GPT-OSS-20B and GPT-OSS-120B with:

```bash
uv run python power-test/plot_best_moe_curves.py
```

This writes `power-test/gpt_oss_best_power_curves.png` and a selection/metric
sidecar at `power-test/gpt_oss_best_power_curves.json`. GPT-OSS-20B uses the
MoE v3 surface; GPT-OSS-120B uses the frozen dense comparator because the MoE
surface does not support TP4/8. Each panel shows the metric-medoid rate-4
repetition for its TP cell rather than the lowest-error trace.

Build and dry-run the exact arbitrary-arrival/agent-session profiling path with:

```bash
uv run python profiling/agentic_traces/build_trace_plan.py \
  syfi_coding_trace.jsonl.gz data/trace_plans/tracelab_code.json \
  --format tracelab-jsonl --revision v0.0.1 \
  --max-sessions 8 --min-rounds-per-session 12 \
  --max-rounds-per-session 32
bash profiling/jobs/run_campaign.sh \
  profiling/campaigns/trace_replay_qwen3-8b_a100_cache_off.json
bash profiling/jobs/run_campaign.sh \
  profiling/campaigns/trace_replay_qwen3-8b_a100_cache_on.json
```

The plan represents open-loop releases and closed-loop tool waits. New cache
replays use deterministic direct prompt IDs plus one seeded,
singleton-allowed output token per turn. The runner verifies every returned
token before it can enter later context, and the pair comparator joins rows by
`(session_id, turn_idx)` rather than completion order. The two Qwen cache
campaigns also record a 60-second pre-idle window with P-state, clock-event,
and power-limit telemetry. This protocol preserves exact dense-model prefix
identity and fails on incomplete token accounting. It is not a content-neutral
MoE cache treatment because forced token identities may alter expert routing.
The collected TP4 state control is not rerun. The pending
`profiling/campaigns/h100_tp8_state_diagnostic.json` job replays its recorded
release/input-length/output-budget marks from the hashed deterministic plan
`data/trace_plans/h100_tp4_state_marks.json`, records 180 seconds of idle, and
persists server launch/ready epochs. Prompt-content identity is unavailable for
the historical TP4 bundle and is not claimed. MoE expert IDs are captured with
`profiling/moe_routing/router_capture.py`. See
`profiling/jobs/README.md` and `profiling/BUNDLE_SCHEMA.md` for the exact inputs.

The paper-final sealed path is defined by four configs:
`sealed_burstgpt_qwen3-8b_a100.json`,
`sealed_openhands_qwen3-8b_a100.json`,
`sealed_qwen3-14b_a100.json`, and
`sealed_qwen3-30b-a3b_h100.json`. The BurstGPT builder can select three
disjoint fixed 900-second Fano strata with `--window-index 0/1/2
--window-count 3`. The OpenHands adapter reads the pinned evaluation JSONL,
preserves real event text and observed action-to-observation gaps, and uses
three disjoint hash packs. Its cache pairs apply the same deterministic
singleton-token protocol as direct trace replay. The pinned vLLM 0.10.1.1
image returns exact IDs through its logprob token-ID transport. OpenHands tool
actions and observations are paired by the observation's explicit `cause`
event ID; both records use `source=agent` in the pinned evaluation output.
Observation records are never replayed as assistant turns, and user follow-up
latency is not counted as tool execution. Before `sbatch`, the submit wrapper
builds all three packs offline and rejects any plan-hash drift or declared wait
limit violation. The sealed hashes bind the selected sessions, text, tokens,
turns, and waits. The corrected packs contain 89/119/102 turns,
1,171,553/1,398,210/1,139,574 cumulative prompt-context tokens, and maximum
per-session tool waits of 15.635/10.054/11.493 seconds. Their maximum individual
tool waits are 12.667/4.887/1.905 seconds. The campaign retains a 48-hour Slurm
limit as an allocation ceiling, not an expected replay horizon.
Cache-on servers also enable vLLM prompt-token details. That vLLM release omits
the per-request field when the cached count is zero, so replay records the
omission as zero and rejects a completed cache-on run unless it contains
positive server-reported cache evidence.
Every server launch now sets prefix caching explicitly: cache-on uses
`--enable-prefix-caching`, while cache-off uses
`--no-enable-prefix-caching` because vLLM V1 enables it by default. Agentic and
direct-replay bundles validate the raw engine cache-hit counter against that
declared treatment before writing a manifest, and sealed scoring repeats the
same check.

Stage the two unseen checkpoints and the pinned OpenHands JSONL before
submission. GPU jobs are offline and the submit wrapper rejects any missing
asset:

```bash
bash profiling/jobs/stage_models.sh Qwen/Qwen3-14B Qwen/Qwen3-30B-A3B
bash profiling/jobs/stage_openhands.sh \
  aa8977805b4cefd317001d80ddf1ad52790e9d23
```

After freezing fits and collecting into a separate sealed root, score all
bundles exactly once:

```bash
uv run python power-test/score_sealed_campaign.py \
  --timing-fit <frozen-timing-fit.json> \
  --power-fit <frozen-power-fit.json> \
  --bundle-dir <sealed-run> \
  --out <new-sealed-report.json>
```

Repeat `--bundle-dir` for every run. The scorer requires top-level
`validation_role=sealed`, validated `measured_ledger` telemetry, unique run
IDs, an unused output path, and exact cache-pair identity when both legs are
present. See `DATA_INVENTORY_CAMPAIGN_PLAN.md` for the fixed gates and complete
launch order.

Audit the collected Qwen3-30B-A3B H100 TP2 MoE transfer cell with:

```bash
uv run python power-test/analyze_qwen3_30b_moe_transfer.py
```

The audit applies both frozen GPT-OSS MoE laws zero-shot, repeats them with
only the pre-declared 60-second target idle anchor updated, evaluates a
source-only H100 energy-per-work adaptation, and includes the dense H100 law as
an explicitly unsupported architecture diagnostic. All candidates use the
same request-derived timing ledger, H100 meter response, observed interval,
and one-second plotting grid. It writes the six-panel comparison
`power-test/qwen3_30b_h100_moe_transfer.png`, exact plotted samples in
`power-test/qwen3_30b_h100_moe_transfer_1s.csv`, and timing, engine-counter,
architecture, metric, and failure diagnostics in
`power-test/qwen3_30b_h100_moe_transfer.json`.

Generate matched paper-style checkpoint-transfer panels and their LaTeX
captions with:

```bash
uv run python power-test/plot_transfer_traces.py
```

This writes standalone dense Qwen3-14B A100 TP1 and MoE Qwen3-30B-A3B H100
TP2 PDFs/PNGs using the same black-measured, Stanford-red-predicted,
one-second per-GPU style as the `power_trace_*` figures. The dense panel is a
retrospective idle-calibrated transfer: it replaces only the frozen A100 idle
floor with Qwen3-14B's declared 60-second pre-request idle measurement and
retains the pure zero-shot score in the report. The MoE panel is also
explicitly retrospective: it applies the target's pre-request idle anchor and
source-only A100/H100 energy-per-work ratios to the frozen GPT-OSS-20B law.
Neither displayed prediction consumes loaded-target power or measured engine
counters. Exact traces and metrics are recorded in
`power-test/transfer_traces_1s.csv` and
`power-test/transfer_trace_report.json`; a ready-to-include two-panel LaTeX
figure is in `power-test/transfer_trace_captions.tex`.

Run the unsupported dense-to-MoE response-law counterfactual for Qwen3-14B
with:

```bash
uv run python power-test/analyze_qwen3_14b_moe_counterfactual.py
```

The diagnostic applies both frozen GPT-OSS MoE laws, with and without the
Qwen3-14B pre-request idle anchor, to the unchanged request-generated dense
ledger. It also includes the dense zero-shot and idle-calibrated comparators on
the same one-second interval. Every MoE result is explicitly marked unsupported
because the target checkpoint is dense. The six-panel comparison, exact plotted
traces, and metric/support report are written to
`power-test/qwen3_14b_moe_counterfactual.{pdf,png}`,
`power-test/qwen3_14b_moe_counterfactual_1s.csv`, and
`power-test/qwen3_14b_moe_counterfactual.json`.

Generate the three idle-calibrated BurstGPT arbitrary-arrival transfer panels
and paper-ready captions with:

```bash
uv run python power-test/plot_burstgpt_transfer.py
```

Each standalone `power_trace_burstgpt_*` PDF/PNG uses the exact replay marks
from one disjoint sealed Fano stratum and changes only that run's idle intercept
to its declared 60-second pre-request measurement. Timing and dynamic-power
coefficients remain frozen, and neither measured engine state nor loaded-run
power enters prediction. The report preserves the corresponding zero-shot
metrics and marks the idle result retrospective. Exact samples, metrics, and a
three-panel LaTeX figure are written to
`power-test/burstgpt_idle_transfer_1s.csv`,
`power-test/burstgpt_idle_transfer_report.json`, and
`power-test/burstgpt_idle_transfer_captions.tex`.

Retrospective analysis of the six OpenHands runs identifies a deployment-level
transfer boundary that the nominal `A100` label does not encode. The source
surface came from a 400 W Azure eight-GPU A100 platform; OpenHands used one GPU
on a different four-GPU A100 platform. Keeping the request/timing simulation
frozen and applying only a target idle anchor, a 1.0425 ordinary-dynamic gain,
and a 1.6316 total prefill-compute gain reduces median energy error from 12.65%
to 1.34%, normalized Soft-DTW from 0.0270 to 0.0049, and range NRMSE from 0.160
to 0.075. Leave-one-pack-out gains and scores remain stable. This supports
few-shot hardware calibration with zero-shot transfer of unseen agentic
workload structure; because the constants were extracted after inspecting the
OpenHands targets, it is retrospective evidence rather than a sealed claim.
The calibrated overlay is
`power-test/openhands_platform_calibrated_prediction_overlay.png`, and the
full finding, evidence map, and calibration equation are in
`FEATURE_TEST_LEARNINGS.md` Sections 15--16.

ACF R2 is reported separately as autocorrelation-profile agreement. It is not
a temporal-fidelity or event-alignment measure: Soft-DTW describes time-warped
shape agreement, range NRMSE describes pointwise magnitude agreement, and
energy error describes integrated magnitude.

The minimal expansion also contains independent same-marks jobs for Qwen
A100/H100 transfer, an off-grid rate of 2.5 requests/s, controlled Gamma arrival
shapes 0.25/1/4, exact BurstGPT arrivals, and Gemma-4-26B-A4B cross-family MoE
transfer. After staging every model, container, local dataset, and trace plan,
submit the jobs in parallel with:

```bash
H100_PARTITION=<partition> \
  bash profiling/jobs/submit_expansion_jobs.sh
```

The submission layer is offline and fail-fast: it rejects incomplete model
caches and missing local inputs before requesting GPUs.

Evaluate the source-only component-accounting revision and the collected Qwen
development bundles without replacing frozen artifacts:

```bash
uv run python timing-test/fit_efficiencies.py \
  --out /tmp/powertrace_fitted_efficiencies_v3_base.json
uv run python timing-test/fit_fp8_bandwidth.py \
  --fitted-in /tmp/powertrace_fitted_efficiencies_v3_base.json \
  --fitted-out /tmp/powertrace_fitted_efficiencies_v3.json \
  --manifest-out /tmp/powertrace_split_manifest_fp8_v3.json
uv run python timing-test/simulated_ledger.py \
  --dt 0.25 --roles all \
  --manifest /tmp/powertrace_split_manifest_fp8_v3.json \
  --fitted /tmp/powertrace_fitted_efficiencies_v3.json \
  --out /tmp/powertrace_ledger_sim_v3.npz
uv run python power-test/join_power.py \
  --cache /tmp/powertrace_ledger_sim_v3.npz \
  --out /tmp/powertrace_ledger_power_v3.npz \
  --provenance-out /tmp/powertrace_ledger_power_v3.provenance.json
uv run python power-test/fit_power_surface.py \
  --cache /tmp/powertrace_ledger_power_v3.npz \
  --surface-out /tmp/powertrace_fitted_surface_v3.json \
  --report-out /tmp/powertrace_power_fit_v3.json
uv run python timing-test/evaluate_expansion.py \
  --timing-fit /tmp/powertrace_fitted_efficiencies_v3.json \
  --power-fit /tmp/powertrace_fitted_surface_v3.json \
  --out /tmp/powertrace_expansion_v3.json
```

The revision treats input embeddings as row gathers, output heads as explicit
BF16 projections, quantized transformer work as a separate sequential roofline,
and concurrent prefill requests as independent attention chunks. Attention,
KV traffic, embedding rows, and the output head do not inherit the FP8
transformer-stream calibration. FP8 evaluation fails when its hardware
calibration is absent. Partial-quantization campaigns bind the actual checkpoint
footprint, embedding storage precision, and quantized FLOP fraction; the 405B
campaign no longer infers a uniform one-byte-per-parameter footprint.
Bundle evaluation binds `max_num_batched_tokens` and `max_num_seqs`; cached
prefixes consume context/KV capacity but not executed prefill. The deployable
power surface fixes a per-hardware loaded-idle floor from the settled portion
of sustained source-run idle gaps, gates checkpoint footprint by busy duty,
and fits nonnegative dynamic work terms at the reported one-second timescale.
A deployment-specific measured idle delta can be applied per GPU before an
explicitly recorded cap. Engine scheduling policy is now
launched and recorded explicitly. Existing idle data support checkpoint
transfer only under the observed allocator/engine/power state; unseen scheduler
policy, P-state, clocks, or cap remain unsupported rather than pooled.

The corrected calibration rejects two batch-256 probe levels that recorded 58
and 66 preemptions because the deployed simulator is non-preemptive. This
improves median cell E2E error over the 150-cell non-training matrix from 4.13%
to 3.63% and p90 from 10.84% to 10.14%. BurstGPT improves from 3.84% to 1.93%
and TraceLab cache-off from 7.57% to 3.08%; legacy GPT-OSS-120B model-transfer
cells regress from 9.41% to 10.22% median, so the change is not presented as a
uniform MoE timing gain. H100 405B FP8 median E2E error is
7.24% across rates: 2.89-8.07% through rate 2 and 12.41% at rate 4. The
remaining high-rate miss is concentrated in mixed prefill/decode token-latency
tails; the recorded 405B validation cells show no waiting or preemption. The
leading hypotheses are a missing phase-specific weight sweep or an
operator-efficiency miss in the mixed FP8 path. The only FP8 calibration
checkpoint is 405B, so this is rate/repeat transfer under one FP8 recipe, not
unseen-FP8-checkpoint validation.

Reproduce the large-checkpoint concurrency and arrival-conditioned ITL
diagnostic with:

```bash
uv run python timing-test/rate4_diagnostic.py \
  --fitted /tmp/powertrace_fitted_efficiencies_v3.json
```

The JSON report records dataset/model hashes and labels the independently
classified measured/predicted arrival windows as retrospective association
evidence, not same-iteration admission evidence.

The three matched A100 Qwen rate/shape runs score 5.69-5.95% timing and
1.64-1.82% energy error. The A100/H100 rate-4 Qwen runs score 6.37%/4.97%
timing and 0.56%/14.22% energy. TraceLab cache-off scores 3.45% timing but
under-predicts energy by 25.75%; cache-on scores 15.55% timing and
under-predicts energy by 2.27%. BurstGPT scores 2.06% timing and 7.89% energy.
The source anchor improves sparse A100 traces but raises A100 legacy low-rate
bias and worsens H100 Qwen energy, demonstrating that engine/power state cannot
be pooled by hardware alone. The v3 surface is therefore a development
candidate and does not replace the frozen artifact.

The cache pair is not a valid treatment comparison: all 136 keyed rows match,
but 10 prompt hashes and 34 output hashes differ. The existing bundles predate
the forced-decode protocol and
cannot be repaired retrospectively; they remain valid only as separate
per-regime agentic model-error reports. All scores are retrospective
development results, not sealed transfer claims.

The baseline power fit uses dense training bins and loads exact simulated
iteration counts, tokens/iteration, prefill/decode duty, and weight-memory
traffic. Each simulated iteration contributes exactly the weight bytes charged
by the timing model; mixed prefill/decode iterations are not counted twice. An
A100 controlled-staircase candidate can be fit jointly, but is deployed only
when it Pareto-improves all source-development objectives under the rule above;
the report preserves both source scores and the rejected candidate fit.
The frozen arrival-only evaluation scores 290 non-training runs without
refitting. Dense energy transfer is strong (A100/H100 held-rate medians
2.85%/1.68%; H100 405B 3.49%), but rate-4 70B autocorrelation-profile agreement
does not pass (ACF-MAE 0.30-0.37). The deployed dense surface remains frozen;
the completed
MoE routing counterfactual is report-only, and sealed MoE validation remains
gated on M2/M3 measurements.
Exact results and contextual M4A/B2 comparisons are in
`power-test/arrival_only_report.json`; see `PIPELINE_PLAN.md` and `TODO.md`
for the stopping condition.
The checked-in power-fit and arrival-only report artifacts predate the exact
weight-traffic correction and are legacy references. Any corrected-ledger
refit is a separately versioned dense-baseline change outside the MoE plan.

Run the training-only thermal-state diagnostic with:

```bash
uv run python power-test/thermal_ablation.py
```

It adds a causal first-order heat coordinate driven by predicted dynamic board
power, resets it at each run boundary, and selects its time constant without
using holdout power. This is an ablation, not part of the deployed surface.
The generic term improves rate-4 70B TP8 median ACF-MAE from 0.314 to
0.114 on A100 and 0.314 to 0.260 on H100, but selects short 30/60-second
constants and worsens the matched TP4 ACF controls. A TP8-only chassis term
selects the 960-second grid boundary and makes no meaningful H100 improvement.
The thermal hypothesis therefore remains unidentifiable from the current
training split; no fitted artifact or arrival-only result is changed.

Run the retrospective changepoint and held-out step diagnostic with:

```bash
uv run python power-test/changepoint_ablation.py
```

It refits the existing surface in memory on dense training bins, inventories
the best two-mean residual split in all 450 runs, and validates
`baseline + before + jump * 1[t >= split]` without reading each test trace.
All 12 rate-4 70B TP8 repetitions change at 304.2-306.5 seconds by
12.35-19.04 W/GPU. On H100, leave-one-model-out ACF-MAE improves from
0.314 to 0.057, energy error from 3.41% to 0.34%, and range NRMSE from
0.304 to 0.187. A100 model transfer improves ACF-MAE from 0.321 to 0.206
but worsens energy and NRMSE. A rate-2 original-training extrapolation also
predicts the H100 rate-4 ACF improvement (0.314 to 0.053), while an ungated
step worsens the other 132 dense TP8 runs. The step is real and predictable
inside the affected cell, but its trigger is not general enough to ship; no
deployment artifact is changed.

Run the rate-2 work-dose extrapolation with:

```bash
uv run python power-test/dose_ablation.py
```

This ablation integrates predicted dynamic node power above the fitted idle and
fabric floors, selects a cooling constant and threshold using only original
rate-2 training traces, scales the learned jump linearly with arrival rate, and
scores an energy-centered correction on frozen rate-4 70B TP8 traces. The
training evidence cannot identify cooling: every A100 constant ties, while
H100 constants from 960 seconds through the cumulative limit tie, so the
simpler cumulative limit is selected. It predicts the A100 transition at
300.5 seconds and improves ACF-MAE from 0.3143 to 0.2501, but predicts the H100
transition at 283.5-284.0 seconds rather than the observed approximately
305 seconds. H100 ACF-MAE improves from 0.3144 to 0.0555, essentially matching
but not beating the fixed rate-2 timing extrapolation at 0.0532. Accumulated
work therefore describes a useful temporal correction but is not identified as
the trigger mechanism and is not added to the deployed surface. The command
also writes `power-test/dose_ablation_heldout.png`, comparing measured held-out
Llama-3-70B TP8 rate-4 traces with predictions made with and without the dose
term.

Run the no-dose H100 TP8 timing diagnostic with:

```bash
uv run python power-test/timing_residual_diagnostic.py
```

It writes `power-test/h100_tp8_timing_diagnostic.png` for held-out
Llama-3-70B TP8 at rate 4. The first panel subtracts the same causal 60-second
trend from measured and predicted power. The remaining panels compare measured
and simulated running/waiting requests and request completions, then show the
simulation-only engine iteration rate and decode batch. This is a
diagnostic of the frozen no-dose model; it does not fit a correction.

Run the one-feature decode-batch power ablation with:

```bash
uv run python power-test/batch_ablation.py
```

It jointly refits the existing surface plus
`TP * log1p(time-averaged decode batch)` on finite dense training bins, scores
the frozen holdouts, and writes `power-test/batch_ablation_heldout.png`.
The feature is rejected. On H100 rate-4 70B TP8, ACF-MAE improves only from
0.3144 to 0.2928 while energy error worsens from 3.41% to 4.87% and range
NRMSE from 0.304 to 0.386. A100 energy and NRMSE also worsen. Within the H100
held-out trace, matched decode-batch ranges consume 12-15 W/GPU more after the
305-second transition than before it, so no static batch-to-power mapping can
represent the observed two-regime relationship. The deployed surface remains
unchanged.

The rebuilt ledger contains exact offered request marks, `A_t`, and
running/waiting state. Transfer results retain the `conditional-timing
transfer` label because request execution timing is measured; the separate
arrival-only scheduler gate has not passed.

After reconstructing post-first decode completions from measured per-request
ITLs, source-only selection chooses the two-timescale active-request response
(`M4A`) on both A100 and H100. The selected scorecard does not pass: failures
remain in H100 S0, A100 gpt-oss scale transfer, H100 405B transfer, A100 TP2/4,
and H100 TP8. Exact metrics are in `feature-test/README.md`; the scorecard is
retrospective development evidence, not sealed validation.

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
Diagnostic bundle evaluation can retain missing meter bins with
`keep_power_gaps=True`: work and causal state remain on the complete uniform
time axis while `power_valid` masks only unavailable targets. This never
interpolates power or compresses causal time across a logger gap; production
cache construction remains fail-fast by default.

New profiling bundles can use the measured hybrid ledger explicitly:

```bash
uv run python feature-test/build_ledger_bundle.py \
  --runs-glob 'data/runs/<campaign>/*' \
  --state-source measured_engine \
  --lambda-prefill 7421 \
  --lambda-prefill-source '<prefill staircase bundle ID>'
```

This path uses stock vLLM counters for executed prompt/decode tokens and total
running/waiting requests, while request TTFT/ITL remains the source of
phase-specific batch and KV geometry. Iteration-rate, tokens-per-iteration, and
GPU-cache-usage diagnostics are persisted in the cache. Stock vLLM does not
provide collective duration, NVLink traffic, router choices, or expert touches;
the campaign identifies those effects only through matched TP/model probes.
Every live run preflights and validates the required metrics, cadence, topology,
counter monotonicity, and cross-stream epoch alignment. Smaller TP-pair legs are
pinned to explicit GPU UUIDs; ingestion checks those UUIDs against the exact
per-GPU columns summed into the TP-group power target.

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
Retrospective reconstruction and arrivals-only physics inference share the same
half-open-bin schedule arithmetic. Retrospective decode completions use the
recorded ITL sequence: the first output is produced at TTFT, and each later
completion is placed at the cumulative ITL time. Rows whose stream-chunk count
does not equal `output_tokens - 1` are excluded from this exact projection and
counted in run-index provenance rather than interpolated. The ledger exposes offered arrivals and
token marks, end-of-bin `A_t`, `delta_A_t`, and running/waiting request counts;
standalone inference rejects horizons that would truncate request work and
labels each `[t, t+dt)` prediction at `t+dt`.
Selected deployment artifacts use the strict
`powertrace-selected-physics-v1` schema: one hardware and one mode per file,
no family/model/TP routing, at most 80 declared learned scalars, and an explicit
conditional-timing or arrival-only validation contract. The legacy physics-v1
reader remains available for existing artifacts.

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

Slurm submissions should use `profiling/jobs/submit_campaign.sh`; it passes the
current checkout to the batch job as `POWERTRACE_REPO`. Override that variable
only when the submitted job must run a different checkout. On Sherlock, submit
A100 campaigns whose maximum TP needs more than two GPUs to `owners` rather than
`ramr`; the wrapper pins owners A100/H100 jobs to the matching 80GB GPU class.
Each allocation derives a job-specific vLLM port and passes the matching base URL
to readiness checks and every workload client, allowing campaigns to safely share
a node without cross-serving requests. Set `POWERTRACE_PORT` only to override it.
Sealed submissions explicitly export `$SCRATCH/ptsim/sealed-runs` as
`SEALED_RUNS`, create it with mode `0700`, and do not depend on the submitting
shell to define that variable. The batch entrypoint applies the same default for
direct `sbatch` use.
For large checkpoint staging, `profiling/jobs/stage_models.sh` defaults
`HF_SNAPSHOT_MAX_WORKERS=2`; lower it to `1` if the login-node downloader is
killed. Native `arch_extract` sanity is opt-in with
`STAGE_MODELS_ARCH_SANITY=1`; the launch container performs the authoritative
parse. Live campaign server processes are pinned with CUDA ordinal indices while
bundle manifests still record the exact active GPU UUIDs. Server teardown
refuses to signal the batch shell's process group if `setsid` has not
isolated the vLLM launcher yet. Campaign server readiness waits for `/health`,
the served model to appear in `/v1/models`, and a one-token `/v1/completions`
smoke request before probes start, so large checkpoints cannot be marked ready
while weights are still loading. Campaign
`server.quantization` is emitted to vLLM and recorded in bundle manifests; the
Llama-3.1-405B H100 TP8 cells use the pre-quantized
`RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8` checkpoint with
`server.quantization=compressed-tensors` and `dtype_hint=fp8` because BF16
weights do not fit on 8x80GB H100. The live power logger
stamps each all-GPU `nvidia-smi` query with one shared timestamp so bundle
ingestion can keep enforcing the 50 ms per-sample skew contract, and the metrics
logger records only successful scrapes so transient HTTP misses do not create
all-NaN evidence rows. Clock headers use canonical `clocks.sm` / `clocks.mem`
names; ingestion also accepts the older `clocks.current.*` display aliases. Probe
campaigns launch direct
`profiling/probes/<probe>.py` entry points, one per `schedule.BUILDERS` probe.
The standard `context_holds` schedule and direct CLI default use a 122880-token
top prefix rather than 131072 so tokenizer expansion cannot exceed the served
Llama context.

Live campaign bundles are written under `data/runs/<campaign_id>/<run_id>/` by
default. The analyzer writes CSVs to `results/occupancy_roofline/` and figures
to `figures/occupancy_roofline/`. The occupancy coordinate is `ell = f / F + g / G`,
where `F` and `G` are p99.5 sustained 5-second rates reconstructed from the
dedicated prefill and decode probes. Treat roofline and agentic claims as
reconstruction-based unless they are rebuilt explicitly with
`--state-source measured_engine`; that path is hybrid rather than phase-complete.
It uses the manifest's `clock.local_utc_offset_s` to compare power,
requests, and probe windows on exact UTC epochs; bundle analysis never applies
the legacy 30-minute timestamp fold.

## Splitwise-Style LUT Baseline Notes

The maintained facility evaluation keeps its Splitwise-style comparator in
`scripts/eval/splitwise.py`, separate from the archived learned baselines:
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

## Testing

The frozen architecture-transfer feature test is run with:

```bash
uv run python feature-test/evaluate_candidates.py
```

It emits the source-only selection, per-run/stratified/bootstrap metrics,
transfer gates, data-efficiency check, and hardware-local deployment artifacts
under `results/feature_test_v1/`. The current honest result selects M4A for both
A100 and H100, but six selected cells fail. Measured ITLs remove the earlier
within-request uniform-timing approximation; the remaining temporal failures
still lack observed engine batch/clock state. See `feature-test/README.md` for
the exact scores and evidence boundary.

```bash
uv run -m pytest -x
```
