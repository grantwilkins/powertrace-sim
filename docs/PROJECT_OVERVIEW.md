# What PowerTrace-Sim is all about

This document is a collaborator-facing overview of the project: the question
we are asking, how the model works, what evidence exists, and where the current
claim boundary lies. It describes the repository as of July 2026.

## The short version

PowerTrace-Sim is a configuration-bounded simulator that predicts request
timing and GPU power traces from an LLM serving request stream.

The central idea is that power should be predicted through the work the serving
system actually executes:

```text
requests
  -> serving scheduler
  -> per-iteration compute and memory work
  -> 250 ms work ledger
  -> calibrated GPU power
```

This makes the model more structured than a black-box time-series predictor.
It knows about prompt prefill, token-by-token decode, continuous batching,
KV-cache admission, tensor parallelism, model architecture, and hardware
limits. The learned part is deliberately small: measured data calibrates
effective hardware rates, overheads, and a compact power equation. The
scheduler and work accounting remain explicit.

The practical goal is to make request-level workload descriptions usable for
power-aware workload studies, capacity planning, and facility analysis while
reducing the need to profile each new arrival pattern directly.

## Why use this structure?

An end-to-end trace predictor can fit a particular collection of traces, but
it is difficult to ask what should happen when prompt lengths, batching,
tensor parallelism, or cache state changes. PowerTrace-Sim keeps those causes
explicit. This gives the project four useful properties:

- **Data efficiency:** measurements calibrate a compact model instead of every
  possible workload combination.
- **Counterfactual control:** a user can change the request stream or supported
  deployment setting and rerun the serving process.
- **Diagnosability:** timing, scheduling, work-ledger, and power-surface errors
  can be evaluated separately.
- **Scalability:** the same request-level model can generate long node traces
  and feed a facility aggregation.

The tradeoff is that the structure is also a claim. If the scheduler,
architecture accounting, cache semantics, or meter response is wrong, that
error propagates into power. This is why the repository reports phase timing
and temporal shape as well as total energy.

## What problem are we solving?

LLM serving power is not determined by request rate alone. Two request streams
at the same rate can produce different power because their prompts, generated
lengths, batching, context sizes, queueing, and cache state differ. Those
effects also change over time as requests overlap.

PowerTrace-Sim therefore takes a request schedule as its starting point. Each
request supplies:

- an arrival time;
- a prompt length;
- a fixed or sampled output length; and
- optionally, a cached-prefix length.

The output is both a request-timing table and a native 250 ms GPU power trace.
The model is conditional on the supplied workload; it does not forecast future
traffic or invent request arrivals.

## The modeling approach

### 1. Simulate the serving policy

The scheduler approximates vLLM V1 continuous batching. It models first-come,
first-served admission, finite KV-cache capacity, chunked prefill, and a
decode-first token budget. Requests can queue when the sequence or KV budget is
full.

A cached prefix remains part of the decoder's context, but its tokens are not
charged again as executed prefill work. This distinction is important:
requested prompt length, executed prompt work, and visible decoder context are
not always the same quantity.

Authoritative implementation:
[`model/timing/scheduler.py`](../model/timing/scheduler.py) and
[`model/request_schedule.py`](../model/request_schedule.py).

### 2. Derive work from the model architecture

For every engine iteration, the model calculates the work implied by the
active requests:

- transformer and output-head FLOPs;
- attention FLOPs;
- model-weight, attention, and KV-cache traffic;
- separate prefill and decode contributions; and
- expected expert-weight traffic for supported mixture-of-experts models,
  under the declared uniform-independent routing assumption.

Iteration time follows a roofline-style model. Each iteration sums separately
roofline-limited GEMM and attention durations, plus calibrated launch,
communication, and sampling overhead. A separate calibrated first-token
overhead is added to the reported TTFT and end-to-end latency. Model structure
such as layer count, hidden size, grouped-query attention, sliding-window
attention, active parameters, and tensor-parallel degree enters explicitly.

Authoritative implementation:
[`model/timing/iteration.py`](../model/timing/iteration.py).

### 3. Project irregular iterations onto a meter-scale ledger

Engine iterations do not line up naturally with a power meter. Their work is
projected by overlap onto fixed 250 ms bins. The resulting ledger includes
busy fraction, prefill/decode work rates, memory traffic, batch occupancy, and
engine-iteration rate.

This ledger is the interface between the serving model and the power model. It
also makes the model inspectable: a power error can be traced back to timing,
scheduling, work accounting, or the final power surface.

Authoritative implementation:
[`model/timing/ledger.py`](../model/timing/ledger.py).

### 4. Map work to power with a small calibrated equation

Dense-model power is a nonnegative combination of:

- idle power;
- the busy fraction weighted by the resident model size;
- compute utilization; and
- a duty-weighted memory-utilization term.

The supported GPT-OSS mixture-of-experts models use separate, model-specific
nonnegative surfaces, claimed only inside their declared configuration
support. Their coordinates include idle power, lagged logical memory
utilization, exact compute utilization, engine-iteration rate, and decode
batch. For dense models, hardware meter delay and averaging are explicit
response transformations rather than hidden recurrent state. The MoE surfaces
instead use a one-bin-lagged logical-memory coordinate.

The model first predicts mean per-GPU power and then applies tensor-parallel
scaling once to obtain node GPU power. It does not fabricate different traces
for individual GPUs inside a colocated TP group.

Authoritative implementation:
[`model/power/predictor.py`](../model/power/predictor.py) and
[`model/power/response.py`](../model/power/response.py).

## What is fixed, and what is fitted?

| Component | Treatment |
| --- | --- |
| Request semantics | Fixed contract |
| Continuous-batching and KV-admission policy | Explicit deterministic simulation |
| Architecture FLOPs and byte accounting | Derived from architecture records |
| Native time resolution | Fixed at 250 ms |
| Timing coefficients | Fitted effective compute/bandwidth and overhead terms |
| Dense power coefficients | Nonnegative fit on frozen work coordinates |
| MoE power coefficients | Nonnegative per-model fit on declared coordinates |
| Dense meter response | Explicit hardware-specific delay/averaging |
| Output-length randomness | Optional and seed-controlled; all other inference is deterministic |

Training refits the selected equations; it does not search over a new model
family. The prepared-data manifest hash-binds the timing data, run index,
frozen splits, probe calibration, and power ledger. The compact JSON release
contains the coefficients, architectures, presets, support rules, and
provenance needed for inference.

See [`docs/MODEL_PIPELINE.md`](MODEL_PIPELINE.md) for the paper-facing pipeline
and [`model/artifacts/powertrace_v1.json`](../model/artifacts/powertrace_v1.json)
for the default release.

## What the model currently supports

The maintained release covers declared Llama 3 and DeepSeek-R1-Distill dense
families on A100/H100 tensor-parallel deployments, plus GPT-OSS-20B and
GPT-OSS-120B configurations on A100. Named presets bind the model, hardware,
TP degree, precision, scheduler, batching limits, and routing assumption.

Unsupported changes fail by default. A caller may explicitly request an
extrapolation, but the outputs are labeled `unsupported_extrapolation`.

The default support contract does **not** include:

- pipeline, expert, data, or context parallelism;
- arbitrary serving schedulers or batching settings;
- distinct device-level behavior within a colocated TP group;
- disaggregated prefill/decode serving; or
- inferred cache behavior when cache state is not supplied.

The historical GMM-BiGRU implementation under
[`archive/gmm_bigru_v1/`](../archive/gmm_bigru_v1/) is a separate,
first-generation artifact. It is not part of the maintained inference path.

## The evidence ladder

The repository contains several kinds of evidence. They answer different
questions and should not be collapsed into one headline accuracy number.

| Evidence | Amount | Calibration relationship | What it supports |
| --- | ---: | --- | --- |
| Broader data inventory | 800 matched request/power run pairs across 29 configurations | Inventory only | Shows the available measurement envelope; not all of it enters the selected release |
| Canonical prepared corpus | 450 runs, 354,125 requests, 1,095,526 native power bins | Contains frozen fit and non-fit roles | Supports deterministic refitting and provenance of the selected model |
| Supported held-out replay | 333 held-out traces across seven model groups | Frozen release scored on non-fit measured runs | Strongest local power-performance evidence inside declared support |
| Timing parity | 735 prefill and 1,200 decode points | Frozen split, probe calibration, and regenerated release | Timing diagnostic; prefill includes calibration probes and queue-free training requests, while decode uses frozen evaluation roles |
| Qwen transfer diagnostics | Two 600-request traces | Target idle and, for the MoE case, source-derived platform ratios | Retrospective boundary evidence, not zero-shot validation |
| BurstGPT arbitrary arrivals | Three traces, 2,613 requests | Per-run target idle | Shows behavior under exact irregular arrivals with minimal retrospective calibration |
| OpenHands agent workloads | Six traces, 620 requests | Target-derived idle plus two calibrated dynamic gains | Retrospective few-shot platform calibration; not a cache-effect result |
| Azure facility study | 240 modeled nodes for 24 hours at 250 ms | Uses the regenerated release; no facility power target | A model-based consequence study, not independent facility validation |
| Existing disaggregated pilot | Nine cells, 11,250 requests | Two role idles and one decoder gain | Exploratory role-transfer evidence with important prefill/cache limitations |
| Cache-disabled disaggregated confirmation | Five cells: one calibration and four heldout, 2,400 requests | Four target scalars in the frozen protocol; six in a separate post-hoc timing diagnostic | Frozen result fails; the diagnostic recovers most prefill cells but not decode |
| External sealed evaluation | None that remains uncontaminated | Must be score-only after freeze | Pending |

The inventory, corpus, power-replay, and timing counts come respectively from
[`results/stage0/data_inventory.json`](../results/stage0/data_inventory.json),
[`results/clean_model/prepared_dataset.json`](../results/clean_model/prepared_dataset.json),
[`results/paper/selected_model_fidelity_table.json`](../results/paper/selected_model_fidelity_table.json),
and
[`results/paper/timing_parity_manifest.json`](../results/paper/timing_parity_manifest.json).

### Local held-out replay

The selected fidelity table reports model-level median energy error between
0.98% and 2.74%, median range-normalized RMSE between 4.21% and 10.92%, and
median ACF R² between 0.948 and 0.995 across the seven reported model groups.
These power metrics use matched, nonoverlapping one-second per-GPU means even
though inference and the underlying work ledger run at 250 ms.
These aggregates are encouraging, but they are not a statement that every
cell passes. Dense 70B TP8 rate-4 runs have a known late-run temporal drift,
and this is one reason the release remains `pre_sealed`.
The release-wide table is a compact summary, not a substitute for the per-run
CSV or the preregistered sealing gates.

The source of truth is
[`results/paper/selected_model_fidelity_table.json`](../results/paper/selected_model_fidelity_table.json).

### Transfer beyond the fitted model labels and arrival processes

The transfer results are deliberately labeled retrospective:

- Qwen3-14B/A100 reaches 0.52% energy error after replacing the source idle
  level, but its ACF R² is -1.54; good energy does not imply good shape.
- Qwen3-30B-A3B/H100 remains difficult: energy error is 32.0% before and
  20.6% after the documented calibration.
- The three BurstGPT strata have idle-calibrated energy errors of 1.16% to
  3.25%, with mixed temporal agreement.
- OpenHands reaches 0.58% to 2.14% calibrated energy error across six runs,
  but this uses target-derived platform calibration; cache-on TTFT remains
  weak, at roughly 44%–46% median absolute error in the reported cache-on runs.

These experiments show which parts of the structure transfer and which
platform effects need calibration. They do not establish broad zero-shot
generalization. Their exact calibration is recorded under
[`results/paper/appendix/`](../results/paper/appendix/).

### Facility-scale application

The Azure study feeds 240 Llama-3-70B A100 TP8 request streams through the same
scheduler and power model for a 24-hour scenario. At 15-minute resolution, the
selected-model scenario reports about 0.539 MW average power and 0.602 MW peak
power after the documented facility additions.

Those numbers are conditional on the workload allocation, serving policy,
model, non-GPU overhead, and PUE. There is no measured 240-node facility power
trace in this repository. The study demonstrates how request-level modeling
changes capacity and ramp estimates; it does not independently validate a
facility.

The Splitwise-style comparator records 5,484,298 LUT support-extrapolation
events and 5,492,641 power-clamp events, and it uses a model-family fallback.
It is therefore a scenario baseline rather than a clean measured oracle.

### Disaggregated prefill/decode inference

The existing pilot separates GPT-OSS-20B prefill and decode onto two A100 TP1
GPUs and applies the colocated source model independently to each role. The
disaggregated data was not used to train the base model, and no frozen timing
or power coefficient was changed. The retrospective analysis adds only two
role-specific idle levels and one accepted decoder dynamic gain. The pilot
also used an 8,192-token scheduler override outside the frozen preset, so it is
retrospective `unsupported_extrapolation` evidence.

On the two primary held-out warm cells:

- the report's native 250 ms held-out warm-cell acceptance passes for decode,
  with median correlation 0.949, standard-deviation ratio 0.962, p95 error
  1.52%, and phase energy error 0.74%; but
- prefill mean energy error is also about 0.74%, while its native trace does
  not pass: median correlation is 0.179, standard-deviation ratio is 0.473,
  and p95 error is 12.24%.

The distinction matters. Summed node energy or a smoothed plot must not be
used to hide a phase-level failure.

The pilot cannot establish cache-aware prefill fidelity. Warm repeats reached
roughly 97% prefix-cache hits, but the saved requests retained requested prompt
length rather than per-request computed versus cached tokens. A naive replay
would therefore charge prefill work that was not executed. The pilot power rows
were also timestamped before each `nvidia-smi` subprocess query, while the
prefill HTTP phases lasted only about 16–20 ms. The report therefore supports
decoder-shape transfer and warm-cell prefill mean behavior, not native
prefill-shape or cache-aware transfer.

The new confirmation campaign isolates the question more cleanly: caching is
disabled, the supported 2,048-token scheduler budget is used, and raw
`nvidia-smi power.draw` remains on a nominal 250 ms schedule. The logger
records query start/end and assigns their host midpoint without changing the
watt value, and prefill/decode roles are gated separately. All collection
gates pass.

The preregistered four-scalar calibration nevertheless fails heldout acceptance
for both roles. Median heldout correlation is 0.515 for prefill and 0.180 for
decode. A separately labeled post-hoc diagnostic uses only the calibration
cell to fit one service-time scale per role, followed by the same role idle and
dynamic-power fits. The inferred scales are 2.068× for prefill and 1.087× for
decode. With them, prefill passes three of four heldout cells and reaches
median correlation 0.919, standard-deviation ratio 0.953, p95 error 2.60%, and
energy error 4.24%. Decode still passes zero cells. Only the 2-request/s
prefill measured replay reaches the preregistered 0.8 correlation floor, so
the raw campaign itself does not support a blanket two-role trace-fidelity
claim.

See
[`results/disaggregated/gpt_oss_20b_a100_pd_confirmation_report.json`](../results/disaggregated/gpt_oss_20b_a100_pd_confirmation_report.json),
the
[`post-hoc timing report`](../results/disaggregated/gpt_oss_20b_a100_pd_confirmation_timing_calibrated_report.json),
and the future cache contract in [`docs/plans/TODO.md`](plans/TODO.md).

## The current measurement contract

Current campaign bundles record:

- raw per-GPU `nvidia-smi power.draw` queried on a nominal 250 ms schedule,
  with cadence and gaps recorded and validated in current campaigns;
- stable GPU index and UUID;
- clocks, utilization, memory use, and temperature;
- vLLM engine metrics at the same nominal cadence;
- request lengths, arrivals, TTFT, and inter-token timing; and
- run metadata binding the model, server settings, GPU identities, clocks,
  and code/data provenance.

This sampling stack is intentionally ordinary and deployable. Its limitation
is equally important: a 250 ms device power sample cannot resolve every short
kernel or 20 ms prefill burst. Stock vLLM metrics also do not reveal every
collective, NVLink transfer, or MoE router decision. The project rejects or
narrows claims when the instrumentation cannot identify them.

See [`profiling/CAMPAIGN.md`](../profiling/CAMPAIGN.md).

## Current limitations and open work

1. **The release is not sealed.** Aggregate held-out behavior is useful, but a
   new untouched external campaign still needs to be scored after the release,
   support, and thresholds are frozen.
2. **Dense high-load temporal drift is unresolved.** Several rate-4 70B TP8
   traces show a coherent late-run power step. A focused temperature, clock,
   P-state, and power-limit profile is needed before adding a correction.
3. **MoE routing is assumed uniform and independent.** Latency and node power
   cannot uniquely identify expert overlap. Direct router-ID instrumentation
   is planned.
4. **Cache-aware disaggregation needs new telemetry.** The prefiller must
   expose per-request computed and cached tokens, while the decoder retains
   full context semantics.
5. **Disaggregated timing is incomplete.** External-KV admission makes decode
   TTFT load-dependent; the pilot intentionally did not fit a timing scalar.
6. **Support is configuration-bound.** A good result on one scheduler,
   hardware, or TP configuration is not automatic support for another.

These items are tracked in [`docs/plans/TODO.md`](plans/TODO.md).

## A provenance detail collaborators should know

Two release snapshots are currently tracked:

- default inference uses
  [`model/artifacts/powertrace_v1.json`](../model/artifacts/powertrace_v1.json),
  SHA-256 `840eae4b…`;
- the regenerated paper replay and facility artifacts bind
  [`results/clean_model/powertrace_v1.json`](../results/clean_model/powertrace_v1.json),
  SHA-256 `5699cb82…`.

The disaggregated report binds the first; the paper manifest binds the second.
They are hash-distinct provenance snapshots, so reported metrics should always
travel with their recorded artifact hash rather than being presented as though
all outputs came from one identical file. The facility trace summary also
records that its source worktree was dirty; the compact outputs are hash-bound,
but this lowers the strength of that run's source-revision provenance.

## Reproducing and navigating the project

The maintained public lifecycle is:

```bash
uv run -m model.scripts.prepare_data
uv run --extra train -m model.scripts.train \
  --prepared-manifest results/clean_model/prepared_dataset.json \
  --out-artifact results/clean_model/powertrace_v1.json
uv run -m model.scripts.infer \
  --requests examples/requests.json \
  --deployment llama-3-70b-a100-tp4 \
  --out-dir outputs/example
uv run -m model.scripts.evaluate \
  --measured measured_power.csv \
  --predicted outputs/example/power.csv \
  --out outputs/example/evaluation.json
```

For the scientific story, start with:

- [`README.md`](../README.md): runnable repository overview;
- [`docs/MODEL_PIPELINE.md`](MODEL_PIPELINE.md): model and leakage boundary;
- [`docs/PAPER_OUTPUTS.md`](PAPER_OUTPUTS.md): evidence contract;
- [`results/paper/manifest.json`](../results/paper/manifest.json): exact local
  paper artifacts and hashes;
- [`results/README.md`](../results/README.md): maintained result families; and
- [`CLEAN_MODEL.md`](../CLEAN_MODEL.md): release status and remaining work.

Candidate studies under `power-test/`, `timing-test/`, and `feature-test/` are
development support. The maintained model interface lives under `model/`, and
the historical neural implementation lives only under `archive/gmm_bigru_v1/`.

## The claim we are ultimately trying to establish

The project is testing whether a compact, inspectable model of serving work can
predict useful GPU power traces across request mixes and deployment scales
with much less calibration than profiling every workload directly.

The evidence is strongest for supported colocated held-out replay, promising
but explicitly calibrated for several transfer workloads, mixed for
disaggregation—with most cache-disabled prefill cells recovered by a post-hoc
timing correction but decode unresolved—and not yet sealed externally. That
graduated statement is the current scientific position of the repository.
