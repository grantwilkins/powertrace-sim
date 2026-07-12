# EENERGY_PLAN: execution specification

Audited against the repository on 2026-07-10, after the common data-path
repair. This document is the source of truth for the paper execution order.
`THEMES.md` remains framing only, and its checked-in result list is stale until
the artifacts are regenerated.

## 1. Paper outcome

The paper answers three separate questions.

1. **Mean-power transfer.** How accurately does a first-order physics kernel
   predict node power across model, hardware, and tensor-parallel settings?
2. **Volatile trace fidelity.** Which generator should be used when 1 s IT-load
   swings, ramps, and temporal structure matter?
3. **Facility uncertainty.** How much do traffic synchronization, observed day,
   and facility size change load-duration curves (LDCs), peaks, and ramps?

The intended claims are deliberately narrower than the motivating language.

| Claim | Main evidence | Wording allowed after its gate |
|---|---|---|
| A | First-order kernel, held-out measured timing, sealed transfer targets | The kernel transfers mean node power across the declared holdout axes when request execution timing is known. |
| B | Paired 1 s held-out comparison on the same requests and metrics | The learned generator preserves the declared in-domain power, energy, ramp, and temporal statistics within its validated support. |
| C | Seven Azure days, explicit traffic assumptions, nested Monte Carlo | Monte Carlo propagation exposes scenario-conditioned LDC, peak, and ramp ranges hidden by a single replay. |

Do not claim universal bounds, calibrated facility prediction intervals, broad
zero-shot transfer, or one globally best generator. The public Azure trace has
timestamps and token counts but no server identity, placement, scheduler state,
or power. It cannot identify cross-server traffic correlation.

## 2. Evidence and language contract

Every artifact and paper row uses one label.

| Label | Meaning |
|---|---|
| measured | Direct request, timing, engine, or power observation |
| reconstructed | Work or state inferred from measured request timing |
| simulated | Output from a traffic, timing, or power model |
| scaled | Extrapolation to another fleet size or topology |

`zero-shot` is allowed only when no target power, execution timing, throughput,
cap, residual, or validation data influenced fitting or model selection.
Otherwise use `conditional-timing transfer`, `workload holdout`, or the exact
holdout axis.

An LDC envelope is the across-realization distribution of a scenario's sorted
power trace. It is a **predictive scenario envelope**, not a confidence band for
all data centers. Pointwise P05/P50/P95 curves are not simultaneous bands.

## 3. Audited repository status

### Ready foundations

- One `RunRecord` gives legacy and bundle data shared ingestion, identity,
  alignment, and failure semantics; GRU preparation adds disjoint splits,
  train-only calibration, lineage, and hash-bound artifacts.
- Main GRU evaluation/inference is IID and non-oracle. The deployable 11-term
  deterministic physics kernel, arrival-only CLI, and Azure adapter exist.
- TP/nameplate accounting, explicit-resolution ramps, exceedance-rank LDCs, all
  seven raw Azure days, and the profiling/probe/replay bundle path are tested.

### Stale or incomplete evidence

- Checked GRU data lack current lineage/bound identities; the ledger run index is
  absent; and the checked physics artifact predates the strict builder.
- Every checked paper output is marked stale. Seven days are split but only
  2024-05-16 is parsed, and the day manifest predates current provenance fields.
- `data/runs/` has no live canonical bundles. Campaign JSONs are schema-valid,
  not live-ready evidence.

### Scientific and correctness blockers

1. Arrival-only timing is queue-free constant throughput, without capacity,
   waiting, batching, backlog, completion, or SLO behavior.
2. The learned F2 path uses only active requests and change in active requests;
   it discards arrival-token, prefill-token, and decode-token channels.
3. Training uses measured completion timing while rollout assumes queue-free
   serial work. This mismatch grows under overload.
4. Measured `engine.csv` ingestion is unimplemented; cache-on/agentic attribution
   remains blocked. Physics residuals are also disabled.
5. New-model inference needs a hash-bound architecture descriptor resolver.
6. Physics labels `[0,dt)` at `0`; GRU output begins at `dt`. Freeze end-of-bin
   labels and lag initialization before comparison.
7. The checked physics artifact was fit at 1 s; it cannot establish 250 ms
   dynamics by running the same coefficients four times faster.
8. GRU sampling uses `node_seed + 23` but records `node_seed`; fix provenance.
9. Azure metrics can skip bad methods, assume fixed filenames/resolutions, and
   require GRU data for physics-only runs.
10. A current 240-node/two-method run uses about 687 MB. Streaming is required.

The repaired **code path** is the foundation. The old generated **artifacts**
are not evidence for the repaired path.

## 4. Decisions frozen for the first implementation

These values prevent each workstream from inventing its own experiment.

| Decision | Specification |
|---|---|
| Native work/power grid | 250 ms half-open bins `[t,t+dt)`, labeled at the bin end; no rounded joins |
| Paper comparison grid | Aggregate native outputs to 1 s; keep 250 ms ramps as a separate fidelity result |
| Build order | Shared work/timing contract -> deterministic physics mean -> learned fidelity/residual -> Monte Carlo |
| Learned feature candidate | arrivals, change in arrivals, executed prefill tokens/s, executed decode tokens/s; add queue/context only by ablation |
| Reference fleet | `N_ref = 240` as a paper scenario assumption, not Azure metadata |
| Reference topology | 10 rows x 6 racks/row x 4 servers/rack |
| Reference configuration | `llama-3-70b_A100_tp8` for the compatibility pilot, subject to the regenerated artifact gate |
| Power domains | Preserve node GPU power, node IT power, and facility power as separate fields |
| Reference overhead | 1,000 W non-GPU IT power/server and PUE 1.3; both are scenario inputs, not measurements |
| Main traffic modes | Existing `partitioned_replay` and one new `correlated_intensity` model |
| Correlation sweep | `rho in {0.0, 0.5, 0.9}`; `rho` is latent log-intensity correlation |
| Facility sizes | 240 main; 2,400 only after the streaming/equivalence gate; 24,000 is cut |
| Observed scenarios | Seven Azure days, reported separately and equally weighted in week summaries |
| Main uncertainty output | Pointwise P05/P50/P95 LDC envelopes plus distributions of daily peak and ramp statistics |
| Tail limit | No conditional P99 outcome until the stopping rule supports it; P50/P95 are primary |
| Agentic scope | Cache-off appendix or secondary result; cache-on waits for engine-state and cache modeling |
| Fleet scope | Homogeneous fleet in the main paper; heterogeneous routing and mixtures are cut |
| Grid scope | PyPSA and grid optimization remain cut |

Changes require a decision record with the old/new value, reason, affected
artifacts, and required reruns.

## 5. Organization, blocking joins, and critical path

```text
Paper lead / claim registry
|
+-- F0 foundation: provenance, roles, regenerated artifacts ----------+
+-- Traffic: golden replay -> rho model -------------------------------+
+-- Shared 250 ms work contract --------------------------------------+-- G0
+-- Physics: probes -> deterministic mean -----------------------------+-- G1
+-- Learned: token features -> ablations -> volatile residual --------+-- G2
+-- MC infra: schema -> adapters -> streaming -> LDC ------------------+-- G3
+-- Statistics/paper: estimands -> stopping rule -> methods -----------+
                                                                         |
G1 + G2 + G3 -> timing/overload gate -> 240-node pilot -> optional 2,400|
        -> final grid -> sealed review -> figures/tables/manuscript ----+
```

The join before the final grid requires all of the following:

- traffic conservation and achieved-correlation tests pass;
- the selected power method is frozen for the use case;
- arrival-only timing passes its operating envelope, or the experiment is
  restricted to the passing load regimes;
- explicit and streaming facility summaries match;
- all stochastic seeds and input identities reproduce the run;
- runtime, RAM, disk, and Monte Carlo convergence meet the pilot budget.

### Work that can start in parallel now

| Lane | Can do now | Must not wait for |
|---|---|---|
| Data/traffic | Parse six remaining days; provenance; golden replay; traffic model and tests | Model retraining or live GPUs |
| Work contract | Build one native 250 ms demand/execution ledger and exact alignment tests | Final physics coefficients |
| Artifact rebuild | Rebuild physics first; learned feature plumbing then proceeds in parallel | Monte Carlo code |
| Learned fidelity | Canonical token features, direct/hybrid ablations, split and leakage tests | Final live campaign |
| MC infrastructure | Run schema, method adapter, streaming accumulator, hand-worked LDC summaries | Final generator choice; use a fake kernel |
| Profiling operations | Role/readiness registry, dry runs, container/weight staging | MC infrastructure |
| Statistics/paper | Estimand sheet, claims matrix, methods skeleton, convergence renderer | Numerical result prose |
| Agentic appendix | Pin SWE-smith/tokenizer revisions and fit or sensitivity-test gap priors | Main paper critical path |

## 6. Work packages and gates

### F0 - Freeze provenance and regenerate the repaired path

Deliverables:

1. Record the current fixed-seed 240-node `partitioned_replay` hashes and request/
   token conservation totals before traffic changes.
2. Parse seven days to isolated paths with source hash, UTC span, request/token
   totals, rejected rows, and command.
3. Register campaign roles (`fit`, `calibration`, `development`,
   `workload_holdout`, `sealed_external_validation`) and readiness (schema,
   container, weights, hardware, dry run, live smoke, complete).
4. Freeze the native 250 ms bin, end-of-bin timestamp, warm-up, meter-lag, and
   seed contracts before rebuilding any model.
5. Rebuild in order:

   ```text
   Stage0/bundles -> shared demand/execution ledger -> physics fit
   -> learned feature ablations -> model selection -> evaluation
   ```

6. Record byte identities; do not overwrite stale evidence without a new version.

Exit gate:

- golden replay matches exactly;
- all seven parsed manifests are repo-relative and hash-complete;
- current consumers accept the rebuilt GRU and physics artifacts;
- a clean rerun reproduces deterministic artifacts or documents the allowed
  floating-point tolerance;
- sealed IDs are committed before their power is inspected.

### F1 - Traffic model

Keep the existing replay unchanged and add only one synthetic sensitivity model.

#### `partitioned_replay`

Each observed request is assigned exactly once, uniformly, to a node. It
preserves the aggregate day and joint token marks exactly. It says nothing about
the original Azure placement.

#### `correlated_intensity`

Use 1 s count bins. For each day:

1. Compute a 300 s centered request-rate envelope `mu[d,t]` with edge-aware
   windows. Let `sigma[d]` be the standard deviation of
   `log(count+0.5)-log(mu+0.5)` and standardize those residuals.
2. Draw independent 300 s circular block-bootstrap residual series `z_site` and
   `z_node[i]`.
3. Set `lambda[i,t] = mu[d,t]/N_ref * exp(sigma[d] * (sqrt(rho)*z_site[t] +
   sqrt(1-rho)*z_node[i,t]) - c)`, where `c` makes the empirical mean multiplier
   one. Thus `rho` changes synchronization without changing expected volume.
4. Draw conditional counts from a Poisson distribution.
5. Sample `(n_in, n_out)` jointly, with replacement, from the same day's hour-of-
   day stratum. Never sample input and output lengths independently.
6. Place arrivals uniformly inside each 1 s bin.

At fleet size `N`, expected requests and both token totals scale by `N/N_ref`,
so expected per-server offered work is constant. The aggregate trace does not
identify the node residual law; reusing the aggregate residual shape is a stated
synthetic assumption. `rho` is verified by achieved detrended count correlation,
not by its input value alone.

Record separate seeds for residual blocks/counts, marks, placement, model
sampling, and optional power residuals. A seed tuple names one replicate.

Traffic gate:

- exact replay conservation and golden equivalence;
- ensemble mean request and token rates within 1% of the target;
- achieved pairwise correlation rises monotonically with `rho`, with intervals
  over sampled node pairs and seeds;
- per-node marginal variance and residual ACF remain stable across `rho` within
  predeclared development tolerances;
- joint token-mark distributions and out-of-support fractions are reported;
- raw Azure and normalized request-column integration tests pass.

The 300 s envelope/block choice gets one appendix sensitivity at 60 s and 900 s.
Do not add Hawkes processes, Gaussian processes, or a common-shock model unless
this minimal model fails its diagnostics.

### F2 - Shared work contract, first-order model, and profiling

#### F2a - One native work contract

Use 250 ms half-open bins `[t,t+dt)` and label each value at `t+dt`. Preserve raw
units and normalize from training runs only. Store:

- **offered demand:** arrivals, input tokens arriving, requested output tokens
  arriving, and backlog;
- **executed work:** prefill tokens/s, decode tokens/s, running/waiting requests,
  context-weighted decode tokens (or KV bytes), KV occupancy, and reused prefix
  tokens when caching is on.

The first-principles equation consumes executed work, not arrival change. Rates
must conserve source tokens. Deltas reset at run boundaries. Tests cover off-grid
requests, simultaneous prefill/decode, queue delay, dense/MoE/SWA/hybrid/FP8
work, TP communication, unused GPUs, long-context KV, cap, lag, and equality of
training and rollout state for the same synthetic schedule.

#### F2b - Profiling preflight

The new campaign code has produced no live canonical bundle. Before GPU use:

1. Record and exclude a 60 s post-health warm-up.
2. Exclude the first/last 2 s of every level and require 30 usable seconds.
3. Require 4 Hz power/engine cadence (median within 5%, no gap over 1 s), >=99%
   finite required engine fields, stable GPU index/UUID, <=50 ms GPU capture
   skew, no counter reset, exact fixed lengths, and no request failure.
4. Add cache query/hit and preemption counters before cache-on campaigns.
5. Run and ingest a short A100 Llama-70B TP4 smoke bundle; stop on any failure.

#### F2c - Exact anchor schedule

| Probe | Levels | Request shape | Dwell |
|---|---|---|---|
| Idle | one | no traffic | 60 s |
| Decode | concurrency 1,2,4,8,16,32,64,128,256 | input 8, output 2048 | 45 s/level |
| Prefill | input 256,1k,4k,16k,65k | concurrency 1, output 1, chunking off | 45 s/level |
| Context decode | prefix 2k,8k,32k,131k | batch 8, new input 8, output 256 | 45 s/level |
| Transient | four idle/load pairs | 20 s idle + 20 s at concurrency 64 | 160 s |
| Mixed | 16 fixed seed-0 points | concurrency 1..256, input 256..16k, output 512 | 45 s/point |

Primary-TP dwell is 29.2 minutes; second-TP decode+prefill is 10.5 minutes.
Record wall time and allocated GPU-hours separately. Run once, then repeat only
levels that fail usable-dwell or pilot-noise gates.

#### F2d - Campaign roles and order

| Order | Role | Campaign |
|---|---|---|
| 1 | FIT | A100 Llama-70B TP4 full; add deferred TP8 decode+prefill when available |
| 2 | FIT | H100 Llama-70B TP8 full + TP4 decode+prefill |
| 3 | DEVELOPMENT | A100 Gemma-4-31B dense TP2 and Gemma-4-26B-A4B MoE TP2 condensed |
| 4 | WORKLOAD HOLDOUT | chat, long-context, agentic cache-off on development models; cache-on diagnostic |
| 5 | SEALED | A100 Qwen3-8B TP1 dense and H100 Qwen3-30B-A3B TP2 MoE |

Freeze sealed IDs before power inspection. Legacy GPT-OSS runs are development
only: they lack canonical manifests/engine logs, and one 120B duplicate stopped
at 32/75 requests.

Physics acceptance requires deterministic refit, physical signs, synthetic
recovery, held-out levels, cross-probe prediction, reported cap/support failures,
and one artifact with equations, architecture, 250 ms lag, fit IDs/hashes,
revision, and validation. Validate measured `engine.csv` bins against reconstructed
bins from the smoke bundle before using them.

### F3 - Learned token features, timing, and fair comparison

Token-aware fidelity is required. Input/output lengths describe offered work;
power needs executed work placed by the same scheduler at training and rollout.
Measured TTFT/decode work is allowed only in a labeled retrospective test.

Use the same architecture, splits, seeds, and 250 ms targets for this ablation:

| Step | Features |
|---|---|
| A | current active requests + change in active requests |
| B | arrivals + change in arrivals + input/output tokens arriving |
| C | arrivals + change in arrivals + executed prefill/decode tokens/s |
| D | C + running and waiting requests |
| E | D + context-weighted decode/KV work; add reused tokens for cache-on |

Compare the current direct learned model with a learned residual around the
physics mean. Choose the smallest combination whose gain holds for every key
regime and seed, not only the pooled average. Report 250 ms peak/ramp error,
1/5/30 s energy, NRMSE, ACF, power-state and LDC-tail error, plus failures. Keep
all turns of a session in one split.

The BiGRU sees future bins, and final output length is known only because the
offline scenario samples it. Call this offline trace generation. Add a one-way
GRU comparison before any online or causal claim.

#### Timing and overload contract

Keep `measured_timing` for retrospective kernel validation and `arrival_only`
for simulation. Validate below-knee, near-knee, and overload load. Report TTFT,
decode duration, running/waiting, offered/completed work, backlog, SLO, and
support. Arrivals remain open-loop: overload never slows the source. Do not drop,
clip, or force-finish work; unfinished work stays in backlog at the horizon.

Compare learned and physics paths on the same requests, native bins, end labels,
horizon, warm-up, initialization, and metric code. No measured-power alignment
or measured initial power is allowed in the main comparison. Report separately:
in-config fidelity, measured-timing kernel fidelity, and arrival-only transfer.

Freeze margins after development. Use the learned path for in-domain Monte Carlo
only if it passes dynamics, energy, support, and failure gates. Use physics for
transfer only where timing and kernel gates pass. Add a stochastic residual only
after training-only mean-zero, variance, ACF, energy, cap-order, and seed tests.

### F4 - Monte Carlo implementation and statistics

Build four small boundaries, not a general simulation framework. Preferred
homes are `model/pipeline/traffic.py`, adapters beside existing inference code,
the accumulator in `scripts/eval/facility.py`, and thin
`scripts/eval/monte_carlo_facility.py` plus a separate renderer.

1. `traffic`: normalized day + scenario + traffic/mark seeds -> node request
   schedules and diagnostics.
2. `power method`: requests + config + horizon + model seed -> node power and
   support/failure diagnostics. Adapters wrap existing GRU and physics code.
3. `facility accumulator`: consume one node trace at a time; retain site trace,
   selected rack/row traces, node-peak sum, and counters. Full node traces are a
   debug option only.
4. `Monte Carlo orchestrator`: read one scenario manifest and own parse ->
   traffic -> power -> accumulation -> summaries in an isolated run directory.
   A separate renderer reads artifacts and never reruns simulation.

For each accepted replicate and power resolution `r in {1 s, 1 min, 15 min}`:

- compute the descending LDC on a fixed 1,001-point exceedance grid;
- compute peak, mean, energy, load factor, maximum rolling 1/15-minute mean;
- compute signed maximum up/down and P95 absolute ramps;
- record cap, support, overload, backlog, completion, SLO, failures, runtime,
  peak RAM, and disk bytes.

Never concatenate bins across replicates before sorting. First compute one LDC
per replicate, then summarize each exceedance rank across replicates.

#### Replication and uncertainty rule

- Pilot: 3 independent seed tuples per day/condition.
- Final sampling: batches of 10, minimum 30, maximum 200 per day/condition.
- P95 is not reported before 100 accepted replicates for that condition.
- After each batch, bootstrap accepted seed tuples within each day. Stop only
  when, for three consecutive batches, the 95% Monte Carlo interval half-width
  is at most 1% of the estimate for the P50 daily peak and at most 2% for the P95
  daily peak and LDC values at exceedance fractions 0.01, 0.05, and 0.50.
- If the maximum is reached, report the unresolved Monte Carlo error and reduce
  claims; do not pool days or loosen the rule after seeing results.

Days are the outer scenario unit; seeds are nested within day. Week summaries
weight the seven days equally and use a hierarchical bootstrap that resamples
days, then seeds within days. Day-specific rows remain primary. These intervals
describe this observed week plus the declared simulation, not a population of
future Azure days.

### F5 - Scale pilot and final grid

Small integration gate:

1. fake deterministic power method, one hour, small hand-worked fleet;
2. explicit files versus streaming accumulator, exact metric equivalence;
3. real selected method, one hour, then 24 hours at 240 nodes;
4. one Azure day x 240 nodes x 3 seed tuples x `rho={0,0.5,0.9}`;
5. record throughput, wall time, RAM, disk, failures, and interval movement.

Only then run the seven-day 240-node grid. Run 2,400 nodes only if streaming or
a cohort method matches explicit simulation at smaller sizes for LDC, peak,
ramps, energy, hierarchy totals, and failure counters. There is no 24,000-node
grid in this paper.

Primary traffic/facility matrix:

| Axis | Main values |
|---|---|
| Day | 2024-05-10 through 2024-05-16 |
| Fleet size | 240; 2,400 only after scale gate |
| Traffic | `partitioned_replay`; `correlated_intensity` with rho 0.0/0.5/0.9 |
| Configuration | regenerated `llama-3-70b_A100_tp8` pilot; one frozen passing method |
| Power resolution | 1 s primary; 1 min and 15 min reductions |
| Repeats | adaptive rule above |

Baselines and ablations:

- nameplate and constant training mean are sizing references, not stochastic
  traces;
- fixed-seed `partitioned_replay` is the single-replay reference;
- independent circular/block shifts remove cross-node alignment while preserving
  node temporal shape;
- IID time shuffle is an explicitly destructive marginal ablation;
- noise-off isolates a validated stochastic power residual;
- Splitwise stays appendix-only with its support/clamp rates;
- normal-sum checks are pointwise only and never stand in for daily maxima or
  ramp distributions.

### F6 - Generalization and sealed validation

Use a holdout taxonomy, not one `generalization` column.

| Question | Development evidence | Confirmatory target |
|---|---|---|
| In-config fidelity | Representative dense/MoE, small/large, A100/H100 configs | Held-out runs from the same config |
| Conditional-timing transfer | Historical 405B and gpt-oss cases | Sealed A100 dense and H100 MoE targets |
| Arrival-only transfer | Passing timing development configs | Same sealed targets with target timing/throughput/power excluded |
| Workload transfer | Chat development runs | Low/near-knee/overload chat; long-context; cache-off agentic |

Seal A100 Qwen3-8B dense TP1 and H100 Qwen3-30B-A3B MoE TP2 before power
inspection. Smoke-test each cell, then run three repeats only for passing cells;
15-minute claims require 45 post-warm-up minutes. Historical 405B/gpt-oss are
appendix development cases. Cache-on and broad sweeps cannot delay the core.

### F7 - Paper and artifact production

| Item | Content |
|---|---|
| Fig 1 | Measured/reconstructed/simulated data flow and two generator roles |
| Fig 2 | Node fidelity plus conditional-timing transfer |
| Table 1 | Use-case validation and support/failure limits |
| Fig 3 | Pointwise LDC P05/P50/P95 versus rho and fleet size |
| Fig 4 | Peak and 1 s/1 min/15 min ramp sensitivity |
| Appendix | Full holdout matrix, retrospective transfer, convergence, traffic diagnostics, block-length sensitivity, scale equivalence, workload details |

Write methods now and numerical claims after gates. Every value maps to a
producer, exact command, immutable input manifest,
hashes, seeds, code revision, model artifact, topology, timing mode, power domain,
failure counts, and expected runtime in `results/eval_paper/README.md`.

## 7. YAGNI and cut order

Build only two traffic modes, one schema/orchestrator/accumulator/renderer, and
adapters for the two existing power paths. Extract a core only when shared.

Cut in order: grid optimization; mixed fleets; cache-on main results; 2,400
servers (24,000 is already cut); stochastic transfer bands; extra rho/baselines;
then P99. Never pool dependent seeds, weaken sealed tests, or hide failures.

The minimum viable paper is: regenerated node evidence, one defensible
conditional-timing transfer result, one validated in-domain volatile generator,
the corrected 240-node replay, and one synthetic synchronization sensitivity
with scenario-conditioned LDC/peak/ramp envelopes.

## 8. Required test gates

For every code phase:

```bash
uv run -m pytest -x
uv run -m pytest -x model/tests profiling feature-test
```

Also run the primary producer or smallest integration case affected by the
change. A task is not complete when only unit tests pass.
