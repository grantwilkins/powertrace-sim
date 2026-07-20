# Data inventory and evidence-gated campaign plan

Status: executable final-stage campaign authority, 2026-07-20.

## Final-stage campaign

This section supersedes the diagnostic ordering in the remainder of this
document for paper-final data collection. The later sections remain the
evidence audit explaining why this campaign is small.

The paper's primary claim is descriptor-based transfer: a simple timing and
power model should generalize to unseen request schedules and unseen
checkpoints without an arrival-rate coefficient, model-name correction, or
target-trace refit. We minimize E2E timing error, energy error, ACF-MAE, and
range-normalized RMSE. ACF R² is the exception: higher is better, so its gate is
a minimum rather than a minimization objective.

Freeze the timing fit, power fit, feature set, architecture registry, metric
code, gates, and all trace-plan hashes before collecting any bundle marked
`validation_role=sealed`. A sealed failure creates a stated support boundary;
it never authorizes a post-hoc coefficient.

### Minimal expressive matrix

| question | frozen campaign | launches | why it is sufficient |
|---|---|---:|---|
| irregular real arrivals and long temporal shape | `sealed_burstgpt_qwen3-8b_a100.json` | 3 | disjoint 15-minute low/median/high Fano strata on a known dense deployment |
| real agent/tool timing and prefix-cache treatment | `sealed_openhands_qwen3-8b_a100.json` | 6 | three disjoint OpenHands packs, each cache-off/on with exact prompt and output-token identity |
| unseen dense checkpoint | `sealed_qwen3-14b_a100.json` | 1 | one new dense scale on a calibrated hardware family |
| unseen MoE checkpoint and hardware | `sealed_qwen3-30b-a3b_h100.json` | 1 | one new sparse architecture on H100; this is a support test, not a router-law fit |

This is 11 workload launches and four scientific contrasts. Do not add a rate
sweep, context sweep, checkpoint ladder, TP cross, synthetic-agent control, or
another arrival family. Those additions either duplicate existing development
support or confound the paper's final transfer claim.

The three BurstGPT windows are full fixed horizons, not capped request prefixes.
The three OpenHands packs are disjoint by a seeded stable hash of
`instance_id`. Every cache pair uses the same real system/user/tool text,
trace reply as subsequent context, singleton-allowed output token, request
seed, and normalized plan hash. The CPU comparator must report `identical`
before a cache effect is scored.

### Freeze and materialize inputs

Pin the BurstGPT CSV revision by its immutable content hash. Generate the three
plans from the same source:

```bash
uv run python profiling/agentic_traces/build_trace_plan.py \
  <burstgpt.csv> data/trace_plans/burstgpt_15min_fano0.json \
  --format burstgpt --revision <sha256> \
  --window-duration-s 900 --window-index 0 --window-count 3
uv run python profiling/agentic_traces/build_trace_plan.py \
  <burstgpt.csv> data/trace_plans/burstgpt_15min_fano1.json \
  --format burstgpt --revision <sha256> \
  --window-duration-s 900 --window-index 1 --window-count 3
uv run python profiling/agentic_traces/build_trace_plan.py \
  <burstgpt.csv> data/trace_plans/burstgpt_15min_fano2.json \
  --format burstgpt --revision <sha256> \
  --window-duration-s 900 --window-index 2 --window-count 3
```

The OpenHands input is pinned to dataset commit
`aa8977805b4cefd317001d80ddf1ad52790e9d23` and the CodeActAgent
Claude-3.5-Sonnet v2.2 output JSONL named in `openhands_adapter.py`. Dataset
text and observed action-to-observation timestamps are preserved. There is no
fitted or sampled tool-gap model on this path. The selected packs contain real
multi-hour waits, so `sealed_openhands_qwen3-8b_a100.json` binds a 48-hour
Slurm limit rather than relying on the four-hour batch default.

Stage the unseen checkpoints and the pinned OpenHands file before submission:

```bash
bash profiling/jobs/stage_models.sh Qwen/Qwen3-14B Qwen/Qwen3-30B-A3B
bash profiling/jobs/stage_openhands.sh \
  aa8977805b4cefd317001d80ddf1ad52790e9d23
```

Dry-run all four configs and inspect their exact commands before submission:

```bash
bash profiling/jobs/run_campaign.sh profiling/campaigns/sealed_burstgpt_qwen3-8b_a100.json
bash profiling/jobs/run_campaign.sh profiling/campaigns/sealed_openhands_qwen3-8b_a100.json
bash profiling/jobs/run_campaign.sh profiling/campaigns/sealed_qwen3-14b_a100.json
bash profiling/jobs/run_campaign.sh profiling/campaigns/sealed_qwen3-30b-a3b_h100.json
```

Live sealed jobs must set a physically separate `SEALED_RUNS` root and use
`submit_campaign.sh` or `run_campaign.sh --execute`. The campaign layer refuses
to write sealed bundles to the development output root.

### Pre-registered scoring

Each bundle must have validated `measured_ledger` instrumentation and at least
120 seconds of power overlap. The primary per-bundle gates are:

| metric | gate |
|---|---:|
| median absolute E2E timing error | ≤ 10% |
| total energy error | ≤ 6% |
| ACF-MAE | ≤ 0.05 |
| ACF R² | ≥ 0.90 |
| range-normalized RMSE | ≤ 0.20 |

Report every run, the median and worst run within each question, and every
failure. The campaign-level claim passes only if every primary run passes and
all three OpenHands cache pairs pass exact keyed identity. Session-level timing
is secondary diagnostic evidence and cannot rescue a failed run-level gate.

Open all sealed results in one score-only invocation. The scorer refuses
development-role bundles, incomplete telemetry, duplicate run IDs, and an
existing output path; it writes hashes rather than fit paths:

```bash
uv run python power-test/score_sealed_campaign.py \
  --timing-fit <frozen-timing-fit.json> \
  --power-fit <frozen-power-fit.json> \
  --bundle-dir <sealed-run-1> \
  --bundle-dir <sealed-run-2> \
  --out <new-sealed-report.json>
```

Repeat `--bundle-dir` for all 11 bundles. Run
`profiling/probes/compare_trace_replays.py <off> <on>` on each OpenHands pair
before scoring. A missing/corrupt bundle may be recollected with the identical
frozen plan. A scientifically valid failure may not be rerun selectively.

### Stop rules

- Stop after these 11 launches if all bundles are valid.
- Recollect only an invalid instrumentation or incomplete-request run.
- Do not fit to, tune on, or choose among models using sealed results.
- Do not average away a failing stratum or checkpoint.
- Do not claim zero-shot loaded-idle transfer; every campaign records a
  60-second deployment idle anchor.
- Do not claim general MoE routing transfer from the single Qwen MoE cell. A
  failure bounds support; a pass is one held-out architecture result.
- Defer additional FP8, TP, router, long-context, and engine-policy axes to
  future work unless the paper changes its primary claim.

This plan serves one mission: build high-fidelity power curves that transfer to
arbitrary marked arrival schedules and unseen model architectures. Arrival
rate, model identity, and elapsed-time steps are not acceptable explanatory
features. A model term must describe observable work, engine state, or hardware
state available before the target power trace is opened.

No broad crossed design is authorized. Each proposed run below resolves one
named ambiguity and has a stopping rule.

## 1. Executive decision

The apparent “rate-4 degradation” is not one mechanism:

- `T4`, a large-checkpoint timing failure, is associated with mixed
  prefill/decode intervals and the timing/concurrency feedback they create;
  the exact operator mechanism is not yet identified.
- `P4`, a 70B TP8 power-shape failure, is a long-horizon operating-state or
  request-composition transition near 305 seconds.
- H100 Qwen rate-4 has a short-run power miss whose idle versus active-state
  cause is not yet separated.
- Existing cache-on/cache-off agent traces are individually useful but are not
  a valid treatment pair because their prompt and output identities differ.

Do not add:

- a rate-4 scalar;
- a model-name correction;
- an unconditional preemption penalty;
- a retrospective 305-second step;
- a checkpoint-size idle law from the present sparse and state-confounded
  anchors.

The immediate order is:

1. use existing controlled data to test a phase-resolved mixed-iteration model;
2. collect one controlled 405B mixed-prefill diagnostic only if the existing
   data leave its FP8 mechanism ambiguous;
3. validate the chosen timing mechanism on one current-stack 405B rate-4 run;
4. collect only the missing TP8 leg of the existing 70B state experiment;
5. resolve loaded-idle and deterministic cache replay with one-factor,
   identity-checked measurements, not a crossed campaign.

## 2. Current inventory

### 2.1 Legacy serving corpus

`timing-test/timing_dataset.npz` contains:

| dimension | coverage |
|---|---|
| runs | 450 |
| requests with exact ITLs | 354,125 |
| hardware | A100 and H100 |
| model identities | 7 |
| configurations | 25 |
| rates | 0.125, 0.25, 0.5, 1, 2, and 4 requests/s |
| repeats | 3 per cell |
| split roles | 160 train, 80 in-domain, 108 twin, 54 model, 48 rate |

It covers dense 8B/70B, FP8 405B, and gpt-oss MoE models across TP1-TP8.
Arrivals are stationary Poisson and prompts are short: prompt
minimum/P50/P90/P99/maximum is 4/95/621/802/1,020 tokens.

### 2.2 Canonical bundles

There are 38 complete bundles under `data/runs/`, not the 26 described by the
previous revision of this plan:

| group | bundles | status |
|---|---:|---|
| original A100/H100 operator and serving bundles | 26 | retained |
| Qwen hardware/rate/shape expansion | 5 | collected and scored |
| TraceLab cache-off/cache-on | 2 | collected; invalid as a paired treatment |
| TraceLab smoke pair | 2 | protocol development only |
| BurstGPT | 1 | collected and scored |
| Gemma-4-26B-A4B | 1 | collected; zero-shot MoE claim remains gated |
| H100 Llama-70B TP4 state control | 1 | collected and scored |

The H100 TP8 state diagnostic is configured in
`profiling/campaigns/h100_tp8_state_diagnostic.json` but has not been
collected. The matched TP4 control exists under
`data/runs/h100_tp4_state_control/`.

### 2.3 Current development scores

The corrected v3 artifacts remain development candidates under `/tmp`; they do
not replace the frozen checked-in surface.

| evidence | timing | power/energy | interpretation |
|---|---:|---:|---|
| frozen non-training matrix | 3.57% median E2E | — | strong general baseline |
| H100 405B, rate 1 | 2.88% E2E | — | pass |
| H100 405B, rate 2 | 8.12% E2E, signed -8.10% | — | miss begins before rate 4 |
| H100 405B, rate 4 | 12.50% E2E, signed -12.46% | 4.84% energy, 0.018 ACF-MAE | timing failure, not power failure |
| Qwen A100/H100, rate 4 | 6.37%/4.97% E2E | 0.56%/14.22% energy | timing transfers; H100 power fails and L1 separates idle from active residuals |
| Qwen A100 rate/shape trio | 5.69-5.95% E2E | 1.64-1.82% energy | no generic high-rate failure |
| BurstGPT A100 | 2.06% E2E | 7.89% energy | timing pass, energy above target |
| TraceLab cache-off | 3.45% E2E | 25.75% underprediction | full-prefill energy fails |
| TraceLab cache-on | 15.55% E2E | 2.27% underprediction | timing fails |
| H100 TP4 state control | 2.23% E2E | 5.70% energy, 0.0166 ACF-MAE | same rate-4 marks pass temporally at TP4 |

The short 52-83 second Qwen arrival experiments cannot grade a 60-second ACF
curve reliably. Their timing and energy summaries remain valid, but long-lag
temporal claims do not.

## 3. Why rate 4 degrades

### 3.1 It is not a universal rate effect

Rate-4 signed E2E error has opposite signs across model families:

| representative cell | signed E2E error |
|---|---:|
| H100 405B TP8 | -12.46% |
| A100 Llama-70B TP4 | -16.98% |
| H100 Llama-70B TP8 | -6.79% |
| H100 Llama-8B TP1 | +0.48% |
| A100 gpt-oss-120B TP4 | +18.24% |

A shared rate correction would improve some cells by worsening others. The
high-load error is checkpoint-, operator-, and engine-path dependent.

### 3.2 Mixed-prefill interference is the leading 405B hypothesis

For 405B/H100/TP8, measured decode concurrency rises from a median near 7.5 at
rate 1, to 14.2 at rate 2, and 29.3 at rate 4. The twin predicts approximately
7.5, 13.3, and 26.1. Its faster service reduces predicted concurrency and
amplifies the original service-time error.

The strongest retrospective evidence is token latency conditioned on offered
prompt arrivals in the 100 ms before each ITL starts:

| arrivals in preceding 100 ms | measured mean ITL | predicted mean ITL |
|---|---:|---:|
| 0 | 36.47 ms | 35.92 ms |
| 1 | 56.98 ms | 40.84 ms |
| 2 or more | 80.83 ms | 48.31 ms |

Measured and predicted token clocks are classified independently, so this is
association evidence rather than a paired residual or proof that a prompt was
admitted in the same engine iteration.

At rate 4, median all-token ITL is close, 36.26 versus 35.92 ms, but the p95 is
103.23 versus 43.84 ms and the p99 is 171.23 versus 89.32 ms. The request-level
median-ITL fitting path discards exactly this mixed-iteration tail.

Preemption is not supported as the cause:

- legacy 405B rate-4 mean concurrency is about 31, below `max_num_seqs=64`;
- canonical 405B rates 1 and 2 record zero waiting and zero preemptions;
- cache use is only 2.2-3.1%;
- the current TP4 rate-4 state control completes 1,680 requests with zero
  waiting and zero preemptions.

The leading alternatives are both physical and falsifiable:

1. mixed decode and prefill execute as separate kernel/weight-sweep groups, but
   `iteration_work` currently charges one transformer weight sweep for the
   flattened mixed iteration; or
2. mixed/FP8 prefill uses a lower operator efficiency than pure decode.

For 405B, intervals preceded by one offered arrival average 20.51 ms above the
no-arrival class in measurement versus 4.92 ms in the model. That increment is
close to the scale of one calibrated FP8 transformer sweep, which motivates
but does not prove the phase/group-sweep hypothesis. It also matters to power:
the simulated ledger consumes the timing model's weight traffic, so a confirmed
omitted sweep would undercount both latency and HBM work.

The explicit BF16 vocabulary head and FP8 transformer accounting fixed earlier
errors, but they do not explain this arrival-conditioned tail. Do not change
the model from correlation alone; use the controlled probe in Section 6.

### 3.3 The 305-second power transition is separate

All twelve legacy 70B TP8 rate-4 traces change by roughly 12-19 W/GPU near
304.2-306.5 seconds. Energy can still be close while ACF-MAE remains about
0.30-0.37. The same rate-4 request marks on the modern H100 TP4 state control
have ACF-MAE 0.0166 and no upward transition.

The TP4 control records:

- 180 seconds of loaded idle near 119 W/GPU;
- P0 and a 700 W cap throughout;
- fixed 2,619 MHz memory clocks;
- no thermal or hardware slowdown;
- intermittent software power-cap events throughout load, not a new event at
  workload age 305 seconds;
- zero waiting and zero preemptions.

The seeded request marks also change composition near 305 seconds: output-token
influx rises about 22-29% while input-token influx falls. Exact changepoint
synchrony therefore does not uniquely imply temperature or server age. A
static batch term already failed because matched decode-batch ranges consume
12-15 W/GPU more after the transition.

## 4. Model appropriateness

### 4.1 Transferable skeleton to retain

The following structure is appropriate:

```text
marked arrivals
    -> manifest-bound discrete-event engine
    -> phase-resolved architecture/operator work
    -> equilibrium power surface conditioned on deployment idle/state
    -> causal state response, only when identified
    -> fixed meter response
```

It transfers arrival schedules through executed work rather than a rate label,
and transfers models through architecture descriptors rather than names. The
simulator accepts arbitrary marks, but empirical validation currently covers
only observed Poisson/Gamma/BurstGPT mark, context, and engine support.

### 4.2 Required timing change

Replace the flattened mixed-iteration closure with a source-supported,
phase-resolved form. The competing forms are:

- one transformer sweep per execution phase/group; or
- an operator-efficiency closure keyed by physical descriptors such as dtype
  recipe, batch, context, prompt/decode composition, and operator intensity.

Use profiler/counter evidence to choose. Do not let both forms absorb the same
residual. Engine version, quantization recipe, chunking policy,
`max_num_batched_tokens`, `max_num_seqs`, KV dtype, prefix-cache policy, and
scheduling policy are part of the deployment contract.

Validation must include held-out batch/context/composition levels and ITL
p50/p95/p99, not only request-level median ITL and E2E.

### 4.3 Required power boundary

The static surface is an appropriate equilibrium conditional mean under a
matched engine and power state. It is not sufficient for a hidden two-regime
TP8 trace.

Add a hardware-state response only when:

- its driver is observable or causally reconstructable before target power is
  read;
- it is trained without the target trace;
- it transfers across the TP8 diagnostic and TP4 negative control;
- it improves energy, ACF, and range error without harming ordinary traces.

Loaded idle remains a measured deployment boundary condition. Current data do
not establish zero-shot idle transfer across checkpoint, engine policy,
P-state, clocks, or caps.

### 4.4 Current unseen-model claim

The descriptor-based timing model is promising for unseen dense BF16
checkpoints inside measured operator support, and the A100 Qwen development
result passes. Unseen dense BF16 power is not generally passed: it still
requires a matched deployment idle/state, and H100 Qwen energy misses by
14.22%. Unseen FP8 transfer is also not established: 405B is the only FP8
calibration checkpoint. A second FP8 checkpoint is a later sealed test, not
part of the present rate-4 diagnosis.

## 5. Offline work before any GPU submission

### O1. Refit from existing controlled probes

Merge and grade the existing canonical decode staircase, context grid, and
mixed-grid bundles. The current calibration path excludes `mixed_grid` and the
request fit collapses mixed token tails to medians. The H100 TP8 mixed grid
reaches 114 cumulative preemptions; operator fitting may use only windows with
zero preemption increments, zero waiting, and nonbinding KV/seat capacity.
Confounded windows are scheduler diagnostics, not operator calibration.

Compare exactly two preregistered phase-resolved timing candidates:

1. explicit additional phase/group weight sweeps, with no fitted rate or model
   coefficient;
2. one operator slowdown
   `t_mix = t_base * (1 + beta_dtype * I[prefill>0, decode>0] * r)`, where
   `r = min(prefill_tokens / max(decode_tokens, 1), r_support)`,
   `beta_dtype >= 0` is the sole new coefficient per measured dtype family,
   and `r_support` is frozen to the source-training maximum.

Candidate 2 is selected source-only with complete batch/context/composition
levels held out. It may not extrapolate beyond `r_support`; unsupported points
fail closed. Profiler traffic distinguishes a time-only slowdown from candidate
1's additional timing and ledger weight work.

Hold out complete batch, context, and mixed-composition levels. Reject either
candidate if it:

- uses arrival rate or model identity;
- improves rate 4 by harming rates at or below 2;
- worsens the existing Qwen timing cells;
- fails to improve mixed-iteration ITL p95/p99;
- changes timing work without making the identical change in ledger work.

### O2. Re-score existing serving data

Report separately:

- dense 8B/70B;
- FP8 405B;
- MoE gpt-oss;
- pure decode versus mixed prompt/decode intervals;
- E2E, TTFT, decode duration, ITL p50/p95/p99, and concurrency error.

`T4` passes only when the source-only candidate:

- brings the 405B rate-4 median E2E error below 10% and improves signed bias;
- holds mixed-composition iteration-duration error to 10% and ITL p95/p99
  error to 20% on held-out levels;
- holds paired request/token ITL-class mean error to 15%, using either the
  controlled T4-A composition label or a measured-clock arrival class applied
  to the corresponding predicted request/token index;
- degrades pure-decode/low-rate median error by no more than one percentage
  point; and
- creates no wrong-sign correction for gpt-oss or 8B.

### O3. Enforce future identity

Future 405B bundles must fail validation unless they persist the corrected
component architecture, checkpoint fingerprint, quantization recipe, engine
version, chunking policy, and scheduler limits. Existing canonical 405B
manifests contain stale aggregate weight metadata and must not silently
override the corrected registry.

## 6. Minimal new data, in decision order

### T4-A. Controlled 405B mixed-prefill diagnostic

Run this only if O1 cannot distinguish the two timing candidates.

Use H100/TP8 and the corrected 405B checkpoint identity. Hold a fixed decode
cohort and context, then inject:

- no prompt;
- one prompt;
- two simultaneous prompts;
- prompt sizes 128, 512, and at least 1,024 tokens.

Before launch, persist one normalized schedule containing exact input token
IDs, output budgets, decode-cohort releases, prompt-injection epochs, engine
identity, and its SHA-256. Use five deterministic repeated injection cycles per
condition in a balanced fixed order. Estimate the incremental iteration-time
effect with cycle-block uncertainty. Permit one additional five-cycle block
only when the preregistered 95% block interval is wider than 20% of the observed
increment; otherwise do not repeat.

Record per iteration or at the finest supported interval:

- decode and prefill scheduled tokens;
- decode cohort, prefill groups, and tokens per iteration;
- iteration duration and kernel grouping;
- HBM traffic or profiler evidence for weight rereads;
- running/waiting requests, KV use, and preemptions;
- exact engine, quantization, scheduler, clock, and cap identity.

Mirror an existing BF16 Llama-70B mixed-grid schedule when possible. Do not
collect a new BF16 sweep unless the existing bundle cannot supply the matched
control.

Decision:

- an extra checkpoint sweep selects the phase/group work model;
- unchanged traffic with longer kernels selects an operator-efficiency model;
- waiting/admission divergence selects a manifest-bound scheduler change;
- no discriminating signal leaves the current FP8 arbitrary-arrival claim
  unsupported.

### T4-B. One current-stack 405B rate-4 validation

After freezing the T4 candidate, run
`profiling/campaigns/h100_405b_rate4_exact_replay.json`. It replays exact
input token IDs, release times, and planned output budgets from the normalized
`data/trace_plans/h100_405b_rate4_200.json` schedule, with exact-length
generation/ignore-EOS. Persist actual output tokens separately for scoring.
Include 60-120 seconds of measured loaded idle and full engine/state telemetry.

This run distinguishes a current-engine model failure from a legacy-engine
support boundary. It is not another rate sweep. Stop after one run unless a
five-block bootstrap interval for median E2E error is wider than two percentage
points or required telemetry is incomplete. Permit at most one exact-plan
repeat, and combine neither run until their manifest identities match.

### P4-A. Missing H100 TP8 state leg

Before launch:

1. persist the server-launch epoch in the run manifest in addition to
   instrumentation and workload epochs;
2. materialize a normalized direct-token plan from the TP4 artifact's relative
   release times, input lengths, and output budgets;
3. generate and persist deterministic token IDs of those exact lengths plus
   the normalized-plan hash.

The collected TP4 artifact did not persist prompt IDs/token IDs or a dataset
content hash. P4 can therefore match its recorded release/length marks but
cannot claim prompt-content identity retrospectively.

Collect the prepared
`profiling/campaigns/h100_tp8_state_diagnostic.json` H100 Llama-70B TP8
diagnostic. Its normalized input is
`data/trace_plans/h100_tp4_state_marks.json`:

| TP | pre-idle | workload | marks | telemetry |
|---:|---:|---:|---|---|
| 8 | 180 s | about 420 s at rate 4 | normalized TP4-derived release/input-length/output-budget plan | per-GPU power, temperature, SM/memory clocks, P-state, cap, throttle/clock-event reasons, full engine counters |

Do not repeat TP4.

Interpret the transition relative to telemetry start, persisted server-start
epoch, and workload start:

- telemetry age near 305 seconds, workload age near 125 seconds: an
  instrumentation/idle-window phase; call it server/runtime age only if the
  persisted server-start epoch supports that interpretation;
- workload age near 305 seconds: accumulated work or request composition;
- an observable clock/cap/temperature threshold: candidate hardware-state
  driver;
- no transition on the current stack: legacy engine/state support boundary.

If the transition remains tied to the same request-mark boundary with no
observable trigger, permit one later TP8 replay that keeps every arrival time
and interarrival interval fixed but cyclically permutes the
`(input_token_ids, output_budget)` mark tuples. Freeze a permutation
that moves the high-output cohort while keeping cumulative predicted work
within 2% of the original at the preregistered 240, 305, and 370 second test
ages. If no such permutation exists, the experiment remains ambiguous and is
not launched. Do not launch a rate/model cross.

P4-A is a diagnostic, not validation of a state law. A driver or threshold
selected from this sole telemetry-rich TP8 trace cannot transfer to the same
trace. If it identifies a modelable mechanism, freeze the law and grade it on
one later independently collected TP8 validation; otherwise record the support
boundary.

### L1. Deployment-calibrated idle and sensitivity boundaries

After T4/P4, run
`profiling/campaigns/h100_qwen3_8b_idle_decomposition.json` for H100 Qwen
rate 4 with:

- the same 200 request marks;
- a measured 60-second loaded-idle window;
- engine policy and full power-state telemetry persisted in the manifest.

Apply the measured idle delta without changing dynamic coefficients. Report
both total energy and idle-subtracted incremental active energy, plus
active-window residual bias/shape. Pass only when total and incremental active
energy errors are each at most 6% and active-window residuals meet the existing
dense shape/bias gate. This prevents a target-idle offset from hiding an
incorrect dynamic surface. On failure, retain a deployment-state support
boundary.

Only after that result, use cheap idle-only, one-factor contrasts if the
deployment claim requires them:

1. hold checkpoint and power state fixed, change engine policy once;
2. hold checkpoint and engine policy fixed, change one verified clock/cap
   state once.

Run each permitted one-factor contrast as same-node A-B-A, with the two A
segments providing a repeated baseline. Before opening B, freeze a
repeatability tolerance from the A segments and existing same-state idle
windows, using the larger of the meter tolerance and a 95% block-mean
uncertainty interval. A material contrast makes measured calibration or
explicit state conditioning mandatory. A null contrast only fails to reject
invariance; it does not establish transfer. Do not spend a
checkpoint-by-policy-by-state crossed design. True zero-shot checkpoint idle
transfer remains deferred with that crossed design.

### C1. Deterministic cache replay identity

The collected TraceLab pair has all 136 keyed rows but differs in 10 prompt
hashes and 34 output hashes. It cannot estimate a cache treatment effect.

Before another GPU pair:

1. pass the CPU plan/hash comparator;
2. pass a short forced-token smoke replay with identical prompt and output
   hashes across cache regimes;
3. persist the normalized-plan hash, tokenizer/checkpoint identity, forced
   output token IDs, and cache policy;
4. require exact keyed identity before scoring power.

Only then rerun cache-off/cache-on. Existing legs remain separate model-error
reports. This paired rerun is deferred until T4/P4 and the identity smoke gate
pass; it is not part of a broader crossed design.

C1 establishes agentic cache/power transfer only if the identity-valid pair
also meets all of:

- median timing error at most 10%;
- exact executed-prefill and cache-hit accounting within one cache block per
  round;
- total and idle-subtracted incremental energy error at most 6%;
- the existing dense temporal gate when the active trace duration supports its
  declared lag.

### U1. Later unseen-FP8 sealed test

After T4 passes on 405B, select one second FP8 checkpoint with a materially
different vocabulary/head fraction or quantization recipe. Freeze the
descriptor-based model before opening:

- one pure-decode point;
- one pure-prefill point;
- one mixed prompt/decode point;
- one serving validation.

This phase alone can establish unseen-FP8 transfer. It is deferred and has no
current budget authorization.

## 7. Agentic trace status

Current evidence supports these limited statements:

- BurstGPT timing transfers well at 2.06% median E2E error; energy error is
  7.89%, above the 6% target.
- TraceLab cache-off timing passes at 3.45%, but energy is underpredicted by
  25.75%.
- TraceLab cache-on energy is close at 2.27% underprediction, but timing fails
  at 15.55%.
- The two TraceLab legs cannot be subtracted because request content diverged.

Therefore agentic timing transfer is promising but not generally passed, and
agentic power transfer has not passed. Prefix-cache causal accuracy remains
ungraded until C1 produces an identity-valid pair.

Agentic evaluation must continue to report:

- exact row/session/turn conservation;
- prompt/output/cache hash identity;
- E2E and TTFT;
- executed prefill and cache-hit accounting;
- energy;
- ACF only when trace duration supports the requested lag;
- sparse-gap idle residuals separately from active-work residuals.

## 8. Reproducing the rate-4 timing evidence

Run:

```bash
uv run python timing-test/rate4_diagnostic.py \
  --fitted /tmp/powertrace_fitted_efficiencies_v3.json
```

The command emits input SHA-256 hashes, per-run measured/predicted decode
concurrency, E2E bias, ITL p50/p95/p99, and the explicitly labeled
arrival-conditioned association diagnostic. It also emits hashes for the
diagnostic, simulator, timing physics, evaluator, and architecture registry,
plus the explicit engine configuration. The exact values in this plan use
dataset hash
`e5696d83d618eba472c9907d9c507120548f52af1970c568a0c2f312a3f617a2`
and v3 fitted-candidate hash
`47c0dd752f1787f59b0686fd0fda6001469fc8bfc019799af12532414f695c32`.
The emitted code/config identity is also part of the evidence key. The `/tmp`
candidate must be frozen to a versioned artifact before these development
numbers are cited as a released result.

## 9. Budget and stopping rules

Approximate workload-only ceilings, excluding model load:

| stage | maximum new cost | authorization |
|---|---:|---|
| O1-O3 | CPU only | now |
| T4-A controlled 405B mixed probe | about 0.7 H100 GPU-hours | conditional on O1 ambiguity |
| T4-B 405B rate-4 validation | about 0.3 H100 GPU-hours | after T4 candidate freeze |
| P4-A missing TP8 state leg | about 1.33 H100 GPU-hours | pending diagnostic |
| later frozen P4 validation | not yet budgeted | only if P4-A identifies a causal law |
| L1 H100 Qwen idle-anchored rerun | less than 0.1 H100 GPU-hours | after T4/P4 |
| C1 cache pair | capped separately | deferred behind identity gate |
| U1 second FP8 checkpoint | not budgeted | deferred |

No job has been launched by this plan update.

Stop rules:

- no generic rate sweep;
- no checkpoint ladder;
- no engine-policy-by-power-state cross;
- no extra TP4 state run;
- no fitted state term without a causal trigger;
- no cache treatment claim without exact replay identity;
- no unseen-FP8 claim from 405B alone;
- no more data when an explicit support boundary is the honest result.

## 10. Claim-to-evidence matrix

| claim | required evidence |
|---|---|
| support-bounded arbitrary-arrival timing, dense BF16 | Poisson/Gamma/BurstGPT data inside observed mark/context/engine support plus T4 held-composition gate |
| arbitrary/sparse-arrival power | L1 dynamic/idle-separated gate plus an identity-valid long sparse replay |
| large-checkpoint high-load timing | O1/O2, then T4-A/T4-B only as triggered |
| FP8 accounting for 405B | corrected component identity plus T4 mixed-phase evidence |
| unseen FP8 model transfer | deferred U1 sealed checkpoint |
| 70B TP8 state mechanism/support boundary | P4-A and TP4 negative control |
| high-fidelity long-horizon 70B TP8 power | a frozen causal law on a later independent TP8 validation |
| deployment-calibrated loaded idle | measured target idle plus L1 dynamic/idle-separated gate |
| zero-shot loaded-idle transfer | deferred checkpoint-by-policy-by-state evidence |
| deterministic prefix-cache effect | identity-valid C1 pair |
| agentic arrival timing | separate BurstGPT and valid TraceLab leg scores |
| agentic cache/power transfer | identity-valid C1 pair that also passes timing, cache-accounting, energy, and duration-eligible temporal gates |
| Gemma cross-family MoE | router-law/freeze-order contract before the collected bundle is opened for the claim |

The model is currently useful inside stated support, but the general claim is
conditional on `T4`, `P4`, and deterministic replay identity. The campaign
adds data only where an existing model comparison cannot decide the mechanism.
