# FEATURE_TEST_PLAN: architecture-aware power model selection

Status: execution plan, audited against the repository and available legacy data
on 2026-07-10.

This plan answers one decision:

> What is the smallest model that preserves energy and trace dynamics while
> transferring to a larger related architecture and an unseen tensor-parallel
> setting on the same hardware generation?

The expected answer is a hybrid, not another end-to-end sequence model:

```text
request (arrival, input tokens, output tokens)
    -> scheduler or measured execution timing
    -> executed-work ledger + A_t and delta A_t
    -> first-principles mean power for one hardware generation
    -> optional small causal correction
    -> node power trace
```

The first-principles layer is responsible for size and parallelism transfer. A
learned correction is responsible only for repeatable short-term error. A100
and H100 are fitted and evaluated separately. No A100-to-H100 transfer claim is
needed or allowed.

## 1. Decision and non-goals

### The model must do all of these

1. Keep full-run energy error low.
2. Preserve temporal structure, measured by ACF R2 and ACF MAE.
3. Compute FLOPs, weight traffic, KV traffic, and communication from request,
   architecture, and parallelism descriptors rather than memorizing a model ID.
4. Transfer within a hardware generation:
   - `gpt-oss-20b -> gpt-oss-120b` on A100;
   - Llama 8B -> 70B and Llama 8B+70B -> 405B on H100;
   - seen tensor-parallel degrees -> an unseen tensor-parallel degree.
5. Use joint `(input_tokens, output_tokens)` request marks and retain `A_t` and
   `delta_A_t` as explicit, inspectable inputs.
6. Produce one small artifact per hardware generation, not one checkpoint per
   model and tensor-parallel configuration.
7. Run fast enough to be used repeatedly inside a Monte Carlo load-duration
   curve simulation.

### This test does not try to do these things

- transfer coefficients between A100 and H100;
- prove pipeline, expert, data, or context-parallel accuracy from the current
  data, which contain tensor parallelism only;
- validate an overload scheduler from traces that do not record enough engine
  and queue state;
- make cache-on or long-context claims from short-context ShareGPT traffic;
- build a general neural architecture search system;
- add stochastic noise before deterministic energy and dynamics pass;
- retain a model merely because its pooled average looks good while a key
  transfer cell fails.

## 2. What the current data can establish

The strict legacy rebuild currently accepts 450 runs: 25 configurations, six
request rates `{0.125, 0.25, 0.5, 1, 2, 4}` requests/s, and three runs per
configuration/rate cell. At 1 s it contains 273,598 valid bins, approximately
76 hours. A100 contributes 131,395 bins and H100 contributes 142,203 bins.

The current data are sufficient for:

- measured-timing mean-power fitting;
- same-configuration held-run evaluation;
- retrospective tensor-parallel holdouts;
- retrospective A100 `gpt-oss-20b -> gpt-oss-120b` transfer;
- retrospective H100 Llama 8B -> 70B -> 405B transfer;
- choosing between physics-only and physics plus a small causal correction.

The current data are not sufficient for:

- a true arrival-only overload/scheduler test;
- long-context prefill or KV-cache identification;
- cache hit, eviction, or preemption modeling;
- measured expert, pipeline, data, or context-parallel transfer;
- a sealed external validation claim. The legacy targets have already been
  inspected and must be called retrospective development evidence.

Four Llama-3-8B A100 configurations lack request timestamps and are excluded
from the strict work ledger. The old GMM datasets and checked physics outputs
also predate the repaired lineage contract. They may orient the work but are
not frozen evidence until regenerated.

## 3. What we are trying to beat

Every method must be rerun through one evaluation harness on the same split and
same 1 s comparison trace. The baselines are:

| ID | Baseline | Why it is required |
|---|---|---|
| B0 | Per-hardware training mean | Detects metrics that reward a flat trace. |
| B1 | Ridge on `A_t` and `delta_A_t` histories | Tests how much the old two features can do without a GRU. |
| B2 | Current K10-F2 GMM+BiGRU | The learned system being replaced; retrain it only on the frozen split. |
| B3 | Current 11-term physics model | The transfer baseline and minimum viable deployable model. |
| B4 | Same-configuration physics fit | An oracle upper reference; never eligible for a transfer claim. |

B2 must use the same requests, timing mode, power bins, warm-up, horizon, and
held-out runs as the candidates. Its existing per-configuration checkpoints do
not count as transfer. It is an in-configuration fidelity baseline only.

### Orientation numbers from the current audit

These numbers came from a strict temporary rebuild and exploratory fits. They
must be reproduced by the permanent harness before they become a baseline.

| Test | Physics | Physics + exploratory causal linear correction |
|---|---:|---:|
| A100 held replicate: energy / ACF R2 / NRMSE | 4.13% / 0.760 / 0.150 | 4.92% / 0.912 / 0.136 |
| H100 held replicate: energy / ACF R2 / NRMSE | 6.46% / 0.654 / 0.176 | 4.94% / 0.864 / 0.148 |
| A100 gpt-oss 20B -> 120B | 5.48% / 0.761 / 0.142 | 4.70% / 0.920 / 0.110 |
| H100 Llama 8B+70B -> 405B | 9.01% / -2.673 / 0.181 | 13.88% / 0.782 / 0.334 |

The last row is the key failure to fix. A correction that raises ACF R2 while
worsening 405B energy and NRMSE is not a successful transferable model.

The current strict physics model also has same-distribution run-CV bin-power R2
of about 0.960 on A100 and 0.956 on H100, with RMSE of about 123 W and 218 W.
Those pooled scores do not replace the run-level transfer gates below.

## 4. Frozen data contract

### Time grid and alignment

- Build one native ledger in 250 ms half-open bins `[t, t+dt)` and label the
  value at `t+dt`.
- Aggregate four consecutive native bins by their mean for the primary 1 s
  comparison. Do not refit or realign after aggregation.
- Never shift a prediction using measured power.
- Never initialize from the measured first power value.
- Reset all deltas and causal histories at a run boundary.
- Fit meter lag on training runs only and express all lag taps in seconds.
- Do not run coefficients fitted at 1 s four times at 250 ms and call that a
  250 ms model.

### Request and state fields

Preserve each request as:

```text
(request_id, session_id, arrival_time, input_tokens, output_tokens)
```

`session_id` is optional in legacy data but required when sessions exist so all
turns can remain in one split. Input and output lengths stay paired; they are
never independently shuffled or sampled.

For every time bin store these separate concepts:

| Group | Required fields |
|---|---|
| Offered demand | arrivals, input tokens arriving, output tokens requested, unfinished backlog |
| Request state | `A_t`, `delta_A_t`, running requests, waiting requests |
| Executed work | prefill tokens/s, decode tokens/s, prefill iterations/s, decode iterations/s |
| Context state | context-weighted decode tokens, KV bytes read/s, KV bytes written/s, KV occupancy when available |
| Physical work | FLOPs/s, weight bytes/s, communication bytes/s, compute utilization, memory utilization |
| Target | measured node GPU power in watts |

Freeze these definitions:

```text
A_t       = number of unfinished requests at the end of bin t
          = waiting requests + running requests
delta A_t = A_t - A_(t-1), with delta A_0 = 0
```

`A_t` is not batch size. Running and waiting are retained separately so a queue
can grow without being mistaken for GPU execution. For a retrospective legacy
run, completion comes from measured request timing. For arrival-only rollout,
completion must come from the scheduler being tested.

### Architecture descriptor

The work calculator consumes values, not model-name one-hot features:

- layer count, hidden size, intermediate size;
- query heads, KV heads, and head dimension;
- total and active parameter counts;
- weight and KV element byte widths;
- dense versus MoE structure;
- total, routed, shared, and active experts when applicable;
- sliding-window/full-attention layer counts and window length;
- quantization/compute dtype;
- the measured or declared maximum context.

For gpt-oss, the descriptor must reflect that the 20B and 120B models change
total parameters, active parameters, layers, and expert count. No `is_120b`
feature is allowed.

### Parallelism descriptor

Use one explicit structure:

```text
ParallelPlan(tp, pp=1, ep=1, dp=1, cp=1, world_size, topology)
```

Only `tp` is empirically validated now. The other fields keep the programming
model extensible but must return an explicit `unsupported_by_evidence` status
until measured data exist. Do not learn arbitrary integer embeddings for these
fields.

Each fit also names an exact `HardwareProfile`: GPU SKU and count, peak compute
rate for each used dtype, HBM capacity and bandwidth, link bandwidth/topology,
and sustained power limit. These are fixed scales, not evidence that A100 and
H100 share coefficients.

## 5. First-principles work calculation

There must be one implementation of request-to-work arithmetic, shared by
training and rollout. Extend `model/training_data/ledger_view.py`; do not create
a second feature calculator in `feature-test/`.

For each bin calculate these rates:

```text
F_pre, F_dec       floating-point operations/s
B_weight           model-weight bytes read/s
B_kv_read          KV-cache bytes read/s
B_kv_write         KV-cache bytes written/s
C_tp               tensor-parallel communication bytes/s
C_pp, C_ep, C_cp   future communication terms, unsupported until measured
```

The first implementation should retain the current audited equations and add
hand-worked tests around them:

- projection and MLP FLOPs are two operations per used matrix weight per token;
- attention score/value FLOPs for `q` query tokens are
  `4 * query_heads * head_dim * q * effective_context`, where
  `effective_context` is the average number of keys visible to those causal
  queries; decode is the `q=1` case;
- one token's KV write is
  `2 * layers * kv_heads * head_dim * bytes_per_element`;
- softmax KV reads grow with effective context, respecting sliding-window and
  full-attention layer counts;
- dense weight traffic uses all dense weights required by an iteration;
- MoE weight traffic uses dense/shared weights plus the unique routed experts
  touched by that iteration;
- when router assignments are absent, expected unique-expert traffic is an
  explicit assumption and not a measurement;
- tensor-parallel ring traffic includes the `2*(tp-1)/tp` collective factor,
  activation width, collective count, and layer count;
- work is divided or replicated according to `ParallelPlan`, never according
  to a fitted TP label.

Every rate must conserve the source request tokens. Synthetic tests cover one
dense request, one MoE request, a mixed prefill/decode bin, a long-context
decode, TP1 and TP>1, an off-grid arrival, and an idle bin.

## 6. Candidate model ladder

Run the ladder in order. Stop at the smallest passing model.

### M0 - Existing deterministic physics mean

Use one coefficient vector per hardware generation:

```text
P_phys(t) = P_idle
          + f_F(F_pre, F_dec; hardware)
          + f_B(B_weight, B_kv_read, B_kv_write; hardware)
          + f_C(C_tp; hardware)
          + P_busy

P0(t) = Cap_hardware(Lag_hardware(P_phys(t)))
```

Start with the maintained 11-term non-negative basis in
`model/classifiers/physics.py`. Fit A100 and H100 independently. For a transfer
score:

- no target power, target cap, target lag, or target residual may enter fitting;
- a `conditional-timing` test may use target measured timing and target prefill
  throughput only to place executed work; an `arrival-only` transfer may not;
- no target-family multiplier is allowed;
- unknown families receive multiplier 1.0;
- report any coefficient that is prior-dominated or physically implausible.

Prune collinear saturation terms when removing them does not hurt the
development holdout. Do not add more physical terms merely to improve the
training score.

### M0b - Monotone resource curves, only if M0's mean fails

This is the one allowed alternative mean formulation. Keep the same calculated
work, but replace overlapping saturation bases with three small, monotone
piecewise-linear response curves:

```text
P_phys(t) = P_idle + g_F(u_compute) + g_B(u_memory) + g_C(u_communication)
            + P_busy
```

Use fixed utilization knots `{0, 0.05, 0.15, 0.40, 1.0, 1.5}` and non-negative
segment slopes. Fit one set per hardware. There are no trees, model IDs, or
interactions. Run M0b only if M0 misses a mean/transfer energy gate or retains
physically conflicting, prior-dominated saturation coefficients. If M0 and M0b
are tied on held data, keep M0.

### M0c - Phase-anchored concave mean (added 2026-07-11 after the v1 failure analysis)

Same calculated work and knot grid as M0b, with three declared changes, each
justified in `FEATURE_TEST_LEARNINGS.md`:

- the response in `u_compute` and `u_memory` is a non-negative sum of
  saturating ramps `min(u, knot)` (concave by construction; the source data
  cannot distinguish shapes, and the unconstrained shapes are NNLS exchange
  artifacts), with no communication column (collinear with compute at
  r ~ 0.99 on every source fit, always fitted to zero);
- fitting is staged: prefill-influence-free bins identify the floors and the
  memory response, prefill-influenced bins identify the compute response from
  the residual, and a final pass refits the non-compute columns on all
  training bins with the compute response frozen (declared assumption with
  published support: decode is memory-bound, prefill is compute-bound);
- the cap is the hardware board power limit and the meter lag is the
  step-identified kernel from `feature-test/meter_kernel.json`, so neither is
  a fitted constant.

M0c carries no request-state columns; `A_t` enters only through the M1-M3
residual ladder. Like every candidate it must earn selection on source
development alone.

### M1 - Physics plus `A_t` dynamics

Fit a causal ridge correction to the selected M0/M0b residual using only:

```text
log1p(A_t), delta A_t
```

Use physical-time taps `{0, 1, 2, 4, 8}` seconds. This is the direct, cheap test
of whether the old GMM+BiGRU inputs need a recurrent neural network.

### M2 - Add offered request marks

Add these per-bin channels while retaining input/output pairs in the ledger:

```text
log1p(arrivals/s)
log1p(input tokens arriving/s)
log1p(output tokens requested/s)
```

M2 tests whether request marks add information beyond `A_t` without using
executed work.

### M3 - Add normalized executed work

Add:

```text
u_compute = (F_pre + F_dec) / hardware_compute_capacity
u_memory  = (B_weight + B_kv_read + B_kv_write) / hardware_memory_bandwidth
busy
prefill_share = F_pre / max(F_pre + F_dec, epsilon)
```

M3 is the preferred final candidate. With nine channels and five time taps, it
has 45 residual coefficients per hardware. All normalization is fitted on the
training partition only.

The correction is:

```text
P_hat(t) = Cap_hardware(Lag_hardware(P_phys(t)) + beta_hardware^T H_t)
```

The filter is causal. It has no model ID, family ID, TP-specific coefficient,
or bidirectional/future input. Level-channel filter weights must have zero DC
gain, and the correction has no free intercept, so the physics layer owns
steady-state mean power. Also report run-energy drift before the final cap.

Fit ridge strengths `{0.001, 0.01, 0.1, 1, 10}` on development runs only. Use
the smallest strength within one standard error of the best development score.
The selection score is lexicographic: first pass energy, then maximize ACF R2,
then minimize NRMSE.

### M4 - Add queue or context state only if an ablation justifies it

Candidate additions are waiting/running split, context-weighted decode, and KV
occupancy. Add one group at a time. Retain a group only if it improves at least
two important development cells and does not make any transfer gate worse.

### M5 - Tiny causal temporal convolution only after M3/M4 fail

This is a stop-controlled fallback, not part of the first implementation. It
may be tried only if the linear correction passes energy but misses the ACF gate
in at least two predeclared cells. Limit it to three causal convolution layers,
eight hidden channels, kernel width three, dilations `{1, 2, 4}`, and fewer than
5,000 learned parameters per hardware. It must beat M3 on untouched test runs,
not merely on development runs.

### Optional stochastic layer

Do not fit it during deterministic model selection. If M0-M4 pass, compare a
training-only block residual sampler with noise-off. It must preserve mean-zero
residuals, energy, cap ordering, marginal power, and ACF under explicit seeds.
A global AR(1) residual is not a candidate: exploratory tests lowered ACF R2 and
worsened energy on both hardware groups.

## 7. Exact split matrix

All source IDs are sorted before assigning repeats. For S0, repeat 0 is
training, repeat 1 is development, and repeat 2 is untouched test. For a
transfer split, fit and select using source-family runs only, freeze the model,
then score every run in the held target family or TP. Target data never select
a feature, coefficient, lag, cap, normalization, or hyperparameter.

With the legacy ledger, S1-S3 are `conditional-timing transfer`: target
measured request timing and prefill throughput place executed work, but target
power never fits the model. Repeat S1-S3 through the arrival-only scheduler
only after Section 13 passes.

### S0 - Same-configuration held repeat

Use every strict configuration/rate cell. Fit hardware coefficients on repeat
0, choose the candidate and ridge strength on repeat 1, refit the frozen
candidate on repeats 0 and 1, and report repeat 2. This tests data efficiency
and in-configuration trace fidelity.

### S1 - MoE size and TP transfer on A100

```text
source: gpt-oss-20b, TP1 and TP2, every rate, all repeats
target: gpt-oss-120b, TP4 and TP8, every rate, all repeats
```

Select the candidate and hyperparameters with source repeats 0 and 1, freeze
them, then refit on all 20B source runs before scoring all 120B target runs. Do
not use any 120B power, cap, lag, residual, normalization, or family multiplier
during fitting. Because size, expert count, and TP all change at once, call this
a joint family-scale/TP transfer test, not isolated expert-count causality.

### S2 - Dense scale transfer on H100

Run both directions in sequence:

```text
S2a train: Llama-3-8B, TP1/2/4/8
    test:  Llama-3-70B, TP4/8

S2b train: Llama-3-8B and Llama-3-70B
    test:  Llama-3-405B, TP8
```

Use every rate. For each step, select with source repeats 0 and 1, freeze the
candidate, refit it on all source runs, and score all target runs. Keep 405B
quantization in the architecture descriptor. Do not learn a special 405B/FP8
multiplier from its power.

### S3 - Tensor-parallel transfer

For each hardware, leave one TP value out entirely, fit on all other available
TP values, and test every model/rate cell at the held TP. Report interpolation
and extrapolation separately. A TP split is valid only when at least two other
TP values remain in training. Select with source-TP repeats 0 and 1, freeze the
candidate, refit on all source-TP runs, and score every held-TP run.

Also report focused H100 Llama-8B tests:

```text
train TP1/TP2/TP4 -> test TP8
train TP1/TP2/TP8 -> test TP4
```

### S4 - Workload and session transfer

The legacy ShareGPT runs support only a request-level repeat check. When
canonical sessions arrive, keep all turns in one split and hold out complete
sessions. Long-context, agentic, and overload workloads remain separate named
tests; never pool them into S0.

### Split leakage checks

For every split, assert that the test source hashes do not appear in:

- coefficient, residual, normalization, cap, or lag fitting;
- throughput or efficiency calibration for a claimed arrival-only transfer;
- hyperparameter selection;
- early stopping;
- family multipliers;
- support thresholds.

Measured target timing may be used only in a result labeled
`conditional-timing transfer`, never `zero-shot arrival-only transfer`.

## 8. Metrics: exact definitions

Compute metrics per run first. Summarize run values by median, P90, worst, and
the number of failures. Never select from a pooled-bin score alone.

### Primary metrics

Full-run absolute energy error:

```text
100 * abs(sum_t P_hat(t)*dt - sum_t P(t)*dt) / sum_t P(t)*dt
```

ACF R2:

- first aggregate native predictions and targets to 1 s;
- compute each trace's ACF after subtracting its own mean;
- compare lags 1 through 60 seconds;
- use `1 - SSE/TSS` without clipping negative values.

ACF MAE:

```text
mean_lag(abs(ACF_hat(lag) - ACF_true(lag))) for lags 1..60 s
```

ACF MAE is mandatory because ACF R2 is unstable when the measured ACF is nearly
flat. A model does not pass temporal fidelity by exploiting that instability.

NRMSE is reported in both forms:

```text
RMSE / (max measured power - min measured power)  # compatibility
RMSE / mean measured power                        # stable interpretation
```

### Secondary metrics

- signed run-mean bias;
- bin-power R2 and RMSE, by hardware;
- absolute energy error in non-overlapping 1 s, 5 s, and 30 s windows;
- P95 and P99 power error;
- 1 s and 250 ms P95 absolute ramp error and maximum up/down ramp error;
- node load-duration values at exceedance fractions 0.01, 0.05, and 0.50;
- cap-hit fraction, out-of-support fraction, skipped runs, and failed runs.

Stratify every result by hardware, split, family, TP, request rate, and
low/middle/high measured load. Include paired bootstrap intervals over runs
with a fixed seed. Bins from one run are not independent bootstrap samples.

## 9. Pass/fail gates

The final choice is the smallest model that passes every hard gate. If no model
passes, report the failure and collect the targeted data in Section 12; do not
relax a gate after seeing the test set.

### G0 - Integrity

- Exact input hashes, split membership, code revision, dirty status, units, and
  `dt` are recorded.
- Token conservation and hand-worked physical calculations pass.
- Training and rollout produce identical ledger state for the same synthetic
  schedule.
- Test data are absent from every fitted artifact.
- There is no measured-power alignment or measured-power initialization.

### G1 - Same-configuration fidelity (S0)

On each hardware independently:

| Metric | Required test result |
|---|---:|
| Full-run energy error | median <= 5%, P90 <= 10%, worst <= 15% |
| ACF R2 | median >= 0.85 |
| ACF MAE | P90 <= 0.10 |
| Range NRMSE | median <= 0.15 |
| Failures | 0 unexplained; all support exclusions reported |

Against the retrained GMM+BiGRU, the selected model may be at most 1 percentage
point worse in median energy, 0.05 worse in median ACF R2, and 0.02 worse in
median range NRMSE. It must use one artifact per hardware and no per-config
training.

### G2 - gpt-oss 20B -> 120B (S1)

This gate explicitly improves the current orientation result:

| Metric | Current physics orientation | Required test result |
|---|---:|---:|
| Energy error | median 5.48%, P90 18.96%, worst 19.5% | median <= 5%, P90 <= 15%, worst <= 20% |
| ACF R2 | median 0.761 | median >= 0.80 |
| Range NRMSE | 0.142 | median <= 0.13 |

### G3 - Llama scale transfer (S2)

For 8B -> 70B, require median energy <= 5%, P90 <= 15%, worst <= 20%, median
ACF R2 >= 0.75, ACF-MAE P90 <= 0.12, and median range NRMSE <= 0.17.

For 8B+70B -> 405B, improve the current physics orientation:

| Metric | Current physics orientation | Required test result |
|---|---:|---:|
| Energy error | median 9.01% | median <= 7.5%, P90 <= 15%, worst <= 20% |
| ACF R2 | median -2.673 | median >= 0.75 |
| ACF MAE | not yet frozen | P90 <= 0.12 |
| Range NRMSE | 0.181 | median <= 0.17 |

### G4 - Tensor-parallel transfer (S3)

For every eligible held TP: median energy <= 7.5%, P90 <= 15%, median ACF R2
>= 0.75, ACF-MAE P90 <= 0.12, and median range NRMSE <= 0.17. Also require no
systematic signed bias greater than 10% at either the lowest or highest request
rate.

### G5 - Learned-correction safety

A correction is enabled for a split only if, relative to physics-only on the
same runs:

- median energy worsens by no more than 0.5 percentage point;
- P90 energy worsens by no more than 1 percentage point;
- median ACF R2 improves by at least 0.05 or ACF MAE improves by at least 20%;
- range NRMSE does not worsen;
- cap-hit rate does not hide an uncapped error increase.

If the correction passes S0 and S1 but fails S2b, ship physics-only for S2b.
Do not describe the residual as transferable across all architectures.

### G6 - Simplicity and data efficiency

- At most one physics artifact and one optional residual vector per hardware.
- At most 80 learned scalars per hardware for M0-M4 combined.
- Artifact size <= 1 MB per hardware.
- CPU-only fitting and inference are required.
- Repeating the fit with the same inputs is deterministic within declared
  floating-point tolerance.
- Fitting one run per configuration/rate cell stays within 2 percentage points
  of full-data median energy and within 0.05 ACF R2. If it does, do not collect
  more random repeats.
- Report fit time, peak RAM, artifact bytes, and predicted bins/s. The selected
  path must be at least 10x smaller and 10x faster on CPU than GMM+BiGRU, using
  the same machine and batch of traces.

## 10. Selection rule

Use this order; do not combine metrics into an opaque weighted score.

1. Reject any model that fails integrity, energy, failure, or correction-safety
   gates.
2. Among remaining models, reject any that fail the named transfer gate for
   the role in which they would be used.
3. Prefer higher ACF R2 and lower ACF MAE.
4. Prefer lower NRMSE.
5. If paired bootstrap intervals overlap on all primary metrics, choose the
   model with fewer learned parameters and features.

The likely deployable result is allowed to have two modes:

```text
transfer mode:  physics only
in-domain mode: physics + validated causal correction
```

That is preferable to forcing one learned correction onto an unsupported
architecture.

## 11. Exact execution order and artifacts

### Phase A - Build the frozen harness

1. Rebuild Stage0 and the strict 250 ms ledger from current source data.
2. Write a split manifest containing source IDs, hashes, roles, and exclusion
   reasons.
3. Implement one metric path used by every baseline and candidate.
4. Add leakage assertions and synthetic conservation/alignment tests.
5. Regenerate B0-B3 before fitting a new correction.

Existing producer commands begin with:

```bash
uv run -m model.scripts.stage0_inventory --data_root_dir data
uv run python feature-test/build_ledger_cache.py \
  --pair-manifest-csv results/stage0/pair_manifest.csv \
  --throughput-db model/throughput_database.json \
  --dt 0.25 \
  --out feature-test/ledger_cache_250ms.npz \
  --run-index-out feature-test/ledger_cache_250ms.runs.json
```

The permanent evaluator should be one thin CLI, not one script per experiment.
Its planned interface is:

```bash
uv run python feature-test/evaluate_candidates.py \
  --ledger-cache feature-test/ledger_cache_250ms.npz \
  --run-index feature-test/ledger_cache_250ms.runs.json \
  --out-dir results/feature_test_v1
```

Do not add CLI flags for every model choice. The frozen ladder and splits belong
in versioned code/data, and the output records them.

### Phase B - Run the feature ladder

Run B0-B4 and M0-M3. Run M0b only when its declared trigger fires. Produce one
row per run and one summary row per
`(candidate, hardware, split, family, TP, rate)`.

Only if M3 fails its predeclared condition, run M4. Only if the M5 trigger is
met, implement and run the tiny causal convolution.

### Phase C - Run transfer and efficiency tests

Run S1, S2, and S3 without refitting on targets. Then run the one-repeat data
efficiency test and the CPU throughput benchmark. Inspect individual worst
runs, not only the summary.

### Required result files

```text
results/feature_test_v2/   # current; results/feature_test_v1/ is the
                           # pre-correction snapshot
  split_manifest.json
  candidate_manifest.json
  per_run_metrics.csv
  aggregate_metrics.csv
  transfer_scorecard.csv
  model_complexity.csv
  support_failures.csv
  selected_model.json
  README.md
```

`selected_model.json` records the chosen role for physics and residual, all
coefficients, architecture schema, supported hardware/parallelism, cap, lag,
training hashes, code revision, and every failed or unsupported gate.

### Phase D - Verification

After each code change:

```bash
uv run -m pytest -x
```

Before accepting the result, rerun the evaluator from a clean output directory
and verify deterministic artifacts and the primary scorecard. Unit tests alone
are not completion.

## 12. Data to collect only if a named gate fails

Do not collect more random ShareGPT repeats first. Existing learning-curve work
shows load/architecture coverage is more valuable than repeats.

| Failure | Smallest useful new measurement |
|---|---|
| Prefill coefficient or peak error | Fixed prefill staircase at several prompt lengths, no decode. |
| KV/context transfer error | Fixed-batch decode at several context lengths. |
| TP transfer error | Same model, request trace, GPU count policy, and clocks at three TP values. |
| Arrival-only timing error | Below-knee, near-knee, and overload runs with engine queue/running/KV state. |
| MoE expert-traffic error | Router/expert counters plus fixed expert-parallel settings. |
| PP/EP/CP claim desired | Hold model, workload, world size, and hardware fixed; vary one parallel axis at a time with at least three settings. |
| Cache-on claim desired | Prefix reuse, cache hit/miss, eviction, and preemption counters. |

Parallelism accuracy is admitted one axis at a time. Synthetic arithmetic tests
can validate conservation for PP/EP/CP, but only measured holdouts can validate
power accuracy. Until then the artifact must say `TP only`.

## 13. Arrival-only timing is a separate gate

Power-model selection first uses measured execution timing so scheduler error
does not contaminate the power test. Deployment from requests alone then needs
a scheduler that turns offered work into executed work:

```text
service_time = launch_overhead
             + max(F / (eta_F * peak_F),
                   B / (eta_B * peak_B),
                   C / (eta_C * peak_C))
```

Fit efficiencies per hardware generation, not per model. The scheduler must
carry continuous batching, waiting work, running work, backlog, KV capacity,
and unfinished work at the horizon. It may not drop or force-finish overload.

Validate below-knee, near-knee, and overload regimes using TTFT, decode time,
throughput, completion rate, backlog, and SLO metrics. Until this passes, call
the main scale tests `conditional-timing transfer`, and use request-only output
only for sensitivity analysis.

For offline Monte Carlo, requested output length may be sampled with input
length as a joint mark before scheduling. Do not imply that final output length
is known to an online serving system.

## 14. YAGNI stop conditions

Stop and ship M0 when physics alone passes the intended transfer role.

Stop and ship M3 for in-domain traces when it passes and its gain over M1/M2 is
real. Do not add queue/context fields merely because they are available.

Do not build:

- another GMM;
- another bidirectional recurrent model;
- per-model or per-TP checkpoints;
- learned model-name embeddings;
- cross-hardware transfer;
- a neural scheduler before the analytical scheduler is tested;
- a TCN unless its trigger fires;
- stochastic residuals before deterministic gates pass;
- PP/EP/CP power terms without measured support.

If no candidate passes Llama-405B energy and ACF together, the correct result is
`physics transfers mean; learned dynamics are in-domain only`, followed by the
targeted probe that identifies the missing work term.

## 15. Technical basis

These sources motivate the programming model; repository measurements decide
the coefficients and pass/fail result.

- The [Roofline model](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf)
  motivates separating compute work from memory traffic.
- NVIDIA's official [A100 data sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf)
  and [H100 product specification](https://www.nvidia.com/en-us/data-center/h100/)
  are the sources for declared hardware scales; actual SKU and profiler values
  still belong in each run manifest.
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)
  demonstrates an analytical approach to inference and multidimensional
  partitioning.
- [Megatron-LM parallelism](https://arxiv.org/abs/2104.04473) and
  [DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) motivate explicit TP/PP/DP
  and expert-parallel communication rather than a generic parallelism label.
- [PagedAttention/vLLM](https://arxiv.org/abs/2309.06180) motivates preserving
  KV state and continuous-batching state at the scheduler boundary.
- OpenAI's [gpt-oss architecture table](https://openai.com/index/introducing-gpt-oss/)
  confirms the 20B/120B differences in layers, total/active parameters, and
  experts that the descriptor must represent.
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) defines the
  dense 8B/70B/405B family used for the scale test.
- A small causal convolution is a reasonable last fallback based on the
  [TCN sequence-model study](https://arxiv.org/abs/1803.01271), but it is not
  justified until the linear causal model fails its declared gate.
