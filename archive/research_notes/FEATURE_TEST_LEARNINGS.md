# FEATURE_TEST_LEARNINGS: current findings and historical failure analysis

Status: current through 2026-07-21. The maintained request-only simulator is
the exact timing-ledger path using `timing-test/fitted_efficiencies.json` and
the separated clean power artifact `power-test/clean_power_surfaces.json`.
Sections 2--7 and 9--14 preserve the July 11--16 feature-ladder investigation
and its adversarial review as historical evidence; they are not the current
deployment or result summary. Section 8 is the maintained resume map, while
Sections 15--16 record the completed OpenHands analysis and the current
pipeline, evidence boundaries, and continuation point.

Read this after `FEATURE_TEST_PLAN.md` and before changing candidates. Numerical
truth is claim-specific:

- clean legacy source/twin/stress results: `power-test/clean_power_report.json`;
- coverage-basis held-out experiment: `power-test/coverage_power_report.json`
  and `power-test/coverage_model_fidelity_table.json`;
- external development bundles: `power-test/clean_expansion_report.json`;
- support-limited MoE v3: `power-test/moe_surface_dev_v3.json` and
  `power-test/moe_surface_stress_v3.json`;
- historical M4A/M0c ladder only: `results/feature_test_v2/`.

No complete sealed-campaign score exists yet. Ten of eleven planned bundles
are present; the Qwen3-30B-A3B H100 MoE run is missing, and the current cache
pair comparator also treats run-specific server epochs as invariant controls.
Do not substitute ad hoc per-run scoring for the absent final campaign report.

## 1. Current state

The current deterministic simulator predicts request timing, emits exact
per-iteration GEMM/attention work into a 250 ms ledger, and applies separate
dense and architecture-specific MoE power laws. The dense law uses a fixed
hardware idle floor, busy-gated resident-weight fraction, exact timing-roofline
compute utilization, and duty-aware square-root HBM utilization. It does not
route on model name, arrival rate, or elapsed trace time. The clean default fit
uses all 225 dense and 60 MoE source repetitions through rate 2; 57 rate-4 runs
are retrospective stress tests and 108 related-model runs are transfer twins.

On that legacy population, dense source/twin/rate-4 medians are respectively
1.68%/1.97%/2.99% energy error, 13.10/12.30/14.09 W/GPU RMSE, and
0.9941/0.9952/0.7295 ACF R2. The separated MoE source/rate-4 medians are
1.27%/2.08% energy error and 0.9676/0.9160 ACF R2. These legacy repetitions
overlap heavily in request content, so source and twin summaries are useful
comparators, not an honest independent validation split.

The alternate coverage-basis experiment is the strongest broad held-out
development result. It trains on 117 traces from 39 complete cells and holds
out 333 traces from 111 cells as new rates, TP setups, or model setups. Across
all held-out cells it obtains 3.26% median end-to-end timing error, 1.97% energy
error, 13.58 W/GPU RMSE, and 0.9894 ACF-profile R2. This experiment is explicitly
versioned and does not silently replace the default artifact.

External evidence remains mixed. Clean Qwen rate/shape development runs score
3.06--3.55% energy and 5.73--6.48% timing error, but current GPT-OSS-120B and
405B bundles cross protocol or telemetry boundaries and fail. The measured
routing ablation is rejected. The support-limited GPT-OSS-20B MoE v3 surface
does improve retrospective development/rate-4 energy to 0.92%/1.16% and range
NRMSE to 0.065/0.052, but it is restricted to A100 TP1/2 with uniform routing.

The completed OpenHands packs establish the newest and most useful boundary.
The frozen simulator transfers unseen agentic event structure, cache behavior,
and tool waits, but absolute watts do not transfer zero-shot between nominally
identical A100 deployment platforms. A retrospective two-gain hardware
calibration reduces median energy error from 12.65% to 1.34%, normalized
Soft-DTW from 0.0270 to 0.0049, and range NRMSE from 0.160 to 0.075. This is
few-shot hardware calibration with zero-shot workload transfer, not a sealed
zero-shot result or workload retraining.

### Historical scope of Sections 2--7 and 9--14

Those sections explain why the July feature ladder failed, which constants and
assumptions were rejected, and how the current clean design was motivated.
Statements such as “current result,” “next candidate,” or “pending campaign”
inside them are dated to July 11--16 unless Section 16 explicitly carries them
forward.

## 2. The geometry of the problem (established, unchanged by v2)

### 2.1 The forward map has five layers and the middle is unobserved

```text
offered requests -> scheduler/engine state -> executed iterations and tensor
work -> hardware occupancy, clocks, memory behavior -> sampled power meter
```

The ledger observes offered requests and measured completion timing, and
computes tensor work from architecture descriptors. It does not observe
iteration batches, chunked-prefill composition, clocks, or achieved
occupancy. Several middle-layer trajectories produce the same request-level
ledger but different power. v2 sharpened this from a general statement into
two specific, named axes (section 5).

### 2.2 Energy and autocorrelation structure constrain different directions

Full-run energy constrains the zero-frequency component; ACF compares a
normalized lag-dependence profile. v2 exhibits the trade concretely: on the
405B cell, M4A gets energy right (4.70%) with poor ACF-profile agreement
(R2 -0.615), while M0c improves that profile (0.602; rates 2-4 positive) with
energy 8-11% over. Neither passes the registered gates. ACF does not measure
event timing, alignment, or temporal fidelity; pointwise error and Soft-DTW
must carry those separate claims.

### 2.3 Identifiability findings that now have numbers

- Communication is unidentifiable next to compute on every source fit
  (group correlation about 0.99); NNLS zeroes it and M0c drops the column.
- The idle-to-busy step migrates freely between the busy floor and the first
  memory segment (jackknife range [0, 435] W on A100); low-utilization curve
  ends and the residence term are exchange-prone, while the idle anchor and
  mid-range memory slopes are stable.
- Concave (non-increasing-slope) response curves fit source development as
  well as the unconstrained spiky shapes (within 0.35 pp energy): the source
  data do not determine the shape, so shape choice is a modeling prior, and
  it matters off-support (section 5.3).
- The measured idle-residency effect is real but small (about 0.1-0.15 W/GB;
  405B idles +7.7 W/GPU above other H100 configs) and family-confounded on
  A100.

## 3. What v2 established

### 3.1 Constants must be audited against served artifacts, not model cards

The served 405B checkpoint is FP8 in FFN matmuls only; attention, first/last
layers, and embeddings stay BF16 (Llama 3 paper section 6.2). Resident weight
bytes are the checkpoint total 487.23e9 — the old 1 byte/param constant
undercounted by 20%. The FLOP dtype scale follows the FP8 parameter share
(0.7996 -> scale 0.60), not a blanket 0.5. gpt-oss MXFP4 weights stay 4-bit
resident on Ampere (Marlin dequantize-on-read; a BF16-resident copy cannot
fit the published 2xA100 serving footprint), so the existing 60.8/12.8 GiB
constants were already correct. KV cache is BF16 everywhere (vLLM default
`auto`). CORRECTED CLAIM (2026-07-16 adversarial review, section 14): these
corrections shifted every physics candidate's 405B bias positive by roughly
8-12 percentage points; they improved the formerly under-predicting
candidates (M4A 5.72 -> 4.70%) by moving them through zero and regressed the
formerly near-zero ones (M0bR 2.39 -> 7.39%, M0dR 2.70 -> 6.81%). On the
cancellation-free measured-engine bundles the whole family over-predicts
(section 10). The constants are still right — they are checkpoint facts —
but the response surface they feed is mis-shaped, so the v2 4.70% must not
be read as model accuracy.

### 3.2 The power meter response is now identified, not assumed

`feature-test/identify_meter_kernel.py` extracts idle-to-busy steps from S0
training runs only (230 A100 / 233 H100 events) and fits a boxcar+EMA kernel:
A100 is near-instant at 250 ms bins; H100 carries a 1.0 s averaging window.
This independently reproduces the published NVML metering behavior
(arXiv:2312.02741: A100 `power.draw` is a ~25 ms boxcar updated ~10 Hz; H100
a 1 s boxcar updated 10 Hz). The kernel is stable across model families and
TP degrees, so it is a hardware property. The A100 offset side shows a
~1.5-1.9 s device power-state relaxation tail (GPU-side, excluded from the
meter kernel; a real phenomenon a future device model could use). The old
per-hardware moving-average/EMA lag constants were the right general shape
with the wrong H100 window (0.5 s instead of 1.0 s).

### 3.3 A physical cap beats a fitted cap

The v1 cap (99.5th percentile of source busy power) clipped 34-35% of bins on
H100 hold-TP8 and flattened predicted dynamics. The board power limit
(400/700 W, datasheet) clips almost nothing and removed that failure mode for
M0c; M4A also passes that cell after the constant corrections.

### 3.4 Phase-anchored fitting breaks the compute/memory exchange

M0c fits floors and the memory response on prefill-influence-free bins,
the compute response on prefill-influenced-bin residuals, then refits the
non-compute columns everywhere with compute frozen (declared assumption with
published support: decode is memory-bound, prefill is compute-bound —
Splitwise ISCA'24, POLCA). Result: a nonzero low-utilization compute slope on
H100, which is what restored 405B burst dynamics (saturated-regime power
correlates +0.5..0.7 with computed compute utilization inside runs; the v1
surface priced those bursts at zero because its low-u compute segment fitted
to zero on 8B/70B source).

### 3.5 Cheap-iteration mechanics that are now proven

B2 is bit-deterministic: the v2 rerun reproduced every v1 B2 per-run metric
to 0.00e+00 (its inputs — A_t, delta A_t, power, splits — are invariant to
work-constant changes). Development iterations can therefore drive the
evaluator's internals through `importlib` and reuse frozen B2 rows, cutting
iteration cost from ~80 minutes to ~4 minutes, without weakening the frozen
comparison. The ledger-rebuild equivalence proof (only `w_read*`/`w_bytes`
scaled, only on 405B bins, by exactly 1.200517) is the template for future
constant changes.

## 4. Why the remaining cells fail (v2 forensics)

### 405B (S2b): the tensor-parallel synchronization axis

M0c over-predicts uniformly (+8..+11.5% at every rate). Single-request 405B
decode sweeps 60.9 GB/GPU per iteration at ~31 ms — about 57% of peak HBM
bandwidth, the same per-GPU utilization band as 8B TP1 source runs. But TP8
decode inserts per-layer all-reduce stalls that TP1 streaming does not have,
so the same per-GPU bytes per second draws measurably less power. The
communication column that would carry this is unidentifiable in source
(r=0.99 with compute). Before the byte correction the curve under-predicted
(-12%); the corrected bytes moved 405B deeper into a curve region identified
by stall-free TP1 streaming and flipped the sign. The remaining ACF failure
is confined to rate 1.0 (all three repeats, ACF about -9, NRMSE 0.78; slow
~100 W regime shifts with no ledger correlate); rates 2-4 are now positive.

### gpt-oss (S1 and the A100 cells): the iteration-granularity axis

Opposite-sign in-domain biases persist under every one-surface candidate
(S0_A100 development: 120b TP4 over-predicted about +10-13%, 20b TP2
under-predicted about -7-9%). Routing-assumption bounds cannot explain it:
at the rates where the bias peaks the decode batch is about 1, where every
routing assumption yields the same traffic. Single-stream throughput shows
120b decode is launch/sync-bound (TP4 and TP8 rates nearly identical), so
equal ledger bytes per second again produce different power. No response
surface over the current features can fix a sign flip inside one hardware.

### A100 TP1/TP2 holdouts: shape choice off-support

M0c under-predicts at high rate (to -19% at rate 4) because hold-TP1/TP2
targets sit above the source utilization support and a concave surface
extrapolates with its smallest slope; the unconstrained shapes happen to
extrapolate steeper and pass. Source development provably cannot distinguish
these shapes, so this is a prior, not a measurement. Choosing the shape that
passes these cells because it passes them would be target fitting; only
up-range probe data (fixed-batch high-utilization sweeps) can decide it.

### H100 S0 versus B2

Nobody passes the B2-relative NRMSE clause (limit 0.0817): M0c reaches
0.08198 — 0.0003 short, the closest any one-artifact candidate has come —
and M4A 0.091. B2 holds 356,226 scalars per configuration against M0c's 16
per hardware; the residual gap is saturated-regime texture (section 4,
405B rate-1 paragraph).

## 5. No-go results that should not be repeated

Carried over from v1, all still binding:

- target-chosen cap, support, or lag thresholds;
- larger fitted power caps; single-alpha sweeps; unconstrained dual lags;
- M4A plus the M1 correction; running/waiting substitutions for A_t
  (waiting is ~0 in every legacy schedule — queues never form);
- batch/iteration/prefill-active substitutions or additive iteration state
  from reconstructed state;
- seeded stochastic residual blocks; elapsed-time lag reinterpretations;
- zero-DC running/waiting residuals; RC high-pass bases;
- ridge shrinkage of state coefficients; residency-scaled active state;
- per-cell mean routing.

New from v2:

- an always-on (idle-active) residence term: NNLS pumps busy-load signal
  into idle bins (fitted 57-69 W at full occupancy against a measured
  ~12 W ceiling); keep residence busy-gated;
- fitting the staged memory response only on prefill-free bins without the
  final all-bins refit: it starves the curve of the saturated mixed-bin
  range (H100 S0 development energy 4.18% -> 1.56% with the refit);
- iterating response-surface shape (concave vs unconstrained) against
  holdout cells: source development cannot distinguish them, so any choice
  made on target scores is target fitting;
- expecting zero-DC causal corrections (M1-M3) to rescue a mean: with the
  v2 bases they fail correction safety everywhere because the bases'
  dynamics are already at the data's ceiling.

## 6. The smallest meaningful next experiment

Historical July 16 plan. Several listed campaigns have since run; use
Section 16.8 for the current continuation point.

Unchanged in spirit from v1, now with sharper targets. Collect synchronized
engine and device state for cells that separate the two named axes:

1. TP-sync axis: same dense model and request marks at TP4/TP8, plus per-GPU
   power/clocks and the stock vLLM token/iteration counters — does power per byte
   fall with TP at fixed per-GPU work? Stock vLLM does **not** expose collective
   duration or NVLink bytes, so this campaign identifies the axis from the
   matched TP contrast; it does not claim to measure the mechanism directly.
2. Iteration-granularity axis: matched gpt-oss 20b/120b fixed-batch decode sweeps
   over batch × context, using stock tokens-per-iteration and iteration rate —
   does the sign flip follow iteration granularity after context is controlled?
   Router/expert-touch counters are not in the maintained runtime path.
   UPDATE 2026-07-16: the expert-touch half no longer needs runtime
   counters — routing is a pure function of weights and text, so one
   offline forward pass over the reconstructed benchmark prompts measures
   the routing law directly. Specified in TODO.md item 1.
3. Up-range shape: fixed-batch decode at several context lengths pushing
   per-GPU memory utilization through 0.4-0.9 on both hardwares — concave
   or not, measured, once;
4. One idle-to-steady step probe per hardware (validates the identified
   meter kernel and measures the A100 power-state relaxation tail).

5. Engine-state serving reruns: repeat the hardest realistic serving cells
   with the engine's internal state logged every 250 ms (running and waiting
   requests, KV-cache occupancy, scheduled tokens per step), synchronized
   with the per-GPU power/clock log. Cells: 405B TP8 at rates 1 and 2;
   70B TP4 and TP8 at rates 2 and 4 on both hardwares; gpt-oss-120b TP4 at
   rate 1. The rate-1.0 405B regime shifts have no correlate in anything we
   currently record, so they need this state to be attributable at all.
   These runs also provide the queue/backlog observations required to
   validate the arrival-only scheduler (plan section 13) — today waiting is
   ~0 in every legacy schedule, so overload behavior is unobserved.
6. A sealed holdout: predeclare a few configuration/rate cells with fresh
   seeds, collect them in the same campaign, and let nobody inspect them
   until final scoring. The current targets are retrospective development
   data; without a sealed set there is no honest validation claim.

The collection scripts and stock-metric path now exist and remain unrun on GPUs:
`profiling/probes/` includes the staircases, transients, and the orthogonal
`decode_context_grid`; `profiling/client/` records stock engine evidence and
per-GPU power/clocks; `feature-test/build_ledger_bundle.py --state-source
measured_engine` carries those fields through `RunRecord` into the cache. Direct
collective/router instrumentation is explicitly out of the current data path.
Add it only with a concrete versioned vLLM patch and an end-to-end persistence
test. See `profiling/CAMPAIGN.md` for the executable campaign order.

## 7. Evaluation discipline for the continuation

1. Freeze feature equations and constants (with provenance classes: cited
   hardware/architecture, source-fitted under frozen procedure, dedicated
   probe, or plan-fixed threshold) before scoring transfer targets.
2. Fit and select on source train/development only; one artifact per
   hardware; no model/family/TP routing.
3. Report every transfer cell including P90/worst temporal metrics and
   signed bias by rate; medians hide the failure structure (the v2 405B
   ACF median of 0.602 averages near-perfect low-rate cells with a
   catastrophic rate-1 cell).
4. Keep B2 same-configuration only; reuse its frozen rows during
   development only under a proven input-invariance argument.
5. After any ledger change, produce an explicit equivalence proof of what
   changed and what did not.
6. The current targets remain retrospective development evidence. Any
   validation claim requires a new sealed campaign.

## 8. Where to resume

- experiment contract: `FEATURE_TEST_PLAN.md`;
- current timing fit and scheduler: `timing-test/fitted_efficiencies.json`,
  `timing-test/scheduler_sim.py`, and `timing-test/simulated_ledger.py`;
- current dense law: `power-test/clean_dense_surface.py`;
- separated fit/evaluation: `power-test/fit_clean_power_pipelines.py` and
  `power-test/clean_power_surfaces.json`;
- external evaluator: `timing-test/evaluate_expansion.py`;
- sealed scorer: `power-test/score_sealed_campaign.py`;
- coverage experiment: `timing-test/build_coverage_split.py` and
  `power-test/evaluate_coverage_split.py`;
- MoE support boundary: `power-test/moe_surface.py` and
  `power-test/fitted_moe_surface_v3.json`;
- historical M4A/M0c ladder: `feature-test/evaluate_candidates.py` and
  `results/feature_test_v2/`.

Rebuild the maintained clean path:

```bash
uv run python timing-test/simulated_ledger.py \
  --dt 0.25 --roles all --moe-routing uniform \
  --out feature-test/ledger_cache_sim_uniform_current_250ms.npz
uv run python power-test/join_power.py \
  --cache feature-test/ledger_cache_sim_uniform_current_250ms.npz \
  --out power-test/sim_ledger_power_uniform_current_250ms.npz \
  --provenance-out power-test/sim_ledger_power_uniform_current_250ms.provenance.json
uv run python power-test/fit_clean_power_pipelines.py
uv run python timing-test/evaluate_expansion.py \
  --timing-fit timing-test/fitted_efficiencies.json \
  --power-fit power-test/clean_power_surfaces.json \
  --out power-test/clean_expansion_report.json
```

Before trusting any result:

```bash
uv run -m pytest -x
uv run -m pytest -x feature-test/tests
```

## 9. Final lesson

Historical July 16 lesson: the data identify a good average power surface over
the source manifold but not a unique map from request timing to power response
when engine iteration structure or device operating state changes. The current
pipeline resolves much of the scheduler/work-accounting ambiguity, and the
coverage experiment demonstrates broad within-platform transfer. OpenHands
adds the sharper deployment lesson: nominal accelerator identity still does
not identify the watt response. Current conclusions and next actions are in
Sections 15--16.

## 10. Repaired-data audit (2026-07-16)

The current legacy raw tree was re-inventoried and rebuilt independently at
250 ms: 800 matched runs across 29 configurations produced the same 450-run,
1,088,106-bin ledger. Its NPZ SHA-256 is byte-for-byte identical to the July 11
ledger (`20ee96c…`). Therefore the repaired legacy files do not change any v2
model input or score; rerunning v2 would reproduce the same result.

Removing the acknowledged A100 Llama-70B TP4 cell also does not explain the
remaining failures. For selected M4A, S3 A100 hold-TP4 median energy changes
4.75% -> 4.20%, but the split still fails; S1 gpt-oss, H100 S0, and H100 405B
are disjoint from that cell and are unchanged.

The new canonical hard-cell bundles do change the evidence because they contain
stock engine counters. Scoring the frozen v2 artifacts on a complete 250 ms
causal feature grid, while masking rather than interpolating missing meter
targets, gives:

| cell | measured-engine signed energy bias | energy error | 1 s ACF R2 | NRMSE-range |
|---|---:|---:|---:|---:|
| A100 gpt-oss-120B TP4 rate 1 | -11.29% | 11.29% | -0.145 | 0.324 |
| A100 gpt-oss-120B TP4 rate 2 | -14.49% | 14.49% | -0.457 | 0.274 |
| H100 Llama-405B TP8 rate 1 | +11.93% | 11.93% | unavailable | 0.204 |
| H100 Llama-405B TP8 rate 2 | +7.48% | 7.48% | unavailable | 0.195 |

H100 ACF is unavailable because no contiguous observed one-second section is
long enough for the predeclared 60-lag statistic; gaps are not bridged. Direct
observed-sample biases (-11.21%, -14.53%, +12.22%, +6.76%) confirm the same
energy conclusion.

Measured engine state improves A100 energy relative to request reconstruction
(24.8-33.9% error -> 11.3-14.5%) but does not reach the gate and makes ACF
negative. On H100, reconstruction's apparently good 0.8-2.3% energy error turns
into 7.5-11.9% overprediction under executed-token/active-state evidence. The
old agreement was cancellation, not a fixed model. The repaired data therefore
clarify the failure but do not fix the frozen model.

## 11. What the working GMM-BiGRU result actually establishes (2026-07-16)

The main IID GMM-BiGRU result is useful evidence for a **state-conditioned**
power model, but it is not evidence that the architecture-transfer problem is
solved. Its contract is materially easier than the feature test:

- each exact `(model, hardware, TP)` configuration has its own GMM power
  centers, normalization, throughput calibration, and BiGRU checkpoint;
- the split is random over repeated traces inside that exact configuration;
  all 120 evaluated test traces have an offered rate that is also present in
  their configuration's training set, and all 25 training sets contain the
  complete `{0.125, 0.25, 0.5, 1, 2, 4}` rate set;
- the classifier is bidirectional. At each time it can use the complete future
  reconstructed active-request trajectory, which is itself built from the
  completed request log including observed `output_tokens` and a
  configuration-local throughput fit;
- it generates a distribution from configuration-local power states. Its good
  energy and ACF do not imply good pointwise prediction.

On `results/continuous_v1_gmm_bigru/k10_f2/eval_metrics_fullheldout`, the 25
configuration medians are 2.33% energy error and 0.993 ACF R2, but median
NRMSE is 0.341. The operating-condition split is sharper:

| offered rate | test traces | median idle fraction | energy error | NRMSE | ACF R2 | KS |
|---:|---:|---:|---:|---:|---:|---:|
| 0.125 | 45 | 0.877 | 2.893% | 0.349 | 0.995 | 0.285 |
| 0.25 | 20 | 0.720 | 3.081% | 0.409 | 0.995 | 0.186 |
| 0.5 | 24 | 0.516 | 2.409% | 0.420 | 0.995 | 0.127 |
| 4.0 | 25 | 0.027 | 0.800% | 0.142 | 0.932 | 0.131 |

Thus it is genuinely strongest pointwise at sustained, high-load operation.
At sparse load, long idle residence makes aggregate ACF easy to preserve while
burst timing/amplitude remains inaccurate. Large configurations also remain
hard: configuration-median energy error is 7.13% for H100 Llama-405B TP8,
7.64% for A100 Llama-70B TP4, and 7.90% for A100 Llama-70B TP8.

### The transferable geometric lesson

The useful part is the latent-state geometry. Fixed request counts do not map
to fixed power. In source training data:

- A100 Llama-70B TP4 at `A=0` spans 284-1048 W (p05-p95). At `A=1`, mean
  power is 736 W on entry (`delta_A>0`), 1211 W while steady, and 1222 W on
  exit (`delta_A<0`).
- H100 Llama-405B TP8 at `A=1` similarly averages 1536 W on entry, 3524 W
  while steady, and 3612 W on exit.
- exact-configuration GMM centers directly encode different floors and
  plateaus: A100 Llama-70B TP4 spans 291-1600 W, while A100 gpt-oss-120B TP4
  spans 263-1056 W.

Some spread is the measured meter/device relaxation already identified in
section 3.2; the rest combines prefill/decode phase, saturation, iteration
granularity, and hardware operating state. The BiGRU can infer proxies for
these regimes from history, future active duration, and configuration-local
calibration. The current physics candidate instead asks one shared response
surface to be nearly single-valued in computed work. That mismatch is the
most obvious missing structure.

The next candidate should therefore be a shared physics mean plus a small
**causal latent operating-state correction**, not a wholesale replacement by
the existing BiGRU. Inputs must follow the declared deployment contract. Stock
vLLM provides completed prompt/decode and iteration-counter deltas plus
running/waiting and cache gauges; it does not provide upcoming scheduled-token
state in this data path. A measured-online model may use only left-edge or
lagged versions of those fields. An arrival-only simulator cannot use them
until its scheduler twin predicts them. Clocks, utilization, and temperature
are valid explanatory diagnostics, but are model inputs only if the deployment
path will actually observe them. Do not add a raw configuration lookup if the
claim remains zero-shot model/TP transfer.

### Minimal tests before building that candidate

No new full profiling grid is required to decide whether this direction is
real. Run these discriminating ablations first:

1. retrain the GMM-GRU with rate-grouped holdouts (one interior rate and rate
   4 as an extrapolation edge), so no test trace shares its rate with training;
2. replace the BiGRU with a causal GRU under the identical split and inputs;
3. hold out an entire model/TP configuration or train one hardware-shared
   state model, removing the exact-configuration GMM/normalization lookup;
4. on the measured-engine hard cells, compare the frozen physics mean against
   (a) a training-only affine DC calibration and (b) a causal latent residual.
   The former isolates mean calibration; only the latter can justify a claim
   about missing dynamics.

These tests separate three hypotheses without confounding them: local power
calibration, future-schedule lookahead, and a genuinely reusable operating
state. There is no direct held-out-power leakage in the audited pipeline; the
problem is that its present evaluation support and information contract are
much narrower than the feature test's transfer contract.

## 12. HSMM critic decision (2026-07-16)

Three independent audits — model structure, maintained data path, and primary
literature — reached the same decision: **do not make a plain HSMM the next
main model**. Test one small input-driven autoregressive switching residual as
a falsifiable ablation, and add explicit duration only if it beats an otherwise
identical input-output HMM on untouched transfer runs.

The old K10 labels do contain non-geometric duration, but mostly as chattering
wattage quantization rather than demonstrated physical modes. Across the 25
configuration training sets there are about 339k state runs: median duration is
one 250 ms bin, 54.9% last one bin, and 64.8% of changes move only to an
adjacent wattage component. An HSMM trained on those labels would first learn a
smoother. It would not establish a reusable scheduler/device regime.

The admissible candidate keeps the frozen shared physics mean and known meter
kernel, then models its residual with two to four hardware-local states and no
configuration ID:

```text
r_t = P_t - meter_h(physics_h(x_t))
Pr(z_t | z_{t-1}, q_t)                       # input-output HMM
r_t = b_z + rho_z * rhat_{t-1} + beta_z q_t # ARX residual emission
```

`rhat_{t-1}` is the previous predicted residual in an open-loop simulator, not
held-out power. The HSMM variant adds elapsed state age only to the exit hazard:

```text
logit Pr(exit_t | z_t, age_t, q_t)
    = a_z + s_z(log(1 + age_t)) + c_z q_t
```

This avoids a fitted maximum-duration constant and allows workload state to
govern residence. Fit states and emissions jointly; do not preassign GMM
labels. Apply the identified meter response outside the switching layer so
duration does not rediscover the H100 one-second window. Do not call the states
DVFS modes: active clocks are nearly fixed in the audited transient probes;
the supported interpretation is scheduler/iteration regime plus device/meter
relaxation.

### Required falsification ladder

Use the same parameter budget, folds, and causal inputs for:

1. physics plus affine DC calibration;
2. physics plus one-state causal ARX/asymmetric response;
3. physics plus a two- or three-state input-output AR-HMM;
4. the identical model with an age-dependent exit hazard;
5. a deterministic engine-feature gate.

Oracle smoothing is diagnostic only. Primary results must be schedule-only
open-loop rollouts; causal filtering with past observed power is a separate
online-forecast contract. Hold out an offered-rate group and entire model/TP
configurations. Use the existing deterministic energy/NRMSE/ACF gates for the
predictive mean and add held-out likelihood or CRPS/calibration for sampled
traces. Reject explicit duration if it does not improve over the matched HMM,
if the gain disappears open-loop, if state occupancy/meaning changes under
configuration holdout, or if the deterministic gate matches it.

### Evidence boundary before implementation

The legacy 450-run cache has no engine stream. The measured-engine builder
persists executed token rates, running/waiting state, iteration rate,
tokens/iteration, and cache usage, but these are retrospective bin aggregates;
same-bin means/deltas are not strict online inputs. Several validated bundles
also contain multi-second logger gaps. A duration model must split sequences at
engine discontinuities and mask power gaps, never count interpolation as state
residence.

There are currently 20 complete validated bundles from four campaigns, versus
35 development bundles across eight campaigns required by the runbook. That is
enough for the HMM-versus-HSMM kill test, not enough to tune and declare a
transferable state model ready. The trainer also needs the still-missing merged
run index with validation role/status; random 250 ms bin splits or ad hoc
directory selection are forbidden.

Primary foundations: [Bengio and Frasconi's input-output
HMM](https://papers.nips.cc/paper_files/paper/1994/file/8065d07da4a77621450aa84fee5656d9-Paper.pdf)
conditions sequence dynamics on external inputs; [Johnson and
Willsky](https://www.jmlr.org/papers/v14/johnson13a.html) show why explicit
duration is needed only beyond geometric dwell;
[Chiappa](https://arxiv.org/abs/1909.05800) separates explicit-duration,
segment, and reset formulations; [Linderman et
al.](https://proceedings.mlr.press/v54/linderman17a.html) show recurrent
switching dynamics but at substantially greater identifiability cost. The
repository's small-data, shared-physics setting justifies the finite
AR-HMM/HSMM ablation, not an rSLDS, neural HSMM, or new per-configuration
mixture.

## 13. Broader model research: the general model should be an input-driven system (2026-07-16)

Section 12 rules on one possible residual model; it is not the overall design.
The broader literature and this repository's failures point to a clearer
decomposition: **requests drive a scheduler and timing model; the resulting
executed phases drive a low-order power system; measurement noise is last**.
Do not ask one sequence network to learn all four maps at once.

```text
(arrival, input tokens, output-token mark/distribution)
                         |
                         v
          scheduler + KV/queue state machine
                         |
                         v
     per-iteration prefill/decode/communication plan
                         |
             timing and exposed-stall model
                         |
                         v
       phase-resolved work and duty on wall time
                         |
                         v
  hardware power equilibrium -> device dynamics -> meter -> noise
```

This follows the system-identification distinction between known exogenous
inputs, internal state, process disturbance, and measurement noise. Schoukens
and Ljung's [nonlinear system-identification
roadmap](https://arxiv.org/abs/1902.00683) emphasizes starting with the
simplest structure supported by prior knowledge, designing experiments for the
intended input domain, and separating structural error from noise. Those are
exactly the three places the configuration-local BiGRU/GMM is weak.

### 13.1 The proposed model interfaces

Let request `i` have arrival `a_i`, input length `l_i`, and output length mark
`o_i`. An engine-policy state machine, not the power model, produces iteration
`k`:

```text
s_(k+1) = scheduler(s_k, newly_arrived_requests; policy, KV_capacity)
q_k = (prefill_tokens, decode_tokens, batch, context_sum, cache,
       collective_count, collective_bytes, expert_work_expectation)
```

Architecture descriptors — layers, width, FFN size, attention/KV heads,
active and resident parameters, dtype/quantization, MoE experts/top-k, sliding
window — transform `q_k` into FLOPs and bytes. Parallelism and topology — TP
now; PP/EP only when declared — transform it into collective messages and
volumes. They should not be free categorical embeddings.

Iteration duration should be modeled before power:

```text
T_compute_memory = smooth_max(F_k / C_eff, B_k / M_eff)
T_collective = N_collective * alpha + V_collective / beta_eff
T_k = T_launch + T_compute_memory + exposed_fraction * T_collective + T_sync
```

`alpha`, `beta_eff`, launch cost, and overlap may depend on hardware, topology,
parallel plan, message size, and iteration granularity, but not model name.
This is the key correction to the current single communication-byte-rate
column. Tensor parallelism pays repeated synchronization/message latency, not
just energy per transferred byte. A single aggregate `comm bytes/s` is
collinear with compute in source data and cannot represent a GPU waiting at a
collective. Standard Megatron tensor parallelism introduces collectives inside
each transformer layer; [Megatron-LM](https://arxiv.org/abs/1909.08053) and
[ASTRA-sim](https://doi.org/10.1109/ISPASS48437.2020.00018) motivate keeping
collective count, volume, topology, and compute/communication overlap explicit.

The scheduler then places each predicted iteration on wall time and exposes
separate fractions for compute/memory work, communication, launch/sync stall,
and idle. This preserves the two coordinates our residual analysis says are
missing: tokens or work **per iteration**, and iterations or synchronization
points **per second**.

The simplest credible power system is a gray-box block model:

```text
P_eq(t) = p_idle_h + g_h(u_compute, u_memory, phase_duty,
                         comm_duty, stall_duty, granularity)
x_(t+1) = A_h(q_t) x_t + B_h(q_t) P_eq(t)
P_device(t) = C_h x_t + D_h P_eq(t)
P_meter(t) = meter_kernel_h(P_device)(t) + epsilon(t)
```

`g_h` is one hardware-local, shape-constrained piecewise-linear/GAM surface in
dimensionless coordinates, with only predeclared low-order interactions. The
state dimension starts at one and may grow to two only if transient probes
identify separate rise/recovery time scales. `A_h(q_t)` may switch smoothly
between idle/busy or rise/fall coefficients; this is a linear-parameter-varying
(LPV) model with observed scheduling variables, not an unobserved-state model.
The identified NVML kernel remains outside the device state.

This is a generalized Hammerstein/LPV structure: a static nonlinear work-to-
equilibrium-power map followed by short linear dynamics. Block-oriented models
are attractive precisely because they separate static nonlinearity from
dynamics and remain understandable; see the [block-oriented identification
survey](https://arxiv.org/abs/1607.01217). Shape-constrained additive models
provide flexible but regularized surfaces without arbitrary off-support spikes
([Pya and Wood](https://doi.org/10.1007/s11222-013-9448-7)). Constraints belong
on achieved utilization or duty, not directly on offered token counts.

### 13.2 Why request tokens and timing remain first-class

The input/output token pair is not merely another neural feature. It defines
the future sequence of prefill work, decode iterations, context growth, KV
traffic, and — through the scheduler — interference and queueing. Systems work
supports this separation:

- [Vidur](https://proceedings.mlsys.org/paper_files/paper/2024/hash/b74a8de47d2b3c928360e0a011f48351-Abstract-Conference.html)
  couples an event-driven scheduler to operator-specific profiled/predictive
  timing models and reports less than 5% error for end-to-end performance in
  its evaluated settings. It explicitly reduces attention batches using token
  and context geometry rather than treating a trace as an opaque sequence.
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310) shows that chunked prefill
  changes decode interference and iteration composition; the same requests can
  therefore produce different wall-time work under a different scheduler.
- [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
  demonstrates that prefill/decode placement and communication change their
  interference and parallelism plan. A power model cannot bury scheduler mode
  inside a learned recurrent state and still claim generality.
- [NeuSight](https://arxiv.org/abs/2407.13853) obtains better unseen-model/GPU
  timing forecasts by decomposing kernels into physically bounded working sets
  instead of regressing whole-kernel latency directly. That is the same design
  principle needed here: learn small efficiency/overhead closures around known
  work, not a configuration-to-trace lookup.

There are three distinct output-length contracts and they must never be mixed:

1. trace replay/offline simulation: final `o_i` is a legitimate input mark;
2. workload generation: sample `(l_i, o_i)` jointly before scheduling and
   propagate that sample through the engine;
3. online prediction: final `o_i` is unknown. Use a declared desired/max length
   or a conditional output-length distribution, and update state as tokens are
   generated. Conditioning online results on the completed output is an oracle.

### 13.3 Model-family decision table

| family | useful role | decision here |
|---|---|---|
| static physics or shape-constrained GAM | equilibrium power over executed work | required, but insufficient without timing/dynamics |
| Hammerstein / low-order LPV state space | nonlinear equilibrium plus short causal device response | **first choice** |
| sparse NFIR/NARX or Volterra model | tests explicit input histories/interactions | strong diagnostic; avoid held-out past power in simulator mode |
| Kalman/linear state-space residual | uncertainty and a small continuous latent response | viable if transient probes identify it |
| GP/NARX or GP state-space | uncertainty with little data | useful on aggregated probe points; too costly/weakly identified for the full ledger first |
| HMM/HSMM/switching LDS | residual conditional multimodality or discrete modes | only after observed-input continuous models fail |
| causal TCN/CNN | flexible nonlinear finite-memory residual | **last practical fallback**, under 5k parameters |
| GRU/LSTM/general neural state-space | unrestricted hidden memory | not justified before the above ladder |

A causal TCN is not an arbitrary idea: TCNs can be viewed as nonlinear FIR/
Volterra or block-oriented system models ([Andersson et
al.](https://arxiv.org/abs/1909.01730)), and causal convolutions have compared
favorably with recurrent networks on broad sequence tasks ([Bai et
al.](https://arxiv.org/abs/1803.01271)). If used here, it consumes the
phase-resolved, dimensionless iteration/work stream, never raw configuration
IDs and never future inputs. Its finite receptive field is a feature: every
second of memory is visible and falsifiable. A TCN directly on request bins
would merely become a more convenient overfit estimator than the BiGRU.

### 13.4 What to fit, in what order

Do not train the complete graph end-to-end initially. Each interface has its
own target and holdout:

1. **Scheduler fidelity:** replay requests through the actual policy; validate
   batch composition, queue/backlog, completion order, cache, TTFT, and ITL.
2. **Timing model:** on controlled holds, predict iteration rate/duration from
   tokens/iteration, context, architecture work, TP collectives, and topology.
   Compare analytical roofline plus alpha-beta communication against a small
   GAM residual and a tree/MLP baseline. Hold out context levels, one TP, and
   one model scale.
3. **Equilibrium power:** use measured executed state and observed iteration
   timing to fit the hardware-local shape-constrained surface. Hold out entire
   operating points, models, and TPs — never random bins.
4. **Dynamics:** identify one- then two-state rise/recovery response from the
   transient probes, with the meter kernel fixed. Retain a state only if it
   improves open-loop transfer.
5. **Flexible residual:** add the tiny input-only causal TCN only if residuals
   retain reproducible correlation with causal work history in at least two
   source-development cells.
6. **Noise:** after the conditional mean passes, estimate heteroscedastic/
   correlated innovations from repeats. Noise never repairs the mean score.

The evaluation matrix must independently hold out: input/context-length bands,
output-length bands, offered-rate or burst pattern, model architecture/scale,
TP, and finally a combined model+TP+workload cell. This is how input/output
tokens and architecture descriptors become demonstrated generalization rather
than decorative columns. Report both component fidelity and end-to-end power;
otherwise a good power score may hide timing cancellation, as the measured-
engine audit already showed.

### 13.5 Data implication

No exhaustive model x TP x rate grid is required. The existing campaign's
orthogonal token/context/batch sweeps and matched TP/model contrasts are the
right experiment design. The remaining gpt-oss 20B/120B contrasts are important
because they identify launch/iteration and TP effects separately. The final
model-readiness claim still requires the runbook's role-aware development index
and sealed cells.

The most important data addition is not another power trace. It is a clean
iteration timing table with, per controlled interval: prompt/decode tokens,
tokens/iteration, iterations/s, context/KV summary, architecture-derived work,
collective count/bytes implied by the declared parallel plan, and synchronized
power. Direct collective duration would improve attribution but remains outside
the stock path; until instrumented, fit exposed communication only from matched
TP contrasts and report it as such.

### 13.6 Decisive next model

The next named candidate should therefore be:

> **shared phase-resolved physics + analytical/profiled scheduler timing + a
> one-state hardware-local LPV power response + the fixed meter kernel**.

It is meaningfully different from M4A. M4A filters reconstructed aggregate
work and two `A_t` channels after per-request timing; the new candidate predicts
iteration timing first, separates collective count/volume and exposed stall,
and drives device dynamics with equilibrium phase power rather than using
active-request EMAs as proxies. Try a shape-constrained interaction surface and
a second device state before any CNN. If those fail with correct timing, test
the predeclared tiny causal TCN on their residual. Only then is a richer latent
or switching model warranted.

## 14. Adversarial review findings (2026-07-16)

Three independent reviews attacked the v2 pipeline's claims. Record of what
broke and what survived; every number below was recomputed, not quoted.

### Confirmed errors and overstatements

1. Section 3.1's original "improved every candidate" claim was false — see
   the corrected paragraph there. The mechanism was a uniform positive bias
   shift, not error reduction.
2. The step-response identification script has a bug: its sub-bin phase
   term cannot represent the ~1.2-bin lag between recorded work onset and
   power response, so the fit laundered that alignment lag into a spurious
   smoothing constant (A100 0.75, H100 0.8). With the lag represented, the
   corrected result is: A100 near-instant, H100 a pure 1.0 s moving average
   (independently confirmed from offset events the onset fit never used),
   plus an explicit ~0.3 s onset delay that belongs in the timing layer,
   not in a filter. Development scores actively prefer over-smoothing
   (alpha 0.5 beats both the shipped and the correct value), i.e. the
   filter absorbs model error — so any smoothing constant must either be
   declared a fitted hyperparameter or replaced by the explicit delay.
3. "M0c passes all four H100 TP holdouts": three are robust (bootstrap
   failure probability <= 0.001); hold-TP8 is a knife edge (ACF-MAE P90
   sits at a bimodal cluster boundary, bootstrap failure probability 0.42,
   one run flips it). M0c's dynamics edge over M4A exists mainly where
   M4A's fitted cap clips. All these passes are retrospective: the cells
   were inspected before the mechanisms were designed.
4. The staged fit enforces a phase prior; it does not break the
   compute/memory exchange in general. In a synthetic regime where prefill
   bins carry elevated memory traffic it silently recovers curves off by
   -37%/+11% while a plain joint fit is near-exact. On the real ledger the
   prior's premise holds in-sample (prefill-bin memory utilization ratio
   0.97-0.98x) and stage-1-to-stage-3 movement is small; a large movement
   is a usable tripwire. The "restored compute response" is one linear
   segment whose extrapolation is exactly the +8-11% 405B over-prediction.
5. Test coverage gaps proven by mutation: a variant of the staged fit that
   trains on test bins survives the current suite (fixtures use all runs
   as refit); the influence-mask tail and the consumption of the shipped
   kernel JSON are untested. The board-power cap is well pinned.
6. Reporting: 405B ACF R2 medians average regimes with 10x different
   denominators (measured-ACF total variance 2.2-3.1 at low rates vs
   0.19-0.33 at rates 1-4); ACF-MAE is the stable metric for
   autocorrelation-profile claims. `energy_error_pct` equals
   `|mean_bias_pct|` per run, which
   is what let a sign flip read as improvement. The deployed v2 artifacts
   (M4A) still carry the 0.5 s window and fitted quantile cap that
   sections 3.2-3.3 describe as superseded — the fixes shipped only in the
   unselected candidate.

### Independently reproduced and confirmed

- Section 10's measured-engine scoring: all four energy biases exact to
  two decimals, NRMSE exact, H100 ACF-unavailability confirmed, and the
  cancellation comparison confirmed (reconstruction-path H100 error
  -0.9..-2.5% vs measured-path +6.8..+12.2%). One value (A100 r1 ACF R2
  -0.145) is not reproducible under any tried convention; all variants
  are more negative, so the failure was understated. Note: section 10
  used the bundle manifests' architecture constants, not the frozen
  artifacts' descriptors (moves biases <= 1.6 pp; should be reconciled).
- The H100 1.0 s reading window, the idle anchors, the ledger equivalence
  proof, the bit-exact B2 reproduction, and the source-only fit machinery
  (no target power enters any fit; artifact loader rejects routing keys)
  all survived attack.
- Provenance honesty: every audited constant is target-blind in value but
  target-driven in selection (each entered after a target cell failed).
  The correct label is the one selected_model.json already carries —
  "source development only after retrospective model design" — and it
  must also be carried on the deployable artifacts and README, and can
  only be discharged by the sealed campaign.

## 15. OpenHands finding: workload transfer after hardware calibration

The six-run OpenHands subset exposed a boundary hidden by the coarse `A100`
hardware label. The source power surface was fit on an Azure eight-GPU
A100 node with a 400 W limit, while OpenHands ran TP1 on a different four-GPU
A100 platform. The OpenHands bundles preserve the unseen agentic structure:
tool waits, growing conversational contexts, large uncached prefills, and
cache-on/off execution. Their request-derived predictions place those events
correctly, but the frozen source watt mapping under-prices compute-bound
prefill on the new platform.

A retrospective two-gain platform calibration explains most of the miss
without changing timing, scheduling, cache accounting, or the source power
surface:

```text
P_calibrated = 83.2 W
             + 1.0425 * (P_source - 70.1188 W)
             + 0.5891 * source_prefill_compute_contribution
```

Equivalently, ordinary dynamic power receives a 1.0425 gain and the total
prefill-compute contribution a 1.6316 gain. Across all six OpenHands traces,
median energy error changes from 12.65% to 1.34%, normalized Soft-DTW from
0.0270 to 0.0049, and range NRMSE from 0.160 to 0.075. Leave-one-pack-out
fits are stable: ordinary gains are 1.039--1.050, total prefill gains are
1.627--1.635, and held-out energy errors are 0.29--2.65% with median Soft-DTW
0.0049. The overlay is
`power-test/openhands_platform_calibrated_prediction_overlay.png`.

The paper-facing finding is therefore:

> Nominal accelerator identity is insufficient for zero-shot absolute-watt
> transfer across deployment platforms. The request/timing simulator still
> transfers unseen workload structure, including agentic pauses and cache
> behavior, while a low-dimensional hardware calibration recovers the local
> watt response. This is few-shot hardware calibration with zero-shot workload
> transfer, not workload-specific retraining.

Metric language must remain precise. Soft-DTW measures time-warped trace-shape
agreement, range NRMSE measures pointwise magnitude agreement, and energy error
measures the integrated magnitude. ACF R2 measures agreement between
autocorrelation profiles over the selected lags; it does not measure event
timing, alignment, or temporal fidelity. A low ACF R2 must therefore be
reported as an autocorrelation-structure disagreement, not used to negate low
Soft-DTW and pointwise error.

These constants were extracted after inspecting OpenHands power and are
retrospective calibration evidence, not a sealed zero-shot result. The clean
confirmatory experiment is to estimate the same idle, ordinary-dynamic, and
prefill gains from short non-agentic idle/prefill/decode probes, freeze them,
and then score untouched agentic traces.

## 16. Current pipeline and evidence map (2026-07-21)

### 16.1 Maintained model contract

The maintained request-only path is no longer the M4A feature-ladder model.
It has three explicit stages:

1. `scheduler_sim.py` predicts admission, prefill, decode, and completion from
   released requests and the recorded server limits.
2. `simulated_ledger.py` conserves exact per-iteration GEMM FLOPs, attention
   FLOPs, attention bytes, phase decomposition, weight traffic, iteration rate,
   and scheduled tokens on a 250 ms grid.
3. `clean_power_surfaces.json` maps those channels to watts with separate dense
   and MoE laws. Dense coefficients are hardware-local; MoE coefficients are
   architecture-specific and fail closed outside declared support.

The clean dense artifact is schema `clean-separated-power-surfaces-v4`. Its
A100/H100 laws each contain four coordinates: idle floor, active resident-weight
fraction, compute utilization, and duty-aware square-root memory utilization.
The current cache rebuild from the maintained simulator is array-identical to
the cache named by the artifact. The artifact binds its cache and timing-fit
hashes, but generated clean artifacts and reports are currently untracked in
the worktree; paper freeze requires versioning them and binding generator and
evaluator code hashes as well.

### 16.2 Clean legacy result

`power-test/clean_power_report.json` is the current result for the 450 legacy
runs:

| surface | role | runs | energy error | RMSE W/GPU | range NRMSE | Soft-DTW | ACF-profile R2 |
|---|---|---:|---:|---:|---:|---:|---:|
| dense | source fit | 225 | 1.68% | 13.10 | 0.057 | 0.0032 | 0.9941 |
| dense | transfer twin | 108 | 1.97% | 12.30 | 0.055 | 0.0029 | 0.9952 |
| dense | rate-4 stress | 45 | 2.99% | 14.09 | 0.089 | 0.0081 | 0.7295 |
| MoE | source fit | 60 | 1.27% | 10.91 | 0.064 | 0.0037 | 0.9676 |
| MoE | rate-4 stress | 12 | 2.08% | 11.66 | 0.063 | 0.0032 | 0.9160 |

These numbers demonstrate fit quality and retrospective stress behavior. The
source and twin workloads reuse nearly identical request sequences, so their
small difference is not an independent transfer claim.

### 16.3 Coverage-basis held-out experiment

The versioned coverage experiment trains on 117 traces from 39 complete cells
and holds out all repetitions from 111 other cells. Its result is:

| held-out axis | traces | cells | E2E timing | energy error | RMSE W/GPU | ACF-profile R2 |
|---|---:|---:|---:|---:|---:|---:|
| new rates | 117 | 39 | 3.91% | 1.06% | 13.10 | 0.9822 |
| new TP setups | 108 | 36 | 3.56% | 3.50% | 14.60 | 0.9776 |
| new model setups | 108 | 36 | 2.55% | 2.10% | 13.17 | 0.9953 |
| all held out | 333 | 111 | 3.26% | 1.97% | 13.58 | 0.9894 |

The model-wise report additionally gives median normalized pointwise errors of
5.34--9.84% of measured range and `100*sqrt(Soft-DTW)` values of 5.27--9.14%
across seven models. That transformed Soft-DTW statistic is explicitly a trace
shape error; it must not be conflated with ACF-profile R2.

This is the best broad development evidence, but the split was designed and
inspected locally. It is a versioned experiment, not the final untouched
external claim and not a replacement for the default fit.

### 16.4 External development boundaries

`power-test/clean_expansion_report.json` scores current external bundles without
refitting:

- three A100 Qwen3-8B rate/shape runs: 5.73--6.48% timing and 3.06--3.55%
  energy error;
- H100 TP4 same-marks state control: 2.15% timing and 5.27% energy error;
- current GPT-OSS-120B A100 runs: 30--33% timing and about 24% energy error;
- current 405B H100 runs: 20--25% timing and 10--14% energy error, with
  insufficient power coverage for Soft-DTW and ACF;
- BurstGPT development run: 1.93% timing but 12.70% energy error under the
  current clean artifact;
- TraceLab cache-off/on: 20.89%/8.58% energy error, with a cache treatment that
  is invalid because prompt/output hashes differ;
- Gemma-4-26B-A4B: unsupported because no architecture-specific MoE surface
  exists.

These failures reject universal rate, elapsed-time, or MoE correction factors.
They also show why engine protocol, power state, telemetry coverage, and exact
cache-pair identity belong in the evidence contract rather than in footnotes.

### 16.5 MoE status

The measured-routing ablation is rejected. Relative to a uniform cache rebuilt
from the same code, measured routing worsens GPT-OSS-20B development energy from
15.52% to 17.75% and rate-4 energy from 17.23% to 22.41%; timing effects are
mixed. Measured routing remains a diagnostic work-coordinate experiment, not
the deployed default.

The separate GPT-OSS-20B MoE v3 surface is supported only for A100 TP1/2 under
independent-uniform routing. On ten retrospective development runs it improves
median energy from 15.52% to 0.92% and range NRMSE from 0.140 to 0.065. On six
opened rate-4 stress runs it improves energy from 17.23% to 1.16% and NRMSE
from 0.233 to 0.052. GPT-OSS-120B TP4/8 remains an unsupported comparator; its
frozen dense baseline is 3.92% energy error and 0.084 NRMSE. Do not generalize
the 20B MoE law across model scale, TP support, or routing mode.

### 16.6 Sealed campaign state

The registered matrix requires eleven bundles: three BurstGPT, six OpenHands,
one unseen Qwen3-14B dense run, and one unseen Qwen3-30B-A3B H100 MoE run. Ten
are present. The missing MoE bundle prevents the scorer's exact campaign-matrix
check from passing. The cache-pair comparator also currently compares
run-specific server launch/ready epochs as if they were treatment invariants;
that must be corrected before an official report can be produced, while still
checking the actual engine configuration and replay identity.

Direct per-run diagnostics with the frozen clean artifacts show:

- BurstGPT: 3.19--7.04% timing, 3.82--15.25% energy, and Soft-DTW
  0.0039--0.0103;
- unseen Qwen3-14B: 1.42% timing, 4.58% energy, 0.143 range NRMSE, and 0.0156
  Soft-DTW;
- OpenHands cache-off: 2.88--4.22% timing and 17.11--20.94% energy;
- OpenHands cache-on: 5.04--7.39% timing and 6.81--8.19% energy.

Those diagnostics are not a substitute for the absent campaign report. All
OpenHands targets have now been inspected, so the hardware-calibrated result in
Section 15 is necessarily retrospective. Any confirmatory claim requires a new
untouched agentic set after the calibration rule is frozen.

### 16.7 Paper-safe findings

The current evidence supports these statements:

1. Exact request/timing-ledger work supports broad held-out rate, TP, and model
   transfer within measured deployment support.
2. Unseen agentic event structure, tool waits, growing contexts, and prefix
   cache behavior can be simulated without agentic traces in source fitting.
3. Nominal accelerator names do not identify deployment-local watt response;
   a small hardware calibration can recover magnitude while leaving workload
   simulation frozen.
4. MoE power requires explicit architecture and routing support; a universal
   dense or entropy multiplier is not supported.
5. Energy, pointwise magnitude, Soft-DTW trace shape, distribution agreement,
   and ACF-profile agreement are distinct measurements. ACF R2 is not a
   temporal-fidelity or event-alignment metric.

The evidence does not yet support these stronger statements:

- zero-shot absolute-watt transfer across arbitrary systems carrying the same
  GPU product name;
- a sealed few-shot hardware-calibration result;
- universal MoE transfer across model scale, TP, hardware, or routing mode;
- a complete paper-final sealed campaign pass.

### 16.8 Current continuation point

Do not tune another shared surface against the opened targets. The smallest
honest continuation is:

1. Add product name, chassis/deployment identity, current/default/enforced power
   limit, P-state, clocks, and cap/thermal event reasons to every calibration
   and validation bundle. Hardware support must include these fields rather
   than the string `A100` alone.
2. On the accessible 300 W A100, collect short non-agentic idle, pure-prefill,
   and sustained-decode probes. Estimate only the predeclared idle,
   ordinary-dynamic, and prefill gains; freeze the adapter before agentic data.
3. Collect fresh untouched agentic packs on that same platform and score the
   frozen scheduler, power surface, and hardware adapter once.
4. Collect or formally retire the missing Qwen3-30B-A3B H100 MoE sealed cell.
   Unsupported retirement narrows the claim; it must not be counted as a pass.
5. Fix cache-pair comparison so run epochs are recorded provenance but only
   treatment-invariant server controls are compared. Add a regression test
   before rescoring.
6. Version the clean and coverage artifacts, their input hashes, generator and
   evaluator code hashes, exact calibration population, and generated reports
   before paper freeze.

The resulting paper story is not that hardware differences are free. It is
that the workload simulator transfers broadly, while the remaining
deployment-specific watt map is low-dimensional, measurable with a short
calibration, and separable from workload-specific retraining.
