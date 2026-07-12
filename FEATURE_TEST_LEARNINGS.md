# FEATURE_TEST_LEARNINGS: failure analysis and continuation map

Status: handoff after the 2026-07-11 v2 evaluation (`results/feature_test_v2/`),
which followed the measured-ITL rebuild and a cited-constant correction pass.
This document records what has been established, why the remaining cells fail,
and how to continue without turning target errors into constants.

Read this after `FEATURE_TEST_PLAN.md` and before changing candidates. Treat
`results/feature_test_v2/transfer_scorecard.csv` and its `selected_model.json`
as the current numerical truth (`results/feature_test_v1/` is the
pre-correction snapshot). Much of the lower half of `feature-test/README.md`
is historical exploration and is not the frozen selection result.

## 1. Current state in one paragraph

Source-only selection still chooses M4A on both hardwares; it passes 7 of 13
cells and no candidate passes all gates. Relative to v1, three input
corrections with cited or measured provenance (405B FP8 weight bytes 487.23e9
instead of 405.85e9; FP8 FLOP fraction 0.7996 instead of a blanket 0.5 scale;
a step-identified meter kernel matching published NVML metering behavior)
improved the 405B energy cell to 4.70% (was 5.72%) and fixed the H100 TP8
ACF-MAE gate (0.114, was 0.143), while regressing H100 hold-TP1 energy (7.64%)
through the changed 405B source fit. A new candidate, M0c (phase-anchored
concave physics mean, 16 scalars, no request-state columns, board-power cap),
is not selected — its source-development energy trails M4A — but it passes all
four H100 TP holdouts including hold-TP1, has the best H100 dynamics in the
ladder (S2a ACF R2 0.991), and lifts 405B median ACF R2 from negative values
to 0.602 while over-predicting 405B energy by 8-11%. This is still not a
passing model. Do not describe it as one.

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

### 2.2 Energy and temporal fidelity constrain nearly orthogonal directions

Full-run energy constrains the zero-frequency component; ACF constrains the
normalized spectrum. v2 exhibits the trade concretely: on the 405B cell, M4A
gets energy right (4.70%) with collapsed dynamics (ACF R2 -0.615), while M0c
gets dynamics largely right (0.602; rates 2-4 positive) with energy 8-11%
over. Neither passes. The information missing from the ledger is exactly what
would let one model do both.

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
`auto`). These corrections improved every candidate's 405B energy and are
independent of model choice.

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

- plan: `FEATURE_TEST_PLAN.md`;
- shared ledger: `model/training_data/ledger_view.py`; arch constants with
  citations: `model/training_data/arch.py`;
- physics kernel (including the M0c concave basis): `model/classifiers/physics.py`;
- evaluator: `feature-test/evaluate_candidates.py` (M0c staged fit:
  `_m0c_coefficients`; board-power caps: `TDP_W_PER_GPU`);
- meter kernel: `feature-test/meter_kernel.json`, regenerated by
  `feature-test/identify_meter_kernel.py`;
- gates: `feature-test/gates.py`;
- current results: `results/feature_test_v2/` (pre-correction snapshot:
  `results/feature_test_v1/`).

Rebuild and evaluate:

```bash
uv run python feature-test/build_ledger_cache.py \
  --dt 0.25 \
  --out feature-test/ledger_cache_250ms.npz

uv run python feature-test/evaluate_candidates.py \
  --ledger-cache feature-test/ledger_cache_250ms.npz \
  --run-index feature-test/ledger_cache_250ms.runs.json \
  --out-dir results/feature_test_v2
```

Before trusting any result:

```bash
uv run -m pytest -x
uv run -m pytest -x feature-test/tests
```

## 9. Final lesson

v1's lesson stands: the data identify a good average power surface over the
source manifold but not a unique map from request timing to power dynamics
where the engine changes its batch and clock behavior. v2 adds the sharper
version: every constant that could be audited against a served artifact or a
published measurement was worth auditing (two of three were wrong in ways
that moved transfer cells), and the two remaining failure axes now have
names — tensor-parallel synchronization power and MoE iteration granularity —
with specific probes that would identify them. The next breakthrough is those
probes, not another surface.
