# MoE Timing and Power Pipeline Plan

Status: adversarially replanned 2026-07-18 from GPT-OSS routing captures,
timing probes, power residuals, deployed runtime logs, primary sources, and
three independent reviews.
## 1. Decision and invariant

The dense pipeline is frozen. MoE models get a separate architecture schema,
expected-execution record, timing artifact, ledger, and power artifact. The
outer request scheduler, wall-clock binning, response chain, metrics, and
reporting contracts remain shared.

The known dense artifacts are:

- `power-test/fitted_surface.json`, SHA-256
  `f92a8c8a6d6d6579bd0fca1a18de09197eebc7324f3dbb0f245071c42baf96e0`;
- `timing-test/fitted_efficiencies.json`, SHA-256
  `55aa7e8ddc86b8ea4c2e4f35d46574e11155cf600fa15511a85693c16e01ecd0`;
- the ledger and response-chain versions to which those fits were made.

The checked-in power fit predates the corrected weight-ledger semantics. It
must not be paired with the corrected-uniform ledger and called a coherent
dense baseline. Stage 0 must bind a versioned tuple `(ledger numerical digest,
fit hash, response-chain version)` for the legacy dense path. Because this plan
may not change dense behavior, the corrected-uniform cache remains retrospective
MoE evidence rather than a reason to refit dense. Do not use an NPZ archive hash
as a numerical invariant; hash canonical arrays by `run_id`.

No MoE stage may refit either dense artifact. Dense dispatch must not load a
routing, MoE timing, or MoE power artifact. Existing dense iteration traces,
ledger channels, design rows, predictions, and metrics must remain bit-identical
by `run_id`; new `moe_*` channels must be zero on dense rows.

## 2. What the completed routing experiment says

The captures contain 114,384 tokens per model and the selected top-4 expert IDs
for every layer. GPT-OSS-20B has 24 layers and 32 experts; GPT-OSS-120B has 36
layers and 128 experts. They identify selected-expert count histograms,
occupancy, and load entropy. They do not contain router logits, gate confidence,
or realized HBM transactions.

Measured routing reduces the expected logical footprint of distinct expert
weights by about 12-14% on matched rate-4 traces. This is not a measured DRAM
traffic reduction: a kernel may reload weights across CTAs, and cache retention
is runtime-dependent.

The like-for-like power result rejects measured routing as a complete model:

| evaluation role | uniform energy / ACF-MAE / ACF R2 / NRMSE | measured routing |
|---|---:|---:|
| GPT-OSS-20B development | 15.52% / 0.0130 / 0.950 / 0.140 | 17.75% / 0.0141 / 0.942 / 0.151 |
| GPT-OSS-20B rate-4 | 17.23% / 0.0274 / 0.913 / 0.233 | 22.41% / 0.0439 / 0.756 / 0.303 |
| GPT-OSS-120B development transfer | 3.92% / 0.0164 / 0.959 / 0.084 | 4.89% / 0.0169 / 0.934 / 0.092 |

ACF R2 should increase; the other three metrics should decrease. Routing-aware
timing improves GPT-OSS-120B median decode error from 12.8% to 10.8% at TP4 and
from 6.6% to 6.2% at TP8, while regressing GPT-OSS-20B. Routing is a physically
motivated candidate work coordinate, but these results show that the current
compressed feature is neither sufficient nor yet demonstrated necessary for
predictive timing or power.

The current routing law is too compressed for causal timing. Its fitted
distinct-expert curve has RMSE of 1.85/32 and 2.79/32 experts for 20B
prefill/decode, and 4.43/128 and 8.30/128 for 120B. Marginal touch probabilities
and one exponent cannot recover maximum load, block rounding, or the complete
expert problem list. The next artifact must preserve empirical joint per-token
top-k assignment records across layers, with expert-count vectors and
aligned-slot histograms stored only as derived views.

## 3. Why the blue trace is a shifted red trace

The frozen A100 dense surface is driven mainly by its static floor, decode duty,
and memory ramp. Its fitted iteration-rate and tokens-per-iteration
coefficients are zero. Changing routing therefore changes one large logical
weight term but leaves almost all waveform coordinates unchanged. That produces
a near-vertical shift rather than the missing load-dependent shape.

Busy-bin residuals, measured minus predicted watts per GPU, show the missing
shape:

| model / TP | rate 0.125 | rate 1 | rate 4 |
|---|---:|---:|---:|
| GPT-OSS-20B TP1 | +21.8 W | +39.1 W | +58.0 W |
| GPT-OSS-20B TP2 | +6.1 W | +22.8 W | +31.9 W |
| GPT-OSS-120B TP4 | -16.3 W | -3.3 W | +24.8 W |
| GPT-OSS-120B TP8 | -13.9 W | -6.4 W | +3.9 W |

A constant MoE offset cannot cross sign with load. A single model-independent
monotone correction in logical traffic cannot explain these opposite-signed
residuals. The pattern motivates testing execution shape and duty before
fitting one MoE-specific energy correction.

At decode batch one, a 20B fixed-batch probe is about 6.9 ms at both TP2 and
TP4. This establishes a poorly scaling floor, but not its cause. It may combine
kernel geometry, dequantization, graph/launch behavior, host scheduling, and
communication.

## 4. Runtime facts that change the model

The realistic A100 serving logs use vLLM 0.11.0, BF16 activations,
`quantization=mxfp4`, async scheduling, CUDA graphs, custom all-reduce enabled,
and tensor parallelism only. They explicitly report that A100 lacks native FP4
compute and selects weight-only FP4 Marlin. Current runs do not enable expert
parallelism, so there is no expert-dispatch all-to-all term to learn.

The existing fixed-batch probes use vLLM 0.10.1.1 without the serving
async-scheduler contract. They can reveal trends, but their absolute intercept
must not calibrate the v0.11.0 serving path.

In the deployed Marlin implementation:

- the hidden dimension is rounded to 256 and the per-TP expert intermediate
  dimension to 128;
- the runtime selects the first `b` in `(8, 16, 32, 48, 64)` satisfying
  `M * top_k / n_experts / b < 0.9`;
- selected token IDs are sorted by expert and each expert group is aligned to
  `b`;
- two Marlin expert GEMMs, activation, top-k weighting, and reduction form the
  fused path.

Applying only this declared grouping rule to the ShareGPT decode captures gives:

| model | useful tokens `M` | Marlin `b` | measured-route aligned expert-token slots / useful routed assignments | uniform top-4 expectation |
|---|---:|---:|---:|---:|
| 20B | 1 / 4 / 16 / 32 / 64 | 8 / 8 / 8 / 8 / 16 | 8.00 / 5.73 / 2.72 / 1.89 / 2.03 | 8.00 / 6.62 / 3.53 / 2.00 / 2.00 |
| 120B | 1 / 4 / 16 / 32 / 64 | 8 / 8 / 8 / 8 / 8 | 8.00 / 6.90 / 4.56 / 3.33 / 2.27 | 8.00 / 7.63 / 6.37 / 5.10 / 3.48 |

The uniform column is the exact aligned-slot expectation under independent
uniform top-4 subsets across tokens:
`E * b * E[ceil(C / b)] / (M * top_k)`, where for each expert
`C ~ Binomial(M, top_k / E)`. Relative to that reference, measured routing
usually reduces aligned slots, most strongly for 120B, but the 20B `M=64` case
slightly increases them because block rounding makes concentration
non-monotone. This gives a concrete timing hypothesis without turning entropy
into a coefficient.

These are aligned expert-token-slot ratios, not physical tile, executed-FLOP,
or DRAM-byte measurements. Invalid rows may be masked or skipped, and physical traffic
depends on the kernel schedule. The nonlinearity and the 20B block change at
`M=64` nevertheless show why an average tokens-per-expert feature is
insufficient.

Primary evidence:

- [OpenAI GPT-OSS architecture and quantization](https://openai.com/index/introducing-gpt-oss/)
- [GPT-OSS-120B released configuration](https://huggingface.co/openai/gpt-oss-120b/blob/bc75b44b8a2a116a0e4c6659bcd1b7969885f423/config.json)
- [vLLM 0.11 Marlin MoE path](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/layers/fused_moe/fused_marlin_moe.py)
- [vLLM 0.11 MXFP4 backend selection and dimension padding](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/layers/quantization/mxfp4.py)
- [vLLM 0.11 expert sorting and alignment](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/layers/fused_moe/moe_align_block_size.py)
- [MARLIN mixed-precision inference](https://arxiv.org/abs/2408.11743)
- [MegaBlocks on padding versus block-sparse MoE execution](https://proceedings.mlsys.org/paper_files/paper/2023/file/5a54f79333768effe7e8927bcccffe40-Paper-mlsys2023.pdf)

## 5. Correct probabilistic timing contract

The scheduler knows the request-visible iteration plan:

```text
q_k = (prefill chunks, decode sequences, contexts, domains, phase composition)
```

It cannot know realized routes from arrival time and request lengths. Let
`A_{k,l}` be the empirical `[M_k, top_k]` selected-expert assignment matrix and
`H_{k,l} = hist(A_{k,l})`. Draw the joint route record across layers from an
empirical distribution conditioned on `q_k`, source domain, checkpoint, and
prompt-rendering contract:

```text
A_k = (A_{k,l})_l ~ R(. | q_k, domain)
```

The first defensible TP-only timing model is:

```text
E[T_k | q_k] =
    T_residual_nonmoe(q_k, TP, runtime)
  + sum_l E_A[g(runtime_hash, TP, execution_mode, A_{k,l}, K_l, N_l)]
```

`g` is the measured inclusive duration of the exact routed MoE path. Its first
version covers router/sort, both expert projections and activation, combine,
and any exposed TP synchronization. Split it only when profiler timelines
prove mutually exclusive critical-path intervals. Summing kernel durations,
host launch time, and asynchronous collectives would double count overlap.
`T_residual_nonmoe` contains sampling and the residual attention/KV/scheduler
critical path, but excludes router, routed expert execution, combine, and any
TP synchronization attributed to `g`. Identify the two terms from one
non-overlapping profiler partition or a full-iteration ablation, then fit and
hold out the residual from request-visible `q_k`. `execution_mode` contains the
graph/eager mode, capture shape, useful tokens by phase, and prefill chunk
layout; semantic phase alone is not a timing coefficient.

The expectation must be evaluated over empirical assignment records. If a
later validated lookup reduces each assignment record to its count vector `H`,
then:

```text
E[g(H)] need not equal g(E[H])
```

because maximum load, block rounding, tactic selection, and kernel duration are
nonlinear. Use deterministic quadrature over stored empirical samples for the
default simulation. A stochastic route-variability experiment must use an
explicit seed.

The Marlin operator consumes the full assignment matrix, not only its count
histogram. If a reduced lookup is desired, first prove by held counterfactuals
that token/expert incidence is immaterial and a sorted histogram of per-expert
aligned blocks is sufficient. Entropy, expected distinct experts, mean tokens
per expert, and global maximum load are diagnostics, not timing coefficients.
Deterministic marginal quadrature suffices for the conditional mean sum. A
stochastic timing/power trace must sample joint route records that preserve
token/request and cross-layer sample IDs; independent per-layer draws are a
separate ablation.

Expert parallelism is a future, separate execution graph:

```text
T_EP = T_dispatch_A2A + T_expert_by_placement + T_return_A2A
```

It requires EP-enabled runs, expert placement, rank-load distributions,
topology, and the all-to-all backend. Current TP data identify none of it.

## 6. Schemas and integration boundary

Do not add nested values to the current scalar `ARCH` dictionaries; existing
paths cast their values to float. Add separate immutable schemas:

```text
MoeArchitecture:
  explicit is_moe, routed/dense layer indices, experts, top_k,
  hidden/intermediate sizes, shared experts, module bytes and formats

MoeRuntime:
  engine version/revision, container, CUDA/GPU, resolved MoE backend,
  rounded dimensions, TP/DP/EP, collective backend, graph mode,
  async scheduling, chunk budget, max sequences

IterationPlan:
  request IDs, decode contexts/domains, prefill chunks and prior contexts

MoeIterationExecution:
  expected routing samples, logical work, artifact hashes, validity/support
```

Add a request adapter with an optional routing-domain label; leave tuple
requests valid. Retrospective `n_out` is an oracle workload mark. Prospective
scheduling must receive an output-length estimate or distribution.

Architecture dispatch must be explicit and fail closed:

- dense plus either routing mode returns the existing dense path without
  inspecting MoE artifacts;
- MoE uniform uses the current reference without loading a measured law;
- MoE measured requires exact checkpoint, capture, domain, backend, and support
  matches;
- missing, mutable, or mismatched artifacts raise.

The next routing artifact stores source labels, token/request and joint
cross-layer sample IDs, chat-template contract, checkpoint/tokenizer revision,
capture runtime/dtype, raw top-k assignment samples or stable references to
them, and derived count vectors by model, source, phase, and supported group
size.
Before use, compare selected expert IDs on a small matched-token subset between
the capture runtime and served vLLM runtime.

Every record/cache sidecar persists schema versions, exact
`routing_artifact_sha256`, `timing_artifact_sha256`, `runtime_hash`, and
support-cell ID. Evaluation consumes these cache-bound values and verifies
hashes; it never rereads a mutable routing law to reconstruct provenance.

The MoE ledger consumes `MoeIterationExecution` directly and labels inferred
channels `expected_*`. M2 contains counts and logical work only, with
`moe_execution_valid=true`, `moe_timing_valid=false`, and timing fields absent
or NaN rather than zero. M3 adds inclusive duration and duty with a verified
timing artifact. Later power fields have a separate `moe_power_valid` mask.
The ledger must not reconstruct nonlinear execution shape from bin averages.

## 7. Power contract

The frozen dense coefficients are aggregate empirical predictors, not
module-level energy prices. Do not decompose them into common, expert, dispatch,
and communication watts.

Because the frozen dense surface and corrected MoE ledger are not a coherent
fit tuple, M4 uses a separate replacement energy contract:

```text
P_equilibrium(t) =
    P_idle(runtime, TP)
  + sum_i J_dynamic_total(q_i, execution_i, runtime, TP) / duration_i
          * 1[t is in iteration interval i]
```

Exact interval/bin overlap conserves each iteration's joules before the shared
response chain maps equilibrium power to meter power.
`J_dynamic_total` is one complete per-server-iteration dynamic-energy surface
trained on matched full-server repeated workloads after idle subtraction. Long
fused-MoE operator loops identify and preregister candidate routing/shape
coordinates, but do not set the absolute total-energy scale. If a later
decomposition is used, it must be a non-overlapping
`J_nonmoe(q_i) + delta_J_moe(execution_i)` identified from paired full-iteration
ablations. Require measured MoE idle to be statistically indistinguishable from
the versioned dense reference or retain an explicit runtime/TP idle value.
Pre-register the finite basis/knots and exclude model/checkpoint ID.
Deterministic prediction uses the same route support as M3 to compute
`E_A[J_dynamic_total(q,A)]`, never energy at a mean route. Stochastic prediction
draws one joint `A_i` for both duration and energy so their covariance is
preserved.

Freeze timing before fitting power. Long profiler-free repeated operator loops
estimate joules per iteration; millisecond NVML samples cannot identify
per-kernel instantaneous power. Randomize experiment blocks, stabilize
clock/temperature conditions, and collect at least five independent
repetitions.

[A measurement study of NVIDIA power telemetry](https://arxiv.org/abs/2312.02741)
documents why sensor update behavior must be handled explicitly.

## 8. Reduced candidate ladder

| candidate | change | acceptance question |
|---|---|---|
| M0 | current uniform reference | frozen comparison |
| M1 | measured logical routing footprint | completed; honest footprint, rejected as final model |
| M2 | empirical assignment-record execution and MoE ledger | can request plans produce supported expected execution coordinates? |
| M3 | complete exact-runtime MoE timing artifact | does `E[g(A)]` plus the non-MoE residual predict held server timing? |
| M4 | complete MoE-server iteration-energy surface plus measured idle | does frozen timing improve held joules and trace metrics? |
| M5 | exposed launch or TP term | only after a direct critical-path experiment identifies it |
| M6 | separate EP model | only after an explicit EP campaign |

There is no entropy rung. Padding survives only as backend geometry inside M3
and only if held tile-boundary probes validate it.

## 9. Identifying experiments

### Stage 0: freeze and audit

1. Bind the legacy dense ledger digest, fit hash, and response-chain version as
   one coherent immutable tuple; do not pair the old fit with corrected rows.
2. Record the exact serving version/commit/container, selected MoE and
   all-reduce kernels, graph mode, scheduler, GPU, and TP/EP settings.
3. Treat all existing GPT-OSS power and timing results as retrospective.

### Stage 1: routing distributions

1. Split by whole request ID before summaries, then recapture explicit
   ShareGPT and SWE/agentic domains with the served chat template and labels.
2. Store full joint assignment records, not only marginal touches or counts.
3. Run capture-versus-serving route parity on matched tokens.
4. Freeze deterministic grouping of contiguous request-level route records into
   supported iteration plans. Measured mode requires an explicit domain/mixture
   artifact ID; legacy tuples remain valid only for dense/uniform mode.
5. Construct mixed-phase distributions by sampling joint assignment records.
   Expected touches are only a logical distinct-weight proxy with no ordering
   guarantee relative to physical HBM traffic.
6. On training requests, capture replayable hidden states, gate weights/logits,
   scales, and reduced pre/post-expert activation norms/sparsity. Hold out whole
   request blocks and reject activation-dependent claims unless these improve
   prediction beyond phase, tokens, and assignment shape.

### Stage 2: exact-backend operator probes

Use the deployed Marlin fused-MoE operator. Hold useful routed assignments
fixed while varying:

- balanced versus skewed full count vectors at equal nonempty expert count;
- different nonempty counts at fixed assignments;
- pairs with equal entropy/active count but different tile histograms;
- global useful-token `M` around every Marlin `b` tactic change;
- individual expert counts `C_e = j*b-1, j*b, j*b+1`, controlling the remaining
  assignments, to isolate alignment from tactic selection;
- actual 20B/120B shapes and both expert projections.

Define start/end GPU events in the deployed layer graph and enumerate included
router, sort, projections, activation, combine, and synchronization operations.
Use identical weights/scales, streams, graph capture, warmup, and kernel
selection. Standalone probes identify shape contrasts; an in-server event check
must validate their absolute mapping. Only split operators if stable profiler
boundaries demonstrate non-overlap. Run TP1 source probes, TP2 source
calibration, and a predeclared TP4 operator-validation subset under the exact
v0.11 graph/custom-all-reduce contract. Run separate long, randomized,
profiler-free power loops. Cross route-shape contrasts with at least two real
activation-value strata while holding values fixed within each pair; otherwise
treat operator energy only as a bound and fit full-server energy.

### Stage 3: matched full-server validation

After the operator shape law is frozen, rerun fixed batches with the exact
v0.11 serving runtime:

- decode batches `1, 4, 16, 64`;
- contexts near `2k` and `32k`;
- prefill and mixed-phase staircases;
- engine-scheduled tokens, exact graph capture size, clocks, temperature,
  power, and iteration duration.

Define server iteration wall time as the union of inclusive MoE GPU intervals
plus a non-overlapping residual non-MoE critical path. On 20B TP2 source cells,
fit and freeze the prospective residual model from request-visible iteration
state; validate on held context/batch cells. The same randomized full-server
source repetitions measure total joules per completed iteration after idle
subtraction for M4; standalone operator loops set candidate shape coordinates,
not absolute server energy. Add a TP communication term only if profiling
identifies exposed critical-path collective time. Freeze this complete M3
artifact before opening 20B TP4 and 120B TP4 server holdouts.
Because TP4 operator validation was used, 120B TP4 is a held system-composition
test, not zero-shot TP/backend transfer.

### Stage 4: realistic traces and transfer

Freeze timing and the M4 energy basis before scoring 20B/120B source stress,
cross-family MoE, and finally sealed H100/Qwen. A different backend outside
measured support is an explicit OOD failure, not evidence of transfer.

### Stage 5: scheduling utility

After M4 is frozen, replay identical sealed arrivals under the baseline and a
routing-aware policy using only prospective expected timing/work. Hold the SLA,
objective, and paired seeds fixed; do not refit from scheduling outcomes.
Report latency/SLO, throughput, and energy, and deploy only for a predeclared
Pareto improvement or tradeoff. Both policies must remain inside frozen M3/M4
support; unsupported actions fail closed to baseline and their rate is reported.

## 10. Splits and gates

Split by whole request for routing and by independent randomized repetition for
timing/power, never by tokens, bins, or alternating shape cells. Use a discarded
pilot to estimate paired run-block variance and preregister the repetitions
needed to detect a 5% mechanism effect at the chosen power. Five repetitions is
a sensor-CV minimum, not a universal inferential sample size. Fit, development,
confirmatory mechanism, and untouched source-test repetitions are disjoint;
keep time/temperature blocks intact. Timing and power have frozen, independent
split manifests, and confidence intervals use repetition-level paired
contrasts.
Before every confirmatory operator, server, trace, or transfer test, bind
artifact, split, support, and metric hashes. A failure retires that candidate;
any revision requires fresh confirmatory data.

A mechanism is admitted only if:

- runtime and support hashes match and the design is identifiable;
- a controlled contrast changes the target by at least 5%;
- its within-cell sign/shape is stable and run-level 95% confidence excludes
  zero; predeclared cross-regime zero crossings are allowed;
- it reduces held operator timing median absolute error by at least 20% and
  0.1 ms, without worsening a held regime by more than one percentage point;
- held operator timing median error is at most 5% and P90 at most 10%;
- padding predicts held boundary discontinuities; launch predicts held layer
  counts; communication predicts directly measured exposed collective time;
- power-loop CV is at most 2%, and M4 reduces held joules/iteration error by at
  least 20% without an unreplicated within-cell sign reversal.

Only then evaluate trace-level gates:

- energy error median at most 5%, P90 at most 10%, worst at most 15%;
- median ACF R2 at least 0.85 and ACF-MAE P90 at most 0.10;
- median range NRMSE at most 0.15;
- median decode and end-to-end timing error at most 10%.

Preregister the ACF lag set, ACF R2/MAE definitions, range-NRMSE denominator
and low-range exclusion, aggregation unit, and absolute-error definitions.
Each held trace is scored before aggregation. M3 is selected only on its timing
manifest and then frozen; M4 is selected only on its independent held-joules
and trace energy/ACF/NRMSE manifest. Both require paired improvement over M0/M1
with run-level uncertainty. Stage 5 has its own scheduling manifest. Plateau
exclusion is fixed before predictions; ACF gates do not apply to plateaus.

## 11. Implementation order and definition of done

1. Add schemas, provenance, explicit `is_moe`, and fail-closed dispatch without
   changing prediction. Introduce
   `moe_iteration_executor(IterationPlan) -> MoeIterationExecution` beside the
   existing dense callback; preserve the full prefill-chunk vector and request
   IDs before scalar aggregation. Dense keeps its callback and trace unchanged.
2. Build the empirical routing distribution and M2 expected-execution ledger.
3. Add measurement ingestion and support checks without fitting.
4. Fit and freeze operator `g` on source operator data, then fit and validate
   `T_residual_nonmoe` on predeclared source-server train/development cells.
   Freeze the complete M3 artifact before opening server holdouts.
5. Fit one M4 total server-iteration energy surface on power-training
   repetitions only.
6. Add M5/M6 only after their own experiments pass the mechanism gates.

Every stage must test:

- bit-identical dense work, scheduler records, existing ledger channels,
  predictions, metrics, and frozen artifact hashes;
- exact hand-worked assignment matrices and count vectors, support failure,
  deterministic expectation, and conservation of routed assignments;
- two assignment matrices with the same count vector remain distinct until a
  held sufficiency test permits reduction;
- direct `E_A[g(A)]` and `E_A[J(q,A)]` quadrature for nonlinear fixtures;
- changing MoE artifacts or targets cannot change dense results;
- missing or mismatched measured artifacts fail for MoE and cannot affect dense;
- the full suite passes with `uv run -m pytest -x`.
The next implementation milestone is M2 plus the measurement schema. It makes
no timing claim. The first fitted milestone is M3, and the first power change is
M4 after M3 passes held timing gates.
