# Data inventory and minimal replay campaign

Status: evidence-backed campaign plan, 2026-07-18.

Implementation status: the bounded canonical-plan builder, TraceLab adapter,
exact-arrival/direct-token replay runner, controlled Gamma arrivals,
cache/reasoning accounting, TP8 telemetry profile and paired campaign, and
direct MoE router capture now live under `profiling/`. Submission rejects
missing local datasets, trace plans, containers, and incomplete model snapshots
before requesting GPUs.

This document answers one question: what is the smallest additional campaign
that can support claims about non-ShareGPT workloads, long contexts, coding
agents, reasoning, Qwen transfer, and mixture-of-experts inference without
wasting Sherlock time?

The decision is to reuse the existing controlled measurements, repair the
software/data handoff first, and collect only measurements that identify a
currently missing mechanism. Broad model, rate, context, or dataset sweeps are
not justified.

## 1. Executive decision

Do not submit the existing Qwen-235B Tier-1 campaigns or another dense/MoE
staircase sweep.

Before interpreting or fitting any new workload bundle:

1. Reconcile architecture metadata between the legacy registry and canonical
   manifests.
2. Project and merge the 26 complete canonical bundles into the timing/power
   path.
3. Regenerate the checked-in power artifacts after the exact weight-traffic
   correction.
4. Bind the timing twin to the recorded engine configuration, especially the
   token budget, KV dtype, preemption, and prefix-cache behavior.
5. Resolve or explicitly exclude the high-load 70B TP8 hardware-state regime.
6. Replace the MoE uniform-routing assumption with directly observed router
   choices.

The independent collection jobs may run in parallel after offline preflight.
Predictions must be frozen before their corresponding power bundles are opened
for grading. The default live campaign is:

- two telemetry-rich Llama-70B runs to identify the TP8 state transition;
- one offline router pass for each existing gpt-oss model;
- one same-marks Qwen3-8B transfer run on A100 and one on H100;
- three A100 jobs that compare an off-grid rate and bursty/Poisson/smooth
  arrivals while holding prompts and mean rate fixed;
- two parallel cache-off/cache-on TraceLab jobs on Qwen3-8B/A100;
- one exact BurstGPT arrival replay;
- one Gemma-4-26B-A4B routing pass and A100 transfer validation.

No separate reasoning, LMSYS, Azure, or LongBench GPU run is in the default
campaign. Reasoning remains ordinary decode; the data sources otherwise produce
deterministic CPU-side schedules before submission.

## 2. What already exists

### 2.1 Legacy serving corpus

`timing-test/timing_dataset.npz` contains:

| dimension | coverage |
|---|---|
| runs | 450 |
| requests with exact ITLs | 354,125 |
| hardware | A100 and H100 |
| model identities | 7 |
| configurations | 25 |
| tensor parallelism | TP1, TP2, TP4, TP8 where applicable |
| rates | 0.125, 0.25, 0.5, 1, 2, 4 requests/s |
| repeats | 3 per cell |
| roles | 160 train, 80 in-domain, 108 twin, 54 model, 48 rate |

Models include dense Llama/DeepSeek 8B and 70B, FP8 Llama-405B, and MoE
gpt-oss-20B/120B. The request scheduler is a deterministic function of
`(arrival_s, input_tokens, output_tokens)`; arbitrary marked schedules can be
replayed without new model calls.

The serving corpus is not long-context evidence. Its prompt
minimum/P50/P90/P99/maximum is 4/95/621/802/1,020 tokens. Output
minimum/P50/P90/P99/maximum is 1/134/485/768/1,838 tokens. All empirically
graded arrivals are stationary Poisson traffic at the six rates above.

### 2.2 Canonical bundles

There are 26 complete bundles under `data/runs/`. Every complete bundle has
validated `manifest.json`, `requests.json`, `power.csv`, and `engine.csv`.

| group | complete bundles | useful coverage |
|---|---:|---|
| A100 Llama-70B TP4 | 6 | idle, decode staircase, 2k/8k/32k context grid, prefill to 65k, context holds to about 123k, transients |
| H100 Llama-70B | 10 | full TP8 probe suite including mixed grid; matched TP4 decode, context, and prefill probes |
| A100 gpt-oss-20B | 4 | TP2/TP4 decode staircases and 2k/8k/32k context grids |
| A100 gpt-oss-120B | 4 | TP4 decode/context probes plus ShareGPT rates 1 and 2 |
| H100 Llama-405B | 2 | ShareGPT rates 1 and 2 |

The expensive operator surface is therefore already broad: batch 1-256,
controlled prompts/contexts 8-122,888 tokens, and outputs 1-2,048 tokens.
What is absent is end-to-end serving validation on realistic long contexts,
non-Poisson arrivals, agent sessions, and Qwen models.

Nine of the original 35 development bundles are absent: the A100 Llama-70B
mixed grid, two gpt-oss-20B realistic controls, and six matched H100
Llama-70B TP4/TP8 realistic controls. They are not all prerequisites for the
new workload campaign.

### 2.3 Current timing and power evidence

- The timing model passes 128/150 frozen cells and transfers zero-shot to
  gpt-oss-120B at 9.4% median end-to-end error.
- Dense arrival-only energy transfer is already strong. The unresolved result
  is temporal fidelity for rate-4 70B TP8.
- All twelve affected legacy runs change power regime at 304.2-306.5 seconds,
  by 12.35-19.04 W/GPU. Timing and occupancy align well, so another timing
  sweep is not the next measurement.
- The power surface was fit on dense bins only. Existing MoE cells have not
  identified or graded the routing-dependent weight traffic.
- The checked-in fitted surface and arrival-only report predate the exact
  weight-traffic correction and must be regenerated before their numbers are
  cited.

### 2.4 Existing replay scaffolding

The repository already has:

- seeded synthetic multi-turn sessions;
- real-text SWE-smith trajectory ingestion;
- monotonically growing prompts;
- per-turn tool classes and pauses;
- paired prefix-cache regimes;
- deterministic temperature-zero, fixed-length generation;
- canonical bundle emission with session/turn metadata.

This path reproduces request work, context growth, and pauses. It does not run
the recorded tools. That is the correct abstraction for power replay: a tool
call is an idle interval followed by new prefill text.

The checked-in `gap_params.json` is not empirical. It has zero fitted samples
and contains literature priors. Also, no agentic bundle has been collected.

## 3. Software and data blockers before GPU submission

These are gates, not optional cleanup.

### G0. Architecture identity

Reconcile and hash one descriptor per served checkpoint.

- Legacy gpt-oss active parameters, weight bytes, and routed fraction disagree
  with the canonical manifests.
- Legacy and canonical Llama-405B weight bytes disagree, the canonical
  manifest omits the FP8 FLOP fraction, and the checkpoint identities differ.
- The campaign path does not currently pass the measured checkpoint footprint
  through every metadata layer.

No cache merge, transfer fit, or MoE result is admissible until one descriptor
identity is used end to end.

### G1. Canonical bundle to timing/power handoff

Implement the post-collection path already required by
`profiling/MODEL_READINESS_RUNBOOK.md`:

1. one prefill calibration per model/hardware/TP;
2. one 250 ms projection per configuration;
3. cache merge with conserved work and provenance;
4. roles derived from manifest metadata;
5. immutable candidate/artifact freeze;
6. score-only validation.

The current timing dataset builder is legacy-only and cannot directly grade
the new bundles.

### G2. Engine configuration in the twin

The replay must use manifest values rather than timing defaults.

- Current twin token budget: 2,048; current campaigns commonly use 8,192.
- Current KV sizing assumes BF16 rather than the recorded KV dtype.
- Prefix-cache hits do not reduce executed prefill in the twin.
- Preemption is absent.
- A request larger than KV capacity can be omitted instead of failing the run.

Every replay must fail explicitly on an unsupported request or configuration.

### G3. Telemetry and token accounting

Before the TP8 diagnostic, add P-state, power limit, and clock-event/throttle
reasons to the existing temperature/clock logger.

Before a reasoning replay, count every generated reasoning token. The current
agent streamer counts normal content deltas but does not establish complete
reasoning-content accounting.

### G4. Current artifact regeneration

On CPU, rebuild the simulated ledger, joined power cache, dense surface, and
arrival-only report after the exact weight-traffic correction. This is the
baseline against which every later decision is made.

Also reconcile the timing README with `fitted_efficiencies.json`; its reported
sampling costs and fitting-point counts currently describe an older artifact.

## 4. Online data source decisions

### Use

| source | use in this project | why |
|---|---|---|
| [TraceLab v0.0.1](https://github.com/uw-syfi/TraceLab) | primary coding-agent schedule and cache replay | 357,161 real Claude/Codex rounds, ordered timing events, prompt/cache splits, input/output lengths, exact tool waits, session structure, a released replay client, and CC BY 4.0 data |
| [OpenHands evaluation outputs](https://huggingface.co/datasets/OpenHands/openhands-evaluation-outputs) | real tool text and an independent tool-gap/routing sample | timestamps, actions, observations, raw response usage, cached-token counts, and tool-call metadata; MIT, but heterogeneous and poorly documented |
| [SWE-smith trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) | real coding/tool text when OpenHands ingestion is inconvenient | 24.1k structured tool trajectories and an existing local adapter; MIT; no timestamps |
| [Azure LLM Inference 2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md) | CPU-only code versus conversation arrival/length schedules | invocation timestamps and input/output token counts with separate code/conversation traces |
| [BurstGPT](https://github.com/HPMLL/BurstGPT) | CPU-only burst and conversation-session schedule | timestamps, session IDs, request/response lengths, and conversation/API labels; CC BY 4.0 |
| [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | optional alternative chat text/length distribution | one million multilingual conversations; no timestamps or tools; gated and non-redistributable |
| [LongBench v2](https://github.com/THUDM/LongBench) | a few real long-context content anchors | real document/dialogue/code contexts; select only examples inside the served native window |
| [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | reasoning output-length strata, offline first | recorded reasoning completions up to 16k tokens; no timestamps |

TraceLab supersedes fabricated gap distributions for the primary agentic
claim. Its sanitized release omits raw private tool inputs, but raw text is not
needed to reproduce dense work or cache shape. Use OpenHands or SWE-smith text
for content-sensitive MoE routing measurements.

### Do not use as primary replay sources

- SWE-bench task rows are task/environment definitions, not serving traces.
- LMSYS-Chat-1M has neither arrivals nor session timing.
- Live SWE-bench/OpenHands repository execution adds package, network, and
  environment variance unrelated to GPU inference power.
- ToolBench live RapidAPI replay is externally unstable.
- A full public MoE routing corpus is unnecessary and may be extremely large;
  measure the target checkpoints directly.

## 5. Replay contract

Every workload is normalized to an ordered table:

```text
session_id
round_idx
arrival_or_ready_s
prefix_tokens
new_input_tokens
output_tokens
post_tool_wait_s
cached_prefix_tokens
source_id
source_revision
```

Rules:

1. Open-loop chat/arrival traces use `arrival_or_ready_s` directly.
2. Agent turns are closed loop: the next turn starts only after the prior
   generation completes and `post_tool_wait_s` elapses.
3. Tool execution is never performed live. The recorded observation or an
   exact-length deterministic token sequence becomes the next turn's added
   prefill.
4. Cache-off executes the full prompt. Cache-on executes only the uncached
   suffix reported/planned by the trace.
5. Live generated output is recorded, but a load-replay run carries the
   recorded or deterministic token sequence forward so the next prompt shape
   does not drift.
6. A pinned source revision, tokenizer, model checkpoint, server image, seed,
   sampler, and normalized-plan SHA-256 are stored in the bundle.
7. Context overflow, KV infeasibility, missing reasoning tokens, or a mismatch
   between planned and measured cached tokens fails the bundle.

TraceLab's released replay client already demonstrates the desired direct
token-ID, closed-loop, exact-prefix approach. Reuse its data contract; the
repository does not need to rerun private tools or synthesize semantic answers.

## 6. Exact minimal campaign

### Phase 0: CPU-only readiness

Run G0-G4. Then build three offline replay families:

1. Azure Code and Conversation: one contiguous 10-minute window each.
2. BurstGPT: one 10-minute exact-timestamp window of independent requests.
3. TraceLab: eight sessions, two from each maximum-context band
   `[4k,8k)`, `[8k,16k)`, `[16k,24k)`, and `[24k,31k]`, selected by a
   fixed seed and preserving all included rounds and exact waits.

Simulate every normalized schedule through the timing/ledger path.

Default decision: no live Azure, BurstGPT, or LMSYS run. Add one only when more
than 5% of its busy 250 ms bins fall outside the min/max support of the
candidate live workloads on at least one of:

- compute utilization;
- memory utilization;
- iterations/s;
- tokens/iteration;
- running requests;
- waiting requests;
- effective context.

This test concerns workload support, not semantic content.

### Phase 1: identify the dense TP8 state

Collect exactly two new H100 runs using the same Llama-70B rate-4 marks and
seed as an affected legacy cell:

| run | TP | schedule | required telemetry |
|---|---:|---|---|
| D1 | 8 | 180 s idle, then 420 s rate-4 workload | per-GPU power, temperature, SM/memory clocks, P-state, power limit, clock-event/throttle reasons |
| D2 | 4 | identical idle and request marks | same |

The existing no-delay 600-second affected traces are the comparison. The
180-second shift distinguishes a roughly 305-second server/runtime phase from
a roughly 305-second workload/temperature/dose threshold without another
duration sweep.

Workload-only cost: `8*10/60 + 4*10/60 = 2.0` H100 GPU-hours.

Decision:

- If a request-visible or telemetry-visible trigger transfers between the
  existing and new TP8 run without harming the TP4 control, implement it and
  rerun the frozen evaluator.
- Otherwise document the support boundary as dense 70B, TP8, sustained
  high-load operation beyond the observed transition. Do not fit another
  retrospective step.

No later power validation is interpreted until this decision is recorded;
independent collection may already be queued or complete.

### Phase 2: identify MoE routing

For `openai/gpt-oss-20b` and `openai/gpt-oss-120b`, run one offline,
temperature-zero forward/generation pass per model and log selected expert IDs
per token and layer.

Use two fixed content strata per model:

- 64 reconstructed ShareGPT sequences;
- 64 real coding/tool sequences from OpenHands or SWE-smith.

Evaluate co-scheduled token group sizes `B = 1, 4, 16, 64, 256`. Double each
stratum to 128 only if either:

- the bootstrap 95% interval for expected distinct experts is wider than 2%
  of the expert count at any `B`; or
- the 64-to-128 estimate changes by more than 1%.

Hard cap: 128 sequences per stratum per model and 1 A100 GPU-hour per model.
The output is a versioned routing-law artifact plus expert-load, entropy, and
cross-distribution comparisons.

Then replace the uniform-independent expectation and re-score all existing
gpt-oss controlled and serving data. Do not recollect a staircase.

Decision:

- If the frozen gpt-oss timing and power gates pass, the existing measurements
  support the initial MoE result.
- If ShareGPT and coding routing laws differ materially, select the law by
  workload class; do not hide the difference in a global scalar.
- If the corrected model still fails, collect only the two missing
  gpt-oss-20B ShareGPT controls at rates 1 and 2 to isolate model scale.

### Phase 3: dense Qwen and arrival transfer

Run five independent jobs with one common ShareGPT sample and seed:

| run | hardware/TP | rate | Gamma shape | identifying comparison |
|---|---|---:|---:|---|
| QD-A | A100/TP1 | 4.0 | 1.0 | dense Qwen transfer |
| QD-H | H100/TP1 | 4.0 | 1.0 | hardware transfer versus QD-A |
| QR | A100/TP1 | 2.5 | 1.0 | off-grid rate versus QD-A |
| QB | A100/TP1 | 2.5 | 0.25 | bursty pattern versus QR |
| QS | A100/TP1 | 2.5 | 4.0 | smooth pattern versus QR |

All use 200 prompts and seed `20260712`. Shape 1 is Poisson; shapes 0.25 and 4
have interarrival coefficients of variation 2 and 0.5. Freeze predictions
before submission. Grade prediction error inside each run; raw total energy
across different rates is not itself a transfer metric.

Decision:

- Pass when median run energy error is at most 6%, timing median absolute
  end-to-end error is at most 10%, and no unmodeled systematic residual exceeds
  the existing dense gate.
- On pass, do not run Qwen3-14B or Qwen3-32B.
- On fail, add the smallest matched anchor that identifies the failed term;
  do not launch the full Qwen-235B Tier-1 campaign.

### Phase 4: one realistic agentic/long-context validation

On Qwen3-8B/A100 TP1, replay the same eight TraceLab sessions from Phase 0
twice:

| run | prefix cache | purpose |
|---|---|---|
| QA-off | disabled | validates full repeated prefill, closed-loop pauses, and growing contexts |
| QA-on | enabled | validates executed-suffix accounting and measured cache hits |

The two runs use identical normalized rows and direct token IDs. They jointly
cover coding-agent arrivals, tool pauses, closed-loop sessions, and contexts
up to 31k; no separate live LongBench run is needed.

Pass criteria:

- exact session/round conservation;
- output-token conservation;
- measured cached prompt tokens within one server cache block per round of the
  plan;
- median timing error at most 10%;
- median energy error at most 6%;
- ACF-MAE at most the existing dense threshold;
- no overflow, silent omission, or unexplained preemption.

If the cache-off run fails, stop: the issue is not prefix caching. If cache-off
passes and cache-on fails, fix executed-prefill/cache accounting and rerun only
cache-on.

### Phase 5: cross-family Gemma MoE transfer

Use `google/gemma-4-26B-A4B-it`, the Gemma 4 MoE checkpoint: 128 experts,
top-8 routing, and about 4B active parameters. Keep A100 hardware fixed.

1. Measure the Gemma router law with the Phase-2 64+64 protocol.
2. Freeze the prediction.
3. Run one Gemma/A100 TP2 ShareGPT validation with the QD-A marks.

Pass threshold: the same 6% energy and 10% timing limits used for Qwen3-8B.
Do not use the existing Gemma calibration or roofline probes before scoring;
otherwise this is no longer zero-shot cross-family transfer. Gemma requires the
dedicated Gemma 4 container. Model facts are bound to the
[official config](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/config.json)
and [vLLM recipe](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html).

## 7. Reasoning decision

Visible chain-of-thought or `<think>...</think>` tokens are autoregressive
decode tokens. They do not justify a separate power phase. Existing
DeepSeek-R1-distill/Llama architecture twins already test that semantic labels
do not alter the architecture-derived work.

The model input must count:

```text
output_tokens = reasoning_tokens + final_answer_tokens + tool_call_tokens
```

Report those components separately for workload interpretation, but sum them
for timing and power. A dedicated reasoning run is added only if the serving
stack uses a mechanically different path such as speculative decoding,
separate hidden-token accounting, or a different model/engine mode.

If that condition is met, select one OpenR1 example from each recorded output
band `<1k`, `1-4k`, `4-8k`, and `8-16k`; run one example per band first and
add a second only when within-band error exceeds the frozen model error.

## 8. Resource ceiling and stopping rule

Default new workload cost, excluding model load and scheduler wait:

| phase | A100 GPU-hours | H100 GPU-hours |
|---|---:|---:|
| TP8 state diagnostic | 0 | 2.0 |
| gpt-oss router logging | at most 2.0 | 0 |
| Qwen3-8B same-marks transfer | less than 0.1 | less than 0.1 |
| controlled arrival rate/pattern | less than 0.2 | 0 |
| paired TraceLab replay | cap at 1.0 | 0 |
| exact BurstGPT replay | cap at 0.5 | 0 |
| Gemma routing and transfer | cap at 1.0 | 0 |
| default total | cap at 4.8 | about 2.1 |

Slurm reservations should include model load and shutdown time, but a job must
request only its TP degree; the current submit script already does this.

Stop the campaign as soon as the declared claim is supported or a support
boundary is identified. In particular:

- no new dense staircase;
- no new gpt-oss staircase;
- no full LMSYS replay;
- no full SWE-bench environment execution;
- no Qwen3-14B/32B ladder after an 8B pass;
- no Qwen3-235B Tier-1 anchor for the cross-family MoE claim;
- no second hardware for agentic distribution validation unless the first
  hardware shows a hardware-specific residual.

## 9. Result matrix

Each final claim has one identifying comparison:

| claim | evidence |
|---|---|
| short-context dense timing/power | existing 450-run corpus |
| controlled long-context operator physics | existing Llama-70B probes to about 123k |
| realistic long-context/session execution | QA-off |
| prefix-cache accounting | QA-on minus QA-off, identical rows |
| arbitrary arrival rate | QD-A versus QR |
| controlled non-Poisson pattern | QR versus QB/QS |
| exact recorded arrival pattern | BurstGPT replay |
| reasoning semantics | all generated tokens counted as decode; existing architecture twins |
| dense Qwen transfer | QD-A and QD-H |
| gpt-oss MoE result | direct router law plus existing gpt-oss bundles |
| cross-family MoE transfer | Gemma router law plus frozen Gemma validation |
| sustained TP8 support boundary | D1/D2 plus existing affected traces |

This matrix is intentionally sparse. Every new run either resolves one named
ambiguity or grades one frozen transfer claim.
