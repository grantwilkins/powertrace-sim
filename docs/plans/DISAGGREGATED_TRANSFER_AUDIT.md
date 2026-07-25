# Disaggregated transfer audit

Date: 2026-07-25

## Conclusion

The confirmation data are internally healthy, but the campaign does not test
the phase-balanced regime its design assumed. It combines a persistently
backlogged prefiller with short, transport-dominated decoder bursts. The
resulting traces are expected from the measured system and do not indicate
corrupt `nvidia-smi` data.

Four effects must be separated:

1. The 2 requests/s prefiller is overloaded, so its trace is nearly constant.
2. The simulator omits the measured NIXL transfer and post-processing stage.
3. A 64-token decode lasts only about 0.3--0.4 seconds and is poorly resolved
   by point samples at 250 ms.
4. The measured-replay analysis uses a marker written before variable client
   initialization, corrupting the 1 request/s replay comparison.

The current campaign therefore does not show that a separate disaggregated
model is necessary. It shows that the next test must avoid saturation, expose
both phases for multiple native samples, and model the causal handoff.

## What changed from the pilot

The pilot and confirmation are different phase regimes:

| Property | Pilot | Confirmation |
| --- | ---: | ---: |
| Mean input length | 286--303 tokens | 8,265--8,325 tokens |
| Mean output length | 198--217 tokens | 64 tokens |
| Scheduler token budget | 8,192 | 2,048 |
| Prefix cache | enabled | disabled |
| Dominant phase | decode | prefill |

The pilot's long outputs and higher offered rates made decoder work nearly
continuous. Its warm repeats also made prefill small or cached. The
confirmation reverses both conditions: each prompt needs several 2,048-token
chunks, while each decode produces only one or two native power observations.
The pilot's smoother decoder is consequently not evidence that the new
decoder data are malformed.

## Confirmed profiling findings

### Prefill is saturated, not demonstrably throttled

The realized 2 requests/s arrival streams offer about 1.88--2.02 requests/s.
Measured completion throughput stabilizes near 1.73 requests/s. The prefiller
therefore accumulates work:

- mean waiting queue: 26--40 requests;
- maximum waiting queue: 55--83 requests;
- mean GPU utilization: about 91%;
- active power after warm-up: about 427--437 W;
- median prefill HTTP duration: 19.8--24.2 seconds, versus about 0.8 seconds
  at 1 request/s.

The SM clock remains exactly 1,410 MHz throughout every captured cell. Token
throughput does not fall as temperature rises. There is no evidence of a
thermal clock collapse. The profile did not retain power-limit or clock-event
reason fields, so a static application power cap cannot be ruled out, but it
is not needed to explain the flat trace.

The campaign selected 2 requests/s because the source timing model predicted a
roughly 372 ms isolated prefill. The first measured request takes 764 ms. The
load-selection error alone is enough to turn the intended 60% duty cell into a
backlogged cell.

### NIXL is successful but absent from the simulator

Every request records one NIXL transfer, with no failures, notification
failures, expirations, or preemptions. Across all five cells, the decoder
reports approximately:

- 43--45 ms transfer time;
- 37.8 ms post-processing time;
- 81--83 ms total handoff time.

Measured decoder first-byte latency is 90--93 ms, while the simulator predicts
about 9.4 ms. The missing handoff almost exactly explains this gap. Post-first-
byte decode service is much closer: 301--307 ms measured versus 296--301 ms
after the diagnostic decode timing scale.

This should be represented as an explicit causal stage between prefill
completion and decoder admission. It is not a fitted trace lag.

### Short decode bursts are under-observed at 250 ms

A 64-token decode lasts only about 0.3--0.4 seconds, giving one or two raw
power samples per request. Small changes in request admission, meter response,
or query phase therefore switch a point between idle and active power. Fixed
one-second means raise the representative decode correlation from 0.172 to
0.505, confirming that sub-second sampling explains part of the dense trace,
but decode amplitude and variance remain underpredicted.

Event-based diagnostics also show a causal power response on the order of
100--150 ms. The dense power path has an explicit response operator, while the
selected MoE path does not. A response term is a plausible later model change,
but it must be frozen from calibration evidence and tested on raw heldout
samples rather than used to align each trace.

### The replay origin is wrong

`workload_start_epoch_s` is written before the benchmark process constructs and
re-tokenizes hundreds of long random prompts. First client traffic occurs
19.8--28.3 seconds later. The two identical 1 request/s repeats differ by 6.16
seconds in this initialization interval even though their recorded request
trajectories are effectively identical.

Pairing replay samples relative to `workload_start_epoch_s` produces the
reported prefill/decode correlations of 0.155/0.031. Re-anchoring each repeat
to its first recorded client or proxy request, an observed boundary rather
than a fitted lag, raises them to roughly 0.90--0.93 for prefill and
0.73--0.76 for decode.

This bug affects the measured-replay gate. It does not affect per-cell
model-versus-measurement scores, which already use recorded request
timestamps.

## Confirmed model limitations

The post-hoc 2.068x prefill timing scale is diagnostic, not a general fix. It
is fitted from one isolated request and scales compute efficiency, bandwidth
efficiency, launch overhead, and sampling overhead together. It predicts
queue-free 1 request/s behavior reasonably, but overpredicts the backlogged
2 requests/s queue:

| Condition | Measured median prefill | Scaled prediction |
| --- | ---: | ---: |
| 1 request/s heldout | 804--811 ms | about 925 ms |
| 2 requests/s heldout | 19.8--19.9 s | 29.9--30.3 s |
| 2 requests/s calibration | 24.2 s | about 37.0 s |

The actual prefiller gains batching efficiency under backlog. One global
service-time scalar cannot represent both isolated and batched service.

The visually strong high-rate prefill trace is also weak evidence for the
detailed model. In one heldout cell, model correlation is 0.934 while the
equal-parameter busy-duty null reaches 0.933; the model/null loss ratio is
0.993. Both predict a flat plateau because both know the engine is always
busy.

The GPT-OSS MoE power surface was learned from colocated, predominantly
short-prompt workloads. It has no explicit transfer/DMA feature and no general
MoE meter-response operator. The roughly 10.7x prefill dynamic gain is evidence
that isolated long-prefill magnitude is outside the source surface's
well-supported regime. Decode mean energy is closer, but its active amplitude
and variance remain too small.

## Ruled out

The retained evidence rules out:

- prefix-cache contamination in the confirmation;
- failed, missing, or expired NIXL transfers;
- request-token accounting errors;
- GPU identity changes or incomplete paired samples;
- preemption;
- broken median 250 ms cadence;
- a progressive temperature-driven throughput collapse.

Engine and proxy debug logs were not retained, so connector selection and
effective vLLM configuration cannot be reconstructed from logs. Runtime
version, topology, proxy events, and aggregate engine counters are retained.

## Minimum next campaign

Keep the existing stack unchanged:

- two A100 TP1 roles;
- vLLM 0.22 and NIXL;
- cache disabled;
- 2,048-token scheduler budget;
- raw `nvidia-smi power.draw` at 250 ms;
- no smoothing, interpolation, fitted lag, or warping.

Run three core five-minute cells:

1. calibration: approximately 8k input, 256 output, 0.5 requests/s;
2. heldout: the same configuration with an independent workload seed;
3. replay: an exact repeat of the heldout request plan.

At the measured service rates, 0.5 requests/s leaves the prefiller
unsaturated. Increasing output length to 256 makes each decode visible for
roughly four or five native samples. This produces useful temporal variation
on both GPUs instead of a prefill plateau and one-sample decode impulses.

If budget permits one additional cell, run 8k input, 64 output, 0.5 requests/s
as a heldout short-decode stress test. Failure there should be labeled a
250-ms observability/response boundary rather than allowed to invalidate the
observable 256-token result.

Before measured traffic:

1. construct and tokenize the complete deterministic request plan;
2. record planned send offsets;
3. write `traffic_start_epoch_s` immediately before scheduling the first
   request;
4. retain actual client and proxy receive timestamps.

Add an abort gate for sustained queueing. The campaign should fail its
phase-transfer purpose if the prefill wait queue is persistently nonzero.
Record `power.limit`, pstate, and clock-event reasons without changing the raw
power field or cadence, subject to the existing query-duration gate.

One short sequential probe of 20--30 requests before the three cells can
identify isolated prefill service, aggregate NIXL handoff, and decoder service.
It must use latency and counters, not power, to choose the safe offered rate.

## Minimal model and decision sequence

1. Correct the replay origin. This is an analysis fix, not calibration.
2. Add an explicit NIXL handoff delay derived from retained transfer/post
   counters. Do not represent it as a fitted trace shift.
3. Replace the one-request global timing scale with a calibration restricted
   to queue-free prefill service. Do not claim backlogged timing transfer from
   it.
4. Retain one idle and one nonnegative dynamic-power gain per role.
5. Evaluate the three core cells at native 250 ms and report fixed one-second
   means only as a secondary diagnostic.

Only add a fixed MoE response kernel if the observable heldout decoder still
shows a causal response mismatch. Only add phase-specific power training if
the role gains fail across the unsaturated heldout cells. Only revise the
scheduler if the intended claim expands to backlogged disaggregated serving.

This sequence tests the narrow claim that the existing work model transfers to
cache-disabled disaggregation with a small handoff term and minimal role
calibration. It does not expand scope to cache prediction, arbitrary overload,
or a separately trained disaggregated model.
