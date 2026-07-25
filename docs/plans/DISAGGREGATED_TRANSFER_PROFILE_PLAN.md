# Minimal disaggregated transfer profile

Status: implemented; collection pending, 2026-07-25.

## Question

Can the existing PowerTrace work model predict separate prefill- and
decode-GPU power with six declared target scalars, without training a new
disaggregated model?

This campaign tests cache-disabled, nonsaturated disaggregation. It does not
test caching or overloaded queue behavior.

## Fixed stack

Do not change:

- GPT-OSS-20B on two A100-80GB TP1 roles;
- vLLM 0.22, NIXL, asynchronous scheduling, and chunked prefill;
- `max_num_batched_tokens=2048`, `max_num_seqs=256`;
- prefix caching disabled;
- raw `nvidia-smi power.draw` queried every 250 ms;
- 30-second pre- and post-traffic idle windows.

Use one engine launch and one allocation. Request 90 minutes to tolerate model
startup; expected measured runtime is about 22 minutes.

## Traffic

Use fixed-deadline arrivals rather than Poisson arrivals:

- input: 8,192 tokens with the existing +/-25% range;
- output: 256 tokens with the existing +/-25% range;
- rate: 0.5 requests/s, one request every 2 seconds;
- traffic duration: 300 seconds per main cell;
- 150 requests and about 1,200 raw power samples per main cell.

Measured service times imply roughly 40% prefill duty and 65--80% decode duty.
Both phases should be visible for several native samples without persistent
queueing.

Create and tokenize each deterministic request plan before starting telemetry.
Persist the prompt, token lengths, output length, fixed send offset, and plan
hash. The replay must reuse the exact heldout plan.

After the first valid power-query midpoint, choose a planned traffic origin at
least 30 seconds later on the same 250 ms cadence. Send against absolute
monotonic deadlines. Persist that origin and use the first `proxy_received`
event as an independent check, not as a fitted alignment.

## Four cells

| Cell | Requests | Seed/plan | Role |
| --- | ---: | --- | --- |
| `stage-probe` | 24 at 0.2 requests/s | probe plan | timing calibration only |
| `rate-0p5-calibration` | 150 | seed 11 | power calibration |
| `rate-0p5-heldout` | 150 | seed 17 | independent evaluation |
| `rate-0p5-replay` | 150 | exact heldout plan | evaluation and repeatability |

The stage probe isolates requests enough to estimate prefill service and NIXL
handoff. It is not a result cell.

Do not add a 64-token or higher-rate cell yet. If the core campaign passes, a
single 8k/64 heldout cell can later test the short-phase observability
boundary. If the core campaign fails, more load cells will not identify the
cause.

## Allowed calibration

Fit exactly six scalars:

1. prefill idle power;
2. decode idle power;
3. prefill nonnegative dynamic-power gain;
4. decode nonnegative dynamic-power gain;
5. one positive queue-free prefill service-time scale;
6. one nonnegative fixed NIXL handoff delay.

Fit the timing scale and handoff delay on the stage probe. Fit idles and power
gains separately by role on the calibration cell using pointwise squared loss,
the no-warp diagonal soft-DTW objective. Freeze all six before opening either
evaluation cell.

Estimate NIXL handoff as the median
`decode_sent -> decode_first_byte` duration minus the frozen decoder
first-iteration prediction. The handoff replaces no decoder compute and must
not double-count the source first-token term.

Keep decoder post-first-byte timing frozen. Do not fit response filters,
per-cell lags, queue parameters, batch-dependent scales, or per-cell gains.
The NIXL delay is an explicit stage between prefill completion and decoder
admission, not trace alignment.

## Instrumentation

Retain:

- raw timed two-GPU power rows and GPU UUIDs;
- clocks, utilization, memory use, and temperature;
- proxy request-stage events;
- per-role engine and NIXL counters;
- exact request IDs, token lengths, planned offsets, and plan hashes;
- benchmark, proxy, and both engine logs;
- image identity, git SHA, effective serve commands, and GPU topology.

Add `power.limit`, pstate, and clock-event reasons to the timed query while
leaving `power.draw`, query midpoint, and 250 ms cadence unchanged. The smoke
test must show that the larger query still satisfies the existing 200 ms query
limit.

Phase-lock the power cadence to the planned traffic origin. Require median
query-midpoint phase error at most 25 ms and p95 at most 50 ms relative to the
250 ms traffic grid. This makes raw replay samples comparable without moving
or interpolating either trace.

## Minimal implementation

The implementation uses one narrow explicit-plan path:

1. a small generator writes the JSON plan before telemetry;
2. a small driver reuses the existing request function and result writer but
   reads that plan and sends at its stored monotonic offsets;
3. the runner selects this path only for the disaggregated campaign.

Do not add a general workload framework or change existing benchmark modes.
The driver reads only `--request-plan`, the `--traffic-start` epoch file, the
proxy base URL, and the output path. The implementation is in
`profiling/disaggregated_prefill/planned_workload.py`; campaign metadata and
gates are in `transfer_campaign.py`, and the Sherlock entry point is
`profiling/jobs/disaggregated_transfer_gpt_oss_20b.sbatch`.

Run the four-cell campaign with:

```bash
sbatch profiling/jobs/disaggregated_transfer_gpt_oss_20b.sbatch
```

Pass `smoke` to
`profiling/jobs/run_disaggregated_transfer_gpt_oss_20b.sh` for the
four-request integration path.

## Gates

Reject a cell if any existing identity, cadence, cache, request, NIXL, or idle
gate fails. Also require:

- prefill waiting-queue median `= 0` and p95 `<= 1`;
- no preemption;
- no power-limit change or thermal/hardware slowdown reason;
- median prefill and decode phase duration each at least 0.75 seconds;
- identical heldout/replay token lengths and planned arrival offsets;
- first `proxy_received` within 50 ms of the planned traffic origin;
- meter phase-error limits above;
- first-to-last-third prompt throughput drift at most 10%.

A failed queue gate invalidates the cell. Start a fresh lower-rate campaign;
do not fit around saturation.

## Frozen evaluation

Score every raw query midpoint against its containing 250 ms model bin.
Perform no smoothing, averaging, interpolation, fitted lag, or DTW warping.
One-second means are secondary diagnostics only.

For each role in both evaluation cells require:

- correlation `>= 0.80`;
- standard-deviation ratio in `[0.80, 1.25]`;
- p95 power error `<= 10%`;
- pointwise model MSE `<= 0.95` times the equal-parameter busy-duty null MSE;
- relevant phase-timing median absolute percentage error `<= 15%`.

The equal-parameter null receives the same timing scale and NIXL handoff but
replaces physical work features with simulated role busy duty.

The measured heldout/replay traces must also reach correlation `>= 0.80` per
role when anchored to the recorded traffic origin.

## Decision

- **Both roles pass:** claim cache-disabled disaggregated transfer with six
  target scalars. Report all six; do not call it zero-shot.
- **Timing passes, power does not beat the busy null:** the existing power
  surface has not demonstrated phase transfer; collect phase-specific power
  training data.
- **Queue-free prefill timing fails:** add validated chunk/batch timing data,
  not another global scale.
- **NIXL timing fails:** collect per-request transfer bytes and latency before
  adding a length-dependent handoff model.
- **Decode timing passes but native power fails:** test one preregistered causal
  MoE response model. Do not tune a lag on heldout traces.
- **Measured replay fails:** fix measurement or host stability and reprofile;
  do not change the model.

No additional campaign branch is authorized until this decision is available.
