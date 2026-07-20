# timing-test: first-principles request-timing model

Read `DESIGN.md` first (frozen 2026-07-16 before any fitting). The model
predicts per-request time to first token, decode duration, and end-to-end
latency from a marked arrival process (arrival time, prompt length, output
length) through three layers: architecture-derived work per iteration, a
roofline iteration-time rule with per-hardware efficiencies and a
layer-scaled launch/synchronization overhead (the latency term of the
alpha-beta communication model), and a discrete-event continuous-batching
scheduler simulation (chunked prefill, decode-first, seat and KV-capacity
admission). Nine fitted scalars per hardware; no per-model constants.

Pipeline:

```bash
uv run python timing-test/build_timing_dataset.py      # per-request dataset + frozen splits
uv run python timing-test/build_probe_calibration.py   # probe-level calibration points
uv run python timing-test/fit_efficiencies.py          # training cells only
uv run python timing-test/evaluate_timing.py --out-dir results/timing_test_v1
uv run -m pytest -q timing-test/tests
```

The retrospective component-accounting revision can be evaluated without
overwriting frozen artifacts:

```bash
uv run python timing-test/fit_efficiencies.py \
  --out /tmp/powertrace_fitted_efficiencies_v2.json
uv run python timing-test/evaluate_timing.py \
  --manifest split_manifest_fp8.json \
  --fitted /tmp/powertrace_fitted_efficiencies_v2.json \
  --out-dir /tmp/powertrace_timing_v2
uv run python timing-test/evaluate_expansion.py \
  --timing-fit /tmp/powertrace_fitted_efficiencies_v2.json \
  --power-fit /tmp/powertrace_fitted_surface_v2.json \
  --out /tmp/powertrace_expansion_v2.json
```

This revision separates embedding lookup, transformer work, and output-head
projection; preserves per-request prefill contexts; binds bundle engine limits;
and represents cached prefixes as context rather than executed prefill. It is
development evidence because Qwen residuals motivated the change. On the
legacy matrix, overall cell-median end-to-end error improves from 4.13% to
3.63%, but the H100 FP8 405B model holdout regresses, so this is not yet a
universal replacement for the frozen model.

Fitted values (`fitted_efficiencies.json`): compute efficiency 0.75 (A100)
/ 0.59 (H100) of peak tensor throughput; bandwidth efficiency 0.89 / 1.00
of peak HBM bandwidth; per-message latency 51-65 us (A100) / 38-45 us
(H100) times two messages per layer; first-token overhead 4.5 / 7.8 ms;
per-generated-token sampling cost 139 / 44 us. Training-fit quality:
log-RMSE 0.10 / 0.07 over 43 / 66 points spanning batch 1-256, contexts to
122k tokens, and six models.

A dense/MoE structural split was tested and REJECTED by the training data:
the implied per-message overhead of the mixture-of-experts model (61-64 us)
sits inside the dense population (37-68 us). What the data demanded instead
was the batch-linear sampling term above, backed out from the high-batch
staircase residuals. Two fitted values deserve suspicion and are flagged,
not hidden: the A100 sampling cost (139 us/token) is implausibly high for
sampling alone and likely absorbs high-batch attention inefficiency; the
H100 bandwidth efficiency sits at its 1.0 bound, meaning dense-BF16 weight
streaming is effectively at peak and the FP8 405B (slower per byte) gets
over-optimistic memory times. Engine iteration counters from the pending
gpt-oss iteration probes would separate both.

Two structural revisions were adopted after training-data diagnostics (no
holdout contact), both standard in the analytical-performance literature
(Vidur MLSys'24, GenZ, LLMCompass, NeuSight): (1) per-operator roofline
times are SUMMED — the GEMM sweep and the batched-attention pass run
sequentially, so a single whole-iteration max under-predicts at high
batch; (2) the fit consumes loaded-run inter-token latencies at
reconstructed concurrency (interval-overlap of decode windows), giving
dense (batch, context) coverage for every training model and GPU count —
profiler-grade signal recovered from the serving logs themselves.

Result summary (`results/timing_test_v1/cell_metrics.csv`, 150 cells;
"baseline" is the per-config log fit from measured median rates, which has
per-model calibration the principled model refuses):

| role | e2e median error | baseline | gates |
|---|---|---|---|
| in-domain test (H100 / A100) | 3.0% / 5.4% | 3.7% / 10.2% | 73/80 |
| held-out rate 4.0 | 6.6% / 5.6% | 6.9% / 11.4% | 15/16 |
| held-out twin (zero-shot) | 2.9% (H100 ds-8b) / 5.6% (A100 ds-70b) | 3.4% / 18.8% | 34/36 |
| held-out model (zero-shot) | 13.6% (H100 405B) / 9.4% (A100 gpt-oss-120b) | 15.0% / 8.9% | 6/18 |

128 of 150 cells pass the frozen targets; the model beats or ties the
per-config log fit in 110 of 150 cells while never seeing per-model data.
Median time-to-first-token error is 3-21 ms everywhere. The zero-shot
mixture-of-experts cell that motivated a dense/MoE split opened at +26%
and closed at +8-17% (median 9.4%) through the two structural revisions —
no expert-class parameters were needed.

The remaining systematic failure in the v1 split was llama-3-405b (-10%
at low rate to -25% at rate 4): a dtype effect with no training signal,
since 405B is the only FP8 checkpoint and was fully held out. Split
amendment v2 (user-directed 2026-07-16; `fit_fp8_bandwidth.py`,
`split_manifest_fp8.json`) moves 405B repeats 0-1 at rates <= 2 into a
`dtype_calibration` role and fits ONE class constant per hardware: FP8
weights stream at 0.781 of the dense-BF16 effective bandwidth (3.5%
residual over 3,470 requests). It is a dtype constant shared by all FP8
models — any FP8 deployment could supply it — not a per-model refit. On
the untouched 405B test cells (repeat 2 and rate 4,
`results/timing_test_v1_fp8/`): 2.9-8.1% median end-to-end error at rates
<= 2 (log-fit baseline: 4.7-21.4%), 12.5% at rate 4 where unmodeled
preemption remains (the new bundles record nonzero preemption counters;
the simulator's count is structurally zero).

The gpt-oss iteration probes (synced 2026-07-16) produced three findings:
(1) an engine-configuration conflict — the legacy gpt-oss serving used
async scheduling and the probes did not, so the same model measures 3.9 vs
6.9 ms per token at batch 1; both are correct for their mode, absolute
probe latencies are therefore excluded from the legacy-serving fit
(`fit_efficiencies.py`), and the llama probes, whose configuration matches
their legacy runs (29.35 vs 28.93 ms), stay. Engine configuration is part
of the model's contract and calibration data must match the deployment
mode. (2) The probes' internal consistency is excellent (client latencies
and engine iteration counters agree to 0.1 ms). (3) An attempted
routing-correlation fit from the staircase's configuration-independent
latency deltas is NOT identifiable (`fit_moe_routing.py`, report-only):
the expert weight sweep hides below the operator roofline max in latency,
so separating routing from per-token overhead requires router/expert
counters — the same instrumentation the power-side analysis called for.
The gpt-oss-120b probes remain quarantined until its zero-shot scoring is
frozen.

A Vidur-style random-forest regressor over the identical fitting points
and architecture features was run as a ceiling diagnostic. In-domain it
has no edge (4.8% cross-validated vs 3.2% analytical). Zero-shot it
collapses exactly where transfer matters: 34% on gpt-oss-120b and 61% on
llama-3-405b (vs 12.9% and 2.4% analytical), while trivially memorizing
the architecture twins (1.4%). Random forests fit Vidur's contract —
profile the target configuration, interpolate — and are the wrong tool
for the no-per-deployment-profiling contract here.
