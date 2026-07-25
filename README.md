# PowerTrace-Sim

PowerTrace-Sim predicts GPU power from LLM requests. The maintained model is a
small deterministic pipeline: a request scheduler produces a 250 ms work
ledger, and frozen dense or bounded-MoE equations convert that work into power.
The historical GMM-BiGRU implementation is preserved under
`archive/gmm_bigru_v1/`; it is not imported by the default package.

## Repository map

- `model/`: maintained training, timing, power, simulation, and CLI code.
- `model/artifacts/powertrace_v1.json`: default frozen release artifact.
- `model/scripts/`: prepare, train, infer, and evaluate entry points.
- `scripts/paper/`: one maintained local paper-regeneration entry point.
- `scripts/eval/`: Azure facility evaluation support.
- `profiling/`: workload collection and Sherlock jobs.
- `archive/gmm_bigru_v1/`: self-contained first-generation model and results.
- `docs/MODEL_PIPELINE.md`: paper-facing end-to-end training/inference flow.
- `docs/PROJECT_OVERVIEW.md`: collaborator-facing model and evidence overview.
- `docs/PAPER_OUTPUTS.md`: paper artifact and evidence contract.
- `docs/plans/DISAGGREGATED_TRANSFER_PROFILE_PLAN.md`: executable minimal
  cache-disabled disaggregated-transfer campaign.
- `docs/plans/DISAGGREGATED_TRANSFER_AUDIT.md`: root-cause evidence behind
  that campaign.
- `docs/plans/`: active paper, campaign, and measurement plans.
- `archive/research_notes/`: superseded plans and detailed failure analyses.
- `CLEAN_MODEL.md`: migration status and remaining external work.

Candidate and ablation scripts under `power-test/`, `timing-test/`, and
`feature-test/` are development support, not additional public model
interfaces. Historical figures and retired paper-support outputs are preserved
under `archive/research_artifacts/`; maintained figures live in
`results/paper/`.

## Setup

Use `uv`; do not install the project with bare `pip`.

```bash
uv sync

# Add only what the task needs.
uv sync --extra train
uv sync --extra paper
uv sync --extra profiling
uv sync --extra archive-bigru
```

## Inference

Run the checked release on an explicit request list:

```bash
uv run -m model.scripts.infer \
    --requests examples/requests.json \
    --deployment llama-3-70b-a100-tp4 \
    --seed 42 \
    --out-dir outputs/example
```

The output directory contains `power.csv`, `requests.csv`, and
`manifest.json`. Inference is fixed at the calibrated 250 ms cadence. The
manifest records the release hash, deployment, support status, seed, and input
identity. Unsupported deployment overrides fail unless explicitly enabled.

Requests contain `arrival_time`, `input_tokens`, and either `output_tokens` or
an `output_tokens_distribution`. `cached_prefix_tokens` is optional and is
subtracted from executed prefill work while remaining part of the initial KV
context. The seed is optional unless output lengths are sampled.

The CLI streams ledger and power rows with memory bounded by meter-response
history. The analysis API `model.simulation.simulate()` materializes arrays.
Bulk facility evaluation uses the same scheduler and power equations in
bounded parallel node batches; request-state accounting is linear-memory.

## Training the frozen equations

Regenerate the selected artifact without reopening model selection:

```bash
uv run --extra train -m model.scripts.prepare_data
uv run --extra train -m model.scripts.train \
    --prepared-manifest results/clean_model/prepared_dataset.json \
    --out-artifact results/clean_model/powertrace_v1.json
```

`prepare_data` validates and hash-binds the 450-run timing and power payloads
without copying their large arrays into Git. `train` refits only the selected
timing calibration and clean v4 dense/bounded-MoE equations. The current input
contains 354,125 requests. Randomness is not used in fitting.

The release supports Llama 3, DeepSeek-R1-Distill, and GPT-OSS presets on the
declared A100/H100 and tensor-parallel configurations. Support checks bind
model family, hardware, TP, scheduler, and MoE-routing assumptions. See
`docs/MODEL_PIPELINE.md` for the equations and train/inference boundary.

To score two aligned power CSVs:

```bash
uv run -m model.scripts.evaluate \
    --measured measured_power.csv \
    --predicted outputs/example/power.csv \
    --out results/clean_model/evaluation.json
```

## Reproducing the paper artifacts

Regenerate every maintained local paper artifact from one selected release:

```bash
uv run --extra train --extra paper -m scripts.paper.regenerate
```

This command performs, in order:

1. prepared-data validation and release refitting;
2. selected-release measured replay, fidelity table, and representative traces;
3. timing parity figures;
4. retrospective Qwen, BurstGPT, and OpenHands transfer diagnostics;
5. all 240 Azure node traces, facility aggregation, metrics, figures,
   oversubscription analysis, and sizing table; and
6. `results/paper/manifest.json`, which hashes every artifact family.

The main replay and facility outputs use the regenerated release directly. The
appendix transfer panels are explicitly retrospective because they include
declared target-derived idle or platform calibration. External sealed scoring
and disaggregated inference are separate campaigns and are not included in the
local manifest. Exact outputs and caption requirements are in
`docs/PAPER_OUTPUTS.md`.

The facility comparison contains the selected scheduler/power model and the
Splitwise-style LUT baseline. The constant mean baseline is derived from the
new selected facility trace; it no longer reads archived BiGRU training data.
The roughly 687 MB of node/rack/row/site `.npy` arrays are reproducible local
intermediates and are intentionally not tracked; compact manifests, metrics,
tables, and paper figures are tracked.

## Disaggregated GPT-OSS profiling

The minimal cache-disabled transfer campaign is the current collection path:

```bash
# Four-request integration check.
POWERTRACE_REPO="$PWD" \
  sbatch --export=ALL,POWERTRACE_DISAGG_MODE=smoke \
  profiling/jobs/disaggregated_transfer_gpt_oss_20b.sbatch

# Probe, calibration, heldout, and exact heldout replay.
POWERTRACE_REPO="$PWD" \
  sbatch profiling/jobs/disaggregated_transfer_gpt_oss_20b.sbatch
```

`planned_workload.py` creates each tokenized request plan before telemetry and
hashes the exact prompts, token targets, and fixed send offsets. The runner
reuses the heldout plan file for replay and issues requests against absolute
deadlines phase-locked to the running 250 ms NVIDIA-SMI cadence. It leaves
`power.draw` raw and adds only query bounds, pstate, power limit, and slowdown
state. Collection gates reject queueing, request drift, cache activity, NIXL
failures, preemption, short phase visibility, unstable power state, missing
idle coverage, and first/last-third prompt-throughput drift. The campaign
collects evidence for the six-scalar transfer analysis; it does not fit those
scalars or modify the frozen model during profiling.

The Sherlock job runs GPT-OSS-20B on two A100-80GB GPUs with one TP1 prefiller
and one TP1 decoder using the pinned vLLM Queue-Haul image:

```bash
# One-minute 2 requests/s integration gate.
POWERTRACE_REPO="$PWD" POWERTRACE_DISAGG_MODE=smoke \
  sbatch profiling/jobs/disaggregated_gpt_oss_20b.sbatch

# One calibration cell and four held-out five-minute cells.
POWERTRACE_REPO="$PWD" POWERTRACE_DISAGG_MODE=campaign \
  sbatch profiling/jobs/disaggregated_gpt_oss_20b.sbatch
```

The prospective confirmation keeps raw `nvidia-smi power.draw` at 250 ms, the
two A100 TP1 roles, vLLM 0.22, and NIXL. It disables prefix caching and uses the
supported 2048-token scheduler budget with deterministic 8192±25% token inputs
and 64±25% token outputs. The source model predicts nonsaturated prefill/decode
duty near 30% at 1 request/s and 60% at 2 requests/s. One 2-request/s cell is
reserved for role-idle and one-gain-per-role calibration; independently seeded
1- and 2-request/s workloads are each replayed twice as held-out evidence, with
the held-out loads interleaved to limit runtime/thermal order confounding.
Synthetic prompt lengths are re-tokenized from the exact text sent to vLLM;
prompts whose sent length falls outside the requested range are resampled
deterministically.

Each cell records 30-second pre/post idle windows, role-separated engine
streams, per-GPU power, exact proxy stages, and NIXL counters. The endpoint
preflight runs before power logging; measured benchmark traffic suppresses the
benchmark client's otherwise hidden test request. The `core_timed` power
profile retains the unmodified `power.draw` value while recording query
start/end and using their midpoint as the shared GPU-row timestamp. Cell gates
select a free prefiller/decoder/proxy/NIXL port group on shared nodes before
engine launch, preventing a health check from attaching to another job's
server. Queries exceeding 200 ms are dropped without modifying neighboring raw
samples; the unchanged cadence-gap gate rejects excessive loss. Role startup
is serialized through the prefiller health gate to avoid
concurrent tokenizer initialization. Gates require zero cached prompt tokens,
prompt-token accounting within 1%, and median prefill duration of at least one
250 ms meter interval. They also reject NIXL failures, expirations,
preemptions, incomplete UUID samples, query durations above 200 ms, sample gaps
above 750 ms, missing idle coverage, or failure to restore idle power and
temperature within 5 W and 5 C.

The run metadata freezes the later analysis: independent per-role pointwise
fits in watts on the single calibration cell, an equal-parameter phase-duty
null, and raw held-out scoring without smoothing, interpolation, fitted lag, or
warping. Per-role acceptance requires correlation at least 0.8, standard
deviation ratio 0.8–1.25, p95 error at most 10%, and at least 5% lower
pointwise loss than the equal-parameter duty null. Measured replays must
themselves reach 0.8 correlation per role and load. These GPUs perform
different phases and must not be mislabeled as GPT-OSS TP2.

The retained pilot campaign under `data/disagg/` can be analyzed with:

```bash
uv run --extra paper python power-test/analyze_disaggregated_inference.py
```

That analyzer applies the frozen GPT-OSS A100 TP1 timing and dynamic-power
coefficients independently to the prefiller and decoder, then sums their power.
It retains zero-shot and shared-idle baselines and reports a minimal
phase-calibrated variant. That variant uses separate prefill/decode idle
baselines from the first cell's settled pre-request window and one decoder
dynamic gain fitted on `rate-2-repeat-1`. The fit minimizes diagonal soft-DTW
separately by role at the recorded native samples with no temporal
warping. A proposed prefill gain is rejected because it worsens mean, p95, and
variance-ratio guardrails; the frozen prefill dynamics remain unchanged. No
frozen timing or power coefficient is refitted.

Primary held-out-rate phase results use repeat 2 at 0.25 and 4 requests/s.
Request latency uses exact request-ID-paired proxy boundaries, never DTW. The
compact report, per-cell metrics, representative native-sample diagnostics, and
two-panel held-out prefill/decode time-series figure are written under
`results/disaggregated/`. The diagnostics and figure use every recorded
approximately 250 ms sample without averaging, interpolation, smoothing, or
time alignment. Each panel reports its own unwarped diagonal soft-DTW and
correlation.

The current 4 Hz power log supports warm-cell prefill mean energy, but the
prefill trace fails the native-shape acceptance gate. Repeat-dependent
prefix-cache state is not present in the request files, and the logger
timestamps before each `nvidia-smi` subprocess query while prefill HTTP phases
last about 16–20 ms. Decode trace shape passes the same gate. The report records
diagonal and one-sample-band soft-DTW separately, constant-mean nulls,
role-specific temporal metrics, exact phase latencies, and rejected prefill
adjustments rather than allowing node totals or time warping to hide this
boundary.

The pilot remains retrospective unsupported-extrapolation evidence because it
used an 8192-token scheduler override outside the frozen preset. The
confirmatory runner instead uses the preset's 2048-token budget. It exposes the
pinned image's `nixl_cu12` installation under the `nixl` package name required
by vLLM 0.22 and fails before model loading when that runtime is unavailable.
Cache-aware disaggregated modeling is explicitly deferred in
`docs/plans/TODO.md`; the confirmatory claim is cache-disabled.

Confirmatory run roots are immutable and cannot resume across allocations:
mixing GPU UUIDs, idle calibration, images, or code across jobs would invalidate
the held-out comparison. An interrupted campaign must restart under a fresh run
root.

The accepted five-cell run from Slurm job `35692922` is retained under
`data/disagg/gpt-oss-20b-a100-pd-confirmatory-35692922/`. It includes the raw
power, request, proxy-event, role-telemetry, timing-boundary, runtime, and GPU
topology evidence needed for validation and analysis; transient engine and
proxy debug logs remain in the immutable Sherlock run root.

Reproduce the frozen confirmation and the separately labeled post-hoc timing
diagnostic with:

```bash
uv run --extra paper python power-test/analyze_disaggregated_confirmation.py
uv run --extra paper python \
  power-test/analyze_disaggregated_confirmation_timing.py
```

Every collection gate passes, but the preregistered four-scalar calibration
does not pass heldout acceptance for either role. A calibration-cell-only
diagnostic adds one positive service-time scale per role without changing the
base timing or power coefficients. Its 2.068× prefill and 1.087× decode scales
substantially improve prefill transfer: three of four heldout prefill cells
pass, with median correlation 0.919, standard-deviation ratio 0.953, p95 error
2.60%, and energy error 4.24%. Decode still fails all four cells, and only one
of four measured role/load replay comparisons reaches the required 0.8
correlation. The diagnostic is therefore useful evidence that a small timing
correction recovers most prefill behavior, not a successful confirmatory
result or grounds for changing the frozen release. Both primary two-panel plots
retain every raw query sample and use no smoothing, averaging, interpolation,
fitted lag, or warping. The timing diagnostic also emits a clearly labeled
one-second view formed by arithmetic means in fixed, non-overlapping bins; it
does not replace the native-sample result.

## Historical GMM-BiGRU artifact

Run historical commands from inside the archive so Python cannot resolve the
maintained root package:

```bash
cd archive/gmm_bigru_v1
uv run --project ../.. --extra archive-bigru python \
    -m model.scripts.train_gmm_bigru --help
```

The archive README documents preparation, inference, evaluation, figures, and
historical facility baselines. Do not copy archived checkpoints or surrogate
paths back into the maintained runtime.

## Azure input preparation

The facility pipeline expects parsed Azure arrivals and per-node request
streams under `data/azure_trace/` and `data/azure_facility/node_streams/`.
Regenerate those inputs only when the source trace or allocation policy
changes; paper regeneration rebuilds model traces and downstream results from
the existing frozen streams.

The selected facility workload is Llama-3-70B A100 TP8. Each node uses the
release's `vllm_v1_decode_first` scheduler, full prefill/decode work ledger,
power equation, and meter response. Facility aggregation adds the documented
non-GPU overhead and PUE once. Splitwise retains its separately labeled
prompt-biased preemptive FIFO scheduler and LUT support/clamp diagnostics.

## Testing

Run the complete maintained test suite after any code change:

```bash
uv run -m pytest -x
```

Research-script tests that live outside the default collection can be run
explicitly, for example:

```bash
uv run -m pytest -x power-test/tests timing-test/tests
```

The profiling proxy route integration test runs when FastAPI is available from
the profiling runtime; dependency-free campaign validation remains in the
default suite.
