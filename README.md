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
- `docs/PAPER_OUTPUTS.md`: paper artifact and evidence contract.
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

The Sherlock job runs GPT-OSS-20B on two A100-80GB GPUs with one TP1 prefiller
and one TP1 decoder using the pinned vLLM Queue-Haul image:

```bash
# Ten-minute 2 requests/s integration gate.
POWERTRACE_DISAGG_MODE=smoke \
  sbatch profiling/jobs/disaggregated_gpt_oss_20b.sbatch

# 0.25, 2, and 4 requests/s; three ten-minute repetitions per rate.
POWERTRACE_DISAGG_MODE=campaign \
  sbatch profiling/jobs/disaggregated_gpt_oss_20b.sbatch
```

The run records role-separated prefill/decode engine streams, per-GPU power,
request results, NIXL transfer counters, and a hash-bound manifest. These GPUs
perform different phases and must not be mislabeled as GPT-OSS TP2. A
role-aware ingestion/composition step is still required before the maintained
model can score the combined deployment; this campaign is therefore outside
the current paper allowlist.

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
