# Model-readiness profiling runbook

This is the only execution checklist for closing the failures in
`FEATURE_TEST_LEARNINGS.md`. `CAMPAIGN.md` is background and design rationale.
Do not add agentic, prefix-cache, long-context replay, generic Tier-2 models, or
custom vLLM counters to this run. They answer different questions.

## Definition of done

The work is complete only when all of these are true:

1. All eight development campaigns below emitted validated canonical bundles.
2. The post-collection adapter described below exists and produces one combined
   250 ms measured ledger plus a `ledger-run-index-v1` index.
3. Candidate equations and selection rules were frozen using development data
   only, then exported as one artifact per hardware.
4. The two sealed campaigns were collected under `SEALED_RUNS` after the freeze.
5. A score-only path evaluated the sealed bundles once without refitting,
   reselection, threshold changes, or feature changes.

Collection is ready now. Items 2, 3, and 5 are explicit software blockers; do
not run the current frozen evaluator directly on the new bundle caches and call
that a completed model update.

## The unconfounded contrasts

Every interpretation must use one of these rows. Do not compare arbitrary runs.

| question | compare | held fixed | only intended change |
|---|---|---|---|
| A100 hardware response | levels inside `a100_tier1_llama70b` | model, TP, server flags | controlled operating point |
| A100 TP effect | gpt-oss-20B TP2 vs TP4 probe levels | model, batch, context, probe schedule | TP |
| A100 model-scale effect | gpt-oss-20B TP4 vs 120B TP4 probe levels | TP, batch, context, server flags | model scale |
| A100 realistic scale check | 20B TP4 vs 120B TP4 ShareGPT | TP, dataset, request marks, rates, seed, server flags | model scale |
| H100 hardware response | levels inside `h100_tier1_llama70b` TP8 | model, TP, server flags | controlled operating point |
| H100 TP effect | Llama-70B TP4 vs TP8 probe levels | model, batch/context schedule, server flags | TP |
| H100 model-scale effect | Llama-70B TP8 vs 405B TP8 ShareGPT | TP, dataset, request marks, rates, seed, server flags | model scale |

Hardware is never compared across A100 and H100 fits. Probe data identifies the
response; realistic ShareGPT data checks whether that response survives normal
scheduling. Sealed data grades the frozen result and identifies nothing.

## Campaigns and order

Run development campaigns in this order:

1. `a100_tier1_llama70b.json`
2. `a100_iteration_gpt-oss-20b.json` — TP2 and TP4
3. `a100_iteration_gpt-oss-120b.json` — TP4
4. `h100_tier1_llama70b.json` — TP8 and matched TP4 subset
5. `a100_hardcells_gpt-oss-20b.json`
6. `a100_hardcells_gpt-oss-120b.json`
7. `h100_hardcells_llama70b.json`
8. `h100_hardcells_llama405b.json`

These produce 35 development bundles: 23 controlled-probe bundles and 12
realistic validation bundles. Do not run the sealed campaigns yet.

After the model and thresholds are frozen, run exactly:

1. `a100_sealed_gpt-oss-120b.json`
2. `h100_sealed_llama405b.json`

These produce six sealed bundles at fresh rates and seeds.

## Before submission

From the repository root:

```bash
uv sync
env UV_CACHE_DIR=/tmp/uv-cache uv run -m pytest -x
env UV_CACHE_DIR=/tmp/uv-cache uv run -m pytest -x profiling feature-test/tests
```

On Sherlock, verify all of the following before spending GPU time:

- `$SCRATCH/ptsim/vllm-openai-v0.10.1.1.sandbox` exists.
- Every campaign model is staged below `$SCRATCH/ptsim/hf/hub/`.
- `$SCRATCH/ptsim/data/ShareGPT_V3_unfiltered_cleaned_split.json` exists.
- The H100 partition and constraint are known. They are site-specific and are
  not safely inferable from this repository. Never submit an H100 campaign to
  the default A100 `ramr` partition.
- Development `RUNS` and sealed `SEALED_RUNS` are different roots.

Dry-run every campaign before submission:

```bash
bash profiling/jobs/run_campaign.sh profiling/campaigns/a100_tier1_llama70b.json
bash profiling/jobs/run_campaign.sh profiling/campaigns/a100_iteration_gpt-oss-20b.json
bash profiling/jobs/run_campaign.sh profiling/campaigns/a100_iteration_gpt-oss-120b.json
bash profiling/jobs/run_campaign.sh profiling/campaigns/h100_tier1_llama70b.json
bash profiling/jobs/run_campaign.sh profiling/campaigns/a100_hardcells_gpt-oss-20b.json
bash profiling/jobs/run_campaign.sh profiling/campaigns/a100_hardcells_gpt-oss-120b.json
bash profiling/jobs/run_campaign.sh profiling/campaigns/h100_hardcells_llama70b.json
bash profiling/jobs/run_campaign.sh profiling/campaigns/h100_hardcells_llama405b.json
```

Each plan must say `evidence: measured_ledger`, `role: development`, and show
only the TP/rate combinations listed above.

## Submission

A100 submission must respect the current shared `ramr` cap. Use `ramr` only
for campaigns whose maximum TP is at most 2; the current model-readiness A100
configs all require TP4, so submit them to `owners`. The wrapper defaults owners
A100 jobs to `GPU_SKU:A100_SXM4&GPU_MEM:80GB` and enables requeue:

```bash
bash profiling/jobs/submit_campaign.sh profiling/campaigns/a100_tier1_llama70b.json -p owners
bash profiling/jobs/submit_campaign.sh profiling/campaigns/a100_iteration_gpt-oss-20b.json -p owners
bash profiling/jobs/submit_campaign.sh profiling/campaigns/a100_iteration_gpt-oss-120b.json -p owners
bash profiling/jobs/submit_campaign.sh profiling/campaigns/a100_hardcells_gpt-oss-20b.json -p owners
bash profiling/jobs/submit_campaign.sh profiling/campaigns/a100_hardcells_gpt-oss-120b.json -p owners
```

H100 submission requires an explicit non-`ramr` GPU partition. On Sherlock,
`owners` is valid; the wrapper defaults owners H100 jobs to
`GPU_SKU:H100_SXM5&GPU_MEM:80GB`. On other sites, replace both values:

```bash
bash profiling/jobs/submit_campaign.sh profiling/campaigns/h100_tier1_llama70b.json -p owners
bash profiling/jobs/submit_campaign.sh profiling/campaigns/h100_hardcells_llama70b.json -p owners
bash profiling/jobs/submit_campaign.sh profiling/campaigns/h100_hardcells_llama405b.json -p owners
```

Resubmit the identical command after preemption or failure. Checkpoints skip
only complete bundles.

## Per-bundle acceptance

A live run is admissible only if it finishes with a manifest containing:

- `instrumentation.status == "validated"`;
- `instrumentation.profile == "measured_ledger"`;
- complete required engine-column coverage;
- the expected model, hardware, TP, seed/rate or probe levels;
- `server.active_gpu_uuids` with exactly `tp` UUIDs;
- four nonempty source files: `power.csv`, `engine.csv`, `requests.json`, and
  `manifest.json`.

The live runner enforces metric availability, monotonic counters, 4 Hz cadence,
GPU identity/topology, active TP UUIDs, and power/engine epoch alignment. A run
that fails those checks is rerun; it is never repaired by interpolation.

## Post-collection blockers to implement before fitting

The repository does not yet provide one correct command for this handoff. Build
and test these pieces in order:

1. **Prefill calibration:** derive one `lambda_prefill` per model/hardware/TP
   from its pure prefill probe and record the source bundle and method.
2. **250 ms bundle projection:** invoke the measured hybrid path at `--dt 0.25`
   for each configuration with its own calibration. The current builder correctly
   refuses to apply one calibration across multiple configurations.
3. **Cache merge/index adapter:** concatenate those per-configuration caches,
   remap run/model/family indices, preserve the measured diagnostic columns, and
   emit `ledger-run-index-v1`. Normalize full Hugging Face IDs to the evaluator's
   declared model identities. Test numerical conservation and provenance.
4. **Development split adapter:** assign probe, realistic-development, and sealed
   roles from manifest metadata. Do not let the frozen evaluator infer roles from
   repeat order; these campaigns intentionally do not collect three redundant
   repeats per cell.
5. **Candidate freeze:** predeclare the smallest candidate terms using iteration
   rate/tokens-per-iteration, select on development data only, and write immutable
   equations, gates, cache hashes, and artifacts.
6. **Score-only sealed path:** load frozen artifacts and thresholds, accept only
   `validation_role=sealed`, produce metrics, and expose no fitting or selection
   operation.

Until all six exist, the correct stopping point is “validated profiling data
collected,” not “model tuned.”

## Sealed execution

Only after the freeze, create a separate restricted root below `$SCRATCH` (the
current Apptainer launch binds `$SCRATCH`, so an arbitrary external path is not
visible in the container) and submit:

```bash
export SEALED_RUNS="$SCRATCH/ptsim/sealed-runs"
mkdir -p "$SEALED_RUNS"
chmod 700 "$SEALED_RUNS"
bash profiling/jobs/submit_campaign.sh profiling/campaigns/a100_sealed_gpt-oss-120b.json -p owners
bash profiling/jobs/submit_campaign.sh profiling/campaigns/h100_sealed_llama405b.json -p owners
```

The 405B sealed and development cells use the pre-quantized
`RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8` checkpoint
(`server.quantization=compressed-tensors`, `dtype_hint=fp8`) because BF16 does
not fit on 8x80GB H100; keep that checkpoint/dtype as part of the reported cell
identity. Do not inspect per-run sealed traces before
score-only evaluation. Do not refit after seeing sealed metrics. Report every
declared cell, including failures.
