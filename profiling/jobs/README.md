# Job Scripts

The authoritative current launch order is
`profiling/MODEL_READINESS_RUNBOOK.md`. The legacy GPT-OSS grids below are not
part of that campaign and must not be mixed into its fit or sealed score.

These scripts launch a vLLM server, run a benchmark client, and record `nvidia-smi` power/utilization logs for profiling runs.

## Campaign Runner Bundle Roots

`run_campaign.sh --execute` writes live bundles under
`data/runs/<campaign_id>/<run_id>/` by default, or under
`$RUNS/<campaign_id>/<run_id>/` when `RUNS` is set. Dry runs write sample bundles
under `data/dry-runs/<campaign_id>/<run_id>/`, or `$DRY_RUNS/<campaign_id>/<run_id>/`
when `DRY_RUNS` is set, so review artifacts do not pollute live roots.

The campaign loader derives `gpus_per_node` from the largest TP degree, matching
the GPU count requested by `submit_campaign.sh`. Installed-but-unallocated GPUs
are never advertised in bundle manifests.
For a smaller TP-pair leg, `run_campaign.sh` pins vLLM to the first TP CUDA
ordinals and records the matching UUID set. Ingestion checks UUIDs against the
first-TP power columns, so idle GPUs from the larger allocation cannot enter the
power target.
`submit_campaign.sh -p owners` adds a hardware-specific 80GB GPU constraint
when none is supplied: A100 campaigns use `GPU_SKU:A100_SXM4&GPU_MEM:80GB`,
and H100 campaigns use `GPU_SKU:H100_SXM5&GPU_MEM:80GB`. The submit wrapper
also exports `POWERTRACE_REPO` so the batch job runs the checkout that submitted
the campaign rather than assuming `$HOME/powertrace-sim`.
`stage_models.sh` sets `HF_SNAPSHOT_MAX_WORKERS=2` by default; lower it to `1`
when staging very large checkpoints on constrained login-node environments.
Native `arch_extract` sanity is opt-in with `STAGE_MODELS_ARCH_SANITY=1`; the
launch container performs the authoritative parse. Probe campaigns launch direct
`profiling/probes/<probe>.py` entry points, one per `schedule.BUILDERS` probe.

## Minimal expansion matrix

The expansion is a set of independent Slurm jobs. It holds the ShareGPT sample
and seed fixed while changing one axis:

The smaller paper transfer-interval follow-up is separate from this expansion.
Its existing-unit audit, four campaign files, A100-hour estimate, preparation,
and guarded submit command are in
`docs/plans/TRANSFER_CI_A100_CAMPAIGN.md`.

| comparison | campaign files | changed axis |
|---|---|---|
| dense Qwen transfer | `validate_qwen3-8b_a100.json`, `validate_qwen3-8b.json` | hardware |
| off-grid arrival rate | `validate_qwen3-8b_a100.json`, `arrival_rate_qwen3-8b_a100_r2p5.json` | 4.0 versus 2.5 requests/s |
| controlled arrival pattern | `arrival_rate_qwen3-8b_a100_r2p5.json`, `arrival_pattern_qwen3-8b_a100_{bursty,smooth}.json` | Gamma shape 1.0, 0.25, 4.0 |
| exact arbitrary timestamps | `burstgpt_qwen3-8b_a100.json` | recorded BurstGPT arrivals |
| agentic context/cache | `trace_replay_qwen3-8b_a100_cache_{off,on}.json` | prefix cache |
| cross-family MoE | `validate_gemma-4-26b-a4b_moe_transfer_a100.json` | Gemma routing/model family |

Gamma shape changes variance without changing mean interarrival: shape 1 is
Poisson, 0.25 has coefficient of variation 2, and 4 has coefficient of
variation 0.5. Every JSON contains one rate and becomes one checkpointed job.

GPU jobs are offline. `stage_models.sh` writes a completion marker after each
selected snapshot is complete; submission rejects a cache directory without
that marker. Submission also checks the model-specific container, local dataset
or trace plan, and frozen MoE JSONL before allocating GPUs.

```bash
bash profiling/jobs/stage_models.sh \
  Qwen/Qwen3-8B meta-llama/Llama-3.1-70B-Instruct \
  openai/gpt-oss-20b openai/gpt-oss-120b \
  google/gemma-4-26B-A4B-it

H100_PARTITION=<partition> \
  bash profiling/jobs/submit_expansion_jobs.sh
```

The wrapper preflights the whole matrix before its first `sbatch`, then submits
each campaign and router capture independently so Sherlock can schedule them in
parallel. Freeze predictions before invoking it. The TraceLab smoke remains a
separate compatibility gate and is not scientific evidence.

## Exact arrivals, sessions, and TP8 state

Normalize a released TraceLab CSV (arrival and tool-wait columns are
milliseconds) into the source/revision-bound replay plan:

```bash
uv run python profiling/agentic_traces/build_trace_plan.py \
  syfi_coding_trace.jsonl.gz data/trace_plans/tracelab_code.json \
  --format tracelab-jsonl --revision v0.0.1 \
  --max-sessions 8 --min-rounds-per-session 12 \
  --max-rounds-per-session 32 \
  --context-band 4096:8192:2 --context-band 8192:16384:2 \
  --context-band 16384:24576:2 --context-band 24576:31745:2

bash profiling/jobs/run_campaign.sh \
  profiling/campaigns/trace_replay_qwen3-8b_a100_cache_off.json
bash profiling/jobs/run_campaign.sh \
  profiling/campaigns/trace_replay_qwen3-8b_a100_cache_on.json
```

On Sherlock, submit the one-session compatibility gate before the full plan:

```bash
bash profiling/jobs/submit_campaign.sh \
  profiling/campaigns/trace_replay_smoke_qwen3-8b_a100.json \
  --time 00:30:00
```

It runs one four-round session in cache-off and cache-on modes. Submit the two
full `trace_replay_qwen3-8b_a100_cache_{off,on}.json` jobs only after both smoke
bundles validate.

Execute mode preserves each plan release before acquiring concurrency and
preserves per-session turn order/tool waits. The direct completion path requires
exact server prompt/completion usage and records cached and reasoning subsets.
After both regimes complete, verify token-level pairing with:

```bash
uv run python profiling/probes/compare_trace_replays.py \
  <cache-off-bundle> <cache-on-bundle>
```

Create the conditional BurstGPT arbitrary-arrival plan on Sherlock with:

```bash
uv run python profiling/agentic_traces/build_trace_plan.py \
  "$GROUP_HOME/gfw/BurstGPT_without_fails_2.csv" \
  data/trace_plans/burstgpt_10min.json \
  --format burstgpt --revision <SOURCE_SHA256> \
  --max-sessions 1000 --window-duration-s 600
```

This keeps exact timestamp gaps from the densest contiguous ten-minute window.
BurstGPT rows are independent requests, not fabricated multi-turn sessions.

The collected TP4 state control is not rerun. Rebuild its normalized
release/length plan and dry-run the pending TP8 leg with:

```bash
uv run python profiling/agentic_traces/build_bundle_replay_plan.py \
  data/runs/h100_tp4_state_control/h100_validate_tp4_r4p0_b1p0_s0_1784435159/requests.json \
  data/trace_plans/h100_tp4_state_marks.json

bash profiling/jobs/run_campaign.sh \
  profiling/campaigns/h100_tp8_state_diagnostic.json
```

The plan hash is
`7be58efca871bcb38fa8d56d7ecbb106c0f09de627df0f42efd80469f7b905a4`.
It preserves the TP4 relative releases and input/output lengths, then uses
deterministic direct tokens because the historical TP4 artifact did not retain
prompt IDs. The TP8 run records a 180-second idle window, server launch/ready
epochs, and the full `tp8_state`/measured-ledger telemetry.

This is the only unconditional new state job:

```bash
bash profiling/jobs/submit_campaign.sh \
  profiling/campaigns/h100_tp8_state_diagnostic.json \
  -p owners --time 01:00:00
```

Two additional exact-plan campaigns are prepared but remain conditional:

- `h100_405b_rate4_exact_replay.json` is T4-B and must not run until the
  mixed-iteration candidate is frozen.
- `h100_qwen3_8b_idle_decomposition.json` is L1 and must not run until T4/P4
  are resolved.

Neither file is included in `submit_expansion_jobs.sh`, so the conditional
budget cannot be spent by the broad expansion wrapper.

Build a small, deterministic router-capture set on a networked login node:

```bash
uv run python profiling/moe_routing/build_routing_samples.py \
  --sharegpt "$GROUP_HOME/gfw/ShareGPT_V3_unfiltered_cleaned_split.json" \
  --output-jsonl "$GROUP_HOME/gfw/moe_routing_samples.jsonl" \
  --n-per-source 64 --seed 0

bash profiling/jobs/stage_models.sh \
  openai/gpt-oss-20b openai/gpt-oss-120b \
  google/gemma-4-26B-A4B-it

bash profiling/jobs/submit_moe_routing.sh \
  openai/gpt-oss-20b 1 \
  "$GROUP_HOME/gfw/moe_routing_samples.jsonl" gpt-oss-20b-routing.npz

bash profiling/jobs/submit_moe_routing.sh \
  openai/gpt-oss-120b 4 \
  "$GROUP_HOME/gfw/moe_routing_samples.jsonl" gpt-oss-120b-routing.npz

bash profiling/jobs/submit_moe_routing.sh \
  google/gemma-4-26B-A4B-it 1 \
  "$GROUP_HOME/gfw/moe_routing_samples.jsonl" gemma-4-26b-a4b-routing.npz \
  vllm-openai-gemma4.sandbox
```

The builder downloads the `tool` split of
`SWE-bench/SWE-smith-trajectories` through `datasets` and selects 64 usable
examples from each source. This networked preparation happens before Slurm; GPU
jobs consume only the frozen JSONL. Capture fails when a sample exceeds 4,096
tokens. The NPZ retains raw token/layer/top-k assignments, a contiguous-token
prefill curve, and a decode curve formed from one completion token per active
sequence. Missing router logits fail instead of falling back to uniform routing.

TraceLab sessions whose reported prefix exceeds the preceding reproducible
context are ineligible for exact replay. Context-band selection excludes them
and fails if a requested band cannot supply its declared session count.

## Current GPT-OSS ShareGPT Runs

### Required environment variables

- `SHAREGPT_DATASET_PATH`: Path to the ShareGPT dataset used by
  `benchmark_serving.py --dataset-name sharegpt`. `campaign.sbatch` sets it to
  `$GROUP_HOME/gfw/ShareGPT_V3_unfiltered_cleaned_split.json` and binds
  `$GROUP_HOME` into the container.
- `BURSTGPT_DATASET_PATH`: `campaign.sbatch` sets this to
  `$GROUP_HOME/gfw/BurstGPT_without_fails_2.csv` for plan preparation/audit.

### Optional profiling-related environment variables

- `VLLM_TORCH_PROFILER_DIR`: If set for the vLLM server process, enables vLLM profiler endpoints.
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN`: Optional for custom long-context configs, if needed by your server setup.
- `DEBUG=1`: Enables shell tracing in the current scripts.

### Run commands

From the repo root:

```bash
bash profiling/jobs/gpt-oss-20b.sh
bash profiling/jobs/gpt-oss-120b.sh
```

### Script behavior and grid

- `gpt-oss-20b.sh`
  - TP: `1, 2`
  - Rates (qps): `0.125, 0.25, 0.5, 1, 2, 4`
  - Iterations: `5`
  - Prompts per run: `round(600 * rate)`
- `gpt-oss-120b.sh`
  - TP: `4, 8`
  - Rates (qps): `0.125, 0.25, 0.5, 1, 2, 4`
  - Iterations: `5`
  - Prompts per run: `round(600 * rate)`
- Benchmark invocation uses:
  - `--backend vllm --endpoint /v1/completions`
  - `--dataset-name sharegpt --dataset-path "$SHAREGPT_DATASET_PATH"`
  - `--save-result --save-detailed`

### Output directories

Current scripts write to:

- `data/sharegpt-benchmark-gpt-oss-20b-a100`
- `data/sharegpt-benchmark-gpt-oss-120b-a100`

The current implementations hardcode the `a100` suffix rather than inferring hardware class from `nvidia-smi`.

### Filename conventions

- Benchmark JSON:
  - `vllm-{rate}qps-tp{tp}-gpt-oss-{size}b-{YYYYMMDD-HHMMSS}.json`
- Power CSV:
  - `gpt-oss-{size}b_tp{tp}_p{rate}_d{YYYYMMDD-HHMMSS}.csv`
