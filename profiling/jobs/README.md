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
For a smaller TP-pair leg, `run_campaign.sh` pins vLLM to the first TP UUIDs and
records that exact set. Ingestion checks it against the first-TP power columns,
so idle GPUs from the larger allocation cannot enter the power target.

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
  profiling/campaigns/trace_replay_qwen3-8b_a100.json
```

On Sherlock, submit the one-session compatibility gate before the full plan:

```bash
bash profiling/jobs/submit_campaign.sh \
  profiling/campaigns/trace_replay_smoke_qwen3-8b_a100.json \
  --time 00:30:00
```

It runs one four-round session in cache-off and cache-on modes. Submit
`trace_replay_qwen3-8b_a100.json` only after both smoke bundles validate.

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

The H100 TP8 transition diagnostic and matched TP4 control are scaffolded in
`profiling/campaigns/h100_tp8_state_diagnostic.json`. Each leg records 180 seconds
of idle before approximately 420 seconds at four requests/second and requires the
`tp8_state` telemetry profile. Set `SHAREGPT_DATASET_PATH` before execution.

Build a small, deterministic router-capture set on a networked login node:

```bash
uv run python profiling/moe_routing/build_routing_samples.py \
  --sharegpt "$GROUP_HOME/gfw/ShareGPT_V3_unfiltered_cleaned_split.json" \
  --output-jsonl "$GROUP_HOME/gfw/moe_routing_samples.jsonl" \
  --n-per-source 64 --seed 0

bash profiling/jobs/stage_models.sh \
  openai/gpt-oss-20b openai/gpt-oss-120b

bash profiling/jobs/submit_moe_routing.sh \
  openai/gpt-oss-20b 1 \
  "$GROUP_HOME/gfw/moe_routing_samples.jsonl" gpt-oss-20b-routing.npz

bash profiling/jobs/submit_moe_routing.sh \
  openai/gpt-oss-120b 4 \
  "$GROUP_HOME/gfw/moe_routing_samples.jsonl" gpt-oss-120b-routing.npz
```

The builder downloads the `tool` split of
`SWE-bench/SWE-smith-trajectories` through `datasets` and selects 64 usable
examples from each source. Each row contains `id`, `source`, `prompt_text`, and
the immediately following `completion_text`. The Slurm jobs write the NPZ and
manifest under `$SCRATCH/ptsim/moe-routing/`; they retain token, layer, top-k
expert IDs, prompt/decode phase, source hash, and distinct-expert curves. Missing
router logits fail the capture instead of falling back to uniform routing.

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
