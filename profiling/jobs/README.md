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

## Current GPT-OSS ShareGPT Runs

### Required environment variables

- `SHAREGPT_DATASET_PATH`: Path to the ShareGPT dataset used by `benchmark_serving.py --dataset-name sharegpt`.

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
