# Job Scripts

These scripts run model servers and benchmark clients across tensor-parallel and arrival-rate grids while `nvidia-smi` records GPU power/utilization.

## GPT-OSS ShareGPT benchmarking

### Required environment variables

- `SHAREGPT_DATASET_PATH`: Path to the ShareGPT dataset used by `benchmark_serving.py --dataset-name sharegpt`.

### Optional profiling-related environment variables

- `VLLM_TORCH_PROFILER_DIR`: If set for the vLLM server process, enables vLLM profiler endpoints.
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN`: Optional for custom long-context configs, if needed by your server setup.

### Run commands

From repo root:

```bash
bash job-scripts/gpt-oss-20b.sh
bash job-scripts/gpt-oss-120b.sh
```

### Script behavior and grid

- `gpt-oss-20b.sh`
  - TP: `1, 2`
  - Rates (qps): `0.125, 0.25, 0.5, 1, 2, 4`
  - Iterations: `5`
- `gpt-oss-120b.sh`
  - TP: `4, 8`
  - Rates (qps): `0.125, 0.25, 0.5, 1, 2, 4`
  - Iterations: `5`
- Prompt count per run: `round(600 * rate)` (10-minute effective profiling horizon).
- Benchmark invocation uses:
  - `--backend vllm --endpoint /v1/completions`
  - `--dataset-name sharegpt --dataset-path "$SHAREGPT_DATASET_PATH"`
  - `--save-result --save-detailed`

### Output directories (auto-detected hardware)

- 20B:
  - `data/sharegpt-benchmark-gpt-oss-20b-a100`
  - `data/sharegpt-benchmark-gpt-oss-20b-h100`
- 120B:
  - `data/sharegpt-benchmark-gpt-oss-120b-a100`
  - `data/sharegpt-benchmark-gpt-oss-120b-h100`

Hardware class is inferred from `nvidia-smi` (`A100`/`H100`) and selected automatically.

### Filename conventions

- Benchmark JSON:
  - `vllm-{rate}qps-tp{tp}-gpt-oss-{size}b-{YYYYMMDD-HHMMSS}.json`
- Power CSV:
  - `gpt-oss-{size}b_tp{tp}_p{rate}_d{YYYYMMDD-HHMMSS}.csv`

### Matrix completeness note

Run both scripts on both hardware classes (A100 and H100) to complete the full GPT-OSS ShareGPT profiling matrix.
