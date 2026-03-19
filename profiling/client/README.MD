# Client

This directory contains the profiling clients and dataset helpers used to collect request timing and GPU power traces from serving endpoints.

## Main Scripts

### `benchmark_serving.py`

Primary benchmark runner for vLLM-compatible endpoints. It supports ShareGPT, BurstGPT, Sonnet, random, realistic, and HuggingFace-backed workloads.

```bash
python profiling/client/benchmark_serving.py \
    --backend vllm \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path /path/to/sharegpt.json \
    --request-rate 1.0 \
    --num-prompts 600 \
    --save-result \
    --save-detailed
```

Key flags:

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model name used for the benchmark request payload | Required |
| `--backend` | Request backend | `vllm` |
| `--base-url` | Server or API base URL; overrides `--host` and `--port` when set | `None` |
| `--host` | Host used when `--base-url` is not provided | `127.0.0.1` |
| `--port` | Port used when `--base-url` is not provided | `8000` |
| `--endpoint` | API endpoint path | `/v1/completions` |
| `--dataset-name` | Dataset family: `sharegpt`, `burstgpt`, `sonnet`, `random`, `realistic`, or `hf` | `sharegpt` |
| `--dataset-path` | Dataset file path or HuggingFace dataset ID, depending on `--dataset-name` | `None` |
| `--request-rate` | Requests per second | `inf` |
| `--num-prompts` | Number of prompts to process | `1000` |
| `--tensor-parallel-size` | Tensor parallelism recorded in the result metadata | `1` |
| `--save-result` | Write the benchmark JSON result file | `False` |
| `--save-detailed` | Include per-request detail in the JSON output | `False` |
| `--append-result` | Append to an existing JSON result file | `False` |

When `--save-result` is enabled, the script writes:

- `{backend}-{request_rate}qps{max_concurrency_str}-tp{tp}-{model}-{timestamp}.json`
- `{same-name-without-.json}.pytorch.json` if the summary conversion succeeds

Power CSVs are recorded separately by the surrounding job or shell process.

### `client.py`

Synchronous benchmark driver that samples `garage-bAInd/Open-Platypus` by default, sends chat-completion requests, and writes per-run CSV output.

Required CLI flags:

- `--date`
- `--model-name`

Other notable flags:

- `--base-url`
- `--api-key`
- `--poisson-arrival-rate`
- `--tensor-parallel-size`
- `--time-window`
- `--reasoning`

Output files are written as:

- `results_<model>_<poisson_arrival_rate>_<tensor_parallel_size>_d<date>.csv`
- `results_<model>_<poisson_arrival_rate>_<tensor_parallel_size>_d<date>_final.csv`

### `client_async.py`

Fully asynchronous version of the same benchmark flow. It uses `AsyncOpenAI` plus `aiohttp` for `/info` and `/metrics` collection and writes the same CSV naming pattern as `client.py`.

### `benchmark_dataset.py`

Dataset sampling helpers used by the benchmark runners.

```python
from benchmark_dataset import RealisticRandomDataset, ShareGPTDataset

dataset = ShareGPTDataset(dataset_path="/path/to/sharegpt.json")
requests = dataset.sample(tokenizer=tokenizer, num_requests=600)
```

### `benchmark_utils.py`

Shared result-format conversion and JSON writing helpers used by `benchmark_serving.py`.

### `backend_request_func.py`

Backend adapters that define the request functions used by the benchmark runner.

## Data Collection Workflow

1. Start the serving endpoint described in [`profiling/server/README.md`](../server/README.md).
2. Run `benchmark_serving.py` or one of the client scripts above.
3. Capture GPU power data separately with the job wrapper or `nvidia-smi`.

## Test Scripts

| Script | Description |
|--------|-------------|
| `test_benchmark_serving_timestamps.py` | Verifies request timestamps stay aligned with mixed benchmark outcomes |
| `test_distributions_standalone.py` | Validates request distribution generation |
| `test_preset_distributions.py` | Validates preset workload distributions |
| `test_simple_integration.py` | Exercises the realistic random dataset sampler |
| `test_integration.sh` | Shell integration test for the client workflow |

## See Also

- [`profiling/server/README.md`](../server/README.md) - Server launch scripts
- [`profiling/jobs/README.md`](../jobs/README.md) - Automated benchmarking jobs
- [`data/README.md`](../../data/README.md) - Collected data layout
