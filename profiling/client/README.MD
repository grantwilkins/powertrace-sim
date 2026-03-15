# Client

This directory contains the benchmark client for collecting power traces from LLM inference servers. The client sends requests to a vLLM server, collects timing information, and records GPU power consumption.

## Main Scripts

### `benchmark_serving.py` - Primary Benchmark Client

The main benchmarking script that sends requests and collects detailed metrics.

```bash
python profiling/client/benchmark_serving.py \
    --backend vllm \
    --endpoint /v1/completions \
    --dataset-name sharegpt \
    --dataset-path /path/to/sharegpt.json \
    --request-rate 1.0 \
    --num-prompts 600 \
    --save-result \
    --save-detailed
```

**Key Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--backend` | Backend type (vllm, openai, etc.) | vllm |
| `--endpoint` | API endpoint | /v1/completions |
| `--dataset-name` | Dataset to use (sharegpt, random) | sharegpt |
| `--dataset-path` | Path to dataset file | Required |
| `--request-rate` | Requests per second | 1.0 |
| `--num-prompts` | Total prompts to send | 100 |
| `--save-result` | Save summary results | False |
| `--save-detailed` | Save per-request details | False |

**Output Files:**

- `vllm-{rate}qps-tp{tp}-{model}-{timestamp}.json` - Detailed benchmark results
- Power CSV files (recorded separately by nvidia-smi)

### `benchmark_dataset.py` - Dataset Utilities

Dataset loading and preprocessing utilities.

```python
from client.benchmark_dataset import load_sharegpt_dataset

dataset = load_sharegpt_dataset("/path/to/sharegpt.json")
```

### `client.py` - Synchronous Client

Basic synchronous HTTP client for simple testing.

```python
from client.client import send_request

response = send_request(
    prompt="Hello, how are you?",
    max_tokens=100,
    endpoint="http://localhost:8000/v1/completions"
)
```

### `client_async.py` - Asynchronous Client

High-performance async client for load testing.

```python
from client.client_async import AsyncBenchmarkClient

client = AsyncBenchmarkClient(base_url="http://localhost:8000")
results = await client.benchmark(
    prompts=prompts,
    request_rate=2.0,
    max_tokens=100
)
```

### `backend_request_func.py` - Backend Adapters

Request functions for different inference backends.

```python
from client.backend_request_func import get_request_func

request_func = get_request_func("vllm", "/v1/completions")
result = await request_func(prompt, max_tokens, session)
```

### `benchmark_utils.py` - Utility Functions

Common utilities for benchmarking.

```python
from client.benchmark_utils import (
    compute_latency_stats,
    compute_throughput,
    format_results,
)
```

## Data Collection Workflow

1. **Start the vLLM server** (see `profiling/server/` directory)

2. **Run the benchmark client:**

   ```bash
   python profiling/client/benchmark_serving.py \
       --backend vllm \
       --endpoint /v1/completions \
       --dataset-name sharegpt \
       --dataset-path $SHAREGPT_DATASET_PATH \
       --request-rate 1.0 \
       --num-prompts 600 \
       --save-result \
       --save-detailed
   ```

3. **Collect power data** (typically done by job scripts):

   ```bash
   nvidia-smi --query-gpu=timestamp,power.draw \
       --format=csv -l 1 > power.csv &
   ```

## Test Scripts

| Script | Description |
|--------|-------------|
| `test_benchmark_serving_timestamps.py` | Test timestamp alignment |
| `test_distributions_standalone.py` | Test request distribution generation |
| `test_preset_distributions.py` | Test preset distribution configurations |
| `test_simple_integration.py` | Basic integration tests |
| `test_integration.sh` | Full integration test script |

## Output Format

### Benchmark JSON

```json
{
  "backend": "vllm",
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "request_rate": 1.0,
  "num_prompts": 600,
  "duration": 600.5,
  "completed_requests": 598,
  "requests": [
    {
      "request_id": 0,
      "prompt_len": 512,
      "output_len": 128,
      "arrival_time": 0.0,
      "start_time": 0.001,
      "first_token_time": 0.045,
      "end_time": 2.134,
      "ttft": 0.044,
      "tpot": 0.016
    }
  ],
  "metrics": {
    "throughput_tps": 45.2,
    "latency_p50": 2.1,
    "latency_p95": 5.4,
    "latency_p99": 8.2
  }
}
```

### Power CSV

```csv
timestamp,power.draw [W]
2025-05-12 01:10:05.123,450.23
2025-05-12 01:10:05.223,487.65
...
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SHAREGPT_DATASET_PATH` | Path to ShareGPT dataset |
| `VLLM_SERVER_URL` | Server URL (default: http://localhost:8000) |

## See Also

- [profiling/server/README.md](../server/README.md) - Server launch scripts
- [profiling/jobs/README.md](../jobs/README.md) - Automated benchmarking
- [data/README.md](../../data/README.md) - Collected data format
