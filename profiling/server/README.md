# Server Scripts

This directory contains shell scripts for launching vLLM inference servers for different model configurations. These scripts are used during data collection to serve models for benchmarking.

## Scripts

| Script | Model | Description |
|--------|-------|-------------|
| `serve-llama-3-8b.sh` | Llama-3 8B | Small Llama model |
| `serve-llama-3-70b.sh` | Llama-3 70B | Large Llama model |
| `serve-llama-3-405b.sh` | Llama-3 405B | Extra-large Llama model |
| `serve-deepseek-r1-distill-8b.sh` | DeepSeek-R1-Distill 8B | Small reasoning model |
| `serve-deepseek-r1-distill-70b.sh` | DeepSeek-R1-Distill 70B | Large reasoning model |
| `serve-deepseek-r1.sh` | DeepSeek-R1 | Full reasoning model |
| `serve-gpt-oss-20b.sh` | GPT-OSS 20B | Small GPT model |
| `serve-gpt-oss-120b.sh` | GPT-OSS 120B | Large GPT model |

## Usage

### Direct Invocation

```bash
# Start a server manually
cd profiling/server
bash serve-llama-3-8b.sh
```

### From Job Scripts (Recommended)

The server scripts are typically invoked by the job scripts in `profiling/jobs/`:

```bash
# This will start the server, run benchmarks, and collect data
bash profiling/jobs/llama-3-8b.sh
```

## Server Configuration

Each script configures vLLM with appropriate settings:

- **Port**: `localhost:8000` (default)
- **Tensor Parallelism**: Set via `--tensor-parallel-size`
- **Model Path**: Configured for each model
- **Max Model Length**: Set appropriately for each model size

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VLLM_TORCH_PROFILER_DIR` | Enable vLLM profiler (optional) |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | Allow longer context lengths |
| `CUDA_VISIBLE_DEVICES` | Specify GPU devices to use |

## Example Script Content

```bash
#!/bin/bash
# serve-llama-3-8b.sh

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 8192
```

## Hardware Requirements

| Model | Minimum GPUs | Recommended |
|-------|--------------|-------------|
| Llama-3 8B | 1x A100/H100 | 1x H100 |
| Llama-3 70B | 4x A100/H100 | 8x H100 |
| Llama-3 405B | 8x H100 | 8x H100 (80GB) |
| DeepSeek-R1 8B | 1x A100/H100 | 1x H100 |
| DeepSeek-R1 70B | 4x A100/H100 | 8x H100 |

## See Also

- [profiling/jobs/README.md](../jobs/README.md) - Automated benchmarking scripts
- [profiling/client/README.MD](../client/README.MD) - Benchmark client documentation
