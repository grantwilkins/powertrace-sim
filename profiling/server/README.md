# Server Scripts

This directory contains thin shell wrappers around `vllm serve` for the model configurations used during profiling.

## Scripts

| Script | Model | Description |
|--------|-------|-------------|
| `serve-llama-3-8b.sh` | `meta-llama/Llama-3.1-8B-Instruct` | Llama 3 8B wrapper |
| `serve-llama-3-70b.sh` | `meta-llama/Llama-3.1-70B-Instruct` | Llama 3 70B wrapper |
| `serve-llama-3-405b.sh` | `meta-llama/Llama-3.1-405B-Instruct-FP8` | Llama 3 405B FP8 wrapper |
| `serve-deepseek-r1-distill-8b.sh` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Distilled reasoning model wrapper |
| `serve-deepseek-r1-distill-70b.sh` | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | Distilled reasoning model wrapper |
| `serve-deepseek-r1.sh` | `deepseek-ai/DeepSeek-R1` | Full DeepSeek-R1 wrapper |
| `serve-gpt-oss-20b.sh` | `openai/gpt-oss-20b` | GPT-OSS 20B wrapper |
| `serve-gpt-oss-120b.sh` | `openai/gpt-oss-120b` | GPT-OSS 120B wrapper |

## Usage

These scripts expect `TENSOR_PARALLEL_SIZE` to be set by the caller. They are usually launched from the job scripts in [`../jobs/`](../jobs/).

### Direct Invocation

```bash
cd profiling/server
TENSOR_PARALLEL_SIZE=1 \
bash serve-llama-3-8b.sh
```

### Job Script Entry Point

Example job wrappers:

```bash
bash profiling/jobs/gpt-oss-20b.sh
bash profiling/jobs/gpt-oss-120b.sh
```

## Environment Variables

- `TENSOR_PARALLEL_SIZE`: Required by the wrappers.
- `OPENAI_API_KEY`: Required by `serve-deepseek-r1.sh`.
- `VLLM_TORCH_PROFILER_DIR`: Used by profiling runs when enabled.
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN`: Used by the long-context wrappers when needed.
- `CUDA_VISIBLE_DEVICES`: Standard CUDA device selection variable.

## Hardware Requirements

| Model | Minimum GPUs | Recommended |
|-------|--------------|-------------|
| Llama 3 8B | 1x A100/H100 | 1x H100 |
| Llama 3 70B | 4x A100/H100 | 8x H100 |
| Llama 3 405B | 8x H100 | 8x H100 (80GB) |
| DeepSeek-R1 Distill 8B | 1x A100/H100 | 1x H100 |
| DeepSeek-R1 Distill 70B | 4x A100/H100 | 8x H100 |
| GPT-OSS 20B | 1x A100/H100 | 1x H100 |
| GPT-OSS 120B | 4x A100/H100 | 8x H100 |

## See Also

- [profiling/jobs/README.md](../jobs/README.md) - Automated benchmarking scripts
- [profiling/client/README.MD](../client/README.MD) - Benchmark client documentation
