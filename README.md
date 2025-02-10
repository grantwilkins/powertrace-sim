# Power Trace Simulator Test Bench

This is a test bench for collecting data to train a power trace generator for AI inference. Using vLLM as our inference engine, this provides scripts and client-side code to collect power traces from NVIDIA GPUs running inference for Poisson-distributed arrivals to LLMs of various sizes.

## Tested Models

- Llama-3 (8B, 70B, 405B)
- DeepSeek-R1 (8B, 70B, 671B)

## Tested Hardware

- NVIDIA A100 & H100

## Dependencies

- Python 3.12
- See `requirements.txt` for Python dependencies

