FROM vllm/vllm-openai:latest
ENV MODEL_NAME meta-llama/Meta-Llama-3-8B-Instruct
ENTRYPOINT python3 -m vllm.entrypoints.openai.api_server --model $MODEL_NAME $VLLM_ARGS