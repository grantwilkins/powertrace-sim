cd ~/
vllm serve meta-llama/Llama-3.1-405B-Instruct
\ --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
\ --api-key ${OPENAI_API_KEY}