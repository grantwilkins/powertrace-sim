cd ~/powertrace-sim/server/
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --api-key ${OPENAI_API_KEY}
