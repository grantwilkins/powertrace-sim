cd ~/powertrace-sim/server/
vllm serve openai/gpt-oss-20b --async-scheduling --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
