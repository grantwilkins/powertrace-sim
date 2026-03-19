cd ~/powertrace-sim/profiling/server/
vllm serve openai/gpt-oss-120b --async-scheduling --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
