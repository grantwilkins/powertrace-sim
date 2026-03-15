cd ~/powertrace-sim/profiling/server/
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}  
