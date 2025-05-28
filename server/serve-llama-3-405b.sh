cd ~/powertrace-sim/server/
vllm serve meta-llama/Llama-3.1-405B-Instruct-FP8  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --max-num-seqs 64 
