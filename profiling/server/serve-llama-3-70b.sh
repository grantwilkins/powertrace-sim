<<<<<<< HEAD:server/serve-llama-3-70b.sh
cd ~/powertrace-sim/server/
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --async-scheduling
=======
cd ~/powertrace-sim/profiling/server/
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}  
>>>>>>> grant/moe-exploration:profiling/server/serve-llama-3-70b.sh
