<<<<<<< HEAD:server/serve-deepseek-r1-distill-70b.sh
cd ~/powertrace-sim/server/
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B --enable-reasoning --reasoning-parser deepseek_r1 --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --async-scheduling
=======
cd ~/powertrace-sim/profiling/server/
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B --enable-reasoning --reasoning-parser deepseek_r1 --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} 
>>>>>>> grant/moe-exploration:profiling/server/serve-deepseek-r1-distill-70b.sh
