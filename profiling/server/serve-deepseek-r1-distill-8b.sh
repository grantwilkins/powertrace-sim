<<<<<<< HEAD:server/serve-deepseek-r1-distill-8b.sh
cd ~/powertrace-sim/server/
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B --enable-reasoning --reasoning-parser deepseek_r1 --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --async-scheduling
=======
cd ~/powertrace-sim/profiling/server/
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B --enable-reasoning --reasoning-parser deepseek_r1 --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
>>>>>>> grant/moe-exploration:profiling/server/serve-deepseek-r1-distill-8b.sh
