<<<<<<< HEAD:server/serve-gpt-oss-20b.sh
cd ~/powertrace-sim/server/
=======
cd ~/powertrace-sim/profiling/server/
>>>>>>> grant/moe-exploration:profiling/server/serve-gpt-oss-20b.sh
vllm serve openai/gpt-oss-20b --async-scheduling --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
