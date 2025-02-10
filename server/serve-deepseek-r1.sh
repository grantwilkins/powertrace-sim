cd ~/
export HF_HOME=~/.cache/huggingface
vllm serve deepseek-ai/DeepSeek-R1 --enable-reasoning 
\ --reasoning-parser deepseek_r1 
\ --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
\ --api-key ${OPENAI_API_KEY}

