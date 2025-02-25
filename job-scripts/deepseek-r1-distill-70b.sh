TENSOR_PARALLEL_SIZES=(2 4 8)
for TENSOR_PARALLEL_SIZE in ${TENSOR_PARALLEL_SIZES[@]}; do
    export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}
    cd ~/powertrace-sim/server
    bash serve-deepseek-r1-distill-70b.sh &
    while ! curl -s http://localhost:8000/v1/completions > /dev/null; do
        echo "Waiting for server to start..."
        sleep 10
    done
    cd ~/powertrace-sim/client
    POISSON_ARRIVAL_RATES=(1 2 4 8 16 32 64)
    for POISSON_ARRIVAL_RATE in ${POISSON_ARRIVAL_RATES[@]}; do
        python3 client.py --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-70B --api-key ${OPENAI_API_KEY} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --poisson-arival-rate ${POISSON_ARRIVAL_RATE} --reasoning True
    done
    pkill -f "vllm serve"
done
