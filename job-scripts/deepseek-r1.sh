export TENSOR_PARALLEL_SIZE=8
cd ~/powertrace-sim/server
bash serve-deepseek-r1.sh &
while ! curl -s http://localhost:8000/v1/completions > /dev/null; do
    echo "Waiting for server to start..."
    sleep 10
done
cd ~/powertrace-sim/client
POISSON_ARRIVAL_RATES=(1 2 4 8 16 32 64)
for POISSON_ARRIVAL_RATE in ${POISSON_ARRIVAL_RATES[@]}; do
    touch deepseek-r1_tp${TENSOR_PARALLEL_SIZE}_p${POISSON_ARRIVAL_RATE}.csv
    nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv -l 1 >> deepseek-r1_tp${TENSOR_PARALLEL_SIZE}_p${POISSON_ARRIVAL_RATE}.csv &
    NVIDIA_SMI_PID=$!
    python3 client.py --model-name deepseek-ai/DeepSeek-R1 --api-key ${OPENAI_API_KEY} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --poisson-arival-rate ${POISSON_ARRIVAL_RATE} --reasoning True
    kill -9 ${NVIDIA_SMI_PID}
done
pkil -9 -f "vllm serve"