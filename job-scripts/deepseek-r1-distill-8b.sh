TENSOR_PARALLEL_SIZES=(2 4 8)
for TENSOR_PARALLEL_SIZE in ${TENSOR_PARALLEL_SIZES[@]}; do
    if [ ${TENSOR_PARALLEL_SIZE} -eq 1 ]; then
        export CUDA_VISIBLE_DEVICES=0
    elif [ ${TENSOR_PARALLEL_SIZE} -eq 2 ]; then
        export CUDA_VISIBLE_DEVICES=0,1
    elif [ ${TENSOR_PARALLEL_SIZE} -eq 4 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3
    elif [ ${TENSOR_PARALLEL_SIZE} -eq 8 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    fi
    export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}
    cd ~/powertrace-sim/server
    bash serve-deepseek-r1-distill-8b.sh &
    export SERVING_PID=$!
    while ! curl -s http://localhost:8000/v1/completions > /dev/null; do
        echo "Waiting for server to start..."
        sleep 10
    done
    cd ~/powertrace-sim/client
    POISSON_ARRIVAL_RATES=(0.125 0.25 0.5 1.0 2.0 4.0)
    for POISSON_ARRIVAL_RATE in ${POISSON_ARRIVAL_RATES[@]}; do
        DATE_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
        touch deepseek-r1-distill-8b_tp${TENSOR_PARALLEL_SIZE}_p${POISSON_ARRIVAL_RATE}_d${DATE_TIME}.csv
        nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv -lms 250 >> deepseek-r1-distill-8b_tp${TENSOR_PARALLEL_SIZE}_p${POISSON_ARRIVAL_RATE}_d${DATE_TIME}.csv &
        NVIDIA_SMI_PID=$!
        python3 client.py --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --api-key ${OPENAI_API_KEY} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --poisson-arrival-rate ${POISSON_ARRIVAL_RATE}
        kill -9 ${NVIDIA_SMI_PID}
    done
    pkill -9 ${SERVING_PID}
done
