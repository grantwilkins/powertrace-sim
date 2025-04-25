TENSOR_PARALLEL_SIZES=(1 2 4 8)
for TENSOR_PARALLEL_SIZE in ${TENSOR_PARALLEL_SIZES[@]}; do
    export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}
    cd ~/powertrace-sim/server
    setsid bash serve-deepseek-r1-distill-8b.sh >/dev/null 2>&1 &
    SERVING_PID=$!
    SERVING_PGID=$(ps -o pgid= -p "$SERVING_PID" | tr -d ' ')
    while ! curl -s -f http://localhost:8000/health &> /dev/null; do
        echo "Waiting for server to start..."
        sleep 10
    done
    cd ~/powertrace-sim/client
    POISSON_ARRIVAL_RATES=(0.125 0.25 0.5 1.0 2.0 4.0)
    NUM_ITERATIONS=5
    for ((i=0; i<NUM_ITERATIONS; i++)); do
        POISSON_ARRIVAL_RATES+=(${POISSON_ARRIVAL_RATES[@]})
    done
    for POISSON_ARRIVAL_RATE in ${POISSON_ARRIVAL_RATES[@]}; do
        DATE_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
        touch deepseek-r1-distill-8b_tp${TENSOR_PARALLEL_SIZE}_p${POISSON_ARRIVAL_RATE}_d${DATE_TIME}.csv
        nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv -lms 250 >> deepseek-r1-distill-8b_tp${TENSOR_PARALLEL_SIZE}_p${POISSON_ARRIVAL_RATE}_d${DATE_TIME}.csv &
        NVIDIA_SMI_PID=$!
        python3 client.py --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --api-key ${OPENAI_API_KEY} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --poisson-arrival-rate ${POISSON_ARRIVAL_RATE}
        kill -9 ${NVIDIA_SMI_PID}
    done
    kill -TERM -- "-$SERVING_PGID"
    sleep 5
    kill -KILL -- "-$SERVING_PGID" 2>/dev/null || true
done
