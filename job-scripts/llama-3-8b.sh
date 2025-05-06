TENSOR_PARALLEL_SIZES=(1 2 4 8)
for TENSOR_PARALLEL_SIZE in ${TENSOR_PARALLEL_SIZES[@]}; do
    export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}
    cd ~/powertrace-sim/server
    setsid bash serve-llama-3-8b.sh 2>&1 &
    SERVING_PID=$!                        
    SERVING_PGID=$(ps -o pgid= -p "$SERVING_PID" | tr -d ' ')
    while ! curl -s -f http://localhost:8000/health &> /dev/null; do
        echo "Waiting for server to start..."
        sleep 10
    done
    cd ~/powertrace-sim/client
    POISSON_ARRIVAL_RATES=(0.125 0.25 0.5 1.0 2.0 4.0 0.125 0.25 0.5 1.0 2.0 4.0 0.125 0.25 0.5 1.0 2.0 4.0 0.125 0.25 0.5 1.0 2.0 4.0 0.125 0.25 0.5 1.0 2.0 4.0)
    for POISSON_ARRIVAL_RATE in ${POISSON_ARRIVAL_RATES[@]}; do
        DATE_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
        touch llama-3-8b_tp${TENSOR_PARALLEL_SIZE}_p${POISSON_ARRIVAL_RATE}_d${DATE_TIME}.csv
        nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv -lms 250 >> llama-3-8b_tp${TENSOR_PARALLEL_SIZE}_p${POISSON_ARRIVAL_RATE}_d${DATE_TIME}.csv &
        NVIDIA_SMI_PID=$!
        NUM_PROMPTS=$(( 600 * ${POISSON_ARRIVAL_RATE} )) # 900 seconds in 15 minutes
        python3 benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --dataset-name sharegpt --dataset-path ${HOME}/ShareGPT_V3_unfiltered_cleaned_split.json --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --request-rate ${POISSON_ARRIVAL_RATE} --num-prompts ${NUM_PROMPTS} --endpoint /v1/completions --save-result --save-detailed
        kill -9 ${NVIDIA_SMI_PID}
    done
    kill -TERM -- "-$SERVING_PGID"
    sleep 5
    kill -KILL -- "-$SERVING_PGID" 2>/dev/null || true
done
