#!/bin/bash

# Configuration
TENSOR_PARALLEL_SIZES=(8)
ALL_DATASETS=(likaixin/InstructCoder AI-MO/aimo-validation-aime vdaita/edit_10k_char)
ITERATIONS=5

ARRIVAL_RATES=(0.0625 0.25 1 4)

random_datasets() {
    echo ${ALL_DATASETS[$RANDOM % ${#ALL_DATASETS[@]}]}
}

for TENSOR_PARALLEL_SIZE in ${TENSOR_PARALLEL_SIZES[@]}; do
    export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}
    cd ~/powertrace-sim/server
    setsid bash serve-gpt-oss-120b.sh 2>&1 &
    SERVING_PID=$!
    SERVING_PGID=$(ps -o pgid= -p "$SERVING_PID" | tr -d ' ')

    echo "Waiting for server to start (TP=${TENSOR_PARALLEL_SIZE})..."
    while ! curl -s -f http://localhost:8000/health &> /dev/null; do
        echo "  Still waiting..."
        sleep 10
    done
    echo "Server ready!"
    cd ~/powertrace-sim/client

    for ARRIVAL_RATE in ${ARRIVAL_RATES[@]}; do
        echo "Running arrival rate: ${ARRIVAL_RATE} req/s (TP=${TENSOR_PARALLEL_SIZE})"
        for ITERATION in $(seq 1 ${ITERATIONS}); do
            WORKLOAD_DATASET=$(random_datasets)
            DATE_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
            OUTPUT_PREFIX="gpt-oss-120b_tp${TENSOR_PARALLEL_SIZE}_rate${ARRIVAL_RATE}_iter${ITERATION}_${DATE_TIME}"
            touch ${OUTPUT_PREFIX}.csv
            nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv -lms 250 >> ${OUTPUT_PREFIX}.csv &
            NVIDIA_SMI_PID=$!
            NUM_PROMPTS=$(printf "%.0f" $(echo "300 * ${ARRIVAL_RATE}" | bc))

            vllm bench serve \
                --model openai/gpt-oss-120b \
                --dataset-name hf \
                --dataset-path ${WORKLOAD_DATASET} \
                --request-rate ${ARRIVAL_RATE} \
                --num-prompts ${NUM_PROMPTS} \
                --save-result \
                --save-detailed \
                --result-dir . \
                --result-filename ${OUTPUT_PREFIX}.json

            kill -9 ${NVIDIA_SMI_PID}
        done
    done

    # Shutdown server
    echo "Shutting down server (TP=${TENSOR_PARALLEL_SIZE})..."
    kill -TERM -- "-$SERVING_PGID"
    sleep 5
    kill -KILL -- "-$SERVING_PGID" 2>/dev/null || true
done

echo "All benchmarks complete!"
