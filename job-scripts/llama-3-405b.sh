#!/bin/bash

# Configuration
TENSOR_PARALLEL_SIZES=(8)
WORKLOAD_INTENSITIES=(low medium high ultra)
WORKLOAD_TASKS=(conversation coding)
ITERATIONS=5

# Generate arrival rates as powers of 4: 4^-3 to 4^3
# 4^-3=0.015625, 4^-2=0.0625, 4^-1=0.25, 4^0=1, 4^1=4, 4^2=16, 4^3=64
ARRIVAL_RATES=(0.015625 0.0625 0.25 1 4 16 64)

for TENSOR_PARALLEL_SIZE in ${TENSOR_PARALLEL_SIZES[@]}; do
    export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}
    cd ~/powertrace-sim/server
    setsid bash serve-llama-3-405b.sh 2>&1 &
    SERVING_PID=$!                        
    SERVING_PGID=$(ps -o pgid= -p "$SERVING_PID" | tr -d ' ')
    
    echo "Waiting for server to start (TP=${TENSOR_PARALLEL_SIZE})..."
    while ! curl -s -f http://localhost:8000/health &> /dev/null; do
        echo "  Still waiting..."
        sleep 10
    done
    echo "Server ready!"
    
    cd ~/powertrace-sim/client

    # Loop over workload types (conversation, coding)
    for WORKLOAD_TASK in ${WORKLOAD_TASKS[@]}; do
        # Loop over workload intensities (low, medium, high, ultra)
        for WORKLOAD_INTENSITY in ${WORKLOAD_INTENSITIES[@]}; do
            echo "Running: Task=${WORKLOAD_TASK}, Intensity=${WORKLOAD_INTENSITY}, TP=${TENSOR_PARALLEL_SIZE}"
            
            # Loop over arrival rates
            for ARRIVAL_RATE in ${ARRIVAL_RATES[@]}; do
                # Run 5 iterations for each configuration
                for ITERATION in $(seq 1 ${ITERATIONS}); do
                    DATE_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
                    OUTPUT_PREFIX="llama-3-405b_tp${TENSOR_PARALLEL_SIZE}_${WORKLOAD_TASK}_${WORKLOAD_INTENSITY}_rate${ARRIVAL_RATE}_iter${ITERATION}_${DATE_TIME}"
                    
                    echo "  Rate=${ARRIVAL_RATE} req/s, Iteration=${ITERATION}/${ITERATIONS}"
                    
                    # Start GPU monitoring
                    touch ${OUTPUT_PREFIX}.csv
                    nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv -lms 250 >> ${OUTPUT_PREFIX}.csv &
                    NVIDIA_SMI_PID=$!
                    
                    # Calculate number of prompts for 10 minutes at this arrival rate
                    NUM_PROMPTS=$(printf "%.0f" $(echo "600 * ${ARRIVAL_RATE}" | bc))
                    
                    # Run benchmark
                    python3 benchmark_serving.py \
                        --model meta-llama/Llama-3.1-405B-Instruct \
                        --backend vllm \
                        --dataset-name realistic \
                        --workload-task ${WORKLOAD_TASK} \
                        --workload-intensity ${WORKLOAD_INTENSITY} \
                        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
                        --request-rate ${ARRIVAL_RATE} \
                        --num-prompts ${NUM_PROMPTS} \
                        --endpoint /v1/completions \
                        --save-result \
                        --save-detailed
                    
                    # Stop GPU monitoring
                    kill -9 ${NVIDIA_SMI_PID}
                done
            done
        done
    done
    
    # Shutdown server
    echo "Shutting down server (TP=${TENSOR_PARALLEL_SIZE})..."
    kill -TERM -- "-$SERVING_PGID"
    sleep 5
    kill -KILL -- "-$SERVING_PGID" 2>/dev/null || true
done

echo "All benchmarks complete!"
