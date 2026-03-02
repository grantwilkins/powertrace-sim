#!/bin/bash

set -euo pipefail

TENSOR_PARALLEL_SIZES=(4 8)
ARRIVAL_RATES=(0.125 0.25 0.5 1 2 4)
ITERATIONS=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${SHAREGPT_DATASET_PATH:-}" ]]; then
    echo "Error: SHAREGPT_DATASET_PATH is required."
    exit 1
fi
if [[ ! -e "${SHAREGPT_DATASET_PATH}" ]]; then
    echo "Error: SHAREGPT_DATASET_PATH does not exist: ${SHAREGPT_DATASET_PATH}"
    exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "${GPU_NAME}" == *"H100"* ]]; then
    HARDWARE_SUFFIX="h100"
elif [[ "${GPU_NAME}" == *"A100"* ]]; then
    HARDWARE_SUFFIX="a100"
else
    echo "Error: Unsupported GPU for this script: ${GPU_NAME}"
    echo "Expected an A100 or H100 host."
    exit 1
fi

OUTPUT_DIR="${REPO_ROOT}/data/sharegpt-benchmark-gpt-oss-120b-${HARDWARE_SUFFIX}"
mkdir -p "${OUTPUT_DIR}"

SERVING_PGID=""
SERVING_PID=""
NVIDIA_SMI_PID=""

stop_nvidia_smi() {
    if [[ -n "${NVIDIA_SMI_PID}" ]] && kill -0 "${NVIDIA_SMI_PID}" 2>/dev/null; then
        kill "${NVIDIA_SMI_PID}" 2>/dev/null || true
        wait "${NVIDIA_SMI_PID}" 2>/dev/null || true
    fi
    NVIDIA_SMI_PID=""
}

stop_server() {
    if [[ -n "${SERVING_PGID}" ]]; then
        kill -TERM -- "-${SERVING_PGID}" 2>/dev/null || true
        sleep 5
        kill -KILL -- "-${SERVING_PGID}" 2>/dev/null || true
    elif [[ -n "${SERVING_PID}" ]] && kill -0 "${SERVING_PID}" 2>/dev/null; then
        kill -TERM "${SERVING_PID}" 2>/dev/null || true
        sleep 2
        kill -KILL "${SERVING_PID}" 2>/dev/null || true
    fi
    SERVING_PGID=""
    SERVING_PID=""
}

cleanup() {
    stop_nvidia_smi
    stop_server
}

trap cleanup EXIT INT TERM

wait_for_server() {
    local max_attempts=90
    local attempt=1
    while ! curl -s -f http://localhost:8000/health >/dev/null; do
        if (( attempt >= max_attempts )); then
            echo "Error: vLLM server did not become healthy in time."
            return 1
        fi
        echo "Waiting for server health endpoint... (${attempt}/${max_attempts})"
        sleep 5
        attempt=$((attempt + 1))
    done
}

start_server() {
    local tp="$1"
    export TENSOR_PARALLEL_SIZE="${tp}"
    pushd "${REPO_ROOT}/server" >/dev/null
    setsid bash serve-gpt-oss-120b.sh > "${OUTPUT_DIR}/server-tp${tp}.log" 2>&1 &
    SERVING_PID=$!
    SERVING_PGID="$(ps -o pgid= -p "${SERVING_PID}" | tr -d ' ')"
    popd >/dev/null
    wait_for_server
}

for TENSOR_PARALLEL_SIZE in "${TENSOR_PARALLEL_SIZES[@]}"; do
    echo "Starting gpt-oss-120b benchmark at TP=${TENSOR_PARALLEL_SIZE}"
    start_server "${TENSOR_PARALLEL_SIZE}"

    pushd "${REPO_ROOT}/client" >/dev/null
    for ARRIVAL_RATE in "${ARRIVAL_RATES[@]}"; do
        NUM_PROMPTS="$(awk -v r="${ARRIVAL_RATE}" 'BEGIN {printf "%.0f", 600 * r}')"
        for ITERATION in $(seq 1 "${ITERATIONS}"); do
            DATE_KEY="$(date '+%Y%m%d-%H%M%S')"
            JSON_FILE="vllm-${ARRIVAL_RATE}qps-tp${TENSOR_PARALLEL_SIZE}-gpt-oss-120b-${DATE_KEY}.json"
            CSV_FILE="gpt-oss-120b_tp${TENSOR_PARALLEL_SIZE}_p${ARRIVAL_RATE}_d${DATE_KEY}.csv"
            CSV_PATH="${OUTPUT_DIR}/${CSV_FILE}"

            echo "TP=${TENSOR_PARALLEL_SIZE} rate=${ARRIVAL_RATE} iter=${ITERATION}/${ITERATIONS} prompts=${NUM_PROMPTS}"

            touch "${CSV_PATH}"
            nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv -lms 250 >> "${CSV_PATH}" &
            NVIDIA_SMI_PID=$!

            python3 benchmark_serving.py \
                --model openai/gpt-oss-120b \
                --backend vllm \
                --dataset-name sharegpt \
                --dataset-path "${SHAREGPT_DATASET_PATH}" \
                --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
                --request-rate "${ARRIVAL_RATE}" \
                --num-prompts "${NUM_PROMPTS}" \
                --endpoint /v1/completions \
                --save-result \
                --save-detailed \
                --result-dir "${OUTPUT_DIR}" \
                --result-filename "${JSON_FILE}"

            stop_nvidia_smi
        done
    done
    popd >/dev/null

    stop_server
done

echo "All GPT-OSS-120B ShareGPT benchmarks complete. Output: ${OUTPUT_DIR}"
