#!/bin/bash

<<<<<<< HEAD:job-scripts/gpt-oss-120b.sh
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
=======
set -Eeuo pipefail
if [[ "${DEBUG:-0}" == "1" ]]; then
    set -x
fi

TENSOR_PARALLEL_SIZES=(4 8)
ARRIVAL_RATES=(0.125 0.25 0.5 1 2 4)
ITERATIONS=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CURRENT_SERVER_LOG=""
CURRENT_JSON_FILE=""
CURRENT_CSV_PATH=""

require_command() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "Error: required command '${cmd}' is not available in PATH."
        exit 1
    fi
}

on_error() {
    local exit_code="$?"
    local line_no="$1"
    echo "Error: command failed at line ${line_no}: ${BASH_COMMAND}"
    echo "Exit code: ${exit_code}"
    if [[ -n "${CURRENT_JSON_FILE}" ]]; then
        echo "Last target JSON: ${CURRENT_JSON_FILE}"
    fi
    if [[ -n "${CURRENT_CSV_PATH}" ]]; then
        echo "Last target CSV: ${CURRENT_CSV_PATH}"
    fi
    if [[ -n "${CURRENT_SERVER_LOG}" && -f "${CURRENT_SERVER_LOG}" ]]; then
        echo "Last 80 lines from server log (${CURRENT_SERVER_LOG}):"
        tail -n 80 "${CURRENT_SERVER_LOG}" || true
    fi
    exit "${exit_code}"
}

trap 'on_error ${LINENO}' ERR

if [[ -z "${SHAREGPT_DATASET_PATH:-}" ]]; then
    echo "Error: SHAREGPT_DATASET_PATH is required."
    exit 1
fi
if [[ ! -e "${SHAREGPT_DATASET_PATH}" ]]; then
    echo "Error: SHAREGPT_DATASET_PATH does not exist: ${SHAREGPT_DATASET_PATH}"
    exit 1
fi

require_command nvidia-smi
require_command curl
require_command awk
require_command python3
require_command setsid
require_command ps
require_command find
require_command wc

echo "Starting GPT-OSS-120B ShareGPT benchmark script"
echo "Repo root: ${REPO_ROOT}"
echo "ShareGPT dataset: ${SHAREGPT_DATASET_PATH}"

HARDWARE_SUFFIX="a100"

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
    local max_attempts=500
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
    CURRENT_SERVER_LOG="${OUTPUT_DIR}/server-tp${tp}.log"
    echo "Launching server for TP=${tp}, log=${CURRENT_SERVER_LOG}"
    pushd "${REPO_ROOT}/profiling/server" >/dev/null
    setsid bash serve-gpt-oss-120b.sh > "${CURRENT_SERVER_LOG}" 2>&1 &
    SERVING_PID=$!
    SERVING_PGID="$(ps -o pgid= -p "${SERVING_PID}" | tr -d ' ')"
    popd >/dev/null
    if ! kill -0 "${SERVING_PID}" 2>/dev/null; then
        echo "Error: server process exited immediately for TP=${tp}."
        tail -n 80 "${CURRENT_SERVER_LOG}" || true
        return 1
    fi
    wait_for_server
    echo "Server is healthy for TP=${tp}"
}

count_completed_json() {
    local tp="$1"
    local rate="$2"
    local pattern="vllm-${rate}qps-tp${tp}-gpt-oss-120b-*.json"
    find "${OUTPUT_DIR}" -maxdepth 1 -type f -name "${pattern}" | wc -l | awk '{print $1}'
}

for TENSOR_PARALLEL_SIZE in "${TENSOR_PARALLEL_SIZES[@]}"; do
    echo "Starting gpt-oss-120b benchmark at TP=${TENSOR_PARALLEL_SIZE}"
    start_server "${TENSOR_PARALLEL_SIZE}"

    pushd "${REPO_ROOT}/profiling/client" >/dev/null
    for ARRIVAL_RATE in "${ARRIVAL_RATES[@]}"; do
        NUM_PROMPTS="$(awk -v r="${ARRIVAL_RATE}" 'BEGIN {printf "%.0f", 600 * r}')"
        COMPLETED_RUNS="$(count_completed_json "${TENSOR_PARALLEL_SIZE}" "${ARRIVAL_RATE}")"
        if (( COMPLETED_RUNS >= ITERATIONS )); then
            echo "Skipping TP=${TENSOR_PARALLEL_SIZE} rate=${ARRIVAL_RATE}: found ${COMPLETED_RUNS}/${ITERATIONS} JSON files."
            continue
        fi
        echo "Resuming TP=${TENSOR_PARALLEL_SIZE} rate=${ARRIVAL_RATE}: completed ${COMPLETED_RUNS}/${ITERATIONS}, running remaining."

        for ((ITERATION=COMPLETED_RUNS + 1; ITERATION<=ITERATIONS; ITERATION++)); do
            DATE_KEY="$(date '+%Y%m%d-%H%M%S')"
            JSON_FILE="vllm-${ARRIVAL_RATE}qps-tp${TENSOR_PARALLEL_SIZE}-gpt-oss-120b-${DATE_KEY}.json"
            CSV_FILE="gpt-oss-120b_tp${TENSOR_PARALLEL_SIZE}_p${ARRIVAL_RATE}_d${DATE_KEY}.csv"
            RUN_LOG="${OUTPUT_DIR}/run-gpt-oss-120b-tp${TENSOR_PARALLEL_SIZE}-rate${ARRIVAL_RATE}-iter${ITERATION}-${DATE_KEY}.log"
            CSV_PATH="${OUTPUT_DIR}/${CSV_FILE}"
            CURRENT_JSON_FILE="${OUTPUT_DIR}/${JSON_FILE}"
            CURRENT_CSV_PATH="${CSV_PATH}"

            echo "TP=${TENSOR_PARALLEL_SIZE} rate=${ARRIVAL_RATE} iter=${ITERATION}/${ITERATIONS} prompts=${NUM_PROMPTS}"
            echo "  -> json: ${CURRENT_JSON_FILE}"
            echo "  -> csv:  ${CURRENT_CSV_PATH}"
            echo "  -> log:  ${RUN_LOG}"

            touch "${CSV_PATH}"
            nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv -lms 250 >> "${CSV_PATH}" &
            NVIDIA_SMI_PID=$!

            if ! python3 benchmark_serving.py \
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
                --result-filename "${JSON_FILE}" 2>&1 | tee "${RUN_LOG}"; then
                echo "Error: benchmark_serving.py failed for TP=${TENSOR_PARALLEL_SIZE}, rate=${ARRIVAL_RATE}, iter=${ITERATION}"
                echo "Tail of run log (${RUN_LOG}):"
                tail -n 120 "${RUN_LOG}" || true
                return 1
            fi

            stop_nvidia_smi
        done
    done
    popd >/dev/null

    stop_server
done

echo "All GPT-OSS-120B ShareGPT benchmarks complete. Output: ${OUTPUT_DIR}"
>>>>>>> grant/moe-exploration:profiling/jobs/gpt-oss-120b.sh
