#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${POWERTRACE_REPO:-$HOME/powertrace-sim}"
ROOT="${SCRATCH:?Sherlock SCRATCH is required}/ptsim"
IMAGE="${POWERTRACE_DISAGG_IMAGE:-$ROOT/lmcache-v0.5.1-vllm0.22.0-cu129-primary.sif}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/gpt-oss-20b-a100-pd-transfer-${SLURM_JOB_ID:-local}}"
MODE="${1:-campaign}"
MODEL=openai/gpt-oss-20b
START_OFFSET="${POWERTRACE_PORT_OFFSET:-$(( (${SLURM_JOB_ID:-1} % 5000) * 4 ))}"
OFFSET="$(python3 profiling/disaggregated_prefill/transfer_campaign.py ports \
    --start-offset "$START_OFFSET")"
PREFILL_PORT=$((21000 + OFFSET))
DECODE_PORT=$((21001 + OFFSET))
PROXY_PORT=$((21002 + OFFSET))
PREFILL_SIDE_PORT=$((31000 + OFFSET))
DECODE_SIDE_PORT=$((31001 + OFFSET))

test "$MODE" = campaign || test "$MODE" = smoke || {
    echo "usage: $0 [campaign|smoke]" >&2
    exit 2
}
test -e "$IMAGE" || { echo "MISSING Apptainer image: $IMAGE" >&2; exit 1; }
test -s "$ROOT/hf/hub/models--openai--gpt-oss-20b/.powertrace-stage-complete" || {
    echo "INCOMPLETE staged model: $MODEL" >&2
    exit 1
}
mkdir -p "$RUN_ROOT/plans"
cd "$REPO"
test ! -e "$RUN_ROOT/run_metadata.json" || {
    echo "transfer runs cannot resume across allocations: $RUN_ROOT" >&2
    exit 1
}
git rev-parse HEAD > "$RUN_ROOT/git_sha.txt"
git status --short > "$RUN_ROOT/git_status.txt"
cp profiling/jobs/run_disaggregated_transfer_gpt_oss_20b.sh \
    "$RUN_ROOT/effective_runner.sh"

GPU_LIST="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}"
IFS=, read -r GPU0 GPU1 REST <<< "$GPU_LIST"
test -n "${GPU0:-}" && test -n "${GPU1:-}" && test -z "${REST:-}" || {
    echo "exactly two allocated GPUs are required; got '$GPU_LIST'" >&2
    exit 1
}

NIXL_SITE=/opt/venv/lib/python3.12/site-packages
NIXL_COMPAT="$RUN_ROOT/python_compat"
mkdir -p "$NIXL_COMPAT"
ln -sfn "$NIXL_SITE/nixl_cu12" "$NIXL_COMPAT/nixl"
NIXL_LIBS="$NIXL_SITE/nixl_cu12.libs:$NIXL_SITE/.nixl_cu12.mesonpy.libs"
APP=(apptainer exec --nv --bind "$SCRATCH" --bind "$GROUP_HOME" --bind "$REPO"
    --env "PYTHONPATH=$NIXL_COMPAT" --env "LD_LIBRARY_PATH=$NIXL_LIBS"
    --env UCX_RCACHE_MAX_UNRELEASED=1024 --env "HF_HOME=$ROOT/hf"
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$IMAGE")
"${APP[@]}" python3 -c 'import fastapi,httpx,nixl,uvicorn,vllm; from vllm.distributed.nixl_utils import NixlWrapper,nixl_agent_config; assert vllm.__version__.startswith("0.22."), vllm.__version__; assert NixlWrapper is not None and nixl_agent_config is not None; print("vllm", vllm.__version__)' \
    | tee "$RUN_ROOT/runtime.txt"
nvidia-smi topo -m > "$RUN_ROOT/gpu_topology.txt"
GPU0_UUID="$(nvidia-smi -i "$GPU0" --query-gpu=uuid --format=csv,noheader | tr -d ' ')"
GPU1_UUID="$(nvidia-smi -i "$GPU1" --query-gpu=uuid --format=csv,noheader | tr -d ' ')"
MODE_ARG=()
test "$MODE" = campaign || MODE_ARG=(--smoke)
python3 profiling/disaggregated_prefill/transfer_campaign.py metadata \
    "$RUN_ROOT/run_metadata.json" --prefill-uuid "$GPU0_UUID" \
    --decode-uuid "$GPU1_UUID" --image "$IMAGE" "${MODE_ARG[@]}"

PIDS=()
POWER_PID=""
METRICS_PID=""
cleanup() {
    test -z "$POWER_PID" || kill -TERM "$POWER_PID" 2>/dev/null || true
    test -z "$METRICS_PID" || kill -TERM "$METRICS_PID" 2>/dev/null || true
    for pid in "${PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
    wait "${PIDS[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start_engine() {
    local role="$1" gpu="$2" port="$3" side_port="$4" log="$5"
    env CUDA_VISIBLE_DEVICES="$gpu" APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpu" \
        NVIDIA_VISIBLE_DEVICES="$gpu" VLLM_NIXL_SIDE_CHANNEL_PORT="$side_port" \
        APPTAINERENV_VLLM_NIXL_SIDE_CHANNEL_PORT="$side_port" \
        UCX_NET_DEVICES=all APPTAINERENV_UCX_NET_DEVICES=all \
        APPTAINERENV_HF_HOME="$ROOT/hf" APPTAINERENV_HF_HUB_OFFLINE=1 \
        APPTAINERENV_VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        "${APP[@]}" vllm serve "$MODEL" --host 127.0.0.1 --port "$port" \
        --tensor-parallel-size 1 --max-model-len 131072 --max-num-seqs 256 \
        --max-num-batched-tokens 2048 --gpu-memory-utilization 0.9 \
        --enable-chunked-prefill --no-enable-prefix-caching --async-scheduling \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail"}' \
        > "$log" 2>&1 &
    PIDS+=("$!")
    echo "$role engine pid=${PIDS[-1]} gpu=$gpu port=$port"
}

wait_health() {
    local url="$1" pid="$2" log="$3"
    for _ in $(seq 1 360); do
        kill -0 "$pid" 2>/dev/null || { tail -n 100 "$log"; return 1; }
        curl -fsS "$url/health" >/dev/null && return
        sleep 5
    done
    echo "health timeout: $url" >&2
    tail -n 100 "$log"
    return 1
}

start_engine prefill "$GPU0" "$PREFILL_PORT" "$PREFILL_SIDE_PORT" "$RUN_ROOT/prefill.log"
wait_health "http://127.0.0.1:$PREFILL_PORT" "${PIDS[0]}" "$RUN_ROOT/prefill.log"
start_engine decode "$GPU1" "$DECODE_PORT" "$DECODE_SIDE_PORT" "$RUN_ROOT/decode.log"
wait_health "http://127.0.0.1:$DECODE_PORT" "${PIDS[1]}" "$RUN_ROOT/decode.log"
"${APP[@]}" python3 profiling/disaggregated_prefill/proxy.py \
    --prefill-url "http://127.0.0.1:$PREFILL_PORT" \
    --decode-url "http://127.0.0.1:$DECODE_PORT" \
    --port "$PROXY_PORT" --events "$RUN_ROOT/proxy_events.jsonl" \
    > "$RUN_ROOT/proxy.log" 2>&1 &
PIDS+=("$!")
wait_health "http://127.0.0.1:$PROXY_PORT" "${PIDS[2]}" "$RUN_ROOT/proxy.log"

PLAN_ARGS=()
test "$MODE" = campaign || PLAN_ARGS=(--smoke)
mapfile -t CELLS < <(
    python3 profiling/disaggregated_prefill/transfer_campaign.py plan "${PLAN_ARGS[@]}"
)
for cell in "${CELLS[@]}"; do
    IFS=$'\t' read -r TAG RATE PROMPTS SEED SPLIT PLAN_KEY <<< "$cell"
    PLAN="$RUN_ROOT/plans/$PLAN_KEY.json"
    test -e "$PLAN" && continue
    "${APP[@]}" python3 profiling/disaggregated_prefill/planned_workload.py generate \
        --output "$PLAN" --model "$MODEL" --request-rate "$RATE" \
        --num-requests "$PROMPTS" --seed "$SEED"
done

for cell in "${CELLS[@]}"; do
    IFS=$'\t' read -r TAG RATE PROMPTS SEED SPLIT PLAN_KEY <<< "$cell"
    DIR="$RUN_ROOT/$TAG"
    PLAN="$RUN_ROOT/plans/$PLAN_KEY.json"
    mkdir -p "$DIR"
    EVENT_START="$(wc -l < "$RUN_ROOT/proxy_events.jsonl")"
    python3 profiling/client/power_logger.py --interval-ms 250 \
        --profile core_timed_state --gpu-ids "$GPU0_UUID,$GPU1_UUID" \
        > "$DIR/power.csv" &
    POWER_PID=$!
    python3 -m profiling.disaggregated_prefill.telemetry \
        --prefill-url "http://127.0.0.1:$PREFILL_PORT" \
        --decode-url "http://127.0.0.1:$DECODE_PORT" --out-dir "$DIR" &
    METRICS_PID=$!
    for _ in $(seq 1 80); do
        test "$(wc -l < "$DIR/power.csv")" -ge 3 \
            && test "$(wc -l < "$DIR/engine_prefill.csv")" -ge 2 \
            && test "$(wc -l < "$DIR/engine_decode.csv")" -ge 2 && break
        sleep 0.25
    done
    kill -0 "$POWER_PID" && kill -0 "$METRICS_PID" \
        && test "$(wc -l < "$DIR/power.csv")" -ge 3 || {
        echo "telemetry failed to initialize for $TAG" >&2
        exit 1
    }
    python3 profiling/disaggregated_prefill/transfer_campaign.py origin \
        "$DIR/power.csv" --lead-s 30 > "$DIR/traffic_start_epoch_s"
    set +e
    "${APP[@]}" python3 profiling/disaggregated_prefill/planned_workload.py run \
        --request-plan "$PLAN" \
        --traffic-start "$DIR/traffic_start_epoch_s" \
        --base-url "http://127.0.0.1:$PROXY_PORT" \
        --output "$DIR/requests.json" > "$DIR/benchmark.log" 2>&1
    STATUS=$?
    set -e
    date +%s.%N > "$DIR/traffic_end_epoch_s"
    test "$STATUS" -ne 0 || sleep 30
    kill -TERM "$POWER_PID" "$METRICS_PID"
    wait "$POWER_PID"
    wait "$METRICS_PID"
    POWER_PID=""
    METRICS_PID=""
    test "$STATUS" -eq 0 || { tail -n 120 "$DIR/benchmark.log"; exit "$STATUS"; }
    python3 profiling/disaggregated_prefill/transfer_campaign.py check \
        "$DIR/requests.json" "$PROMPTS" --plan "$PLAN" \
        --events "$RUN_ROOT/proxy_events.jsonl" --event-start-line "$EVENT_START" \
        --prefill-metrics "$DIR/engine_prefill.csv" \
        --decode-metrics "$DIR/engine_decode.csv" --power "$DIR/power.csv" \
        --prefill-uuid "$GPU0_UUID" --decode-uuid "$GPU1_UUID" \
        --traffic-start "$DIR/traffic_start_epoch_s" \
        --traffic-end "$DIR/traffic_end_epoch_s"
    touch "$DIR/.complete"
    echo "COMPLETE $TAG split=$SPLIT plan=$PLAN_KEY sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"sha256\"])' "$PLAN")"
done

echo "DISAGGREGATED_TRANSFER_OK run_root=$RUN_ROOT"
