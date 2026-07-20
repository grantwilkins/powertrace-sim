#!/bin/bash
# Reusable vLLM server lifecycle (extracted from profiling/jobs/llama-3-70b.sh).
# Source this file, then use start_server / stop_server. The existing per-model
# job scripts are left untouched.

# start_server "<full vllm serve command>"
start_server() {
    local serve_cmd="$1"
    local log="${SERVER_LOG:-server.log}"
    local model="${SERVER_MODEL:-}"
    local base_url="${POWERTRACE_BASE_URL:-http://localhost:${POWERTRACE_PORT:-8000}}"
    POWERTRACE_SERVER_LAUNCH_EPOCH_S="$(date +%s)"
    export POWERTRACE_SERVER_LAUNCH_EPOCH_S
    setsid bash -c "$serve_cmd" > "$log" 2>&1 &
    SERVING_PID=$!
    SERVING_PGID=""
    local parent_pgid
    parent_pgid=$(ps -o pgid= -p $$ | tr -d ' ')
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        SERVING_PGID=$(ps -o pgid= -p "$SERVING_PID" 2>/dev/null | tr -d ' ' || true)
        [ -z "$SERVING_PGID" ] && break
        [ "$SERVING_PGID" != "$parent_pgid" ] && break
        sleep 0.2
    done
    if [ -z "$SERVING_PGID" ] || [ "$SERVING_PGID" = "$parent_pgid" ]; then
        echo "WARNING: server PGID was not isolated from parent PGID $parent_pgid; will stop PID tree only" >&2
        SERVING_PGID=""
    fi
    echo "Launched server (PID=$SERVING_PID${SERVING_PGID:+ PGID=$SERVING_PGID}); log -> $log"
    local tries=0
    while ! curl -s -f "$base_url/health" &> /dev/null; do
        sleep 10
        tries=$((tries + 1))
        if [ "$tries" -gt 180 ]; then
            echo "ERROR: server did not become healthy in ~30 min" >&2
            stop_server
            return 1
        fi
    done
    if [ -n "$model" ]; then
        local payload
        payload="{\"model\":\"$model\",\"prompt\":\"ready\",\"max_tokens\":1,\"temperature\":0}"
        while ! curl -s -f "$base_url/v1/models" | grep -F "\"id\":\"$model\"" &> /dev/null \
              && ! curl -s -f "$base_url/v1/models" | grep -F "\"id\": \"$model\"" &> /dev/null; do
            sleep 10
            tries=$((tries + 1))
            if [ "$tries" -gt 180 ]; then
                echo "ERROR: model $model did not appear in /v1/models in ~30 min" >&2
                stop_server
                return 1
            fi
        done
        while ! curl -s -f --max-time 300 \
              -H 'Content-Type: application/json' \
              -d "$payload" \
              "$base_url/v1/completions" &> /dev/null; do
            sleep 10
            tries=$((tries + 1))
            if [ "$tries" -gt 180 ]; then
                echo "ERROR: model $model did not accept a completion probe in ~30 min" >&2
                stop_server
                return 1
            fi
        done
        POWERTRACE_SERVER_READY_EPOCH_S="$(date +%s)"
        export POWERTRACE_SERVER_READY_EPOCH_S
        echo "Server ready for model $model."
    else
        POWERTRACE_SERVER_READY_EPOCH_S="$(date +%s)"
        export POWERTRACE_SERVER_READY_EPOCH_S
        echo "Server ready."
    fi
}

stop_server() {
    local parent_pgid
    parent_pgid=$(ps -o pgid= -p $$ | tr -d ' ')
    if [ -n "${SERVING_PGID:-}" ] && [ "$SERVING_PGID" != "$parent_pgid" ]; then
        echo "Shutting down server (PGID=$SERVING_PGID)..."
        kill -TERM -- "-$SERVING_PGID" 2>/dev/null || true
        sleep 5
        kill -KILL -- "-$SERVING_PGID" 2>/dev/null || true
    elif [ -n "${SERVING_PID:-}" ]; then
        echo "Shutting down server (PID=$SERVING_PID)..."
        pkill -TERM -P "$SERVING_PID" 2>/dev/null || true
        kill -TERM "$SERVING_PID" 2>/dev/null || true
        sleep 5
        pkill -KILL -P "$SERVING_PID" 2>/dev/null || true
        kill -KILL "$SERVING_PID" 2>/dev/null || true
    fi
    SERVING_PID=""
    SERVING_PGID=""
}
