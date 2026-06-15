#!/bin/bash
# Reusable vLLM server lifecycle (extracted from profiling/jobs/llama-3-70b.sh).
# Source this file, then use start_server / stop_server. The existing per-model
# job scripts are left untouched.

# start_server "<full vllm serve command>"
start_server() {
    local serve_cmd="$1"
    local log="${SERVER_LOG:-server.log}"
    setsid bash -c "$serve_cmd" > "$log" 2>&1 &
    SERVING_PID=$!
    SERVING_PGID=$(ps -o pgid= -p "$SERVING_PID" | tr -d ' ')
    echo "Launched server (PID=$SERVING_PID PGID=$SERVING_PGID); log -> $log"
    local tries=0
    while ! curl -s -f http://localhost:8000/health &> /dev/null; do
        sleep 10
        tries=$((tries + 1))
        if [ "$tries" -gt 180 ]; then
            echo "ERROR: server did not become healthy in ~30 min" >&2
            stop_server
            return 1
        fi
    done
    echo "Server ready."
}

stop_server() {
    if [ -n "${SERVING_PGID:-}" ]; then
        echo "Shutting down server (PGID=$SERVING_PGID)..."
        kill -TERM -- "-$SERVING_PGID" 2>/dev/null || true
        sleep 5
        kill -KILL -- "-$SERVING_PGID" 2>/dev/null || true
    fi
    SERVING_PGID=""
}
