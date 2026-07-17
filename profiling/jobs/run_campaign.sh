#!/bin/bash
# Thin campaign orchestrator (CAMPAIGN.md §5-F).
#
#   bash profiling/jobs/run_campaign.sh <campaign.json>            # DRY RUN (default)
#   bash profiling/jobs/run_campaign.sh <campaign.json> --execute  # launch on GPUs
#
# Dry run prints the full server+probe plan and writes a sample synthetic bundle
# under data/dry-runs/<campaign_id>/ WITHOUT launching a server. Execute mode
# reuses server_lifecycle.sh per TP.
#
# Container / Sherlock: the server and probe processes are wrapped with $APP
# (e.g. "apptainer exec --nv --bind $SCRATCH <sandbox>") so modern vLLM runs
# inside the image; campaign_config itself is pure Python and runs natively.
# Env knobs (all optional; defaults preserve the local, no-container behaviour):
#   APP    prefix for in-container exec (default empty -> run on host PATH)
#   PYBIN  interpreter for campaign_config and sample bundles (default "uv run python"; the sbatch
#          sets "python3" after `ml devel python/3.12.1`)
#   RUNS   live bundle parent root (default data/runs; live runs use $RUNS/<campaign_id>)
#   DRY_RUNS sample-bundle parent root (default data/dry-runs)
#   LOGS   server-log dir (default $RUNS/logs)
# Checkpoint/restart: each (TP, probe) — and each (TP, regime) for validate/
# agentic — writes a marker under $RUNS/.done once its bundle is complete. A
# resubmitted job SKIPs any step whose marker exists, so resume = just resubmit.
set -euo pipefail

CAMPAIGN="${1:?usage: run_campaign.sh <campaign.json> [--execute]}"
shift || true
EXECUTE=false
for arg in "$@"; do [ "$arg" = "--execute" ] && EXECUTE=true; done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

APP="${APP:-}"
PY="${PYBIN:-uv run python}"
BASE_RUNS="${RUNS:-data/runs}"
# Do NOT name this "CC": that is the C-compiler env var. `ml devel` exports CC, a
# plain reassignment keeps the export, apptainer forwards it into the container, and
# torch.compile (vLLM) then execs our string as the compiler -> FileNotFoundError.
# Also scrub any inherited CC/CXX so the container's torch inductor uses its own
# toolchain rather than a host compiler path that doesn't exist inside the image.
unset CC CXX 2>/dev/null || true
CCFG="$PY -m profiling.jobs.campaign_config"
CAMP_ID="$(basename "$CAMPAIGN" .json)"
CTYPE="$($CCFG "$CAMPAIGN" --emit type)"
ROLE="$($CCFG "$CAMPAIGN" --emit role)"
if [ "$EXECUTE" = true ] && [ "$ROLE" = "sealed" ]; then
    BASE_RUNS="${SEALED_RUNS:?sealed campaigns require SEALED_RUNS on unmounted/restricted storage}"
fi
RUNS="${BASE_RUNS%/}/$CAMP_ID"
LOGS="${LOGS:-$RUNS/logs}"
export RUNS          # campaign_config.out_root() reads $RUNS to place bundles

echo "=== Campaign plan ==="
$CCFG "$CAMPAIGN" --emit plan

if [ "$EXECUTE" = false ]; then
    echo
    echo "=== DRY RUN (no server launched). Writing sample bundle... ==="
    $PY "$SCRIPT_DIR/_sample_bundle.py" "$CAMPAIGN"
    echo "Re-run with --execute to launch on GPUs."
    exit 0
fi

# ----------------------------- live execution ----------------------------- #
source "$SCRIPT_DIR/server_lifecycle.sh"

# A max-TP allocation may run a smaller TP-pair leg. Pin vLLM to the first TP
# nvidia-smi UUIDs, record that exact set in the bundle, and let RunRecord verify
# that its first-TP power sum is the same device set.
configure_active_gpus() {  # <tp>
    local tp="$1" count
    POWERTRACE_ACTIVE_GPU_UUIDS="$(
        nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
        | sort -t, -k1,1n | head -n "$tp" | cut -d, -f2 | tr -d ' ' | paste -sd, -
    )"
    count="$(awk -F, '{print NF}' <<< "$POWERTRACE_ACTIVE_GPU_UUIDS")"
    if [ -z "$POWERTRACE_ACTIVE_GPU_UUIDS" ] || [ "$count" -ne "$tp" ]; then
        echo "ERROR: could not select exactly $tp active GPU UUIDs" >&2
        return 1
    fi
    export POWERTRACE_ACTIVE_GPU_UUIDS
    echo "TP=$tp active GPUs: $POWERTRACE_ACTIVE_GPU_UUIDS"
}

DONE_DIR="$RUNS/.done"
mkdir -p "$DONE_DIR" "$LOGS"
echo "Checkpoint dir: $DONE_DIR (existing markers are SKIPped)"

# Touch a marker only once the run directory just emitted by the command is complete;
# aborts the job otherwise so a resubmit retries the same step instead of silently
# marking it done.
checkpoint_bundle() {  # <marker> <run_dir>
    local mark="$1" rd="$2"
    if [ -n "$rd" ] && [ -f "$rd/manifest.json" ] && [ -s "$rd/power.csv" ] \
       && [ -s "$rd/engine.csv" ] \
       && awk -F, 'NR > 1 { for (i = 1; i <= NF; i++) if ($i != "") found = 1 } END { exit !found }' "$rd/engine.csv"; then
        touch "$mark"; echo "checkpoint: $(basename "$mark") -> $rd"
    else
        echo "ERROR: incomplete emitted bundle for $(basename "$mark"): $rd" >&2
        return 1
    fi
}

run_bundle_command() {  # <marker> <command>
    local mark="$1" cmd="$2" log status rd
    log="$(mktemp "${TMPDIR:-/tmp}/ptsim-run.XXXXXX")"
    set +e
    ( cd "$REPO_ROOT" && $APP $cmd ) | tee "$log"
    status=${PIPESTATUS[0]}
    set -e
    if [ "$status" -ne 0 ]; then
        rm -f "$log"
        return "$status"
    fi
    rd="$(awk 'NF { line = $0 } END { print line }' "$log")"
    rm -f "$log"
    checkpoint_bundle "$mark" "$rd"
}

if [ "$CTYPE" = "validate" ] || [ "$CTYPE" = "agentic" ] \
   || [ "$CTYPE" = "trace_replay" ]; then
    # One prefix-cache regime per pass (1 for validate; cache off+on for agentic).
    # The server is relaunched per regime so --enable-prefix-caching always matches
    # the run's --prefix-cache (same regime index -> they can't disagree).
    NREG="$($CCFG "$CAMPAIGN" --emit regimes)"
    for TP in $($CCFG "$CAMPAIGN" --emit tps); do
        configure_active_gpus "$TP"
        for R in $(seq 0 $((NREG - 1))); do
            MARK="$DONE_DIR/${CAMP_ID}_tp${TP}_r${R}"
            if [ -f "$MARK" ]; then
                echo "### SKIP (checkpointed): $CTYPE TP=$TP regime=$R ###"; continue
            fi
            echo "### $CTYPE TP=$TP regime=$R/$((NREG - 1)) ###"
            SERVER_LOG="$LOGS/server-${CAMP_ID}-tp${TP}-r${R}.log" \
                start_server "$APP env CUDA_VISIBLE_DEVICES=$POWERTRACE_ACTIVE_GPU_UUIDS $($CCFG "$CAMPAIGN" --emit serve --tp "$TP" --regime-idx "$R")" || exit 1
            RUNCMD="$($CCFG "$CAMPAIGN" --emit run-cmd --tp "$TP" --regime-idx "$R")"
            RUNCMD="$RUNCMD --active-gpu-uuids $POWERTRACE_ACTIVE_GPU_UUIDS"
            echo "+ $APP $RUNCMD"
            run_bundle_command "$MARK" "$RUNCMD" || { stop_server; exit 1; }
            stop_server
        done
    done
    echo "Campaign complete."
    exit 0
fi

if [ "$CTYPE" = "roofline" ]; then
    for TP in $($CCFG "$CAMPAIGN" --emit tps); do
        configure_active_gpus "$TP"
        echo "### roofline TP=$TP ###"
        mapfile -t SERVES < <($CCFG "$CAMPAIGN" --emit probe-serves --tp "$TP")
        mapfile -t PROBES < <($CCFG "$CAMPAIGN" --emit probes --tp "$TP")
        mapfile -t PNAMES < <($CCFG "$CAMPAIGN" --emit probe-names --tp "$TP")
        for i in "${!PROBES[@]}"; do
            PROBE="${PNAMES[$i]}"
            MARK="$DONE_DIR/${CAMP_ID}_tp${TP}_${PROBE}"
            if [ -f "$MARK" ]; then
                echo "--- SKIP (checkpointed): TP=$TP $PROBE ---"; continue
            fi
            echo "--- roofline probe $((i + 1))/${#PROBES[@]}: $PROBE (TP=$TP) ---"
            SERVER_LOG="$LOGS/server-${CAMP_ID}-tp${TP}-${PROBE}.log" \
                start_server "$APP env CUDA_VISIBLE_DEVICES=$POWERTRACE_ACTIVE_GPU_UUIDS ${SERVES[$i]}" || exit 1
            PROBECMD="${PROBES[$i]} --active-gpu-uuids $POWERTRACE_ACTIVE_GPU_UUIDS"
            echo "+ $APP $PROBECMD"
            run_bundle_command "$MARK" "$PROBECMD" || { stop_server; exit 1; }
            stop_server
        done

        MARK="$DONE_DIR/${CAMP_ID}_tp${TP}_long_agentic"
        if [ -f "$MARK" ]; then
            echo "--- SKIP (checkpointed): TP=$TP long_agentic ---"
        else
            echo "--- roofline long agentic (TP=$TP) ---"
            SERVER_LOG="$LOGS/server-${CAMP_ID}-tp${TP}-long-agentic.log" \
                start_server "$APP env CUDA_VISIBLE_DEVICES=$POWERTRACE_ACTIVE_GPU_UUIDS $($CCFG "$CAMPAIGN" --emit serve --tp "$TP")" || exit 1
            RUNCMD="$($CCFG "$CAMPAIGN" --emit roofline-agentic --tp "$TP")"
            RUNCMD="$RUNCMD --active-gpu-uuids $POWERTRACE_ACTIVE_GPU_UUIDS"
            echo "+ $APP $RUNCMD"
            run_bundle_command "$MARK" "$RUNCMD" || { stop_server; exit 1; }
            stop_server
        fi

        MARK="$DONE_DIR/${CAMP_ID}_tp${TP}_analyze"
        if [ -f "$MARK" ]; then
            echo "--- SKIP (checkpointed): TP=$TP analyze ---"
        else
            ANALYZE_CMD="$($CCFG "$CAMPAIGN" --emit analyze-cmd --tp "$TP")"
            echo "+ $APP $ANALYZE_CMD"
            ( cd "$REPO_ROOT" && $APP $ANALYZE_CMD )
            touch "$MARK"
        fi
    done
    echo "Campaign complete."
    exit 0
fi

for TP in $($CCFG "$CAMPAIGN" --emit tps); do
    configure_active_gpus "$TP"
    echo "### TP=$TP ###"
    # One server per probe: probes need different launch flags (e.g. prefill
    # staircase requires chunked-prefill OFF, context holds a long max-model-len).
    mapfile -t SERVES < <($CCFG "$CAMPAIGN" --emit probe-serves --tp "$TP")
    mapfile -t PROBES < <($CCFG "$CAMPAIGN" --emit probes --tp "$TP")
    mapfile -t PNAMES < <($CCFG "$CAMPAIGN" --emit probe-names --tp "$TP")
    for i in "${!PROBES[@]}"; do
        PROBE="${PNAMES[$i]}"
        MARK="$DONE_DIR/${CAMP_ID}_tp${TP}_${PROBE}"
        if [ -f "$MARK" ]; then
            echo "--- SKIP (checkpointed): TP=$TP $PROBE ---"; continue
        fi
        echo "--- probe $((i + 1))/${#PROBES[@]}: $PROBE (TP=$TP) ---"
        SERVER_LOG="$LOGS/server-${CAMP_ID}-tp${TP}-${PROBE}.log" \
            start_server "$APP env CUDA_VISIBLE_DEVICES=$POWERTRACE_ACTIVE_GPU_UUIDS ${SERVES[$i]}" || exit 1
        PROBECMD="${PROBES[$i]} --active-gpu-uuids $POWERTRACE_ACTIVE_GPU_UUIDS"
        echo "+ $APP $PROBECMD"
        run_bundle_command "$MARK" "$PROBECMD" || { stop_server; exit 1; }
        stop_server
    done
done
echo "Campaign complete."
