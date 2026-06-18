#!/bin/bash
# Thin campaign orchestrator (CAMPAIGN.md §5-F).
#
#   bash profiling/jobs/run_campaign.sh <campaign.json>            # DRY RUN (default)
#   bash profiling/jobs/run_campaign.sh <campaign.json> --execute  # launch on GPUs
#
# Dry run prints the full server+probe plan and writes a sample synthetic bundle
# WITHOUT launching a server. Execute mode reuses server_lifecycle.sh per TP.
#
# Container / Sherlock: the server and probe processes are wrapped with $APP
# (e.g. "apptainer exec --nv --bind $SCRATCH <sandbox>") so modern vLLM runs
# inside the image; campaign_config itself is pure Python and runs natively.
# Env knobs (all optional; defaults preserve the local, no-container behaviour):
#   APP    prefix for in-container exec (default empty -> run on host PATH)
#   PYBIN  interpreter for campaign_config (default "uv run python"; the sbatch
#          sets "python3" after `ml devel python/3.12.1`)
#   RUNS   bundle output root (default data/runs; the sbatch points it at $SCRATCH)
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
RUNS="${RUNS:-data/runs}"
LOGS="${LOGS:-$RUNS/logs}"
export RUNS          # campaign_config.out_root() reads $RUNS to place bundles
# Do NOT name this "CC": that is the C-compiler env var. `ml devel` exports CC, a
# plain reassignment keeps the export, apptainer forwards it into the container, and
# torch.compile (vLLM) then execs our string as the compiler -> FileNotFoundError.
# Also scrub any inherited CC/CXX so the container's torch inductor uses its own
# toolchain rather than a host compiler path that doesn't exist inside the image.
unset CC CXX 2>/dev/null || true
CCFG="${PYBIN:-uv run python} -m profiling.jobs.campaign_config"

echo "=== Campaign plan ==="
$CCFG "$CAMPAIGN" --emit plan

if [ "$EXECUTE" = false ]; then
    echo
    echo "=== DRY RUN (no server launched). Writing sample bundle... ==="
    uv run python "$SCRIPT_DIR/_sample_bundle.py" "$CAMPAIGN"
    echo "Re-run with --execute to launch on GPUs."
    exit 0
fi

# ----------------------------- live execution ----------------------------- #
source "$SCRIPT_DIR/server_lifecycle.sh"

CAMP_ID="$(basename "$CAMPAIGN" .json)"
DONE_DIR="$RUNS/.done"
mkdir -p "$DONE_DIR" "$LOGS"
echo "Checkpoint dir: $DONE_DIR (existing markers are SKIPped)"

# Touch a marker only once a complete bundle exists; aborts the job otherwise so a
# resubmit retries the same step instead of silently marking it done.
checkpoint_bundle() {  # <marker> <run_dir_glob>
    local mark="$1" glob="$2"
    local rd; rd="$(ls -dt $glob 2>/dev/null | head -1 || true)"
    if [ -n "$rd" ] && [ -f "$rd/manifest.json" ] && [ -s "$rd/power.csv" ] \
       && [ -s "$rd/engine.csv" ]; then
        touch "$mark"; echo "checkpoint: $(basename "$mark") -> $rd"
    else
        echo "ERROR: no complete bundle for $(basename "$mark") (glob: $glob)" >&2
        return 1
    fi
}

CTYPE="$($CCFG "$CAMPAIGN" --emit type)"
if [ "$CTYPE" = "validate" ] || [ "$CTYPE" = "agentic" ]; then
    # One prefix-cache regime per pass (1 for validate; cache off+on for agentic).
    # The server is relaunched per regime so --enable-prefix-caching always matches
    # the run's --prefix-cache (same regime index -> they can't disagree).
    NREG="$($CCFG "$CAMPAIGN" --emit regimes)"
    for TP in $($CCFG "$CAMPAIGN" --emit tps); do
        for R in $(seq 0 $((NREG - 1))); do
            MARK="$DONE_DIR/${CAMP_ID}_tp${TP}_r${R}"
            if [ -f "$MARK" ]; then
                echo "### SKIP (checkpointed): $CTYPE TP=$TP regime=$R ###"; continue
            fi
            echo "### $CTYPE TP=$TP regime=$R/$((NREG - 1)) ###"
            SERVER_LOG="$LOGS/server-${CAMP_ID}-tp${TP}-r${R}.log" \
                start_server "$APP $($CCFG "$CAMPAIGN" --emit serve --tp "$TP" --regime-idx "$R")" || exit 1
            RUNCMD="$($CCFG "$CAMPAIGN" --emit run-cmd --tp "$TP" --regime-idx "$R")"
            echo "+ $APP $RUNCMD"
            ( cd "$REPO_ROOT" && $APP $RUNCMD )
            stop_server
            checkpoint_bundle "$MARK" "$RUNS/*_${CTYPE}_tp${TP}_*" || exit 1
        done
    done
    echo "Campaign complete."
    exit 0
fi

for TP in $($CCFG "$CAMPAIGN" --emit tps); do
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
            start_server "$APP ${SERVES[$i]}" || exit 1
        echo "+ $APP ${PROBES[$i]}"
        ( cd "$REPO_ROOT" && $APP ${PROBES[$i]} )
        stop_server
        checkpoint_bundle "$MARK" "$RUNS/*_${PROBE}_tp${TP}_*" || exit 1
    done
done
echo "Campaign complete."
