#!/bin/bash
# Thin campaign orchestrator (CAMPAIGN.md §5-F).
#
#   bash profiling/jobs/run_campaign.sh <campaign.json>            # DRY RUN (default)
#   bash profiling/jobs/run_campaign.sh <campaign.json> --execute  # launch on GPUs
#
# Dry run prints the full server+probe plan and writes a sample synthetic bundle
# WITHOUT launching a server. Execute mode reuses server_lifecycle.sh per TP.
set -euo pipefail

CAMPAIGN="${1:?usage: run_campaign.sh <campaign.json> [--execute]}"
shift || true
EXECUTE=false
for arg in "$@"; do [ "$arg" = "--execute" ] && EXECUTE=true; done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CC="uv run python -m profiling.jobs.campaign_config"

echo "=== Campaign plan ==="
$CC "$CAMPAIGN" --emit plan

if [ "$EXECUTE" = false ]; then
    echo
    echo "=== DRY RUN (no server launched). Writing sample bundle... ==="
    uv run python "$SCRIPT_DIR/_sample_bundle.py" "$CAMPAIGN"
    echo "Re-run with --execute to launch on GPUs."
    exit 0
fi

# ----------------------------- live execution ----------------------------- #
source "$SCRIPT_DIR/server_lifecycle.sh"

# validate / agentic campaigns: one server per TP, then their own entrypoint
# (which carries the campaign's max_model_len, so length pruning auto-tracks the
# served context for any model size).
CTYPE="$($CC "$CAMPAIGN" --emit type)"
if [ "$CTYPE" = "validate" ] || [ "$CTYPE" = "agentic" ]; then
    for TP in $($CC "$CAMPAIGN" --emit tps); do
        echo "### $CTYPE TP=$TP ###"
        SERVER_LOG="$REPO_ROOT/server-tp${TP}.log" \
            start_server "$($CC "$CAMPAIGN" --emit serve --tp "$TP")" || exit 1
        RUNCMD="$($CC "$CAMPAIGN" --emit run-cmd --tp "$TP")"
        echo "+ uv run $RUNCMD"
        ( cd "$REPO_ROOT" && uv run $RUNCMD )
        stop_server
    done
    echo "Campaign complete."
    exit 0
fi

for TP in $($CC "$CAMPAIGN" --emit tps); do
    echo "### TP=$TP ###"
    # One server per probe: probes need different launch flags (e.g. prefill
    # staircase requires chunked-prefill OFF, context holds a long max-model-len).
    mapfile -t SERVES < <($CC "$CAMPAIGN" --emit probe-serves --tp "$TP")
    mapfile -t PROBES < <($CC "$CAMPAIGN" --emit probes --tp "$TP")
    for i in "${!PROBES[@]}"; do
        echo "--- probe $((i + 1))/${#PROBES[@]} (TP=$TP) ---"
        SERVER_LOG="$REPO_ROOT/server-tp${TP}-p${i}.log" start_server "${SERVES[$i]}" || exit 1
        echo "+ uv run ${PROBES[$i]}"
        ( cd "$REPO_ROOT" && uv run ${PROBES[$i]} )
        stop_server
    done
done
echo "Campaign complete."
