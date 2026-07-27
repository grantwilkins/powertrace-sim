#!/bin/bash
# Prepare, review, or submit the minimal A100 transfer-CI campaign.
set -euo pipefail

MODE="${1:---dry-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CAMPAIGNS=(
    profiling/campaigns/transfer_ci_burstgpt_qwen3-8b_a100.json
    profiling/campaigns/transfer_ci_openhands_qwen3-8b_a100.json
    profiling/campaigns/transfer_ci_qwen3-14b_a100.json
    profiling/campaigns/transfer_ci_qwen3-30b-a3b_a100.json
)

case "$MODE" in
    --dry-run)
        echo "18 new regimes; estimate 3.5 serial allocation-hours / 4.5 A100 GPU-hours"
        for campaign in "${CAMPAIGNS[@]}"; do
            uv run python -m profiling.jobs.campaign_config \
                "$campaign" --emit plan
        done
        ;;
    --freeze-openhands)
        ROOT="${SCRATCH:?Sherlock SCRATCH is required}/ptsim"
        DATA_REVISION="aa8977805b4cefd317001d80ddf1ad52790e9d23"
        OPENHANDS_DIR="$ROOT/data/openhands/$DATA_REVISION"
        CAMPAIGN="${CAMPAIGNS[1]}"
        test -s "$OPENHANDS_DIR/output.jsonl"
        APPTAINERENV_OPENHANDS_DATASET_PATH="$OPENHANDS_DIR/output.jsonl" \
        APPTAINERENV_HF_HOME="$ROOT/hf" \
        APPTAINERENV_HF_HUB_OFFLINE=1 \
            apptainer exec \
            --bind "$SCRATCH" --bind "${GROUP_HOME:?Sherlock GROUP_HOME is required}" \
            --pwd "$REPO_ROOT" "$ROOT/vllm-openai-v0.10.1.1.sandbox" \
            python3 profiling/agentic_traces/openhands_preflight.py \
            "$CAMPAIGN" --freeze-hashes
        ;;
    --submit)
        test -s data/trace_plans/burstgpt_transfer_ci_0.json
        uv run python -c '
import json
from pathlib import Path
p = Path("profiling/campaigns/transfer_ci_openhands_qwen3-8b_a100.json")
values = json.loads(p.read_text())["sessions"]["expected_plan_sha256"]
if any(value == "0" * 64 for value in values):
    raise SystemExit("freeze OpenHands hashes before submission")
'
        for campaign in "${CAMPAIGNS[@]}"; do
            bash "$SCRIPT_DIR/submit_campaign.sh" "$campaign"
        done
        ;;
    *)
        echo "usage: $0 [--dry-run|--freeze-openhands|--submit]" >&2
        exit 2
        ;;
esac
