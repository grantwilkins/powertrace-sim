#!/bin/bash
# Submit a campaign on the right number of GPUs for its TP (CAMPAIGN.md §5-F).
#
#   bash profiling/jobs/submit_campaign.sh <campaign.json> [--time HH:MM:SS] [-p PART]
#
# Computes --gres=gpu:N from the campaign's max TP degree (tp=4 Tier-1 -> 4 GPUs;
# tp=1 Tier-2 -> 1 GPU) and submits profiling/jobs/campaign.sbatch on -p ramr
# (the group's reserved 4x A100-80GB node) by default. Resume after a failure is
# just re-running this same line: campaign.sbatch -> run_campaign.sh SKIPs the
# (TP,probe) steps already checkpointed under $SCRATCH/ptsim/runs/.done.
set -euo pipefail

CAMPAIGN="${1:?usage: submit_campaign.sh <campaign.json> [--time HH:MM:SS] [-p PART]}"
shift || true
TIME=""
PART="ramr"
PART_EXPLICIT=false
CONS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --time) TIME="$2"; shift 2;;
        -p|--partition) PART="$2"; PART_EXPLICIT=true; shift 2;;
        -C|--constraint) CONS="$2"; shift 2;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

REQUEUE=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CAMPAIGN_ABS="$(readlink -f "$CAMPAIGN")"
test -f "$CAMPAIGN_ABS" || { echo "no such campaign: $CAMPAIGN" >&2; exit 1; }

# Max TP degree -> GPU count for the single node (tp_pair second leg uses fewer).
# Parse only the submission fields with stdlib json. Importing campaign_config here
# also imports the scientific NumPy stack, which is not guaranteed on a login node.
source /etc/profile.d/modules.sh 2>/dev/null || true
ml devel python/3.12.1 2>/dev/null || true
read -r N HARDWARE MODEL SANDBOX_NAME CTYPE TRACE_PLAN DATASET CORPUS DATASET_REVISION PACK_COUNT DEFAULT_TIME ROLE < <(
    python3 -c '
import json, sys
c = json.load(open(sys.argv[1]))
tps = [int(c["server"]["tp"])] + [int(x) for x in c.get("tp_pair", [])]
print(
    max(tps), c["hardware"], c["model"],
    c.get("container") or "vllm-openai-v0.10.1.1.sandbox",
    c["campaign_type"], c.get("trace", {}).get("plan", "-"),
    c.get("workload", {}).get("dataset", "-"),
    c.get("sessions", {}).get("corpus", "-"),
    c.get("sessions", {}).get("dataset_revision", "-"),
    c.get("sessions", {}).get("pack_count", 1),
    c.get("slurm_time", "-"),
    c.get("validation_role", "development"),
)
' "$CAMPAIGN_ABS"
)
[ -n "$TIME" ] || [ "$DEFAULT_TIME" = "-" ] || TIME="$DEFAULT_TIME"
[ -n "$N" ] || { echo "could not determine TP for $CAMPAIGN" >&2; exit 1; }
# owners is mixed-GPU and preemptible. Pin the GPU class and memory size so
# large-model campaigns do not land on another accelerator or a 40GB A100.
case ",$PART," in
    *,owners,*)
        if [ -z "$CONS" ]; then
            case "$HARDWARE" in
                A100) CONS="GPU_SKU:A100_SXM4&GPU_MEM:80GB";;
                H100) CONS="GPU_SKU:H100_SXM5&GPU_MEM:80GB";;
            esac
        fi
        REQUEUE=1
        ;;
esac
if [ "$HARDWARE" = "H100" ]; then
    [ "$PART_EXPLICIT" = true ] || {
        echo "H100 campaign requires an explicit -p <H100_PARTITION>" >&2; exit 1; }
    case ",$PART," in
        *,ramr,*) echo "H100 campaign cannot use the A100 ramr partition" >&2; exit 1;;
    esac
fi

ROOT="${SCRATCH:?Sherlock SCRATCH is required}/ptsim"
GROUP_DATA="${GROUP_HOME:?Sherlock GROUP_HOME is required}/gfw"
SEALED_RUNS="$ROOT/sealed-runs"
if [ "$ROLE" = "sealed" ]; then
    mkdir -p "$SEALED_RUNS"
    chmod 700 "$SEALED_RUNS"
    [ "$(stat -c %a "$SEALED_RUNS")" = "700" ] || {
        echo "SEALED_RUNS must have mode 700: $SEALED_RUNS" >&2; exit 1;
    }
fi
test -d "$ROOT/$SANDBOX_NAME" || {
    echo "MISSING container: $ROOT/$SANDBOX_NAME" >&2; exit 1;
}
CACHE_DIR="$ROOT/hf/hub/models--${MODEL//\//--}"
test -s "$CACHE_DIR/.powertrace-stage-complete" || {
    echo "INCOMPLETE staged model: $MODEL" >&2
    echo "  run: bash profiling/jobs/stage_models.sh $MODEL" >&2
    exit 1
}
if [ "$CTYPE" = "validate" ]; then
    case "$DATASET" in
        sharegpt)
            test -s "$GROUP_DATA/ShareGPT_V3_unfiltered_cleaned_split.json" || {
                echo "MISSING ShareGPT data under $GROUP_DATA" >&2; exit 1; };;
        burstgpt)
            test -s "$GROUP_DATA/BurstGPT_without_fails_2.csv" || {
                echo "MISSING BurstGPT data under $GROUP_DATA" >&2; exit 1; };;
    esac
elif [ "$CTYPE" = "trace_replay" ]; then
    python3 -c '
import json, sys
from pathlib import Path
campaign = json.load(open(sys.argv[1]))
limit = int(campaign["server"]["max_model_len"])
plans = {
    row.get("plan", campaign["trace"]["plan"])
    for row in campaign["trace"]["regimes"]
}
for value in sorted(plans):
    path = Path(value)
    if not path.is_absolute():
        path = Path(sys.argv[2]) / path
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"MISSING trace plan: {path}")
    plan = json.loads(path.read_text())
    rounds = plan.get("rounds", [])
    if not rounds:
        raise SystemExit(f"trace plan has no rounds: {path}")
    peak = max(
        int(row["prefix_tokens"]) + int(row["input_tokens"])
        + int(row["output_tokens"])
        for row in rounds
    )
    if peak > limit:
        raise SystemExit(f"{path} peak {peak} exceeds max_model_len {limit}")
' "$CAMPAIGN_ABS" "$REPO_ROOT"
elif [ "$CTYPE" = "agentic" ] && [ "$CORPUS" = "openhands" ]; then
    OPENHANDS_DIR="$ROOT/data/openhands/$DATASET_REVISION"
    test -s "$OPENHANDS_DIR/output.jsonl" \
        && test -s "$OPENHANDS_DIR/source.json" || {
        echo "MISSING pinned OpenHands data for $DATASET_REVISION" >&2
        echo "  run: bash profiling/jobs/stage_openhands.sh $DATASET_REVISION" >&2
        exit 1
    }
    for PACK_INDEX in $(seq 0 $((PACK_COUNT - 1))); do
        APPTAINERENV_OPENHANDS_DATASET_PATH="$OPENHANDS_DIR/output.jsonl" \
        APPTAINERENV_HF_HOME="$ROOT/hf" \
        APPTAINERENV_HF_HUB_OFFLINE=1 \
            apptainer exec \
            --bind "$SCRATCH" --bind "$GROUP_HOME" --pwd "$REPO_ROOT" \
            "$ROOT/$SANDBOX_NAME" \
            python3 profiling/agentic_traces/openhands_preflight.py \
            "$CAMPAIGN_ABS" --pack-index "$PACK_INDEX"
    done
fi

echo "Submitting $(basename "$CAMPAIGN_ABS") on -p $PART --gres=gpu:$N${CONS:+ -C $CONS}${REQUEUE:+ --requeue}${TIME:+ --time $TIME}"
EXPORTS="ALL,CAMPAIGN=$CAMPAIGN_ABS,POWERTRACE_REPO=$REPO_ROOT"
[ "$ROLE" != "sealed" ] || EXPORTS="$EXPORTS,SEALED_RUNS=$SEALED_RUNS"
set -x
# ${VAR:+...} keeps each flag optional without empty-array expansion (fails under
# `set -u` on Sherlock's bash); none of these values word-split.
exec sbatch -p "$PART" --gres=gpu:"$N" \
    ${CONS:+--constraint "$CONS"} ${REQUEUE:+--requeue} ${TIME:+--time "$TIME"} \
    --export="$EXPORTS" \
    "$SCRIPT_DIR/campaign.sbatch" "$CAMPAIGN_ABS"
