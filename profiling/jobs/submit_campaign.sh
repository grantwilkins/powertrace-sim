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
read -r N HARDWARE MODEL SANDBOX_NAME CTYPE TRACE_PLAN DATASET < <(
    python3 -c '
import json, sys
c = json.load(open(sys.argv[1]))
tps = [int(c["server"]["tp"])] + [int(x) for x in c.get("tp_pair", [])]
print(
    max(tps), c["hardware"], c["model"],
    c.get("container") or "vllm-openai-v0.10.1.1.sandbox",
    c["campaign_type"], c.get("trace", {}).get("plan", "-"),
    c.get("workload", {}).get("dataset", "-"),
)
' "$CAMPAIGN_ABS"
)
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
    case "$TRACE_PLAN" in
        /*) ;;
        *) TRACE_PLAN="$REPO_ROOT/$TRACE_PLAN";;
    esac
    test -s "$TRACE_PLAN" || {
        echo "MISSING trace plan: $TRACE_PLAN" >&2; exit 1;
    }
    python3 -c '
import json, sys
plan = json.load(open(sys.argv[1]))
campaign = json.load(open(sys.argv[2]))
limit = int(campaign["server"]["max_model_len"])
rounds = plan.get("rounds", [])
if not rounds:
    raise SystemExit("trace plan has no rounds")
peak = max(
    int(row["prefix_tokens"]) + int(row["input_tokens"]) + int(row["output_tokens"])
    for row in rounds
)
if peak > limit:
    raise SystemExit(f"trace plan peak {peak} exceeds max_model_len {limit}")
' "$TRACE_PLAN" "$CAMPAIGN_ABS"
fi

echo "Submitting $(basename "$CAMPAIGN_ABS") on -p $PART --gres=gpu:$N${CONS:+ -C $CONS}${REQUEUE:+ --requeue}${TIME:+ --time $TIME}"
set -x
# ${VAR:+...} keeps each flag optional without empty-array expansion (fails under
# `set -u` on Sherlock's bash); none of these values word-split.
exec sbatch -p "$PART" --gres=gpu:"$N" \
    ${CONS:+--constraint "$CONS"} ${REQUEUE:+--requeue} ${TIME:+--time "$TIME"} \
    --export=ALL,CAMPAIGN="$CAMPAIGN_ABS",POWERTRACE_REPO="$REPO_ROOT" \
    "$SCRIPT_DIR/campaign.sbatch" "$CAMPAIGN_ABS"
