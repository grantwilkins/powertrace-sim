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
CONS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --time) TIME="$2"; shift 2;;
        -p|--partition) PART="$2"; shift 2;;
        -C|--constraint) CONS="$2"; shift 2;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

# owners is mixed-GPU and preemptible: pin the 80GB A100 SKU (so we never land on a
# smaller/other GPU our configs aren't sized for) and allow requeue — the per-(TP,probe)
# done-markers make a requeued job resume instead of restart. `-p ramr,owners` lets
# Slurm place the job wherever an A100 frees first. ramr-only needs no constraint.
REQUEUE=""
case ",$PART," in
    *,owners,*) [ -n "$CONS" ] || CONS="GPU_SKU:A100_SXM4"; REQUEUE=1;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CAMPAIGN_ABS="$(readlink -f "$CAMPAIGN")"
test -f "$CAMPAIGN_ABS" || { echo "no such campaign: $CAMPAIGN" >&2; exit 1; }

# Max TP degree -> GPU count for the single node (tp_pair second leg uses fewer).
# Use the python module directly: `uv run` here tries to sync the heavy project env
# (torch has no glibc-2.17 wheel on Sherlock) and fails. campaign_config is pure stdlib.
source /etc/profile.d/modules.sh 2>/dev/null || true
ml devel python/3.12.1 2>/dev/null || true
N="$(cd "$REPO_ROOT" && python3 -m profiling.jobs.campaign_config \
        "$CAMPAIGN_ABS" --emit tps | sort -n | tail -1)"
[ -n "$N" ] || { echo "could not determine TP for $CAMPAIGN" >&2; exit 1; }

echo "Submitting $(basename "$CAMPAIGN_ABS") on -p $PART --gres=gpu:$N${CONS:+ -C $CONS}${REQUEUE:+ --requeue}${TIME:+ --time $TIME}"
set -x
# ${VAR:+...} keeps each flag optional without empty-array expansion (fails under
# `set -u` on Sherlock's bash); none of these values word-split.
exec sbatch -p "$PART" --gres=gpu:"$N" \
    ${CONS:+--constraint "$CONS"} ${REQUEUE:+--requeue} ${TIME:+--time "$TIME"} \
    --export=ALL,CAMPAIGN="$CAMPAIGN_ABS" \
    "$SCRIPT_DIR/campaign.sbatch" "$CAMPAIGN_ABS"
