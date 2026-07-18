#!/bin/bash
set -euo pipefail

H100_PARTITION="${H100_PARTITION:?set the Sherlock H100 partition}"
ROOT="${SCRATCH:?Sherlock SCRATCH is required}/ptsim"
GROUP_DATA="${GROUP_HOME:?Sherlock GROUP_HOME is required}/gfw"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROUTING_INPUT="${ROUTING_INPUT:-$GROUP_DATA/moe_routing_samples.jsonl}"

require_model() {
    local model="$1"
    test -s "$ROOT/hf/hub/models--${model//\//--}/.powertrace-stage-complete" || {
        echo "INCOMPLETE staged model: $model" >&2
        exit 1
    }
}

check_plan() {
    python3 -c '
import json, sys
plan = json.load(open(sys.argv[1]))
limit = int(sys.argv[2])
rounds = plan.get("rounds", [])
if not rounds:
    raise SystemExit("trace plan has no rounds")
peak = max(
    int(row["prefix_tokens"]) + int(row["input_tokens"]) + int(row["output_tokens"])
    for row in rounds
)
if peak > limit:
    raise SystemExit(f"{sys.argv[1]} peak {peak} exceeds {limit}")
' "$1" "$2"
}

test -d "$ROOT/vllm-openai-v0.10.1.1.sandbox"
test -d "$ROOT/vllm-openai-gemma4.sandbox"
test -s "$GROUP_DATA/ShareGPT_V3_unfiltered_cleaned_split.json"
test -s "$ROUTING_INPUT"
test -s "$REPO/data/trace_plans/tracelab_code.json"
test -s "$REPO/data/trace_plans/burstgpt_10min.json" || {
    echo "Generate data/trace_plans/burstgpt_10min.json before submission" >&2
    exit 1
}
check_plan "$REPO/data/trace_plans/tracelab_code.json" 131072
check_plan "$REPO/data/trace_plans/burstgpt_10min.json" 32768
for model in \
    Qwen/Qwen3-8B \
    meta-llama/Llama-3.1-70B-Instruct \
    openai/gpt-oss-20b \
    openai/gpt-oss-120b \
    google/gemma-4-26B-A4B-it
do
    require_model "$model"
done

submit_a100() {
    bash "$REPO/profiling/jobs/submit_campaign.sh" "$REPO/$1" --time "${2:-00:30:00}"
}
submit_h100() {
    bash "$REPO/profiling/jobs/submit_campaign.sh" "$REPO/$1" \
        -p "$H100_PARTITION" --time "${2:-00:30:00}"
}

submit_a100 profiling/campaigns/validate_qwen3-8b_a100.json
submit_h100 profiling/campaigns/validate_qwen3-8b.json
submit_a100 profiling/campaigns/arrival_rate_qwen3-8b_a100_r2p5.json
submit_a100 profiling/campaigns/arrival_pattern_qwen3-8b_a100_bursty.json
submit_a100 profiling/campaigns/arrival_pattern_qwen3-8b_a100_smooth.json
submit_a100 profiling/campaigns/trace_replay_qwen3-8b_a100_cache_off.json 01:30:00
submit_a100 profiling/campaigns/trace_replay_qwen3-8b_a100_cache_on.json 01:30:00
submit_a100 profiling/campaigns/burstgpt_qwen3-8b_a100.json 01:30:00
submit_h100 profiling/campaigns/h100_tp8_state_diagnostic.json 01:00:00
submit_h100 profiling/campaigns/h100_tp4_state_control.json 01:00:00
submit_a100 profiling/campaigns/validate_gemma-4-26b-a4b_moe_transfer_a100.json

bash "$REPO/profiling/jobs/submit_moe_routing.sh" \
    openai/gpt-oss-20b 1 "$ROUTING_INPUT" gpt-oss-20b-routing.npz
bash "$REPO/profiling/jobs/submit_moe_routing.sh" \
    openai/gpt-oss-120b 4 "$ROUTING_INPUT" gpt-oss-120b-routing.npz
bash "$REPO/profiling/jobs/submit_moe_routing.sh" \
    google/gemma-4-26B-A4B-it 1 "$ROUTING_INPUT" gemma-4-26b-a4b-routing.npz \
    vllm-openai-gemma4.sandbox
