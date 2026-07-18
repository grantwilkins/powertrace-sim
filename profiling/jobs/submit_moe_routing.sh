#!/bin/bash
set -euo pipefail

USAGE="submit_moe_routing.sh MODEL GPUS INPUT_JSONL OUTPUT_NAME [SANDBOX_NAME]"
MODEL="${1:?usage: $USAGE}"
GPUS="${2:?usage: $USAGE}"
INPUT_JSONL="${3:?usage: $USAGE}"
OUTPUT_NAME="${4:?usage: $USAGE}"
SANDBOX_NAME="${5:-vllm-openai-v0.10.1.1.sandbox}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRATCH:?Sherlock SCRATCH is required}/ptsim"

case "$GPUS" in
    1|2|4|8) ;;
    *) echo "GPUS must be one of 1,2,4,8" >&2; exit 1;;
esac
case "$OUTPUT_NAME" in
    *.npz) ;;
    *) echo "OUTPUT_NAME must end in .npz" >&2; exit 1;;
esac

test -s "$INPUT_JSONL" || { echo "MISSING routing input: $INPUT_JSONL" >&2; exit 1; }
test -d "$ROOT/$SANDBOX_NAME" || {
    echo "MISSING container: $ROOT/$SANDBOX_NAME" >&2; exit 1;
}
test -s "$ROOT/hf/hub/models--${MODEL//\//--}/.powertrace-stage-complete" || {
    echo "INCOMPLETE staged model: $MODEL" >&2; exit 1;
}

exec sbatch --gres=gpu:"$GPUS" \
    --export=ALL,MODEL="$MODEL",INPUT_JSONL="$INPUT_JSONL",OUTPUT_NAME="$OUTPUT_NAME",SANDBOX_NAME="$SANDBOX_NAME" \
    "$SCRIPT_DIR/moe_routing.sbatch"
