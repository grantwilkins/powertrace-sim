#!/bin/bash
set -euo pipefail

MODEL="${1:?usage: submit_moe_routing.sh MODEL GPUS INPUT_JSONL OUTPUT_NAME}"
GPUS="${2:?usage: submit_moe_routing.sh MODEL GPUS INPUT_JSONL OUTPUT_NAME}"
INPUT_JSONL="${3:?usage: submit_moe_routing.sh MODEL GPUS INPUT_JSONL OUTPUT_NAME}"
OUTPUT_NAME="${4:?usage: submit_moe_routing.sh MODEL GPUS INPUT_JSONL OUTPUT_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$GPUS" in
    1|2|4|8) ;;
    *) echo "GPUS must be one of 1,2,4,8" >&2; exit 1;;
esac
case "$OUTPUT_NAME" in
    *.npz) ;;
    *) echo "OUTPUT_NAME must end in .npz" >&2; exit 1;;
esac

exec sbatch --gres=gpu:"$GPUS" \
    --export=ALL,MODEL="$MODEL",INPUT_JSONL="$INPUT_JSONL",OUTPUT_NAME="$OUTPUT_NAME" \
    "$SCRIPT_DIR/moe_routing.sbatch"
