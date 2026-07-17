#!/bin/bash
#SBATCH --job-name=ptsim-moe-routing
#SBATCH --partition=ramr
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --output=/scratch/users/gfw/ptsim/logs/moe-routing-%j.out
#SBATCH --error=/scratch/users/gfw/ptsim/logs/moe-routing-%j.out
set -Eeuo pipefail

ROOT="$SCRATCH/ptsim"
REPO="$HOME/powertrace-sim"
MODEL="${MODEL:?MODEL is required}"
INPUT_JSONL="${INPUT_JSONL:?INPUT_JSONL is required}"
OUTPUT_NAME="${OUTPUT_NAME:?OUTPUT_NAME is required}"
SANDBOX="$ROOT/${SANDBOX_NAME:-vllm-openai-v0.10.1.1.sandbox}"
OUTPUT="$ROOT/moe-routing/$OUTPUT_NAME"

mkdir -p "$ROOT/logs" "$ROOT/moe-routing"
export HF_HOME="$ROOT/hf"
export HF_HUB_OFFLINE=1
cd "$REPO"

test -f "$INPUT_JSONL" || { echo "MISSING routing input: $INPUT_JSONL"; exit 1; }
test -d "$SANDBOX" || { echo "MISSING container: $SANDBOX"; exit 1; }
test -d "$HF_HOME/hub/models--${MODEL//\//--}" || {
    echo "MISSING staged model: $MODEL"; exit 1;
}

APP=(apptainer exec --nv --bind "$SCRATCH" --bind "$GROUP_HOME" "$SANDBOX")
"${APP[@]}" python3 -c \
    "import accelerate, transformers; print(transformers.__version__)"
"${APP[@]}" python3 profiling/moe_routing/router_capture.py \
    --model "$MODEL" \
    --input-jsonl "$INPUT_JSONL" \
    --output-npz "$OUTPUT"

test -s "$OUTPUT"
test -s "${OUTPUT%.npz}.manifest.json"
echo "ALL_MOE_ROUTING_OK output=$OUTPUT"
