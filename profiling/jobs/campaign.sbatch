#!/bin/bash
#SBATCH --job-name=ptsim-campaign
#SBATCH --partition=ramr
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --output=/scratch/users/gfw/ptsim/logs/campaign-%j.out
#SBATCH --error=/scratch/users/gfw/ptsim/logs/campaign-%j.out
#
# Parameterized Tier-1 / Tier-2 (and validate/agentic) campaign launcher.
# Submit via profiling/jobs/submit_campaign.sh, which overrides --gres / --time /
# --partition for the campaign's TP and exports CAMPAIGN. The vLLM server and the
# probe clients run inside the official Apptainer image ($APP); campaign_config and
# the bundle checks run natively. Checkpoint/restart lives in run_campaign.sh:
# resubmit the same line and completed (TP,probe) steps are SKIPped.
set -Eeuo   # no pipefail: diagnostic `| head` pipes would SIGPIPE-kill the script
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

ROOT="$SCRATCH/ptsim"
REPO="$HOME/powertrace-sim"
RUNS="$ROOT/runs"
LOGS="$ROOT/logs"
NATIVE_VENV="$ROOT/venv-native"
GROUP_DATA="${GROUP_HOME:?Sherlock GROUP_HOME is required}/gfw"
CAMPAIGN="${1:-${CAMPAIGN:?set CAMPAIGN env or pass campaign.json as arg}}"
mkdir -p "$RUNS" "$LOGS"

# caches -> scratch (never NFS $HOME), and force them into the container too
export HF_HOME="$ROOT/hf"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME="$ROOT/cache/xdg"
export VLLM_CACHE_ROOT="$ROOT/cache/vllm"
export TRITON_CACHE_DIR="$ROOT/cache/triton"
export OUTLINES_CACHE_DIR="$ROOT/cache/outlines"
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$OUTLINES_CACHE_DIR"
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_HF_DATASETS_OFFLINE=1
export APPTAINERENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"
export APPTAINERENV_VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT"
export APPTAINERENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export APPTAINERENV_OUTLINES_CACHE_DIR="$OUTLINES_CACHE_DIR"
# context-hold probes set max_model_len up to 131072; allow it inside the container.
# (campaign JSON server.extra_env is documentation only; the server inherits this.)
export APPTAINERENV_VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# validate campaigns read ShareGPT from $SHAREGPT_DATASET_PATH (so the config need
# not hardcode the scratch path); harmless for probe/agentic campaigns.
export SHAREGPT_DATASET_PATH="$GROUP_DATA/ShareGPT_V3_unfiltered_cleaned_split.json"
export BURSTGPT_DATASET_PATH="$GROUP_DATA/BurstGPT_without_fails_2.csv"
export APPTAINERENV_SHAREGPT_DATASET_PATH="$SHAREGPT_DATASET_PATH"
export APPTAINERENV_BURSTGPT_DATASET_PATH="$BURSTGPT_DATASET_PATH"

echo "### preflight"
source /etc/profile.d/modules.sh 2>/dev/null || true
ml devel python/3.12.1 2>/dev/null || true
[ -x "$NATIVE_VENV/bin/python" ] && source "$NATIVE_VENV/bin/activate"
cd "$REPO"
python3 --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Per-campaign container (gemma-4 needs the newer vLLM image; others use v0.10.1.1)
# and the served model, parsed together from the campaign JSON.
read -r SANDBOX_NAME MODEL < <(python3 profiling/jobs/campaign_config.py "$CAMPAIGN" --emit json \
    | python3 -c 'import json,sys; c=json.load(sys.stdin); print(c.get("container") or "vllm-openai-v0.10.1.1.sandbox", c["model"])')
CTYPE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["campaign_type"])' "$CAMPAIGN")"
SANDBOX="$ROOT/$SANDBOX_NAME"
APP="apptainer exec --nv --bind $SCRATCH --bind $GROUP_HOME $SANDBOX"
echo "container: $SANDBOX"
test -d "$SANDBOX" || { echo "MISSING $SANDBOX; build it: bash profiling/jobs/build_container.sh"; exit 1; }

# weights must be pre-staged (HF_HUB_OFFLINE=1 -> no on-the-fly download)
CACHE_DIR="$HF_HOME/hub/models--$(echo "$MODEL" | sed 's#/#--#g')"
test -d "$CACHE_DIR" || {
    echo "MISSING weights for $MODEL"
    echo "  expected: $CACHE_DIR"
    echo "  -> bash profiling/jobs/stage_models.sh   (login node; after 'hf auth login')"
    exit 1; }
test -s "$CACHE_DIR/.powertrace-stage-complete" || {
    echo "INCOMPLETE staged model: $MODEL"
    echo "  rerun: bash profiling/jobs/stage_models.sh $MODEL"
    exit 1; }
echo "weights present: $CACHE_DIR"

if [ "$CTYPE" = "validate" ]; then
    test -s "$SHAREGPT_DATASET_PATH" || {
        echo "MISSING ShareGPT data: $SHAREGPT_DATASET_PATH"; exit 1; }
elif [ "$CTYPE" = "trace_replay" ]; then
    TRACE_PLAN="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["trace"]["plan"])' "$CAMPAIGN")"
    case "$TRACE_PLAN" in
        /*) ;;
        *) TRACE_PLAN="$REPO/$TRACE_PLAN";;
    esac
    test -s "$TRACE_PLAN" || { echo "MISSING trace plan: $TRACE_PLAN"; exit 1; }
fi

$APP python3 -c \
    "import aiohttp,datasets,numpy,pandas,PIL,transformers,vllm; print('container vllm', vllm.__version__)"
$APP python3 -c \
    'import sys; from transformers import AutoConfig,AutoTokenizer; AutoConfig.from_pretrained(sys.argv[1], local_files_only=True); AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True); print("offline model metadata OK")' \
    "$MODEL"
if [ "$CTYPE" = "roofline" ]; then
    $APP python3 -c "import matplotlib; print('matplotlib', matplotlib.__version__)"
fi

echo "### run campaign (server+probes in-container; campaign_config native)"
RUNS="$RUNS" LOGS="$LOGS" APP="$APP" PYBIN=python3 \
    bash profiling/jobs/run_campaign.sh "$CAMPAIGN" --execute

echo "### checkpoints written:"
ls -1 "$RUNS/.done/$(basename "$CAMPAIGN" .json)"* 2>/dev/null || true
echo "ALL_CAMPAIGN_OK  campaign=$(basename "$CAMPAIGN")  runs=$RUNS"
