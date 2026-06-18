#!/bin/bash
# Login-node model staging (no Slurm) — pre-download gated weights to $SCRATCH so
# the campaign jobs run with HF_HUB_OFFLINE=1 (no on-the-fly downloads on the GPU
# node). This is network + disk I/O only (no GPU, light CPU), so it is fine to run
# on a login node; for the ~140GB Llama-70B pull the DTN
# (dtn.sherlock.stanford.edu) or an `sh_dev` shell is gentler on the login node.
#
# Auth: `hf auth login` once, with the SAME HF_HOME this script uses, so the token
# lands where the campaign jobs read it (no token in env or .bashrc):
#     export HF_HOME=$SCRATCH/ptsim/hf
#     hf auth login                      # paste a READ token; accept model licenses on HF first
#     bash profiling/jobs/stage_models.sh [model ...]
# With no args it stages the Tier-1 anchor + all three gemma-4 Tier-2 models.
set -euo pipefail

ROOT="$SCRATCH/ptsim"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NATIVE_VENV="$ROOT/venv-native"
export HF_HOME="$ROOT/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(
        meta-llama/Llama-3.1-70B-Instruct
        google/gemma-4-12B-it
        google/gemma-4-26B-A4B-it
        google/gemma-4-31B-it
    )
fi

# Modern Python + huggingface_hub (the login-node system python3 is too old).
source /etc/profile.d/modules.sh 2>/dev/null || true
ml devel python/3.12.1 2>/dev/null || true
if [ ! -x "$NATIVE_VENV/bin/python" ]; then
    echo "ERROR: native venv missing at $NATIVE_VENV" >&2
    echo "  create it: uv venv $NATIVE_VENV --python 3.12 && \\" >&2
    echo "    source $NATIVE_VENV/bin/activate && uv pip install huggingface_hub hf_transfer transformers" >&2
    exit 1
fi
source "$NATIVE_VENV/bin/activate"
cd "$REPO"

# A token must resolve — from `hf auth login` ($HF_HOME/token) or HF_TOKEN env.
python3 - <<'PY'
import os, sys
from huggingface_hub import get_token
tok = os.environ.get("HF_TOKEN") or get_token()
if not tok:
    sys.exit("No HF token found. Run:\n"
             "  export HF_HOME=$SCRATCH/ptsim/hf && hf auth login")
try:
    from huggingface_hub import whoami
    print("HF token OK (user: %s)" % whoami(tok).get("name"))
except Exception as e:
    print("HF token present; whoami check skipped (%s)" % e)
PY

for MODEL in "${MODELS[@]}"; do
    echo "### staging $MODEL"
    python3 - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(
    sys.argv[1],
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken"],
)
print("cached at:", p)
PY
    # arch_extract uses transformers.AutoConfig; gemma-4 needs transformers>=5.5.0,
    # which lives in the run container, not this native venv — so this is a
    # best-effort sanity check, not a gate. The authoritative parse runs in-container.
    HF_HUB_OFFLINE=1 python3 - "$MODEL" <<'PY' || echo "  (arch_extract sanity skipped — likely needs transformers>=5.5.0; parsed in-container at run time)"
import sys
sys.path.insert(0, "profiling/client")
import arch_extract
a = arch_extract.extract_arch(arch_extract.load_config(sys.argv[1]))
assert a.get("n_layers", 0) > 0, a
print("  arch_extract OK | n_layers", a["n_layers"])
PY
done

echo "### staged sizes"
du -sh "$HF_HOME"/hub/models--* 2>/dev/null || true
echo "ALL_STAGE_OK  (HF_HOME=$HF_HOME)"
