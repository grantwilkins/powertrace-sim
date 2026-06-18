#!/bin/bash
# Build a vLLM Apptainer sandbox for a given image tag (campaign "container" field).
# Gemma-4 Tier-2 needs vLLM >=0.19 (transformers >=5.5), a newer image than the
# v0.10.1.1 sandbox the Llama/validate campaigns use. The build is multi-GB and
# CPU/IO-heavy (no GPU): run it inside `sh_dev` or via sbatch, NOT bare on a login
# node. mksquashfs segfaults on these images, so we build a --sandbox dir, not a SIF.
#
#   sh_dev -t 02:00:00          # grab an interactive compute shell first
#   bash profiling/jobs/build_container.sh \
#        [docker://vllm/vllm-openai:gemma4] [vllm-openai-gemma4.sandbox]
set -euo pipefail

IMG="${1:-docker://vllm/vllm-openai:gemma4}"
SANDBOX_NAME="${2:-vllm-openai-gemma4.sandbox}"
ROOT="$SCRATCH/ptsim"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SANDBOX="$ROOT/$SANDBOX_NAME"
export APPTAINER_CACHEDIR="$ROOT/.apptainer"
export APPTAINER_TMPDIR="$ROOT/.apptainer_tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

source /etc/profile.d/modules.sh 2>/dev/null || true
ml devel python/3.12.1 2>/dev/null || true
apptainer --version

echo "### build sandbox  $SANDBOX  from  $IMG"
if [ ! -d "$SANDBOX" ] || ! apptainer inspect "$SANDBOX" >/dev/null 2>&1; then
    rm -rf "$SANDBOX"
    apptainer build --sandbox "$SANDBOX" "$IMG"
fi
apptainer inspect "$SANDBOX" >/dev/null
du -sh "$SANDBOX" 2>/dev/null || true

echo "### vLLM + transformers versions in container"
apptainer exec --bind "$SCRATCH" "$SANDBOX" python3 - <<'PY'
import vllm, transformers, sys
print("vllm", vllm.__version__, "| transformers", transformers.__version__,
      "| py", sys.version.split()[0])
PY

echo "### ensure client deps (pandas, datasets) in the sandbox"
if ! apptainer exec "$SANDBOX" python3 -c "import pandas, datasets" 2>/dev/null; then
    apptainer exec --writable --no-home "$SANDBOX" \
        python3 -m pip install --no-cache-dir pandas datasets
fi

echo "### vendored bench import smoke (the fragile vLLM imports, in-container)"
if apptainer exec --bind "$SCRATCH" "$SANDBOX" \
        python3 "$REPO/profiling/client/benchmark_serving.py" --help >/dev/null; then
    echo "benchmark_serving.py --help OK (in container)"
else
    echo "FAIL: vendored bench imports broke under this vLLM version." >&2
    echo "  -> patch the vLLM imports in profiling/client/{benchmark_serving,benchmark_dataset,backend_request_func}.py" >&2
    exit 1
fi

echo "ALL_BUILD_OK  sandbox=$SANDBOX  ($IMG)"
