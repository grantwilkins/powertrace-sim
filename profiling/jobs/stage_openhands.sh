#!/bin/bash
# Stage the pinned OpenHands evaluation JSONL for offline GPU jobs.
set -euo pipefail

ROOT="${SCRATCH:?Sherlock SCRATCH is required}/ptsim"
REVISION="${1:-aa8977805b4cefd317001d80ddf1ad52790e9d23}"
NATIVE_VENV="$ROOT/venv-native"
OUT_DIR="$ROOT/data/openhands/$REVISION"
OUT="$OUT_DIR/output.jsonl"
MANIFEST="$OUT_DIR/source.json"

source /etc/profile.d/modules.sh 2>/dev/null || true
ml devel python/3.12.1 2>/dev/null || true
test -x "$NATIVE_VENV/bin/python" || {
    echo "ERROR: native venv missing at $NATIVE_VENV" >&2
    exit 1
}
source "$NATIVE_VENV/bin/activate"
export HF_HOME="$ROOT/hf"
mkdir -p "$OUT_DIR"

python3 - "$REVISION" "$OUT" "$MANIFEST" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

dataset = "OpenHands/openhands-evaluation-outputs"
data_file = (
    "outputs/SWE-bench_Lite-test/CodeActAgent/"
    "claude-3-5-sonnet-20241022_maxiter_100_N_v2.2-no-hint/output.jsonl"
)
revision, out_value, manifest_value = sys.argv[1:]
source = Path(hf_hub_download(
    repo_id=dataset, repo_type="dataset", filename=data_file, revision=revision
))
out = Path(out_value)
if not out.exists() or out.stat().st_size != source.stat().st_size:
    shutil.copyfile(source, out)
digest = hashlib.sha256(out.read_bytes()).hexdigest()
Path(manifest_value).write_text(json.dumps({
    "dataset": dataset,
    "revision": revision,
    "data_file": data_file,
    "sha256": digest,
}, indent=2, sort_keys=True) + "\n")
print(f"staged {out} ({out.stat().st_size} bytes, sha256={digest})")
PY
