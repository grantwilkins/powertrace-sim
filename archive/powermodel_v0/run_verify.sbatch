#!/bin/bash
#SBATCH -p dev
#SBATCH --job-name=pm_verify
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=%x_%j.out
# End-to-end verification of the powermodel package (numpy-only venv on -p dev).
#   1. analytic-physics + inference-chain unit tests
#   2. build the measured-state ledger from the gemma staircase/validate/agentic bundles
#   3. calibrate on idle+staircases, validate on never-fit realistic runs
set -euo pipefail

ml python/3.12.1 2>/dev/null || true
PY="$SCRATCH/npm-venv/bin/python"
REPO="/home/users/gfw/powertrace-sim"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

LEDGER="$SCRATCH/ptsim/powermodel_ledger.npz"
OUT="$SCRATCH/ptsim/powermodel_results"

echo "########## 1. unit tests"
"$PY" -m powermodel.tests.runner

echo "########## 2. build measured-state ledger"
"$PY" -m powermodel.ingest --runs-glob "$SCRATCH/ptsim/runs/*" --out "$LEDGER"

echo "########## 3. calibrate + validate"
"$PY" -m powermodel.validate --ledger "$LEDGER" --out "$OUT"

echo "########## done. results in $OUT"
