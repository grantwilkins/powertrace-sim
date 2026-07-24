#!/usr/bin/env bash
#SBATCH --job-name=ptsim-gptoss-pd
#SBATCH --partition=ramr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPU_SKU:A100_SXM4&GPU_MEM:80GB
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/gfw/ptsim/logs/gptoss-pd-%j.out
#SBATCH --error=/scratch/users/gfw/ptsim/logs/gptoss-pd-%j.out

set -euo pipefail
MODULES_INIT=/etc/profile.d/modules.sh
test ! -r "$MODULES_INIT" || source "$MODULES_INIT"
command -v ml >/dev/null
ml devel python/3.12.1
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
REPO="${POWERTRACE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
export POWERTRACE_REPO="$REPO"
cd "$REPO"
exec bash profiling/jobs/run_disaggregated_gpt_oss_20b.sh "${POWERTRACE_DISAGG_MODE:-campaign}"
