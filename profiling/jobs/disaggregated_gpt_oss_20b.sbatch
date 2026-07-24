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
source /etc/profile.d/modules.sh 2>/dev/null || true
ml devel python/3.12.1 2>/dev/null || true
REPO="${POWERTRACE_REPO:-$HOME/powertrace-sim}"
cd "$REPO"
exec bash profiling/jobs/run_disaggregated_gpt_oss_20b.sh "${POWERTRACE_DISAGG_MODE:-campaign}"
