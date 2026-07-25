#!/usr/bin/env bash
#SBATCH --job-name=ptsim-pd-transfer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%x-%j.out

set -Eeuo pipefail
cd "${POWERTRACE_REPO:-$HOME/powertrace-sim}"
bash profiling/jobs/run_disaggregated_transfer_gpt_oss_20b.sh \
    "${POWERTRACE_DISAGG_MODE:-campaign}"
