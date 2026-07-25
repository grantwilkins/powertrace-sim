#!/usr/bin/env bash
#SBATCH --job-name=ptsim-pd-transfer
#SBATCH --partition=ramr
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%x-%j.out

set -Eeuo pipefail
MODULES_INIT=/etc/profile.d/modules.sh
test ! -r "$MODULES_INIT" || source "$MODULES_INIT"
command -v ml >/dev/null
ml devel python/3.12.1
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
cd "${POWERTRACE_REPO:-$HOME/powertrace-sim}"
bash profiling/jobs/run_disaggregated_transfer_gpt_oss_20b.sh \
    "${POWERTRACE_DISAGG_MODE:-campaign}"
