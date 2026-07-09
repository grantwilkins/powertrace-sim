#!/usr/bin/env bash

# Ablation Study Orchestration Script
# This script runs the full 3-stage ablation study for all model/hardware/TP combinations

set -e  # Exit on error

# Configuration
DATA_DIR="data"
RESULTS_BASE="results"
SEED=42

# Function to get TP values for a model
get_model_tps() {
    case "$1" in
        "llama-3-8b") echo "1 2" ;;
        "llama-3-70b") echo "4 8" ;;
        "gpt-oss-20b") echo "1 2" ;;
        "gpt-oss-120b") echo "4 8" ;;
        *) echo "" ;;
    esac
}

# Stage 1 parameters (LR sweep)
STAGE1_EPOCHS=100
STAGE1_H=64

# Stage 2 parameters (Hidden size)
STAGE2_EPOCHS=1000

# Stage 3 parameters (Directionality)
STAGE3_EPOCHS=1000

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run a training stage
run_stage() {
    local model=$1
    local hardware=$2
    local tp=$3
    local stage=$4
    local extra_args=$5

    local data_file="${DATA_DIR}/random_${model}_${hardware}.npz"
    local output_dir="${RESULTS_BASE}/stage_${stage}/${model}_${hardware}_tp${tp}"

    log_info "Running ${stage} for ${model} ${hardware} TP=${tp}"

    mkdir -p "${output_dir}"

    python model/train_entry.py \
        --data_file "${data_file}" \
        --model "${model}" \
        --hardware_accelerator "${hardware}" \
        --tp "${tp}" \
        --stage "${stage}" \
        --output_dir "${output_dir}" \
        --seed ${SEED} \
        ${extra_args}

    log_success "Completed ${stage} for ${model} ${hardware} TP=${tp}"
}

# Main execution
main() {
    log_info "Starting ablation study pipeline"
    log_info "Seed: ${SEED}"
    log_info "Results will be saved to: ${RESULTS_BASE}/"
    mkdir -p "${RESULTS_BASE}"
    for model in "llama-3-8b"; do

        TPS=$(get_model_tps "$model")

        for hardware in "h100"; do
            data_file="${DATA_DIR}/random_${model}_${hardware}.npz"

            # Check if data file exists
            if [ ! -f "${data_file}" ]; then
                log_error "Data file not found: ${data_file}"
                continue
            fi

            log_info "========================================"
            log_info "Processing: ${model} on ${hardware}"
            log_info "TP values: ${TPS}"
            log_info "========================================"

            for tp in ${TPS}; do
                echo ""
                log_info "----------------------------------------"
                log_info "Model: ${model} | Hardware: ${hardware} | TP: ${tp}"
                log_info "----------------------------------------"

                # # STAGE 1: LR Sweep
                # echo ""
                # log_info "=== STAGE 1: LR Sweep ==="
                # stage1_dir="${RESULTS_BASE}/stage1_lr_sweep/${model}_${hardware}_tp${tp}"
                # run_stage "${model}" "${hardware}" "${tp}" "lr_sweep" \
                #     "--num_epochs ${STAGE1_EPOCHS} --hidden_size ${STAGE1_H}"

                log_info "=== STAGE 2: Hidden Size Ablation ==="
                stage2_dir="${RESULTS_BASE}/stage2_hidden_size/${model}_${hardware}_tp${tp}"
                run_stage "${model}" "${hardware}" "${tp}" "hidden_size" \
                    "--num_epochs ${STAGE2_EPOCHS} --lr 0.0005"
                best_h_file="${stage2_dir}/best_hidden_size.txt"
                if [ -f "${best_h_file}" ]; then
                    best_h=$(cat "${best_h_file}")
                    log_success "Best hidden size: ${best_h}"
                else
                    log_error "Could not find best_hidden_size.txt, using default 64"
                    best_h="64"
                fi

                # STAGE 3: Directionality
                echo ""
                log_info "=== STAGE 3: Directionality (UniGRU vs BiGRU) ==="
                stage3_dir="${RESULTS_BASE}/stage3_directionality/${model}_${hardware}_tp${tp}"
                run_stage "${model}" "${hardware}" "${tp}" "directionality" \
                    "--num_epochs ${STAGE3_EPOCHS} --lr ${best_lr} --hidden_size ${best_h}"

                log_success "Completed all stages for ${model} ${hardware} TP=${tp}"
                echo ""
            done
        done
    done

    log_success "==============================================="
    log_success "Ablation study pipeline complete!"
    log_success "Results saved to: ${RESULTS_BASE}/"
    log_success "==============================================="
}

# Run main function
main "$@"
