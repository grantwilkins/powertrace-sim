#!/usr/bin/env bash

# Train GRU classifiers across all models, hardwares, and TPs
# Fixed hyperparameters: H=64, LR=1e-4, bidirectional GRU

set -e

# Configuration
DATA_DIR="data"
RESULTS_BASE="results/training"
HIDDEN_SIZE=64
NUM_EPOCHS=500
BATCH_SIZE=8
SEED=42
WANDB_PROJECT="powertrace-classifier-training"

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

get_model_lr() {
    case "$1" in
        "llama-3-8b") echo "2e-4" ;;
        "llama-3-70b") echo "1e-4" ;;
        "gpt-oss-20b") echo "2e-4" ;;
        "gpt-oss-120b") echo "1e-4" ;;
        *) echo "1e-4" ;;
    esac
}

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

# Main execution
main() {
    log_info "=========================================="
    log_info "PowerTrace Classifier Training"
    log_info "=========================================="
    log_info "Hidden Size: $HIDDEN_SIZE"
    log_info "Learning Rate: $LR"
    log_info "Num Epochs: $NUM_EPOCHS"
    log_info "Batch Size: $BATCH_SIZE"
    log_info "Bidirectional: True"
    log_info "Scheduler: cosine with warmup"
    log_info "Seed: $SEED"
    log_info "=========================================="
    echo ""

    mkdir -p "${RESULTS_BASE}"

    # Models and hardware combinations
    for model in "llama-3-8b" "llama-3-70b" "gpt-oss-20b" "gpt-oss-120b"; do
        TPS=$(get_model_tps "$model")
        LR=$(get_model_lr "$model")

        for hardware in "h100" "a100"; do
            data_file="${DATA_DIR}/random_${model}_${hardware}.npz"

            # Check if data file exists
            if [ ! -f "${data_file}" ]; then
                log_error "Data file not found: ${data_file}, skipping..."
                continue
            fi

            log_info "========================================"
            log_info "Processing: ${model} on ${hardware}"
            log_info "TP values: ${TPS}"
            log_info "Data file: ${data_file}"
            log_info "========================================"

            for tp in ${TPS}; do
                echo ""
                log_info "--- Training TP=${tp} ---"

                # Create output directory
                output_dir="${RESULTS_BASE}/${model}_${hardware}_tp${tp}"
                mkdir -p "${output_dir}"

                # Run training
                python model/train_entry.py \
                    --data_file "${data_file}" \
                    --model "${model}" \
                    --hardware_accelerator "${hardware}" \
                    --tp "${tp}" \
                    --stage "train" \
                    --hidden_size "${HIDDEN_SIZE}" \
                    --lr "${LR}" \
                    --num_epochs "${NUM_EPOCHS}" \
                    --seed "${SEED}" \
                    --output_dir "${output_dir}" \
                    --bidirectional \
                    --save_model \
                    --wandb_project "${WANDB_PROJECT}" \
                    --wandb_run_name "${model}_${hardware}_tp${tp}_H${HIDDEN_SIZE}_lr${LR}"

                log_success "Completed TP=${tp} for ${model} on ${hardware}"
                log_info "Results saved to: ${output_dir}"
            done

            echo ""
            log_success "Completed all TPs for ${model} on ${hardware}"
            echo ""
        done
    done

    log_success "=========================================="
    log_success "All training completed!"
    log_success "Results saved to: ${RESULTS_BASE}/"
    log_success "=========================================="
}

# Run main function
main "$@"
