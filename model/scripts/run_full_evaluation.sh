#!/bin/bash
# Full evaluation pipeline for paper results
# Generates Main Figure 1, Main Figure 2, and Main Table 1

set -e  # Exit on error

# Configuration
DATA_DIR="./training_data/processed"
WEIGHTS_DIR="./gru_classifier_weights"
SUMMARY_DIR="./summary_data"
OUTPUT_DIR="./evaluation_results"
FIGURES_DIR="./figures"
NUM_SAMPLES=100

echo "=========================================="
echo "Power Trace Inference Evaluation Pipeline"
echo "=========================================="

# Step 1: Run evaluation for all configurations
echo ""
echo "Step 1: Running evaluation for all configurations..."
echo ""

# LLaMA-3-8B on A100
python -m evaluate \
    --data-file "${DATA_DIR}/llama-3-8b_a100.npz" \
    --weights-dir "${WEIGHTS_DIR}" \
    --summary-dir "${SUMMARY_DIR}" \
    --output-dir "${OUTPUT_DIR}/llama-3-8b_a100" \
    --num-samples ${NUM_SAMPLES} \
    --configs "llama-3-8b:a100:1" "llama-3-8b:a100:2"

# LLaMA-3-8B on H100
python -m evaluate \
    --data-file "${DATA_DIR}/llama-3-8b_h100.npz" \
    --weights-dir "${WEIGHTS_DIR}" \
    --summary-dir "${SUMMARY_DIR}" \
    --output-dir "${OUTPUT_DIR}/llama-3-8b_h100" \
    --num-samples ${NUM_SAMPLES} \
    --configs "llama-3-8b:h100:1" "llama-3-8b:h100:2"

# LLaMA-3-70B on H100
python -m evaluate \
    --data-file "${DATA_DIR}/llama-3-70b_h100.npz" \
    --weights-dir "${WEIGHTS_DIR}" \
    --summary-dir "${SUMMARY_DIR}" \
    --output-dir "${OUTPUT_DIR}/llama-3-70b_h100" \
    --num-samples ${NUM_SAMPLES} \
    --configs "llama-3-70b:h100:4" "llama-3-70b:h100:8"

# DeepSeek-R1-Distill-8B on H100
python -m evaluate \
    --data-file "${DATA_DIR}/deepseek-r1-distill-8b_h100.npz" \
    --weights-dir "${WEIGHTS_DIR}" \
    --summary-dir "${SUMMARY_DIR}" \
    --output-dir "${OUTPUT_DIR}/deepseek-r1-distill-8b_h100" \
    --num-samples ${NUM_SAMPLES} \
    --configs "deepseek-r1-distill-8b:h100:1" "deepseek-r1-distill-8b:h100:2"

# DeepSeek-R1-Distill-70B on H100
python -m evaluate \
    --data-file "${DATA_DIR}/deepseek-r1-distill-70b_h100.npz" \
    --weights-dir "${WEIGHTS_DIR}" \
    --summary-dir "${SUMMARY_DIR}" \
    --output-dir "${OUTPUT_DIR}/deepseek-r1-distill-70b_h100" \
    --num-samples ${NUM_SAMPLES} \
    --configs "deepseek-r1-distill-70b:h100:4"

# Step 2: Aggregate metrics from all evaluations
echo ""
echo "Step 2: Aggregating metrics into Main Table 1..."
echo ""

python -c "
import pandas as pd
import os
from glob import glob

# Find all metrics_table.csv files
metrics_files = glob('${OUTPUT_DIR}/*/metrics_table.csv')

# Concatenate all metrics
all_metrics = []
for f in metrics_files:
    df = pd.read_csv(f)
    all_metrics.append(df)

combined = pd.concat(all_metrics, ignore_index=True)

# Sort by hardware, model, tp
combined = combined.sort_values(['hardware', 'model', 'tp'])

# Save combined table
combined.to_csv('${OUTPUT_DIR}/main_table_1.csv', index=False, float_format='%.4f')
print(f'Saved Main Table 1 to ${OUTPUT_DIR}/main_table_1.csv')
print('')
print(combined.to_string(index=False))
"

# Step 3: Generate visualizations
echo ""
echo "Step 3: Generating Main Figure 1 and Main Figure 2..."
echo ""

# Generate all visualizations
python -m visualization.plot_evaluation \
    --evaluation-dir "${OUTPUT_DIR}" \
    --output-dir "${FIGURES_DIR}"

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Main Table 1: ${OUTPUT_DIR}/main_table_1.csv"
echo "  - Individual config metrics: ${OUTPUT_DIR}/*/metrics_table.csv"
echo "  - Main Figure 1 (power traces): ${FIGURES_DIR}/*/power_trace_overlay_*.png"
echo "  - Main Figure 2 (CDFs): ${FIGURES_DIR}/*/phase_duration_cdfs.png"
echo "  - Phase statistics: ${OUTPUT_DIR}/*/phase_stats/"
echo "  - Trace examples: ${OUTPUT_DIR}/*/trace_examples/"
echo ""
