#!/bin/bash
# Quick test of evaluation pipeline on a single configuration

set -e

echo "Testing evaluation pipeline on llama-3-8b A100 TP=1..."
echo ""

cd /Users/grantwilkins/powertrace-sim/model

python evaluate.py \
    --data-file "./training_data/vllm-benchmark_llama-3-8b_a100.npz" \
    --weights-dir "./gru_classifier_weights" \
    --summary-dir "./summary_data" \
    --output-dir "./evaluation_results_test" \
    --num-samples 10 \
    --configs "llama-3-8b:a100:1"

echo ""
echo "Test complete! Check results in ./evaluation_results_test/"
echo ""

# Show metrics
if [ -f "./evaluation_results_test/metrics_table.csv" ]; then
    echo "Metrics:"
    cat "./evaluation_results_test/metrics_table.csv"
    echo ""
fi
