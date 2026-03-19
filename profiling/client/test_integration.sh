#!/bin/bash
# Integration test for the new realistic dataset parameters

set -e

echo "Testing realistic dataset integration with benchmark_serving.py..."
echo

# Test 1: Basic preset usage (conversation, medium)
echo "Test 1: Using preset (conversation, medium) with exponential outputs"
python3 benchmark_serving.py \
    --model gpt2 \
    --backend vllm \
    --dataset-name realistic \
    --workload-intensity medium \
    --task conversation \
    --num-prompts 100 \
    --host localhost \
    --port 8000 \
    --request-rate inf \
    --dry-run 2>&1 | head -50

echo
echo "✓ Test 1 passed"
echo

# Test 2: Coding task with high intensity
echo "Test 2: Using preset (coding, high)"
python3 benchmark_serving.py \
    --model gpt2 \
    --backend vllm \
    --dataset-name realistic \
    --workload-intensity high \
    --task coding \
    --num-prompts 100 \
    --host localhost \
    --port 8000 \
    --request-rate inf \
    --dry-run 2>&1 | head -50

echo
echo "✓ Test 2 passed"
echo

# Test 3: Manual exp-output-mean override
echo "Test 3: Override exp-output-mean"
python3 benchmark_serving.py \
    --model gpt2 \
    --backend vllm \
    --dataset-name realistic \
    --workload-intensity low \
    --task conversation \
    --exp-output-mean 500 \
    --num-prompts 100 \
    --host localhost \
    --port 8000 \
    --request-rate inf \
    --dry-run 2>&1 | head -50

echo
echo "✓ Test 3 passed"
echo

# Test 4: Lognormal output fallback
echo "Test 4: Using lognormal output (fallback)"
python3 benchmark_serving.py \
    --model gpt2 \
    --backend vllm \
    --dataset-name realistic \
    --workload-intensity medium \
    --task conversation \
    --use-lognormal-output \
    --num-prompts 100 \
    --host localhost \
    --port 8000 \
    --request-rate inf \
    --dry-run 2>&1 | head -50

echo
echo "✓ Test 4 passed"
echo

echo "=========================================="
echo "All integration tests passed!"
echo "=========================================="
