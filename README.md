# PowerTrace-Sim: GPU Power Trace Simulator for LLM Inference

PowerTrace-Sim is a comprehensive framework for simulating realistic GPU power traces during Large Language Model (LLM) inference workloads. It combines empirical data collection from real hardware with machine learning models to generate statistically accurate power consumption patterns at multiple scales: from individual inference servers to datacenter-level aggregations.

## Overview

The core contribution is the **GMM-BiGRU pipeline**: a Gaussian Mixture Model combined with a Bidirectional GRU classifier that learns to generate power traces conditioned on workload characteristics (request arrivals, token counts, tensor parallelism). The generated traces preserve key statistical properties including:

- Power distribution (KS statistic)
- Temporal autocorrelation (ACF R^2)
- Peak power percentiles (P95, P99)
- Total energy consumption

## Tested Configurations

### Models
- **Llama-3 Family**: 8B, 70B, 405B parameters
- **DeepSeek-R1**: 8B, 70B (distilled versions)
- **GPT-OSS**: 20B, 120B

### Hardware
- NVIDIA A100 (40GB/80GB)
- NVIDIA H100 (80GB)

### Tensor Parallelism
- TP1, TP2, TP4, TP8 (configuration-dependent)

## Project Structure

```
powertrace-sim/
├── model/                    # Core ML models and simulation code
│   ├── classifiers/          # GMM-BiGRU and GRU classifier implementations
│   ├── core/                 # Dataset utilities and feature computation
│   ├── examples/             # Runnable demonstration scripts
│   ├── scripts/              # Training, evaluation, and inference pipelines
│   ├── simulators/           # Request arrival and power simulation
│   ├── tests/                # Unit and integration tests
│   ├── training_data/        # Data preprocessing and preparation utilities
│   ├── predictors/           # Power smoothing and sampling utilities
│   └── utils/                # Performance statistics extraction
├── scripts/
│   └── eval/                 # Paper evaluation scripts and baselines
├── data/                     # Benchmark data and traces (NPZ, JSON, CSV)
├── results/                  # Training outputs, checkpoints, and metrics
├── ServeGen/                 # Workload generation (git submodule)
├── client/                   # Benchmark client for data collection
├── server/                   # vLLM server launch scripts
├── job-scripts/              # End-to-end benchmarking automation
├── figures/                  # Generated evaluation figures
└── docs/                     # Additional documentation
```

## Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training/inference)

### Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/<repo>/powertrace-sim.git
cd powertrace-sim

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install ServeGen (workload generator)
cd ServeGen && pip install -e . && cd ..
```

## Quick Start

### 1. Training the GMM-BiGRU Model

```bash
# Train on all available configurations
python -m model.scripts.train_gmm_bigru \
    --stage0-manifest results/stage0/manifest.json \
    --out-dir results/continuous_v1_gmm_bigru/k10_f2 \
    --num-components 10 \
    --feature-set f2

# Train on specific configurations only
python -m model.scripts.train_gmm_bigru \
    --stage0-manifest results/stage0/manifest.json \
    --config-ids llama-3-8b_H100_tp1 llama-3-70b_H100_tp4 \
    --out-dir results/my_experiment
```

### 2. Evaluating the Model

```bash
# Evaluate on test set
python -m model.scripts.eval_gmm_bigru \
    --stage0-manifest results/stage0/manifest.json \
    --checkpoint-dir results/continuous_v1_gmm_bigru/k10_f2/checkpoints \
    --norm-dir results/continuous_v1_gmm_bigru/k10_f2/norm_params \
    --gmm-dir results/continuous_v1_gmm_bigru/k10_f2/gmms \
    --out-dir results/continuous_v1_gmm_bigru/k10_f2/eval_metrics
```

### 3. Running Inference (Generating Power Traces)

```bash
# Generate power trace from request timeline
python -m model.scripts.infer_gmm_bigru \
    --requests input_requests.csv \
    --checkpoint results/continuous_v1_gmm_bigru/k10_f2/checkpoints/llama-3-8b_H100_tp1.pt \
    --norm results/continuous_v1_gmm_bigru/k10_f2/norm_params/llama-3-8b_H100_tp1.json \
    --gmm results/continuous_v1_gmm_bigru/k10_f2/gmms/llama-3-8b_H100_tp1.json \
    --out-csv generated_power.csv
```

### 4. Running Baselines Comparison

```bash
# Compare against TDP, Mean, and Splitwise baselines
python -m scripts.eval.run_baselines_node \
    --config-id llama-3-8b_H100_tp1 \
    --num-seeds 5 \
    --out-dir results/eval_paper
```

## Data Collection

To collect your own training data:

1. **Start the vLLM server** (see `server/` scripts)
2. **Run the benchmark client** with power logging:
   ```bash
   python client/benchmark_serving.py \
       --backend vllm \
       --endpoint /v1/completions \
       --dataset-name sharegpt \
       --request-rate 1.0 \
       --num-prompts 600 \
       --save-result \
       --save-detailed
   ```
3. **Process the collected data** into training format:
   ```bash
   python -m model.training_data.utils.prepare_training_data \
       --input-dir data/sharegpt-benchmark-llama-3-8b-h100 \
       --output results/stage0
   ```

## Key Metrics

The evaluation framework computes:

| Metric | Description |
|--------|-------------|
| **KS Statistic** | Kolmogorov-Smirnov distance between generated and ground truth power CDFs |
| **ACF R^2** | R-squared of autocorrelation function fit (temporal dynamics fidelity) |
| **NRMSE** | Normalized root mean squared error |
| **P95/P99 Error** | Percentage error in 95th/99th percentile power |
| **Delta Energy** | Percentage error in total energy consumption |

## Citation

If you use PowerTrace-Sim in your research, please cite:

```bibtex
@article{powertrace2025,
  title={PowerTrace-Sim: Realistic GPU Power Trace Generation for LLM Inference},
  author={...},
  year={2025}
}
```

## Dependencies

See `requirements.txt` for the complete list. Key dependencies:
- PyTorch >= 2.0
- NumPy >= 1.24
- scikit-learn >= 1.3
- scipy >= 1.11
- matplotlib >= 3.7 (for visualization)

