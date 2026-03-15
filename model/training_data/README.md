# Training Data

This directory contains utilities for preparing, processing, and visualizing training data for the PowerTrace-Sim models.

## Directory Structure

```
training_data/
├── utils/                    # Data preparation utilities
│   ├── prepare_training_data.py      # Main data preparation script
│   ├── request_timestamps.py         # Request timestamp alignment
│   ├── stage0_inventory_and_throughput.py  # Stage0 data processing
│   ├── data_file_renamer.py          # File naming utilities
│   └── test_*.py                     # Unit tests
└── losses/                   # Training loss visualization
    └── plot_loss.py          # Loss curve plotting
```

## Data Preparation Pipeline

### Stage 0: Raw Data Processing

The `prepare_training_data.py` script converts raw benchmark JSON files into the Stage 0 format used for training.

```bash
python -m model.training_data.prepare_training_data \
    --input-dir data/sharegpt-benchmark-llama-3-8b-h100 \
    --output results/stage0
```

**Input:** Raw benchmark JSON files with request timings and power CSV files

**Output:**
```
results/stage0/
├── manifest.json              # Index of all processed data
├── {config_id}/
│   ├── run_{n}.json           # Per-run request data
│   └── power_{n}.csv          # Per-run power trace
└── throughput.json            # Aggregate throughput statistics
```

### Manifest Format

```json
{
  "configs": {
    "llama-3-8b_H100_tp1": {
      "runs": [
        {
          "run_id": "run_0",
          "requests_path": "llama-3-8b_H100_tp1/run_0.json",
          "power_path": "llama-3-8b_H100_tp1/power_0.csv",
          "num_requests": 600,
          "duration_sec": 600.0,
          "qps": 1.0
        }
      ],
      "model": "llama-3-8b",
      "hardware": "H100",
      "tp": 1
    }
  }
}
```

## Utility Modules

### `utils/request_timestamps.py`

Handles alignment between request arrival times and power measurement timestamps.

```python
def align_request_timestamps(
    requests: List[Dict],
    power_timestamps: np.ndarray,
    power_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align request timeline with power measurements.

    Returns:
        timestamps: Aligned time grid
        power: Power values on aligned grid
        active_requests: Concurrent request count per timestamp
    """
```

### `utils/stage0_inventory_and_throughput.py`

Processes Stage 0 data to compute throughput statistics and create training inventories.

```python
def compute_throughput_stats(manifest_path: str) -> Dict[str, Dict]:
    """
    Compute TTFT/TPOT statistics for each configuration.

    Returns dict with per-config statistics:
    {
        "llama-3-8b_H100_tp1": {
            "ttft_p50": 0.045,
            "ttft_p95": 0.123,
            "tpot_p50": 0.012,
            "tpot_p95": 0.028,
            ...
        }
    }
    """
```

### `utils/data_file_renamer.py`

Utilities for renaming and organizing data files to match expected conventions.

```python
def rename_benchmark_files(
    input_dir: str,
    output_dir: str,
    config_id: str,
) -> None:
    """
    Rename benchmark files to standard naming convention.

    Converts: vllm-1.0qps-tp1-Llama-3.1-8B-....json
    To:       llama-3-8b_H100_tp1/run_0.json
    """
```

## Loss Visualization

### `losses/plot_loss.py`

Plot training loss curves from training logs.

```bash
python -m model.training_data.losses.plot_loss \
    --log-dir results/continuous_v1_gmm_bigru/k10_f2/training_curves \
    --out-dir figures/training
```

**Output:**
- Per-configuration loss curves
- Aggregated loss comparison plots
- Training convergence analysis

## Data Flow

```
Raw Benchmark Data
├── vllm-*.json (request timings)
└── *.csv (power measurements)
         │
         ▼
    prepare_training_data.py
         │
         ▼
Stage 0 Data
├── manifest.json
├── {config}/run_*.json
└── {config}/power_*.csv
         │
         ▼
    train_gmm_bigru.py
         │
         ├── Fit GMM to power values
         ├── Build features from requests
         ├── Generate state labels
         └── Train GRU classifier
         │
         ▼
Trained Model Artifacts
├── checkpoints/*.pt
├── gmms/*.json
└── norm_params/*.json
```

## Adding New Data

1. **Collect benchmark data** using `profiling/client/benchmark_serving.py`
2. **Organize files** into `data/sharegpt-benchmark-{model}-{hardware}/`
3. **Run preparation** with `prepare_training_data.py`
4. **Verify manifest** contains all expected configurations
5. **Train models** using the updated manifest

## File Naming Conventions

**Benchmark JSON:**
```
vllm-{rate}qps-tp{tp}-{model}-{timestamp}.json
```

**Power CSV:**
```
{model}_tp{tp}_p{rate}_d{timestamp}.csv
```

**Stage 0 Output:**
```
{config_id}/run_{n}.json
{config_id}/power_{n}.csv
```

**Config ID Format:**
```
{model}_{hardware}_tp{n}
Examples: llama-3-8b_H100_tp1, deepseek-r1-distill-70b_A100_tp4
```
