# Utils

This directory contains utility modules for performance statistics extraction and analysis.

## Files

### `extract_performance_stats.py`

Extracts performance statistics (TTFT, TPOT, latency) from benchmark data for use in simulation and analysis.

#### Key Functions

```python
def extract_ttft_tpot_from_benchmark(
    benchmark_json_path: str,
) -> Dict[str, np.ndarray]:
    """
    Extract TTFT and TPOT arrays from a benchmark JSON file.

    Returns:
        {
            'ttft': np.ndarray,  # Time to first token (seconds)
            'tpot': np.ndarray,  # Time per output token (seconds)
            'latency': np.ndarray,  # Total request latency (seconds)
            'input_tokens': np.ndarray,
            'output_tokens': np.ndarray,
        }
    """
```

```python
def compute_percentiles(
    values: np.ndarray,
    percentiles: List[float] = [50, 90, 95, 99],
) -> Dict[str, float]:
    """
    Compute percentile statistics for an array.

    Returns:
        {'p50': value, 'p90': value, 'p95': value, 'p99': value}
    """
```

```python
def aggregate_stats_by_config(
    data_dir: str,
    config_pattern: str = "*",
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate performance statistics across all benchmark files for a configuration.

    Returns nested dict:
        {
            'llama-3-8b_H100_tp1': {
                'ttft_p50': 0.045,
                'ttft_p95': 0.123,
                'tpot_p50': 0.012,
                'tpot_p95': 0.028,
                'latency_p50': 2.5,
                'latency_p95': 8.1,
            }
        }
    """
```

```python
def fit_distribution(
    values: np.ndarray,
    dist_type: str = "lognormal",
) -> Dict[str, float]:
    """
    Fit a distribution to observed values.

    Args:
        values: Observed data points
        dist_type: Distribution type ("lognormal", "gamma", "exponential")

    Returns:
        Distribution parameters (shape, loc, scale)
    """
```

#### Usage

```python
from model.utils.extract_performance_stats import (
    extract_ttft_tpot_from_benchmark,
    compute_percentiles,
    aggregate_stats_by_config,
)

# Extract from single file
stats = extract_ttft_tpot_from_benchmark(
    "data/sharegpt-benchmark-llama-3-8b-h100/vllm-1.0qps-tp1-....json"
)
print(f"Median TTFT: {np.median(stats['ttft'])*1000:.1f} ms")
print(f"Median TPOT: {np.median(stats['tpot'])*1000:.1f} ms")

# Compute percentiles
ttft_pcts = compute_percentiles(stats['ttft'])
print(f"TTFT P95: {ttft_pcts['p95']*1000:.1f} ms")

# Aggregate across all configs
all_stats = aggregate_stats_by_config("data/")
for config, config_stats in all_stats.items():
    print(f"{config}: TTFT P50={config_stats['ttft_p50']*1000:.1f}ms")
```

#### Performance Database Generation

This module is used to generate the `config/performance_database.json` file used by the simulators:

```python
from model.utils.extract_performance_stats import aggregate_stats_by_config
import json

# Aggregate all benchmark data
stats = aggregate_stats_by_config("data/")

# Save as performance database
with open("model/config/performance_database.json", "w") as f:
    json.dump(stats, f, indent=2)
```

## Related Configuration

The extracted statistics populate `model/config/performance_database.json`, which contains:

```json
{
  "llama-3-8b_H100_tp1": {
    "ttft_p50": 0.045,
    "ttft_p95": 0.123,
    "ttft_p99": 0.187,
    "tpot_p50": 0.012,
    "tpot_p95": 0.028,
    "tpot_p99": 0.041,
    "ttft_dist": {"type": "lognormal", "shape": 0.5, "loc": 0.01, "scale": 0.04},
    "tpot_dist": {"type": "gamma", "shape": 2.0, "loc": 0.005, "scale": 0.006}
  }
}
```

This database is used by `PerformanceSampler` in `arrival_simulator.py` to generate realistic request processing times.
