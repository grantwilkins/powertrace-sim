# Core

This directory contains core dataset classes and utility functions for data loading and feature matrix construction.

## Files

### `dataset.py` - PowerTraceDataset

PyTorch Dataset implementation for loading and preprocessing power trace data.

```python
class PowerTraceDataset(Dataset):
    """
    Dataset for power trace training data.

    Loads NPZ files containing:
    - timestamps: Time bins for each trace
    - power: Power measurements (target variable)
    - prefill_tokens: Prefill token counts per bin
    - decode_tokens: Decode token counts per bin
    - active_requests: Concurrent request counts
    - request_ts: Individual request arrival times
    - input_tokens: Input token lengths per request
    - output_tokens: Output token lengths per request

    Automatically:
    - Fits per-TP GMMs to discretize power into states
    - Constructs feature matrices using make_schedule_matrix()
    - Computes state statistics (mu, sigma per state)

    Args:
        npz_file: Path to NPZ data file
        K: Number of GMM components (default: 6)

    Returns per __getitem__:
        x: Features tensor (T, 3) - [prefill_tokens, decode_tokens, active_requests]
        y: Power tensor (T, 1)
        z: State labels tensor (T,)
    """
```

#### Usage

```python
from model.core.dataset import PowerTraceDataset
from torch.utils.data import DataLoader

# Load dataset
dataset = PowerTraceDataset('data/benchmark_llama-3-8b_h100.npz', K=10)

# Access individual traces
x, y, z = dataset[0]
print(f"Sequence length: {x.shape[0]}")
print(f"Features: {x.shape[1]}")
print(f"Power range: {y.min():.0f} - {y.max():.0f} W")

# Create dataloader
loader = DataLoader(dataset, batch_size=1, shuffle=True)
```

#### Attributes

- `traces`: List of trace dictionaries with raw data
- `tp_all`: List of tensor parallelism settings per trace
- `hw_accelerator`: Hardware type extracted from filename
- `llm_name`: Model name extracted from filename
- `state_labels`: Per-TP GMM models
- `mu`, `sigma`: State statistics (mean, std per GMM component)

### `utils.py` - Feature Utilities

Utility functions for feature matrix construction.

```python
def make_schedule_matrix(trace: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Construct feature matrix from trace dictionary.

    Args:
        trace: Dictionary with keys:
            - prefill_tokens: (T,) prefill counts
            - decode_tokens: (T,) decode counts
            - active_requests: (T,) concurrent requests

    Returns:
        Feature matrix (T, 3) with columns:
            [prefill_tokens, decode_tokens, active_requests]
    """
```

#### Usage

```python
from model.core.utils import make_schedule_matrix
import numpy as np

trace = {
    'prefill_tokens': np.array([100, 200, 150, 0, 0]),
    'decode_tokens': np.array([0, 50, 100, 150, 100]),
    'active_requests': np.array([1, 2, 2, 1, 1]),
}

features = make_schedule_matrix(trace)
print(features.shape)  # (5, 3)
```

## Data Flow

```
NPZ File
    │
    ▼
PowerTraceDataset.__init__()
    │
    ├── Load arrays from NPZ
    ├── Build trace dictionaries
    ├── Fit per-TP GMMs
    ├── Compute state labels
    └── Calculate state statistics
    │
    ▼
PowerTraceDataset.__getitem__(idx)
    │
    ├── make_schedule_matrix(trace)
    ├── Convert to tensors
    └── Return (x, y, z)
```

## NPZ File Format

Expected structure for input NPZ files:

```python
{
    'timestamps': np.ndarray,        # (n_traces, max_len)
    'power_traces': np.ndarray,      # (n_traces, max_len)
    'prefill_tokens': np.ndarray,    # (n_traces, max_len)
    'decode_tokens': np.ndarray,     # (n_traces, max_len)
    'active_requests': np.ndarray,   # (n_traces, max_len)
    'request_timestamps': np.ndarray,# (n_traces, max_requests)
    'input_tokens': np.ndarray,      # (n_traces, max_requests)
    'output_tokens': np.ndarray,     # (n_traces, max_requests)
    'tensor_parallelism': np.ndarray # (n_traces,)
}
```

Zero-padding indicates end of variable-length sequences.
