# Classifiers

This directory contains the machine learning model implementations for power state classification and trace generation.

## Files

### `gmm_bigru.py` - Main Pipeline (Primary)

The GMM-BiGRU pipeline is the primary method for power trace generation. It combines:

1. **Gaussian Mixture Model (GMM)**: Discretizes continuous power values into K states
2. **Bidirectional GRU**: Predicts state sequences conditioned on workload features

#### Key Functions

**GMM Fitting:**

```python
def fit_power_gmm(
    power_values: ArrayLike,
    k: int = 10,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Fit a GMM to power values and return sorted parameters.
    Returns dict with means, stds, weights, and label_map for sorted ordering.
    """
```

**Feature Construction:**

```python
def build_features_from_active(
    active_requests: np.ndarray,
    norm_params: Mapping[str, float],
    feature_set: str = "f2",
) -> np.ndarray:
    """
    Build feature matrix from active request counts.
    - f2: [active_requests_norm, delta_active_norm]
    - f3: [active_requests_norm, delta_active_norm, cumulative_tokens_norm]
    """

def build_rollout_features_from_requests(
    requests: Sequence[ServeGenRequest],
    config: ServingConfig,
    duration: float,
    dt: float,
    norm_params: Mapping[str, float],
    feature_set: str = "f2",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build features for inference from a request stream.
    Returns (features, timestamps) tuple.
    """
```

**State Labels:**

```python
def build_state_labels(
    power_values: np.ndarray,
    gmm_params: Mapping[str, object],
) -> np.ndarray:
    """
    Assign GMM state labels to power values using sorted component ordering.
    """
```

**Trace Generation:**

```python
def generate_gmm_bigru_trace(
    features: np.ndarray,
    model: GRUClassifier,
    gmm_params: Mapping[str, float],
    device: str = "cpu",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate power trace by sampling from GMM based on GRU predictions.
    """

def generate_gmm_bigru_trace_ar1_thresholded(
    features: np.ndarray,
    model: GRUClassifier,
    gmm_params: Mapping[str, float],
    ar1_phi: float,
    ar1_sigma: float,
    idle_threshold: float,
    idle_power: float,
    ...
) -> np.ndarray:
    """
    Generate trace with AR(1) smoothing and idle state thresholding.
    Produces more realistic temporal dynamics.
    """
```

**I/O Utilities:**

```python
def load_gmm_params_json_dict(path: str) -> Dict[str, object]:
    """Load GMM parameters from JSON file."""

def gmm_params_to_json_dict(params: Dict[str, object]) -> Dict[str, object]:
    """Convert GMM parameters to JSON-serializable format."""
```

### `gru.py` - Neural Network Architecture

Bidirectional GRU classifier for state sequence prediction.

```python
class GRUClassifier(nn.Module):
    """
    Bidirectional GRU for power state classification.

    Args:
        input_size: Number of input features (2 for f2, 3 for f3)
        hidden_size: GRU hidden dimension (default: 64)
        num_layers: Number of GRU layers (default: 2)
        num_classes: Number of GMM components K (default: 10)
        dropout: Dropout probability (default: 0.1)
        bidirectional: Use bidirectional GRU (default: True)

    Forward:
        Input: (batch, seq_len, input_size)
        Output: (batch, seq_len, num_classes) - logits
    """
```

### `feature_utils.py` - Feature Computation

Extracted utility functions for feature computation used across the pipeline.

```python
def compute_inference_features(
    requests: List[ServeGenRequest],
    config: ServingConfig,
    T: float,
    dt: float,
) -> np.ndarray:
    """
    Compute raw features from request stream for inference.
    Returns unnormalized feature matrix.
    """

def compute_delta_active_requests(active_requests: np.ndarray) -> np.ndarray:
    """
    Compute first-order difference of active request counts.
    """

def normalize_delta_active_requests(
    delta_active: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """
    Standardize delta values using stored normalization parameters.
    """
```

### `metrics.py` - Evaluation Metrics

Comprehensive evaluation metrics for power trace quality.

```python
def compute_power_metrics(
    ground_truth_w: np.ndarray,
    generated_w: np.ndarray,
    dt: float = 0.1,
    acf_max_lag: int = 100,
) -> Dict[str, float]:
    """
    Compute full metric suite comparing generated to ground truth traces.

    Returns:
        ks_stat: Kolmogorov-Smirnov statistic (lower is better)
        acf_r2: ACF R-squared fit (higher is better)
        nrmse: Normalized RMSE (lower is better)
        p95_error_pct: 95th percentile error percentage
        p99_error_pct: 99th percentile error percentage
        delta_energy_pct: Total energy error percentage
    """
```

## Usage Example

```python
import torch
import numpy as np
from model.classifiers.gmm_bigru import (
    fit_power_gmm,
    build_features_from_active,
    build_state_labels,
    generate_gmm_bigru_trace,
)
from model.classifiers.gru import GRUClassifier

# 1. Fit GMM to training power values
gmm_params = fit_power_gmm(power_values, k=10)

# 2. Build features and labels
features = build_features_from_active(active_requests, norm_params, "f2")
labels = build_state_labels(power_values, gmm_params)

# 3. Train GRU classifier
model = GRUClassifier(input_size=2, num_classes=10)
# ... training loop ...

# 4. Generate new trace
new_features = build_features_from_active(new_active, norm_params, "f2")
generated_power = generate_gmm_bigru_trace(new_features, model, gmm_params)
```

## Feature Sets

| Feature Set | Dimensions | Features |
|-------------|------------|----------|
| f2 | 2 | active_requests_norm, delta_active_norm |
| f3 | 3 | active_requests_norm, delta_active_norm, cumulative_tokens_norm |

The `f2` feature set is recommended for most use cases as it provides good performance with lower complexity.
