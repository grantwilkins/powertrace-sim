# Predictors

This directory contains prediction and smoothing utilities for power trace generation.

## Files

### `smooth_sampler.py`

Provides smoothing and sampling utilities for generated power traces to improve temporal realism.

#### Key Functions

```python
def smooth_power_trace(
    power: np.ndarray,
    window_size: int = 5,
    method: str = "median",
) -> np.ndarray:
    """
    Apply smoothing to a power trace to reduce noise.

    Args:
        power: Raw power trace array
        window_size: Smoothing window size
        method: Smoothing method ("median", "mean", "gaussian")

    Returns:
        Smoothed power trace
    """
```

```python
def resample_trace(
    timestamps: np.ndarray,
    power: np.ndarray,
    target_dt: float,
    method: str = "linear",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample power trace to different time resolution.

    Args:
        timestamps: Original timestamps
        power: Original power values
        target_dt: Target time step (seconds)
        method: Interpolation method ("linear", "nearest", "cubic")

    Returns:
        (new_timestamps, resampled_power)
    """
```

```python
def apply_ar1_smoothing(
    power: np.ndarray,
    phi: float,
    sigma: float,
    seed: int = None,
) -> np.ndarray:
    """
    Apply AR(1) process smoothing to power trace.

    The AR(1) model: x[t] = phi * x[t-1] + epsilon
    where epsilon ~ N(0, sigma^2)

    Args:
        power: Input power trace
        phi: AR(1) coefficient (0 < phi < 1 for stationarity)
        sigma: Innovation standard deviation

    Returns:
        AR(1) smoothed power trace
    """
```

```python
def sample_from_gmm_with_smoothing(
    state_sequence: np.ndarray,
    gmm_params: Dict[str, np.ndarray],
    ar1_phi: float = 0.9,
    ar1_sigma: float = 10.0,
    seed: int = None,
) -> np.ndarray:
    """
    Sample power values from GMM with AR(1) correlation.

    Combines GMM sampling with AR(1) smoothing for
    realistic temporal dynamics.

    Args:
        state_sequence: Predicted state indices
        gmm_params: GMM parameters (means, stds, weights)
        ar1_phi: AR(1) coefficient
        ar1_sigma: AR(1) innovation std

    Returns:
        Smoothed power trace
    """
```

#### Usage

```python
from model.predictors.smooth_sampler import (
    smooth_power_trace,
    resample_trace,
    apply_ar1_smoothing,
)
import numpy as np

# Smooth a noisy trace
raw_power = np.random.randn(1000) * 50 + 400
smooth_power = smooth_power_trace(raw_power, window_size=5, method="median")

# Resample to different resolution
timestamps = np.arange(0, 100, 0.1)  # 100ms resolution
power = np.random.randn(1000) * 50 + 400
new_ts, new_power = resample_trace(timestamps, power, target_dt=1.0)  # 1s resolution

# Apply AR(1) smoothing
ar1_power = apply_ar1_smoothing(power, phi=0.95, sigma=5.0)
```

## AR(1) Smoothing

The AR(1) (autoregressive order 1) model is used to add realistic temporal correlation to generated traces:

```
power[t] = phi * power[t-1] + (1 - phi) * gmm_sample[t] + noise
```

Where:
- `phi` controls the persistence (higher = more smoothing)
- `noise` ~ N(0, sigma^2) adds realistic variation

Typical values:
- `phi = 0.9-0.95` for smooth transitions
- `sigma = 5-15 W` for realistic variation

## Thresholding

The smoothing module also supports idle state thresholding:

```python
def apply_idle_threshold(
    power: np.ndarray,
    active_requests: np.ndarray,
    idle_power: float,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Apply idle power when request count falls below threshold.

    Args:
        power: Power trace
        active_requests: Concurrent request counts
        idle_power: Power value for idle state
        threshold: Request count threshold for idle

    Returns:
        Power trace with idle thresholding applied
    """
```

This ensures that power drops to idle levels when no requests are active, which is important for realistic power profiles during low-traffic periods.
