from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_normalization_stats(
    traces: List[Dict[str, object]],
) -> Dict[str, float]:
    """Compute normalization statistics across all traces for a config."""
    all_power: List[np.ndarray] = []
    all_active: List[np.ndarray] = []
    all_t_arrive_log: List[np.ndarray] = []

    for tr in traces:
        all_power.append(tr["power"])
        all_active.append(tr["active_requests"])
        all_t_arrive_log.append(tr["t_arrive_log"])

    power_cat = np.concatenate(all_power)
    active_cat = np.concatenate(all_active)
    t_log_cat = np.concatenate(all_t_arrive_log)

    return {
        "power_mean": float(np.mean(power_cat)),
        "power_std": float(np.std(power_cat) + 1e-6),
        "power_min": float(np.min(power_cat)),
        "power_max": float(np.max(power_cat)),
        "active_mean": float(np.mean(active_cat)),
        "active_std": float(np.std(active_cat) + 1e-6),
        "t_arrive_log_mean": float(np.mean(t_log_cat)),
        "t_arrive_log_std": float(np.std(t_log_cat) + 1e-6),
    }


def create_train_val_test_split(
    n_traces: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Create train/val/test split indices."""
    if n_traces < 3:
        raise ValueError("At least three traces are required for disjoint train/val/test splits")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_traces).tolist()

    n_train = min(max(1, int(n_traces * train_ratio)), n_traces - 2)
    n_val = min(max(1, int(n_traces * val_ratio)), n_traces - n_train - 1)

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    return {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }

