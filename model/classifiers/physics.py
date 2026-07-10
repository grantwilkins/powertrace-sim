"""Pure first-principles node-power kernel and deployment artifact reader.

The fit artifact's ``training_ledger_artifact`` records the exact ledger and
run-index hashes, while ``training_source_index`` records stable source
identities for each hardware fit, and whether the fitting Git tree was dirty.
Those provenance fields explain the coefficients; the kernel itself remains a
deterministic consumer and never substitutes a different training source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

SCHEMA_VERSION = "powertrace-physics-v1"
FEATURE_ORDER = (
    "tp",
    "tp_link",
    "busy_tp",
    "sat_bw_a",
    "sat_bw_b",
    "sat_bw_c",
    "flops_pre",
    "flops_dec",
    "w_read_pre",
    "kv_write",
    "comm",
)


def load_physics_artifact(path: str | Path) -> dict:
    artifact = json.loads(Path(path).read_text())
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported physics artifact: {artifact.get('schema_version')}")
    if tuple(artifact.get("feature_order", ())) != FEATURE_ORDER:
        raise ValueError("Physics artifact feature order does not match the kernel")
    return artifact


def ledger_power_features(
    ledger: Mapping[str, np.ndarray],
    arch: Mapping[str, object],
    *,
    tp: int,
    hbm_bandwidth_bytes_s: float,
) -> dict[str, np.ndarray]:
    """Map per-bin work rates to the deployed 11-term power basis."""
    tp = int(tp)
    bandwidth = float(hbm_bandwidth_bytes_s)
    if tp < 1 or bandwidth <= 0.0:
        raise ValueError("tp and hbm_bandwidth_bytes_s must be positive")
    required = ("pre_tok", "dec_tok", "w_read", "kv_read", "w_read_pre", "kv_write", "comm")
    arrays = {key: np.asarray(ledger[key], dtype=np.float64).reshape(-1) for key in required}
    lengths = {arr.size for arr in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("Ledger feature columns have different lengths")

    n_active = float(arch["n_active"])
    fp8_scale = 0.5 if bool(arch.get("fp8", 0)) else 1.0
    util_mem = np.clip(
        (arrays["w_read"] + arrays["kv_read"]) / (tp * bandwidth), 0.0, 1.5
    )
    busy = (arrays["pre_tok"] + arrays["dec_tok"] > 0.0).astype(np.float64)
    ones = np.ones_like(busy)
    return {
        "tp": tp * ones,
        "tp_link": tp * (tp > 1) * ones,
        "busy_tp": tp * busy,
        "sat_bw_a": tp * (1.0 - np.exp(-util_mem / 0.05)),
        "sat_bw_b": tp * (1.0 - np.exp(-util_mem / 0.15)),
        "sat_bw_c": tp * (1.0 - np.exp(-util_mem / 0.4)),
        "flops_pre": fp8_scale * 2.0 * n_active * arrays["pre_tok"],
        "flops_dec": fp8_scale * 2.0 * n_active * arrays["dec_tok"],
        "w_read_pre": arrays["w_read_pre"],
        "kv_write": arrays["kv_write"],
        "comm": arrays["comm"],
    }


def apply_meter_lag(
    values: np.ndarray, *, moving_average_bins: int, ema_alpha: float
) -> np.ndarray:
    """Apply the artifact's causal moving-average then EMA convention."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    window = int(moving_average_bins)
    alpha = float(ema_alpha)
    if window < 1 or not 0.0 < alpha <= 1.0:
        raise ValueError("moving_average_bins >= 1 and 0 < ema_alpha <= 1 required")
    if x.size == 0:
        return x.copy()
    if window > 1:
        padded = np.concatenate([np.full(window - 1, x[0]), x])
        x = np.convolve(padded, np.ones(window) / window, mode="valid")
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def predict_mean_node_power(
    ledger: Mapping[str, np.ndarray],
    arch: Mapping[str, object],
    *,
    tp: int,
    hardware: str,
    artifact: Mapping[str, object],
    dt_s: float | None = None,
) -> np.ndarray:
    """Compute capped, lagged mean node power from one work ledger."""
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Physics artifact schema mismatch")
    hardware_artifacts = artifact.get("hardware")
    if not isinstance(hardware_artifacts, Mapping) or hardware not in hardware_artifacts:
        raise ValueError(f"No physics artifact for hardware {hardware!r}")
    spec = hardware_artifacts[hardware]
    coefficients = spec["coefficients"]
    if set(coefficients) != set(FEATURE_ORDER):
        raise ValueError("Physics coefficient set does not match the kernel")

    features = ledger_power_features(
        ledger,
        arch,
        tp=tp,
        hbm_bandwidth_bytes_s=float(spec["hbm_bandwidth_bytes_s"]),
    )
    idle = sum(features[key] * float(coefficients[key]) for key in ("tp", "tp_link"))
    dynamic = sum(
        features[key] * float(coefficients[key])
        for key in FEATURE_ORDER
        if key not in {"tp", "tp_link"}
    )
    family = str(arch.get("family", ""))
    multiplier = float(spec.get("family_multipliers", {}).get(family, 1.0))
    power = idle + multiplier * dynamic
    artifact_dt = float(artifact.get("dt_s", 1.0))
    inference_dt = artifact_dt if dt_s is None else float(dt_s)
    if artifact_dt <= 0.0 or inference_dt <= 0.0:
        raise ValueError("Artifact and inference timesteps must be positive")
    lag = spec["lag"]
    window = max(
        1,
        int(round(int(lag["moving_average_bins"]) * artifact_dt / inference_dt)),
    )
    alpha_fit = float(lag["ema_alpha"])
    alpha = 1.0 - (1.0 - alpha_fit) ** (inference_dt / artifact_dt)
    power = apply_meter_lag(
        power,
        moving_average_bins=window,
        ema_alpha=alpha,
    )
    return np.minimum(power, float(spec["cap_w_per_gpu"]) * int(tp))


def add_stochastic_residual(
    mean_power_w: np.ndarray, *, sigma_w: float, phi: float = 0.0, seed: int
) -> np.ndarray:
    """Add a seeded, mean-zero stationary AR(1) residual after meter lag."""
    mean = np.asarray(mean_power_w, dtype=np.float64).reshape(-1)
    sigma_w, phi = float(sigma_w), float(phi)
    if sigma_w < 0.0 or not -0.99 <= phi <= 0.99:
        raise ValueError("Require sigma_w >= 0 and -0.99 <= phi <= 0.99")
    if sigma_w == 0.0 or mean.size == 0:
        return mean.copy()
    rng = np.random.default_rng(int(seed))
    residual = np.empty_like(mean)
    residual[0] = rng.normal(0.0, sigma_w)
    innovation_sigma = sigma_w * np.sqrt(1.0 - phi**2)
    for i in range(1, mean.size):
        residual[i] = phi * residual[i - 1] + rng.normal(0.0, innovation_sigma)
    return mean + residual
