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
from scipy.signal import lfilter

SCHEMA_VERSION = "powertrace-physics-v1"
SELECTED_SCHEMA_VERSION = "powertrace-selected-physics-v1"
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

UTILIZATION_KNOTS = (0.0, 0.05, 0.15, 0.4, 1.0, 1.5)
_ROUTING_KEYS = {
    "family_multipliers", "roles", "selected_by_split", "routing",
    "tp_modes", "family_modes", "model_modes",
}


def physics_feature_order(
    mean_kind: str, *, residence: bool = False,
    state_filter_names: tuple[str, ...] = ("A_fast", "A_slow"),
) -> tuple[str, ...]:
    """Return the fixed coefficient contract for one selected mean mode."""
    if mean_kind in {"M0", "M0d"}:
        names = FEATURE_ORDER
    elif mean_kind in {"M0b", "M4A"}:
        segments = tuple(
            f"{resource}_{lo:g}_{hi:g}"
            for resource in ("compute", "memory", "communication")
            for lo, hi in zip(UTILIZATION_KNOTS[:-1], UTILIZATION_KNOTS[1:])
        )
        names = FEATURE_ORDER[:3] + segments
    elif mean_kind == "M0c":
        ramps = tuple(
            f"{resource}_ramp_{knot:g}"
            for resource in ("compute", "memory")
            for knot in UTILIZATION_KNOTS[1:]
        )
        names = FEATURE_ORDER[:3] + ramps
    else:
        raise ValueError(f"Unsupported selected physics mean: {mean_kind!r}")
    if residence:
        names += ("residence",)
    if mean_kind == "M4A":
        names += state_filter_names
    return names


def _vector(value, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 1:
        return np.full(length, float(array[0]))
    if array.size != length:
        raise ValueError(f"{name} has length {array.size}, expected {length}")
    return array


def _filter_by_run(
    values: np.ndarray, run_ids: np.ndarray, *, moving_average_bins: int,
    ema_alpha: float,
) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    if values.shape[0] == 0:
        return out
    starts = np.r_[0, np.flatnonzero(np.diff(run_ids)) + 1]
    ends = np.r_[starts[1:], values.shape[0]]
    if np.unique(run_ids).size != starts.size:
        raise ValueError("Each run must occupy one contiguous ledger segment")
    for start, end in zip(starts, ends):
        segment = values[start:end]
        if moving_average_bins > 1:
            padded = np.vstack([
                np.repeat(segment[:1], moving_average_bins - 1, axis=0), segment
            ])
            cumulative = np.vstack([np.zeros((1, segment.shape[1])), np.cumsum(padded, axis=0)])
            segment = (cumulative[moving_average_bins:] - cumulative[:-moving_average_bins]) / moving_average_bins
        for column in range(values.shape[1]):
            out[start:end, column], _ = lfilter(
                [ema_alpha], [1.0, -(1.0 - ema_alpha)], segment[:, column],
                zi=[(1.0 - ema_alpha) * segment[0, column]],
            )
    return out


def physics_design(
    ledger: Mapping[str, np.ndarray], arch: Mapping[str, object], *, tp,
    hardware_profile: Mapping[str, float], mean_kind: str,
    residence: bool = False, dt_s: float = 1.0,
    moving_average_s: float = 1.0, ema_alpha: float = 1.0,
    run_ids=None,
    state_filters: tuple[Mapping[str, object], ...] = (
        {"name": "A_fast", "ema_alpha": 0.03},
        {"name": "A_slow", "ema_alpha": 0.5},
    ),
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the selected physics basis with causal filters reset per run."""
    required = (
        "pre_tok", "dec_tok", "w_read", "kv_read", "w_read_pre",
        "kv_write", "comm",
    )
    if mean_kind == "M0d":
        required += ("w_read_dec",)
    arrays = {key: np.asarray(ledger[key], dtype=np.float64).reshape(-1) for key in required}
    lengths = {array.size for array in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("Ledger feature columns have different lengths")
    n = lengths.pop()
    tp_values = _vector(tp, n, "tp")
    if np.any(tp_values < 1.0):
        raise ValueError("tp must be positive")
    runs = np.zeros(n, dtype=np.int64) if run_ids is None else np.asarray(run_ids).reshape(-1)
    if runs.size != n:
        raise ValueError("run_ids length does not match the ledger")
    dt_s, moving_average_s = float(dt_s), float(moving_average_s)
    if dt_s <= 0.0 or moving_average_s <= 0.0:
        raise ValueError("dt_s and moving_average_s must be positive")
    moving_average_bins = max(1, int(round(moving_average_s / dt_s)))

    n_active = _vector(arch["n_active"], n, "n_active")
    w_bytes = _vector(arch["w_bytes"], n, "w_bytes")
    fp8 = _vector(arch.get("fp8", 0), n, "fp8")
    bandwidth = float(hardware_profile["hbm_bandwidth_bytes_s"])
    peak = float(hardware_profile["compute_peak_flops_s"])
    link = float(hardware_profile["link_bandwidth_bytes_s"])
    residence_bytes = float(hardware_profile["residence_bytes_per_gpu"])
    if min(bandwidth, peak, link, residence_bytes) <= 0.0:
        raise ValueError("Hardware profile scales must be positive")

    busy = (arrays["pre_tok"] + arrays["dec_tok"] > 0.0).astype(np.float64)
    if "fp8_flop_frac" in arch:
        # Fraction of FLOPs executed at the double-rate dtype. Llama-3 FP8
        # quantizes FFN matmuls only (arXiv:2407.21783 section 6.2), so the
        # effective rate scale is 1 - 0.5 * fraction, not a blanket 0.5.
        fp8_frac = np.clip(_vector(arch["fp8_flop_frac"], n, "fp8_flop_frac"), 0.0, 1.0)
        dtype_scale = 1.0 - 0.5 * fp8_frac
    else:
        dtype_scale = np.where(fp8 > 0.0, 0.5, 1.0)
    pre = dtype_scale * 2.0 * n_active * arrays["pre_tok"]
    dec = dtype_scale * 2.0 * n_active * arrays["dec_tok"]
    weight = arrays["w_read_dec"] if mean_kind == "M0d" else arrays["w_read"]
    memory = weight + arrays["kv_read"] + arrays["kv_write"]
    u_compute = (pre + dec) / (tp_values * peak)
    u_memory = memory / (tp_values * bandwidth)
    u_comm = arrays["comm"] / (tp_values * link)
    columns = [tp_values, tp_values * (tp_values > 1.0), tp_values * busy]
    if mean_kind in {"M0", "M0d"}:
        columns += [
            tp_values * (1.0 - np.exp(-u_memory / knot))
            for knot in (0.05, 0.15, 0.4)
        ]
        columns += [pre, dec, arrays["w_read_pre"], arrays["kv_write"], arrays["comm"]]
    elif mean_kind in {"M0b", "M4A"}:
        columns += [
            tp_values * np.clip(utilization - lo, 0.0, hi - lo)
            for utilization in (u_compute, u_memory, u_comm)
            for lo, hi in zip(UTILIZATION_KNOTS[:-1], UTILIZATION_KNOTS[1:])
        ]
    elif mean_kind == "M0c":
        # Concave monotone response: non-negative sums of saturating ramps
        # min(u, knot) cannot produce increasing marginal power, which the
        # source data never demand. Communication has no column: on every
        # source fit it is collinear with compute (r ~ 0.99) and carries
        # zero identified energy, so TP enters through work division only.
        columns += [
            tp_values * np.minimum(utilization, knot)
            for utilization in (u_compute, u_memory)
            for knot in UTILIZATION_KNOTS[1:]
        ]
    else:
        raise ValueError(f"Unsupported selected physics mean: {mean_kind!r}")
    if residence:
        columns.append(
            tp_values * busy * np.clip(w_bytes / (tp_values * residence_bytes), 0.0, 1.0)
        )
    design = _filter_by_run(
        np.column_stack(columns), runs, moving_average_bins=moving_average_bins,
        ema_alpha=float(ema_alpha),
    )
    state_names = tuple(str(spec["name"]) for spec in state_filters)
    if mean_kind == "M4A":
        state = tp_values * np.log1p(np.asarray(ledger["A_t"], dtype=np.float64).reshape(-1))
        if state.size != n:
            raise ValueError("A_t length does not match the ledger")
        state_columns = [
            _filter_by_run(
                state[:, None], runs, moving_average_bins=moving_average_bins,
                ema_alpha=float(spec["ema_alpha"]),
            )[:, 0]
            for spec in state_filters
        ]
        design = np.column_stack([design, *state_columns])
    names = physics_feature_order(
        mean_kind, residence=residence, state_filter_names=state_names
    )
    if design.shape[1] != len(names):
        raise AssertionError("Selected physics design does not match its coefficient order")
    return design, names


def load_physics_artifact(path: str | Path) -> dict:
    artifact = json.loads(Path(path).read_text())
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported physics artifact: {artifact.get('schema_version')}")
    if tuple(artifact.get("feature_order", ())) != FEATURE_ORDER:
        raise ValueError("Physics artifact feature order does not match the kernel")
    return artifact


def _contains_routing_key(value) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key).lower()
            routed_map = isinstance(item, Mapping) and (
                "routing" in name or "by_tp" in name
                or ("family" in name and "mode" in name)
                or ("model" in name and "mode" in name)
            )
            if key in _ROUTING_KEYS or routed_map or _contains_routing_key(item):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_routing_key(item) for item in value)
    return False


def _validate_selected_artifact(artifact: Mapping[str, object]) -> dict:
    if artifact.get("schema_version") != SELECTED_SCHEMA_VERSION:
        raise ValueError("Selected physics artifact schema mismatch")
    if _contains_routing_key(artifact):
        raise ValueError("Deployable physics artifacts cannot contain target routing maps")
    hardware = artifact.get("hardware")
    if not isinstance(hardware, str) or not hardware:
        raise ValueError("Selected physics artifact must name exactly one hardware")
    if not isinstance(artifact.get("architectures"), Mapping):
        raise ValueError("Selected physics artifact requires architecture descriptors")
    timing = artifact.get("timing_contract")
    if timing not in {"conditional_timing", "arrival_only_validated"}:
        raise ValueError("Unknown selected physics timing contract")
    dt_s = float(artifact.get("dt_s", 0.0))
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("Selected physics artifact dt_s must be positive")

    profile = artifact.get("hardware_profile")
    profile_keys = {
        "hbm_bandwidth_bytes_s", "compute_peak_flops_s",
        "link_bandwidth_bytes_s", "residence_bytes_per_gpu",
    }
    if not isinstance(profile, Mapping) or not profile_keys <= set(profile):
        raise ValueError("Selected physics artifact has an incomplete hardware profile")
    if any(not np.isfinite(float(profile[key])) or float(profile[key]) <= 0.0 for key in profile_keys):
        raise ValueError("Selected physics hardware scales must be finite and positive")

    mode = artifact.get("mode")
    if not isinstance(mode, Mapping):
        raise ValueError("Selected physics artifact has no mode")
    mean_kind = str(mode.get("mean_kind", ""))
    residence = mode.get("residence", False)
    if not isinstance(residence, bool):
        raise ValueError("Selected physics residence flag must be boolean")
    state_filters_raw = mode.get("state_filters", [])
    if not isinstance(state_filters_raw, list):
        raise ValueError("Selected physics state_filters must be a list")
    if any(not isinstance(spec, Mapping) for spec in state_filters_raw):
        raise ValueError("Selected physics state filter entries must be mappings")
    if mean_kind == "M4A" and len(state_filters_raw) != 2:
        raise ValueError("M4A requires exactly two causal A_t filters")
    if mean_kind != "M4A" and state_filters_raw:
        raise ValueError("Only M4A accepts state filters")
    state_names = tuple(str(spec.get("name", "")) for spec in state_filters_raw)
    if len(set(state_names)) != len(state_names) or any(not name for name in state_names):
        raise ValueError("Selected physics state filter names must be unique")
    for spec in state_filters_raw:
        alpha = float(spec.get("ema_alpha", 0.0))
        if not 0.0 < alpha <= 1.0:
            raise ValueError("Selected physics state filter alpha is invalid")
    names = physics_feature_order(
        mean_kind, residence=residence, state_filter_names=state_names
    )
    coefficients = mode.get("coefficients")
    if not isinstance(coefficients, Mapping) or set(coefficients) != set(names):
        raise ValueError("Selected physics coefficient set does not match its mode")
    values = np.asarray([float(coefficients[name]) for name in names])
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Selected physics coefficients must be finite and non-negative")
    lag = mode.get("lag")
    if not isinstance(lag, Mapping):
        raise ValueError("Selected physics mode has no lag")
    moving_average_s = float(lag.get("moving_average_s", 0.0))
    ema_alpha = float(lag.get("ema_alpha", 0.0))
    if not np.isfinite(moving_average_s) or moving_average_s <= 0.0 or not 0.0 < ema_alpha <= 1.0:
        raise ValueError("Selected physics lag is invalid")
    cap = float(mode.get("cap_w_per_gpu", 0.0))
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError("Selected physics cap must be finite and positive")
    learned = len(names) + 2 + len(state_filters_raw)
    declared_raw = artifact.get("learned_scalar_count", -1)
    if isinstance(declared_raw, bool) or not isinstance(declared_raw, int):
        raise ValueError("Selected physics learned scalar count must be an integer")
    declared = declared_raw
    if declared > 80:
        raise ValueError("Selected physics artifact exceeds 80 learned scalars")
    if declared != learned:
        raise ValueError(
            f"Selected physics learned scalar count mismatch: {declared} != {learned}"
        )

    provenance = artifact.get("provenance")
    source_fields = (
        "training_source_ids", "selection_source_ids", "excluded_target_source_ids"
    )
    if not isinstance(provenance, Mapping) or any(
        not isinstance(provenance.get(key), list) for key in source_fields
    ):
        raise ValueError("Selected physics artifact has incomplete source provenance")
    fitted = set(provenance["training_source_ids"]) | set(provenance["selection_source_ids"])
    if fitted & set(provenance["excluded_target_source_ids"]):
        raise ValueError("Selected physics artifact source provenance leaks target data")
    return dict(artifact)


def load_selected_physics_artifact(path: str | Path) -> dict:
    """Load one strict, hardware-local selected physics artifact."""
    return _validate_selected_artifact(json.loads(Path(path).read_text()))


def predict_selected_physics(
    ledger: Mapping[str, np.ndarray], arch: Mapping[str, object], *, tp: int,
    hardware: str, artifact: Mapping[str, object], dt_s: float | None = None,
    run_ids=None,
) -> np.ndarray:
    """Predict one selected hardware mean using the shared evaluator basis."""
    artifact = _validate_selected_artifact(artifact)
    if hardware != artifact["hardware"]:
        raise ValueError(
            f"Selected physics artifact is for {artifact['hardware']!r}, not {hardware!r}"
        )
    mode = artifact["mode"]
    state_filters = tuple(mode.get("state_filters", ()))
    resolved_dt = float(artifact["dt_s"] if dt_s is None else dt_s)
    design, names = physics_design(
        ledger, arch, tp=tp, hardware_profile=artifact["hardware_profile"],
        mean_kind=str(mode["mean_kind"]), residence=bool(mode.get("residence", False)),
        dt_s=resolved_dt, moving_average_s=float(mode["lag"]["moving_average_s"]),
        ema_alpha=float(mode["lag"]["ema_alpha"]), run_ids=run_ids,
        state_filters=state_filters,
    )
    coefficients = np.asarray([float(mode["coefficients"][name]) for name in names])
    return np.minimum(design @ coefficients, float(mode["cap_w_per_gpu"]) * int(tp))


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

