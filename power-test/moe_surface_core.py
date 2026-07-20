"""Core feature, fit, support, and provenance logic for the MoE power surface."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from fit_power_surface import run_slices  # noqa: E402
from response_chain import apply_chain  # noqa: E402

SCHEMA = "moe-power-surface-v3"
HARDWARE = "A100"
MODEL = "gpt-oss-20b"
SUPPORTED_TP = (1, 2)
ROUTING_MODE = "uniform"
BIN_S = 0.25
DELAY_S = 0.0
HBM_BYTES_S = 2.0e12
TDP_W_PER_GPU = 400.0
FEATURE_NAMES = (
    "tp_floor",
    "tp_link",
    "logical_memory_util_node_lag_250ms",
    "engine_iterations_rate_tp",
    "log_batch_tp",
)
DESIGN_DIGEST_KEYS = (
    "run_id",
    "tp",
    "rate",
    "w_read",
    "kv_read",
    "kv_write",
    "engine_iterations_rate",
    "batch",
)
MOE_MODELS = frozenset(("gpt-oss-20b", "gpt-oss-120b"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_array(digest, key: str, value) -> None:
    value = np.asarray(value)
    digest.update(key.encode())
    digest.update(json.dumps(value.shape).encode())
    if value.dtype.kind in "OUS":
        digest.update(json.dumps(value.tolist(), sort_keys=True).encode())
    else:
        digest.update(value.dtype.str.encode())
        digest.update(np.ascontiguousarray(value).tobytes())


def canonical_design_digest(cache: dict, runs: set[int]) -> str:
    """Hash selected model inputs and identities, excluding measured targets."""
    digest = hashlib.sha256()
    selected = np.isin(cache["run_id"], sorted(runs))
    for key in DESIGN_DIGEST_KEYS:
        if key not in cache:
            raise ValueError(f"MoE cache is missing {key!r}")
        _hash_array(digest, key, np.asarray(cache[key])[selected])
    _hash_array(digest, "model", model_names_for_rows(cache)[selected])
    _hash_array(digest, "hardware", hardware_names_for_rows(cache)[selected])
    _hash_array(digest, "role", role_names_for_rows(cache)[selected])
    _hash_array(digest, "dt_s", cache["dt_s"])
    _hash_array(digest, "routing_mode", routing_mode(cache))
    return digest.hexdigest()


def target_digest(cache: dict, runs: set[int]) -> str:
    digest = hashlib.sha256()
    selected = np.isin(cache["run_id"], sorted(runs))
    _hash_array(digest, "run_id", cache["run_id"][selected])
    _hash_array(digest, "power", cache["power"][selected])
    _hash_array(digest, "power_valid", cache["power_valid"][selected])
    return digest.hexdigest()


def routing_mode(cache: dict) -> str:
    return str(np.asarray(cache["moe_routing_mode"]).item())


def model_names_for_rows(cache: dict) -> np.ndarray:
    return np.asarray(cache["model_names"])[cache["model_idx"]].astype(str)


def hardware_names_for_rows(cache: dict) -> np.ndarray:
    return np.asarray(cache["hw_names"])[cache["hw_idx"]].astype(str)


def role_names_for_rows(cache: dict) -> np.ndarray:
    return np.asarray(cache["role_names"])[cache["role_idx"]].astype(str)


def surface_design(cache: dict) -> np.ndarray:
    """Five offline-ledger node-power features for the MoE surface."""
    tp = np.asarray(cache["tp"], float)
    memory = lag_one_bin_by_run((
        np.asarray(cache["w_read"], float)
        + np.asarray(cache["kv_read"], float)
        + np.asarray(cache["kv_write"], float)
    ), cache["run_id"])
    return np.column_stack((
        tp,
        tp * (tp > 1),
        memory / HBM_BYTES_S,
        tp * np.asarray(cache["engine_iterations_rate"], float) / 1000.0,
        tp * np.log1p(np.asarray(cache["batch"], float)),
    ))


def lag_one_bin_by_run(values: np.ndarray, run_id: np.ndarray) -> np.ndarray:
    """Lag one bin per run, holding the first source value at the boundary."""
    values = np.asarray(values, float)
    output = np.empty_like(values)
    for _, lo, hi in run_slices(np.asarray(run_id)):
        output[lo] = values[lo]
        output[lo + 1:hi] = values[lo:hi - 1]
    return output


def predict(design: np.ndarray, coefficients: np.ndarray,
            tp: np.ndarray) -> np.ndarray:
    return np.minimum(
        np.asarray(design, float) @ np.asarray(coefficients, float),
        TDP_W_PER_GPU * np.asarray(tp, float),
    )


def fit_coefficients(design: np.ndarray, power: np.ndarray,
                     run_id: np.ndarray, fit_runs: set[int]) -> np.ndarray:
    """Run-balanced NNLS: every training run contributes its mean squared error."""
    design = np.asarray(design, float)
    power = np.asarray(power, float)
    run_id = np.asarray(run_id)
    weights = np.zeros(power.size, float)
    for run in sorted(fit_runs):
        rows = (run_id == run) & np.isfinite(power)
        if not rows.any():
            raise ValueError(f"Training run {run} has no finite power bins")
        weights[rows] = 1.0 / np.sqrt(rows.sum())
    selected = weights > 0
    x = design[selected] * weights[selected, None]
    y = power[selected] * weights[selected]
    scale = np.sqrt(np.mean(x ** 2, axis=0))
    keep = scale > 0
    coefficients = np.zeros(design.shape[1])
    scaled, _ = nnls(x[:, keep] / scale[keep], y)
    coefficients[keep] = scaled / scale[keep]
    return coefficients


def supported_run(model: str, hardware: str, tp: int, mode: str) -> bool:
    return (
        model == MODEL
        and hardware == HARDWARE
        and tp in SUPPORTED_TP
        and mode == ROUTING_MODE
    )


def validate_artifact_contract(artifact: dict) -> np.ndarray:
    exact = {
        "schema_version": SCHEMA,
        "model": MODEL,
        "hardware": HARDWARE,
        "supported_tp": list(SUPPORTED_TP),
        "routing_mode": ROUTING_MODE,
        "target_units": "TP-summed node watts",
        "feature_names": list(FEATURE_NAMES),
        "response_delay_s": DELAY_S,
        "input_bin_s": BIN_S,
        "hbm_bytes_s": HBM_BYTES_S,
        "tdp_w_per_gpu": TDP_W_PER_GPU,
        "fit_role": "train",
    }
    if any(artifact.get(key) != value for key, value in exact.items()):
        raise ValueError("MoE artifact support or feature contract mismatch")
    coefficients = np.asarray(artifact.get("coefficients"), float)
    if coefficients.shape != (len(FEATURE_NAMES),) or (
            not np.isfinite(coefficients).all()) or np.any(coefficients < 0):
        raise ValueError("MoE artifact coefficients are invalid")
    return coefficients


def load_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        cache = {key: data[key] for key in data.files}
    if routing_mode(cache) != ROUTING_MODE:
        raise ValueError(
            f"MoE surface requires routing mode {ROUTING_MODE!r}, "
            f"got {routing_mode(cache)!r}")
    if float(cache["dt_s"]) != BIN_S:
        raise ValueError(f"MoE surface requires {BIN_S:g} s cache bins")
    return cache


def load_run_index(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text())
    return {int(row["run_id"]): row for row in payload["runs"]}


def load_split_roles(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text())
    return {
        int(run): role
        for role, runs in payload["role_runs"].items()
        for run in runs
    }


def run_metadata(cache: dict) -> dict[int, dict]:
    models = model_names_for_rows(cache)
    hardware = hardware_names_for_rows(cache)
    roles = role_names_for_rows(cache)
    output = {}
    for run, lo, hi in run_slices(cache["run_id"]):
        for key in ("model_idx", "hw_idx", "role_idx", "tp", "rate"):
            if np.unique(cache[key][lo:hi]).size != 1:
                raise ValueError(f"Run {run} has non-constant {key}")
        output[run] = {
            "model": models[lo],
            "hardware": hardware[lo],
            "role": roles[lo],
            "tp": int(cache["tp"][lo]),
            "rate": float(cache["rate"][lo]),
        }
    return output


def validate_cache(cache: dict, index: dict[int, dict],
                   split_roles: dict[int, str]) -> dict[int, dict]:
    n = cache["run_id"].size
    required = DESIGN_DIGEST_KEYS + (
        "power", "power_valid", "model_idx", "hw_idx", "role_idx")
    if any(np.asarray(cache[key]).shape != (n,) for key in required):
        raise ValueError("MoE cache per-bin columns disagree in length")
    if not np.isfinite(float(cache["dt_s"])) or float(cache["dt_s"]) <= 0:
        raise ValueError("MoE cache dt_s must be positive and finite")
    tp = np.asarray(cache["tp"], float)
    if (
        not np.isfinite(tp).all()
        or np.any(tp <= 0)
        or not np.array_equal(tp, np.floor(tp))
    ):
        raise ValueError("MoE cache TP values must be positive integers")
    metadata = run_metadata(cache)
    if set(metadata) != set(index) or set(metadata) != set(split_roles):
        raise ValueError("Cache, run index, and split manifest disagree on run IDs")
    for run, row in metadata.items():
        if row["model"] in MOE_MODELS and row["role"] != split_roles[run]:
            raise ValueError(f"Run {run} role disagrees with split manifest")
        expected = f"{row['model']}_{row['hardware']}_tp{row['tp']}"
        if index[run]["config_id"] != expected:
            raise ValueError(f"Run {run} identity disagrees with run index")
    if not np.isfinite(surface_design(cache)).all():
        raise ValueError("MoE cache contains non-finite design channels")
    return metadata


def training_runs(metadata: dict[int, dict]) -> set[int]:
    return {
        run for run, row in metadata.items()
        if row["model"] == MODEL
        and row["hardware"] == HARDWARE
        and row["tp"] in SUPPORTED_TP
        and row["role"] == "train"
    }


def apply_by_run(values: np.ndarray, run_id: np.ndarray, dt: float,
                 hardware: str, delay_s: float) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    for _, lo, hi in run_slices(run_id):
        output[lo:hi] = apply_chain(
            values[lo:hi], dt, hardware, delay_s)
    return output
