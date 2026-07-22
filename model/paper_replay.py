"""Score the frozen release artifact on the canonical measured replay cache."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np

from model.metrics import compute_power_metrics, downsample_mean
from model.power import predict_power
from model.release import load_artifact, support_violations

LEDGER_CHANNELS = (
    "busy", "batch", "w_read", "kv_read", "kv_write",
    "engine_iterations_rate", "gemm_flops_rate", "attn_flops_rate",
    "prefill_gemm_flops_rate", "decode_gemm_flops_rate",
    "prefill_attn_flops_rate", "decode_attn_flops_rate",
    "prefill_attn_bytes_rate", "decode_attn_bytes_rate",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as source:
        cache = {key: source[key] for key in source.files}
    required = {
        "run_id", "power", "tp", "rate", "model_idx", "model_names",
        "hw_idx", "hw_names", "family_idx", "family_names", "dt_s",
        *LEDGER_CHANNELS,
    }
    missing = sorted(required - cache.keys())
    if missing:
        raise ValueError(f"replay cache missing fields: {missing}")
    if float(cache["dt_s"]) != 0.25:
        raise ValueError("paper replay requires the native 250 ms cache")
    return cache


def run_slices(run_ids: np.ndarray) -> Iterator[tuple[int, int, int]]:
    values = np.asarray(run_ids, dtype=int)
    if values.size == 0:
        return
    edges = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, values.size]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.any(values[lo:hi] != values[lo]):
            raise ValueError("run IDs must be contiguous")
        yield int(values[lo]), int(lo), int(hi)


def _constant(cache: Mapping[str, np.ndarray], key: str, lo: int, hi: int):
    values = np.unique(cache[key][lo:hi])
    if values.size != 1:
        raise ValueError(f"run has non-constant {key}")
    return values[0]


def run_metadata(cache: Mapping[str, np.ndarray], lo: int, hi: int) -> dict:
    model_idx = int(_constant(cache, "model_idx", lo, hi))
    hardware_idx = int(_constant(cache, "hw_idx", lo, hi))
    family_idx = int(_constant(cache, "family_idx", lo, hi))
    return {
        "model": str(cache["model_names"][model_idx]),
        "hardware": str(cache["hw_names"][hardware_idx]),
        "family": str(cache["family_names"][family_idx]),
        "tp": int(_constant(cache, "tp", lo, hi)),
        "rate": float(_constant(cache, "rate", lo, hi)),
    }


def _supported(meta: Mapping[str, object], artifact: Mapping[str, object]) -> bool:
    config = {
        **meta,
        "scheduler": "vllm_v1_decode_first",
        "moe_routing": artifact["routing"]["mode"],
    }
    return not support_violations(config, dict(artifact))


def predict_cache(
    cache: Mapping[str, np.ndarray], artifact: Mapping[str, object],
) -> tuple[np.ndarray, dict[int, bool]]:
    """Predict TP-summed node watts, resetting meter response at every run."""
    prediction = np.full(cache["run_id"].size, np.nan, dtype=float)
    support = {}
    for run_id, lo, hi in run_slices(cache["run_id"]):
        meta = run_metadata(cache, lo, hi)
        model = meta["model"]
        ledger = {key: np.asarray(cache[key][lo:hi]) for key in LEDGER_CHANNELS}
        power = predict_power(
            ledger,
            arch=artifact["architectures"][model],
            model=model,
            hardware=meta["hardware"],
            tp=meta["tp"],
            artifact=artifact,
            dt_s=float(cache["dt_s"]),
        )
        prediction[lo:hi] = power["node_gpu_power_w"]
        support[run_id] = _supported(meta, artifact)
    if not np.isfinite(prediction).all():
        raise AssertionError("release artifact left replay bins unpredicted")
    return prediction, support


def interpolate_power(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Interpolate internal gaps and hold the nearest observed edge value."""
    source = np.asarray(values, dtype=float)
    missing = ~np.isfinite(source)
    if not missing.any():
        return source, 0
    if missing.all():
        raise ValueError("run has no measured power")
    output = source.copy()
    observed = np.flatnonzero(~missing)
    output[missing] = np.interp(np.flatnonzero(missing), observed, source[observed])
    return output, int(missing.sum())


def one_second_per_gpu(
    measured_node: np.ndarray, predicted_node: np.ndarray, *, tp: int, dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if tp <= 0:
        raise ValueError("TP must be positive")
    measured = downsample_mean(measured_node, dt=dt_s, resolution_s=1.0)
    predicted = downsample_mean(predicted_node, dt=dt_s, resolution_s=1.0)
    size = min(measured.size, predicted.size)
    if size == 0:
        raise ValueError("run has no complete one-second interval")
    return measured[:size] / tp, predicted[:size] / tp


def score_cache(
    cache: Mapping[str, np.ndarray], artifact: Mapping[str, object],
    roles: Mapping[int, str],
) -> tuple[list[dict], np.ndarray]:
    prediction, support = predict_cache(cache, artifact)
    rows = []
    for run_id, lo, hi in run_slices(cache["run_id"]):
        if run_id not in roles:
            raise ValueError(f"split manifest has no role for run {run_id}")
        meta = run_metadata(cache, lo, hi)
        measured, interpolated = interpolate_power(cache["power"][lo:hi])
        observed, predicted = one_second_per_gpu(
            measured, prediction[lo:hi], tp=meta["tp"], dt_s=float(cache["dt_s"]),
        )
        metrics = compute_power_metrics(observed, predicted, dt=1.0, acf_max_lag=60)
        rows.append({
            "run_id": run_id,
            **meta,
            "role": roles[run_id],
            "surface_supported": support[run_id],
            "interpolated_power_bins": interpolated,
            "seconds": observed.size,
            "energy_error_pct": metrics["delta_energy_pct"],
            "rmse_w_per_gpu": float(np.sqrt(np.mean((predicted - observed) ** 2))),
            "nrmse_range": metrics["nrmse"],
            "ks_agreement": 1.0 - metrics["ks_stat"],
            "acf_r2": metrics["acf_r2"],
        })
    return rows, prediction


def load_roles(path: str | Path) -> dict[int, str]:
    payload = json.loads(Path(path).read_text())
    return {int(run): str(role) for run, role in payload["roles"].items()}


def write_scores(
    rows: list[dict], *, out_csv: str | Path, out_json: str | Path,
    cache_path: str | Path, artifact_path: str | Path, split_path: str | Path,
) -> None:
    with Path(out_csv).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "powertrace-release-replay-v1",
        "artifact": {"path": str(artifact_path), "sha256": sha256_file(artifact_path)},
        "cache": {"path": str(cache_path), "sha256": sha256_file(cache_path)},
        "split": {"path": str(split_path), "sha256": sha256_file(split_path)},
        "aggregation": "matched nonoverlapping one-second per-GPU means",
        "runs": len(rows),
        "supported_runs": sum(row["surface_supported"] for row in rows),
        "output_csv": str(out_csv),
    }
    Path(out_json).write_text(json.dumps(report, indent=2) + "\n")


def evaluate_replay(
    *, cache_path: str | Path, artifact_path: str | Path,
    split_path: str | Path, out_csv: str | Path, out_json: str | Path,
) -> tuple[list[dict], np.ndarray, dict[str, np.ndarray]]:
    cache = load_cache(cache_path)
    artifact = load_artifact(artifact_path)
    rows, prediction = score_cache(cache, artifact, load_roles(split_path))
    write_scores(
        rows, out_csv=out_csv, out_json=out_json, cache_path=cache_path,
        artifact_path=artifact_path, split_path=split_path,
    )
    return rows, prediction, cache
