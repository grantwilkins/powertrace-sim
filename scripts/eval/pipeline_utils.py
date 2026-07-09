#!/usr/bin/env python3
"""
Pipeline utilities for evaluation scripts.

Re-exports from the GMM-BiGRU pipeline modules for a flat import surface,
plus shared helpers consolidated from the eval scripts.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from model.utils.io import load_json
from model.classifiers.gmm_bigru import (
    load_gmm_params_json_dict,
    predict_sorted_gmm_labels_from_params,
)
from model.classifiers.features import (
    build_rollout_features_from_requests,
    extract_norm_params,
)
from model.classifiers.trace_generation import (
    AR1_MIN_RUN_LENGTH,
    AR1_PHI_THRESHOLD,
    estimate_ar1_params,
    generate_gmm_bigru_trace,
    generate_gmm_bigru_trace_ar1_thresholded,
)
from model.classifiers.model_loading import load_gru_classifier
from model.pipeline.artifact_resolution import (
    resolve_checkpoint_norm_gmm_paths,
    resolve_experimental_paths,
    resolve_throughput,
)

__all__ = [
    "AR1_MIN_RUN_LENGTH",
    "AR1_PHI_THRESHOLD",
    "build_rollout_features_from_requests",
    "estimate_ar1_params",
    "extract_norm_params",
    "generate_gmm_bigru_trace",
    "generate_gmm_bigru_trace_ar1_thresholded",
    "load_gmm_params_json_dict",
    "load_gru_classifier",
    "predict_sorted_gmm_labels_from_params",
    "resolve_checkpoint_norm_gmm_paths",
    "resolve_experimental_paths",
    "resolve_throughput",
]


# ---------------------------------------------------------------------------
# Shared helpers consolidated from the eval scripts.
# ---------------------------------------------------------------------------

CONFIG_70B_TP4_RE = re.compile(r"^.+-70b_(A100|H100)_tp4$")


def _is_70b_tp4_config(config_id: str) -> bool:
    return CONFIG_70B_TP4_RE.match(str(config_id).strip()) is not None


def _finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _fmt(value: Optional[float], decimals: int, dash: str = "---") -> str:
    if value is None:
        return dash
    if not np.isfinite(float(value)):
        return dash
    return f"{float(value):.{int(decimals)}f}"


def _parse_config_ids(config_ids: Optional[Sequence[str]]) -> List[str]:
    if not config_ids:
        return []
    out: List[str] = []
    for token in config_ids:
        if token is None:
            continue
        out.extend([x.strip() for x in str(token).split(",") if x.strip()])
    deduped: List[str] = []
    seen = set()
    for cid in out:
        if cid in seen:
            continue
        deduped.append(cid)
        seen.add(cid)
    return deduped


def _resolve_existing_path(path_str: str, base_dir: str) -> Optional[str]:
    raw = Path(path_str)
    if raw.is_absolute():
        return str(raw) if raw.exists() else None
    local = Path(path_str)
    if local.exists():
        return str(local)
    from_base = Path(base_dir) / raw
    if from_base.exists():
        return str(from_base)
    return None


def _load_pair_manifest_map(pair_manifest_csv: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    base_dir = str(Path(pair_manifest_csv).resolve().parent)
    with open(pair_manifest_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("status", "")).strip() != "matched":
                continue
            key = str(row.get("pair_key", "")).strip()
            json_path_raw = str(row.get("json_path", "")).strip()
            if key == "" or json_path_raw == "":
                continue
            json_path = _resolve_existing_path(json_path_raw, base_dir)
            if json_path is not None:
                out[key] = json_path
    return out


def _load_or_estimate_ar1_params(
    *,
    config_id: str,
    gmm_params: Dict[str, object],
    train_power_traces: Sequence[np.ndarray],
    ar1_params_dir: str,
) -> Dict[str, np.ndarray]:
    ar1_path = Path(ar1_params_dir) / f"{config_id}_ar1_params.json"
    k = int(gmm_params["k"])
    if ar1_path.exists():
        payload = load_json(str(ar1_path))
        phi = np.asarray(payload.get("phi", []), dtype=np.float64).reshape(-1)
        sigma_innov = np.asarray(payload.get("sigma_innov", []), dtype=np.float64).reshape(-1)
        sigma_marginal = np.asarray(payload.get("sigma_marginal", []), dtype=np.float64).reshape(-1)
        if phi.size == k and sigma_innov.size == k and sigma_marginal.size == k:
            return {
                "phi": phi,
                "sigma_innov": sigma_innov,
                "sigma_marginal": sigma_marginal,
                "phi_threshold": float(payload.get("phi_threshold", 0.3)),
            }

    train_labels = [
        predict_sorted_gmm_labels_from_params(trace, gmm_params).astype(np.int64)
        for trace in train_power_traces
    ]
    phi, sigma_innov, sigma_marginal = estimate_ar1_params(
        gmm_params=gmm_params,
        training_power_traces=train_power_traces,
        training_labels_traces=train_labels,
        K=k,
    )
    return {
        "phi": np.asarray(phi, dtype=np.float64).reshape(-1),
        "sigma_innov": np.asarray(sigma_innov, dtype=np.float64).reshape(-1),
        "sigma_marginal": np.asarray(sigma_marginal, dtype=np.float64).reshape(-1),
        "phi_threshold": 0.3,
    }
