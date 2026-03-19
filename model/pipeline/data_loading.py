from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from model.classifiers.features import build_features_from_active
from model.classifiers.gmm_bigru import build_state_labels
from model.pipeline.manifest_validation import validate_manifest
from model.utils.io import resolve_existing_path as _resolve_existing_path


def _tensor_from_array(
    values: object,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    arr = np.asarray(values)
    try:
        tensor = torch.from_numpy(arr)
    except RuntimeError as exc:
        if "Numpy is not available" not in str(exc):
            raise
        tensor = torch.tensor(arr.tolist())
    return tensor.to(device=device, dtype=dtype)


def _extract_raw_trace(
    *,
    idx: int,
    pair_key_arr: np.ndarray,
    power_arr: np.ndarray,
    active_arr: np.ndarray,
    feature_set: str,
) -> Optional[Dict[str, object]]:
    n = int(min(len(pair_key_arr), len(power_arr), len(active_arr)))
    if idx < 0 or idx >= n:
        return None
    try:
        power = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
        active = np.asarray(active_arr[idx], dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if power.size < 2 or active.size < 2:
        return None
    if not (np.all(np.isfinite(power)) and np.all(np.isfinite(active))):
        return None

    if str(feature_set).strip().lower() == "f3":
        raise ValueError("feature_set='f3' is no longer supported; use 'f2'.")

    pair_key = str(pair_key_arr[idx]) if idx < len(pair_key_arr) else f"trace-{idx}"
    return {
        "trace_idx": int(idx),
        "pair_key": pair_key,
        "power_w": power.astype(np.float64),
        "active_requests": active.astype(np.float64),
        "t_arrive_log": None,
    }


def _build_split_raw(
    *,
    indices: Sequence[int],
    pair_key_arr: np.ndarray,
    power_arr: np.ndarray,
    active_arr: np.ndarray,
    feature_set: str,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for idx in indices:
        tr = _extract_raw_trace(
            idx=int(idx),
            pair_key_arr=pair_key_arr,
            power_arr=power_arr,
            active_arr=active_arr,
            feature_set=feature_set,
        )
        if tr is None:
            continue
        out.append(tr)
    return out


def load_config_data(
    config_id: str,
    config_manifest_entry: Dict[str, object],
    *,
    manifest_dir: str,
    feature_set: str,
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    dataset_path = _resolve_existing_path(
        str(config_manifest_entry.get("dataset_npz", "")), manifest_dir
    )
    split_path = _resolve_existing_path(
        str(config_manifest_entry.get("split_json", "")), manifest_dir
    )
    norm_path = _resolve_existing_path(
        str(config_manifest_entry.get("norm_params_json", "")), manifest_dir
    )
    if dataset_path is None:
        return None, "missing_dataset_npz"
    if split_path is None:
        return None, "missing_split_json"
    if norm_path is None:
        return None, "missing_norm_params_json"

    try:
        with open(split_path, "r") as f:
            split_payload = json.load(f)
        validate_manifest(split_payload, "split_manifest")
    except Exception as exc:
        return None, f"split_json_error:{type(exc).__name__}"
    try:
        with open(norm_path, "r") as f:
            norm_payload = json.load(f)
    except Exception as exc:
        return None, f"norm_json_error:{type(exc).__name__}"

    try:
        with np.load(dataset_path, allow_pickle=True) as data:
            pair_key = (
                np.asarray(data["pair_key"], dtype=object)
                if "pair_key" in data
                else np.asarray(
                    [f"trace-{i}" for i in range(len(data["power"]))], dtype=object
                )
            )
            power = np.asarray(data["power"], dtype=object)
            active = np.asarray(data["active_requests"], dtype=object)
    except Exception as exc:
        return None, f"dataset_npz_error:{type(exc).__name__}"

    train_idx = [int(i) for i in split_payload.get("train_indices", [])]
    val_idx = [int(i) for i in split_payload.get("val_indices", [])]
    test_idx = [int(i) for i in split_payload.get("test_indices", [])]

    train_raw = _build_split_raw(
        indices=train_idx,
        pair_key_arr=pair_key,
        power_arr=power,
        active_arr=active,
        feature_set=feature_set,
    )
    val_raw = _build_split_raw(
        indices=val_idx,
        pair_key_arr=pair_key,
        power_arr=power,
        active_arr=active,
        feature_set=feature_set,
    )
    test_raw = _build_split_raw(
        indices=test_idx,
        pair_key_arr=pair_key,
        power_arr=power,
        active_arr=active,
        feature_set=feature_set,
    )
    if len(train_raw) == 0:
        return None, "empty_train_split"
    if len(val_raw) == 0:
        return None, "empty_val_split"
    if len(test_raw) == 0:
        return None, "empty_test_split"

    return {
        "config_id": config_id,
        "dataset_path": dataset_path,
        "split_path": split_path,
        "norm_path": norm_path,
        "split_payload": split_payload,
        "norm_payload": norm_payload,
        "raw": {
            "train": train_raw,
            "val": val_raw,
            "test": test_raw,
        },
    }, None


def _compute_delta_stats(
    train_raw: Sequence[Dict[str, object]],
    *,
    source_norm: Dict[str, object],
    feature_set: str,
) -> Tuple[Optional[Tuple[float, float]], Optional[str]]:
    tmp_norm = {
        "active_mean": float(
            source_norm.get("active_mean", source_norm.get("A_mean", 0.0))
        ),
        "active_std": float(
            source_norm.get("active_std", source_norm.get("A_std", 1.0))
        ),
        "t_arrive_log_mean": float(
            source_norm.get(
                "t_arrive_log_mean", source_norm.get("T_arrive_log_mean", 0.0)
            )
        ),
        "t_arrive_log_std": float(
            source_norm.get(
                "t_arrive_log_std", source_norm.get("T_arrive_log_std", 1.0)
            )
        ),
        "delta_A_mean": 0.0,
        "delta_A_std": 1.0,
    }
    all_delta: List[np.ndarray] = []
    for tr in train_raw:
        power = np.asarray(tr["power_w"], dtype=np.float64)
        active = np.asarray(tr["active_requests"], dtype=np.float64)
        t_log = tr.get("t_arrive_log")
        built = build_features_from_active(
            active_requests=active,
            t_arrive_log=np.asarray(t_log, dtype=np.float64)
            if t_log is not None
            else None,
            norm=tmp_norm,
            feature_set=feature_set,
            max_length=int(max(0, power.size - 1)),
        )
        d = np.asarray(built["delta_A_raw"], dtype=np.float64).reshape(-1)
        if d.size == 0:
            continue
        finite = d[np.isfinite(d)]
        if finite.size > 0:
            all_delta.append(finite)
    if len(all_delta) == 0:
        return None, "empty_train_delta_A"
    cat = np.concatenate(all_delta, axis=0)
    mean = float(np.mean(cat))
    std = float(np.std(cat) + 1e-6)
    return (mean, std), None


def _prepare_split(
    raw_traces: Sequence[Dict[str, object]],
    *,
    norm: Dict[str, float],
    feature_set: str,
    gmm_fit: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for tr in raw_traces:
        power = np.asarray(tr["power_w"], dtype=np.float64).reshape(-1)
        active = np.asarray(tr["active_requests"], dtype=np.float64).reshape(-1)
        t_log = tr.get("t_arrive_log")
        built = build_features_from_active(
            active_requests=active,
            t_arrive_log=np.asarray(t_log, dtype=np.float64)
            if t_log is not None
            else None,
            norm=norm,
            feature_set=feature_set,
            max_length=int(max(0, power.size - 1)),
        )
        x = np.asarray(built["features_norm"], dtype=np.float32)
        L = int(len(x))
        if L <= 0:
            continue
        y_power = power[1 : L + 1].astype(np.float64)
        if y_power.size != L:
            continue
        if gmm_fit is not None:
            labels = build_state_labels(y_power, gmm_fit).astype(np.int64)
            if labels.size != L:
                continue
        else:
            labels = np.zeros((L,), dtype=np.int64)
        out.append(
            {
                "trace_idx": int(tr["trace_idx"]),
                "pair_key": str(tr["pair_key"]),
                "features_norm": x,
                "target_power_w": y_power.astype(np.float64),
                "state_labels": labels.astype(np.int64),
            }
        )
    return out
