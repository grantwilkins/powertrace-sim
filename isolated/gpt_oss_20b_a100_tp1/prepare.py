#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG_ID = "gpt-oss-20b_A100_tp1"
DEFAULT_EXPERIMENTAL_MANIFEST = (
    REPO_ROOT / "results" / "experimental_continuous_v1_gptoss_a100" / "manifest.json"
)
DEFAULT_DATA_INVENTORY = (
    REPO_ROOT / "results" / "stage0_sharegpt_gptoss_a100" / "data_inventory.json"
)
DEFAULT_THROUGHPUT_DB = (
    REPO_ROOT / "results" / "stage0_sharegpt_gptoss_a100" / "throughput_database.json"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "prepared"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    _ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _resolve_existing_path(path_str: str, base_dir: Path) -> Path:
    raw = Path(str(path_str).strip())
    if raw.is_absolute():
        if raw.exists():
            return raw
        raise FileNotFoundError(f"Path does not exist: {raw}")
    if raw.exists():
        return raw.resolve()
    from_base = (base_dir / raw).resolve()
    if from_base.exists():
        return from_base
    from_repo = (REPO_ROOT / raw).resolve()
    if from_repo.exists():
        return from_repo
    raise FileNotFoundError(f"Unable to resolve path '{path_str}' from {base_dir}")


def _extract_request_arrays(json_path: Path) -> Dict[str, np.ndarray]:
    payload = _load_json(json_path)
    input_lens = payload.get("input_lens", [])
    output_lens = payload.get("output_lens", [])
    request_timestamps = payload.get("request_timestamps", [])

    if not (
        isinstance(input_lens, list)
        and isinstance(output_lens, list)
        and isinstance(request_timestamps, list)
    ):
        raise ValueError(f"Request JSON missing required arrays: {json_path}")

    n = int(min(len(input_lens), len(output_lens), len(request_timestamps)))
    if n <= 0:
        raise ValueError(f"Request JSON has no aligned requests: {json_path}")

    arrivals: List[float] = []
    input_tokens: List[float] = []
    output_tokens: List[float] = []
    for i in range(n):
        try:
            ts = float(request_timestamps[i])
            nin = float(input_lens[i])
            nout = float(output_lens[i])
        except Exception as exc:
            raise ValueError(f"Invalid request value at index {i} in {json_path}") from exc
        if not (np.isfinite(ts) and np.isfinite(nin) and np.isfinite(nout)):
            continue
        if ts <= 0.0:
            continue
        arrivals.append(ts)
        input_tokens.append(max(0.0, nin))
        output_tokens.append(max(0.0, nout))

    if len(arrivals) == 0:
        raise ValueError(f"Request JSON has no finite request_timestamps: {json_path}")

    return {
        "request_arrival_time_s": np.asarray(arrivals, dtype=np.float64),
        "input_tokens": np.asarray(input_tokens, dtype=np.float64),
        "output_tokens": np.asarray(output_tokens, dtype=np.float64),
    }


def _load_inventory_json_path_map(data_inventory_path: Path) -> Dict[str, Path]:
    payload = _load_json(data_inventory_path)
    paired_runs = payload.get("paired_runs", [])
    if not isinstance(paired_runs, list):
        raise ValueError(f"Invalid data inventory format: {data_inventory_path}")

    out: Dict[str, Path] = {}
    for row in paired_runs:
        if not isinstance(row, dict):
            continue
        pair_key = str(row.get("pair_key", "")).strip()
        json_path_raw = str(row.get("json_path", "")).strip()
        if pair_key == "" or json_path_raw == "":
            continue
        out[pair_key] = _resolve_existing_path(json_path_raw, data_inventory_path.parent)

    if not out:
        raise ValueError(f"No pair_key -> json_path mappings found in {data_inventory_path}")
    return out


def _extract_raw_trace(
    *,
    idx: int,
    pair_key_arr: np.ndarray,
    power_arr: np.ndarray,
    active_arr: np.ndarray,
    t_arrive_log_arr: Optional[np.ndarray],
    feature_set: str,
) -> Optional[Dict[str, object]]:
    n = int(min(len(pair_key_arr), len(power_arr), len(active_arr)))
    if t_arrive_log_arr is not None:
        n = int(min(n, len(t_arrive_log_arr)))
    if idx < 0 or idx >= n:
        return None

    power = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
    active = np.asarray(active_arr[idx], dtype=np.float64).reshape(-1)
    if power.size < 2 or active.size < 2:
        return None
    if not (np.all(np.isfinite(power)) and np.all(np.isfinite(active))):
        return None

    t_arrive_log: Optional[np.ndarray] = None
    if feature_set == "f3":
        if t_arrive_log_arr is None:
            return None
        t_arrive_log = np.asarray(t_arrive_log_arr[idx], dtype=np.float64).reshape(-1)
        if t_arrive_log.size < 2 or not np.all(np.isfinite(t_arrive_log)):
            return None

    return {
        "trace_idx": int(idx),
        "pair_key": str(pair_key_arr[idx]),
        "power_w": power.astype(np.float64),
        "active_requests": active.astype(np.float64),
        "t_arrive_log": t_arrive_log.astype(np.float64) if t_arrive_log is not None else None,
    }


def _build_split_raw(
    *,
    indices: Sequence[int],
    pair_key_arr: np.ndarray,
    power_arr: np.ndarray,
    active_arr: np.ndarray,
    t_arrive_log_arr: Optional[np.ndarray],
    feature_set: str,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for idx in indices:
        trace = _extract_raw_trace(
            idx=int(idx),
            pair_key_arr=pair_key_arr,
            power_arr=power_arr,
            active_arr=active_arr,
            t_arrive_log_arr=t_arrive_log_arr,
            feature_set=feature_set,
        )
        if trace is not None:
            out.append(trace)
    return out


def _load_prepared_training_bundle(prepared_dir: Path, feature_set: str) -> Dict[str, object]:
    dataset_path = prepared_dir / "dataset.npz"
    split_path = prepared_dir / "split.json"
    norm_path = prepared_dir / "norm_params.json"

    split_payload = _load_json(split_path)
    norm_payload = _load_json(norm_path)

    with np.load(dataset_path, allow_pickle=True) as data:
        config_arr = np.asarray(data["config_id"], dtype=object).reshape(-1)
        pair_key_arr = np.asarray(data["pair_key"], dtype=object)
        power_arr = np.asarray(data["power"], dtype=object)
        active_arr = np.asarray(data["active_requests"], dtype=object)
        t_arrive_log_arr = (
            np.asarray(data["t_arrive_log"], dtype=object) if "t_arrive_log" in data else None
        )

    config_id = str(config_arr[0]) if config_arr.size > 0 else DEFAULT_CONFIG_ID
    train_indices = [int(idx) for idx in split_payload.get("train_indices", [])]
    val_indices = [int(idx) for idx in split_payload.get("val_indices", [])]
    test_indices = [int(idx) for idx in split_payload.get("test_indices", [])]

    raw = {
        "train": _build_split_raw(
            indices=train_indices,
            pair_key_arr=pair_key_arr,
            power_arr=power_arr,
            active_arr=active_arr,
            t_arrive_log_arr=t_arrive_log_arr,
            feature_set=feature_set,
        ),
        "val": _build_split_raw(
            indices=val_indices,
            pair_key_arr=pair_key_arr,
            power_arr=power_arr,
            active_arr=active_arr,
            t_arrive_log_arr=t_arrive_log_arr,
            feature_set=feature_set,
        ),
        "test": _build_split_raw(
            indices=test_indices,
            pair_key_arr=pair_key_arr,
            power_arr=power_arr,
            active_arr=active_arr,
            t_arrive_log_arr=t_arrive_log_arr,
            feature_set=feature_set,
        ),
    }
    if not raw["train"]:
        raise ValueError("Prepared bundle has an empty train split")
    if not raw["val"]:
        raise ValueError("Prepared bundle has an empty val split")
    if not raw["test"]:
        raise ValueError("Prepared bundle has an empty test split")

    return {
        "config_id": config_id,
        "norm_payload": norm_payload,
        "raw": raw,
    }


def _compute_delta_stats(
    train_raw: Sequence[Dict[str, object]],
    *,
    source_norm: Mapping[str, object],
    feature_set: str,
) -> Tuple[float, float]:
    from model.classifiers.gmm_bigru import build_features_from_active

    tmp_norm = {
        "active_mean": float(source_norm.get("active_mean", source_norm.get("A_mean", 0.0))),
        "active_std": float(source_norm.get("active_std", source_norm.get("A_std", 1.0))),
        "t_arrive_log_mean": float(
            source_norm.get("t_arrive_log_mean", source_norm.get("T_arrive_log_mean", 0.0))
        ),
        "t_arrive_log_std": float(
            source_norm.get("t_arrive_log_std", source_norm.get("T_arrive_log_std", 1.0))
        ),
        "delta_A_mean": 0.0,
        "delta_A_std": 1.0,
    }

    delta_values: List[np.ndarray] = []
    for trace in train_raw:
        power = np.asarray(trace["power_w"], dtype=np.float64)
        active = np.asarray(trace["active_requests"], dtype=np.float64)
        t_arrive_log = trace.get("t_arrive_log")
        built = build_features_from_active(
            active_requests=active,
            t_arrive_log=np.asarray(t_arrive_log, dtype=np.float64) if t_arrive_log is not None else None,
            norm=tmp_norm,
            feature_set=feature_set,
            max_length=int(max(0, power.size - 1)),
        )
        delta_raw = np.asarray(built["delta_A_raw"], dtype=np.float64).reshape(-1)
        finite = delta_raw[np.isfinite(delta_raw)]
        if finite.size > 0:
            delta_values.append(finite)

    if not delta_values:
        raise ValueError("Unable to compute delta_A statistics from the train split")
    cat = np.concatenate(delta_values, axis=0)
    return float(np.mean(cat)), float(np.std(cat) + 1e-6)


def _prepare_supervised_split(
    raw_traces: Sequence[Dict[str, object]],
    *,
    norm_cfg: Mapping[str, float],
    feature_set: str,
    gmm_fit: Optional[Mapping[str, object]],
) -> List[Dict[str, object]]:
    from model.classifiers.gmm_bigru import build_features_from_active, build_state_labels

    out: List[Dict[str, object]] = []
    for trace in raw_traces:
        power = np.asarray(trace["power_w"], dtype=np.float64).reshape(-1)
        active = np.asarray(trace["active_requests"], dtype=np.float64).reshape(-1)
        t_arrive_log = trace.get("t_arrive_log")
        built = build_features_from_active(
            active_requests=active,
            t_arrive_log=np.asarray(t_arrive_log, dtype=np.float64) if t_arrive_log is not None else None,
            norm=norm_cfg,
            feature_set=feature_set,
            max_length=int(max(0, power.size - 1)),
        )
        features_norm = np.asarray(built["features_norm"], dtype=np.float32)
        length = int(features_norm.shape[0])
        if length <= 0:
            continue
        target_power = power[1 : length + 1].astype(np.float64)
        if target_power.size != length:
            continue
        if gmm_fit is None:
            state_labels = np.zeros((length,), dtype=np.int64)
        else:
            state_labels = build_state_labels(target_power, gmm_fit).astype(np.int64)
            if state_labels.size != length:
                continue
        out.append(
            {
                "features_norm": features_norm,
                "state_labels": state_labels,
                "target_power_w": target_power,
            }
        )
    return out


def _fit_candidate_scores(
    power_values: np.ndarray,
    *,
    k_candidates: Sequence[int],
    seed: int,
) -> Dict[str, Dict[str, float]]:
    from model.classifiers.gmm_bigru import fit_power_gmm

    out: Dict[str, Dict[str, float]] = {}
    for k in k_candidates:
        if int(k) < 1 or power_values.size < int(k):
            out[str(int(k))] = {"aic": float("nan"), "bic": float("nan")}
            continue
        fitted = fit_power_gmm(
            power_values=power_values,
            k=int(k),
            random_state=int(seed),
            n_init=10,
            max_iter=300,
            reg_covar=1e-6,
        )
        out[str(int(k))] = {
            "aic": float(fitted["aic"]),
            "bic": float(fitted["bic"]),
        }
    return out


def _select_optimal_k(
    bic_scores: Mapping[str, Mapping[str, float]],
    *,
    fallback_k: int,
) -> Tuple[int, str]:
    valid_scores: Dict[int, float] = {}
    for k_str, scores in bic_scores.items():
        try:
            k = int(k_str)
            bic = float(scores["bic"])
        except Exception:
            continue
        if np.isfinite(bic):
            valid_scores[k] = bic
    if not valid_scores:
        return int(fallback_k), "fallback_no_valid_bic_scores"
    best_k = min(valid_scores, key=valid_scores.get)
    sorted_ks = sorted(valid_scores)
    if len(sorted_ks) >= 2 and best_k == sorted_ks[-1]:
        second_best_edge = sorted_ks[-2]
        if valid_scores[best_k] < valid_scores[second_best_edge]:
            return int(best_k), f"bic_minimum_at_max_k={best_k}_may_need_higher"
    return int(best_k), f"bic_minimum_at_k={best_k}"


def build_training_inputs(
    *,
    prepared_dir: Path,
    config_id: str,
    feature_set: str,
    hidden_dim: int,
    num_layers: int,
    seed: int,
    fixed_k: Optional[int],
    bic_candidates: Sequence[int],
) -> Dict[str, object]:
    from model.classifiers.gmm_bigru import fit_power_gmm

    input_dim = 2 if feature_set == "f2" else 3
    bundle = _load_prepared_training_bundle(prepared_dir, feature_set)
    resolved_config_id = str(bundle["config_id"])
    if resolved_config_id != str(config_id):
        raise ValueError(f"Unexpected config_id in prepared bundle: {resolved_config_id}")

    delta_mean, delta_std = _compute_delta_stats(
        bundle["raw"]["train"],
        source_norm=bundle["norm_payload"],
        feature_set=feature_set,
    )
    norm_payload = {
        **bundle["norm_payload"],
        "delta_A_mean": float(delta_mean),
        "delta_A_std": float(delta_std),
        "feature_set": feature_set,
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(max(1, num_layers)),
    }
    norm_cfg = {
        "active_mean": float(norm_payload.get("active_mean", norm_payload.get("A_mean", 0.0))),
        "active_std": float(norm_payload.get("active_std", norm_payload.get("A_std", 1.0))),
        "t_arrive_log_mean": float(
            norm_payload.get("t_arrive_log_mean", norm_payload.get("T_arrive_log_mean", 0.0))
        ),
        "t_arrive_log_std": float(
            norm_payload.get("t_arrive_log_std", norm_payload.get("T_arrive_log_std", 1.0))
        ),
        "delta_A_mean": float(delta_mean),
        "delta_A_std": float(delta_std),
    }

    train_unlabeled = _prepare_supervised_split(
        bundle["raw"]["train"],
        norm_cfg=norm_cfg,
        feature_set=feature_set,
        gmm_fit=None,
    )
    val_unlabeled = _prepare_supervised_split(
        bundle["raw"]["val"],
        norm_cfg=norm_cfg,
        feature_set=feature_set,
        gmm_fit=None,
    )
    if not train_unlabeled or not val_unlabeled:
        raise ValueError("Unable to build train/val features from the prepared bundle")

    train_power = np.concatenate(
        [np.asarray(trace["target_power_w"], dtype=np.float64).reshape(-1) for trace in train_unlabeled],
        axis=0,
    )
    bic_scan = _fit_candidate_scores(train_power, k_candidates=bic_candidates, seed=int(seed))
    selected_k, k_selection_reason = (
        (int(fixed_k), "fixed")
        if fixed_k is not None
        else _select_optimal_k(bic_scan, fallback_k=10)
    )
    gmm_fit = fit_power_gmm(
        power_values=train_power,
        k=int(selected_k),
        random_state=int(seed),
        n_init=10,
        max_iter=300,
        reg_covar=1e-6,
    )

    train_data = _prepare_supervised_split(
        bundle["raw"]["train"],
        norm_cfg=norm_cfg,
        feature_set=feature_set,
        gmm_fit=gmm_fit,
    )
    val_data = _prepare_supervised_split(
        bundle["raw"]["val"],
        norm_cfg=norm_cfg,
        feature_set=feature_set,
        gmm_fit=gmm_fit,
    )
    if not train_data or not val_data:
        raise ValueError("Unable to build labeled train/val data from the prepared bundle")

    return {
        "config_id": resolved_config_id,
        "input_dim": int(input_dim),
        "norm_payload": norm_payload,
        "train_data": train_data,
        "val_data": val_data,
        "gmm_fit": gmm_fit,
        "selected_k": int(selected_k),
        "k_selection_reason": str(k_selection_reason),
        "bic_scan": bic_scan,
    }


def run_prepare(
    *,
    config_id: str,
    experimental_manifest: Path,
    data_inventory: Path,
    throughput_db: Path,
    out_dir: Path,
    force: bool,
) -> Dict[str, object]:
    manifest_payload = _load_json(experimental_manifest)
    configs = manifest_payload.get("configs", {})
    if not isinstance(configs, dict):
        raise ValueError(f"Invalid experimental manifest format: {experimental_manifest}")

    cfg = configs.get(config_id)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config '{config_id}' not found in {experimental_manifest}")
    if not bool(cfg.get("written", False)):
        raise ValueError(f"Config '{config_id}' is not marked written in {experimental_manifest}")

    manifest_base = experimental_manifest.resolve().parent
    dataset_src = _resolve_existing_path(str(cfg.get("dataset_npz", "")), manifest_base)
    split_src = _resolve_existing_path(str(cfg.get("split_json", "")), manifest_base)
    norm_src = _resolve_existing_path(str(cfg.get("norm_params_json", "")), manifest_base)

    _ensure_dir(out_dir)

    dataset_dst = out_dir / "dataset.npz"
    split_dst = out_dir / "split.json"
    norm_dst = out_dir / "norm_params.json"
    throughput_dst = out_dir / "throughput_database.json"
    heldout_dst = out_dir / "heldout_requests.npz"
    bundle_dst = out_dir / "bundle_manifest.json"

    if (not force) and any(
        path.exists()
        for path in (dataset_dst, split_dst, norm_dst, throughput_dst, heldout_dst, bundle_dst)
    ):
        raise FileExistsError(
            f"Prepared artifacts already exist in {out_dir}. Re-run with --force to overwrite."
        )

    shutil.copy2(dataset_src, dataset_dst)
    shutil.copy2(split_src, split_dst)
    shutil.copy2(norm_src, norm_dst)

    throughput_payload = _load_json(throughput_db)
    throughput_configs = throughput_payload.get("configs", {})
    if not isinstance(throughput_configs, dict):
        raise ValueError(f"Invalid throughput DB format: {throughput_db}")
    throughput_row = throughput_configs.get(config_id)
    if not isinstance(throughput_row, dict):
        raise ValueError(f"Config '{config_id}' not found in {throughput_db}")

    filtered_throughput_payload = {
        "schema_version": str(throughput_payload.get("schema_version", "stage0-throughput-db")),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_throughput_db": str(throughput_db),
        "configs": {
            config_id: throughput_row,
        },
    }
    _write_json(throughput_dst, filtered_throughput_payload)

    with np.load(dataset_dst, allow_pickle=True) as data:
        pair_key_arr = np.asarray(data["pair_key"], dtype=object)
        rate_arr = np.asarray(data["rate"], dtype=object)

    split_payload = _load_json(split_dst)
    test_indices_raw = split_payload.get("test_indices", [])
    if not isinstance(test_indices_raw, list) or len(test_indices_raw) == 0:
        raise ValueError(f"Prepared split has no test indices: {split_dst}")
    test_indices = [int(idx) for idx in test_indices_raw]

    inventory_map = _load_inventory_json_path_map(data_inventory)

    heldout_trace_idx: List[int] = []
    heldout_pair_key: List[str] = []
    heldout_rate: List[str] = []
    heldout_arrivals: List[np.ndarray] = []
    heldout_input_tokens: List[np.ndarray] = []
    heldout_output_tokens: List[np.ndarray] = []

    for idx in test_indices:
        if idx < 0 or idx >= len(pair_key_arr):
            raise IndexError(f"Test index out of bounds for dataset: {idx}")
        pair_key = str(pair_key_arr[idx])
        json_path = inventory_map.get(pair_key)
        if json_path is None:
            raise KeyError(f"Missing heldout request JSON for pair_key '{pair_key}' in {data_inventory}")
        request_arrays = _extract_request_arrays(json_path)
        heldout_trace_idx.append(int(idx))
        heldout_pair_key.append(pair_key)
        heldout_rate.append(str(rate_arr[idx]) if idx < len(rate_arr) else "")
        heldout_arrivals.append(request_arrays["request_arrival_time_s"])
        heldout_input_tokens.append(request_arrays["input_tokens"])
        heldout_output_tokens.append(request_arrays["output_tokens"])

    np.savez(
        heldout_dst,
        trace_idx=np.asarray(heldout_trace_idx, dtype=np.int64),
        pair_key=np.asarray(heldout_pair_key, dtype=object),
        rate=np.asarray(heldout_rate, dtype=object),
        request_arrival_time_s=np.asarray(heldout_arrivals, dtype=object),
        input_tokens=np.asarray(heldout_input_tokens, dtype=object),
        output_tokens=np.asarray(heldout_output_tokens, dtype=object),
    )

    bundle_manifest = {
        "config_id": config_id,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": {
            "experimental_manifest": str(experimental_manifest),
            "data_inventory": str(data_inventory),
            "throughput_db": str(throughput_db),
        },
        "prepared": {
            "dataset_npz": str(dataset_dst),
            "split_json": str(split_dst),
            "norm_params_json": str(norm_dst),
            "throughput_database_json": str(throughput_dst),
            "heldout_requests_npz": str(heldout_dst),
        },
        "summary": {
            "num_test_traces": int(len(test_indices)),
            "test_indices": [int(idx) for idx in test_indices],
        },
    }
    _write_json(bundle_dst, bundle_manifest)
    return bundle_manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the isolated gpt-oss-20b_A100_tp1 training/eval bundle."
    )
    parser.add_argument("--config-id", default=DEFAULT_CONFIG_ID)
    parser.add_argument(
        "--source-experimental-manifest",
        default=str(DEFAULT_EXPERIMENTAL_MANIFEST),
    )
    parser.add_argument(
        "--source-data-inventory",
        default=str(DEFAULT_DATA_INVENTORY),
    )
    parser.add_argument(
        "--source-throughput-db",
        default=str(DEFAULT_THROUGHPUT_DB),
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    bundle = run_prepare(
        config_id=str(args.config_id),
        experimental_manifest=Path(args.source_experimental_manifest).resolve(),
        data_inventory=Path(args.source_data_inventory).resolve(),
        throughput_db=Path(args.source_throughput_db).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        force=bool(args.force),
    )
    print("[prepare] Done")
    print(f"config_id: {bundle['config_id']}")
    print(f"prepared_dir: {Path(args.out_dir).resolve()}")
    print(f"num_test_traces: {bundle['summary']['num_test_traces']}")


if __name__ == "__main__":
    main()
