#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import numpy as np
import torch

from model.classifiers.gmm_bigru import (
    build_features_from_active,
    build_state_labels,
    fit_power_gmm,
    gmm_params_to_json_dict,
)
from model.classifiers.gru import GRUClassifier

BRACKET_CONFIG_SUBSET = [
    "deepseek-r1-distill-8b_A100_tp1",
    "deepseek-r1-distill-70b_H100_tp8",
    "gpt-oss-20b_A100_tp2",
    "gpt-oss-120b_H100_tp8",
    "llama-3-8b_A100_tp1",
    "llama-3-8b_H100_tp8",
    "llama-3-70b_A100_tp4",
    "llama-3-405b_H100_tp8",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text)


def _write_json(path: str, payload: Dict[str, object]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _resolve_device(device: Optional[torch.device | str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def _parse_k_candidates(csv_text: str) -> List[int]:
    out: List[int] = []
    for tok in str(csv_text).split(","):
        tok = tok.strip()
        if tok == "":
            continue
        val = int(tok)
        if val < 1:
            continue
        out.append(val)
    deduped: List[int] = []
    seen = set()
    for val in out:
        if val in seen:
            continue
        deduped.append(val)
        seen.add(val)
    return deduped


def _extract_t_arrive_log(data: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    if "t_arrive_log" in data:
        return np.asarray(data["t_arrive_log"], dtype=object)
    if "t_arrive" in data:
        arr = np.asarray(data["t_arrive"], dtype=object)
        out = []
        for x in arr:
            vals = np.asarray(x, dtype=np.float64).reshape(-1)
            out.append(np.log1p(np.clip(vals, a_min=0.0, a_max=None)))
        return np.asarray(out, dtype=object)
    return None


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
    try:
        power = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
        active = np.asarray(active_arr[idx], dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if power.size < 2 or active.size < 2:
        return None
    if not (np.all(np.isfinite(power)) and np.all(np.isfinite(active))):
        return None

    t_arrive_log: Optional[np.ndarray] = None
    if feature_set == "f3":
        if t_arrive_log_arr is None:
            return None
        try:
            t_arrive_log = np.asarray(t_arrive_log_arr[idx], dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if t_arrive_log.size < 2:
            return None
        if not np.all(np.isfinite(t_arrive_log)):
            return None

    pair_key = str(pair_key_arr[idx]) if idx < len(pair_key_arr) else f"trace-{idx}"
    return {
        "trace_idx": int(idx),
        "pair_key": pair_key,
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
        tr = _extract_raw_trace(
            idx=int(idx),
            pair_key_arr=pair_key_arr,
            power_arr=power_arr,
            active_arr=active_arr,
            t_arrive_log_arr=t_arrive_log_arr,
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
    dataset_path = _resolve_existing_path(str(config_manifest_entry.get("dataset_npz", "")), manifest_dir)
    split_path = _resolve_existing_path(str(config_manifest_entry.get("split_json", "")), manifest_dir)
    norm_path = _resolve_existing_path(str(config_manifest_entry.get("norm_params_json", "")), manifest_dir)
    if dataset_path is None:
        return None, "missing_dataset_npz"
    if split_path is None:
        return None, "missing_split_json"
    if norm_path is None:
        return None, "missing_norm_params_json"

    try:
        with open(split_path, "r") as f:
            split_payload = json.load(f)
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
                else np.asarray([f"trace-{i}" for i in range(len(data["power"]))], dtype=object)
            )
            power = np.asarray(data["power"], dtype=object)
            active = np.asarray(data["active_requests"], dtype=object)
            t_arrive_log = _extract_t_arrive_log(data)
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
        t_arrive_log_arr=t_arrive_log,
        feature_set=feature_set,
    )
    val_raw = _build_split_raw(
        indices=val_idx,
        pair_key_arr=pair_key,
        power_arr=power,
        active_arr=active,
        t_arrive_log_arr=t_arrive_log,
        feature_set=feature_set,
    )
    test_raw = _build_split_raw(
        indices=test_idx,
        pair_key_arr=pair_key,
        power_arr=power,
        active_arr=active,
        t_arrive_log_arr=t_arrive_log,
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
        "active_mean": float(source_norm.get("active_mean", source_norm.get("A_mean", 0.0))),
        "active_std": float(source_norm.get("active_std", source_norm.get("A_std", 1.0))),
        "t_arrive_log_mean": float(source_norm.get("t_arrive_log_mean", source_norm.get("T_arrive_log_mean", 0.0))),
        "t_arrive_log_std": float(source_norm.get("t_arrive_log_std", source_norm.get("T_arrive_log_std", 1.0))),
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
            t_arrive_log=np.asarray(t_log, dtype=np.float64) if t_log is not None else None,
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
            t_arrive_log=np.asarray(t_log, dtype=np.float64) if t_log is not None else None,
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


def _fit_candidate_scores(
    power_values: np.ndarray,
    *,
    k_candidates: Sequence[int],
    seed: int,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k in k_candidates:
        kc = int(k)
        if kc < 1:
            continue
        if power_values.size < kc:
            out[str(kc)] = {"aic": float("nan"), "bic": float("nan")}
            continue
        try:
            fitted = fit_power_gmm(
                power_values=power_values,
                k=kc,
                random_state=int(seed),
                n_init=10,
                max_iter=300,
                reg_covar=1e-6,
            )
            out[str(kc)] = {
                "aic": float(fitted["aic"]),
                "bic": float(fitted["bic"]),
            }
        except Exception:
            out[str(kc)] = {"aic": float("nan"), "bic": float("nan")}
    return out


def _select_optimal_k(
    bic_scores: Dict[str, Dict[str, float]],
    *,
    max_k: int = 20,
    min_k: int = 4,
) -> Tuple[int, str]:
    """
    Select optimal K based on BIC scores.

    Returns:
        Tuple of (selected_k, selection_reason)
    """
    valid_scores: Dict[int, float] = {}
    for k_str, scores in bic_scores.items():
        k_val = int(k_str)
        bic_val = scores.get("bic", float("nan"))
        if np.isfinite(bic_val) and min_k <= k_val <= max_k:
            valid_scores[k_val] = bic_val

    if not valid_scores:
        return max(min_k, 10), "no_valid_bic_scores_fallback"

    # Find K with minimum BIC
    best_k = min(valid_scores, key=lambda k: valid_scores[k])
    best_bic = valid_scores[best_k]

    # Check if BIC is still decreasing at max tested K (suggests higher K might be better)
    sorted_ks = sorted(valid_scores.keys())
    if len(sorted_ks) >= 2:
        max_tested_k = sorted_ks[-1]
        second_max_k = sorted_ks[-2]
        if best_k == max_tested_k and valid_scores[max_tested_k] < valid_scores[second_max_k]:
            reason = f"bic_minimum_at_max_k={best_k}_may_need_higher"
        else:
            reason = f"bic_minimum_at_k={best_k}"
    else:
        reason = f"bic_minimum_at_k={best_k}"

    return best_k, reason


def train_one_config(
    *,
    config_id: str,
    config_data: Dict[str, List[Dict[str, object]]],
    k: int = 10,
    input_dim: int = 2,
    hidden_dim: int = 64,
    num_layers: int = 1,
    n_epochs: int = 500,
    lr: float = 1e-3,
    patience: int = 50,
    scheduler_patience: int = 20,
    scheduler_factor: float = 0.5,
    seed: int = 42,
    device: Optional[torch.device | str] = None,
    checkpoint_path: Optional[str] = None,
    curve_path: Optional[str] = None,
) -> Dict[str, object]:
    k = int(k)
    input_dim = int(input_dim)
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if input_dim < 1:
        raise ValueError(f"input_dim must be >= 1; got {input_dim}")

    resolved_device = _resolve_device(device)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    model = GRUClassifier(
        Dx=input_dim,
        K=k,
        H=int(hidden_dim),
        num_layers=int(max(1, num_layers)),
    ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=int(scheduler_patience),
        factor=float(scheduler_factor),
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    patience_counter = 0
    history: List[Dict[str, float]] = []

    for epoch in range(int(max(1, n_epochs))):
        model.train()
        train_losses: List[float] = []
        for tr in config_data["train"]:
            x = torch.from_numpy(np.asarray(tr["features_norm"], dtype=np.float32)).to(resolved_device)
            y = torch.from_numpy(np.asarray(tr["state_labels"], dtype=np.int64)).to(resolved_device)
            if x.ndim != 2 or y.ndim != 1:
                continue
            if len(x) == 0 or len(y) != len(x):
                continue
            if x.shape[1] != input_dim:
                continue
            if int(y.max().item()) >= k or int(y.min().item()) < 0:
                continue

            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

            logits = model(x)
            loss = criterion(logits.reshape(-1, k), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for tr in config_data["val"]:
                x = torch.from_numpy(np.asarray(tr["features_norm"], dtype=np.float32)).to(resolved_device)
                y = torch.from_numpy(np.asarray(tr["state_labels"], dtype=np.int64)).to(resolved_device)
                if x.ndim != 2 or y.ndim != 1:
                    continue
                if len(x) == 0 or len(y) != len(x):
                    continue
                if x.shape[1] != input_dim:
                    continue
                if int(y.max().item()) >= k or int(y.min().item()) < 0:
                    continue

                x = x.unsqueeze(0)
                y = y.unsqueeze(0)
                logits = model(x)
                loss = criterion(logits.reshape(-1, k), y.reshape(-1))
                val_losses.append(float(loss.item()))

        mean_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        sched_metric = mean_val_loss if np.isfinite(mean_val_loss) else mean_train_loss
        if np.isfinite(sched_metric):
            scheduler.step(float(sched_metric))
        lr_now = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(mean_train_loss),
                "val_loss": float(mean_val_loss),
                "lr": float(lr_now),
            }
        )

        if epoch % 25 == 0 or epoch == (int(max(1, n_epochs)) - 1):
            print(
                f"  [{config_id}] epoch={epoch:4d} train_loss={mean_train_loss:.5f} "
                f"val_loss={mean_val_loss:.5f} lr={lr_now:.2e}"
            )

        if np.isfinite(mean_val_loss) and mean_val_loss < best_val_loss:
            best_val_loss = float(mean_val_loss)
            best_epoch = int(epoch)
            patience_counter = 0
            if checkpoint_path:
                _ensure_dir(os.path.dirname(checkpoint_path) or ".")
                torch.save(model.state_dict(), checkpoint_path)
            else:
                best_state = {k0: v.detach().cpu().clone() for k0, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= int(max(1, patience)):
                print(f"  [{config_id}] early stopping at epoch {epoch}")
                break

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            state = torch.load(checkpoint_path, map_location=resolved_device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint_path, map_location=resolved_device)
        model.load_state_dict(state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    if curve_path:
        _write_csv(
            curve_path,
            history,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "lr",
            ],
        )

    final = history[-1] if history else {}
    return {
        "model": model,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(final.get("train_loss", float("nan"))),
        "final_val_loss": float(final.get("val_loss", float("nan"))),
        "checkpoint_path": checkpoint_path or "",
        "curve_path": curve_path or "",
        "device": str(resolved_device),
    }


def run_training_from_manifest(
    *,
    manifest_path: str = "results/experimental_continuous_v1/manifest.json",
    out_root: str = "results/continuous_v1_gmm_bigru",
    config_ids: Optional[Sequence[str]] = None,
    k: int = 10,
    feature_set: str = "f2",
    hidden_dim: int = 64,
    num_layers: int = 1,
    epochs: int = 500,
    lr: float = 1e-3,
    patience: int = 50,
    scheduler_patience: int = 20,
    scheduler_factor: float = 0.5,
    bic_candidates: Sequence[int] = (6, 8, 10, 12, 14, 16, 18, 20),
    seed: int = 42,
    device: str = "auto",
    force_all_configs: bool = False,
    auto_k: bool = False,
    max_k: int = 20,
) -> Dict[str, object]:
    feature_set = str(feature_set).strip().lower()
    if feature_set not in {"f2", "f3"}:
        raise ValueError(f"feature_set must be one of {{'f2','f3'}}; got {feature_set}")
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    manifest_dir = str(Path(manifest_path).resolve().parent)
    config_map = dict(manifest.get("configs", {}))
    requested = _parse_config_ids(config_ids)

    if requested:
        targets: List[str] = []
        for cid in requested:
            entry = config_map.get(cid)
            if entry is None:
                continue
            if not bool(entry.get("written", False)):
                continue
            targets.append(cid)
    else:
        if (not bool(force_all_configs)) and feature_set == "f2" and int(k) in {8, 12}:
            targets = [
                cid
                for cid in BRACKET_CONFIG_SUBSET
                if cid in config_map and bool(config_map[cid].get("written", False))
            ]
        else:
            targets = sorted([cid for cid, entry in config_map.items() if bool(entry.get("written", False))])

    if auto_k:
        variant = f"kauto_max{int(max_k)}_{feature_set}"
    else:
        variant = f"k{int(k)}_{feature_set}"
    out_dir = os.path.join(out_root, variant)
    checkpoints_dir = os.path.join(out_dir, "checkpoints")
    curves_dir = os.path.join(out_dir, "training_curves")
    norms_dir = os.path.join(out_dir, "norm_params")
    gmms_dir = os.path.join(out_dir, "gmms")
    _ensure_dir(checkpoints_dir)
    _ensure_dir(curves_dir)
    _ensure_dir(norms_dir)
    _ensure_dir(gmms_dir)

    summary_rows: List[Dict[str, object]] = []
    config_results: Dict[str, Dict[str, object]] = {}
    resolved_device = str(_resolve_device(device))

    for cid in targets:
        entry = config_map[cid]
        payload, err = load_config_data(
            config_id=cid,
            config_manifest_entry=entry,
            manifest_dir=manifest_dir,
            feature_set=feature_set,
        )
        if payload is None:
            row = {"config_id": cid, "status": "skipped", "reason": err or "load_failed"}
            summary_rows.append(row)
            config_results[cid] = dict(row)
            continue

        try:
            delta_stats, d_err = _compute_delta_stats(
                payload["raw"]["train"],
                source_norm=payload["norm_payload"],
                feature_set=feature_set,
            )
            if delta_stats is None:
                raise ValueError(d_err or "delta_stats_failed")
            delta_mean, delta_std = delta_stats

            norm_payload = dict(payload["norm_payload"])
            # Note: k will be updated after BIC selection if auto_k is enabled
            norm_payload.update(
                {
                    "config_id": cid,
                    "source_norm_path": payload["norm_path"],
                    "delta_A_mean": float(delta_mean),
                    "delta_A_std": float(delta_std),
                    "feature_set": feature_set,
                    "input_dim": int(2 if feature_set == "f2" else 3),
                    "hidden_dim": int(hidden_dim),
                    "num_layers": int(max(1, num_layers)),
                    "seed": int(seed),
                }
            )

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

            train_prepared = _prepare_split(
                payload["raw"]["train"],
                norm=norm_cfg,
                feature_set=feature_set,
                gmm_fit=None,
            )
            val_prepared = _prepare_split(
                payload["raw"]["val"],
                norm=norm_cfg,
                feature_set=feature_set,
                gmm_fit=None,
            )
            test_prepared = _prepare_split(
                payload["raw"]["test"],
                norm=norm_cfg,
                feature_set=feature_set,
                gmm_fit=None,
            )
            if len(train_prepared) == 0:
                raise ValueError("empty_train_after_feature_build")
            if len(val_prepared) == 0:
                raise ValueError("empty_val_after_feature_build")
            if len(test_prepared) == 0:
                raise ValueError("empty_test_after_feature_build")

            train_power = np.concatenate(
                [np.asarray(tr["target_power_w"], dtype=np.float64).reshape(-1) for tr in train_prepared],
                axis=0,
            )
            if train_power.size < int(k):
                raise ValueError(f"insufficient_train_points_for_gmm_k{int(k)}")

            # Run BIC sweep first to evaluate all candidate K values
            bic_scan = _fit_candidate_scores(
                train_power,
                k_candidates=[int(v) for v in bic_candidates],
                seed=int(seed),
            )

            # Select K: either auto-select based on BIC or use provided fixed K
            if auto_k:
                selected_k, k_selection_reason = _select_optimal_k(
                    bic_scan,
                    max_k=int(max_k),
                    min_k=4,
                )
                print(f"  [{cid}] auto-k selected K={selected_k} ({k_selection_reason})")
            else:
                selected_k = int(k)
                k_selection_reason = "fixed"

            gmm_fit = fit_power_gmm(
                power_values=train_power,
                k=selected_k,
                random_state=int(seed),
                n_init=10,
                max_iter=300,
                reg_covar=1e-6,
            )

            train_data = _prepare_split(
                payload["raw"]["train"],
                norm=norm_cfg,
                feature_set=feature_set,
                gmm_fit=gmm_fit,
            )
            val_data = _prepare_split(
                payload["raw"]["val"],
                norm=norm_cfg,
                feature_set=feature_set,
                gmm_fit=gmm_fit,
            )
            test_data = _prepare_split(
                payload["raw"]["test"],
                norm=norm_cfg,
                feature_set=feature_set,
                gmm_fit=gmm_fit,
            )
            if len(train_data) == 0:
                raise ValueError("empty_train_after_label_build")
            if len(val_data) == 0:
                raise ValueError("empty_val_after_label_build")
            if len(test_data) == 0:
                raise ValueError("empty_test_after_label_build")

            slug = _safe_slug(cid)
            checkpoint_path = os.path.join(checkpoints_dir, f"{slug}_k{selected_k}_{feature_set}_best.pt")
            curve_path = os.path.join(curves_dir, f"{slug}_k{selected_k}_{feature_set}.csv")
            norm_out_path = os.path.join(norms_dir, f"{slug}.json")
            gmm_out_path = os.path.join(gmms_dir, f"{slug}_k{selected_k}.json")

            # Update norm_payload with selected K before writing
            norm_payload["k"] = selected_k
            norm_payload["k_selection_reason"] = k_selection_reason
            norm_payload["auto_k"] = bool(auto_k)
            _write_json(norm_out_path, norm_payload)
            _write_json(
                gmm_out_path,
                {
                    **gmm_params_to_json_dict(gmm_fit),
                    "config_id": cid,
                    "seed": int(seed),
                },
            )

            train_result = train_one_config(
                config_id=cid,
                config_data={
                    "train": train_data,
                    "val": val_data,
                    "test": test_data,
                },
                k=selected_k,
                input_dim=int(2 if feature_set == "f2" else 3),
                hidden_dim=int(hidden_dim),
                num_layers=int(max(1, num_layers)),
                n_epochs=int(epochs),
                lr=float(lr),
                patience=int(patience),
                scheduler_patience=int(scheduler_patience),
                scheduler_factor=float(scheduler_factor),
                seed=int(seed),
                device=resolved_device,
                checkpoint_path=checkpoint_path,
                curve_path=curve_path,
            )

            row = {
                "config_id": cid,
                "status": "trained",
                "reason": "",
                "k": selected_k,
                "k_selection_reason": k_selection_reason,
                "auto_k": bool(auto_k),
                "feature_set": feature_set,
                "input_dim": int(2 if feature_set == "f2" else 3),
                "seed": int(seed),
                "device": resolved_device,
                "best_epoch": int(train_result["best_epoch"]),
                "best_val_loss": float(train_result["best_val_loss"]),
                "final_train_loss": float(train_result["final_train_loss"]),
                "final_val_loss": float(train_result["final_val_loss"]),
                "num_train_traces": int(len(train_data)),
                "num_val_traces": int(len(val_data)),
                "num_test_traces": int(len(test_data)),
                "num_train_points": int(np.sum([len(t["state_labels"]) for t in train_data])),
                "gmm_aic": float(gmm_fit["aic"]),
                "gmm_bic": float(gmm_fit["bic"]),
                "bic_candidates": json.dumps(bic_scan, sort_keys=True),
                "checkpoint_path": checkpoint_path,
                "curve_path": curve_path,
                "norm_params_path": norm_out_path,
                "gmm_params_path": gmm_out_path,
            }
            summary_rows.append(row)
            config_results[cid] = {
                **row,
                "hidden_dim": int(hidden_dim),
                "num_layers": int(max(1, num_layers)),
                "bic_candidates": bic_scan,
            }
        except Exception as exc:
            row = {
                "config_id": cid,
                "status": "failed",
                "reason": f"{type(exc).__name__}:{exc}",
            }
            summary_rows.append(row)
            config_results[cid] = dict(row)

    summary_fields = [
        "config_id",
        "status",
        "reason",
        "k",
        "k_selection_reason",
        "auto_k",
        "feature_set",
        "input_dim",
        "seed",
        "device",
        "best_epoch",
        "best_val_loss",
        "final_train_loss",
        "final_val_loss",
        "num_train_traces",
        "num_val_traces",
        "num_test_traces",
        "num_train_points",
        "gmm_aic",
        "gmm_bic",
        "bic_candidates",
        "checkpoint_path",
        "curve_path",
        "norm_params_path",
        "gmm_params_path",
    ]
    for row in summary_rows:
        for field in summary_fields:
            row.setdefault(field, "")
    run_summary_path = os.path.join(out_dir, "run_summary.csv")
    _write_csv(run_summary_path, summary_rows, summary_fields)

    num_trained = int(sum(1 for r in summary_rows if r.get("status") == "trained"))
    num_skipped = int(sum(1 for r in summary_rows if r.get("status") == "skipped"))
    num_failed = int(sum(1 for r in summary_rows if r.get("status") == "failed"))
    run_manifest = {
        "schema_version": "continuous-v1-gmm-bigru-train-run-v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "manifest_path": manifest_path,
        },
        "defaults": {
            "out_root": out_root,
            "out_dir": out_dir,
            "variant": variant,
            "k": int(k),
            "auto_k": bool(auto_k),
            "max_k": int(max_k),
            "feature_set": feature_set,
            "hidden_dim": int(hidden_dim),
            "num_layers": int(max(1, num_layers)),
            "epochs": int(epochs),
            "lr": float(lr),
            "patience": int(patience),
            "scheduler_patience": int(scheduler_patience),
            "scheduler_factor": float(scheduler_factor),
            "bic_candidates": [int(v) for v in bic_candidates],
            "seed": int(seed),
            "device": resolved_device,
            "force_all_configs": bool(force_all_configs),
        },
        "summary": {
            "num_requested_configs": int(len(requested) if requested else len(targets)),
            "num_target_configs": int(len(targets)),
            "num_trained": int(num_trained),
            "num_skipped": int(num_skipped),
            "num_failed": int(num_failed),
        },
        "artifacts": {
            "run_summary_csv": run_summary_path,
            "checkpoints_dir": checkpoints_dir,
            "training_curves_dir": curves_dir,
            "norm_params_dir": norms_dir,
            "gmms_dir": gmms_dir,
        },
        "configs": config_results,
    }
    run_manifest_path = os.path.join(out_dir, "run_manifest.json")
    _write_json(run_manifest_path, run_manifest)
    return run_manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train continuous v1 GMM+BiGRU models from experimental manifest.")
    parser.add_argument("--manifest", default="results/experimental_continuous_v1/manifest.json")
    parser.add_argument("--out-root", default="results/continuous_v1_gmm_bigru")
    parser.add_argument("--config-id", action="append", default=[])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--feature-set", choices=["f2", "f3"], default="f2")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--scheduler-patience", type=int, default=20)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--bic-candidates", default="6,8,10,12,14,16,18,20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--auto-k",
        action="store_true",
        help="Automatically select K per config based on BIC minimum (up to --max-k).",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=20,
        help="Maximum K to consider when using --auto-k (default: 20).",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Disable phased default scoping and target all written configs.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_manifest = run_training_from_manifest(
        manifest_path=args.manifest,
        out_root=args.out_root,
        config_ids=args.config_id,
        k=args.k,
        feature_set=args.feature_set,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        bic_candidates=_parse_k_candidates(args.bic_candidates),
        seed=args.seed,
        device=args.device,
        force_all_configs=bool(args.all_configs),
        auto_k=bool(args.auto_k),
        max_k=args.max_k,
    )
    print("[train_gmm_bigru] Summary:")
    for k, v in run_manifest.get("summary", {}).items():
        print(f"  {k}: {v}")
    print(f"  run_manifest: {os.path.join(run_manifest['defaults']['out_dir'], 'run_manifest.json')}")
    print(f"  run_summary : {os.path.join(run_manifest['defaults']['out_dir'], 'run_summary.csv')}")


if __name__ == "__main__":
    main()
