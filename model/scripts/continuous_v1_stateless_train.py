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

from model.classifiers.stateless_mean_reverting import (
    StatelessMeanReverting,
    normalize_delta_active_requests,
    stateless_mean_reverting_nll,
)


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


def _extract_trace_arrays(
    x_obj: object,
    y_obj: object,
    active_obj: object,
) -> Optional[Dict[str, np.ndarray]]:
    try:
        x = np.asarray(x_obj, dtype=np.float32)
        y = np.asarray(y_obj, dtype=np.float32).reshape(-1)
        active = np.asarray(active_obj, dtype=np.float64).reshape(-1)
    except Exception:
        return None

    if x.ndim != 2 or x.shape[1] < 2:
        return None

    L = int(min(len(x), len(y), max(0, len(active) - 1)))
    if L <= 0:
        return None

    p_prev = x[:L, 0:1].astype(np.float32)
    A_norm = x[:L, 1:2].astype(np.float32)
    p_target = y[:L].reshape(-1, 1).astype(np.float32)
    delta_raw = (active[1 : L + 1] - active[:L]).reshape(-1, 1).astype(np.float64)

    if not (
        np.all(np.isfinite(p_prev))
        and np.all(np.isfinite(A_norm))
        and np.all(np.isfinite(p_target))
        and np.all(np.isfinite(delta_raw))
    ):
        return None

    return {
        "p_prev_norm": p_prev,
        "A_norm": A_norm,
        "p_target_norm": p_target,
        "delta_A_raw": delta_raw.astype(np.float32),
    }


def _build_split_traces(
    indices: Sequence[int],
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    active_requests: np.ndarray,
    pair_key: np.ndarray,
) -> List[Dict[str, object]]:
    traces: List[Dict[str, object]] = []
    n = int(min(len(x_norm), len(y_norm), len(active_requests)))
    for idx in indices:
        i = int(idx)
        if i < 0 or i >= n:
            continue
        arrs = _extract_trace_arrays(x_norm[i], y_norm[i], active_requests[i])
        if arrs is None:
            continue
        key = str(pair_key[i]) if i < len(pair_key) else f"trace-{i}"
        traces.append({"pair_key": key, **arrs})
    return traces


def _attach_delta_norm(config_data: Dict[str, List[Dict[str, object]]]) -> Tuple[Optional[Tuple[float, float]], Optional[str]]:
    train_deltas: List[np.ndarray] = []
    for tr in config_data.get("train", []):
        d = np.asarray(tr.get("delta_A_raw", []), dtype=np.float64).reshape(-1)
        if d.size == 0:
            continue
        finite = d[np.isfinite(d)]
        if finite.size > 0:
            train_deltas.append(finite)
    if len(train_deltas) == 0:
        return None, "empty_train_delta_A"

    cat = np.concatenate(train_deltas, axis=0)
    mean = float(np.mean(cat))
    std = float(np.std(cat) + 1e-6)

    for split in ("train", "val", "test"):
        for tr in config_data.get(split, []):
            d_raw = np.asarray(tr["delta_A_raw"], dtype=np.float64).reshape(-1)
            d_norm = normalize_delta_active_requests(d_raw, mean=mean, std=std).reshape(-1, 1)
            tr["delta_A_norm"] = d_norm.astype(np.float32)
    return (mean, std), None


def load_config_data(
    config_id: str,
    config_manifest_entry: Dict[str, object],
    *,
    manifest_dir: str,
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
            x_norm = np.asarray(data["x_norm"], dtype=object)
            y_norm = np.asarray(data["y_norm"], dtype=object)
            active_requests = np.asarray(data["active_requests"], dtype=object)
            pair_key = (
                np.asarray(data["pair_key"], dtype=object)
                if "pair_key" in data
                else np.asarray([f"trace-{i}" for i in range(len(x_norm))], dtype=object)
            )
    except Exception as exc:
        return None, f"dataset_npz_error:{type(exc).__name__}"

    train_indices = [int(i) for i in split_payload.get("train_indices", [])]
    val_indices = [int(i) for i in split_payload.get("val_indices", [])]
    test_indices = [int(i) for i in split_payload.get("test_indices", [])]

    train_traces = _build_split_traces(train_indices, x_norm, y_norm, active_requests, pair_key)
    val_traces = _build_split_traces(val_indices, x_norm, y_norm, active_requests, pair_key)
    test_traces = _build_split_traces(test_indices, x_norm, y_norm, active_requests, pair_key)

    if len(train_traces) == 0:
        return None, "empty_train_split"
    if len(val_traces) == 0:
        return None, "empty_val_split"
    if len(test_traces) == 0:
        return None, "empty_test_split"

    config_data = {
        "train": train_traces,
        "val": val_traces,
        "test": test_traces,
    }
    delta_stats, delta_err = _attach_delta_norm(config_data)
    if delta_stats is None:
        return None, delta_err or "delta_norm_failed"

    delta_A_mean, delta_A_std = delta_stats
    return {
        "config_id": config_id,
        "dataset_path": dataset_path,
        "split_path": split_path,
        "norm_path": norm_path,
        "split_payload": split_payload,
        "norm_payload": norm_payload,
        "delta_A_mean": float(delta_A_mean),
        "delta_A_std": float(delta_A_std),
        "data": config_data,
    }, None


def _resolve_device(device: Optional[torch.device | str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def train_one_config(
    *,
    config_id: str,
    config_data: Dict[str, List[Dict[str, object]]],
    config_norm: Dict[str, float],
    hidden_dim: int = 16,
    n_epochs: int = 500,
    lr: float = 1e-3,
    patience: int = 50,
    scheduler_patience: int = 20,
    scheduler_factor: float = 0.5,
    lambda_mu: float = 0.1,
    seed: int = 42,
    device: Optional[torch.device | str] = None,
    checkpoint_path: Optional[str] = None,
    curve_path: Optional[str] = None,
) -> Dict[str, object]:
    del config_norm  # kept for API compatibility

    lambda_mu = float(lambda_mu)
    if lambda_mu < 0.0:
        raise ValueError(f"lambda_mu must be >= 0; got {lambda_mu}")

    resolved_device = _resolve_device(device)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    model = StatelessMeanReverting(hidden_dim=int(hidden_dim)).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=int(scheduler_patience),
        factor=float(scheduler_factor),
    )

    best_val_total_loss = float("inf")
    best_val_nll = float("inf")
    best_val_mu_loss = float("inf")
    best_epoch = -1
    best_state = None
    patience_counter = 0
    history: List[Dict[str, float]] = []

    for epoch in range(int(max(1, n_epochs))):
        model.train()
        train_total_losses: List[float] = []
        train_nll_losses: List[float] = []
        train_mu_losses: List[float] = []
        sigma_accum: List[float] = []
        alpha_accum: List[float] = []

        for trace in config_data["train"]:
            p_prev = torch.from_numpy(np.asarray(trace["p_prev_norm"], dtype=np.float32)).to(
                device=resolved_device, dtype=torch.float32
            )
            A = torch.from_numpy(np.asarray(trace["A_norm"], dtype=np.float32)).to(
                device=resolved_device, dtype=torch.float32
            )
            dA = torch.from_numpy(np.asarray(trace["delta_A_norm"], dtype=np.float32)).to(
                device=resolved_device, dtype=torch.float32
            )
            p_target = torch.from_numpy(np.asarray(trace["p_target_norm"], dtype=np.float32)).to(
                device=resolved_device, dtype=torch.float32
            )
            if p_prev.ndim != 2 or A.ndim != 2 or dA.ndim != 2 or p_target.ndim != 2:
                continue
            if len(p_prev) == 0:
                continue

            p_prev = p_prev.unsqueeze(0)
            A = A.unsqueeze(0)
            dA = dA.unsqueeze(0)
            p_target = p_target.unsqueeze(0)

            mu, alpha, log_sigma = model(A, dA)
            parts = stateless_mean_reverting_nll(
                mu=mu,
                alpha=alpha,
                log_sigma=log_sigma,
                p_prev=p_prev,
                p_target=p_target,
                lambda_mu=lambda_mu,
                return_parts=True,
            )
            loss = parts["total_loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_total_losses.append(float(parts["total_loss"].item()))
            train_nll_losses.append(float(parts["nll_loss"].item()))
            train_mu_losses.append(float(parts["mu_loss"].item()))
            with torch.no_grad():
                sigma_accum.append(float(torch.exp(log_sigma).mean().item()))
                alpha_accum.append(float(alpha.mean().item()))

        mean_train_total = float(np.mean(train_total_losses)) if train_total_losses else float("nan")
        mean_train_nll = float(np.mean(train_nll_losses)) if train_nll_losses else float("nan")
        mean_train_mu = float(np.mean(train_mu_losses)) if train_mu_losses else float("nan")
        mean_sigma = float(np.mean(sigma_accum)) if sigma_accum else float("nan")
        mean_alpha = float(np.mean(alpha_accum)) if alpha_accum else float("nan")

        model.eval()
        val_total_losses: List[float] = []
        val_nll_losses: List[float] = []
        val_mu_losses: List[float] = []
        with torch.no_grad():
            for trace in config_data["val"]:
                p_prev = torch.from_numpy(np.asarray(trace["p_prev_norm"], dtype=np.float32)).to(
                    device=resolved_device, dtype=torch.float32
                )
                A = torch.from_numpy(np.asarray(trace["A_norm"], dtype=np.float32)).to(
                    device=resolved_device, dtype=torch.float32
                )
                dA = torch.from_numpy(np.asarray(trace["delta_A_norm"], dtype=np.float32)).to(
                    device=resolved_device, dtype=torch.float32
                )
                p_target = torch.from_numpy(np.asarray(trace["p_target_norm"], dtype=np.float32)).to(
                    device=resolved_device, dtype=torch.float32
                )
                if p_prev.ndim != 2 or A.ndim != 2 or dA.ndim != 2 or p_target.ndim != 2:
                    continue
                if len(p_prev) == 0:
                    continue

                p_prev = p_prev.unsqueeze(0)
                A = A.unsqueeze(0)
                dA = dA.unsqueeze(0)
                p_target = p_target.unsqueeze(0)

                mu, alpha, log_sigma = model(A, dA)
                parts = stateless_mean_reverting_nll(
                    mu=mu,
                    alpha=alpha,
                    log_sigma=log_sigma,
                    p_prev=p_prev,
                    p_target=p_target,
                    lambda_mu=lambda_mu,
                    return_parts=True,
                )
                val_total_losses.append(float(parts["total_loss"].item()))
                val_nll_losses.append(float(parts["nll_loss"].item()))
                val_mu_losses.append(float(parts["mu_loss"].item()))

        mean_val_total = float(np.mean(val_total_losses)) if val_total_losses else float("nan")
        mean_val_nll = float(np.mean(val_nll_losses)) if val_nll_losses else float("nan")
        mean_val_mu = float(np.mean(val_mu_losses)) if val_mu_losses else float("nan")
        sched_metric = mean_val_total if np.isfinite(mean_val_total) else mean_train_total
        if np.isfinite(sched_metric):
            scheduler.step(float(sched_metric))

        lr_now = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "train_total_loss": float(mean_train_total),
                "train_nll": float(mean_train_nll),
                "train_mu_loss": float(mean_train_mu),
                "val_total_loss": float(mean_val_total),
                "val_nll": float(mean_val_nll),
                "val_mu_loss": float(mean_val_mu),
                "mean_sigma": float(mean_sigma),
                "mean_alpha": float(mean_alpha),
                "lr": float(lr_now),
            }
        )

        if epoch % 25 == 0 or epoch == (int(max(1, n_epochs)) - 1):
            print(
                f"  [{config_id}] epoch={epoch:4d} train_total={mean_train_total:.5f} val_total={mean_val_total:.5f} "
                f"train_nll={mean_train_nll:.5f} val_nll={mean_val_nll:.5f} "
                f"train_mu={mean_train_mu:.5f} val_mu={mean_val_mu:.5f} "
                f"sigma={mean_sigma:.5f} alpha={mean_alpha:.5f} lr={lr_now:.2e}"
            )

        if np.isfinite(mean_val_total) and mean_val_total < best_val_total_loss:
            best_val_total_loss = float(mean_val_total)
            best_val_nll = float(mean_val_nll)
            best_val_mu_loss = float(mean_val_mu)
            best_epoch = int(epoch)
            patience_counter = 0
            if checkpoint_path:
                _ensure_dir(os.path.dirname(checkpoint_path) or ".")
                torch.save(model.state_dict(), checkpoint_path)
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= int(max(1, patience)):
                print(f"  [{config_id}] early stopping at epoch {epoch}")
                break

    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=resolved_device))
    elif best_state is not None:
        model.load_state_dict(best_state)

    if curve_path:
        _write_csv(
            curve_path,
            history,
            fieldnames=[
                "epoch",
                "train_total_loss",
                "train_nll",
                "train_mu_loss",
                "val_total_loss",
                "val_nll",
                "val_mu_loss",
                "mean_sigma",
                "mean_alpha",
                "lr",
            ],
        )

    final = history[-1] if history else {}
    return {
        "model": model,
        "history": history,
        "best_val_loss": float(best_val_total_loss),
        "best_val_total_loss": float(best_val_total_loss),
        "best_val_nll": float(best_val_nll),
        "best_val_mu_loss": float(best_val_mu_loss),
        "best_epoch": int(best_epoch),
        "final_train_total_loss": float(final.get("train_total_loss", float("nan"))),
        "final_val_total_loss": float(final.get("val_total_loss", float("nan"))),
        "final_train_nll": float(final.get("train_nll", float("nan"))),
        "final_val_nll": float(final.get("val_nll", float("nan"))),
        "final_train_mu_loss": float(final.get("train_mu_loss", float("nan"))),
        "final_val_mu_loss": float(final.get("val_mu_loss", float("nan"))),
        "final_mean_sigma": float(final.get("mean_sigma", float("nan"))),
        "final_mean_alpha": float(final.get("mean_alpha", float("nan"))),
        "lambda_mu": float(lambda_mu),
        "checkpoint_path": checkpoint_path or "",
        "curve_path": curve_path or "",
        "device": str(resolved_device),
    }


def run_training_from_manifest(
    *,
    manifest_path: str = "results/experimental_continuous_v1/manifest.json",
    out_dir: str = "results/continuous_v1_stateless",
    config_ids: Optional[Sequence[str]] = None,
    hidden_dim: int = 16,
    epochs: int = 500,
    lr: float = 1e-3,
    patience: int = 50,
    scheduler_patience: int = 20,
    scheduler_factor: float = 0.5,
    lambda_mu: float = 0.1,
    seed: int = 42,
    device: str = "auto",
) -> Dict[str, object]:
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    manifest_dir = str(Path(manifest_path).resolve().parent)
    config_map = dict(manifest.get("configs", {}))
    requested = _parse_config_ids(config_ids)

    targets: List[str] = []
    summary_rows: List[Dict[str, object]] = []
    if requested:
        for cid in requested:
            entry = config_map.get(cid)
            if entry is None:
                summary_rows.append({"config_id": cid, "status": "skipped", "reason": "config_not_in_manifest"})
                continue
            if not bool(entry.get("written", False)):
                summary_rows.append({"config_id": cid, "status": "skipped", "reason": "config_not_written"})
                continue
            targets.append(cid)
    else:
        targets = sorted([cid for cid, entry in config_map.items() if bool(entry.get("written", False))])

    checkpoints_dir = os.path.join(out_dir, "checkpoints")
    curves_dir = os.path.join(out_dir, "training_curves")
    norms_dir = os.path.join(out_dir, "norm_params")
    _ensure_dir(checkpoints_dir)
    _ensure_dir(curves_dir)
    _ensure_dir(norms_dir)

    config_results: Dict[str, Dict[str, object]] = {}
    resolved_device = str(_resolve_device(device))

    for cid in targets:
        entry = config_map[cid]
        payload, err = load_config_data(cid, entry, manifest_dir=manifest_dir)
        if payload is None:
            row = {"config_id": cid, "status": "skipped", "reason": err or "load_failed"}
            summary_rows.append(row)
            config_results[cid] = dict(row)
            continue

        slug = _safe_slug(cid)
        checkpoint_path = os.path.join(checkpoints_dir, f"{slug}_stateless_best.pt")
        curve_path = os.path.join(curves_dir, f"{slug}_stateless.csv")
        norm_out_path = os.path.join(norms_dir, f"{slug}.json")

        norm_payload = dict(payload["norm_payload"])
        norm_payload.update(
            {
                "config_id": cid,
                "source_norm_path": payload["norm_path"],
                "model_kind": "stateless",
                "hidden_dim": int(hidden_dim),
                "lambda_mu": float(lambda_mu),
                "seed": int(seed),
                "delta_A_mean": float(payload["delta_A_mean"]),
                "delta_A_std": float(payload["delta_A_std"]),
            }
        )
        _write_json(norm_out_path, norm_payload)

        try:
            train_result = train_one_config(
                config_id=cid,
                config_data=payload["data"],
                config_norm=payload["norm_payload"],
                hidden_dim=int(hidden_dim),
                n_epochs=int(epochs),
                lr=float(lr),
                patience=int(patience),
                scheduler_patience=int(scheduler_patience),
                scheduler_factor=float(scheduler_factor),
                lambda_mu=float(lambda_mu),
                seed=int(seed),
                device=resolved_device,
                checkpoint_path=checkpoint_path,
                curve_path=curve_path,
            )
            row = {
                "config_id": cid,
                "status": "trained",
                "reason": "",
                "model_kind": "stateless",
                "hidden_dim": int(hidden_dim),
                "lambda_mu": float(lambda_mu),
                "seed": int(seed),
                "device": resolved_device,
                "delta_A_mean": float(payload["delta_A_mean"]),
                "delta_A_std": float(payload["delta_A_std"]),
                "best_epoch": int(train_result["best_epoch"]),
                "best_val_total_loss": float(train_result["best_val_total_loss"]),
                "best_val_nll": float(train_result["best_val_nll"]),
                "best_val_mu_loss": float(train_result["best_val_mu_loss"]),
                "final_train_total_loss": float(train_result["final_train_total_loss"]),
                "final_val_total_loss": float(train_result["final_val_total_loss"]),
                "final_train_nll": float(train_result["final_train_nll"]),
                "final_val_nll": float(train_result["final_val_nll"]),
                "final_train_mu_loss": float(train_result["final_train_mu_loss"]),
                "final_val_mu_loss": float(train_result["final_val_mu_loss"]),
                "final_mean_sigma": float(train_result["final_mean_sigma"]),
                "final_mean_alpha": float(train_result["final_mean_alpha"]),
                "num_train_traces": int(len(payload["data"]["train"])),
                "num_val_traces": int(len(payload["data"]["val"])),
                "num_test_traces": int(len(payload["data"]["test"])),
                "checkpoint_path": checkpoint_path,
                "curve_path": curve_path,
                "norm_params_path": norm_out_path,
            }
            summary_rows.append(row)
            config_results[cid] = dict(row)
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
        "model_kind",
        "hidden_dim",
        "lambda_mu",
        "seed",
        "device",
        "delta_A_mean",
        "delta_A_std",
        "best_epoch",
        "best_val_total_loss",
        "best_val_nll",
        "best_val_mu_loss",
        "final_train_total_loss",
        "final_val_total_loss",
        "final_train_nll",
        "final_val_nll",
        "final_train_mu_loss",
        "final_val_mu_loss",
        "final_mean_sigma",
        "final_mean_alpha",
        "num_train_traces",
        "num_val_traces",
        "num_test_traces",
        "checkpoint_path",
        "curve_path",
        "norm_params_path",
    ]
    for row in summary_rows:
        for f in summary_fields:
            row.setdefault(f, "")

    run_summary_path = os.path.join(out_dir, "run_summary.csv")
    _write_csv(run_summary_path, summary_rows, fieldnames=summary_fields)

    num_trained = int(sum(1 for r in summary_rows if r["status"] == "trained"))
    num_skipped = int(sum(1 for r in summary_rows if r["status"] == "skipped"))
    num_failed = int(sum(1 for r in summary_rows if r["status"] == "failed"))
    run_manifest = {
        "schema_version": "continuous-v1-stateless-train-run-v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "manifest_path": manifest_path,
        },
        "defaults": {
            "out_dir": out_dir,
            "model_kind": "stateless",
            "hidden_dim": int(hidden_dim),
            "epochs": int(epochs),
            "lr": float(lr),
            "patience": int(patience),
            "scheduler_patience": int(scheduler_patience),
            "scheduler_factor": float(scheduler_factor),
            "lambda_mu": float(lambda_mu),
            "seed": int(seed),
            "device": resolved_device,
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
        },
        "configs": config_results,
    }
    run_manifest_path = os.path.join(out_dir, "run_manifest.json")
    _write_json(run_manifest_path, run_manifest)
    return run_manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train stateless mean-reverting models from experimental continuous v1 data.")
    parser.add_argument(
        "--manifest",
        default="results/experimental_continuous_v1/manifest.json",
        help="Path to experimental continuous v1 manifest.json.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/continuous_v1_stateless",
        help="Output directory for checkpoints and training artifacts.",
    )
    parser.add_argument(
        "--config-id",
        action="append",
        default=[],
        help="Optional config ID filter. Repeatable and/or comma-separated.",
    )
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--scheduler-patience", type=int, default=20)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lambda-mu", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device (auto, cpu, cuda, cuda:0, ...).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_manifest = run_training_from_manifest(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        config_ids=args.config_id,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        lambda_mu=args.lambda_mu,
        seed=args.seed,
        device=args.device,
    )
    print("[continuous_v1_stateless_train] Summary:")
    for k, v in run_manifest.get("summary", {}).items():
        print(f"  {k}: {v}")
    print(f"  run_manifest: {os.path.join(args.out_dir, 'run_manifest.json')}")
    print(f"  run_summary : {os.path.join(args.out_dir, 'run_summary.csv')}")


if __name__ == "__main__":
    main()
