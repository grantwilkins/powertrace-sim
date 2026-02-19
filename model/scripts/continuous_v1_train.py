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

from model.classifiers.continuous_gru import MeanRevertingGRU, PowerNoiseInjector, mean_reverting_nll


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


def _extract_trace_arrays(x_obj: object, y_obj: object) -> Optional[Dict[str, np.ndarray]]:
    try:
        x = np.asarray(x_obj, dtype=np.float32)
        y = np.asarray(y_obj, dtype=np.float32).reshape(-1)
    except Exception:
        return None

    if x.ndim != 2 or x.shape[1] < 3:
        return None
    L = int(min(len(x), len(y)))
    if L <= 0:
        return None

    p_prev = x[:L, 0:1].astype(np.float32)
    features = x[:L, 1:3].astype(np.float32)
    p_target = y[:L].reshape(-1, 1).astype(np.float32)
    if not (
        np.all(np.isfinite(p_prev))
        and np.all(np.isfinite(features))
        and np.all(np.isfinite(p_target))
    ):
        return None

    return {
        "p_prev_norm": p_prev,
        "features_norm": features,
        "p_target_norm": p_target,
    }


def _build_split_traces(
    indices: Sequence[int],
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    pair_key: np.ndarray,
) -> List[Dict[str, object]]:
    traces: List[Dict[str, object]] = []
    n = int(min(len(x_norm), len(y_norm)))
    for idx in indices:
        i = int(idx)
        if i < 0 or i >= n:
            continue
        arrs = _extract_trace_arrays(x_norm[i], y_norm[i])
        if arrs is None:
            continue
        key = str(pair_key[i]) if i < len(pair_key) else f"trace-{i}"
        traces.append({"pair_key": key, **arrs})
    return traces


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

    train_traces = _build_split_traces(train_indices, x_norm, y_norm, pair_key)
    val_traces = _build_split_traces(val_indices, x_norm, y_norm, pair_key)
    test_traces = _build_split_traces(test_indices, x_norm, y_norm, pair_key)

    if len(train_traces) == 0:
        return None, "empty_train_split"
    if len(val_traces) == 0:
        return None, "empty_val_split"
    if len(test_traces) == 0:
        return None, "empty_test_split"

    return {
        "config_id": config_id,
        "dataset_path": dataset_path,
        "split_path": split_path,
        "norm_path": norm_path,
        "split_payload": split_payload,
        "norm_payload": norm_payload,
        "data": {
            "train": train_traces,
            "val": val_traces,
            "test": test_traces,
        },
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
    n_mix: int = 1,
    hidden_dim: int = 64,
    num_layers: int = 1,
    n_epochs: int = 500,
    lr: float = 1e-3,
    patience: int = 50,
    scheduler_patience: int = 20,
    scheduler_factor: float = 0.5,
    warmup_epochs: int = 100,
    ramp_epochs: int = 200,
    max_noise_std: float = 0.1,
    lambda_mu: float = 0.1,
    seed: int = 42,
    device: Optional[torch.device | str] = None,
    checkpoint_path: Optional[str] = None,
    curve_path: Optional[str] = None,
) -> Dict[str, object]:
    del config_norm  # kept for API consistency

    lambda_mu = float(lambda_mu)
    if lambda_mu < 0.0:
        raise ValueError(f"lambda_mu must be >= 0; got {lambda_mu}")

    resolved_device = _resolve_device(device)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    model = MeanRevertingGRU(
        input_dim=3,
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        n_mix=int(n_mix),
    ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=int(scheduler_patience),
        factor=float(scheduler_factor),
    )
    noise = PowerNoiseInjector(
        warmup_epochs=int(warmup_epochs),
        ramp_epochs=int(ramp_epochs),
        max_noise_std=float(max_noise_std),
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
            features = torch.from_numpy(np.asarray(trace["features_norm"], dtype=np.float32)).to(
                device=resolved_device, dtype=torch.float32
            )
            p_target = torch.from_numpy(np.asarray(trace["p_target_norm"], dtype=np.float32)).to(
                device=resolved_device, dtype=torch.float32
            )
            if p_prev.ndim != 2 or features.ndim != 2 or p_target.ndim != 2:
                continue
            if len(p_prev) == 0:
                continue

            p_prev = p_prev.unsqueeze(0)
            features = features.unsqueeze(0)
            p_target = p_target.unsqueeze(0)

            p_prev_noisy = noise(p_prev.clone(), epoch=epoch)
            x = torch.cat([p_prev_noisy, features], dim=-1)
            params, _ = model(x)
            parts = mean_reverting_nll(
                params,
                p_prev,
                p_target,
                n_mix=int(n_mix),
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
                sigma_accum.append(float(torch.exp(params["log_sigma"]).mean().item()))
                alpha_accum.append(float(params["alpha"].mean().item()))

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
                features = torch.from_numpy(np.asarray(trace["features_norm"], dtype=np.float32)).to(
                    device=resolved_device, dtype=torch.float32
                )
                p_target = torch.from_numpy(np.asarray(trace["p_target_norm"], dtype=np.float32)).to(
                    device=resolved_device, dtype=torch.float32
                )
                if p_prev.ndim != 2 or features.ndim != 2 or p_target.ndim != 2:
                    continue
                if len(p_prev) == 0:
                    continue
                p_prev = p_prev.unsqueeze(0)
                features = features.unsqueeze(0)
                p_target = p_target.unsqueeze(0)
                x = torch.cat([p_prev, features], dim=-1)
                params, _ = model(x)
                parts = mean_reverting_nll(
                    params,
                    p_prev,
                    p_target,
                    n_mix=int(n_mix),
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
        noise_std = float(noise.get_std(epoch))
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
                "noise_std": float(noise_std),
                "lr": float(lr_now),
            }
        )

        if epoch % 25 == 0 or epoch == (int(max(1, n_epochs)) - 1):
            print(
                f"  [{config_id}] epoch={epoch:4d} train_total={mean_train_total:.5f} val_total={mean_val_total:.5f} "
                f"train_nll={mean_train_nll:.5f} val_nll={mean_val_nll:.5f} "
                f"train_mu={mean_train_mu:.5f} val_mu={mean_val_mu:.5f} "
                f"sigma={mean_sigma:.5f} alpha={mean_alpha:.5f} noise={noise_std:.3f} lr={lr_now:.2e}"
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
                "noise_std",
                "lr",
            ],
        )

    final = history[-1] if history else {}
    return {
        "model": model,
        "history": history,
        "best_val_loss": float(best_val_total_loss),  # legacy alias for compatibility
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
    out_dir: str = "results/continuous_v1",
    config_ids: Optional[Sequence[str]] = None,
    n_mix: int = 1,
    hidden_dim: int = 64,
    num_layers: int = 1,
    epochs: int = 500,
    lr: float = 1e-3,
    patience: int = 50,
    scheduler_patience: int = 20,
    scheduler_factor: float = 0.5,
    warmup_epochs: int = 100,
    ramp_epochs: int = 200,
    max_noise_std: float = 0.1,
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
        checkpoint_path = os.path.join(checkpoints_dir, f"{slug}_mix{int(n_mix)}_best.pt")
        curve_path = os.path.join(curves_dir, f"{slug}_mix{int(n_mix)}.csv")
        norm_out_path = os.path.join(norms_dir, f"{slug}.json")

        norm_payload = dict(payload["norm_payload"])
        norm_payload.update(
            {
                "config_id": cid,
                "source_norm_path": payload["norm_path"],
                "n_mix": int(n_mix),
                "hidden_dim": int(hidden_dim),
                "num_layers": int(num_layers),
                "lambda_mu": float(lambda_mu),
                "seed": int(seed),
            }
        )
        _write_json(norm_out_path, norm_payload)

        try:
            train_result = train_one_config(
                config_id=cid,
                config_data=payload["data"],
                config_norm=payload["norm_payload"],
                n_mix=int(n_mix),
                hidden_dim=int(hidden_dim),
                num_layers=int(num_layers),
                n_epochs=int(epochs),
                lr=float(lr),
                patience=int(patience),
                scheduler_patience=int(scheduler_patience),
                scheduler_factor=float(scheduler_factor),
                warmup_epochs=int(warmup_epochs),
                ramp_epochs=int(ramp_epochs),
                max_noise_std=float(max_noise_std),
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
                "n_mix": int(n_mix),
                "lambda_mu": float(lambda_mu),
                "seed": int(seed),
                "device": resolved_device,
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
        "n_mix",
        "lambda_mu",
        "seed",
        "device",
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
        "schema_version": "continuous-v1-train-run-v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "manifest_path": manifest_path,
        },
        "defaults": {
            "out_dir": out_dir,
            "n_mix": int(n_mix),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "epochs": int(epochs),
            "lr": float(lr),
            "patience": int(patience),
            "scheduler_patience": int(scheduler_patience),
            "scheduler_factor": float(scheduler_factor),
            "warmup_epochs": int(warmup_epochs),
            "ramp_epochs": int(ramp_epochs),
            "max_noise_std": float(max_noise_std),
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
    parser = argparse.ArgumentParser(description="Stage 2+3 standalone trainer for experimental continuous v1 datasets.")
    parser.add_argument(
        "--manifest",
        default="results/experimental_continuous_v1/manifest.json",
        help="Path to experimental continuous v1 manifest.json.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/continuous_v1",
        help="Output directory for checkpoints and training artifacts.",
    )
    parser.add_argument(
        "--config-id",
        action="append",
        default=[],
        help="Optional config ID filter. Repeatable and/or comma-separated.",
    )
    parser.add_argument("--n-mix", type=int, default=1, choices=[1, 3])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--scheduler-patience", type=int, default=20)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--warmup-epochs", type=int, default=100)
    parser.add_argument("--ramp-epochs", type=int, default=200)
    parser.add_argument("--max-noise-std", type=float, default=0.1)
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
        n_mix=args.n_mix,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        warmup_epochs=args.warmup_epochs,
        ramp_epochs=args.ramp_epochs,
        max_noise_std=args.max_noise_std,
        lambda_mu=args.lambda_mu,
        seed=args.seed,
        device=args.device,
    )
    print("[continuous_v1_train] Summary:")
    for k, v in run_manifest.get("summary", {}).items():
        print(f"  {k}: {v}")
    print(f"  run_manifest: {os.path.join(args.out_dir, 'run_manifest.json')}")
    print(f"  run_summary : {os.path.join(args.out_dir, 'run_summary.csv')}")


if __name__ == "__main__":
    main()
