from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from model.classifiers.gmm_bigru import fit_power_gmm, gmm_params_to_json_dict
from model.classifiers.gru import GRUClassifier
from model.pipeline.data_loading import (
    _compute_delta_stats,
    _prepare_split,
    _tensor_from_array,
    load_config_data,
)
from model.pipeline.gmm_fitting import (
    BRACKET_CONFIG_SUBSET,
    _fit_candidate_scores,
    _select_optimal_k,
)
from model.pipeline.manifest_validation import validate_manifest
from model.utils.config import (
    parse_config_ids as _parse_config_ids,
    resolve_device as _resolve_device,
)
from model.utils.io import (
    ensure_dir as _ensure_dir,
    safe_slug as _safe_slug,
    write_csv as _write_csv,
    write_json as _write_json,
)


def train_one_config(
    *,
    config_id: str,
    config_data: Dict[str, List[Dict[str, object]]],
    k: int = 10,
    input_dim: int = 2,
    hidden_dim: int = 64,
    num_layers: int = 1,
    n_epochs: int = 1000,
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
    np.random.default_rng(int(seed))

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
            x = _tensor_from_array(
                np.asarray(tr["features_norm"], dtype=np.float32),
                dtype=torch.float32,
                device=resolved_device,
            )
            y = _tensor_from_array(
                np.asarray(tr["state_labels"], dtype=np.int64),
                dtype=torch.int64,
                device=resolved_device,
            )
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
                x = _tensor_from_array(
                    np.asarray(tr["features_norm"], dtype=np.float32),
                    dtype=torch.float32,
                    device=resolved_device,
                )
                y = _tensor_from_array(
                    np.asarray(tr["state_labels"], dtype=np.int64),
                    dtype=torch.int64,
                    device=resolved_device,
                )
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
                best_state = {
                    k0: v.detach().cpu().clone() for k0, v in model.state_dict().items()
                }
        else:
            patience_counter += 1
            if patience_counter >= int(max(1, patience)):
                print(f"  [{config_id}] early stopping at epoch {epoch}")
                break

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            state = torch.load(
                checkpoint_path, map_location=resolved_device, weights_only=True
            )
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
    epochs: int = 1000,
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
    if feature_set == "f3":
        raise ValueError("feature_set='f3' is no longer supported; use 'f2'.")
    if feature_set != "f2":
        raise ValueError(f"feature_set must be 'f2'; got {feature_set}")
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    validate_manifest(manifest, "experimental_manifest")
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
            targets = sorted(
                [
                    cid
                    for cid, entry in config_map.items()
                    if bool(entry.get("written", False))
                ]
            )

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
            row = {
                "config_id": cid,
                "status": "skipped",
                "reason": err or "load_failed",
            }
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
                    "input_dim": 2,
                    "hidden_dim": int(hidden_dim),
                    "num_layers": int(max(1, num_layers)),
                    "seed": int(seed),
                }
            )

            norm_cfg = {
                "active_mean": float(
                    norm_payload.get("active_mean", norm_payload.get("A_mean", 0.0))
                ),
                "active_std": float(
                    norm_payload.get("active_std", norm_payload.get("A_std", 1.0))
                ),
                "t_arrive_log_mean": float(
                    norm_payload.get(
                        "t_arrive_log_mean", norm_payload.get("T_arrive_log_mean", 0.0)
                    )
                ),
                "t_arrive_log_std": float(
                    norm_payload.get(
                        "t_arrive_log_std", norm_payload.get("T_arrive_log_std", 1.0)
                    )
                ),
                "delta_A_mean": float(delta_mean),
                "delta_A_std": float(delta_std),
            }

            train_power = np.concatenate(
                [
                    np.asarray(tr["power_w"], dtype=np.float64).reshape(-1)
                    for tr in payload["raw"]["train"]
                ],
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
                print(
                    f"  [{cid}] auto-k selected K={selected_k} ({k_selection_reason})"
                )
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
            checkpoint_path = os.path.join(
                checkpoints_dir, f"{slug}_k{selected_k}_{feature_set}_best.pt"
            )
            curve_path = os.path.join(
                curves_dir, f"{slug}_k{selected_k}_{feature_set}.csv"
            )
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
                input_dim=2,
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
                "input_dim": 2,
                "seed": int(seed),
                "device": resolved_device,
                "best_epoch": int(train_result["best_epoch"]),
                "best_val_loss": float(train_result["best_val_loss"]),
                "final_train_loss": float(train_result["final_train_loss"]),
                "final_val_loss": float(train_result["final_val_loss"]),
                "num_train_traces": int(len(train_data)),
                "num_val_traces": int(len(val_data)),
                "num_test_traces": int(len(test_data)),
                "num_train_points": int(
                    np.sum([len(t["state_labels"]) for t in train_data])
                ),
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
