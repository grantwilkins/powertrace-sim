from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

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
from model.classifiers.metrics import (
    compute_aggregate_power_metrics,
    compute_power_metrics,
)
from model.pipeline.request_builder import (
    _build_requests_from_stage0_json,
    _load_pair_manifest_map,
    _synthesize_request_timestamps,
)
from model.pipeline.artifact_resolution import (
    resolve_checkpoint_norm_gmm_paths,
    resolve_experimental_paths,
    resolve_throughput,
)
from model.pipeline.manifest_validation import validate_manifest
from model.utils.config import (
    parse_config_ids as _parse_config_ids,
    resolve_device as _resolve_device,
)
from model.utils.io import (
    ensure_dir as _ensure_dir,
    load_json as _load_json,
    safe_slug as _safe_slug,
    write_csv as _write_csv,
    write_json as _write_json,
)

def _nanmedian(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _total_energy_from_trace(power_w: np.ndarray, *, dt: float) -> float:
    arr = np.asarray(power_w, dtype=np.float64).reshape(-1)
    return float(np.sum(arr) * float(dt))


def _detect_first_power_spike(
    power_trace: np.ndarray,
    *,
    active_threshold: float = 250.0,
    window_bins: int = 3,
) -> int:
    arr = np.asarray(power_trace, dtype=np.float64).reshape(-1)
    if arr.size < int(window_bins):
        return 0
    w = int(max(1, window_bins))
    for i in range(0, int(arr.size) - w + 1):
        if np.all(arr[i : i + w] >= float(active_threshold)):
            return int(i)
    above = np.where(arr >= float(active_threshold))[0]
    return int(above[0]) if above.size > 0 else 0


def _detect_first_at_activation(a_t: np.ndarray) -> int:
    arr = np.asarray(a_t, dtype=np.float64).reshape(-1)
    nonzero = np.where(arr > 1e-9)[0]
    return int(nonzero[0]) if nonzero.size > 0 else 0


def _estimate_request_alignment_offset_seconds(
    *,
    power_trace: np.ndarray,
    a_t: np.ndarray,
    dt: float,
    active_threshold: float = 250.0,
    window_bins: int = 3,
) -> float:
    power_spike_bin = _detect_first_power_spike(
        power_trace,
        active_threshold=float(active_threshold),
        window_bins=int(window_bins),
    )
    at_spike_bin = _detect_first_at_activation(a_t)
    offset_bins = int(power_spike_bin - at_spike_bin)
    return float(offset_bins) * float(dt)


def _plot_overlay(path: str, *, dt: float, gt: np.ndarray, pred: np.ndarray, title: str) -> None:
    n = int(min(len(gt), len(pred)))
    t = np.arange(n, dtype=np.float64) * float(dt)
    fig, ax = plt.subplots(figsize=(12, 4))
    try:
        ax.plot(t, gt[:n], label="Measured", linewidth=1.5)
        ax.plot(t, pred[:n], label="Generated", linewidth=1.2, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (W)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _plot_ar1_params(
    path: str,
    *,
    gmm_means: np.ndarray,
    phi: np.ndarray,
    sigma_marginal: np.ndarray,
    sigma_innov: np.ndarray,
    title: str,
    phi_threshold: float = AR1_PHI_THRESHOLD,
) -> None:
    means = np.asarray(gmm_means, dtype=np.float64).reshape(-1)
    phi_arr = np.asarray(phi, dtype=np.float64).reshape(-1)
    sigma_m = np.asarray(sigma_marginal, dtype=np.float64).reshape(-1)
    sigma_i = np.asarray(sigma_innov, dtype=np.float64).reshape(-1)
    K = int(means.size)
    if phi_arr.size != K or sigma_m.size != K or sigma_i.size != K:
        raise ValueError("AR(1) plot parameter size mismatch")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    try:
        x = np.arange(K, dtype=np.int64)

        threshold = float(phi_threshold)
        colors = ["#d62728" if p >= threshold else "#2ca02c" for p in phi_arr]
        ax1.bar(x, phi_arr, color=colors, alpha=0.8)
        ax1.set_xlabel("GMM State (sorted by mean power)")
        ax1.set_ylabel("phi (AR(1) persistence)")
        ax1.set_title("Within-state persistence")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{int(m)}W" for m in means], rotation=45, fontsize=8)
        ax1.axhline(y=threshold, color="gray", linestyle="--", alpha=0.5, label=f"phi={threshold:.1f} threshold")
        ax1.legend(fontsize=8)
        ax1.set_ylim(0.0, 1.0)

        ax2.bar(x - 0.15, sigma_m, width=0.3, label="sigma_marginal", alpha=0.8)
        ax2.bar(x + 0.15, sigma_i, width=0.3, label="sigma_innovation", alpha=0.8)
        ax2.set_xlabel("GMM State")
        ax2.set_ylabel("Std Dev (W)")
        ax2.set_title("Marginal vs Innovation Noise")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{int(m)}W" for m in means], rotation=45, fontsize=8)
        ax2.legend(fontsize=8)

        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _build_trace_record(
    *,
    trace_idx: int,
    pair_key: str,
    rate: str,
    power_start_epoch_s: float,
    power: np.ndarray,
    dt: float,
) -> Dict[str, Any]:
    p = np.asarray(power, dtype=np.float64).reshape(-1)
    if p.size < 2:
        raise ValueError(f"Trace {trace_idx} has length < 2")
    L = int(len(p) - 1)
    if L <= 0:
        raise ValueError(f"Trace {trace_idx} has no aligned points")
    return {
        "trace_idx": int(trace_idx),
        "pair_key": str(pair_key),
        "rate": str(rate),
        "power_start_epoch_s": float(power_start_epoch_s),
        "power": p[: L + 1],
        "ground_truth": p[1 : L + 1],
        "p0": float(p[0]),
        "dt": float(dt),
        "num_points": int(L),
    }


def evaluate_from_artifacts(
    *,
    run_manifest: str = "results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json",
    experimental_manifest: str = "results/experimental_continuous_v1/manifest.json",
    throughput_db: str = "model/config/throughput_database.json",
    pair_manifest_csv: str = "results/stage0/pair_manifest.csv",
    out_dir: str = "results/continuous_v1_gmm_bigru/k10_f2/eval_metrics",
    config_ids: Optional[Sequence[str]] = None,
    num_seeds: int = 5,
    base_seed: int = 42,
    device: str = "auto",
    acf_max_lag: int = 50,
    generation_mode: str = "ar1_thresholded",
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    plots: bool = True,
) -> Dict[str, object]:
    if int(num_seeds) <= 0:
        raise ValueError("num_seeds must be >= 1")
    generation_mode_resolved = str(generation_mode).strip().lower()
    if generation_mode_resolved not in {"iid", "ar1", "ar1_thresholded"}:
        raise ValueError(
            "generation_mode must be one of {'iid', 'ar1', 'ar1_thresholded'}"
        )
    if decode_mode not in {"stochastic", "argmax"}:
        raise ValueError(f"decode_mode must be one of {{'stochastic','argmax'}}; got {decode_mode}")

    run_manifest_payload = _load_json(run_manifest)
    validate_manifest(run_manifest_payload, "run_manifest")
    run_cfgs = run_manifest_payload.get("configs", {})
    run_manifest_base = str(Path(run_manifest).resolve().parent)

    experimental_payload = _load_json(experimental_manifest)
    validate_manifest(experimental_payload, "experimental_manifest")
    experimental_base = str(Path(experimental_manifest).resolve().parent)

    throughput_payload = _load_json(throughput_db)
    validate_manifest(throughput_payload, "throughput_db")
    pair_map = _load_pair_manifest_map(pair_manifest_csv)

    requested = _parse_config_ids(config_ids)
    if requested:
        targets = requested
    else:
        targets = sorted([cid for cid, row in run_cfgs.items() if isinstance(row, dict) and row.get("status") == "trained"])

    resolved_device = _resolve_device(device)
    _ensure_dir(out_dir)
    plots_dir = os.path.join(out_dir, "plots")
    _ensure_dir(plots_dir)
    ar1_params_dir = os.path.join(str(Path(out_dir).parent), "ar1_params")
    _ensure_dir(ar1_params_dir)

    per_seed_rows: List[Dict[str, object]] = []
    per_trace_rows: List[Dict[str, object]] = []
    per_config_seed_rows: List[Dict[str, object]] = []
    config_rows: List[Dict[str, object]] = []
    config_results: Dict[str, Dict[str, object]] = {}
    seeds = [int(base_seed) + i for i in range(int(num_seeds))]

    for config_id in targets:
        row = run_cfgs.get(config_id)
        if not isinstance(row, dict):
            cfg_row = {"config_id": config_id, "status": "skipped", "reason": "config_not_in_run_manifest"}
            config_rows.append(cfg_row)
            config_results[config_id] = dict(cfg_row)
            continue
        if row.get("status") != "trained":
            cfg_row = {
                "config_id": config_id,
                "status": "skipped",
                "reason": f"config_status_{row.get('status', 'unknown')}",
            }
            config_rows.append(cfg_row)
            config_results[config_id] = dict(cfg_row)
            continue

        try:
            checkpoint_path, norm_path, gmm_path = resolve_checkpoint_norm_gmm_paths(row, run_manifest_base)
            norm_payload = _load_json(norm_path)
            norm_cfg = extract_norm_params(norm_payload)
            gmm_payload = _load_json(gmm_path)
            gmm_cfg = load_gmm_params_json_dict(gmm_payload)

            k = int(row.get("k", gmm_cfg["k"]))
            feature_set = str(row.get("feature_set", norm_payload.get("feature_set", "f2"))).lower()
            if feature_set == "f3":
                raise ValueError("feature_set='f3' is no longer supported; use 'f2'.")
            if feature_set != "f2":
                raise ValueError(f"invalid feature_set for '{config_id}': {feature_set}")
            input_dim = int(row.get("input_dim", 2))
            hidden_dim = int(row.get("hidden_dim", norm_payload.get("hidden_dim", 64)))
            num_layers = int(row.get("num_layers", norm_payload.get("num_layers", 1)))
            if k != int(gmm_cfg["k"]):
                raise ValueError(f"k mismatch between run manifest ({k}) and gmm payload ({int(gmm_cfg['k'])})")

            model = load_gru_classifier(
                checkpoint_path=checkpoint_path,
                k=k,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                device=resolved_device,
            )
            throughput = resolve_throughput(throughput_payload, config_id)
            dataset_path, split_path = resolve_experimental_paths(
                experimental_payload,
                config_id=config_id,
                experimental_base=experimental_base,
            )
            split_payload = _load_json(split_path)
            validate_manifest(split_payload, "split_manifest")
            test_indices = [int(x) for x in split_payload.get("test_indices", [])]
            train_indices = [int(x) for x in split_payload.get("train_indices", [])]
            if len(test_indices) == 0:
                raise ValueError("empty test split")

            with np.load(dataset_path, allow_pickle=True) as data:
                pair_key_arr = np.asarray(data["pair_key"], dtype=object)
                power_arr = np.asarray(data["power"], dtype=object)
                power_start_arr = np.asarray(data["power_start_epoch_s"], dtype=np.float64)
                rate_arr = np.asarray(data["rate"], dtype=object) if "rate" in data else np.asarray([], dtype=object)
                dt_arr = np.asarray(data["dt"], dtype=np.float64).reshape(-1)
            if dt_arr.size == 0:
                raise ValueError("dataset dt missing")
            dt = float(dt_arr[0])
            if (not np.isfinite(dt)) or dt <= 0.0:
                raise ValueError(f"invalid dt in dataset: {dt}")
            n_total = int(min(len(pair_key_arr), len(power_arr), len(power_start_arr)))

            training_power_traces: List[np.ndarray] = []
            training_labels_traces: List[np.ndarray] = []
            for idx in train_indices:
                if idx < 0 or idx >= n_total:
                    continue
                p_train = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
                if p_train.size == 0:
                    continue
                labels_train = predict_sorted_gmm_labels_from_params(p_train, gmm_cfg)
                training_power_traces.append(p_train.astype(np.float64))
                training_labels_traces.append(labels_train.astype(np.int64))

            phi, sigma_innov, sigma_marginal = estimate_ar1_params(
                gmm_params=gmm_cfg,
                training_power_traces=training_power_traces,
                training_labels_traces=training_labels_traces,
                K=int(k),
                min_run_length=AR1_MIN_RUN_LENGTH,
            )
            phi_above_threshold = phi >= float(AR1_PHI_THRESHOLD)
            slug = _safe_slug(config_id)
            ar1_params_path = os.path.join(ar1_params_dir, f"{slug}_ar1_params.json")
            _write_json(
                ar1_params_path,
                {
                    "config_id": config_id,
                    "phi": phi.tolist(),
                    "phi_threshold": float(AR1_PHI_THRESHOLD),
                    "phi_above_threshold": [bool(v) for v in phi_above_threshold.tolist()],
                    "num_ar1_states": int(np.sum(phi_above_threshold)),
                    "num_iid_states": int(np.sum(~phi_above_threshold)),
                    "sigma_innov": sigma_innov.tolist(),
                    "sigma_marginal": sigma_marginal.tolist(),
                    "gmm_means": np.asarray(gmm_cfg["means"], dtype=np.float64).reshape(-1).tolist(),
                    "min_run_length": int(AR1_MIN_RUN_LENGTH),
                },
            )

            trace_records: List[Dict[str, Any]] = []
            for idx in test_indices:
                if idx < 0 or idx >= n_total:
                    per_trace_rows.append(
                        {
                            "config_id": config_id,
                            "trace_idx": int(idx),
                            "pair_key": "",
                            "status": "skipped",
                            "reason": "test_index_out_of_bounds",
                        }
                    )
                    continue
                try:
                    tr = _build_trace_record(
                        trace_idx=int(idx),
                        pair_key=str(pair_key_arr[idx]),
                        rate=str(rate_arr[idx]) if idx < len(rate_arr) else "",
                        power_start_epoch_s=float(power_start_arr[idx]),
                        power=np.asarray(power_arr[idx], dtype=np.float64),
                        dt=dt,
                    )
                    trace_records.append(tr)
                except Exception as exc:
                    per_trace_rows.append(
                        {
                            "config_id": config_id,
                            "trace_idx": int(idx),
                            "pair_key": str(pair_key_arr[idx]) if idx < len(pair_key_arr) else "",
                            "status": "skipped",
                            "reason": f"trace_load_error:{type(exc).__name__}:{exc}",
                        }
                    )

            if len(trace_records) == 0:
                raise ValueError("no valid test traces to evaluate")

            eval_trace_rows: List[Dict[str, object]] = []
            representative_trace_idx = int(trace_records[0]["trace_idx"])
            representative_seed = int(seeds[0])
            representative_gt: Optional[np.ndarray] = None
            representative_pred: Optional[np.ndarray] = None
            gt_traces_by_seed: Dict[int, List[np.ndarray]] = {
                int(seed): [] for seed in seeds
            }
            pred_traces_by_seed: Dict[int, List[np.ndarray]] = {
                int(seed): [] for seed in seeds
            }

            for tr in trace_records:
                trace_idx = int(tr["trace_idx"])
                pair_key = str(tr["pair_key"])
                json_path = pair_map.get(pair_key)
                if json_path is None:
                    per_trace_rows.append(
                        {
                            "config_id": config_id,
                            "trace_idx": trace_idx,
                            "pair_key": pair_key,
                            "status": "skipped",
                            "reason": "pair_key_not_found_in_pair_manifest",
                        }
                    )
                    continue

                try:
                    gt = np.asarray(tr["ground_truth"], dtype=np.float64).reshape(-1)
                    if gt.size == 0:
                        raise ValueError("empty ground truth trace")

                    requests = _build_requests_from_stage0_json(
                        json_path,
                        power_start_epoch_s=float(tr["power_start_epoch_s"]),
                        trace_duration_s=float((int(tr["num_points"]) + 1) * dt),
                        dt=dt,
                    )
                    feat = build_rollout_features_from_requests(
                        requests=requests,
                        throughput=throughput,
                        norm=norm_cfg,
                        T=int(tr["num_points"]),
                        dt=dt,
                        feature_set=feature_set,
                    )

                    # Align request schedule to measured power by matching
                    # first sustained power activation and first A_t activation.
                    a_raw_initial = np.asarray(feat.get("A_raw", []), dtype=np.float64).reshape(-1)
                    offset_seconds = _estimate_request_alignment_offset_seconds(
                        power_trace=gt,
                        a_t=a_raw_initial,
                        dt=float(dt),
                        active_threshold=250.0,
                        window_bins=3,
                    )
                    if abs(float(offset_seconds)) >= (0.5 * float(dt)):
                        requests = _build_requests_from_stage0_json(
                            json_path,
                            power_start_epoch_s=float(tr["power_start_epoch_s"]),
                            trace_duration_s=float((int(tr["num_points"]) + 1) * dt),
                            dt=dt,
                            alignment_offset_s=float(offset_seconds),
                        )
                        feat = build_rollout_features_from_requests(
                            requests=requests,
                            throughput=throughput,
                            norm=norm_cfg,
                            T=int(tr["num_points"]),
                            dt=dt,
                            feature_set=feature_set,
                        )

                    features_norm = np.asarray(feat["features_norm"], dtype=np.float32)
                    if features_norm.ndim != 2 or features_norm.shape[1] != input_dim:
                        raise ValueError(f"rollout feature shape mismatch: {features_norm.shape} vs input_dim={input_dim}")

                    with torch.no_grad():
                        x = torch.tensor(
                            features_norm.tolist(),
                            dtype=torch.float32,
                            device=resolved_device,
                        ).unsqueeze(0)
                        logits_t = model(x)[0].detach().cpu()
                        try:
                            logits = np.asarray(logits_t.numpy(), dtype=np.float64)
                        except Exception:
                            logits = np.asarray(logits_t.tolist(), dtype=np.float64)

                    seed_rows: List[Dict[str, object]] = []
                    pred_by_seed: Dict[int, np.ndarray] = {}
                    for seed_value in seeds:
                        if generation_mode_resolved == "iid":
                            gen = generate_gmm_bigru_trace(
                                logits=logits,
                                gmm_params=gmm_cfg,
                                seed=int(seed_value),
                                decode_mode=decode_mode,
                                median_filter_window=int(median_filter_window),
                                clamp_range=(
                                    norm_cfg["power_min"],
                                    norm_cfg["power_max"],
                                ),
                            )
                        elif generation_mode_resolved == "ar1":
                            gen = generate_gmm_bigru_trace_ar1_thresholded(
                                logits=logits,
                                gmm_params=gmm_cfg,
                                phi=phi,
                                sigma_innov=sigma_innov,
                                sigma_marginal=sigma_marginal,
                                p0=float(tr["p0"]),
                                seed=int(seed_value),
                                decode_mode=decode_mode,
                                median_filter_window=int(median_filter_window),
                                phi_threshold=0.0,
                                clamp_range=(
                                    norm_cfg["power_min"],
                                    norm_cfg["power_max"],
                                ),
                            )
                        else:
                            gen = generate_gmm_bigru_trace_ar1_thresholded(
                                logits=logits,
                                gmm_params=gmm_cfg,
                                phi=phi,
                                sigma_innov=sigma_innov,
                                sigma_marginal=sigma_marginal,
                                p0=float(tr["p0"]),
                                seed=int(seed_value),
                                decode_mode=decode_mode,
                                median_filter_window=int(median_filter_window),
                                phi_threshold=float(AR1_PHI_THRESHOLD),
                                clamp_range=(
                                    norm_cfg["power_min"],
                                    norm_cfg["power_max"],
                                ),
                            )
                        pred = np.asarray(gen["power_w"], dtype=np.float64).reshape(-1)
                        n = int(min(len(gt), len(pred)))
                        if n <= 0:
                            raise ValueError("no aligned points after generation")
                        gt_n = gt[:n]
                        pred_n = pred[:n]
                        metrics = compute_power_metrics(
                            gt_n,
                            pred_n,
                            dt=dt,
                            acf_max_lag=int(acf_max_lag),
                        )
                        energy_gt_j = _total_energy_from_trace(gt_n, dt=dt)
                        energy_pred_j = _total_energy_from_trace(pred_n, dt=dt)
                        seed_row = {
                            "config_id": config_id,
                            "trace_idx": trace_idx,
                            "pair_key": pair_key,
                            "seed": int(seed_value),
                            "num_points": int(n),
                            "status": "ok",
                            "reason": "",
                            "energy_gt_j": float(energy_gt_j),
                            "energy_pred_j": float(energy_pred_j),
                            **metrics,
                        }
                        per_seed_rows.append(seed_row)
                        seed_rows.append(seed_row)
                        pred_by_seed[int(seed_value)] = pred_n
                        gt_traces_by_seed[int(seed_value)].append(gt_n.copy())
                        pred_traces_by_seed[int(seed_value)].append(pred_n.copy())

                    trace_row = {
                        "config_id": config_id,
                        "trace_idx": trace_idx,
                        "pair_key": pair_key,
                        "rate": str(tr["rate"]),
                        "status": "evaluated",
                        "reason": "",
                        "num_requests": int(len(requests)),
                        "num_points": int(seed_rows[0]["num_points"]) if seed_rows else int(tr["num_points"]),
                        "dt": float(dt),
                        "num_seeds": int(len(seed_rows)),
                        "seeds": ";".join(str(x) for x in seeds),
                        "ks_stat_median": _nanmedian(r["ks_stat"] for r in seed_rows),
                        "acf_r2_median": _nanmedian(r["acf_r2"] for r in seed_rows),
                        "nrmse_median": _nanmedian(r["nrmse"] for r in seed_rows),
                        "p95_error_pct_median": _nanmedian(r["p95_error_pct"] for r in seed_rows),
                        "p99_error_pct_median": _nanmedian(r["p99_error_pct"] for r in seed_rows),
                        "delta_energy_pct_median": _nanmedian(r["delta_energy_pct"] for r in seed_rows),
                    }
                    per_trace_rows.append(trace_row)
                    eval_trace_rows.append(trace_row)

                    if trace_idx == representative_trace_idx:
                        nrmse_vals = np.asarray([float(r["nrmse"]) for r in seed_rows], dtype=np.float64)
                        med_nrmse = float(np.median(nrmse_vals))
                        best_i = int(np.argmin(np.abs(nrmse_vals - med_nrmse)))
                        representative_seed = int(seed_rows[best_i]["seed"])
                        representative_gt = gt[: len(pred_by_seed[representative_seed])]
                        representative_pred = pred_by_seed[representative_seed]
                except Exception as exc:
                    per_trace_rows.append(
                        {
                            "config_id": config_id,
                            "trace_idx": trace_idx,
                            "pair_key": pair_key,
                            "status": "failed",
                            "reason": f"{type(exc).__name__}:{exc}",
                        }
                    )

            if len(eval_trace_rows) == 0:
                raise ValueError("all test traces failed or were skipped")

            config_seed_rows: List[Dict[str, object]] = []
            for seed_value in seeds:
                seed_int = int(seed_value)
                gt_traces = gt_traces_by_seed.get(seed_int, [])
                pred_traces = pred_traces_by_seed.get(seed_int, [])
                if len(gt_traces) == 0 or len(pred_traces) == 0:
                    continue
                aggregate_metrics = compute_aggregate_power_metrics(
                    gt_traces,
                    pred_traces,
                    dt=dt,
                    acf_max_lag=int(acf_max_lag),
                )
                total_points = int(
                    sum(len(np.asarray(arr, dtype=np.float64).reshape(-1)) for arr in gt_traces)
                )
                config_seed_row = {
                    "config_id": config_id,
                    "seed": seed_int,
                    "status": "evaluated",
                    "reason": "",
                    "num_eval_traces": int(len(gt_traces)),
                    "num_points": total_points,
                    **aggregate_metrics,
                }
                per_config_seed_rows.append(config_seed_row)
                config_seed_rows.append(config_seed_row)

            if len(config_seed_rows) == 0:
                raise ValueError("no config-seed aggregate metrics were computed")

            plot_paths: Dict[str, str] = {}
            ar1_params_plot_path = ""
            if plots and representative_gt is not None and representative_pred is not None:
                stem = f"{slug}_trace{representative_trace_idx}"
                overlay_path = os.path.join(plots_dir, f"{stem}_overlay.png")
                _plot_overlay(
                    overlay_path,
                    dt=dt,
                    gt=representative_gt,
                    pred=representative_pred,
                    title=f"{config_id} trace={representative_trace_idx} generated vs measured",
                )
                plot_paths = {
                    "overlay_plot": overlay_path,
                }
            if plots:
                ar1_params_plot_path = os.path.join(plots_dir, f"{slug}_ar1_params.png")
                _plot_ar1_params(
                    ar1_params_plot_path,
                    gmm_means=np.asarray(gmm_cfg["means"], dtype=np.float64).reshape(-1),
                    phi=phi,
                    sigma_marginal=sigma_marginal,
                    sigma_innov=sigma_innov,
                    title=f"{config_id} AR(1) parameter diagnostics",
                    phi_threshold=float(AR1_PHI_THRESHOLD),
                )

            cfg_row = {
                "config_id": config_id,
                "status": "evaluated",
                "reason": "",
                "generation_mode": generation_mode_resolved,
                "k": int(k),
                "feature_set": feature_set,
                "decode_mode": decode_mode,
                "median_filter_window": int(median_filter_window),
                "gmm_covariance_type": str(gmm_cfg.get("covariance_type", "full")),
                "num_test_traces": int(len(test_indices)),
                "num_eval_traces": int(len(eval_trace_rows)),
                "num_skipped_or_failed_traces": int(len(test_indices) - len(eval_trace_rows)),
                "num_seeds": int(num_seeds),
                "ks_stat_median": _nanmedian(r["ks_stat_median"] for r in eval_trace_rows),
                "acf_r2_median": _nanmedian(r["acf_r2_median"] for r in eval_trace_rows),
                "nrmse_median": _nanmedian(r["nrmse_median"] for r in eval_trace_rows),
                "p95_error_pct_median": _nanmedian(r["p95_error_pct_median"] for r in eval_trace_rows),
                "p99_error_pct_median": _nanmedian(r["p99_error_pct_median"] for r in eval_trace_rows),
                "delta_energy_pct_median": _nanmedian(r["delta_energy_pct_median"] for r in eval_trace_rows),
                "ks_stat_all_heldout": _nanmedian(r["ks_stat"] for r in config_seed_rows),
                "acf_r2_all_heldout": _nanmedian(r["acf_r2"] for r in config_seed_rows),
                "nrmse_all_heldout": _nanmedian(r["nrmse"] for r in config_seed_rows),
                "p95_error_pct_all_heldout": _nanmedian(
                    r["p95_error_pct"] for r in config_seed_rows
                ),
                "p99_error_pct_all_heldout": _nanmedian(
                    r["p99_error_pct"] for r in config_seed_rows
                ),
                "delta_energy_pct_all_heldout": _nanmedian(
                    r["delta_energy_pct"] for r in config_seed_rows
                ),
                "phi_median": _nanmedian(phi),
                "representative_trace_idx": int(representative_trace_idx),
                "representative_seed": int(representative_seed),
                "ar1_params_json": ar1_params_path,
                "ar1_params_plot": ar1_params_plot_path,
                **plot_paths,
            }
            config_rows.append(cfg_row)
            config_results[config_id] = dict(cfg_row)
        except Exception as exc:
            cfg_row = {
                "config_id": config_id,
                "status": "failed",
                "reason": f"{type(exc).__name__}:{exc}",
            }
            config_rows.append(cfg_row)
            config_results[config_id] = dict(cfg_row)

    per_seed_fields = [
        "config_id",
        "trace_idx",
        "pair_key",
        "seed",
        "status",
        "reason",
        "num_points",
        "energy_gt_j",
        "energy_pred_j",
        "ks_stat",
        "acf_r2",
        "nrmse",
        "p95_error_pct",
        "p99_error_pct",
        "delta_energy_pct",
    ]
    for r in per_seed_rows:
        for f in per_seed_fields:
            r.setdefault(f, "")
    per_seed_csv = os.path.join(out_dir, "per_seed_metrics.csv")
    _write_csv(per_seed_csv, per_seed_rows, per_seed_fields)

    per_trace_fields = [
        "config_id",
        "trace_idx",
        "pair_key",
        "rate",
        "status",
        "reason",
        "num_requests",
        "num_points",
        "dt",
        "num_seeds",
        "seeds",
        "ks_stat_median",
        "acf_r2_median",
        "nrmse_median",
        "p95_error_pct_median",
        "p99_error_pct_median",
        "delta_energy_pct_median",
    ]
    for r in per_trace_rows:
        for f in per_trace_fields:
            r.setdefault(f, "")
    per_trace_csv = os.path.join(out_dir, "per_trace_metrics.csv")
    _write_csv(per_trace_csv, per_trace_rows, per_trace_fields)

    per_config_seed_fields = [
        "config_id",
        "seed",
        "status",
        "reason",
        "num_eval_traces",
        "num_points",
        "ks_stat",
        "acf_r2",
        "nrmse",
        "p95_error_pct",
        "p99_error_pct",
        "delta_energy_pct",
    ]
    for r in per_config_seed_rows:
        for f in per_config_seed_fields:
            r.setdefault(f, "")
    per_config_seed_csv = os.path.join(out_dir, "per_config_seed_metrics.csv")
    _write_csv(per_config_seed_csv, per_config_seed_rows, per_config_seed_fields)

    config_fields = [
        "config_id",
        "status",
        "reason",
        "generation_mode",
        "k",
        "feature_set",
        "decode_mode",
        "median_filter_window",
        "gmm_covariance_type",
        "num_test_traces",
        "num_eval_traces",
        "num_skipped_or_failed_traces",
        "num_seeds",
        "ks_stat_median",
        "acf_r2_median",
        "nrmse_median",
        "p95_error_pct_median",
        "p99_error_pct_median",
        "delta_energy_pct_median",
        "ks_stat_all_heldout",
        "acf_r2_all_heldout",
        "nrmse_all_heldout",
        "p95_error_pct_all_heldout",
        "p99_error_pct_all_heldout",
        "delta_energy_pct_all_heldout",
        "phi_median",
        "representative_trace_idx",
        "representative_seed",
        "ar1_params_json",
        "ar1_params_plot",
        "overlay_plot",
    ]
    for r in config_rows:
        for f in config_fields:
            r.setdefault(f, "")
    config_csv = os.path.join(out_dir, "config_summary.csv")
    _write_csv(config_csv, config_rows, config_fields)

    run_manifest_payload = {
        "schema_version": "continuous-v1-gmm-bigru-eval-run-v2",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "run_manifest": run_manifest,
            "experimental_manifest": experimental_manifest,
            "throughput_db": throughput_db,
            "pair_manifest_csv": pair_manifest_csv,
        },
        "defaults": {
            "out_dir": out_dir,
            "num_seeds": int(num_seeds),
            "base_seed": int(base_seed),
            "acf_max_lag": int(acf_max_lag),
            "decode_mode": str(decode_mode),
            "median_filter_window": int(median_filter_window),
            "device": str(resolved_device),
            "plots": bool(plots),
            "generation_mode": generation_mode_resolved,
            "phi_threshold": (
                float(AR1_PHI_THRESHOLD)
                if generation_mode_resolved == "ar1_thresholded"
                else (0.0 if generation_mode_resolved == "ar1" else None)
            ),
            "min_run_length": int(AR1_MIN_RUN_LENGTH),
        },
        "summary": {
            "num_target_configs": int(len(targets)),
            "num_evaluated_configs": int(sum(1 for r in config_rows if r.get("status") == "evaluated")),
            "num_failed_configs": int(sum(1 for r in config_rows if r.get("status") == "failed")),
            "num_skipped_configs": int(sum(1 for r in config_rows if r.get("status") == "skipped")),
        },
        "artifacts": {
            "per_seed_metrics_csv": per_seed_csv,
            "per_trace_metrics_csv": per_trace_csv,
            "per_config_seed_metrics_csv": per_config_seed_csv,
            "config_summary_csv": config_csv,
            "plots_dir": plots_dir,
            "ar1_params_dir": ar1_params_dir,
        },
        "configs": config_results,
    }
    run_manifest_out = os.path.join(out_dir, "run_manifest.json")
    _write_json(run_manifest_out, run_manifest_payload)
    return run_manifest_payload
