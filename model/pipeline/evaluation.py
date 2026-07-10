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

from model.classifiers.gmm_bigru import load_gmm_params_json_dict
from model.classifiers.features import (
    build_next_step_features_from_requests,
    extract_norm_params,
)
from model.classifiers.trace_generation import generate_gmm_bigru_trace
from model.classifiers.model_loading import load_gru_classifier
from model.classifiers.metrics import (
    compute_aggregate_power_metrics,
    compute_power_metrics,
)
from model.pipeline.request_builder import _build_requests_from_stage0_json
from model.pipeline.artifact_resolution import (
    resolve_checkpoint_norm_gmm_paths,
    resolve_bound_throughput,
    resolve_experimental_paths,
)
from model.pipeline.manifest_validation import validate_manifest
from model.utils.config import (
    parse_config_ids as _parse_config_ids,
    resolve_device as _resolve_device,
)
from model.utils.io import (
    ensure_dir as _ensure_dir,
    load_json as _load_json,
    repo_relative_or_absolute as _repo_relative_or_absolute,
    resolve_input_path as _resolve_input_path,
    resolve_existing_path as _resolve_existing_path,
    safe_slug as _safe_slug,
    write_csv as _write_csv,
    write_json as _write_json,
)
from model.utils.provenance import assert_file_identity, file_identity, git_state, sha256_file

def _nanmedian(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _total_energy_from_trace(power_w: np.ndarray, *, dt: float) -> float:
    arr = np.asarray(power_w, dtype=np.float64).reshape(-1)
    return float(np.sum(arr) * float(dt))


# Requests are placed at their recorded arrival times; the oracle alignment
# offset is computed as a diagnostic only and never applied to the schedule.
TIMING_MODE = "arrival_only"
REQUEST_ALIGNMENT_MODE = "none"
REQUEST_ALIGNMENT_SIGNAL = "measured_power_first_sustained_activation"
FALLBACK_STATUS = "none"


def _request_timestamp_source(request_json_path: str) -> str:
    payload = _load_json(request_json_path)
    if isinstance(payload.get("request_timestamps"), list) and payload["request_timestamps"]:
        return "recorded"
    return "missing"


def _request_json_from_lineage(
    lineage: Dict[str, object], *, experimental_base: str
) -> str:
    paths = lineage.get("source_paths")
    hashes = lineage.get("source_sha256")
    if not isinstance(paths, dict) or not isinstance(hashes, dict):
        raise ValueError("trace lineage is missing source paths or hashes")
    key = "requests_json" if "requests_json" in paths else "requests.json"
    raw_path = str(paths.get(key, ""))
    resolved = _resolve_existing_path(raw_path, experimental_base)
    if resolved is None:
        raise ValueError(f"lineage request source not found: {raw_path}")
    expected_hash = str(hashes.get(key, ""))
    if not expected_hash or sha256_file(resolved) != expected_hash:
        raise ValueError(f"lineage request source hash mismatch: {resolved}")
    return resolved


def _build_eval_command(
    *,
    run_manifest: str,
    experimental_manifest: str,
    throughput_db: str,
    pair_manifest_csv: str,
    out_dir: str,
    config_ids: Optional[Sequence[str]],
    num_seeds: int,
    base_seed: int,
    acf_max_lag: int,
    generation_mode: str,
    decode_mode: str,
    median_filter_window: int,
    device: str,
    plots: bool,
) -> List[str]:
    command = [
        "uv",
        "run",
        "-m",
        "model.scripts.eval_gmm_bigru",
        "--run-manifest",
        run_manifest,
        "--experimental-manifest",
        experimental_manifest,
        "--throughput-db",
        throughput_db,
        "--pair-manifest-csv",
        pair_manifest_csv,
        "--out-dir",
        out_dir,
        "--num-seeds",
        str(int(num_seeds)),
        "--base-seed",
        str(int(base_seed)),
        "--acf-max-lag",
        str(int(acf_max_lag)),
        "--generation-mode",
        generation_mode,
        "--decode-mode",
        decode_mode,
        "--median-filter-window",
        str(int(median_filter_window)),
        "--device",
        device,
    ]
    for config_id in config_ids or []:
        command.extend(["--config-id", str(config_id)])
    if not plots:
        command.append("--no-plots")
    return command


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


def _build_evaluation_rollout_features(
    *,
    requests: Sequence[Dict[str, object]],
    throughput: Dict[str, float],
    norm: Dict[str, float],
    num_points: int,
    dt: float,
    feature_set: str,
) -> Dict[str, np.ndarray]:
    """Build features at t=dt..N*dt, matching training targets power[1:]."""
    return build_next_step_features_from_requests(
        requests=requests,
        throughput=throughput,
        norm=norm,
        num_points=int(num_points),
        dt=float(dt),
        feature_set=feature_set,
    )


def evaluate_from_artifacts(
    *,
    run_manifest: str = "results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json",
    experimental_manifest: str = "results/experimental_continuous_v1/manifest.json",
    throughput_db: str = "model/throughput_database.json",
    pair_manifest_csv: str = "results/stage0/pair_manifest.csv",
    out_dir: str = "results/continuous_v1_gmm_bigru/k10_f2/eval_metrics",
    config_ids: Optional[Sequence[str]] = None,
    num_seeds: int = 5,
    base_seed: int = 42,
    device: str = "auto",
    acf_max_lag: int = 50,
    generation_mode: str = "iid",
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    plots: bool = True,
) -> Dict[str, object]:
    if int(num_seeds) <= 0:
        raise ValueError("num_seeds must be >= 1")
    generation_mode_resolved = str(generation_mode).strip().lower()
    if generation_mode_resolved != "iid":
        raise ValueError(
            "generation_mode must be 'iid'; AR(1) generation modes "
            f"('ar1', 'ar1_thresholded') were removed. Got: {generation_mode!r}"
        )
    generation_mode_label = generation_mode_resolved
    if decode_mode not in {"stochastic", "argmax"}:
        raise ValueError(f"decode_mode must be one of {{'stochastic','argmax'}}; got {decode_mode}")

    run_manifest_path = _resolve_input_path(run_manifest)
    experimental_manifest_path = _resolve_input_path(experimental_manifest)
    run_manifest_payload = _load_json(run_manifest_path)
    validate_manifest(run_manifest_payload, "run_manifest")
    run_cfgs = run_manifest_payload.get("configs", {})
    run_manifest_base = str(Path(run_manifest_path).resolve().parent)

    experimental_payload = _load_json(experimental_manifest_path)
    validate_manifest(experimental_payload, "experimental_manifest")
    experimental_base = str(Path(experimental_manifest_path).resolve().parent)

    requested = _parse_config_ids(config_ids)
    if requested:
        targets = requested
    else:
        targets = sorted([cid for cid, row in run_cfgs.items() if isinstance(row, dict) and row.get("status") == "trained"])

    resolved_device = _resolve_device(device)
    _ensure_dir(out_dir)
    plots_dir = os.path.join(out_dir, "plots")
    _ensure_dir(plots_dir)

    per_seed_rows: List[Dict[str, object]] = []
    per_trace_rows: List[Dict[str, object]] = []
    per_config_seed_rows: List[Dict[str, object]] = []
    config_rows: List[Dict[str, object]] = []
    config_results: Dict[str, Dict[str, object]] = {}
    resolved_artifacts: Dict[str, Dict[str, object]] = {}
    seeds = [int(base_seed) + i for i in range(int(num_seeds))]
    command = _build_eval_command(
        run_manifest=run_manifest,
        experimental_manifest=experimental_manifest,
        throughput_db=throughput_db,
        pair_manifest_csv=pair_manifest_csv,
        out_dir=out_dir,
        config_ids=config_ids,
        num_seeds=int(num_seeds),
        base_seed=int(base_seed),
        acf_max_lag=int(acf_max_lag),
        generation_mode=generation_mode_resolved,
        decode_mode=str(decode_mode),
        median_filter_window=int(median_filter_window),
        device=str(device),
        plots=bool(plots),
    )

    for config_id in targets:
        row = run_cfgs.get(config_id)
        if not isinstance(row, dict):
            cfg_row = {
                "config_id": config_id,
                "status": "skipped",
                "reason": "config_not_in_run_manifest",
                "generation_mode": generation_mode_resolved,
                "generation_mode_label": generation_mode_label,
                "timing_mode": TIMING_MODE,
                "alignment_mode": REQUEST_ALIGNMENT_MODE,
                "request_alignment_mode": REQUEST_ALIGNMENT_MODE,
                "timestamp_source": "",
                "fallback_status": FALLBACK_STATUS,
            }
            config_rows.append(cfg_row)
            config_results[config_id] = dict(cfg_row)
            continue
        if row.get("status") != "trained":
            cfg_row = {
                "config_id": config_id,
                "status": "skipped",
                "reason": f"config_status_{row.get('status', 'unknown')}",
                "generation_mode": generation_mode_resolved,
                "generation_mode_label": generation_mode_label,
                "timing_mode": TIMING_MODE,
                "alignment_mode": REQUEST_ALIGNMENT_MODE,
                "request_alignment_mode": REQUEST_ALIGNMENT_MODE,
                "timestamp_source": "",
                "fallback_status": FALLBACK_STATUS,
            }
            config_rows.append(cfg_row)
            config_results[config_id] = dict(cfg_row)
            continue

        try:
            checkpoint_path, norm_path, gmm_path = resolve_checkpoint_norm_gmm_paths(row, run_manifest_base)
            identities = row.get("artifact_identities")
            if not isinstance(identities, dict):
                raise ValueError("trained config is missing artifact identities")
            assert_file_identity(checkpoint_path, identities.get("checkpoint"), label="checkpoint")
            assert_file_identity(norm_path, identities.get("trained_norm"), label="normalization")
            assert_file_identity(gmm_path, identities.get("gmm"), label="GMM")
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
            throughput = resolve_bound_throughput(row, config_id)
            dataset_path, split_path = resolve_experimental_paths(
                experimental_payload,
                config_id=config_id,
                experimental_base=experimental_base,
            )
            experimental_cfg = experimental_payload["configs"][config_id]
            lineage_path = _resolve_existing_path(
                str(experimental_cfg.get("lineage_json", "")), experimental_base
            )
            if lineage_path is None:
                raise ValueError("experimental config is missing lineage_json")
            assert_file_identity(dataset_path, identities.get("dataset"), label="dataset")
            assert_file_identity(split_path, identities.get("split"), label="split")
            assert_file_identity(lineage_path, identities.get("lineage"), label="lineage")
            lineage_by_trace: Dict[int, Dict[str, object]] = {}
            lineage_payload = _load_json(lineage_path)
            if lineage_payload.get("schema_version") != "gru-dataset-lineage-v1":
                raise ValueError("unsupported GRU dataset lineage schema")
            lineage_rows = lineage_payload.get("traces")
            if not isinstance(lineage_rows, list):
                raise ValueError("GRU dataset lineage traces must be a list")
            lineage_by_trace = {
                int(entry["trace_index"]): entry
                for entry in lineage_rows
                if isinstance(entry, dict)
            }
            if len(lineage_by_trace) != len(lineage_rows):
                raise ValueError("GRU dataset lineage has duplicate or invalid trace rows")
            resolved_artifacts[config_id] = {
                "checkpoint": file_identity(checkpoint_path),
                "norm": file_identity(norm_path),
                "gmm": file_identity(gmm_path),
                "dataset": file_identity(dataset_path),
                "split": file_identity(split_path),
                "lineage": file_identity(lineage_path),
            }
            split_payload = _load_json(split_path)
            validate_manifest(split_payload, "split_manifest")
            test_indices = [int(x) for x in split_payload.get("test_indices", [])]
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
            if set(lineage_by_trace) != set(range(n_total)):
                raise ValueError("GRU dataset lineage must bind every trace exactly once")

            slug = _safe_slug(config_id)
            trace_records: List[Dict[str, Any]] = []
            for idx in test_indices:
                if idx < 0 or idx >= n_total:
                    per_trace_rows.append(
                        {
                            "config_id": config_id,
                            "trace_idx": int(idx),
                            "pair_key": "",
                            "generation_mode": generation_mode_resolved,
                            "generation_mode_label": generation_mode_label,
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
                    tr["lineage"] = lineage_by_trace.get(int(idx))
                    trace_records.append(tr)
                except Exception as exc:
                    per_trace_rows.append(
                        {
                            "config_id": config_id,
                            "trace_idx": int(idx),
                            "pair_key": str(pair_key_arr[idx]) if idx < len(pair_key_arr) else "",
                            "generation_mode": generation_mode_resolved,
                            "generation_mode_label": generation_mode_label,
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
            config_timestamp_sources: set[str] = set()

            for tr in trace_records:
                trace_idx = int(tr["trace_idx"])
                pair_key = str(tr["pair_key"])
                lineage = tr.get("lineage")
                if not isinstance(lineage, dict):
                    raise ValueError(f"trace {trace_idx} is missing hash-bound lineage")
                json_path = _request_json_from_lineage(
                    lineage, experimental_base=experimental_base
                )

                try:
                    timestamp_source = _request_timestamp_source(json_path)
                    config_timestamp_sources.add(timestamp_source)
                    gt = np.asarray(tr["ground_truth"], dtype=np.float64).reshape(-1)
                    if gt.size == 0:
                        raise ValueError("empty ground truth trace")

                    requests = _build_requests_from_stage0_json(
                        json_path,
                        power_start_epoch_s=float(tr["power_start_epoch_s"]),
                        trace_duration_s=float((int(tr["num_points"]) + 1) * dt),
                        dt=dt,
                    )
                    feat = _build_evaluation_rollout_features(
                        requests=requests,
                        throughput=throughput,
                        norm=norm_cfg,
                        num_points=int(tr["num_points"]),
                        dt=dt,
                        feature_set=feature_set,
                    )

                    # Diagnostic only: estimate the oracle alignment offset between
                    # first sustained power activation and first A_t activation.
                    # It is recorded but NOT applied; requests stay at their
                    # recorded arrival times (timing_mode="arrival_only").
                    a_raw_initial = np.asarray(feat.get("A_raw", []), dtype=np.float64).reshape(-1)
                    oracle_offset_seconds = _estimate_request_alignment_offset_seconds(
                        power_trace=gt,
                        a_t=a_raw_initial,
                        dt=float(dt),
                        active_threshold=250.0,
                        window_bins=3,
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
                            "generation_mode": generation_mode_resolved,
                            "generation_mode_label": generation_mode_label,
                            "timing_mode": TIMING_MODE,
                            "timestamp_source": timestamp_source,
                            "alignment_mode": REQUEST_ALIGNMENT_MODE,
                            "fallback_status": FALLBACK_STATUS,
                            "oracle_alignment_offset_s": float(oracle_offset_seconds),
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
                        "generation_mode": generation_mode_resolved,
                        "generation_mode_label": generation_mode_label,
                        "timing_mode": TIMING_MODE,
                        "status": "evaluated",
                        "reason": "",
                        "timestamp_source": timestamp_source,
                        "alignment_mode": REQUEST_ALIGNMENT_MODE,
                        "fallback_status": FALLBACK_STATUS,
                        "request_alignment_mode": REQUEST_ALIGNMENT_MODE,
                        "request_alignment_signal": REQUEST_ALIGNMENT_SIGNAL,
                        "oracle_alignment_offset_s": float(oracle_offset_seconds),
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
                            "generation_mode": generation_mode_resolved,
                            "generation_mode_label": generation_mode_label,
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
                    "generation_mode": generation_mode_resolved,
                    "generation_mode_label": generation_mode_label,
                    "timing_mode": TIMING_MODE,
                    "alignment_mode": REQUEST_ALIGNMENT_MODE,
                    "fallback_status": FALLBACK_STATUS,
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

            trace_rows_for_config = [
                r for r in per_trace_rows if str(r.get("config_id", "")) == config_id
            ]
            num_skipped_traces = int(
                sum(1 for r in trace_rows_for_config if r.get("status") == "skipped")
            )
            num_failed_traces = int(
                sum(1 for r in trace_rows_for_config if r.get("status") == "failed")
            )

            cfg_row = {
                "config_id": config_id,
                "status": "evaluated",
                "reason": "",
                "generation_mode": generation_mode_resolved,
                "generation_mode_label": generation_mode_label,
                "timing_mode": TIMING_MODE,
                "timestamp_source": (
                    ";".join(sorted(config_timestamp_sources))
                    if config_timestamp_sources
                    else ""
                ),
                "alignment_mode": REQUEST_ALIGNMENT_MODE,
                "fallback_status": FALLBACK_STATUS,
                "k": int(k),
                "feature_set": feature_set,
                "decode_mode": decode_mode,
                "median_filter_window": int(median_filter_window),
                "request_alignment_mode": REQUEST_ALIGNMENT_MODE,
                "request_alignment_signal": REQUEST_ALIGNMENT_SIGNAL,
                "gmm_covariance_type": str(gmm_cfg.get("covariance_type", "full")),
                "num_test_traces": int(len(test_indices)),
                "num_eval_traces": int(len(eval_trace_rows)),
                "num_skipped_traces": int(num_skipped_traces),
                "num_failed_traces": int(num_failed_traces),
                "num_skipped_or_failed_traces": int(
                    num_skipped_traces + num_failed_traces
                ),
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
                "representative_trace_idx": int(representative_trace_idx),
                "representative_seed": int(representative_seed),
                **plot_paths,
            }
            config_rows.append(cfg_row)
            config_results[config_id] = dict(cfg_row)
        except Exception as exc:
            cfg_row = {
                "config_id": config_id,
                "status": "failed",
                "reason": f"{type(exc).__name__}:{exc}",
                "generation_mode": generation_mode_resolved,
                "generation_mode_label": generation_mode_label,
                "timing_mode": TIMING_MODE,
                "alignment_mode": REQUEST_ALIGNMENT_MODE,
                "request_alignment_mode": REQUEST_ALIGNMENT_MODE,
                "timestamp_source": "",
                "fallback_status": FALLBACK_STATUS,
                "num_test_traces": 0,
                "num_eval_traces": 0,
                "num_skipped_traces": 0,
                "num_failed_traces": 0,
                "num_skipped_or_failed_traces": 0,
            }
            config_rows.append(cfg_row)
            config_results[config_id] = dict(cfg_row)

    per_seed_fields = [
        "config_id",
        "trace_idx",
        "pair_key",
        "seed",
        "generation_mode",
        "generation_mode_label",
        "timing_mode",
        "timestamp_source",
        "alignment_mode",
        "fallback_status",
        "oracle_alignment_offset_s",
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
        r["timing_mode"] = TIMING_MODE
        for f in per_seed_fields:
            r.setdefault(f, "")
    per_seed_csv = os.path.join(out_dir, "per_seed_metrics.csv")
    _write_csv(per_seed_csv, per_seed_rows, per_seed_fields)

    per_trace_fields = [
        "config_id",
        "trace_idx",
        "pair_key",
        "rate",
        "generation_mode",
        "generation_mode_label",
        "timing_mode",
        "status",
        "reason",
        "timestamp_source",
        "alignment_mode",
        "fallback_status",
        "request_alignment_mode",
        "request_alignment_signal",
        "oracle_alignment_offset_s",
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
        r["timing_mode"] = TIMING_MODE
        for f in per_trace_fields:
            r.setdefault(f, "")
    per_trace_csv = os.path.join(out_dir, "per_trace_metrics.csv")
    _write_csv(per_trace_csv, per_trace_rows, per_trace_fields)

    per_config_seed_fields = [
        "config_id",
        "seed",
        "generation_mode",
        "generation_mode_label",
        "timing_mode",
        "alignment_mode",
        "fallback_status",
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
        r["timing_mode"] = TIMING_MODE
        for f in per_config_seed_fields:
            r.setdefault(f, "")
    per_config_seed_csv = os.path.join(out_dir, "per_config_seed_metrics.csv")
    _write_csv(per_config_seed_csv, per_config_seed_rows, per_config_seed_fields)

    config_fields = [
        "config_id",
        "status",
        "reason",
        "generation_mode",
        "generation_mode_label",
        "timing_mode",
        "timestamp_source",
        "alignment_mode",
        "fallback_status",
        "k",
        "feature_set",
        "decode_mode",
        "median_filter_window",
        "request_alignment_mode",
        "request_alignment_signal",
        "gmm_covariance_type",
        "num_test_traces",
        "num_eval_traces",
        "num_skipped_traces",
        "num_failed_traces",
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
        "representative_trace_idx",
        "representative_seed",
        "overlay_plot",
    ]
    for r in config_rows:
        r["timing_mode"] = TIMING_MODE
        for f in config_fields:
            r.setdefault(f, "")
    config_csv = os.path.join(out_dir, "config_summary.csv")
    _write_csv(config_csv, config_rows, config_fields)

    revision = git_state()
    input_identities = {
        "run_manifest": file_identity(run_manifest_path),
        "experimental_manifest": file_identity(experimental_manifest_path),
    }
    run_manifest_payload = {
        "schema_version": "continuous-v1-gmm-bigru-eval-run-v2",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": command,
        **revision,
        "config_ids": list(targets),
        "seed": int(base_seed),
        "seeds": list(seeds),
        "generation_mode": generation_mode_resolved,
        "timing_mode": TIMING_MODE,
        "alignment_mode": REQUEST_ALIGNMENT_MODE,
        "timestamp_source": "recorded",
        "fallback_status": FALLBACK_STATUS,
        "inputs": {
            "run_manifest": run_manifest,
            "experimental_manifest": experimental_manifest,
        },
        "input_paths": {
            "run_manifest": _repo_relative_or_absolute(run_manifest_path),
            "experimental_manifest": _repo_relative_or_absolute(experimental_manifest_path),
        },
        "provenance": {
            "command": command,
            "source_revision": revision,
            "config_ids": list(targets),
            "seed": int(base_seed),
            "seeds": list(seeds),
            "generation_mode": generation_mode_resolved,
            "timing_mode": TIMING_MODE,
            "alignment_mode": REQUEST_ALIGNMENT_MODE,
            "timestamp_source": "recorded",
            "fallback_status": FALLBACK_STATUS,
            "inputs": input_identities,
            "model_artifacts": resolved_artifacts,
            "artifacts": {
                "per_seed_metrics_csv": _repo_relative_or_absolute(per_seed_csv),
                "per_trace_metrics_csv": _repo_relative_or_absolute(per_trace_csv),
                "per_config_seed_metrics_csv": _repo_relative_or_absolute(per_config_seed_csv),
                "config_summary_csv": _repo_relative_or_absolute(config_csv),
                "plots_dir": _repo_relative_or_absolute(plots_dir),
            },
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
            "generation_mode_label": generation_mode_label,
            "timing_mode": TIMING_MODE,
            "request_alignment_mode": REQUEST_ALIGNMENT_MODE,
            "request_alignment_signal": REQUEST_ALIGNMENT_SIGNAL,
        },
        "summary": {
            "num_target_configs": int(len(targets)),
            "num_evaluated_configs": int(sum(1 for r in config_rows if r.get("status") == "evaluated")),
            "num_failed_configs": int(sum(1 for r in config_rows if r.get("status") == "failed")),
            "num_skipped_configs": int(sum(1 for r in config_rows if r.get("status") == "skipped")),
            "num_test_traces": int(
                sum(int(r.get("num_test_traces") or 0) for r in config_rows)
            ),
            "num_eval_traces": int(
                sum(int(r.get("num_eval_traces") or 0) for r in config_rows)
            ),
            "num_skipped_traces": int(
                sum(int(r.get("num_skipped_traces") or 0) for r in config_rows)
            ),
            "num_failed_traces": int(
                sum(int(r.get("num_failed_traces") or 0) for r in config_rows)
            ),
            "num_skipped_or_failed_traces": int(
                sum(
                    int(r.get("num_skipped_or_failed_traces") or 0)
                    for r in config_rows
                )
            ),
        },
        "artifacts": {
            "per_seed_metrics_csv": per_seed_csv,
            "per_trace_metrics_csv": per_trace_csv,
            "per_config_seed_metrics_csv": per_config_seed_csv,
            "config_summary_csv": config_csv,
            "plots_dir": plots_dir,
        },
        "configs": config_results,
    }
    run_manifest_out = os.path.join(out_dir, "run_manifest.json")
    _write_json(run_manifest_out, run_manifest_payload)
    return run_manifest_payload
