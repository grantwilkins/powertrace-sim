#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from model.scripts.continuous_v1_infer import run_inference_from_artifacts


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text)


def _load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _synthesize_request_timestamps(payload: Dict[str, object], n: int) -> Optional[List[float]]:
    if n <= 0:
        return []

    duration = _finite_float(payload.get("duration"))
    if duration is not None and duration > 0:
        step = float(duration) / float(max(n, 1))
        if step > 0:
            values = (np.arange(n, dtype=np.float64) + 0.5) * step + 1.0
            return [float(x) for x in values]

    request_rate = _finite_float(payload.get("request_rate"))
    poisson_rate = _finite_float(payload.get("poisson_rate"))
    rate = request_rate if request_rate is not None else poisson_rate
    if rate is not None and rate > 0:
        step = 1.0 / float(rate)
        values = (np.arange(n, dtype=np.float64) + 1.0) * step + 1.0
        return [float(x) for x in values]

    return None


def _resolve_trace_index(
    split_path: str,
    test_pick: int,
    trace_idx: Optional[int],
) -> int:
    if trace_idx is not None:
        return int(trace_idx)

    split_payload = _load_json(split_path)
    test_indices = split_payload.get("test_indices", [])
    if not isinstance(test_indices, list) or len(test_indices) == 0:
        raise ValueError(f"No test_indices found in split file: {split_path}")
    if test_pick < 0 or test_pick >= len(test_indices):
        raise ValueError(
            f"--test-pick {test_pick} out of range for {split_path} "
            f"(num_test_indices={len(test_indices)})"
        )
    return int(test_indices[int(test_pick)])


def _load_trace_from_dataset(dataset_path: str, trace_idx: int) -> Dict[str, object]:
    with np.load(dataset_path, allow_pickle=True) as data:
        pair_keys = np.asarray(data["pair_key"], dtype=object)
        power_all = np.asarray(data["power"], dtype=object)
        power_start_all = np.asarray(data["power_start_epoch_s"], dtype=np.float64)
        dt = float(np.asarray(data["dt"], dtype=np.float64).reshape(-1)[0])

    if trace_idx < 0 or trace_idx >= len(power_all):
        raise ValueError(f"trace_idx {trace_idx} out of bounds for dataset {dataset_path}")

    power = np.asarray(power_all[trace_idx], dtype=np.float64).reshape(-1)
    if len(power) < 2:
        raise ValueError(f"Trace {trace_idx} in {dataset_path} has length < 2")

    pair_key = str(pair_keys[trace_idx]) if trace_idx < len(pair_keys) else f"trace-{trace_idx}"
    power_start_epoch_s = float(power_start_all[trace_idx])
    return {
        "pair_key": pair_key,
        "power": power,
        "power_start_epoch_s": power_start_epoch_s,
        "dt": dt,
    }


def _find_pair_row(pair_manifest_csv: str, pair_key: str) -> Dict[str, str]:
    rows = _load_csv_rows(pair_manifest_csv)
    matches = [r for r in rows if str(r.get("status", "")).strip() == "matched" and str(r.get("pair_key", "")) == pair_key]
    if len(matches) == 0:
        raise ValueError(f"pair_key '{pair_key}' not found in matched rows of {pair_manifest_csv}")
    return matches[0]


def _build_requests_from_json(
    request_json_path: str,
    power_start_epoch_s: float,
    trace_duration_s: float,
    dt: float,
) -> List[Dict[str, float]]:
    payload = _load_json(request_json_path)
    required = ("input_lens", "output_lens")
    if any(not isinstance(payload.get(k), list) for k in required):
        raise ValueError(
            f"Request JSON missing required arrays {required}: {request_json_path}"
        )

    in_lens = payload["input_lens"]
    out_lens = payload["output_lens"]
    n_base = int(min(len(in_lens), len(out_lens)))
    req_ts_raw = payload.get("request_timestamps")
    if isinstance(req_ts_raw, list):
        n = int(min(n_base, len(req_ts_raw)))
        req_ts = req_ts_raw[:n]
    else:
        n = int(n_base)
        synth = _synthesize_request_timestamps(payload, n)
        if synth is None:
            raise ValueError(
                f"Request JSON missing required arrays ('request_timestamps',): {request_json_path}"
            )
        req_ts = synth
    if n <= 0:
        raise ValueError(f"No aligned request rows found in {request_json_path}")

    arrivals = np.asarray(req_ts[:n], dtype=np.float64) - float(power_start_epoch_s)
    if arrivals.size > 0 and (
        float(np.min(arrivals)) < -float(dt) or float(np.max(arrivals)) > float(trace_duration_s) + float(dt)
    ):
        arrivals = arrivals - float(np.min(arrivals))

    out: List[Dict[str, float]] = []
    for i in range(n):
        a = float(arrivals[i])
        nin = float(in_lens[i])
        nout = float(out_lens[i])
        if not (np.isfinite(a) and np.isfinite(nin) and np.isfinite(nout)):
            continue
        out.append(
            {
                "arrival_time": float(a),
                "input_tokens": float(max(0.0, nin)),
                "output_tokens": float(max(0.0, nout)),
            }
        )

    if len(out) == 0:
        raise ValueError(f"No valid requests after filtering in {request_json_path}")
    return out


def _write_requests_json(path: str, requests: Sequence[Dict[str, float]]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump({"requests": list(requests)}, f, indent=2)


def _read_generated_csv(path: str) -> np.ndarray:
    rows = _load_csv_rows(path)
    return np.asarray([float(r["power_w"]) for r in rows], dtype=np.float64)


def _write_comparison_csv(path: str, dt: float, gt: np.ndarray, pred: np.ndarray) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    n = int(min(len(gt), len(pred)))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["t_bin", "time_s", "ground_truth_w", "generated_w", "error_w", "abs_error_w"],
        )
        writer.writeheader()
        for i in range(n):
            err = float(pred[i] - gt[i])
            writer.writerow(
                {
                    "t_bin": int(i),
                    "time_s": float(i * dt),
                    "ground_truth_w": float(gt[i]),
                    "generated_w": float(pred[i]),
                    "error_w": float(err),
                    "abs_error_w": float(abs(err)),
                }
            )


def run_compare(
    *,
    config_id: str,
    test_pick: int = 0,
    trace_idx: Optional[int] = None,
    seed: int = 42,
    out_dir: str = "results/continuous_v1/eval",
    run_manifest: str = "results/continuous_v1/run_manifest.json",
    throughput_db: str = "model/config/throughput_database.json",
    pair_manifest_csv: str = "results/stage0/pair_manifest.csv",
    experimental_root: str = "results/experimental_continuous_v1",
    device: str = "auto",
) -> Dict[str, object]:
    split_path = os.path.join(experimental_root, "splits", f"{config_id}.json")
    dataset_path = os.path.join(experimental_root, "datasets", f"{config_id}.npz")
    if not os.path.exists(split_path):
        raise ValueError(f"Split path not found: {split_path}")
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path not found: {dataset_path}")

    resolved_trace_idx = _resolve_trace_index(
        split_path=split_path,
        test_pick=int(test_pick),
        trace_idx=trace_idx,
    )
    tr = _load_trace_from_dataset(dataset_path=dataset_path, trace_idx=resolved_trace_idx)
    pair_key = str(tr["pair_key"])
    power = np.asarray(tr["power"], dtype=np.float64).reshape(-1)
    power_start_epoch_s = float(tr["power_start_epoch_s"])
    dt = float(tr["dt"])

    manifest_row = _find_pair_row(pair_manifest_csv=pair_manifest_csv, pair_key=pair_key)
    request_json_path = str(manifest_row.get("json_path", "")).strip()
    if request_json_path == "" or (not os.path.exists(request_json_path)):
        raise ValueError(f"Request JSON path invalid for pair_key '{pair_key}': {request_json_path}")

    requests = _build_requests_from_json(
        request_json_path=request_json_path,
        power_start_epoch_s=power_start_epoch_s,
        trace_duration_s=float(len(power) * dt),
        dt=dt,
    )

    _ensure_dir(out_dir)
    slug = _safe_slug(config_id)
    stem = f"{slug}_trace{resolved_trace_idx}"
    requests_json_out = os.path.join(out_dir, f"{stem}_requests.json")
    generated_csv_out = os.path.join(out_dir, f"{stem}_generated.csv")
    comparison_csv_out = os.path.join(out_dir, f"{stem}_comparison.csv")
    metrics_json_out = os.path.join(out_dir, f"{stem}_metrics.json")

    _write_requests_json(requests_json_out, requests)

    infer_result = run_inference_from_artifacts(
        config_id=config_id,
        requests_json=requests_json_out,
        out_csv=generated_csv_out,
        run_manifest=run_manifest,
        throughput_db=throughput_db,
        device=device,
        seed=int(seed),
        dt=dt,
        T=int(len(power) - 1),
        p0=float(power[0]),
    )

    pred = _read_generated_csv(generated_csv_out)
    gt = power[1 : 1 + len(pred)]
    n = int(min(len(gt), len(pred)))
    gt = gt[:n]
    pred = pred[:n]
    if n == 0:
        raise ValueError("No aligned points for comparison.")

    err = pred - gt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    nrmse = float(rmse / (float(np.max(gt) - np.min(gt)) + 1e-12))
    energy_err_pct = float(100.0 * ((float(np.sum(pred)) - float(np.sum(gt))) / (float(np.sum(gt)) + 1e-12)))

    _write_comparison_csv(comparison_csv_out, dt=dt, gt=gt, pred=pred)

    metrics = {
        "config_id": config_id,
        "trace_idx": int(resolved_trace_idx),
        "pair_key": pair_key,
        "num_points": int(n),
        "dt": float(dt),
        "seed": int(seed),
        "mae_w": float(mae),
        "rmse_w": float(rmse),
        "nrmse": float(nrmse),
        "energy_err_pct": float(energy_err_pct),
        "paths": {
            "requests_json": requests_json_out,
            "generated_csv": generated_csv_out,
            "comparison_csv": comparison_csv_out,
            "metrics_json": metrics_json_out,
            "request_source_json": request_json_path,
            "dataset_npz": dataset_path,
            "split_json": split_path,
        },
        "inference": infer_result,
    }
    with open(metrics_json_out, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and compare Stage-4 v1 trace against ground truth.")
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--test-pick", type=int, default=0, help="Index into split test_indices.")
    parser.add_argument("--trace-idx", type=int, default=None, help="Direct dataset trace index override.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results/continuous_v1/eval")
    parser.add_argument("--run-manifest", default="results/continuous_v1/run_manifest.json")
    parser.add_argument("--throughput-db", default="model/config/throughput_database.json")
    parser.add_argument("--pair-manifest-csv", default="results/stage0/pair_manifest.csv")
    parser.add_argument("--experimental-root", default="results/experimental_continuous_v1")
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = run_compare(
        config_id=args.config_id,
        test_pick=args.test_pick,
        trace_idx=args.trace_idx,
        seed=args.seed,
        out_dir=args.out_dir,
        run_manifest=args.run_manifest,
        throughput_db=args.throughput_db,
        pair_manifest_csv=args.pair_manifest_csv,
        experimental_root=args.experimental_root,
        device=args.device,
    )

    print("[continuous_v1_compare] Done")
    print(f"  config_id: {metrics['config_id']}")
    print(f"  trace_idx: {metrics['trace_idx']}")
    print(f"  pair_key : {metrics['pair_key']}")
    print(f"  MAE      : {metrics['mae_w']:.6f} W")
    print(f"  RMSE     : {metrics['rmse_w']:.6f} W")
    print(f"  NRMSE    : {metrics['nrmse']:.6f}")
    print(f"  Energy%  : {metrics['energy_err_pct']:.6f}")
    print(f"  metrics  : {metrics['paths']['metrics_json']}")
    print(f"  compare  : {metrics['paths']['comparison_csv']}")


if __name__ == "__main__":
    main()
