"""Per-dtype FP8 weight-streaming calibration (split amendment v2).

Contract amendment, user-directed 2026-07-16: the frozen holdout left
llama-3-405b fully unseen, which makes the FP8 streaming efficiency
unlearnable — it is the only FP8 checkpoint in the dataset. This script
declares a v2 split in which 405B repeats 0 and 1 at rates <= 2 become a
`dtype_calibration` role used to fit ONE scalar per hardware: the ratio of
effective HBM streaming efficiency for FP8-quantized weights to the dense
BF16 efficiency. It is a dtype-class constant (any FP8 deployment could
supply it; it is shared by all FP8 models), not a per-model refit. 405B
repeat 2 and all rate-4 runs remain untouched test data.

The public training command owns file I/O. FP8 consumers fail closed when this
calibration is absent.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize_scalar

from model.timing.iteration import (
    iteration_time_s,
    iteration_work,
    launch_overhead_s,
)
from model.training_data.arch import get_arch

CALIBRATION_MODEL = "llama-3-405b"
CALIBRATION_REPEATS = (0, 1)
CALIBRATION_MAX_RATE = 2.0
SCALE_BOUNDS = (0.5, 1.1)


def calibration_run_ids(data, manifest):
    roles = {int(k): v for k, v in manifest["roles"].items()}
    out = []
    for rid, role in roles.items():
        if (role in {"holdout_model", "train"}
                and str(data["run_model"][rid]) == CALIBRATION_MODEL
                and int(data["run_repeat"][rid]) in CALIBRATION_REPEATS
                and float(data["run_rate"][rid]) <= CALIBRATION_MAX_RATE):
            out.append(rid)
    return sorted(out)


def itl_points(data, run_ids):
    """Per-request median inter-token latency at reconstructed concurrency."""
    offsets = data["itl_offsets"]
    points = []
    for rid in run_ids:
        idx = np.flatnonzero(data["req_run_id"] == rid)
        start = data["arrival_time_s"][idx] + data["ttft_s"][idx]
        end = start + data["decode_duration_s"][idx]
        keep = data["output_tokens"][idx] >= 16
        overlap = (np.minimum(end[keep, None], end[None, :])
                   - np.maximum(start[keep, None], start[None, :]))
        concurrency = 1.0 + (
            (np.clip(overlap, 0.0, None).sum(axis=1) - (end[keep] - start[keep]))
            / np.maximum(end[keep] - start[keep], 1e-9))
        for j, i in enumerate(idx[keep]):
            itls = data["itl_values"][offsets[i]:offsets[i + 1]]
            points.append({
                "tp": int(data["run_tp"][rid]),
                "hardware": str(data["run_hardware"][rid]),
                "batch": float(concurrency[j]),
                "context": float(data["input_tokens"][i])
                + float(data["output_tokens"][i]) / 2.0,
                "seconds": float(np.median(itls))})
    return points


def calibrate(data, manifest, fitted) -> tuple[dict, dict, dict]:
    """Return calibrated fit, amended roles, and calibration summary."""
    run_ids = calibration_run_ids(data, manifest)
    if not run_ids:
        raise ValueError("no calibration runs found")
    points = itl_points(data, run_ids)
    arch = get_arch(CALIBRATION_MODEL)
    hardware = points[0]["hardware"]
    if any(point["hardware"] != hardware for point in points):
        raise ValueError("one FP8 calibration may cover only one hardware")
    params = fitted[hardware]

    def loss(scale):
        residuals = []
        for p in points:
            t_launch = launch_overhead_s(
                arch, base_s=params["base_overhead_s"],
                per_message_s=params["per_message_s"][str(p["tp"])])
            work = iteration_work(
                arch, decode_batch=p["batch"], context_mean=p["context"]
            )
            pred = iteration_time_s(
                work, hardware=hardware, tp=p["tp"],
                eff_flops=params["eff_flops"], eff_bw=params["eff_bw"],
                t_launch_s=t_launch, t_sample_s=params["per_token_sample_s"],
                transformer_bw_scale=scale,
            )
            residuals.append(np.log(pred) - np.log(p["seconds"]))
        return float(np.mean(np.asarray(residuals) ** 2))

    result = minimize_scalar(loss, bounds=SCALE_BOUNDS, method="bounded")
    scale = float(result.x)
    calibrated = json.loads(json.dumps(fitted))
    calibrated[hardware]["fp8_stream_scale"] = scale
    calibrated[hardware]["fp8_stream_support"] = {
        "calibration_model": CALIBRATION_MODEL,
        "hardware": hardware,
        "repeats": list(CALIBRATION_REPEATS),
        "max_rate": CALIBRATION_MAX_RATE,
        "runs": run_ids,
        "scope": "FP8 recipe and engine represented by calibration model",
    }
    calibrated[hardware].setdefault("provenance_notes", []).append(
        f"fp8_stream_scale={scale:.4f} fitted on {len(run_ids)} "
        f"dtype_calibration runs ({CALIBRATION_MODEL} repeats "
        f"{CALIBRATION_REPEATS} rates<= {CALIBRATION_MAX_RATE}), "
        "split amendment v2, user-directed 2026-07-16")
    roles = dict(manifest["roles"])
    for rid in run_ids:
        roles[str(rid)] = "dtype_calibration"
    amended = {**manifest, "roles": roles,
               "schema_version": manifest.get("schema_version", "v1") + "+fp8",
               "amendment": ("405B repeats 0,1 at rates<=2 reassigned to "
                             "dtype_calibration to fit one FP8 streaming "
                             "scalar per hardware; repeat 2 and rate 4.0 "
                             "remain test")}
    summary = {
        "hardware": hardware,
        "fp8_stream_scale": scale,
        "rmse_log": float(np.sqrt(result.fun)),
        "requests": len(points),
        "runs": len(run_ids),
    }
    return calibrated, amended, summary
