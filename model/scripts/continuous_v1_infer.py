#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import numpy as np
import torch

from model.classifiers.continuous_gru import (
    MeanRevertingGRU,
    compute_inference_features,
    generate_mean_reverting_trace,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def _resolve_device(device: Optional[torch.device | str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def _read_json(path: str) -> Dict[str, object]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_requests_json(path: str) -> List[Dict[str, object]]:
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        reqs = payload
    elif isinstance(payload, dict):
        reqs = payload.get("requests")
    else:
        reqs = None
    if not isinstance(reqs, list):
        raise ValueError("requests JSON must be a list or object with key 'requests'.")

    out: List[Dict[str, object]] = []
    for i, req in enumerate(reqs):
        if not isinstance(req, dict):
            raise ValueError(f"request[{i}] must be an object")
        for key in ("arrival_time", "input_tokens", "output_tokens"):
            if key not in req:
                raise ValueError(f"request[{i}] missing '{key}'")
        out.append(req)
    return out


def _resolve_config_entry(run_manifest_path: str, config_id: str) -> Tuple[Dict[str, object], str]:
    payload = _read_json(run_manifest_path)
    config_map = payload.get("configs", {})
    if not isinstance(config_map, dict):
        raise ValueError(f"Invalid run manifest format: {run_manifest_path}")
    if config_id not in config_map:
        raise ValueError(f"config_id '{config_id}' not found in run manifest")
    entry = config_map[config_id]
    if not isinstance(entry, dict):
        raise ValueError(f"Invalid config entry for '{config_id}' in run manifest")
    return entry, str(Path(run_manifest_path).resolve().parent)


def _resolve_throughput(throughput_db_path: str, config_id: str) -> Dict[str, float]:
    payload = _read_json(throughput_db_path)
    configs = payload.get("configs", {})
    if not isinstance(configs, dict):
        raise ValueError(f"Invalid throughput DB format: {throughput_db_path}")
    if config_id not in configs:
        raise ValueError(f"config_id '{config_id}' not found in throughput DB")
    row = configs[config_id]
    if not isinstance(row, dict):
        raise ValueError(f"Invalid throughput row for '{config_id}'")

    prefill = float(row.get("prefill_rate_median_toks_per_s", float("nan")))
    decode = float(row.get("decode_rate_median_toks_per_s", float("nan")))
    if (not np.isfinite(prefill)) or prefill <= 0.0:
        raise ValueError(f"Invalid prefill throughput for '{config_id}': {prefill}")
    if (not np.isfinite(decode)) or decode <= 0.0:
        raise ValueError(f"Invalid decode throughput for '{config_id}': {decode}")
    return {"lambda_prefill": prefill, "lambda_decode": decode}


def _resolve_checkpoint_and_norm_paths(
    config_entry: Dict[str, object],
    base_dir: str,
    checkpoint: Optional[str],
    norm_params: Optional[str],
) -> Tuple[str, str]:
    checkpoint_raw = checkpoint if checkpoint is not None else str(config_entry.get("checkpoint_path", ""))
    norm_raw = norm_params if norm_params is not None else str(config_entry.get("norm_params_path", ""))

    checkpoint_path = _resolve_existing_path(checkpoint_raw, base_dir)
    norm_path = _resolve_existing_path(norm_raw, base_dir)
    if checkpoint_path is None:
        raise ValueError(f"Checkpoint path not found: {checkpoint_raw}")
    if norm_path is None:
        raise ValueError(f"Norm params path not found: {norm_raw}")
    return checkpoint_path, norm_path


def _extract_norm_for_inference(norm_payload: Dict[str, object]) -> Dict[str, float]:
    required = (
        "active_mean",
        "active_std",
        "t_arrive_log_mean",
        "t_arrive_log_std",
        "power_mean",
        "power_std",
        "power_min",
        "power_max",
    )
    missing = [k for k in required if k not in norm_payload]
    if missing:
        raise ValueError(f"Norm params missing keys: {missing}")

    out = {
        "A_mean": float(norm_payload["active_mean"]),
        "A_std": float(norm_payload["active_std"]),
        "T_arrive_log_mean": float(norm_payload["t_arrive_log_mean"]),
        "T_arrive_log_std": float(norm_payload["t_arrive_log_std"]),
        "power_mean": float(norm_payload["power_mean"]),
        "power_std": float(norm_payload["power_std"]),
        "power_min": float(norm_payload["power_min"]),
        "power_max": float(norm_payload["power_max"]),
    }
    for key, val in out.items():
        if not np.isfinite(val):
            raise ValueError(f"Non-finite norm value for '{key}': {val}")
    if out["A_std"] <= 0.0:
        raise ValueError("A_std must be positive")
    if out["T_arrive_log_std"] <= 0.0:
        raise ValueError("T_arrive_log_std must be positive")
    if out["power_std"] <= 0.0:
        raise ValueError("power_std must be positive")
    return out


def _write_trace_csv(path: str, power: np.ndarray, dt: float) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t_bin", "time_s", "power_w"])
        writer.writeheader()
        for i, p in enumerate(np.asarray(power, dtype=np.float64).reshape(-1)):
            writer.writerow(
                {
                    "t_bin": int(i),
                    "time_s": float(i * dt),
                    "power_w": float(p),
                }
            )


def run_inference_from_artifacts(
    *,
    config_id: str,
    requests_json: str,
    out_csv: str,
    run_manifest: str = "results/continuous_v1/run_manifest.json",
    throughput_db: str = "model/config/throughput_database.json",
    device: str = "auto",
    seed: Optional[int] = None,
    dt: Optional[float] = None,
    T: Optional[int] = None,
    p0: Optional[float] = None,
    n_mix: Optional[int] = None,
    checkpoint: Optional[str] = None,
    norm_params: Optional[str] = None,
    hidden_dim: Optional[int] = None,
    num_layers: Optional[int] = None,
) -> Dict[str, object]:
    config_entry, manifest_dir = _resolve_config_entry(run_manifest, config_id)
    if (checkpoint is None or norm_params is None) and str(config_entry.get("status", "")) != "trained":
        raise ValueError(
            f"Config '{config_id}' is not marked trained in run manifest. "
            "Provide explicit --checkpoint and --norm-params to override."
        )

    checkpoint_path, norm_path = _resolve_checkpoint_and_norm_paths(
        config_entry=config_entry,
        base_dir=manifest_dir,
        checkpoint=checkpoint,
        norm_params=norm_params,
    )
    norm_payload = _read_json(norm_path)
    throughput = _resolve_throughput(throughput_db, config_id)
    requests = _load_requests_json(requests_json)

    infer_norm = _extract_norm_for_inference(norm_payload)
    feat_cfg = {
        "lambda_prefill": throughput["lambda_prefill"],
        "lambda_decode": throughput["lambda_decode"],
        "A_mean": infer_norm["A_mean"],
        "A_std": infer_norm["A_std"],
        "T_arrive_log_mean": infer_norm["T_arrive_log_mean"],
        "T_arrive_log_std": infer_norm["T_arrive_log_std"],
    }

    resolved_dt = float(norm_payload.get("dt", 0.25) if dt is None else dt)
    if (not np.isfinite(resolved_dt)) or resolved_dt <= 0.0:
        raise ValueError(f"dt must be positive; got {resolved_dt}")
    features_norm = compute_inference_features(
        requests=requests,
        config=feat_cfg,
        T=T,
        dt=resolved_dt,
    )

    resolved_T = int(features_norm.shape[0])
    if resolved_T <= 0:
        raise ValueError("Computed horizon T is zero; increase --T or provide non-empty requests.")

    resolved_n_mix = int(config_entry.get("n_mix", norm_payload.get("n_mix", 1)) if n_mix is None else n_mix)
    if resolved_n_mix < 1:
        raise ValueError(f"n_mix must be >= 1; got {resolved_n_mix}")

    resolved_hidden_dim = int(norm_payload.get("hidden_dim", (64 if hidden_dim is None else hidden_dim)))
    resolved_num_layers = int(norm_payload.get("num_layers", (1 if num_layers is None else num_layers)))
    resolved_p0 = float(infer_norm["power_min"] if p0 is None else p0)

    resolved_device = _resolve_device(device)
    model = MeanRevertingGRU(
        input_dim=3,
        hidden_dim=resolved_hidden_dim,
        num_layers=resolved_num_layers,
        n_mix=resolved_n_mix,
    ).to(resolved_device)
    state = torch.load(checkpoint_path, map_location=resolved_device)
    if isinstance(state, dict) and "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
        state = state["model_state_dict"]
    model.load_state_dict(state)

    power_w = generate_mean_reverting_trace(
        model=model,
        features_norm=features_norm,
        P_0=resolved_p0,
        config_norm={
            "power_mean": infer_norm["power_mean"],
            "power_std": infer_norm["power_std"],
            "power_min": infer_norm["power_min"],
            "power_max": infer_norm["power_max"],
        },
        n_mix=resolved_n_mix,
        seed=seed,
    )
    _write_trace_csv(out_csv, power_w, dt=resolved_dt)

    return {
        "config_id": config_id,
        "checkpoint_path": checkpoint_path,
        "norm_params_path": norm_path,
        "throughput_db": throughput_db,
        "requests_json": requests_json,
        "out_csv": out_csv,
        "dt": float(resolved_dt),
        "T": int(resolved_T),
        "p0": float(resolved_p0),
        "n_mix": int(resolved_n_mix),
        "hidden_dim": int(resolved_hidden_dim),
        "num_layers": int(resolved_num_layers),
        "device": str(resolved_device),
        "num_requests": int(len(requests)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 4 v1 mean-reverting power trace inference.")
    parser.add_argument("--run-manifest", default="results/continuous_v1/run_manifest.json")
    parser.add_argument("--throughput-db", default="model/config/throughput_database.json")
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--requests-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--p0", type=float, default=None)
    parser.add_argument("--n-mix", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--norm-params", default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_inference_from_artifacts(
        config_id=args.config_id,
        requests_json=args.requests_json,
        out_csv=args.out_csv,
        run_manifest=args.run_manifest,
        throughput_db=args.throughput_db,
        device=args.device,
        seed=args.seed,
        dt=args.dt,
        T=args.T,
        p0=args.p0,
        n_mix=args.n_mix,
        checkpoint=args.checkpoint,
        norm_params=args.norm_params,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )

    print("[continuous_v1_infer] Inference complete")
    print(f"  config_id: {result['config_id']}")
    print(f"  checkpoint: {result['checkpoint_path']}")
    print(f"  dt: {result['dt']}")
    print(f"  T: {result['T']}")
    print(f"  p0: {result['p0']}")
    print(f"  n_mix: {result['n_mix']}")
    print(f"  out_csv: {result['out_csv']}")


if __name__ == "__main__":
    main()
