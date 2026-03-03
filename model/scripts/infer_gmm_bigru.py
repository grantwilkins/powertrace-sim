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

from model.classifiers.gmm_bigru import (
    build_rollout_features_from_requests,
    generate_gmm_bigru_trace,
    load_gmm_params_json_dict,
)
from model.classifiers.gru import GRUClassifier


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
    cfgs = payload.get("configs", {})
    if not isinstance(cfgs, dict):
        raise ValueError(f"Invalid run manifest format: {run_manifest_path}")
    if config_id not in cfgs:
        raise ValueError(f"config_id '{config_id}' not found in run manifest")
    row = cfgs[config_id]
    if not isinstance(row, dict):
        raise ValueError(f"Invalid config entry for '{config_id}'")
    return row, str(Path(run_manifest_path).resolve().parent)


def _resolve_throughput(throughput_db_path: str, config_id: str) -> Dict[str, float]:
    payload = _read_json(throughput_db_path)
    cfgs = payload.get("configs", {})
    if not isinstance(cfgs, dict):
        raise ValueError(f"Invalid throughput DB format: {throughput_db_path}")
    row = cfgs.get(config_id)
    if not isinstance(row, dict):
        raise ValueError(f"config_id '{config_id}' not found in throughput DB")

    prefill = float(row.get("prefill_rate_median_toks_per_s", float("nan")))
    decode = float(row.get("decode_rate_median_toks_per_s", float("nan")))
    if (not np.isfinite(prefill)) or prefill <= 0.0:
        raise ValueError(f"Invalid prefill throughput for '{config_id}': {prefill}")
    if (not np.isfinite(decode)) or decode <= 0.0:
        raise ValueError(f"Invalid decode throughput for '{config_id}': {decode}")
    return {"lambda_prefill": prefill, "lambda_decode": decode}


def _resolve_paths(
    *,
    config_entry: Dict[str, object],
    base_dir: str,
    checkpoint: Optional[str],
    gmm_params: Optional[str],
    norm_params: Optional[str],
) -> Tuple[str, str, str]:
    checkpoint_raw = checkpoint if checkpoint is not None else str(config_entry.get("checkpoint_path", ""))
    gmm_raw = gmm_params if gmm_params is not None else str(config_entry.get("gmm_params_path", ""))
    norm_raw = norm_params if norm_params is not None else str(config_entry.get("norm_params_path", ""))

    checkpoint_path = _resolve_existing_path(checkpoint_raw, base_dir)
    gmm_path = _resolve_existing_path(gmm_raw, base_dir)
    norm_path = _resolve_existing_path(norm_raw, base_dir)
    if checkpoint_path is None:
        raise ValueError(f"Checkpoint path not found: {checkpoint_raw}")
    if gmm_path is None:
        raise ValueError(f"GMM params path not found: {gmm_raw}")
    if norm_path is None:
        raise ValueError(f"Norm params path not found: {norm_raw}")
    return checkpoint_path, gmm_path, norm_path


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
        "delta_A_mean",
        "delta_A_std",
    )
    missing = [k for k in required if k not in norm_payload]
    if missing:
        raise ValueError(f"Norm params missing keys: {missing}")

    out = {k: float(norm_payload[k]) for k in required}
    for k, v in out.items():
        if not np.isfinite(v):
            raise ValueError(f"Non-finite norm value for '{k}': {v}")
    if out["active_std"] <= 0.0:
        raise ValueError("active_std must be positive")
    if out["t_arrive_log_std"] <= 0.0:
        raise ValueError("t_arrive_log_std must be positive")
    if out["power_std"] <= 0.0:
        raise ValueError("power_std must be positive")
    if out["delta_A_std"] <= 0.0:
        raise ValueError("delta_A_std must be positive")
    return out


def _write_trace_csv(path: str, power: np.ndarray, dt: float) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t_bin", "time_s", "power_w"])
        writer.writeheader()
        for i, p in enumerate(np.asarray(power, dtype=np.float64).reshape(-1)):
            writer.writerow({"t_bin": int(i), "time_s": float(i * dt), "power_w": float(p)})


def run_inference_from_artifacts(
    *,
    config_id: str,
    requests_json: str,
    out_csv: str,
    run_manifest: str = "results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2/run_manifest.json",
    throughput_db: str = "model/config/throughput_database.json",
    device: str = "auto",
    seed: Optional[int] = None,
    dt: Optional[float] = None,
    T: Optional[int] = None,
    p0: Optional[float] = None,
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    checkpoint: Optional[str] = None,
    gmm_params: Optional[str] = None,
    norm_params: Optional[str] = None,
    k: Optional[int] = None,
    feature_set: Optional[str] = None,
    hidden_dim: Optional[int] = None,
    num_layers: Optional[int] = None,
) -> Dict[str, object]:
    config_entry, manifest_base = _resolve_config_entry(run_manifest, config_id)
    if (
        (checkpoint is None or gmm_params is None or norm_params is None)
        and str(config_entry.get("status", "")) != "trained"
    ):
        raise ValueError(
            f"Config '{config_id}' is not marked trained in run manifest. "
            "Provide explicit --checkpoint, --gmm-params, and --norm-params to override."
        )

    checkpoint_path, gmm_path, norm_path = _resolve_paths(
        config_entry=config_entry,
        base_dir=manifest_base,
        checkpoint=checkpoint,
        gmm_params=gmm_params,
        norm_params=norm_params,
    )
    norm_payload = _read_json(norm_path)
    gmm_payload = _read_json(gmm_path)
    norm_cfg = _extract_norm_for_inference(norm_payload)
    gmm_cfg = load_gmm_params_json_dict(gmm_payload)
    throughput = _resolve_throughput(throughput_db, config_id)
    requests = _load_requests_json(requests_json)

    resolved_feature_set = str(
        feature_set
        if feature_set is not None
        else config_entry.get("feature_set", norm_payload.get("feature_set", "f2"))
    ).lower()
    if resolved_feature_set not in {"f2", "f3"}:
        raise ValueError(f"feature_set must be one of {{'f2','f3'}}; got {resolved_feature_set}")

    resolved_dt = float(norm_payload.get("dt", 0.25) if dt is None else dt)
    if (not np.isfinite(resolved_dt)) or resolved_dt <= 0.0:
        raise ValueError(f"dt must be positive; got {resolved_dt}")

    feat = build_rollout_features_from_requests(
        requests=requests,
        throughput=throughput,
        norm=norm_cfg,
        T=T,
        dt=resolved_dt,
        feature_set=resolved_feature_set,
    )
    features_norm = np.asarray(feat["features_norm"], dtype=np.float32)
    resolved_T = int(features_norm.shape[0])
    if resolved_T <= 0:
        raise ValueError("Computed horizon T is zero; increase --T or provide non-empty requests.")

    resolved_k = int(k if k is not None else config_entry.get("k", gmm_cfg["k"]))
    if resolved_k != int(gmm_cfg["k"]):
        raise ValueError(f"K mismatch between args/config ({resolved_k}) and GMM ({int(gmm_cfg['k'])})")
    resolved_input_dim = int(features_norm.shape[1])
    resolved_hidden_dim = int(hidden_dim if hidden_dim is not None else config_entry.get("hidden_dim", 64))
    resolved_num_layers = int(num_layers if num_layers is not None else config_entry.get("num_layers", 1))
    resolved_p0 = float(norm_cfg["power_min"] if p0 is None else p0)

    resolved_device = _resolve_device(device)
    model = GRUClassifier(
        Dx=resolved_input_dim,
        K=resolved_k,
        H=resolved_hidden_dim,
        num_layers=int(max(1, resolved_num_layers)),
    ).to(resolved_device)
    try:
        state = torch.load(checkpoint_path, map_location=resolved_device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=resolved_device)
    if isinstance(state, dict) and "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(features_norm).to(device=resolved_device, dtype=torch.float32).unsqueeze(0)
        logits = model(x)[0].detach().cpu().numpy()
    generated = generate_gmm_bigru_trace(
        logits=logits,
        gmm_params=gmm_cfg,
        seed=seed,
        decode_mode=decode_mode,
        median_filter_window=int(median_filter_window),
        clamp_range=(norm_cfg["power_min"], norm_cfg["power_max"]),
    )
    power_w = np.asarray(generated["power_w"], dtype=np.float64).reshape(-1)
    _write_trace_csv(out_csv, power_w, dt=resolved_dt)

    return {
        "config_id": config_id,
        "checkpoint_path": checkpoint_path,
        "gmm_params_path": gmm_path,
        "norm_params_path": norm_path,
        "throughput_db": throughput_db,
        "requests_json": requests_json,
        "out_csv": out_csv,
        "dt": float(resolved_dt),
        "T": int(resolved_T),
        "p0": float(resolved_p0),
        "k": int(resolved_k),
        "feature_set": resolved_feature_set,
        "input_dim": int(resolved_input_dim),
        "hidden_dim": int(resolved_hidden_dim),
        "num_layers": int(max(1, resolved_num_layers)),
        "decode_mode": str(decode_mode),
        "median_filter_window": int(median_filter_window),
        "device": str(resolved_device),
        "num_requests": int(len(requests)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate traces from continuous v1 GMM+BiGRU artifacts.")
    parser.add_argument("--run-manifest", default="results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2/run_manifest.json")
    parser.add_argument("--throughput-db", default="model/config/throughput_database.json")
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--requests-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--p0", type=float, default=None)
    parser.add_argument("--decode-mode", choices=["stochastic", "argmax"], default="stochastic")
    parser.add_argument("--median-filter-window", type=int, default=1)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gmm-params", default=None)
    parser.add_argument("--norm-params", default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--feature-set", choices=["f2", "f3"], default=None)
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
        decode_mode=args.decode_mode,
        median_filter_window=args.median_filter_window,
        checkpoint=args.checkpoint,
        gmm_params=args.gmm_params,
        norm_params=args.norm_params,
        k=args.k,
        feature_set=args.feature_set,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )

    print("[infer_gmm_bigru] Inference complete")
    print(f"  config_id: {result['config_id']}")
    print(f"  dt: {result['dt']}")
    print(f"  T: {result['T']}")
    print(f"  decode_mode: {result['decode_mode']}")
    print(f"  out_csv: {result['out_csv']}")


if __name__ == "__main__":
    main()
