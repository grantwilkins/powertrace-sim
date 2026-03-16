from __future__ import annotations

from typing import Dict, Mapping, Tuple

from model.utils.io import resolve_existing_path
from model.utils.numeric import finite_float


def resolve_experimental_paths(
    experimental_manifest: Mapping[str, object],
    *,
    config_id: str,
    experimental_base: str,
) -> Tuple[str, str]:
    cfgs = experimental_manifest.get("configs", {})
    if not isinstance(cfgs, dict):
        raise ValueError("Invalid experimental manifest format")
    row = cfgs.get(config_id)
    if not isinstance(row, dict):
        raise ValueError(f"config_id '{config_id}' not found in experimental manifest")
    dataset_path = resolve_existing_path(str(row.get("dataset_npz", "")), experimental_base)
    split_path = resolve_existing_path(str(row.get("split_json", "")), experimental_base)
    if dataset_path is None:
        raise ValueError(f"Dataset path not found for '{config_id}'")
    if split_path is None:
        raise ValueError(f"Split path not found for '{config_id}'")
    return dataset_path, split_path


def resolve_checkpoint_norm_gmm_paths(
    config_entry: Mapping[str, object], base_dir: str
) -> Tuple[str, str, str]:
    checkpoint_raw = str(config_entry.get("checkpoint_path", ""))
    norm_raw = str(config_entry.get("norm_params_path", ""))
    gmm_raw = str(config_entry.get("gmm_params_path", ""))
    checkpoint_path = resolve_existing_path(checkpoint_raw, base_dir)
    norm_path = resolve_existing_path(norm_raw, base_dir)
    gmm_path = resolve_existing_path(gmm_raw, base_dir)
    if checkpoint_path is None:
        raise ValueError(f"Checkpoint path not found: {checkpoint_raw}")
    if norm_path is None:
        raise ValueError(f"Norm params path not found: {norm_raw}")
    if gmm_path is None:
        raise ValueError(f"GMM path not found: {gmm_raw}")
    return checkpoint_path, norm_path, gmm_path


def resolve_throughput(
    throughput_payload: Mapping[str, object], config_id: str
) -> Dict[str, float]:
    cfgs = throughput_payload.get("configs", {})
    if not isinstance(cfgs, dict):
        raise ValueError("Invalid throughput database format")
    row = cfgs.get(config_id)
    if not isinstance(row, dict):
        raise ValueError(f"config_id '{config_id}' not found in throughput DB")
    prefill = finite_float(row.get("prefill_rate_median_toks_per_s"))
    decode = finite_float(row.get("decode_rate_median_toks_per_s"))
    if prefill is None or prefill <= 0.0:
        raise ValueError(f"Invalid prefill throughput for '{config_id}'")
    if decode is None or decode <= 0.0:
        raise ValueError(f"Invalid decode throughput for '{config_id}'")
    return {"lambda_prefill": prefill, "lambda_decode": decode}


__all__ = [
    "resolve_checkpoint_norm_gmm_paths",
    "resolve_experimental_paths",
    "resolve_throughput",
]
