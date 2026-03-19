from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

CONFIG_MODEL_SIZE_RE = re.compile(r"^(.+)-(\d+)b_(A100|H100)_tp(\d+)$")


def resolve_device(device: Optional[torch.device | str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def parse_config_ids(config_ids: Optional[Sequence[str]]) -> List[str]:
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


def parse_config_id(config_id: str) -> Dict[str, str]:
    match = CONFIG_MODEL_SIZE_RE.match(str(config_id).strip())
    if not match:
        raise ValueError(
            f"Invalid config_id format: '{config_id}'. "
            "Expected format: '{model}-{size}b_{hardware}_tp{digits}'"
        )

    model_family, model_size, hardware, tp = match.groups()
    return {
        "model_family": model_family,
        "model_size": model_size,
        "hardware": hardware,
        "tp": tp,
    }


def is_moe_config(config_id: str) -> bool:
    match = CONFIG_MODEL_SIZE_RE.match(str(config_id).strip())
    if match is None:
        return False

    model_family = str(match.group(1)).lower()
    model_size = int(match.group(2))

    # DeepSeek-R1-Distill is treated as dense.
    if "deepseek-r1-distill" in model_family:
        return False

    # GPT-OSS 20B+ models are treated as MoE.
    if "gpt-oss" in model_family and model_size >= 20:
        return True

    return False


def safe_float(value: object, field_name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"Unable to parse float for '{field_name}': {value}") from exc
    if not np.isfinite(out):
        raise ValueError(f"Non-finite float for '{field_name}': {value}")
    return out


def parse_csv_list(values: str) -> list[str]:
    out = [x.strip() for x in str(values).split(",") if x.strip()]
    if not out:
        raise ValueError("Expected a non-empty comma-separated list.")
    return out
