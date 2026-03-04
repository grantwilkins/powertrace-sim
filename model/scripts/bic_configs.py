#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Mapping

# Shared BIC configuration policy for methods + D1 figures.
# These are restricted to configs with recorded request_timestamps in ShareGPT data.
BIC_CONFIGS_RECORDED_BALANCED: Dict[str, Dict[str, str]] = {
    "bic_config1": {
        "config_id": "deepseek-r1-distill-8b_A100_tp1",
        "title": "DeepSeek-R1-Distill-8B / A100 / TP=1",
        "legend": "DeepSeek A100 TP1",
    },
    "bic_config2": {
        "config_id": "llama-3-8b_H100_tp2",
        "title": "Llama-3.1-8B / H100 / TP=2",
        "legend": "Llama H100 TP2",
    },
    "bic_config3": {
        "config_id": "gpt-oss-120b_A100_tp4",
        "title": "GPT-OSS-120B / A100 / TP=4 (MoE Proxy)",
        "legend": "GPT-OSS A100 TP4",
    },
    "bic_config4": {
        "config_id": "deepseek-r1-distill-70b_H100_tp4",
        "title": "DeepSeek-R1-Distill-70B / H100 / TP=4 (Dense)",
        "legend": "DeepSeek H100 TP4",
    },
}

# Canonical alias used by figure scripts.
BIC_CONFIGS = BIC_CONFIGS_RECORDED_BALANCED


def clone_bic_configs(configs: Mapping[str, Mapping[str, str]] = BIC_CONFIGS) -> Dict[str, Dict[str, str]]:
    return {str(k): {str(kk): str(vv) for kk, vv in dict(v).items()} for k, v in dict(configs).items()}

