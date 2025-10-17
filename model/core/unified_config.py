"""
Unified configuration system - single source of truth in performance_database.json.
All model configs (TTFT, TPOT, power stats) in one place.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


def _resolve_path(path: str) -> str:
    """Resolve path relative to project root."""
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    resolved = os.path.join(project_root, path)
    if os.path.exists(resolved):
        return resolved
    return path


@dataclass
class ModelConfig:
    """Complete configuration for a model/hardware/TP combination."""
    model_name: str
    hardware: str
    tensor_parallelism: int
    state_means: np.ndarray
    state_stds: np.ndarray
    num_states: int = 6
    ttft_mean: Optional[float] = None
    ttft_std: Optional[float] = None
    tpot_mean: Optional[float] = None
    tpot_std: Optional[float] = None
    classifier_weights_path: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.model_name}-TP{self.tensor_parallelism}-{self.hardware.upper()}"


def load_performance_database(path: str = "model/config/performance_database.json") -> Dict:
    """Load unified performance database."""
    path = _resolve_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Performance database not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def load_model_config(
    model_name: str,
    hardware: str,
    tp: int,
    performance_db_path: str = "model/config/performance_database.json",
    weights_base_path: str = "model/gru_classifier_weights",
) -> ModelConfig:
    """Load complete model configuration from unified performance database."""
    performance_db_path = _resolve_path(performance_db_path)
    weights_base_path = _resolve_path(weights_base_path)
    
    perf_db = load_performance_database(performance_db_path)
    
    # Map model names to database format
    model_map = {
        "llama-3-8b": "llama-3.1_8b",
        "llama-3-70b": "llama-3.1_70b",
        "llama-3-405b": "llama-3.1_405b",
        "deepseek-r1-8b": "deepseek-r1-distill_8b",
        "deepseek-r1-70b": "deepseek-r1-distill_70b",
        "deepseek-r1-distill-8b": "deepseek-r1-distill_8b",
        "deepseek-r1-distill-70b": "deepseek-r1-distill_70b",
    }
    
    db_model_name = model_map.get(model_name, model_name)
    db_key = f"{db_model_name}_{hardware}_tp{tp}"
    
    if db_key not in perf_db:
        available = list(perf_db.keys())[:10]
        raise KeyError(f"Config not found: {db_key}. Available: {available}...")
    
    entry = perf_db[db_key]
    
    if "power_states" not in entry:
        raise ValueError(f"No power stats for {db_key}")
    
    power_stats = entry["power_states"]
    state_means = np.array(power_stats["state_means"])
    state_stds = np.array(power_stats["state_stds"])
    num_states = power_stats["num_states"]
    
    ttft_mean = ttft_std = tpot_mean = tpot_std = None
    if "ttft_model" in entry:
        ttft_mean = entry["ttft_model"]["summary_stats"]["mean_seconds"]
        ttft_std = entry["ttft_model"]["summary_stats"]["std_seconds"]
    if "tpot_distribution" in entry:
        tpot_mean = entry["tpot_distribution"]["mean"]
        tpot_std = entry["tpot_distribution"]["std"]
    
    weights_path = os.path.join(weights_base_path, f"{model_name}_{hardware}_tp{tp}.pt")
    
    return ModelConfig(
        model_name=model_name,
        hardware=hardware,
        tensor_parallelism=tp,
        state_means=state_means,
        state_stds=state_stds,
        num_states=num_states,
        ttft_mean=ttft_mean,
        ttft_std=ttft_std,
        tpot_mean=tpot_mean,
        tpot_std=tpot_std,
        classifier_weights_path=weights_path,
    )
