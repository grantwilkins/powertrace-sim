"""
Unified configuration system - single source of truth for model configurations.
Combines power state stats and performance data from existing JSON files.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


def _resolve_path(path: str) -> str:
    """
    Resolve path relative to project root.
    Handles execution from different directories.
    """
    if os.path.isabs(path):
        return path

    # Try the path as-is first
    if os.path.exists(path):
        return path

    # Try relative to current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # Try relative to project root
    resolved = os.path.join(project_root, path)
    if os.path.exists(resolved):
        return resolved

    # Return original path (will fail later with clear error)
    return path


@dataclass
class ModelConfig:
    """Complete configuration for a model/hardware/TP combination."""

    model_name: str
    hardware: str
    tensor_parallelism: int

    # Power state statistics
    state_means: np.ndarray  # Shape: (K,)
    state_stds: np.ndarray  # Shape: (K,)
    num_states: int = 6

    # Performance data (from performance_database.json)
    ttft_mean: Optional[float] = None
    ttft_std: Optional[float] = None
    tpot_mean: Optional[float] = None
    tpot_std: Optional[float] = None

    # Paths
    classifier_weights_path: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.model_name}-TP{self.tensor_parallelism}-{self.hardware.upper()}"


def load_model_config(
    model_name: str,
    hardware: str,
    tp: int,
    summary_data_dir: str = "model/summary_data",
    performance_db_path: str = "model/config/performance_database.json",
    weights_base_path: str = "gru_classifier_weights",
) -> ModelConfig:
    """
    Load complete model configuration from existing JSON files.

    Args:
        model_name: Model name (e.g., "llama-3-8b", "deepseek-r1-distill-70b")
        hardware: Hardware type (e.g., "a100", "h100")
        tp: Tensor parallelism
        summary_data_dir: Directory containing model summary JSON files
        performance_db_path: Path to performance database JSON
        weights_base_path: Base path for classifier weights

    Returns:
        ModelConfig with all necessary data
    """
    # Resolve paths relative to project root
    summary_data_dir = _resolve_path(summary_data_dir)
    performance_db_path = _resolve_path(performance_db_path)
    weights_base_path = _resolve_path(weights_base_path)

    # Load power state statistics from summary file
    summary_file = os.path.join(
        summary_data_dir, f"model_summary_{model_name}_{hardware}.json"
    )

    if not os.path.exists(summary_file):
        raise FileNotFoundError(
            f"Summary file not found: {summary_file}\n"
            f"Available models: {list_available_models(summary_data_dir)}"
        )

    with open(summary_file, "r") as f:
        summary_data = json.load(f)

    # Get stats for this TP configuration
    tp_key = str(tp)
    if tp_key not in summary_data:
        available_tps = list(summary_data.keys())
        raise ValueError(
            f"TP={tp} not found for {model_name}_{hardware}. "
            f"Available TPs: {available_tps}"
        )

    tp_data = summary_data[tp_key]
    state_means = np.array(tp_data["mu_values"])
    state_stds = np.array(tp_data["sigma_values"])

    # Load performance data from performance database
    perf_db = load_performance_database(performance_db_path)

    # Try to find matching performance entry
    # Map model names to performance database format
    model_name_map = {
        "llama-3-8b": "llama-3.1_8b",
        "llama-3-70b": "llama-3.1_70b",
        "llama-3-405b": "llama-3.1_405b",
        "deepseek-r1-8b": "deepseek-r1-distill_8b",
        "deepseek-r1-70b": "deepseek-r1-distill_70b",
        "deepseek-r1-distill-8b": "deepseek-r1-distill_8b",
        "deepseek-r1-distill-70b": "deepseek-r1-distill_70b",
    }

    perf_model_name = model_name_map.get(model_name, model_name)
    perf_key = f"{perf_model_name}_{hardware}_tp{tp}"

    ttft_mean = ttft_std = tpot_mean = tpot_std = None
    if perf_key in perf_db:
        perf_data = perf_db[perf_key]
        ttft_mean = perf_data["ttft_model"]["summary_stats"]["mean_seconds"]
        ttft_std = perf_data["ttft_model"]["summary_stats"]["std_seconds"]
        tpot_mean = perf_data["tpot_distribution"]["mean"]
        tpot_std = perf_data["tpot_distribution"]["std"]

    # Construct classifier weights path
    weights_path = os.path.join(
        weights_base_path, f"{model_name}_{hardware}_tp{tp}.pt"
    )

    return ModelConfig(
        model_name=model_name,
        hardware=hardware,
        tensor_parallelism=tp,
        state_means=state_means,
        state_stds=state_stds,
        num_states=len(state_means),
        ttft_mean=ttft_mean,
        ttft_std=ttft_std,
        tpot_mean=tpot_mean,
        tpot_std=tpot_std,
        classifier_weights_path=weights_path,
    )


def load_performance_database(
    performance_db_path: str = "model/config/performance_database.json",
) -> Dict:
    """Load performance database."""
    if not os.path.exists(performance_db_path):
        return {}

    with open(performance_db_path, "r") as f:
        return json.load(f)


def list_available_models(summary_data_dir: str = "model/summary_data") -> list:
    """List available model configurations."""
    import glob

    pattern = os.path.join(summary_data_dir, "model_summary_*.json")
    files = glob.glob(pattern)

    models = []
    for f in files:
        basename = os.path.basename(f)
        # Extract model_name and hardware from filename
        # Format: model_summary_{model_name}_{hardware}.json
        parts = basename.replace("model_summary_", "").replace(".json", "")
        models.append(parts)

    return sorted(models)


def get_all_configs(
    summary_data_dir: str = "model/summary_data",
    performance_db_path: str = "model/config/performance_database.json",
    weights_base_path: str = "gru_classifier_weights",
) -> Dict[str, ModelConfig]:
    """
    Load all available model configurations.

    Returns:
        Dictionary mapping config key to ModelConfig
        Key format: "{model_name}_{hardware}_tp{tp}"
    """
    configs = {}
    available = list_available_models(summary_data_dir)

    for model_hw in available:
        parts = model_hw.rsplit("_", 1)
        if len(parts) != 2:
            continue

        model_name, hardware = parts
        summary_file = os.path.join(
            summary_data_dir, f"model_summary_{model_name}_{hardware}.json"
        )

        with open(summary_file, "r") as f:
            summary_data = json.load(f)

        # Load all TP configurations for this model/hardware
        for tp_key in summary_data.keys():
            tp = int(tp_key)
            config_key = f"{model_name}_{hardware}_tp{tp}"

            try:
                config = load_model_config(
                    model_name=model_name,
                    hardware=hardware,
                    tp=tp,
                    summary_data_dir=summary_data_dir,
                    performance_db_path=performance_db_path,
                    weights_base_path=weights_base_path,
                )
                configs[config_key] = config
            except Exception as e:
                print(f"Warning: Could not load {config_key}: {e}")

    return configs


if __name__ == "__main__":
    # Demo usage
    print("Available model configurations:")
    print("=" * 60)

    configs = get_all_configs()
    for key, config in sorted(configs.items()):
        print(f"\n{key}:")
        print(f"  State means: {config.state_means}")
        print(f"  State stds: {config.state_stds}")
        if config.ttft_mean:
            print(f"  TTFT: {config.ttft_mean:.4f}s ± {config.ttft_std:.4f}s")
            print(f"  TPOT: {config.tpot_mean:.4f}s ± {config.tpot_std:.4f}s")
        print(f"  Weights: {config.classifier_weights_path}")
