"""
Utility for managing power state statistics independently from training data.
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class PowerStateStats:
    """Power state statistics for a specific model/hardware/TP configuration."""

    model_name: str
    hardware: str
    tensor_parallelism: int
    num_states: int
    state_means: np.ndarray  # Shape: (K,)
    state_stds: np.ndarray  # Shape: (K,)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": str(self.model_name),
            "hardware": str(self.hardware),
            "tensor_parallelism": int(self.tensor_parallelism),
            "num_states": int(self.num_states),
            "state_means": self.state_means.tolist(),
            "state_stds": self.state_stds.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PowerStateStats":
        """Load from dictionary."""
        return cls(
            model_name=data["model_name"],
            hardware=data["hardware"],
            tensor_parallelism=data["tensor_parallelism"],
            num_states=data["num_states"],
            state_means=np.array(data["state_means"]),
            state_stds=np.array(data["state_stds"]),
        )

    def save_json(self, path: str):
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "PowerStateStats":
        """Load from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_npz(self, path: str):
        """Save to NPZ file."""
        np.savez(
            path,
            model_name=self.model_name,
            hardware=self.hardware,
            tensor_parallelism=self.tensor_parallelism,
            num_states=self.num_states,
            state_means=self.state_means,
            state_stds=self.state_stds,
        )

    @classmethod
    def load_npz(cls, path: str) -> "PowerStateStats":
        """Load from NPZ file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            model_name=str(data["model_name"]),
            hardware=str(data["hardware"]),
            tensor_parallelism=int(data["tensor_parallelism"]),
            num_states=int(data["num_states"]),
            state_means=data["state_means"],
            state_stds=data["state_stds"],
        )


def extract_stats_from_dataset(
    dataset_path: str, output_dir: str = "../power_state_stats/"
):
    """
    Extract state statistics from training dataset and save individually.

    Args:
        dataset_path: Path to training NPZ file
        output_dir: Directory to save individual stat files
    """
    from model.core.dataset import PowerTraceDataset

    # Load dataset
    dataset = PowerTraceDataset(dataset_path, K=6)

    os.makedirs(output_dir, exist_ok=True)

    # Save stats for each TP configuration
    for tp in dataset.mu.keys():
        stats = PowerStateStats(
            model_name=dataset.llm_name,
            hardware=dataset.hw_accelerator,
            tensor_parallelism=tp,
            num_states=6,
            state_means=dataset.mu[tp],
            state_stds=dataset.sigma[tp],
        )

        # Save as JSON
        json_path = os.path.join(
            output_dir,
            f"{stats.model_name}_{stats.hardware}_tp{tp}_stats.json",
        )
        stats.save_json(json_path)
        print(f"Saved {json_path}")

        # Also save as NPZ for faster loading
        npz_path = os.path.join(
            output_dir,
            f"{stats.model_name}_{stats.hardware}_tp{tp}_stats.npz",
        )
        stats.save_npz(npz_path)
        print(f"Saved {npz_path}")


def load_stats(
    model_name: str,
    hardware: str,
    tp: int,
    stats_dir: str = "../power_state_stats/",
    format: str = "json",
) -> PowerStateStats:
    """
    Load power state statistics.

    Args:
        model_name: Model name (e.g., "llama-3-8b")
        hardware: Hardware type (e.g., "a100")
        tp: Tensor parallelism
        stats_dir: Directory containing stats files
        format: File format ("json" or "npz")

    Returns:
        PowerStateStats object
    """
    filename = f"{model_name}_{hardware}_tp{tp}_stats.{format}"
    path = os.path.join(stats_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Stats file not found: {path}\n"
            f"Run extract_stats_from_dataset() to generate stats files."
        )

    if format == "json":
        return PowerStateStats.load_json(path)
    elif format == "npz":
        return PowerStateStats.load_npz(path)
    else:
        raise ValueError(f"Unknown format: {format}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract power state statistics from training data"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to training NPZ file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../power_state_stats/",
        help="Output directory for stats files",
    )
    args = parser.parse_args()

    extract_stats_from_dataset(args.dataset, args.output_dir)
