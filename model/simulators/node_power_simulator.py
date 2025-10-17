"""
Node-level power simulator for data center simulation.
Generates power traces for individual compute nodes based on LLM serving workloads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from model.classifiers.gru import GRUClassifier
from model.core.unified_config import ModelConfig, load_model_config
from model.core.utils import load_classifier
from model.predictors.smooth_sampler import SmoothingSampler
from model.simulators.arrival_simulator import (
    ServeGenPowerSimulator,
    ServingConfig,
)


@dataclass
class PowerTrace:
    """Structured output for node power consumption over time."""

    node_id: str
    timestamps: np.ndarray  # Simulation time in seconds
    power_watts: np.ndarray  # Power consumption in watts
    metadata: Dict  # Additional info (model, hardware, tp, etc.)

    def to_dict(self) -> Dict:
        """Convert to dictionary for easy serialization."""
        return {
            "node_id": self.node_id,
            "timestamps": self.timestamps.tolist(),
            "power_watts": self.power_watts.tolist(),
            "metadata": self.metadata,
        }

    def save_npz(self, path: str):
        """Save power trace to NPZ file."""
        np.savez(
            path,
            node_id=self.node_id,
            timestamps=self.timestamps,
            power_watts=self.power_watts,
            **self.metadata,
        )


@dataclass
class NodeConfig:
    """Configuration for a compute node."""

    node_id: str
    model_name: str  # e.g., "llama-3-8b", "deepseek-r1-distill-70b"
    hardware: str  # e.g., "a100", "h100"
    tensor_parallelism: int
    weights_base_path: str = "../gru_classifier_weights/"

    def get_weights_path(self) -> str:
        """Construct path to classifier weights."""
        # Format: <model_name>_<hardware>_tp<tp>.pt
        filename = f"{self.model_name}_{self.hardware}_tp{self.tensor_parallelism}.pt"
        return os.path.join(self.weights_base_path, filename)

    def __str__(self) -> str:
        return f"Node({self.node_id}:{self.model_name}-TP{self.tensor_parallelism}-{self.hardware.upper()})"


class NodePowerSimulator:
    """
    Simulates power consumption for a single compute node serving LLM requests.
    Integrates with ServeGen arrival simulator and GRU-based power models.
    """

    def __init__(
        self,
        node_config: NodeConfig,
        state_means: np.ndarray,
        state_stds: np.ndarray,
        num_states: int = 6,
        feature_dim: int = 6,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize node power simulator.

        Args:
            node_config: Node configuration (model, hardware, TP)
            state_means: Power state means from training (shape: [K])
            state_stds: Power state standard deviations (shape: [K])
            num_states: Number of power states (K)
            feature_dim: Input feature dimensions (Dx)
            device: PyTorch device for inference
        """
        self.node_config = node_config
        self.num_states = num_states
        self.feature_dim = feature_dim

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Load classifier
        weights_path = node_config.get_weights_path()
        self.classifier = load_classifier(
            weights_path, device=self.device, Dx=feature_dim, K=num_states
        )

        # Store state statistics
        self.state_means = state_means
        self.state_stds = state_stds

        # Initialize smoothing sampler
        self.sampler = SmoothingSampler(state_stats={})

        print(f"Initialized {node_config}")

    def generate_power_trace(
        self,
        feature_matrix: np.ndarray,
        timeline: Dict[str, np.ndarray],
        dt: float = 0.25,
        smoothing_window: int = 5,
        target_duration: Optional[float] = None,
    ) -> PowerTrace:
        """
        Generate power trace from workload features.

        Args:
            feature_matrix: (T, Dx) z-scored feature matrix from arrival simulator
            timeline: Timeline dictionary with timestamps and other metadata
            dt: Time step in seconds
            smoothing_window: Window size for power smoothing
            target_duration: Optional target duration to extend/clip trace

        Returns:
            PowerTrace with node_id, timestamps, and power values
        """
        # Sample power from classifier
        timestamps, power_watts, states = self.sampler.sample_power(
            net=self.classifier,
            mu=self.state_means,
            sigma=self.state_stds,
            schedule_x=feature_matrix,
            dt=dt,
            smoothing_window=smoothing_window,
        )

        # Extend or clip to target duration if specified
        if target_duration is not None:
            current_duration = timestamps[-1]
            if current_duration < target_duration:
                # Extend: repeat last power value
                n_extend = int((target_duration - current_duration) / dt)
                timestamps = np.concatenate(
                    [
                        timestamps,
                        timestamps[-1] + np.arange(1, n_extend + 1) * dt,
                    ]
                )
                power_watts = np.concatenate(
                    [power_watts, np.full(n_extend, power_watts[-1])]
                )
            elif current_duration > target_duration:
                # Clip to target duration
                mask = timestamps <= target_duration
                timestamps = timestamps[mask]
                power_watts = power_watts[mask]

        # Create metadata
        metadata = {
            "model_name": self.node_config.model_name,
            "hardware": self.node_config.hardware,
            "tensor_parallelism": self.node_config.tensor_parallelism,
            "time_step": dt,
            "num_states": self.num_states,
            "smoothing_window": smoothing_window,
            "mean_power_watts": float(np.mean(power_watts)),
            "max_power_watts": float(np.max(power_watts)),
            "total_energy_joules": float(np.trapz(power_watts, timestamps)),
        }

        return PowerTrace(
            node_id=self.node_config.node_id,
            timestamps=timestamps,
            power_watts=power_watts,
            metadata=metadata,
        )

    def simulate_from_serving_config(
        self,
        serving_config: ServingConfig,
        workload_category: str = "language",
        model_type: str = "m-large",
        duration: float = 3600,
        rate_requests_per_sec: float = 1.0,
        time_window: Optional[str] = None,
        seed: int = 0,
        dt: float = 0.25,
        smoothing_window: int = 5,
    ) -> PowerTrace:
        """
        End-to-end simulation: generate workload and predict power.

        Args:
            serving_config: LLM serving configuration
            workload_category: ServeGen category ("language", "reason", "multimodal")
            model_type: ServeGen model type ("m-small", "m-mid", "m-large")
            duration: Simulation duration in seconds
            rate_requests_per_sec: Request arrival rate
            time_window: Optional time window for realistic patterns
            seed: Random seed
            dt: Time step for power trace
            smoothing_window: Smoothing window size

        Returns:
            PowerTrace for this node
        """
        # Generate workload using arrival simulator
        power_simulator = ServeGenPowerSimulator(serving_config)
        simulation_data = power_simulator.generate_power_simulation_data(
            category=workload_category,
            model_type=model_type,
            duration=duration,
            rate_requests_per_sec=rate_requests_per_sec,
            time_window=time_window,
            seed=seed,
        )

        # Extract feature matrix and timeline
        feature_matrix = simulation_data["feature_matrix"]
        timeline = simulation_data["timeline"]

        # Generate power trace
        return self.generate_power_trace(
            feature_matrix=feature_matrix,
            timeline=timeline,
            dt=dt,
            smoothing_window=smoothing_window,
            target_duration=duration,
        )


def create_node_simulator(
    node_id: str,
    model_name: str,
    hardware: str,
    tp: int,
    weights_base_path: str = "model/gru_classifier_weights",
    device: Optional[torch.device] = None,
) -> NodePowerSimulator:
    """
    Create NodePowerSimulator from unified configuration (recommended).
    Loads ALL config from performance_database.json (TTFT, TPOT, power stats).

    Args:
        node_id: Unique node identifier
        model_name: Model name (e.g., "llama-3-8b", "deepseek-r1-distill-70b")
        hardware: Hardware type (e.g., "a100", "h100")
        tp: Tensor parallelism
        weights_base_path: Base path for classifier weights
        device: PyTorch device

    Returns:
        Initialized NodePowerSimulator
    """
    # Load unified configuration from performance_database.json
    model_config = load_model_config(
        model_name=model_name,
        hardware=hardware,
        tp=tp,
        weights_base_path=weights_base_path,
    )

    # Create node config
    node_config = NodeConfig(
        node_id=node_id,
        model_name=model_name,
        hardware=hardware,
        tensor_parallelism=tp,
        weights_base_path=weights_base_path,
    )

    return NodePowerSimulator(
        node_config=node_config,
        state_means=model_config.state_means,
        state_stds=model_config.state_stds,
        num_states=model_config.num_states,
        feature_dim=6,
        device=device,
    )


def create_node_simulator_from_dataset(
    node_config: NodeConfig,
    dataset_path: str,
    device: Optional[torch.device] = None,
) -> NodePowerSimulator:
    """
    LEGACY: Create NodePowerSimulator from training dataset.
    Prefer using create_node_simulator() with pre-extracted stats instead.

    Args:
        node_config: Node configuration
        dataset_path: Path to training NPZ file (for state statistics)
        device: PyTorch device

    Returns:
        Initialized NodePowerSimulator
    """
    from model.core.dataset import PowerTraceDataset

    # Load dataset to get state statistics
    dataset = PowerTraceDataset(dataset_path, K=6)

    tp = node_config.tensor_parallelism
    if tp not in dataset.mu:
        available_tps = list(dataset.mu.keys())
        raise ValueError(f"TP={tp} not found in dataset. Available: {available_tps}")

    return NodePowerSimulator(
        node_config=node_config,
        state_means=dataset.mu[tp],
        state_stds=dataset.sigma[tp],
        num_states=6,
        feature_dim=6,
        device=device,
    )


# Convenience function for quick simulation
def quick_node_simulation(
    node_id: str,
    model: str = "llama-3-8b",
    hardware: str = "a100",
    tp: int = 1,
    workload: str = "language",
    duration: float = 3600,
    rate: float = 1.0,
    weights_base_path: str = "model/gru_classifier_weights",
) -> PowerTrace:
    """
    Quick end-to-end node simulation using unified configuration.

    **Single source of truth**: Loads everything from performance_database.json
    (TTFT, TPOT, power state statistics).

    Args:
        node_id: Unique node identifier
        model: Model name (e.g., "llama-3-8b", "deepseek-r1-distill-70b")
        hardware: Hardware type ("a100", "h100")
        tp: Tensor parallelism
        workload: Workload type ("language", "reason", "multimodal")
        duration: Simulation duration in seconds
        rate: Request rate (requests/sec)
        weights_base_path: Path to classifier weights directory

    Returns:
        PowerTrace for the simulated node
    """
    from model.simulators.arrival_simulator import (
        create_deepseek_config,
        create_llama_config,
    )

    # Create simulator from unified config (loads from performance_database.json)
    simulator = create_node_simulator(
        node_id=node_id,
        model_name=model,
        hardware=hardware,
        tp=tp,
        weights_base_path=weights_base_path,
    )

    # Create serving config
    if "llama" in model.lower():
        size = int(model.split("-")[-1].replace("b", ""))
        serving_config = create_llama_config(size, tp, hardware.upper())
    elif "deepseek" in model.lower():
        size = int(model.split("-")[-1].replace("b", ""))
        serving_config = create_deepseek_config(size, tp, hardware.upper())
    else:
        raise ValueError(f"Unknown model: {model}")

    # Run simulation
    return simulator.simulate_from_serving_config(
        serving_config=serving_config,
        workload_category=workload,
        duration=duration,
        rate_requests_per_sec=rate,
    )
