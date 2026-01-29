import json
from typing import TYPE_CHECKING, Dict, Optional, Union

import numpy as np
import torch

from model.classifiers.gru import GRUClassifier

if TYPE_CHECKING:
    # Import only for type checking to avoid heavy deps at runtime
    from model.core.dataset import PowerTraceDataset


class SmoothingSampler:
    """
    Sampler for power traces that applies smoothing.

    Can be initialized either with:
    1. A PowerTraceDataset (legacy)
    2. GMM parameters loaded from performance database (recommended for inference)
    """

    def __init__(self, state_stats: Union["PowerTraceDataset", Dict] = None):
        """
        Initialize sampler with either a dataset or GMM parameters.

        Args:
            state_stats: Either a PowerTraceDataset or a dict with GMM parameters
                        Dict format: {tp: {"mu": [...], "sigma": [...]}}
        """
        if isinstance(state_stats, dict):
            self.state_stats = state_stats
            self.mu = {}
            self.sigma = {}
            for tp, params in state_stats.items():
                self.mu[tp] = np.array(params["mu"])
                self.sigma[tp] = np.array(params["sigma"])
        elif state_stats is not None:
            # Lazy import to avoid pulling sklearn/scipy when not needed
            try:
                from model.core.dataset import (
                    PowerTraceDataset as _PowerTraceDataset,  # type: ignore
                )

                if isinstance(state_stats, _PowerTraceDataset):
                    self.state_stats = state_stats
                    self.mu = state_stats.mu
                    self.sigma = state_stats.sigma
                else:
                    raise ValueError(
                        "state_stats must be either a dict or PowerTraceDataset instance"
                    )
            except Exception as e:
                raise ValueError(
                    "state_stats must be a dict with GMM parameters; "
                    "PowerTraceDataset import failed or unavailable"
                ) from e
        else:
            raise ValueError(
                "state_stats must be either PowerTraceDataset or dict with GMM parameters"
            )

    @classmethod
    def from_performance_database(
        cls,
        database_path: str,
        model_name: str,
        model_size_b: int,
        hardware: str,
        tensor_parallelism: int,
    ) -> "SmoothingSampler":
        """
        Create a SmoothingSampler from performance database.

        Args:
            database_path: Path to performance_database.json
            model_name: Model name (e.g., "deepseek-r1-distill", "llama-3.1")
            model_size_b: Model size in billions
            hardware: Hardware type ("A100", "H100")
            tensor_parallelism: Tensor parallelism value

        Returns:
            SmoothingSampler instance
        """
        with open(database_path, "r") as f:
            database = json.load(f)

        key = f"{model_name}_{model_size_b}b_{hardware.lower()}_tp{tensor_parallelism}"

        if key not in database:
            raise ValueError(f"Configuration {key} not found in database")

        config = database[key]

        if "gmm_parameters" not in config:
            raise ValueError(
                f"No GMM parameters found for {key}. Run extract_performance_stats.py with --training_data_dir"
            )

        # Create dict with single TP entry
        gmm_params = {tensor_parallelism: config["gmm_parameters"]}

        return cls(gmm_params)

    def sample_power(
        self,
        net: GRUClassifier,
        mu: np.ndarray,
        sigma: np.ndarray,
        schedule_x: np.ndarray,
        dt: float = 0.25,
        smoothing_window: int = 5,
    ):
        """
        Sample power values from the model and apply smoothing automatically.

        Args:
            net: GRU classifier model
            mu: mean values for each state
            sigma: standard deviation values for each state
            schedule_x: (T,Dx) – already z-scored feature matrix on desired Δt grid
            dt: time step
            smoothing_window: window size for smoothing

        Returns:
            t: time array
            watts: smoothed power values
            z: chosen states
        """
        net.eval()
        with torch.no_grad():
            logits = net(torch.from_numpy(schedule_x[None]).float()).squeeze(0)  # (T,K)
            probs = torch.softmax(logits, -1).cpu().numpy()
        K = probs.shape[1]
        # chosen_states = np.array([np.random.choice(K, p=p) for p in probs])
        chosen_states = np.argmax(probs, axis=1)
        watts = np.random.normal(mu[chosen_states], sigma[chosen_states])

        t = np.arange(len(chosen_states)) * dt

        smoothed_power = watts.copy()
        for i in range(len(watts)):
            start = max(0, i - smoothing_window // 2)
            end = min(len(watts), i + smoothing_window // 2 + 1)
            current_state = chosen_states[i]
            same_state_mask = chosen_states[start:end] == current_state

            if same_state_mask.sum() > 0:
                window_values = watts[start:end][same_state_mask]
                smoothed_power[i] = np.median(window_values)

        return t, smoothed_power, chosen_states
