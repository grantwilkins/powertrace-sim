from typing import Optional

import numpy as np
import torch
from classifiers.gru import GRUClassifier
from core.dataset import PowerTraceDataset


class SmoothingSampler:
    def __init__(self, state_stats: dict):
        self.state_stats = state_stats

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
        chosen_states = np.array([np.random.choice(K, p=p) for p in probs])
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
