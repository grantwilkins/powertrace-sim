from typing import Dict, List

import numpy as np


class SmoothingSampler:
    def __init__(self, dataset, tp=None):
        self.state_stats = {}
        self.tp = tp

        for k in range(6):
            powers = []
            for tr in dataset.traces:
                if tp is not None and tr.get("tp", None) != tp:
                    continue

                mask = tr["z"] == k
                if mask.sum() > 0:
                    powers.extend(tr["y"][mask])

            if powers:
                self.state_stats[k] = {
                    "median": np.median(powers),
                    "mad": np.median(np.abs(powers - np.median(powers))),
                    "iqr": np.percentile(powers, 75) - np.percentile(powers, 25),
                    "tp": tp,
                }

    def smooth_sample(self, time, power, states, smoothing_window=5, tp=None):
        tp = tp if tp is not None else self.tp
        smoothed = power.copy()
        for i in range(len(power)):
            start = max(0, i - smoothing_window // 2)
            end = min(len(power), i + smoothing_window // 2 + 1)
            current_state = states[i]
            same_state_mask = states[start:end] == current_state

            if same_state_mask.sum() > 0:
                window_values = power[start:end][same_state_mask]
                smoothed[i] = np.median(window_values)

        return time, smoothed, states
