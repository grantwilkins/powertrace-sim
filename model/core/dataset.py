import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from numpy import ndarray
from sklearn.mixture import GaussianMixture
from torch.utils.data import Dataset

from .utils import make_schedule_matrix


class PowerTraceDataset(Dataset):
    """
    Returns (x_t, y_t, z_t) for one whole trace.
    (T,3) (T,1) (T,)
    """

    def __init__(self, npz_file: str, K: int = 6):
        d = np.load(npz_file, allow_pickle=True)
        self.traces: List[Dict[str, np.ndarray]] = []
        self.tp_all: List[int] = []

        self.hw_accelerator = npz_file.split("/")[-1].split("_")[2]
        self.llm_name = npz_file.split("/")[-1].split("_")[1]

        tp_array = d["tensor_parallelism"]
        poisson_rate_array = d.get("poisson_rate", None)
        n_exp = d["timestamps"].shape[0]

        for i in range(n_exp):
            bin_mask = d["timestamps"][i] > 0
            req_mask = d["request_timestamps"][i] > 0

            arrival_rate = (
                float(poisson_rate_array[i]) if poisson_rate_array is not None else None
            )

            trace = dict(
                timestamps=d["timestamps"][i][bin_mask],
                power=d["power_traces"][i][bin_mask].astype("float32"),
                prefill_tokens=d["prefill_tokens"][i][bin_mask],
                decode_tokens=d["decode_tokens"][i][bin_mask],
                active_requests=d["active_requests"][i][bin_mask],
                request_ts=d["request_timestamps"][i][req_mask],
                input_tokens=d["input_tokens"][i][req_mask],
                output_tokens=d["output_tokens"][i][req_mask],
                # arrival_rate=arrival_rate,
            )
            trace["x"] = make_schedule_matrix(
                trace, arrival_rate=arrival_rate, add_diff_features=True
            )
            trace["y"] = trace["power"][:, None]
            self.traces.append(trace)
            self.tp_all.append(int(tp_array[i]))

        self.state_labels: Dict[int, np.ndarray] = {}
        unique_tp = np.unique(self.tp_all)

        for tp in unique_tp:
            y_concat = (
                np.concatenate(
                    [
                        tr["y"]
                        for tr, tp_i in zip(self.traces, self.tp_all)
                        if tp_i == tp
                    ]
                )
                .reshape(-1, 1)
                .astype(np.float64)
            )

            model = GaussianMixture(
                n_components=K,
                covariance_type="diag",
                n_init=10,
                random_state=0,
                reg_covar=1e-6,
            ).fit(y_concat)

            self.state_labels[tp] = model

        for tr, tp in zip(self.traces, self.tp_all):
            model = self.state_labels[tp]
            tr["z"] = model.predict(tr["y"]).astype("int64")

        self.mu, self.sigma = self.compute_state_statistics(K=K)

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tr = self.traces[idx]
        return (
            torch.from_numpy(tr["x"]),
            torch.from_numpy(tr["y"]),
            torch.from_numpy(tr["z"]),
        )

    def compute_state_statistics(self, K: int) -> Tuple[dict, dict]:
        mu = {}
        sigma = {}

        for tp in self.state_labels:
            model = self.state_labels[tp]
            mu[tp] = np.zeros(K)
            sigma[tp] = np.zeros(K)

            for k in range(K):
                mu[tp][k] = float(model.means_[k][0])
                sigma[tp][k] = float(np.sqrt(model.covariances_[k][0]))

        return mu, sigma
