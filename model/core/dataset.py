from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from torch.utils.data import Dataset

from model.core.utils import histogram_requests, make_schedule_matrix


class PowerTraceDataset(Dataset):
    """
    Returns  (x_t  ,  y_t  ,  z_t)  for one whole trace.
              (T,3)   (T,1)   (T,)
    """

    def __init__(self, npz_file: str, K: int = 6, use_gmm: bool = False):
        d = np.load(npz_file, allow_pickle=True)
        self.traces: List[Dict[str, np.ndarray]] = []
        self.tp_all: List[int] = []
        tp_array = d["tensor_parallelism"]

        n_exp = d["timestamps"].shape[0]
        for i in range(n_exp):
            bin_mask = d["timestamps"][i] > 0
            req_mask = d["request_timestamps"][i] > 0

            trace = dict(
                timestamps=d["timestamps"][i][bin_mask],
                power=d["power_traces"][i][bin_mask].astype("float32"),
                prefill_tokens=d["prefill_tokens"][i][bin_mask],
                decode_tokens=d["decode_tokens"][i][bin_mask],
                active_requests=d["active_requests"][i][bin_mask],
                request_ts=d["request_timestamps"][i][req_mask],
                input_tokens=d["input_tokens"][i][req_mask],
                output_tokens=d["output_tokens"][i][req_mask],
            )

            trace["x"] = make_schedule_matrix(trace)
            trace["y"] = trace["power"][:, None]  # (T,1)
            self.traces.append(trace)
            self.tp_all.append(int(tp_array[i]))

        self.state_labels: Dict[int, np.ndarray] = {}
        unique_tp = np.unique(self.tp_all)
        for tp in unique_tp:
            y_concat = np.concatenate(
                [tr["y"] for tr, tp_i in zip(self.traces, self.tp_all) if tp_i == tp]
            ).reshape(-1, 1)

            if use_gmm:
                model = GaussianMixture(
                    n_components=K, covariance_type="diag", n_init=10, random_state=0
                ).fit(y_concat)
            else:
                model = KMeans(n_clusters=K, n_init=10, random_state=0).fit(y_concat)
            self.state_labels[tp] = model

        for tr, tp in zip(self.traces, self.tp_all):
            model = self.state_labels[tp]
            tr["z"] = model.predict(tr["y"]).astype("int64")

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tr = self.traces[idx]
        return (
            torch.from_numpy(tr["x"]),
            torch.from_numpy(tr["y"]),
            torch.from_numpy(tr["z"]),
        )
