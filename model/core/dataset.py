import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from core.utils import make_schedule_matrix
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from torch.utils.data import Dataset


class PowerTraceDataset(Dataset):
    """
    Returns  (x_t  ,  y_t  ,  z_t)  for one whole trace.
              (T,3)   (T,1)   (T,)
    """

    def __init__(self, npz_file: str, K: int = 6):
        d = np.load(npz_file, allow_pickle=True)
        self.traces: List[Dict[str, np.ndarray]] = []
        self.tp_all: List[int] = []
        # For specific naming convention of the npz file:
        # *_<llm_name>_<hw_accelerator>*.npz
        self.hw_accelerator = npz_file.split("/")[-1].split("_")[2]
        self.llm_name = npz_file.split("/")[-1].split("_")[1]
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
            # Convert to float64 for better numerical precision and add regularization
            y_concat_64 = y_concat.astype(np.float64)
            model = GaussianMixture(
                n_components=K,
                covariance_type="diag",
                n_init=10,
                random_state=0,
                reg_covar=1e-6
            ).fit(y_concat_64)
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


@dataclass
class PowerInferenceConfig:
    """Configuration for power inference from JSON."""

    model_name: str
    hardware: str
    tensor_parallelism: int
    prefill_throughput: float
    decode_throughput: float
    ttft: float
    num_states: int
    state_means: np.ndarray
    state_stds: np.ndarray
    input_token_samples: np.ndarray
    output_token_samples: np.ndarray
    model_weights_path: str
    feature_dimensions: int

    @classmethod
    def from_json_file(cls, json_path: str) -> "PowerInferenceConfig":
        """Load configuration from JSON file."""
        with open(json_path, "r") as f:
            data = json.load(f)

        return cls(
            model_name=data["model"]["name"],
            hardware=data["model"]["hardware"],
            tensor_parallelism=data["model"]["tensor_parallelism"],
            prefill_throughput=data["throughput"]["prefill_tokens_per_second"],
            decode_throughput=data["throughput"]["decode_tokens_per_second"],
            ttft=data["throughput"]["time_to_first_token"],
            num_states=data["power_states"]["num_states"],
            state_means=np.array(data["power_states"]["state_means"]),
            state_stds=np.array(data["power_states"]["state_stds"]),
            clustering_method=data["power_states"]["clustering_method"],
            model_weights_path=data["inference"]["model_weights_path"],
            feature_dimensions=data["inference"]["feature_dimensions"],
        )

    def to_json_file(self, json_path: str):
        """Save configuration to JSON file."""
        data = {
            "model": {
                "name": self.model_name,
                "hardware": self.hardware,
                "tensor_parallelism": self.tensor_parallelism,
            },
            "throughput": {
                "prefill_tokens_per_second": self.prefill_throughput,
                "decode_tokens_per_second": self.decode_throughput,
                "time_to_first_token": self.ttft,
            },
            "power_states": {
                "num_states": self.num_states,
                "clustering_method": self.clustering_method,
                "state_means": self.state_means.tolist(),
                "state_stds": self.state_stds.tolist(),
            },
            "inference": {
                "model_weights_path": self.model_weights_path,
                "feature_dimensions": self.feature_dimensions,
            },
        }

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)


def extract_config_from_npz(
    npz_path: str, weights_path: str, tp: int = 1
) -> PowerInferenceConfig:
    """Extract configuration from existing NPZ file to create JSON config."""
    from core.dataset import PowerTraceDataset

    dataset = PowerTraceDataset(npz_path, K=6)
    data = np.load(npz_path, allow_pickle=True)

    mask = data["tensor_parallelism"] == tp

    config = PowerInferenceConfig(
        model_name=dataset.llm_name,
        hardware=dataset.hw_accelerator,
        tensor_parallelism=tp,
        prefill_throughput=np.mean(data["prefill_throughputs"][mask].flatten()),
        decode_throughput=np.mean(data["decode_throughputs"][mask].flatten()),
        ttft=np.mean(data["prefill_times"][mask].flatten()),
        num_states=6,
        state_means=dataset.mu[tp],
        state_stds=dataset.sigma[tp],
        model_weights_path=weights_path,
        feature_dimensions=6,
    )

    return config
