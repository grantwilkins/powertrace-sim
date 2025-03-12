import os
import json
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import re


class GPUPowerDataProcessor:
    """
    Process and prepare GPU power trace data for training.
    Handles loading and aligning data from JSON config, request CSV, and power measurement CSV files.
    """

    def __init__(self, data_root_dir: str):
        """
        Initialize the data processor.

        Args:
            data_root_dir: Root directory containing experiment data folders
        """
        self.data_root_dir = data_root_dir

    def discover_experiment_folders(self) -> List[str]:
        """
        Discover all experiment folders in the data root directory.

        Returns:
            List of paths to experiment folders
        """
        experiment_folders = []

        for folder in glob.glob(os.path.join(self.data_root_dir, "*/")):
            json_files = glob.glob(os.path.join(folder, "*.json"))
            csv_files = glob.glob(os.path.join(folder, "*.csv"))

            if json_files and csv_files:
                experiment_folders.append(folder)

        print(f"Discovered {len(experiment_folders)} experiment folders")
        return experiment_folders

    def extract_model_size_in_billions(self, model_name: str) -> float:
        """
        Extract model size in billions from model name.

        Args:
            model_name: Name of the model (e.g., "llama-7b", "claude-3-70b", etc.)

        Returns:
            Model size in billions of parameters
        """
        match = re.search(r"(\d+)[bB]", model_name)
        if match:
            return float(match.group(1))

        print(
            f"Warning: Unable to extract model size from {model_name}, using default value of 13B"
        )
        return 13.0

    def parse_json_config(self, json_path: str) -> Dict:
        """
        Parse JSON configuration file.

        Args:
            json_path: Path to the JSON config file

        Returns:
            Dictionary with configuration parameters
        """
        with open(json_path, "r") as f:
            config = json.load(f)

        model_name = config.get("Model", "")
        model_size = self.extract_model_size_in_billions(model_name)

        is_reasoning = config.get("Reasoning", False)
        if not isinstance(is_reasoning, bool):
            is_reasoning = (
                "reasoning" in model_name.lower() or "opus" in model_name.lower()
            )

        is_h100 = config.get("GPU", "").lower().startswith("h100")
        poisson_rate = float(config.get("Poisson Arrival Rate", 1.0))
        tensor_parallelism = int(config.get("Tensor Parallel Size", 1))

        return {
            "model_name": model_name,
            "model_size": model_size,
            "is_reasoning": is_reasoning,
            "is_h100": is_h100,
            "poisson_rate": poisson_rate,
            "tensor_parallelism": tensor_parallelism,
        }

    def parse_request_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Parse request CSV file.

        Args:
            csv_path: Path to the request CSV file

        Returns:
            DataFrame with request data
        """
        try:
            requests_df = pd.read_csv(csv_path)
            if "Request Time" in requests_df.columns and isinstance(
                requests_df["Request Time"].iloc[0], str
            ):
                requests_df["Request Time"] = pd.to_datetime(
                    requests_df["Request Time"]
                )

            return requests_df
        except Exception as e:
            print(f"Error parsing request CSV {csv_path}: {e}")
            return pd.DataFrame()

    def parse_power_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Parse NVIDIA power consumption CSV file.

        Args:
            csv_path: Path to the power CSV file

        Returns:
            DataFrame with power measurements
        """
        try:
            power_df = pd.read_csv(csv_path, skipinitialspace=True)
            if "timestamp" not in power_df.columns[0].lower():
                power_df = pd.read_csv(
                    csv_path,
                    names=["timestamp", "power_draw", "gpu_util", "memory_util"],
                    skiprows=1,
                    skipinitialspace=True,
                )
            power_df.columns = [col.strip() for col in power_df.columns]
            timestamp_col = [col for col in power_df.columns if "time" in col.lower()][
                0
            ]
            power_df[timestamp_col] = pd.to_datetime(power_df[timestamp_col])
            power_cols = [
                col
                for col in power_df.columns
                if "power" in col.lower() or "draw" in col.lower()
            ]

            if not power_cols:
                raise ValueError(f"No power columns found in {csv_path}")
            for col in power_cols:
                if power_df[col].dtype == object:  # If it's a string
                    power_df[col] = power_df[col].str.extract(r"([\d.]+)").astype(float)

            return power_df
        except Exception as e:
            print(f"Error parsing power CSV {csv_path}: {e}")
            return pd.DataFrame()

    def align_timestamps(
        self, power_df: pd.DataFrame, requests_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align power measurements with request timestamps.

        Args:
            power_df: DataFrame with power measurements
            requests_df: DataFrame with request data

        Returns:
            DataFrame with power measurements aligned with requests
        """
        if power_df.empty or requests_df.empty:
            return pd.DataFrame()

        power_time_col = [col for col in power_df.columns if "time" in col.lower()][0]
        request_time_col = (
            "Request Time"
            if "Request Time" in requests_df.columns
            else requests_df.columns[0]
        )

        power_df = power_df.sort_values(by=power_time_col)
        requests_df = requests_df.sort_values(by=request_time_col)
        start_time = min(
            requests_df[request_time_col].min(), power_df[power_time_col].min()
        )

        power_df["relative_time"] = (
            power_df[power_time_col] - start_time
        ).dt.total_seconds()
        end_time = max(
            requests_df[request_time_col].max(), power_df[power_time_col].max()
        )
        experiment_duration = (end_time - start_time).total_seconds()

        power_df["experiment_duration"] = experiment_duration

        return power_df

    def extract_power_trace(
        self,
        power_df: pd.DataFrame,
        duration_seconds: int = 600,
        sampling_rate_hz: int = 1,
    ) -> np.ndarray:
        """
        Extract uniform power trace from power measurements.

        Args:
            power_df: DataFrame with power measurements
            duration_seconds: Duration of the trace to extract (in seconds)
            sampling_rate_hz: Sampling rate for the uniform trace

        Returns:
            Numpy array with uniform power trace
        """
        if power_df.empty:
            return np.zeros(duration_seconds * sampling_rate_hz)

        power_cols = [
            col
            for col in power_df.columns
            if "power" in col.lower() or "draw" in col.lower()
        ]

        if not power_cols:
            print("No power column found in DataFrame")
            return np.zeros(duration_seconds * sampling_rate_hz)

        power_col = power_cols[0]

        # Create uniform time grid
        uniform_time = np.linspace(
            0,
            min(duration_seconds, power_df["experiment_duration"]),
            num=duration_seconds * sampling_rate_hz,
        )

        # Get original time and power values
        time_values = power_df["relative_time"].values
        power_values = power_df[power_col].values

        # Handle multi-GPU tensor parallelism by averaging if multiple power columns
        if len(power_cols) > 1:
            # Take the average across all GPUs
            power_values = power_df[power_cols].mean(axis=1).values

        uniform_power = np.interp(
            uniform_time,
            time_values,
            power_values,
            left=power_values[0] if len(power_values) > 0 else 0,
            right=power_values[-1] if len(power_values) > 0 else 0,
        )

        # Ensure the trace is exactly the right length
        if len(uniform_power) < duration_seconds * sampling_rate_hz:
            # Pad with the last value if too short
            pad_length = duration_seconds * sampling_rate_hz - len(uniform_power)
            uniform_power = np.pad(uniform_power, (0, pad_length), "edge")
        elif len(uniform_power) > duration_seconds * sampling_rate_hz:
            # Truncate if too long
            uniform_power = uniform_power[: duration_seconds * sampling_rate_hz]

        return uniform_power

    def process_experiment(
        self,
        experiment_folder: str,
        duration_seconds: int = 600,
        sampling_rate_hz: int = 1,
    ) -> Tuple[Dict, np.ndarray]:
        """
        Process a single experiment folder.

        Args:
            experiment_folder: Path to experiment folder
            duration_seconds: Duration of power trace to extract
            sampling_rate_hz: Sampling rate for power trace

        Returns:
            Tuple of (config_dict, power_trace)
        """
        json_files = glob.glob(os.path.join(experiment_folder, "*.json"))
        if not json_files:
            print(f"No JSON config file found in {experiment_folder}")
            return None, None

        config = self.parse_json_config(json_files[0])

        request_csv_files = [
            f
            for f in glob.glob(os.path.join(experiment_folder, "*.csv"))
            if "request" in f.lower() or "model" in f.lower()
        ]

        power_csv_files = [
            f
            for f in glob.glob(os.path.join(experiment_folder, "*.csv"))
            if "power" in f.lower() or "nvidia" in f.lower()
        ]

        if not request_csv_files or not power_csv_files:
            print(f"Missing required CSV files in {experiment_folder}")
            return None, None

        requests_df = self.parse_request_csv(request_csv_files[0])
        power_df = self.parse_power_csv(power_csv_files[0])

        if power_df.empty:
            print(f"Failed to parse power data from {power_csv_files[0]}")
            return None, None

        aligned_power_df = self.align_timestamps(power_df, requests_df)

        power_trace = self.extract_power_trace(
            aligned_power_df, duration_seconds, sampling_rate_hz
        )

        return config, power_trace

    def process_all_experiments(
        self, duration_seconds: int = 600, sampling_rate_hz: int = 1
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Process all experiment folders and collect data for training.

        Args:
            duration_seconds: Duration of power traces to extract
            sampling_rate_hz: Sampling rate for power traces

        Returns:
            Tuple of (config_params, power_traces)
        """
        experiment_folders = self.discover_experiment_folders()

        config_keys = [
            "poisson_rate",
            "tensor_parallelism",
            "model_size",
            "is_reasoning",
            "is_h100",
        ]

        config_params = {key: [] for key in config_keys}
        power_traces = []

        skipped_count = 0
        for folder in tqdm(experiment_folders, desc="Processing experiments"):
            config, power_trace = self.process_experiment(
                folder, duration_seconds, sampling_rate_hz
            )

            if config is None or power_trace is None:
                skipped_count += 1
                continue

            config_params["poisson_rate"].append(config["poisson_rate"])
            config_params["tensor_parallelism"].append(config["tensor_parallelism"])
            config_params["model_size"].append(config["model_size"])
            config_params["is_reasoning"].append(1.0 if config["is_reasoning"] else 0.0)
            config_params["is_h100"].append(1.0 if config["is_h100"] else 0.0)
            power_traces.append(power_trace)

        for key in config_keys:
            config_params[key] = np.array(config_params[key])

        power_traces = np.array(power_traces)

        print(
            f"Processed {len(power_traces)} experiments, skipped {skipped_count} due to missing or invalid data"
        )

        return config_params, power_traces

    def visualize_samples(
        self,
        config_params: Dict[str, np.ndarray],
        power_traces: np.ndarray,
        num_samples: int = 5,
    ):
        """
        Visualize a few sample power traces from the dataset.

        Args:
            config_params: Dictionary of configuration parameters
            power_traces: Array of power traces
            num_samples: Number of samples to visualize
        """
        if len(power_traces) == 0:
            print("No data to visualize")
            return

        # Select a subset of samples
        indices = np.random.choice(
            len(power_traces), min(num_samples, len(power_traces)), replace=False
        )

        # Create figure
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))
        if num_samples == 1:
            axes = [axes]

        for i, idx in enumerate(indices):
            trace = power_traces[idx]
            time_axis = np.arange(len(trace)) / 60  # Convert to minutes

            # Get configuration for this trace
            model_size = config_params["model_size"][idx]
            poisson_rate = config_params["poisson_rate"][idx]
            tensor_parallel = config_params["tensor_parallelism"][idx]
            is_reasoning = config_params["is_reasoning"][idx] > 0.5
            is_h100 = config_params["is_h100"][idx] > 0.5

            # Plot trace
            axes[i].plot(time_axis, trace)
            axes[i].set_title(
                f"Model: {model_size:.1f}B, {'Reasoning' if is_reasoning else 'Standard'}, "
                f"{'H100' if is_h100 else 'A100'}, TP={tensor_parallel}, Rate={poisson_rate:.1f}"
            )
            axes[i].set_xlabel("Time (minutes)")
            axes[i].set_ylabel("Power (W)")
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def create_dataset(
        self,
        config_params: Dict[str, np.ndarray],
        power_traces: np.ndarray,
        sequence_length: int = 600,
        stride: int = 60,
    ) -> Tuple[Dataset, Dataset]:
        """
        Create PyTorch datasets for training and validation.

        Args:
            config_params: Dictionary of configuration parameters
            power_traces: Array of power traces
            sequence_length: Length of sequences to use for training
            stride: Stride for sliding window

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        from torch.utils.data import random_split

        # Create dataset
        dataset = PowerTraceDataset(
            power_traces=power_traces,
            config_params=config_params,
            sequence_length=sequence_length,
            stride=stride,
        )

        # Split into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        print(f"Created dataset with {len(dataset)} samples")
        print(
            f"Train set: {len(train_dataset)} samples, Validation set: {len(val_dataset)} samples"
        )

        return train_dataset, val_dataset


class PowerTraceDataset(Dataset):
    """Dataset for power trace data with configuration parameters."""

    def __init__(
        self,
        power_traces: np.ndarray,
        config_params: Dict[str, np.ndarray],
        sequence_length: int = 600,
        stride: int = 60,
    ):
        """
        Args:
            power_traces: Array of power measurements
            config_params: Dictionary of configuration parameters
                - poisson_rate: Poisson arrival rate of queries
                - tensor_parallelism: Degree of tensor parallelism
                - model_size: Model size in parameter count (in billions)
                - is_reasoning: Boolean indicating if reasoning model (1) or not (0)
                - is_h100: Boolean indicating if H100 (1) or A100 (0)
            sequence_length: Length of output sequences
            stride: Stride for sliding window
        """
        self.power_traces = power_traces
        self.config_params = config_params
        self.sequence_length = sequence_length
        self.stride = stride
        self.valid_indices = []
        for i in range(len(power_traces)):
            for j in range(0, len(power_traces[i]) - sequence_length + 1, stride):
                self.valid_indices.append((i, j))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        sample_idx, start_idx = self.valid_indices[idx]
        sequence = self.power_traces[sample_idx][
            start_idx : start_idx + self.sequence_length
        ]

        config = {
            "poisson_rate": self.config_params["poisson_rate"][sample_idx],
            "tensor_parallelism": self.config_params["tensor_parallelism"][sample_idx],
            "model_size": self.config_params["model_size"][sample_idx],
            "is_reasoning": self.config_params["is_reasoning"][sample_idx],
            "is_h100": self.config_params["is_h100"][sample_idx],
        }
        sequence_tensor = torch.FloatTensor(sequence)
        config_tensor = torch.FloatTensor(
            [
                config["poisson_rate"],
                config["tensor_parallelism"],
                config["model_size"],
                config["is_reasoning"],
                config["is_h100"],
            ]
        )

        return {
            "config": config_tensor,
            "power_trace": sequence_tensor,
            "input_trace": torch.cat([torch.zeros(1), sequence_tensor[:-1]]),
        }


def normalize_config_params(
    config_params: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Normalize configuration parameters to have similar ranges.

    Args:
        config_params: Dictionary of configuration parameters

    Returns:
        Dictionary of normalized configuration parameters
    """
    MAX_MODEL_SIZE = 671.0
    MAX_POISSON_ARRIVAL_RATE = 64.0
    MAX_TENSOR_PARALLELISM = 8.0
    normalized_params = {}
    normalized_params["poisson_rate"] = (
        config_params["poisson_rate"] / MAX_POISSON_ARRIVAL_RATE
    )

    normalized_params["tensor_parallelism"] = (
        config_params["tensor_parallelism"] / MAX_TENSOR_PARALLELISM
    )
    normalized_params["model_size"] = config_params["model_size"] / MAX_MODEL_SIZE
    normalized_params["is_reasoning"] = config_params["is_reasoning"]
    normalized_params["is_h100"] = config_params["is_h100"]

    return normalized_params


if __name__ == "__main__":
    data_root = "/path/to/your/data"
    processor = GPUPowerDataProcessor(data_root)
    config_params, power_traces = processor.process_all_experiments(
        duration_seconds=600, sampling_rate_hz=1
    )
    normalized_params = normalize_config_params(config_params)

    train_dataset, val_dataset = processor.create_dataset(
        normalized_params, power_traces, sequence_length=600, stride=60
    )

    save_dir = "data/processed"
    os.makedirs(save_dir, exist_ok=True)

    train_data = {
        "config": train_dataset.dataset.config_params,
        "power_traces": train_dataset.dataset.power_traces,
        "indices": train_dataset.indices,
    }
    torch.save(train_data, os.path.join(save_dir, "train_dataset.pt"))
    val_data = {
        "config": val_dataset.dataset.config_params,
        "power_traces": val_dataset.power_traces,
        "indices": val_dataset.indices,
    }
    torch.save(val_data, os.path.join(save_dir, "val_dataset.pt"))

    print(f"Saved processed datasets to {save_dir}")
