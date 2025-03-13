import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
from datetime import datetime


class GPUPowerDataProcessor:
    def __init__(self, data_root_dir: str):
        self.data_root_dir = data_root_dir

    def discover_experiment_folders(self) -> List[str]:
        experiment_folders = []
        for folder in glob.glob(self.data_root_dir):
            csv_files = glob.glob(os.path.join(folder, "*.csv"))
            if len(csv_files) >= 2:
                experiment_folders.append(folder)

        print(f"Discovered {len(experiment_folders)} experiment folders")
        return experiment_folders

    def extract_model_size_in_billions(self, model_name: str) -> float:
        match = re.search(r"(\d+\.?\d*)[-]?[bB]", model_name)
        if match:
            return float(match.group(1))
        print(
            f"Warning: Unable to extract model size from {model_name}, using default value of 13B"
        )
        return 13.0

    def determine_is_reasoning(self, model_name: str) -> bool:
        reasoning_indicators = ["deepseek"]

        return any(
            indicator in model_name.lower() for indicator in reasoning_indicators
        )

    def determine_gpu_type(self, power_df: pd.DataFrame) -> bool:
        if "power.draw [W]" in power_df.columns:
            max_power = power_df["power.draw [W]"].max()
            if max_power > 400:  # H100 has higher peak power
                return True

        return False

    def parse_request_csv(self, csv_path: str) -> pd.DataFrame:
        try:
            requests_df = pd.read_csv(csv_path)

            # Convert Unix timestamp to datetime if needed
            if "Request Time" in requests_df.columns:
                if pd.api.types.is_numeric_dtype(requests_df["Request Time"]):
                    requests_df["Request Time"] = pd.to_datetime(
                        requests_df["Request Time"], unit="s"
                    )
                else:
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
            # First try to read with header row
            power_df = pd.read_csv(csv_path, skipinitialspace=True)

            # If columns don't look correct, try with explicit column names
            if not any("power" in col.lower() for col in power_df.columns):
                power_df = pd.read_csv(
                    csv_path,
                    names=[
                        "timestamp",
                        "power.draw [W]",
                        "utilization.gpu [%]",
                        "memory.used [MiB]",
                    ],
                    skiprows=1,
                    skipinitialspace=True,
                )

            # Clean up column names
            power_df.columns = [col.strip() for col in power_df.columns]

            # Convert timestamp to datetime
            timestamp_col = [col for col in power_df.columns if "time" in col.lower()][
                0
            ]
            power_df[timestamp_col] = pd.to_datetime(power_df[timestamp_col])

            # Extract numeric values from columns if they contain units
            for col in power_df.columns:
                if power_df[col].dtype == object and any(
                    c.isdigit() for c in power_df[col].iloc[0]
                ):
                    power_df[col] = power_df[col].str.extract(r"([\d.]+)").astype(float)

            return power_df
        except Exception as e:
            print(f"Error parsing power CSV {csv_path}: {e}")
            return pd.DataFrame()

    def identify_active_gpus(self, power_df: pd.DataFrame) -> List[int]:
        """
        Identify which GPUs are active based on utilization and memory usage.

        Args:
            power_df: DataFrame with power measurements for multiple GPUs

        Returns:
            List of indices of active GPUs
        """
        # Reshape the data to identify individual GPUs
        # Each set of N consecutive rows represents measurements for N GPUs at one time
        num_gpus = self._determine_num_gpus(power_df)
        if num_gpus <= 1:
            return [0]  # Only one GPU

        # Group readings by timestamp to identify patterns
        timestamps = power_df["timestamp"].unique()

        # Keep track of utilization and memory across all timestamps
        gpu_util_sum = np.zeros(num_gpus)
        gpu_memory_sum = np.zeros(num_gpus)

        for i, ts in enumerate(timestamps):
            ts_data = power_df[power_df["timestamp"] == ts]
            if len(ts_data) == num_gpus:
                if "utilization.gpu [%]" in ts_data.columns:
                    gpu_util_sum += ts_data["utilization.gpu [%]"].values
                if "memory.used [MiB]" in ts_data.columns:
                    gpu_memory_sum += ts_data["memory.used [MiB]"].values

        # Identify active GPUs based on utilization or memory
        active_gpus = []
        for i in range(num_gpus):
            # A GPU is considered active if it has significant utilization or memory usage
            if (gpu_util_sum[i] > 0) or (gpu_memory_sum[i] > 1000 * len(timestamps)):
                active_gpus.append(i)

        # If no active GPUs identified, fall back to the one with highest memory
        if not active_gpus and len(gpu_memory_sum) > 0:
            active_gpus = [np.argmax(gpu_memory_sum)]

        print(f"Identified {len(active_gpus)} active GPUs: {active_gpus}")
        return active_gpus

    def _determine_num_gpus(self, power_df: pd.DataFrame) -> int:
        """
        Determine the number of GPUs by analyzing the pattern in the DataFrame.

        Args:
            power_df: DataFrame with power measurements

        Returns:
            Number of GPUs
        """
        # Look for repeating timestamp patterns
        timestamps = power_df["timestamp"].tolist()

        # Find the first repeated timestamp
        for i in range(1, min(len(timestamps), 20)):
            if timestamps[i] == timestamps[0]:
                return i

        # If we can't determine, assume 1 GPU
        return 1

    def filter_active_gpu_data(self, power_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter power data to include only active GPUs.

        Args:
            power_df: DataFrame with power measurements for multiple GPUs

        Returns:
            DataFrame with power measurements for active GPUs only
        """
        num_gpus = self._determine_num_gpus(power_df)
        if num_gpus <= 1:
            return power_df  # Only one GPU, no filtering needed

        active_gpus = self.identify_active_gpus(power_df)
        if not active_gpus:
            print("No active GPUs identified. Using all GPUs.")
            return power_df

        # Create a new DataFrame with only the active GPU data
        filtered_rows = []
        timestamps = power_df["timestamp"].unique()

        for ts in timestamps:
            ts_data = power_df[power_df["timestamp"] == ts]
            if len(ts_data) >= max(active_gpus) + 1:
                for gpu_idx in active_gpus:
                    filtered_rows.append(ts_data.iloc[gpu_idx])

        return pd.DataFrame(filtered_rows)

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

        # Identify timestamp columns
        power_time_col = [col for col in power_df.columns if "time" in col.lower()][0]
        request_time_col = (
            "Request Time"
            if "Request Time" in requests_df.columns
            else requests_df.columns[0]
        )

        # Sort both dataframes by timestamp
        power_df = power_df.sort_values(by=power_time_col)
        requests_df = requests_df.sort_values(by=request_time_col)

        # Align timestamps - find the start time of the experiment
        start_time = min(
            requests_df[request_time_col].min(), power_df[power_time_col].min()
        )

        # Reset timestamps relative to start time
        power_df["relative_time"] = (
            power_df[power_time_col] - start_time
        ).dt.total_seconds()

        # Add experiment duration (in seconds)
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
            # Return zero trace if no data
            return np.zeros(duration_seconds * sampling_rate_hz)

        # Get power column
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

        # Perform interpolation to create uniform trace
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

    def extract_config_from_request(
        self, requests_df: pd.DataFrame, power_df: pd.DataFrame
    ) -> Dict:
        """
        Extract configuration parameters from request data.

        Args:
            requests_df: DataFrame with request data
            power_df: DataFrame with power measurements (for GPU type detection)

        Returns:
            Dictionary with configuration parameters
        """
        if requests_df.empty:
            return None

        # Get the first model name to determine model size and if it's a reasoning model
        model_name = (
            requests_df["Model"].iloc[0]
            if "Model" in requests_df.columns
            else "unknown-13B"
        )

        # Extract model size
        model_size = self.extract_model_size_in_billions(model_name)

        # Determine if it's a reasoning model
        is_reasoning = self.determine_is_reasoning(model_name)

        # Get tensor parallelism
        tensor_parallelism = 1
        if "Tensor Parallel Size" in requests_df.columns:
            tensor_parallelism = int(requests_df["Tensor Parallel Size"].iloc[0])

        # Get Poisson arrival rate
        poisson_rate = 1.0
        if "Poisson Arrival Rate" in requests_df.columns:
            poisson_rate = float(requests_df["Poisson Arrival Rate"].iloc[0])

        # Determine GPU type
        is_h100 = self.determine_gpu_type(power_df)

        return {
            "model_name": model_name,
            "model_size": model_size,
            "is_reasoning": is_reasoning,
            "is_h100": is_h100,
            "poisson_rate": poisson_rate,
            "tensor_parallelism": tensor_parallelism,
        }

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
        # Find CSV files
        csv_files = glob.glob(os.path.join(experiment_folder, "*.csv"))
        if len(csv_files) < 2:
            print(f"Not enough CSV files found in {experiment_folder}")
            return None, None

        # Identify request and power CSV files
        request_csv_files = [
            f
            for f in csv_files
            if "request" in f.lower()
            or any(x in f.lower() for x in ["model", "input", "tokens"])
        ]
        power_csv_files = [
            f
            for f in csv_files
            if any(x in f.lower() for x in ["power", "nvidia", "smi", "gpu"])
        ]

        if not request_csv_files or not power_csv_files:
            # Try to identify by examining content
            for f in csv_files:
                with open(f, "r") as file:
                    header = file.readline().lower()
                    if any(x in header for x in ["request", "model", "tokens"]):
                        request_csv_files.append(f)
                    elif any(x in header for x in ["power", "draw", "utilization"]):
                        power_csv_files.append(f)

        if not request_csv_files or not power_csv_files:
            print(
                f"Could not identify request and power CSV files in {experiment_folder}"
            )
            return None, None

        # Parse CSV files
        requests_df = self.parse_request_csv(request_csv_files[0])
        power_df = self.parse_power_csv(power_csv_files[0])

        if power_df.empty or requests_df.empty:
            print(f"Failed to parse data from {experiment_folder}")
            return None, None

        # Filter power data to include only active GPUs
        filtered_power_df = self.filter_active_gpu_data(power_df)

        # Extract configuration
        config = self.extract_config_from_request(requests_df, filtered_power_df)
        if config is None:
            print(f"Failed to extract configuration from {experiment_folder}")
            return None, None

        # Align timestamps
        aligned_power_df = self.align_timestamps(filtered_power_df, requests_df)

        # Extract power trace
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

        # Initialize data containers
        config_keys = [
            "poisson_rate",
            "tensor_parallelism",
            "model_size",
            "is_reasoning",
            "is_h100",
        ]

        config_params = {key: [] for key in config_keys}
        power_traces = []

        # Process each experiment
        skipped_count = 0
        for folder in tqdm(experiment_folders, desc="Processing experiments"):
            config, power_trace = self.process_experiment(
                folder, duration_seconds, sampling_rate_hz
            )

            if config is None or power_trace is None:
                skipped_count += 1
                continue

            # Collect configuration parameters
            config_params["poisson_rate"].append(config["poisson_rate"])
            config_params["tensor_parallelism"].append(config["tensor_parallelism"])
            config_params["model_size"].append(config["model_size"])
            config_params["is_reasoning"].append(1.0 if config["is_reasoning"] else 0.0)
            config_params["is_h100"].append(1.0 if config["is_h100"] else 0.0)

            # Collect power trace
            power_traces.append(power_trace)

        # Convert lists to arrays
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

        return fig

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

        # Calculate valid starting indices
        self.valid_indices = []
        for i in range(len(power_traces)):
            for j in range(0, len(power_traces[i]) - sequence_length + 1, stride):
                self.valid_indices.append((i, j))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        sample_idx, start_idx = self.valid_indices[idx]

        # Get sequence of power values
        sequence = self.power_traces[sample_idx][
            start_idx : start_idx + self.sequence_length
        ]

        # Get configuration parameters for this sample
        config = {
            "poisson_rate": self.config_params["poisson_rate"][sample_idx],
            "tensor_parallelism": self.config_params["tensor_parallelism"][sample_idx],
            "model_size": self.config_params["model_size"][sample_idx],
            "is_reasoning": self.config_params["is_reasoning"][sample_idx],
            "is_h100": self.config_params["is_h100"][sample_idx],
        }

        # Convert to tensors
        sequence_tensor = torch.FloatTensor(sequence)

        # Create config tensor
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
            # During training, input = target shifted by one step for autoregressive training
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
    # Constants based on typical ranges in your data
    MAX_MODEL_SIZE = 671.0  # Maximum model size in billions
    MAX_POISSON_ARRIVAL_RATE = 64.0  # Maximum Poisson arrival rate
    MAX_TENSOR_PARALLELISM = 8.0  # Maximum tensor parallelism

    normalized_params = {}

    # Normalize poisson_rate
    normalized_params["poisson_rate"] = (
        config_params["poisson_rate"] / MAX_POISSON_ARRIVAL_RATE
    )

    # Normalize tensor_parallelism
    normalized_params["tensor_parallelism"] = (
        config_params["tensor_parallelism"] / MAX_TENSOR_PARALLELISM
    )

    # Normalize model_size
    normalized_params["model_size"] = config_params["model_size"] / MAX_MODEL_SIZE

    # is_reasoning and is_h100 are already 0 or 1
    normalized_params["is_reasoning"] = config_params["is_reasoning"]
    normalized_params["is_h100"] = config_params["is_h100"]

    return normalized_params


def save_processed_data(save_dir: str, train_dataset, val_dataset):
    """
    Save processed datasets to disk.

    Args:
        save_dir: Directory to save datasets
        train_dataset: Training dataset
        val_dataset: Validation dataset
    """
    os.makedirs(save_dir, exist_ok=True)

    # Get indices from the random split
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices

    # Get the original dataset
    original_dataset = train_dataset.dataset

    # Save data
    np.savez(
        os.path.join(save_dir, "power_trace_data.npz"),
        power_traces=original_dataset.power_traces,
        poisson_rate=original_dataset.config_params["poisson_rate"],
        tensor_parallelism=original_dataset.config_params["tensor_parallelism"],
        model_size=original_dataset.config_params["model_size"],
        is_reasoning=original_dataset.config_params["is_reasoning"],
        is_h100=original_dataset.config_params["is_h100"],
        train_indices=train_indices,
        val_indices=val_indices,
    )

    print(f"Saved processed data to {os.path.join(save_dir, 'power_trace_data.npz')}")


def example_usage():
    """Example usage of the data processor."""
    # Initialize data processor
    data_root = "/Users/grantwilkins/powertrace-sim/client"
    processor = GPUPowerDataProcessor(data_root)

    # Process all experiments
    config_params, power_traces = processor.process_all_experiments(
        duration_seconds=600, sampling_rate_hz=4
    )

    fig = processor.visualize_samples(config_params, power_traces, num_samples=3)
    fig.savefig("sample_traces.png")

    # Normalize configuration parameters
    normalized_params = normalize_config_params(config_params)

    # Create datasets
    train_dataset, val_dataset = processor.create_dataset(
        normalized_params, power_traces, sequence_length=600, stride=60
    )

    # Save processed data
    save_processed_data("processed_data", train_dataset, val_dataset)

    print("Data processing complete!")


if __name__ == "__main__":
    example_usage()
