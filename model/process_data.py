import os
import glob
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


class GPUPowerDataProcessor:
    """
    A simplified processor class to:
      1) match power traces and results CSVs,
      2) parse and sum multi-GPU power logs,
      3) align them in time,
      4) extract uniform, fixed-duration power traces,
      5) gather config metadata from filenames.
    """

    def __init__(self, data_root_dir: str):
        """
        Args:
            data_root_dir: Root directory containing all CSV files.
                           e.g. /powertrace_sim/client
        """
        self.data_root_dir = data_root_dir

    def discover_experiment_pairs(self) -> List[Tuple[str, str]]:
        """
        Finds matching pairs of power CSVs and results CSVs based on
        {MODEL_NAME}_tp{TP}_p{RATE}_d{DATE}.csv and
        results_{MODEL_NAME}_{RATE}_{TP}_final.csv.

        Returns:
            A list of (power_csv_path, results_csv_path).
        """
        all_csvs = glob.glob(
            os.path.join(self.data_root_dir, "**", "*.csv"), recursive=True
        )
        print(f"Found {len(all_csvs)} CSV files in {self.data_root_dir}")

        # Separate into "power" files vs "results" files
        power_files = []
        results_files = []

        for csv_path in all_csvs:
            base = os.path.basename(csv_path)
            if base.startswith("results_"):
                results_files.append(csv_path)
            else:
                # Likely a power file if it has _tp and _p in the name
                if "_tp" in base and "_p" in base:
                    power_files.append(csv_path)

        # Attempt to match them by extracting model_name, tp, rate
        def extract_power_info(filename: str):
            # Example:  llama-3-8b_tp2_p2.0_d2025-03-14-07-32-35.csv
            base = os.path.basename(filename)
            model_match = re.match(r"(.*)_tp(\d+)_p([\d\.]+)_d", base)
            print(model_match.groups())
            if not model_match:
                return None
            model_name = model_match.group(1)
            tp = model_match.group(2)
            rate = model_match.group(3)
            return model_name, tp, rate

        def extract_results_info(filename: str):
            # Example: results_Llama-3.1-8B-Instruct_0.5_2_final.csv
            base = os.path.basename(filename)
            result_match = re.match(r"results_(.*)_(\d+\.\d+)_(\d+)_final", base)
            if not result_match:
                return None
            model_name = result_match.group(1)
            rate = result_match.group(2)
            tp = result_match.group(3)
            return model_name, tp, rate

        matched_pairs = []
        for pfile in power_files:
            pinfo = extract_power_info(pfile)
            if not pinfo:
                continue
            p_model, p_tp, p_rate = pinfo

            # Look for a results file that matches
            for rfile in results_files:
                rinfo = extract_results_info(rfile)
                if not rinfo:
                    continue
                r_model, r_tp, r_rate = rinfo

                # Check if they match
                if p_model == r_model and p_tp == r_tp and p_rate == r_rate:
                    matched_pairs.append((pfile, rfile))
                    break  # Found our match, no need to look further

        print(
            f"Found {len(power_files)} power files, {len(results_files)} results files, "
            f"matched {len(matched_pairs)} pairs."
        )
        return matched_pairs

    def parse_results_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Parse the results CSV (e.g. results_{MODEL_NAME}_{RATE}_{TP}_final.csv).

        Expects columns:
        [Request Time, Model, Data Source, Poisson Arrival Rate, Tensor Parallel Size,
         Input Tokens, Output Tokens, E2E Latency]

        Returns:
            Pandas DataFrame with a 'request_time' column (datetime64).
        """
        try:
            df = pd.read_csv(csv_path)
            if "Request Time" in df.columns:
                # Convert to datetime (the example data uses numeric timestamps in seconds)
                if pd.api.types.is_numeric_dtype(df["Request Time"]):
                    df["request_time"] = pd.to_datetime(df["Request Time"], unit="s")
                else:
                    df["request_time"] = pd.to_datetime(df["Request Time"])
            else:
                # Fallback if column not found
                df["request_time"] = pd.date_range(
                    start=pd.Timestamp.now(), periods=len(df), freq="s"
                )
            return df
        except Exception as e:
            print(f"Error reading results file {csv_path}: {e}")
            return pd.DataFrame()

    def parse_power_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Parse the GPU power trace CSV and sum up the multi-GPU rows
        so that each timestamp has a single total power value.

        Expects columns (for each GPU row):
            timestamp, power.draw [W], utilization.gpu [%], memory.used [MiB]

        Because the example data has repeated timestamps for each GPU,
        we group by timestamp and sum the power.
        """
        try:
            df = pd.read_csv(csv_path, skipinitialspace=True)
            df.columns = [col.strip().lower() for col in df.columns]
            if "power.draw [w]" in df.columns:
                df.rename(columns={"power.draw [w]": "power"}, inplace=True)
            if df["power"].dtype == object:
                df["power"] = df["power"].replace(r"[^\d.]", "", regex=True)
                df["power"] = pd.to_numeric(df["power"])

            # Parse timestamp column
            # The example data might have "2025/03/13 20:25:44.374"
            # We convert that to datetime
            time_col = None
            for c in df.columns:
                if "time" in c:
                    time_col = c
                    break
            if not time_col:
                raise ValueError("No timestamp column found in power CSV")

            df[time_col] = pd.to_datetime(df[time_col])

            # Now group by the exact timestamp to sum GPU rows
            grouped = df.groupby(time_col, as_index=False)["power"].sum()
            grouped.rename(columns={time_col: "timestamp"}, inplace=True)

            return grouped.sort_values(by="timestamp").reset_index(drop=True)

        except Exception as e:
            print(f"Error reading power file {csv_path}: {e}")
            return pd.DataFrame()

    def align_and_interpolate(
        self,
        power_df: pd.DataFrame,
        results_df: pd.DataFrame,
        duration_seconds: int,
        sampling_rate_hz: int,
    ) -> np.ndarray:
        """
        Align the power measurements with the request timeline and
        extract a uniform power trace of length (duration_seconds * sampling_rate_hz).

        1) Determine start_time = min(power_df.timestamp, results_df.request_time)
        2) Determine end_time   = max(power_df.timestamp, results_df.request_time)
        3) Create a uniform time axis from 0..duration_seconds at sampling_rate_hz.
        4) np.interp the power values onto that axis.

        If there's no data or something is empty, return zeros.
        """
        if power_df.empty or results_df.empty:
            return np.zeros(duration_seconds * sampling_rate_hz)

        start_time = min(power_df["timestamp"].min(), results_df["request_time"].min())
        end_time = max(power_df["timestamp"].max(), results_df["request_time"].max())
        total_expt_duration = (end_time - start_time).total_seconds()
        print(f"Experiment duration: {total_expt_duration:.2f} seconds")
        # We'll only consider up to duration_seconds (clamped to the experiment's length)
        effective_duration = min(duration_seconds, int(np.ceil(total_expt_duration)))

        # Build the uniform time axis in *seconds* from 0..effective_duration
        uniform_time = np.linspace(
            0, effective_duration, num=effective_duration * sampling_rate_hz
        )

        # Convert actual power timestamps to "seconds since start_time"
        power_seconds = (power_df["timestamp"] - start_time).dt.total_seconds().values
        power_values = power_df["power"].values

        # Interpolate
        uniform_power = np.interp(
            uniform_time,
            power_seconds,
            power_values,
            left=power_values[0],
            right=power_values[-1],
        )

        # If we need exactly duration_seconds * sampling_rate_hz points, pad or truncate
        desired_len = duration_seconds * sampling_rate_hz
        if len(uniform_power) < desired_len:
            # pad with the last value
            padding = np.full(desired_len - len(uniform_power), uniform_power[-1])
            uniform_power = np.concatenate([uniform_power, padding])
        elif len(uniform_power) > desired_len:
            uniform_power = uniform_power[:desired_len]

        return uniform_power

    def process_all_experiments(
        self, duration_seconds: int = 600, sampling_rate_hz: int = 4
    ) -> Tuple[Dict[str, List[float]], np.ndarray]:
        """
        1) Finds all matching power/results pairs
        2) Parses and aligns them
        3) Extracts a fixed-size power trace
        4) Builds up lists of config parameters, and an array of power traces.

        Returns:
            (config_params, power_traces)
            where:
               config_params is a dict of lists, e.g. {
                 "model_name": [...],
                 "tensor_parallelism": [...],
                 "poisson_rate": [...]
               }
               power_traces is shape [N, duration_seconds * sampling_rate_hz]
        """
        pairs = self.discover_experiment_pairs()
        if not pairs:
            print("No experiment pairs found. Returning empty data.")
            return {}, np.array([])

        # Prepare containers
        config_params = {
            "model_name": [],
            "tensor_parallelism": [],
            "poisson_rate": [],
        }
        all_traces = []

        for pfile, rfile in pairs:
            power_df = self.parse_power_csv(pfile)
            results_df = self.parse_results_csv(rfile)
            if power_df.empty or results_df.empty:
                print(f"Skipping pair: {pfile}, {rfile} (empty data).")
                continue

            # 2) extract config from filenames
            # power file example: {MODEL_NAME}_tp{TP}_p{RATE}_d...
            base_p = os.path.basename(pfile)
            match = re.match(r"(.*)_tp(\d+)_p([\d\.]+)_d", base_p)
            if not match:
                print(f"Could not parse power filename {base_p}")
                continue
            model_name, tp_str, rate_str = match.groups()

            # 3) align & interpolate
            power_trace = self.align_and_interpolate(
                power_df, results_df, duration_seconds, sampling_rate_hz
            )

            # 4) accumulate results
            config_params["model_name"].append(model_name)
            config_params["tensor_parallelism"].append(float(tp_str))
            config_params["poisson_rate"].append(float(rate_str))
            all_traces.append(power_trace)

        if len(all_traces) == 0:
            print("All experiment pairs were invalid or empty.")
            return {}, np.array([])

        # Stack into a single array of shape [N, T]
        power_traces = np.stack(all_traces, axis=0)
        return config_params, power_traces

    def visualize_samples(
        self,
        config_params: Dict[str, List[float]],
        power_traces: np.ndarray,
        num_samples: int = 3,
    ):
        """
        Visualize random samples from the dataset
        """
        if len(power_traces) == 0:
            print("No power traces to visualize.")
            return

        indices = np.random.choice(
            len(power_traces), min(num_samples, len(power_traces)), replace=False
        )
        for i, idx in enumerate(indices):
            plt.figure(figsize=(10, 3))
            trace = power_traces[idx]
            time_axis = np.arange(len(trace)) / 60.0  # show in minutes, for example
            plt.plot(time_axis, trace)
            plt.title(
                f"Sample {idx} | Model={config_params['model_name'][idx]}, "
                f"TP={config_params['tensor_parallelism'][idx]}, "
                f"Rate={config_params['poisson_rate'][idx]}"
            )
            plt.xlabel("Time (minutes)")
            plt.ylabel("Power (W)")
            plt.grid(True)
            plt.show()


class PowerTraceDataset(Dataset):
    """
    A simple PyTorch Dataset to create (config, power_trace) pairs
    from the processed data. Demonstrates a sliding-window approach.
    """

    def __init__(
        self,
        power_traces: np.ndarray,
        config_params: Dict[str, np.ndarray],
        sequence_length: int = 600,
        stride: int = 60,
    ):
        """
        Args:
            power_traces: shape [N, T], each row is a power time series
            config_params: dictionary with keys like
               ["model_name", "tensor_parallelism", "poisson_rate"]
               you'll want numeric or one-hot versions for training
               This example just converts them to float for demonstration.
            sequence_length: number of points in each returned sequence
            stride: how many steps to move the window for the next sample
        """
        self.power_traces = power_traces
        self.config_params = config_params
        self.sequence_length = sequence_length
        self.stride = stride
        self.tp_array = np.array(config_params["tensor_parallelism"], dtype=np.float32)
        self.rate_array = np.array(config_params["poisson_rate"], dtype=np.float32)

        self.valid_indices = []
        for i in range(len(self.power_traces)):
            T = len(self.power_traces[i])
            for start_idx in range(0, T - sequence_length + 1, stride):
                self.valid_indices.append((i, start_idx))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        sample_i, start_i = self.valid_indices[idx]
        seq = self.power_traces[sample_i][start_i : start_i + self.sequence_length]

        config_tensor = torch.tensor(
            [
                self.tp_array[sample_i],
                self.rate_array[sample_i],
            ],
            dtype=torch.float32,
        )

        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        input_tensor = torch.cat([torch.zeros(1), seq_tensor[:-1]])  # shift by 1

        return {
            "config": config_tensor,  # [2]
            "power_trace": seq_tensor,  # [sequence_length]
            "input_trace": input_tensor,  # [sequence_length]
        }


def normalize_config_params(
    config_params: Dict[str, List[float]]
) -> Dict[str, np.ndarray]:
    """
    Example normalization. Adjust as needed for your actual data ranges.
    We'll ignore 'model_name' in numeric normalization.
    """
    print(config_params)
    tp = np.array(config_params["tensor_parallelism"], dtype=np.float32)
    rate = np.array(config_params["poisson_rate"], dtype=np.float32)
    tp_max = 8.0
    rate_max = 4.0

    return {
        "model_name": np.array(config_params["model_name"]),  # keep as-is
        "tensor_parallelism": tp / tp_max,
        "poisson_rate": rate / rate_max,
    }


def save_processed_data(save_dir: str, dataset: Dataset):
def save_processed_data(save_dir: str, dataset: Dataset):
    """
    Save final data to disk in a single .npz file. This includes the original
    power traces and valid indices.
    power traces and valid indices.

    Adjust as needed based on how you want to load data in the future.
    """
    os.makedirs(save_dir, exist_ok=True)

    np.savez(
        os.path.join(save_dir, "power_trace_data.npz"),
        power_traces=dataset.power_traces,
        model_name=dataset.config_params["model_name"],
        tensor_parallelism=dataset.config_params["tensor_parallelism"],
        poisson_rate=dataset.config_params["poisson_rate"],
        valid_indices=dataset.valid_indices,
        power_traces=dataset.power_traces,
        model_name=dataset.config_params["model_name"],
        tensor_parallelism=dataset.config_params["tensor_parallelism"],
        poisson_rate=dataset.config_params["poisson_rate"],
        valid_indices=dataset.valid_indices,
    )
    print(
        f"Saved processed dataset to: {os.path.join(save_dir, 'power_trace_data.npz')}"
    )


if __name__ == "__main__":
    data_root = "../client"
    processor = GPUPowerDataProcessor(data_root_dir=data_root)
    config_params, power_traces = processor.process_all_experiments(
        duration_seconds=600, sampling_rate_hz=4
    )
    print("Done processing, final shapes:")
    for k, v in config_params.items():
        print(f"  {k}: {len(v)} entries")
    print("  power_traces:", power_traces.shape)
    processor.visualize_samples(config_params, power_traces, num_samples=3)
    norm_params = normalize_config_params(config_params)
    dataset = PowerTraceDataset(
    dataset = PowerTraceDataset(
        power_traces=power_traces,
        config_params=norm_params,
        sequence_length=600,
        stride=60,
    )
    save_processed_data("processed_data", dataset)
    save_processed_data("processed_data", dataset)
