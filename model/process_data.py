import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
import datetime


def discover_experiment_pairs(data_root_dir: str) -> List[Tuple[str, str]]:
    """
    Finds matching pairs of power CSVs and results CSVs based on
    {MODEL_NAME}_tp{TP}_p{RATE}_d{DATE}.csv and
    results_{MODEL_NAME}_{RATE}_{TP}_d{DATE}_final.csv.

    Returns:
        A list of (power_csv_path, results_csv_path).
    """
    all_csvs = glob.glob(os.path.join(data_root_dir, "**", "*.csv"), recursive=True)
    print(f"Found {len(all_csvs)} CSV files in {data_root_dir}")

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

    print(
        f"Found {len(power_files)} power files and {len(results_files)} results files"
    )

    # Attempt to match them by extracting model_name, tp, rate, and date
    def extract_power_info(filename: str):
        # Example: llama-3-8b_tp2_p1.0_d2025-04-07-20-19-17.csv
        base = os.path.basename(filename)
        model_match = re.match(r"(.*)_tp(\d+)_p([\d\.]+)_d(.*).csv", base)
        if not model_match:
            return None
        model_name = model_match.group(1)
        tp = model_match.group(2)
        rate = model_match.group(3)
        date = model_match.group(4)
        return model_name, tp, rate, date

    def extract_results_info(filename: str):
        # Example: results_llama-3-8b_1.0_2_d2025-04-07-20-19-17_final.csv
        base = os.path.basename(filename)
        # Try pattern with date
        result_match = re.match(r"results_(.*)_([\d\.]+)_(\d+)_d(.*)_final.csv", base)
        if not result_match:
            # Try the old pattern without date
            result_match = re.match(r"results_(.*)_([\d\.]+)_(\d+)_final.csv", base)
            if not result_match:
                return None
            model_name = result_match.group(1)
            rate = result_match.group(2)
            tp = result_match.group(3)
            date = None
        else:
            model_name = result_match.group(1)
            rate = result_match.group(2)
            tp = result_match.group(3)
            date = result_match.group(4)
        return model_name, tp, rate, date

    matched_pairs = []
    for pfile in power_files:
        pinfo = extract_power_info(pfile)
        if not pinfo:
            continue
        p_model, p_tp, p_rate, p_date = pinfo

        # Look for a results file that matches
        for rfile in results_files:
            rinfo = extract_results_info(rfile)
            if not rinfo:
                continue
            r_model, r_tp, r_rate, r_date = rinfo

            # Check if they match on model, tp, and rate
            if p_model == r_model and p_tp == r_tp and p_rate == r_rate:
                # If date is available in both, check if it matches
                if p_date and r_date:
                    if p_date == r_date:
                        matched_pairs.append((pfile, rfile))
                        break  # Found our match with date
                else:
                    # No date information, match by other criteria
                    matched_pairs.append((pfile, rfile))
                    break  # Found our match without date

    print(
        f"Found {len(power_files)} power files, {len(results_files)} results files, "
        f"matched {len(matched_pairs)} pairs."
    )
    return matched_pairs


def parse_results_csv(csv_path: str) -> pd.DataFrame:
    """
    Parse the results CSV.

    Expects columns:
    [Request Time, Model, Data Source, Poisson Arrival Rate, Tensor Parallel Size,
     Input Tokens, Output Tokens, E2E Latency, Batch Size, Effective Batch Size,
     Prefill Tokens, Decode Tokens, Prefill Throughput, Decode Throughput]

    Returns:
        Pandas DataFrame with a 'Completion Time' column.
    """
    try:
        df = pd.read_csv(csv_path)
        df.sort_values("Request Time", inplace=True)
        df["Completion Time"] = df["Request Time"] + df["E2E Latency"]

        # Extract model size from model name
        if "deepseek" in csv_path:
            df["Model Size"] = df["Model"].apply(
                lambda x: int(x.split("-")[-1].replace("B", ""))
            )
        elif "llama-3" in csv_path or "Llama-3" in csv_path:
            # Handle multiple naming conventions
            def extract_model_size(model_name):
                if "llama-3-" in model_name.lower():
                    return int(model_name.split("-")[-2].replace("B", ""))
                elif "llama-3.1-" in model_name.lower():
                    return int(model_name.split("-")[-2].replace("B", ""))
                else:
                    # Try to find any occurrence of a number followed by B
                    size_match = re.search(r"(\d+)B", model_name)
                    if size_match:
                        return int(size_match.group(1))
                    return 0

            df["Model Size"] = df["Model"].apply(extract_model_size)
        else:
            print(f"Unknown model size format in {csv_path}, setting model size to 0.")
            df["Model Size"] = 0

        # Ensure required columns exist
        required_columns = [
            "Prefill Tokens",
            "Decode Tokens",
            "Batch Size",
            "Effective Batch Size",
            "Prefill Throughput",
            "Decode Throughput",
        ]
        for col in required_columns:
            if col not in df.columns:
                print(
                    f"Warning: {col} column missing in {csv_path}. Adding empty column."
                )
                df[col] = 0

        return df
    except Exception as e:
        print(f"Error reading results file {csv_path}: {e}")
        return pd.DataFrame()


def parse_power_csv(csv_path: str) -> pd.DataFrame:
    """
    Parse the GPU power trace CSV and sum up the first X power values of each group of 8 rows,
    where X is the tensor parallelism extracted from the filename (_tpX).
    The timestamp for each sample is taken as the minimum timestamp within the group.

    Expects columns (for each GPU row):
        timestamp, power.draw [W], utilization.gpu [%], memory.used [MiB]
    """
    try:
        # Read CSV and clean up column names
        df = pd.read_csv(csv_path, skipinitialspace=True)
        df.columns = [col.strip().lower() for col in df.columns]

        # Rename power and memory columns if needed
        if "power.draw [w]" in df.columns:
            df.rename(columns={"power.draw [w]": "power"}, inplace=True)
            df.rename(columns={"memory.used [mib]": "memory"}, inplace=True)

        # Clean up the power column (remove non-numeric characters and convert to numeric)
        if df["power"].dtype == object:
            df["power"] = df["power"].replace(r"[^\d.]", "", regex=True)
            df["power"] = pd.to_numeric(df["power"])

        # Clean up the memory column if present
        if "memory" in df.columns and df["memory"].dtype == object:
            df["memory"] = df["memory"].replace(r"[^\d.]", "", regex=True)
            df["memory"] = pd.to_numeric(df["memory"])

        # Identify the timestamp column (any column containing "time")
        time_col = None
        for c in df.columns:
            if "time" in c:
                time_col = c
                break
        if not time_col:
            raise ValueError("No timestamp column found in power CSV")

        # Convert timestamps to datetime and rename column to "timestamp"
        df[time_col] = pd.to_datetime(df[time_col])
        df.rename(columns={time_col: "timestamp"}, inplace=True)

        # Extract tensor parallelism from the filename (_tpX -> X)
        tp_match = re.search(r"_tp(\d+)", csv_path)
        tensor_parallelism = int(tp_match.group(1)) if tp_match else 1
        print(f"Extracted tensor parallelism {tensor_parallelism} from {csv_path}")

        # Group by 8 rows (8 GPUs per node)
        num_rows = len(df)
        num_complete_groups = num_rows // 8
        df = df.iloc[: num_complete_groups * 8]
        groups = df.groupby(np.arange(len(df)) // 8)

        result = groups.apply(
            lambda x: pd.Series(
                {
                    "timestamp": x["timestamp"].min(),
                    "power": x.iloc[:tensor_parallelism]["power"].sum(),
                }
            )
        ).reset_index(drop=True)

        result["timestamp"] = (
            pd.to_datetime(result["timestamp"]).astype(np.int64) / 10**9
        )

        return result

    except Exception as e:
        print(f"Error reading power file {csv_path}: {e}")
        return pd.DataFrame()


def extract_experiment_info(power_csv_path: str) -> Dict:
    """
    Extract experiment information from the power CSV filename.

    Args:
        power_csv_path: Path to the power CSV file

    Returns:
        Dict with model_name, tp, rate, and date
    """
    base = os.path.basename(power_csv_path)
    model_match = re.match(r"(.*)_tp(\d+)_p([\d\.]+)_d(.*).csv", base)
    if not model_match:
        return None

    return {
        "model_name": model_match.group(1),
        "tp": int(model_match.group(2)),
        "rate": float(model_match.group(3)),
        "date": model_match.group(4),
    }


def load_and_process_experiments(data_root_dir: str, exclude_patterns=None):
    """
    Load and process all experiment pairs, organizing them into coordinated lists.

    Args:
        data_root_dir: Root directory to search for CSV files
        exclude_patterns: List of patterns to exclude (e.g., ['deepseek'])

    Returns:
        power_dfs: List of power dataframes
        results_dfs: List of results dataframes
        experiment_info: List of dictionaries with experiment metadata
    """
    # Find matching experiment pairs
    matched_pairs = discover_experiment_pairs(data_root_dir)

    # Filter out excluded patterns
    if exclude_patterns:
        filtered_pairs = []
        for pair in matched_pairs:
            should_include = True
            for pattern in exclude_patterns:
                if (
                    pattern.lower() in pair[0].lower()
                    or pattern.lower() in pair[1].lower()
                ):
                    should_include = False
                    break
            if should_include:
                filtered_pairs.append(pair)

        print(
            f"Filtered out {len(matched_pairs) - len(filtered_pairs)} pairs. Remaining pairs: {len(filtered_pairs)}"
        )
        matched_pairs = filtered_pairs

    # Process each pair
    power_dfs = []
    results_dfs = []
    experiment_info = []

    for power_csv, results_csv in matched_pairs:
        print(f"Processing {power_csv} and {results_csv}")

        # Extract experiment information
        info = extract_experiment_info(power_csv)
        if not info:
            print(f"Could not extract experiment info from {power_csv}")
            continue

        # Parse CSV files
        power_df = parse_power_csv(power_csv)
        results_df = parse_results_csv(results_csv)

        # Skip if parsing failed
        if power_df.empty or results_df.empty:
            print(f"Skipping empty dataframes for {power_csv} and {results_csv}")
            continue

        # Add to our lists
        power_dfs.append(power_df)
        results_dfs.append(results_df)
        experiment_info.append(info)

    print(f"Successfully processed {len(power_dfs)} experiment pairs")
    return power_dfs, results_dfs, experiment_info


def add_token_columns_to_power_df(
    power_df: pd.DataFrame, results_df: pd.DataFrame
) -> pd.DataFrame:
    """
    For each time window in power_df, compute the tokens that are being processed by
    tracking prefill and decode tokens across the processing duration.

    Args:
        power_df: DataFrame with a 'timestamp' column (in seconds relative to experiment start).
        results_df: DataFrame from the results CSV. Expects columns:
            - "Request Time" (datetime64)
            - "Completion Time" (datetime64)
            - "Prefill Tokens"
            - "Decode Tokens"

    Returns:
        The original power_df with added columns:
            - "prefill_tokens": the prefill tokens being processed in that window
            - "decode_tokens": the decode tokens being processed in that window
            - "batch_size": the batch size being processed in that window
            - "effective_batch_size": the effective batch size in that window
    """
    # Use the earliest request time as the reference
    start_time = results_df["Request Time"].min()

    # Create new columns with times in seconds relative to start_time
    results_df = results_df.copy()
    results_df["req_sec"] = results_df["Request Time"] - start_time
    results_df["comp_sec"] = results_df["Completion Time"] - start_time

    window_duration = 0.25  # 250ms window
    prefill_tokens_list = []
    decode_tokens_list = []
    batch_size_list = []
    effective_batch_list = []
    prefill_throughput_list = []
    decode_throughput_list = []

    for t in power_df["timestamp"]:
        window_start = t
        window_end = t + window_duration

        # Find all requests that overlap with this time window
        overlapping = results_df[
            (results_df["comp_sec"] > window_start)
            & (results_df["req_sec"] < window_end)
        ]

        total_prefill = 0.0
        total_decode = 0.0
        total_batch = 0.0
        total_effective_batch = 0.0
        total_prefill_throughput = 0.0
        total_decode_throughput = 0.0

        # For each overlapping query, compute the tokens being processed in this window
        for _, row in overlapping.iterrows():
            q_start = row["req_sec"]
            q_end = row["comp_sec"]

            # Compute the overlap (in seconds) between the query interval and the window
            overlap = max(0.0, min(window_end, q_end) - max(window_start, q_start))
            query_duration = q_end - q_start

            if query_duration > 0:
                fraction = overlap / query_duration
                # Use prefill and decode columns instead of input/output tokens
                total_prefill += row["Prefill Tokens"] * fraction
                total_decode += row["Decode Tokens"] * fraction
                total_batch += row["Batch Size"] * fraction
                total_effective_batch += row["Effective Batch Size"] * fraction
                total_prefill_throughput += row["Prefill Throughput"] * fraction
                total_decode_throughput += row["Decode Throughput"] * fraction

        prefill_tokens_list.append(total_prefill)
        decode_tokens_list.append(total_decode)
        batch_size_list.append(total_batch)
        effective_batch_list.append(total_effective_batch)
        prefill_throughput_list.append(total_prefill_throughput)
        decode_throughput_list.append(total_decode_throughput)

    # Add the new columns to the power_df
    power_df["prefill_tokens"] = prefill_tokens_list
    power_df["decode_tokens"] = decode_tokens_list
    power_df["batch_size"] = batch_size_list
    power_df["effective_batch_size"] = effective_batch_list
    power_df["prefill_throughput"] = prefill_throughput_list
    power_df["decode_throughput"] = decode_throughput_list

    return power_df


def align_and_truncate_data(power_dfs, results_dfs):
    """
    Align power and results dataframes by their timestamps and truncate
    power data to only include the time periods when inference was happening.

    Args:
        power_dfs: List of power dataframes
        results_dfs: List of results dataframes

    Returns:
        truncated_power_dfs: List of truncated and aligned power dataframes
    """
    truncated_power_dfs = []

    for i in range(len(power_dfs)):
        # Normalize timestamps relative to start
        power_start = power_dfs[i]["timestamp"].min()
        print(
            f"Power start timestamp: {datetime.datetime.fromtimestamp(power_start).strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Find request time boundaries
        request_start = results_dfs[i]["Request Time"].min()
        request_end = results_dfs[i]["Completion Time"].max()

        print(f"Start delay: {request_start - power_start:.2f} seconds")
        print(f"End at: {request_end - power_start:.2f} seconds")

        # Truncate power data to only include the period when inference was happening
        truncated_df = power_dfs[i][
            (power_dfs[i]["timestamp"] >= request_start)
            & (power_dfs[i]["timestamp"] <= request_end)
        ].copy()

        # Reset indices
        truncated_df.reset_index(drop=True, inplace=True)

        # Normalize timestamps relative to first request
        truncated_df["timestamp"] -= request_start

        # Adjust results timestamps as well (in-place)
        results_dfs[i]["Request Time"] -= request_start
        results_dfs[i]["Completion Time"] -= request_start

        truncated_power_dfs.append(truncated_df)

    return truncated_power_dfs


def plot_power_trace(power_df, results_df, title="Power Trace", save_path=None):
    """
    Plot power trace with request/completion markers.

    Args:
        power_df: Processed power dataframe
        results_df: Processed results dataframe
        title: Plot title
        save_path: Path to save the figure
    """
    plt.figure(figsize=(15, 8))

    # Plot power trace
    plt.plot(power_df["timestamp"], power_df["power"], label="Power (W)")

    # Plot token metrics if available
    if "prefill_tokens" in power_df.columns:
        plt.plot(
            power_df["timestamp"],
            power_df["prefill_tokens"],
            label="Prefill Tokens",
            alpha=0.5,
        )
    if "decode_tokens" in power_df.columns:
        plt.plot(
            power_df["timestamp"],
            power_df["decode_tokens"],
            label="Decode Tokens",
            alpha=0.5,
        )

    # Mark request and completion times
    plt.scatter(
        results_df["Request Time"],
        [0] * len(results_df),
        c="blue",
        marker="^",
        label="Request Time",
        s=40,
    )
    plt.scatter(
        results_df["Completion Time"],
        [0] * len(results_df),
        c="red",
        marker="v",
        label="Completion Time",
        s=40,
    )

    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Power (W) / Token Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def save_processed_data(
    truncated_power_dfs, results_dfs, output_path="processed_data/power_trace_data.npz"
):
    """
    Save processed power and token data to an NPZ file.

    Args:
        truncated_power_dfs: List of processed power dataframes
        results_dfs: List of results dataframes
        output_path: Path to save the NPZ file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Extract metadata from results
    tensor_parallelism = []
    poisson_rate = []
    model_sizes = []

    # First pass to get max length for padding
    max_len = max([len(df) for df in truncated_power_dfs])

    # Initialize arrays
    power_traces = []
    prefill_tokens = []
    decode_tokens = []
    batch_sizes = []
    effective_batch_sizes = []
    prefill_throughputs = []
    decode_throughputs = []

    for i in range(len(truncated_power_dfs)):
        print(
            f"Processing dataset {i+1}/{len(truncated_power_dfs)}: "
            f"{len(truncated_power_dfs[i])} power samples, {len(results_dfs[i])} results"
        )

        # Extract metadata
        tensor_parallelism.append(results_dfs[i]["Tensor Parallel Size"].iloc[0])
        poisson_rate.append(results_dfs[i]["Poisson Arrival Rate"].iloc[0])
        model_sizes.append(results_dfs[i]["Model Size"].iloc[0])

        # Extract and pad data traces
        power_trace = truncated_power_dfs[i]["power"].values
        prefill = truncated_power_dfs[i]["prefill_tokens"].values
        decode = truncated_power_dfs[i]["decode_tokens"].values
        batch = truncated_power_dfs[i]["batch_size"].values
        eff_batch = truncated_power_dfs[i]["effective_batch_size"].values
        prefill_tput = truncated_power_dfs[i]["prefill_throughput"].values
        decode_tput = truncated_power_dfs[i]["decode_throughput"].values

        # Pad arrays to max_len for consistent shape
        power_trace = np.pad(power_trace, (0, max_len - len(power_trace)), "edge")
        prefill = np.pad(
            prefill, (0, max_len - len(prefill)), "constant", constant_values=0
        )
        decode = np.pad(
            decode, (0, max_len - len(decode)), "constant", constant_values=0
        )
        batch = np.pad(batch, (0, max_len - len(batch)), "constant", constant_values=0)
        eff_batch = np.pad(
            eff_batch, (0, max_len - len(eff_batch)), "constant", constant_values=0
        )
        prefill_tput = np.pad(
            prefill_tput,
            (0, max_len - len(prefill_tput)),
            "constant",
            constant_values=0,
        )
        decode_tput = np.pad(
            decode_tput, (0, max_len - len(decode_tput)), "constant", constant_values=0
        )

        # Append to lists
        power_traces.append(power_trace)
        prefill_tokens.append(prefill)
        decode_tokens.append(decode)
        batch_sizes.append(batch)
        effective_batch_sizes.append(eff_batch)
        prefill_throughputs.append(prefill_tput)
        decode_throughputs.append(decode_tput)

    # Convert to numpy arrays
    power_traces = np.array(power_traces)
    prefill_tokens = np.array(prefill_tokens)
    decode_tokens = np.array(decode_tokens)
    batch_sizes = np.array(batch_sizes)
    effective_batch_sizes = np.array(effective_batch_sizes)
    prefill_throughputs = np.array(prefill_throughputs)
    decode_throughputs = np.array(decode_throughputs)
    tensor_parallelism = np.array(tensor_parallelism)
    poisson_rate = np.array(poisson_rate)
    model_sizes = np.array(model_sizes)

    # Save to NPZ file
    np.savez(
        output_path,
        power_traces=power_traces,
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
        batch_sizes=batch_sizes,
        effective_batch_sizes=effective_batch_sizes,
        prefill_throughputs=prefill_throughputs,
        decode_throughputs=decode_throughputs,
        tensor_parallelism=tensor_parallelism,
        poisson_rate=poisson_rate,
        model_sizes=model_sizes,
    )

    print(f"Saved processed data to {output_path}")


def main():
    """Main function to run the entire data processing pipeline."""
    data_root_dir = "../client"
    exclude_patterns = ["deepseek"]  # Models to exclude
    output_dir = "processed_data"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load and process experiments
    print("Loading and processing experiments...")
    power_dfs, results_dfs, experiment_info = load_and_process_experiments(
        data_root_dir, exclude_patterns=exclude_patterns
    )

    if not power_dfs:
        print("No valid experiment pairs found. Exiting.")
        return

    # Align and truncate data
    print("Aligning and truncating data...")
    truncated_power_dfs = align_and_truncate_data(power_dfs, results_dfs)

    # Add token columns to power dataframes
    print("Adding token processing information to power data...")
    for i in range(len(truncated_power_dfs)):
        truncated_power_dfs[i] = add_token_columns_to_power_df(
            truncated_power_dfs[i], results_dfs[i]
        )

    # Plot examples
    print("Generating example plots...")
    for i in range(min(3, len(truncated_power_dfs))):
        info = experiment_info[i]
        plot_title = (
            f"Model: {info['model_name']}, TP: {info['tp']}, Rate: {info['rate']}"
        )
        plot_path = os.path.join(output_dir, f"power_trace_{i}.png")
        plot_power_trace(
            truncated_power_dfs[i],
            results_dfs[i],
            title=plot_title,
            save_path=plot_path,
        )

    # Save processed data
    print("Saving processed data...")
    save_processed_data(
        truncated_power_dfs,
        results_dfs,
        output_path=os.path.join(output_dir, "power_trace_data.npz"),
    )

    print("Data processing complete!")


if __name__ == "__main__":
    main()
