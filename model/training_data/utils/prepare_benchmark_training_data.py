"""
Prepare training data from benchmark experiments with streamlined naming convention.
This script processes data where CSV and JSON files share identical base names,
differing only in extension (e.g., model_tp1_rate0.25_iter1_timestamp.{csv,json}).
Output is saved per (model, hardware) combination with all TP values included.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import utility functions from the original prepare_training_data.py
from .prepare_training_data import (
    align_request_timestamps,
    calculate_token_workload_efficient,
    infer_hardware_from_path,
    parse_power_csv,
)


def discover_benchmark_pairs(data_root_dir: str) -> List[Tuple[str, str]]:
    """
    Finds matching pairs of power CSVs and config JSONs based on identical base names.

    New naming format:
    - Power: model_tp{TP}_rate{RATE}_iter{ITER}_{TIMESTAMP}.csv
    - Config: model_tp{TP}_rate{RATE}_iter{ITER}_{TIMESTAMP}.json

    Returns:
        A list of (power_csv_path, config_json_path) tuples.
    """
    all_csvs = glob.glob(os.path.join(data_root_dir, "**", "*.csv"), recursive=True)
    all_jsons = glob.glob(os.path.join(data_root_dir, "**", "*.json"), recursive=True)

    print(
        f"Found {len(all_csvs)} CSV files and {len(all_jsons)} JSON files in {data_root_dir}"
    )

    # Create a mapping from base filename (without extension) to full paths
    csv_by_base = {}
    json_by_base = {}

    for csv_path in all_csvs:
        basename = os.path.basename(csv_path)
        # Extract base name without extension
        base_without_ext = os.path.splitext(basename)[0]
        csv_by_base[base_without_ext] = csv_path

    for json_path in all_jsons:
        basename = os.path.basename(json_path)
        # Extract base name without extension
        base_without_ext = os.path.splitext(basename)[0]
        json_by_base[base_without_ext] = json_path

    # Find matching pairs
    matched_pairs = []
    for base_name in csv_by_base:
        if base_name in json_by_base:
            matched_pairs.append((csv_by_base[base_name], json_by_base[base_name]))
            print(
                f"Matched: {os.path.basename(csv_by_base[base_name])} <-> {os.path.basename(json_by_base[base_name])}"
            )
        else:
            print(f"No JSON match for CSV: {os.path.basename(csv_by_base[base_name])}")

    for base_name in json_by_base:
        if base_name not in csv_by_base:
            print(f"No CSV match for JSON: {os.path.basename(json_by_base[base_name])}")

    print(
        f"\nMatched {len(matched_pairs)} pairs out of {len(all_csvs)} CSVs and {len(all_jsons)} JSONs"
    )
    return matched_pairs


def extract_benchmark_info(file_path: str) -> Optional[dict]:
    """
    Extract experiment information from the benchmark file path.

    Expected pattern: {model_name}_tp{tp}_rate{rate}_iter{iter}_{timestamp}.{ext}
    Examples:
    - gpt-oss-20b_tp1_rate0.25_iter1_2025-10-21-04-23-19.csv
    - gpt-oss-120b_tp2_rate1_iter2_2025-10-21-07-55-42.json

    Args:
        file_path: Path to the benchmark file (CSV or JSON)

    Returns:
        Dict with model_name, tp, rate, iter, and timestamp
    """
    base = os.path.basename(file_path)

    # Pattern: {model}_tp{tp}_rate{rate}_iter{iter}_{timestamp}.{ext}
    pattern = r"(.+?)_tp(\d+)_rate([\d\.]+)_iter(\d+)_([\d-]+)\.\w+$"
    match = re.match(pattern, base)

    if match:
        model_name = match.group(1)
        tp = int(match.group(2))
        rate = float(match.group(3))
        iteration = int(match.group(4))
        timestamp = match.group(5)

        return {
            "model_name": model_name,
            "tp": tp,
            "rate": rate,
            "iteration": iteration,
            "timestamp": timestamp,
        }

    print(f"Could not extract benchmark info from: {base}")
    return None


def analyze_benchmark_results(json_file: str) -> pd.DataFrame:
    """
    Analyze benchmark results from a JSON file with the new naming convention.

    Args:
        json_file: Path to the benchmark JSON file

    Returns:
        DataFrame with per-request results including timing and token information
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    results = []

    # Extract info from filename using our new function
    experiment_info = extract_benchmark_info(json_file)
    if not experiment_info:
        print(f"Could not extract info from {json_file}")
        return pd.DataFrame()

    # Extract model size from model name (e.g., "gpt-oss-20b" -> 20)
    model_name = experiment_info["model_name"]
    # Try to extract numeric size from model name
    size_match = re.search(r"(\d+)b", model_name.lower())
    if size_match:
        model_size = int(size_match.group(1))
    else:
        print(f"Could not extract model size from {model_name}, defaulting to 0")
        model_size = 0

    tp = experiment_info["tp"]
    poisson_rate = experiment_info["rate"]

    if (
        "ttfts" in data
        and "input_lens" in data
        and "output_lens" in data
        and "itls" in data
    ):
        for i in range(len(data["ttfts"])):
            if i < len(data["input_lens"]) and i < len(data["output_lens"]):
                if data["ttfts"][i] is not None:
                    prefill_time = data["ttfts"][i]
                    e2e_latency = (
                        data.get("e2el", [])[i]
                        if "e2el" in data and i < len(data.get("e2el", []))
                        else None
                    )
                    if e2e_latency is not None:
                        decode_time = e2e_latency - prefill_time
                    else:
                        itls_for_request = (
                            data["itls"][i]
                            if i < len(data["itls"])
                            and isinstance(data["itls"][i], list)
                            else []
                        )
                        decode_time = sum(itls_for_request) if itls_for_request else 0
                    results.append(
                        {
                            "input_tokens": data["input_lens"][i],
                            "output_tokens": data["output_lens"][i],
                            "prefill_time": prefill_time,
                            "decode_time": decode_time,
                            "ttft_ms": prefill_time * 1000,
                            "e2e_latency": (
                                e2e_latency
                                if e2e_latency is not None
                                else (prefill_time + decode_time)
                            ),
                            "prefill_throughput": (
                                data["input_lens"][i] / prefill_time
                                if prefill_time
                                else 0
                            ),
                            "decode_throughput": (
                                data["output_lens"][i] / decode_time
                                if decode_time
                                else 0
                            ),
                            "model_size": model_size,
                            "tp": tp,
                            "poisson_rate": poisson_rate,
                        }
                    )
    else:
        print("Detailed per-request data not available. Using aggregated statistics.")
        prefill_time_ms = data.get("median_ttft_ms", 0)
        e2e_latency_ms = data.get("median_e2el_ms", 0)
        decode_time_ms = e2e_latency_ms - prefill_time_ms

        results.append(
            {
                "prefill_time": prefill_time_ms / 1000,
                "decode_time": decode_time_ms / 1000,
                "ttft_ms": prefill_time_ms,
                "e2e_latency": e2e_latency_ms / 1000,
                "model_size": model_size,
                "tp": tp,
                "poisson_rate": poisson_rate,
            }
        )

    # Generate synthetic timestamps using Poisson process
    time_steps = []
    time = 0
    for i in range(len(results)):
        time += np.random.exponential(1.0 / poisson_rate)
        time_steps.append(time)
    for i in range(len(results)):
        results[i]["time_step"] = time_steps[i]
    df = pd.DataFrame(results)

    return df


def load_and_process_benchmark_experiments(
    data_root_dir: str, exclude_patterns: Optional[List[str]] = None
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[dict]]:
    """
    Load and process all benchmark experiment pairs.

    Args:
        data_root_dir: Root directory to search for benchmark files
        exclude_patterns: List of patterns to exclude (e.g., ['gpt-oss-120b'])

    Returns:
        power_dfs: List of power dataframes with token workload information
        results_dfs: List of results dataframes with aligned timestamps
        experiment_info: List of dictionaries with experiment metadata
    """
    matched_pairs = discover_benchmark_pairs(data_root_dir)

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
            f"Filtered out {len(matched_pairs) - len(filtered_pairs)} pairs. "
            f"Remaining pairs: {len(filtered_pairs)}"
        )
        matched_pairs = filtered_pairs

    power_dfs = []
    results_dfs = []
    experiment_info = []

    for power_csv, config_json in matched_pairs:
        print(
            f"\nProcessing {os.path.basename(power_csv)} and {os.path.basename(config_json)}"
        )

        # Extract experiment info
        info = extract_benchmark_info(power_csv)
        if not info:
            print(f"Could not extract experiment info from {power_csv}, skipping")
            continue

        # Infer hardware from path
        info["hardware"] = infer_hardware_from_path(power_csv)

        # Parse power CSV and analyze benchmark results
        power_df = parse_power_csv(power_csv)
        results_df = analyze_benchmark_results(config_json)

        if power_df.empty or results_df.empty:
            print(f"Skipping empty dataframes for {power_csv} and {config_json}")
            continue

        # Align timestamps and calculate token workload
        results_df = align_request_timestamps(results_df, power_df, info["rate"])
        power_df = calculate_token_workload_efficient(power_df, results_df)

        power_dfs.append(power_df)
        results_dfs.append(results_df)
        experiment_info.append(info)

    print(f"\n=== Successfully processed {len(power_dfs)} experiment pairs ===")
    return power_dfs, results_dfs, experiment_info


def organize_by_model_hardware(
    power_dfs: List[pd.DataFrame],
    results_dfs: List[pd.DataFrame],
    experiment_info: List[dict],
) -> Dict[tuple, Dict[str, List]]:
    """
    Organize experiments by (model, hardware) combination, keeping all TP values together.

    Returns:
        Dict mapping (model_name, hardware) -> {'power_dfs': [...], 'results_dfs': [...], 'info': [...]}
    """
    organized = defaultdict(lambda: {"power_dfs": [], "results_dfs": [], "info": []})

    for i, info in enumerate(experiment_info):
        # Key is (model, hardware), TP will be stored in the data
        key = (info["model_name"], info["hardware"])
        organized[key]["power_dfs"].append(power_dfs[i])
        organized[key]["results_dfs"].append(results_dfs[i])
        organized[key]["info"].append(info)

    return organized


def save_npz_for_group(
    group_data: Dict[str, List],
    model_name: str,
    hardware: str,
    output_dir: str,
):
    """
    Save NPZ file for a specific (model, hardware) combination with all TP values.
    """
    power_dfs = group_data["power_dfs"]
    results_dfs = group_data["results_dfs"]
    experiment_info = group_data["info"]

    if not power_dfs:
        print(f"No data for {model_name} {hardware}, skipping")
        return

    # Find max lengths for padding
    max_len_power = max([len(df) for df in power_dfs])
    max_len_results = max([len(df) for df in results_dfs])

    # Get unique TPs and rates for reporting
    unique_tps = sorted(set(info["tp"] for info in experiment_info))
    unique_rates = sorted(set(info["rate"] for info in experiment_info))
    unique_iters = sorted(set(info["iteration"] for info in experiment_info))

    print(
        f"Processing {model_name} {hardware}: {len(power_dfs)} experiments\n"
        f"  TPs: {unique_tps}, Rates: {unique_rates}, Iterations: {unique_iters}\n"
        f"  max_power_len={max_len_power}, max_results_len={max_len_results}"
    )

    # Initialize lists for all data
    timestamps = []
    power_traces = []
    prefill_tokens = []
    decode_tokens = []
    active_requests = []
    input_tokens = []
    output_tokens = []
    ttfts = []
    e2e_latencies = []
    prefill_times = []
    decode_times = []
    prefill_throughputs = []
    decode_throughputs = []
    request_timestamps = []
    tensor_parallelism = []
    poisson_rate = []
    model_sizes = []
    hardware_list = []
    iterations = []

    for i in range(len(power_dfs)):
        power_df = power_dfs[i]
        results_df = results_dfs[i]
        info = experiment_info[i]

        # Calculate average timestamp interval for padding
        if len(power_df) > 1:
            avg_interval = (
                power_df["timestamp"].max() - power_df["timestamp"].min()
            ) / (len(power_df) - 1)
        else:
            avg_interval = 0.25  # default 250ms

        # Pad power trace: use minimum power as idle power
        power_trace = power_df["power"].values
        idle_power = power_trace.min() if len(power_trace) > 0 else 0
        power_trace = np.pad(
            power_trace,
            (0, max_len_power - len(power_trace)),
            mode="constant",
            constant_values=idle_power,
        )

        # Pad timestamps: continue incrementing at average interval for monotonicity
        timestamp_arr = power_df["timestamp"].values
        if len(timestamp_arr) < max_len_power:
            last_timestamp = timestamp_arr[-1] if len(timestamp_arr) > 0 else 0
            padding_length = max_len_power - len(timestamp_arr)
            padded_timestamps = last_timestamp + avg_interval * (
                np.arange(1, padding_length + 1)
            )
            timestamp_arr = np.concatenate([timestamp_arr, padded_timestamps])

        # Pad token counts and requests: use 0 (no activity after workload ends)
        prefill_tokens_arr = power_df["prefill_tokens"].values
        prefill_tokens_arr = np.pad(
            prefill_tokens_arr,
            (0, max_len_power - len(prefill_tokens_arr)),
            mode="constant",
            constant_values=0,
        )
        decode_tokens_arr = power_df["decode_tokens"].values
        decode_tokens_arr = np.pad(
            decode_tokens_arr,
            (0, max_len_power - len(decode_tokens_arr)),
            mode="constant",
            constant_values=0,
        )
        active_requests_arr = power_df["active_requests"].values
        active_requests_arr = np.pad(
            active_requests_arr,
            (0, max_len_power - len(active_requests_arr)),
            mode="constant",
            constant_values=0,
        )

        # Pad results data
        input_tokens_arr = results_df["input_tokens"].values
        input_tokens_arr = np.pad(
            input_tokens_arr,
            (0, max_len_results - len(input_tokens_arr)),
            mode="constant",
            constant_values=0,
        )
        output_tokens_arr = results_df["output_tokens"].values
        output_tokens_arr = np.pad(
            output_tokens_arr,
            (0, max_len_results - len(output_tokens_arr)),
            mode="constant",
            constant_values=0,
        )
        ttft_arr = results_df["ttft_ms"].values
        ttft_arr = np.pad(
            ttft_arr,
            (0, max_len_results - len(ttft_arr)),
            mode="constant",
            constant_values=0,
        )
        e2e_latency_arr = results_df["e2e_latency"].values
        e2e_latency_arr = np.pad(
            e2e_latency_arr,
            (0, max_len_results - len(e2e_latency_arr)),
            mode="constant",
            constant_values=0,
        )
        prefill_time_arr = results_df["prefill_time"].values
        prefill_time_arr = np.pad(
            prefill_time_arr,
            (0, max_len_results - len(prefill_time_arr)),
            mode="constant",
            constant_values=0,
        )
        decode_time_arr = results_df["decode_time"].values
        decode_time_arr = np.pad(
            decode_time_arr,
            (0, max_len_results - len(decode_time_arr)),
            mode="constant",
            constant_values=0,
        )
        prefill_throughput_arr = results_df["prefill_throughput"].values
        prefill_throughput_arr = np.pad(
            prefill_throughput_arr,
            (0, max_len_results - len(prefill_throughput_arr)),
            mode="constant",
            constant_values=0,
        )
        decode_throughput_arr = results_df["decode_throughput"].values
        decode_throughput_arr = np.pad(
            decode_throughput_arr,
            (0, max_len_results - len(decode_throughput_arr)),
            mode="constant",
            constant_values=0,
        )
        request_timestamp_arr = results_df["request_timestamp"].values
        request_timestamp_arr = np.pad(
            request_timestamp_arr,
            (0, max_len_results - len(request_timestamp_arr)),
            mode="constant",
            constant_values=0,
        )

        # Append to lists
        power_traces.append(power_trace)
        timestamps.append(timestamp_arr)
        prefill_tokens.append(prefill_tokens_arr)
        decode_tokens.append(decode_tokens_arr)
        active_requests.append(active_requests_arr)

        request_timestamps.append(request_timestamp_arr)
        ttfts.append(ttft_arr)
        prefill_times.append(prefill_time_arr)
        decode_times.append(decode_time_arr)
        e2e_latencies.append(e2e_latency_arr)
        input_tokens.append(input_tokens_arr)
        output_tokens.append(output_tokens_arr)
        prefill_throughputs.append(prefill_throughput_arr)
        decode_throughputs.append(decode_throughput_arr)

        tensor_parallelism.append(results_df["tp"].values[0])
        poisson_rate.append(results_df["poisson_rate"].values[0])
        model_sizes.append(results_df["model_size"].values[0])
        hardware_list.append(info["hardware"])
        iterations.append(info["iteration"])

    # Convert lists to numpy arrays
    timestamps = np.array(timestamps)
    power_traces = np.array(power_traces)
    prefill_tokens = np.array(prefill_tokens)
    decode_tokens = np.array(decode_tokens)
    active_requests = np.array(active_requests)

    input_tokens = np.array(input_tokens)
    output_tokens = np.array(output_tokens)
    ttfts = np.array(ttfts)
    e2e_latencies = np.array(e2e_latencies)
    prefill_times = np.array(prefill_times)
    decode_times = np.array(decode_times)
    prefill_throughputs = np.array(prefill_throughputs)
    decode_throughputs = np.array(decode_throughputs)
    request_timestamps = np.array(request_timestamps)
    tensor_parallelism = np.array(tensor_parallelism)
    poisson_rate = np.array(poisson_rate)
    model_sizes = np.array(model_sizes)
    hardware_arr = np.array(hardware_list)
    iterations = np.array(iterations)

    # Create output filename (no TP in filename, all TPs combined)
    output_filename = f"benchmark_{model_name}_{hardware}.npz"
    output_path = os.path.join(output_dir, output_filename)

    # Save NPZ file
    np.savez(
        output_path,
        timestamps=timestamps,
        power_traces=power_traces,
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
        active_requests=active_requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        ttfts=ttfts,
        e2e_latencies=e2e_latencies,
        prefill_times=prefill_times,
        decode_times=decode_times,
        prefill_throughputs=prefill_throughputs,
        decode_throughputs=decode_throughputs,
        request_timestamps=request_timestamps,
        tensor_parallelism=tensor_parallelism,
        poisson_rate=poisson_rate,
        model_sizes=model_sizes,
        hardware=hardware_arr,
        iterations=iterations,
    )

    print(f"Saved {output_path} with {len(power_dfs)} experiments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process benchmark data with streamlined naming and save per (model, hardware)"
    )
    parser.add_argument(
        "--data_root_dir",
        type=str,
        required=True,
        help="Root directory containing benchmark-* subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed NPZ files",
    )
    parser.add_argument(
        "--exclude_patterns",
        nargs="*",
        default=None,
        help="List of patterns to exclude from processing",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and process all experiments
    print(f"Loading benchmark experiments from {args.data_root_dir}...")
    power_dfs, results_dfs, experiment_info = load_and_process_benchmark_experiments(
        args.data_root_dir, exclude_patterns=args.exclude_patterns
    )

    # Organize by (model, hardware) - all TPs combined
    print("\nOrganizing by (model, hardware)...")
    organized_data = organize_by_model_hardware(power_dfs, results_dfs, experiment_info)

    # Save each group
    print(f"\nSaving {len(organized_data)} groups...")
    for (model_name, hardware), group_data in organized_data.items():
        save_npz_for_group(group_data, model_name, hardware, args.output_dir)

    print("\nDone!")
