"""
Prepare training data from random workload experiments.
This script processes data and saves it per (model, hardware, tp) combination.
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
from prepare_training_data import load_and_process_experiments


def organize_by_model_hardware(
    power_dfs, results_dfs, experiment_info
) -> Dict[tuple, Dict[str, List]]:
    """
    Organize experiments by (model, hardware) combination, keeping all TP values together.

    Returns:
        Dict mapping (model_name, hardware) -> {'power_dfs': [...], 'results_dfs': [...], 'info': [...]}
    """
    organized = defaultdict(lambda: {"power_dfs": [], "results_dfs": [], "info": []})

    for i, info in enumerate(experiment_info):
        # Use model_name from info (extracted from power CSV filename)
        model_name = info["model_name"]

        # Key is just (model, hardware), TP will be stored in the data
        key = (model_name, info["hardware"])
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

    # Get unique TPs for reporting
    unique_tps = sorted(set(info["tp"] for info in experiment_info))

    print(
        f"Processing {model_name} {hardware} (TPs: {unique_tps}): {len(power_dfs)} experiments, "
        f"max_power_len={max_len_power}, max_results_len={max_len_results}"
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

    for i in range(len(power_dfs)):
        power_df = power_dfs[i]
        results_df = results_dfs[i]
        info = experiment_info[i]

        # Calculate average timestamp interval for padding
        if len(power_df) > 1:
            avg_interval = (power_df["timestamp"].max() - power_df["timestamp"].min()) / (len(power_df) - 1)
        else:
            avg_interval = 0.25  # default 250ms

        # Pad power trace: use minimum power as idle power
        power_trace = power_df["power"].values
        idle_power = power_trace.min() if len(power_trace) > 0 else 0
        power_trace = np.pad(
            power_trace, (0, max_len_power - len(power_trace)), mode="constant", constant_values=idle_power
        )

        # Pad timestamps: continue incrementing at average interval for monotonicity
        timestamp_arr = power_df["timestamp"].values
        if len(timestamp_arr) < max_len_power:
            last_timestamp = timestamp_arr[-1] if len(timestamp_arr) > 0 else 0
            padding_length = max_len_power - len(timestamp_arr)
            padded_timestamps = last_timestamp + avg_interval * (np.arange(1, padding_length + 1))
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
            decode_tokens_arr, (0, max_len_power - len(decode_tokens_arr)), mode="constant", constant_values=0
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

    # Create output filename (no TP in filename, all TPs combined)
    output_filename = f"random_{model_name}_{hardware}.npz"
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
    )

    print(f"Saved {output_path} with {len(power_dfs)} experiments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process random workload data and save per (model, hardware, tp)"
    )
    parser.add_argument(
        "--data_root_dir",
        type=str,
        required=True,
        help="Root directory containing random-* subdirectories",
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
    print(f"Loading experiments from {args.data_root_dir}...")
    power_dfs, results_dfs, experiment_info = load_and_process_experiments(
        args.data_root_dir, exclude_patterns=args.exclude_patterns
    )

    # Organize by (model, hardware) - all TPs combined
    print("\nOrganizing by (model, hardware)...")
    organized_data = organize_by_model_hardware(
        power_dfs, results_dfs, experiment_info
    )

    # Save each group
    print(f"\nSaving {len(organized_data)} groups...")
    for (model_name, hardware), group_data in organized_data.items():
        save_npz_for_group(group_data, model_name, hardware, args.output_dir)

    print("\nDone!")
