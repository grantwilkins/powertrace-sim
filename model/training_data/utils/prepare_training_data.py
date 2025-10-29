import argparse
import glob
import json
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def calculate_token_workload_efficient(
    power_df: pd.DataFrame, results_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate token workload accounting for short prefill times relative to power sampling interval.

    Args:
        power_df: DataFrame with power measurements and timestamps
        results_df: DataFrame with benchmark results

    Returns:
        Updated power_df with token workload columns
    """
    power_df = power_df.copy()
    power_df["prefill_tokens"] = 0
    power_df["decode_tokens"] = 0
    power_df["active_requests"] = 0
    power_df["prefill_requests"] = 0

    results_df["prefill_start"] = results_df["request_timestamp"]
    results_df["prefill_end"] = results_df["prefill_start"] + results_df["prefill_time"]
    results_df["decode_end"] = results_df["prefill_end"] + results_df["decode_time"]

    if len(power_df) > 1:
        avg_interval = (power_df["timestamp"].max() - power_df["timestamp"].min()) / (
            len(power_df) - 1
        )
    else:
        avg_interval = 0.25

    for i in range(len(power_df)):
        current_time = power_df.iloc[i]["timestamp"]

        if i == 0:
            interval_start = current_time - avg_interval / 2
        else:
            interval_start = (current_time + power_df.iloc[i - 1]["timestamp"]) / 2

        if i == len(power_df) - 1:
            interval_end = current_time + avg_interval / 2
        else:
            interval_end = (current_time + power_df.iloc[i + 1]["timestamp"]) / 2

        prefill_in_interval = (results_df["prefill_start"] >= interval_start) & (
            results_df["prefill_start"] < interval_end
        )
        prefill_tokens = results_df.loc[prefill_in_interval, "input_tokens"].sum()
        prefill_request_count = prefill_in_interval.sum()

        decode_active = (results_df["prefill_end"] <= current_time) & (
            current_time <= results_df["decode_end"]
        )
        decode_tokens = results_df.loc[decode_active, "output_tokens"].sum()

        active_requests = (
            (results_df["prefill_start"] <= current_time)
            & (current_time <= results_df["decode_end"])
        ).sum()

        power_df.iloc[i, power_df.columns.get_loc("prefill_tokens")] = prefill_tokens
        power_df.iloc[i, power_df.columns.get_loc("decode_tokens")] = decode_tokens
        power_df.iloc[i, power_df.columns.get_loc("active_requests")] = active_requests
        power_df.iloc[i, power_df.columns.get_loc("prefill_requests")] = (
            prefill_request_count
        )

    return power_df


def discover_experiment_pairs(data_root_dir: str) -> List[Tuple[str, str]]:
    """
    Finds matching pairs of power CSVs and results CSVs based on
    {MODEL_NAME}_tp{TP}_p{RATE}_d{DATE}.csv and
    results_{MODEL_NAME}_{RATE}_{TP}_d{DATE}_final.csv.

    Returns:
        A list of (power_csv_path, results_csv_path).
    """
    all_csvs = glob.glob(os.path.join(data_root_dir, "**", "*.csv"), recursive=True)
    all_jsons = glob.glob(os.path.join(data_root_dir, "**", "*.json"), recursive=True)
    print(f"Found {len(all_csvs)} CSV files in {data_root_dir}")

    power_files = []
    results_files = []

    for csv_path in all_csvs:
        base = os.path.basename(csv_path)
        # Accept both old pattern (_p) and new pattern (rate)
        if "_tp" in base and ("_p" in base or "rate" in base):
            power_files.append(csv_path)
        else:
            print(
                f"Skipping power file {csv_path} as it doesn't match expected pattern"
            )

    for json_path in all_jsons:
        base = os.path.basename(json_path)
        # Accept vllm- prefix OR files that will match our patterns
        if "vllm-" in base or any(pattern in base for pattern in ["llama-3", "gpt-oss", "deepseek"]):
            results_files.append(json_path)
        else:
            print(
                f"Skipping results file {json_path} as it doesn't match expected pattern"
            )

    print(
        f"Found {len(power_files)} power files and {len(results_files)} results files"
    )

    # Organize files by (tp, rate) for matching
    from collections import defaultdict
    power_by_key = defaultdict(list)
    results_by_key = defaultdict(list)

    for pfile in power_files:
        pinfo = extract_power_info(pfile)
        if not pinfo:
            continue
        p_model, p_tp, p_rate, p_date = pinfo
        # Key is just (tp, rate) since model is the same within a directory
        # Normalize rate to float for consistent matching
        key = (p_tp, float(p_rate))
        power_by_key[key].append((pfile, p_date))

    for rfile in results_files:
        rinfo = extract_results_info(rfile)
        if not rinfo:
            continue
        r_model, r_tp, r_rate, r_date = rinfo
        # Normalize rate to float for consistent matching
        key = (r_tp, float(r_rate))
        results_by_key[key].append((rfile, r_date))

    matched_pairs = []

    # For each (tp, rate) combination, match CSVs to JSONs by closest following timestamp
    for key in power_by_key:
        if key not in results_by_key:
            print(f"No results files found for tp={key[0]}, rate={key[1]}")
            continue

        power_list = sorted(power_by_key[key], key=lambda x: x[1].replace('-', ''))  # Sort by normalized date
        results_list = sorted(results_by_key[key], key=lambda x: x[1].replace('-', ''))  # Sort by normalized date

        used_results = set()

        for pfile, p_date in power_list:
            # Find the closest following JSON file
            best_match = None
            min_time_diff = float('inf')

            # Normalize timestamps to comparable format (remove all hyphens)
            p_date_clean = p_date.replace('-', '')

            for rfile, r_date in results_list:
                if rfile in used_results:
                    continue

                r_date_clean = r_date.replace('-', '')

                # JSON file should come after CSV file (or very close to it)
                if r_date_clean >= p_date_clean:
                    try:
                        time_diff = int(r_date_clean) - int(p_date_clean)
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            best_match = rfile
                    except ValueError:
                        # If conversion fails, skip this pair
                        continue

            if best_match:
                matched_pairs.append((pfile, best_match))
                used_results.add(best_match)
                print(f"Matched: {os.path.basename(pfile)} -> {os.path.basename(best_match)} (time_diff={min_time_diff})")
            else:
                print(f"No match found for {os.path.basename(pfile)} (date={p_date_clean})")

    print(
        f"Found {len(power_files)} power files, {len(results_files)} results files, "
        f"matched {len(matched_pairs)} pairs."
    )
    return matched_pairs


def extract_results_info(filename: str) -> Optional[Tuple[str, str, str, str]]:
    base = os.path.basename(filename)
    # Pattern for Llama-3.1 files (old format)
    llama_match = re.match(
        r"vllm-([\d\.]+)qps-tp(\d+)-Llama-3.1-(\d+)B-Instruct(-FP8)?-(.*).json", base
    )
    if llama_match:
        rate = llama_match.group(1)
        tp = llama_match.group(2)
        model_size = llama_match.group(3)
        date = llama_match.group(5)
        return model_size, tp, rate, date

    # Pattern for new Llama-3.1 format with timestamp at end (timestamp may have hyphens)
    llama_new_match = re.match(
        r"vllm-([\d\.]+)qps-tp(\d+)-Llama-3\.1-(\d+)B-Instruct-([\d-]+).json", base
    )
    if llama_new_match:
        rate = llama_new_match.group(1)
        tp = llama_new_match.group(2)
        model_size = llama_new_match.group(3)
        date = llama_new_match.group(4)
        return model_size, tp, rate, date

    # Pattern for gpt-oss files: vllm-{rate}qps-tp{tp}-gpt-oss-{size}b-{timestamp}.json
    gpt_oss_match = re.match(
        r"vllm-([\d\.]+)qps-tp(\d+)-gpt-oss-(\d+)b-([\d-]+).json", base
    )
    if gpt_oss_match:
        rate = gpt_oss_match.group(1)
        tp = gpt_oss_match.group(2)
        model_size = gpt_oss_match.group(3)
        date = gpt_oss_match.group(4)
        return model_size, tp, rate, date

    # Pattern for new gpt-oss files: gpt-oss-{size}b_tp{tp}_rate{rate}_iter{iter}_{timestamp}.json
    gpt_oss_new_match = re.match(
        r"gpt-oss-(\d+)b_tp(\d+)_rate([\d\.]+)_iter\d+_([\d-]+).json", base
    )
    if gpt_oss_new_match:
        model_size = gpt_oss_new_match.group(1)
        tp = gpt_oss_new_match.group(2)
        rate = gpt_oss_new_match.group(3)
        date = gpt_oss_new_match.group(4)
        return model_size, tp, rate, date

    deepseek_match = re.match(
        r"vllm-([\d\.]+)qps-tp(\d+)-DeepSeek-R1-Distill-Llama-(\d+)B-(.*).json", base
    )
    if deepseek_match:
        rate = deepseek_match.group(1)
        tp = deepseek_match.group(2)
        model_size = deepseek_match.group(3)
        date = deepseek_match.group(4)
        return model_size, tp, rate, date

    print(f"Could not extract results info from: {base}")
    return None


def extract_power_info(filename: str) -> Optional[Tuple[str, str, str, str]]:
    base = os.path.basename(filename)
    # Pattern for old Llama-3 files: llama-3-{size}b_tp{tp}_p{rate}_d{date}.csv
    llama_match = re.match(r"llama-3-(\d+)b_tp(\d+)_p([\d\.]+)_d(.*).csv", base)
    if llama_match:
        model_size = llama_match.group(1)
        tp = llama_match.group(2)
        rate = llama_match.group(3)
        date = llama_match.group(4)
        return model_size, tp, rate, date

    # Pattern for new Llama-3 files: llama-3-{size}b_tp{tp}_{category}_{distribution}_rate{rate}_iter{iter}_{timestamp}.csv
    llama_new_match = re.match(
        r"llama-3-(\d+)b_tp(\d+)_\w+_\w+_rate([\d\.]+)_iter\d+_([\d-]+).csv", base
    )
    if llama_new_match:
        model_size = llama_new_match.group(1)
        tp = llama_new_match.group(2)
        rate = llama_new_match.group(3)
        date = llama_new_match.group(4)
        return model_size, tp, rate, date

    # Pattern for gpt-oss files with category/distribution: gpt-oss-{size}b_tp{tp}_{category}_{distribution}_rate{rate}_iter{iter}_{timestamp}.csv
    gpt_oss_match = re.match(
        r"gpt-oss-(\d+)b_tp(\d+)_\w+_\w+_rate([\d\.]+)_iter\d+_([\d-]+).csv", base
    )
    if gpt_oss_match:
        model_size = gpt_oss_match.group(1)
        tp = gpt_oss_match.group(2)
        rate = gpt_oss_match.group(3)
        date = gpt_oss_match.group(4)
        return model_size, tp, rate, date

    # Pattern for simpler gpt-oss files: gpt-oss-{size}b_tp{tp}_rate{rate}_iter{iter}_{timestamp}.csv
    gpt_oss_simple_match = re.match(
        r"gpt-oss-(\d+)b_tp(\d+)_rate([\d\.]+)_iter\d+_([\d-]+).csv", base
    )
    if gpt_oss_simple_match:
        model_size = gpt_oss_simple_match.group(1)
        tp = gpt_oss_simple_match.group(2)
        rate = gpt_oss_simple_match.group(3)
        date = gpt_oss_simple_match.group(4)
        return model_size, tp, rate, date

    # Pattern for DeepSeek files
    deepseek_match = re.match(
        r"deepseek-r1-distill-(\d+)b_tp(\d+)_p([\d\.]+)_d(.*).csv", base
    )
    if deepseek_match:
        model_size = deepseek_match.group(1)
        tp = deepseek_match.group(2)
        rate = deepseek_match.group(3)
        date = deepseek_match.group(4)
        return model_size, tp, rate, date

    print(f"Could not extract power info from: {base}")
    return None


def analyze_benchmark_results(json_file: str) -> pd.DataFrame:
    with open(json_file, "r") as f:
        data = json.load(f)

    results = []

    experiment_info = extract_results_info(json_file)
    model_size = int(experiment_info[0])
    tp = int(experiment_info[1])
    poisson_rate = float(experiment_info[2])

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
                    e2e_latency = data.get("e2el", [])[i] if "e2el" in data and i < len(data.get("e2el", [])) else None
                    if e2e_latency is not None:
                        decode_time = e2e_latency - prefill_time
                    else:
                        itls_for_request = (
                            data["itls"][i] if i < len(data["itls"]) and isinstance(data["itls"][i], list) else []
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
            }
        )

    time_steps = []
    time = 0
    for i in range(len(results)):
        time += np.random.exponential(1.0 / poisson_rate)
        time_steps.append(time)
    for i in range(len(results)):
        results[i]["time_step"] = time_steps[i]
    df = pd.DataFrame(results)

    return df


def parse_power_csv(csv_path: str) -> pd.DataFrame:
    """
    Parse the GPU power trace CSV and sum up the first X power values of each group of 8 rows,
    where X is the tensor parallelism extracted from the filename (_tpX).
    The timestamp for each sample is taken as the minimum timestamp within the group.

    Expects columns (for each GPU row):
        timestamp, power.draw [W], utilization.gpu [%], memory.used [MiB]
    """
    try:
        df = pd.read_csv(csv_path, skipinitialspace=True)
        df.columns = [col.strip().lower() for col in df.columns]

        if "power.draw [w]" in df.columns:
            df.rename(columns={"power.draw [w]": "power"}, inplace=True)
            df.rename(columns={"memory.used [mib]": "memory"}, inplace=True)
        if df["power"].dtype == object:
            df["power"] = df["power"].replace(r"[^\d.]", "", regex=True)
            df["power"] = pd.to_numeric(df["power"])
        if "memory" in df.columns and df["memory"].dtype == object:
            df["memory"] = df["memory"].replace(r"[^\d.]", "", regex=True)
            df["memory"] = pd.to_numeric(df["memory"])
        if "utilization.gpu [%]" in df.columns:
            # format is 0 %
            df.rename(columns={"utilization.gpu [%]": "gpu_utilization"}, inplace=True)
            df["gpu_utilization"] = df["gpu_utilization"].replace(
                r"[^\d.]", "", regex=True
            )
            df["gpu_utilization"] = pd.to_numeric(df["gpu_utilization"])

        time_col = None
        for c in df.columns:
            if "time" in c:
                time_col = c
                break
        if not time_col:
            raise ValueError("No timestamp column found in power CSV")
        df[time_col] = pd.to_datetime(df[time_col])
        df.rename(columns={time_col: "timestamp"}, inplace=True)

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
                    "utilization": x.iloc[:tensor_parallelism][
                        "gpu_utilization"
                    ].mean(),
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


def extract_experiment_info(power_csv_path: str) -> Optional[dict]:
    """
    Extract experiment information from the power CSV filename.

    Args:
        power_csv_path: Path to the power CSV file

    Returns:
        Dict with model_name, tp, rate, and date
    """
    base = os.path.basename(power_csv_path)

    # Try old format first: {model}_tp{tp}_p{rate}_d{date}.csv
    model_match = re.match(r"(.*)_tp(\d+)_p([\d\.]+)_d(.*).csv", base)
    if model_match:
        return {
            "model_name": model_match.group(1),
            "tp": int(model_match.group(2)),
            "rate": float(model_match.group(3)),
            "date": model_match.group(4),
        }

    # Try new format with category/distribution: {model}_tp{tp}_{category}_{distribution}_rate{rate}_iter{iter}_{timestamp}.csv
    new_match = re.match(
        r"(.*)_tp(\d+)_\w+_\w+_rate([\d\.]+)_iter\d+_(.*).csv", base
    )
    if new_match:
        return {
            "model_name": new_match.group(1),
            "tp": int(new_match.group(2)),
            "rate": float(new_match.group(3)),
            "date": new_match.group(4),
        }

    # Try simpler format: {model}_tp{tp}_rate{rate}_iter{iter}_{timestamp}.csv
    simple_match = re.match(
        r"(.*)_tp(\d+)_rate([\d\.]+)_iter\d+_(.*).csv", base
    )
    if simple_match:
        return {
            "model_name": simple_match.group(1),
            "tp": int(simple_match.group(2)),
            "rate": float(simple_match.group(3)),
            "date": simple_match.group(4),
        }

    return None


def infer_hardware_from_path(file_path: str) -> str:
    """
    Infer hardware type from the directory path.

    Args:
        file_path: Path to the file

    Returns:
        Hardware type (e.g., 'a100', 'h100')
    """
    path_lower = file_path.lower()
    if '-a100' in path_lower or '_a100' in path_lower:
        return 'a100'
    elif '-h100' in path_lower or '_h100' in path_lower:
        return 'h100'
    else:
        # Default to a100 if can't determine
        return 'a100'


def load_and_process_experiments(
    data_root_dir: str, exclude_patterns: Optional[str] = None
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[dict]]:
    """
    Load and process all experiment pairs, organizing them into coordinated lists.

    Args:
        data_root_dir: Root directory to search for CSV files
        exclude_patterns: List of patterns to exclude (e.g., ['deepseek'])

    Returns:
        power_dfs: List of power dataframes with token workload information
        results_dfs: List of results dataframes with aligned timestamps
        experiment_info: List of dictionaries with experiment metadata
    """
    matched_pairs = discover_experiment_pairs(data_root_dir)
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

    power_dfs = []
    results_dfs = []
    experiment_info = []

    for power_csv, results_json in matched_pairs:
        print(f"Processing {power_csv} and {results_json}")
        info = extract_experiment_info(power_csv)
        if not info:
            print(f"Could not extract experiment info from {power_csv}")
            continue

        # Infer hardware from path
        info['hardware'] = infer_hardware_from_path(power_csv)

        power_df = parse_power_csv(power_csv)
        results_df = analyze_benchmark_results(results_json)
        if power_df.empty or results_df.empty:
            print(f"Skipping empty dataframes for {power_csv} and {results_json}")
            continue

        results_df = align_request_timestamps(results_df, power_df, info["rate"])
        power_df = calculate_token_workload_efficient(power_df, results_df)
        power_dfs.append(power_df)
        results_dfs.append(results_df)
        experiment_info.append(info)

    print(f"Successfully processed {len(power_dfs)} experiment pairs")
    return power_dfs, results_dfs, experiment_info


def align_request_timestamps(
    results_df: pd.DataFrame, power_df: pd.DataFrame, poisson_rate: float
) -> pd.DataFrame:
    """
    Generate synthetic timestamps for requests using a Poisson process,
    aligned with the min and max timestamps from the power measurements.

    Args:
        results_df: DataFrame with benchmark results
        power_df: DataFrame with power measurements and timestamps
        poisson_rate: Rate parameter for the Poisson process (requests per second)

    Returns:
        Updated results_df with aligned timestamps
    """
    min_timestamp = power_df["timestamp"].min()
    max_timestamp = power_df["timestamp"].max()
    time_range = max_timestamp - min_timestamp
    if "time_step" in results_df.columns:
        max_relative = results_df["time_step"].max()
        results_df["request_timestamp"] = (
            min_timestamp + (results_df["time_step"] / max_relative) * time_range
        )
    else:
        num_requests = len(results_df)
        timestamps = []
        current_time = min_timestamp
        for _ in range(num_requests):
            interarrival = np.random.exponential(1.0 / poisson_rate)
            current_time += interarrival
            if current_time > max_timestamp:
                current_time = (
                    min_timestamp + (current_time - min_timestamp) % time_range
                )
            timestamps.append(current_time)

        timestamps.sort()
        results_df["request_timestamp"] = timestamps

    results_df["request_datetime"] = pd.to_datetime(
        results_df["request_timestamp"], unit="s"
    )

    return results_df


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Process power and benchmark results CSV files"
    )
    argparser.add_argument(
        "--data_root_dir",
        type=str,
        required=True,
        help="Root directory containing power and results CSV files",
    )
    argparser.add_argument(
        "--save_path",
        type=str,
        default=None,
        required=True,
        help="Directory to save processed data",
    )
    argparser.add_argument(
        "--exclude_patterns",
        nargs="*",
        default=None,
        help="List of patterns to exclude from processing (e.g., ['deepseek'])",
    )
    args = argparser.parse_args()

    all_power_dfs, all_results_dfs, all_experiment_info = load_and_process_experiments(
        args.data_root_dir, exclude_patterns=args.exclude_patterns
    )

    if len(all_power_dfs) == 0:
        print("ERROR: No experiment pairs were successfully processed!")
        print("Please check:")
        print("  1. The data_root_dir contains matching power CSV and results JSON files")
        print("  2. Files follow the expected naming pattern")
        print("  3. Files are not empty or corrupted")
        exit(1)

    timestamps = []  # from power_df
    prefill_tokens = []  # from power_df
    decode_tokens = []  # from power_df
    active_requests = []  # from power_df
    input_tokens = []  # from results_df
    output_tokens = []  # from results_df
    ttfts = []  # from results_df
    e2e_latencies = []  # from results_df
    prefill_times = []  # from results_df
    decode_times = []  # from results_df
    prefill_throughputs = []  # from results_df
    decode_throughputs = []  # from results_df
    request_timestamps = []  # from results_df

    tensor_parallelism = []
    poisson_rate = []
    model_sizes = []
    power_traces = []
    hardware = []

    max_len_power = max([len(df) for df in all_power_dfs])
    max_len_results = max([len(df) for df in all_results_dfs])

    for i in range(len(all_power_dfs)):
        power_df = all_power_dfs[i]
        results_df = all_results_dfs[i]
        info = all_experiment_info[i]

        # pad power trace to max_len with edge values
        power_trace = power_df["power"].values
        power_trace = np.pad(
            power_trace, (0, max_len_power - len(power_trace)), mode="edge"
        )
        timestamp_arr = power_df["timestamp"].values
        timestamp_arr = np.pad(
            timestamp_arr, (0, max_len_power - len(timestamp_arr)), mode="linear_ramp"
        )
        prefill_tokens_arr = power_df["prefill_tokens"].values
        prefill_tokens_arr = np.pad(
            prefill_tokens_arr,
            (0, max_len_power - len(prefill_tokens_arr)),
            mode="edge",
        )
        decode_tokens_arr = power_df["decode_tokens"].values
        decode_tokens_arr = np.pad(
            decode_tokens_arr, (0, max_len_power - len(decode_tokens_arr)), mode="edge"
        )
        active_requests_arr = power_df["active_requests"].values
        active_requests_arr = np.pad(
            active_requests_arr,
            (0, max_len_power - len(active_requests_arr)),
            mode="edge",
        )

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

        # Extract relevant data
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
        hardware.append("A100")

    # convert lists to numpy arrays
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
    hardware = np.array(hardware)

    np.savez(
        args.save_path,
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
        hardware=hardware,
    )
