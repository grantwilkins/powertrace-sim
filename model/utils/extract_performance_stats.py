"""
Extract TTFT and TPOT statistics from benchmark data and create distribution parameters.
This creates a structured JSON file with performance parameters for different model/hardware/TP configurations.
"""

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class PerformanceStats:
    """Container for performance statistics."""

    model_name: str
    model_size_b: int
    hardware: str
    tensor_parallelism: int

    # TTFT statistics (in seconds)
    ttft_mean: float
    ttft_median: float
    ttft_std: float
    ttft_p99: float

    # TPOT statistics (in seconds)
    tpot_mean: float
    tpot_median: float
    tpot_std: float
    tpot_p99: float

    # Distribution parameters
    ttft_gamma_shape: float
    ttft_gamma_scale: float
    tpot_gaussian_mean: float
    tpot_gaussian_std: float

    # Metadata
    num_requests: int
    benchmark_date: Optional[str] = None
    request_rate_qps: Optional[float] = None


def extract_model_info(
    filename: str,
) -> Optional[Tuple[str, int, str, int, float, str]]:
    """
    Extract model information from benchmark filename.

    Returns:
        Tuple of (model_name, model_size_b, hardware, tensor_parallelism, qps, date)
    """
    base = os.path.basename(filename)

    # Pattern for DeepSeek files
    deepseek_match = re.match(
        r"vllm-([\d\.]+)qps-tp(\d+)-DeepSeek-R1-Distill-Llama-(\d+)B-(\d{8}-\d{6})\.json",
        base,
    )
    if deepseek_match:
        qps = float(deepseek_match.group(1))
        tp = int(deepseek_match.group(2))
        size_b = int(deepseek_match.group(3))
        date = deepseek_match.group(4)

        # Infer hardware from path
        hardware = "H100" if "h100" in filename.lower() else "A100"

        return "deepseek-r1-distill", size_b, hardware, tp, qps, date

    # Pattern for Llama files
    llama_match = re.match(
        r"vllm-([\d\.]+)qps-tp(\d+)-Llama-3\.1-(\d+)B-Instruct(?:-FP8)?-(\d{8}-\d{6})\.json",
        base,
    )
    if llama_match:
        qps = float(llama_match.group(1))
        tp = int(llama_match.group(2))
        size_b = int(llama_match.group(3))
        date = llama_match.group(4)

        # Infer hardware from path
        hardware = "H100" if "h100" in filename.lower() else "A100"

        return "llama-3.1", size_b, hardware, tp, qps, date

    return None


def fit_ttft_heteroskedastic(input_lens: List[int], ttft_s: List[float]) -> Dict:
    """
    Fit heteroskedastic log-linear model with input-dependent variance,
    filtering out P99 outliers to improve CDF matching in the bulk distribution.

    Model:
    log(TTFT) = a0 + a1 * log(input_tokens) + N(0, sigma_total^2)
    where sigma_total = sqrt(sigma_base^2 + sigma_input^2)

    Args:
        input_lens: List of input token counts
        ttft_s: List of TTFT values in seconds

    Returns:
        Dict with model parameters:
        - type, intercept, slope, sigma_base, sigma_intercept, sigma_slope
        - n_train, nin_min, nin_max
        - summary_stats
    """
    from sklearn.linear_model import LinearRegression

    if not input_lens or not ttft_s or len(input_lens) != len(ttft_s):
        # Fallback for missing data
        return {
            "type": "heteroskedastic_log_linear",
            "intercept": 0.0,
            "slope": 1.0,
            "sigma_base": 0.1,
            "sigma_intercept": -2.3,
            "sigma_slope": 0.0,
            "n_train": 0,
            "nin_min": 1,
            "nin_max": 1024,
            "summary_stats": {
                "mean_seconds": 0.0,
                "median_seconds": 0.0,
                "std_seconds": 0.0,
                "min_observed": 0.0,
                "max_observed": 0.0,
            },
        }

    input_lens = np.array(input_lens)
    ttft_s = np.array(ttft_s)

    # Filter out invalid values
    valid_mask = (input_lens > 0) & (ttft_s > 0)
    input_lens = input_lens[valid_mask]
    ttft_s = ttft_s[valid_mask]

    if len(input_lens) < 10:
        # Not enough data
        return {
            "type": "heteroskedastic_log_linear",
            "intercept": np.log(np.mean(ttft_s)) if len(ttft_s) > 0 else 0.0,
            "slope": 0.5,
            "sigma_base": 0.1,
            "sigma_intercept": -2.3,
            "sigma_slope": 0.0,
            "n_train": len(ttft_s),
            "nin_min": int(input_lens.min()) if len(input_lens) > 0 else 1,
            "nin_max": int(input_lens.max()) if len(input_lens) > 0 else 1024,
            "summary_stats": {
                "mean_seconds": float(np.mean(ttft_s)) if len(ttft_s) > 0 else 0.0,
                "median_seconds": float(np.median(ttft_s)) if len(ttft_s) > 0 else 0.0,
                "std_seconds": float(np.std(ttft_s)) if len(ttft_s) > 0 else 0.0,
                "min_observed": float(np.min(ttft_s)) if len(ttft_s) > 0 else 0.0,
                "max_observed": float(np.max(ttft_s)) if len(ttft_s) > 0 else 0.0,
            },
        }

    # Filter out P99 outliers to focus on bulk distribution
    p99_threshold = np.quantile(ttft_s, 0.99)
    bulk_mask = ttft_s <= p99_threshold
    input_lens_bulk = input_lens[bulk_mask]
    ttft_s_bulk = ttft_s[bulk_mask]

    print(f"  Filtered {(~bulk_mask).sum()} P99 outliers (>{p99_threshold:.6f}s), keeping {len(ttft_s_bulk)} samples")

    # Log transform
    log_input = np.log(input_lens_bulk)
    log_ttft = np.log(ttft_s_bulk)

    # Step 1: Fit mean model in log-log space
    X = log_input.reshape(-1, 1)
    mean_model = LinearRegression()
    mean_model.fit(X, log_ttft)

    a0 = float(mean_model.intercept_)
    a1 = float(mean_model.coef_[0])

    # Step 2: Decompose variance into base + input-dependent components
    predictions = mean_model.predict(X)
    residuals = log_ttft - predictions

    # Base variance (median of squared residuals)
    residual_sq = residuals**2
    sigma_base = float(np.sqrt(np.median(residual_sq)))

    # Model input-dependent variance
    log_residual_sq = np.log(residual_sq + 1e-10)

    var_model = LinearRegression()
    var_model.fit(X, log_residual_sq)

    b0 = float(var_model.intercept_)
    b1 = float(var_model.coef_[0])

    sigma_intercept = b0 / 2.0
    sigma_slope = b1 / 2.0

    return {
        "type": "heteroskedastic_log_linear",
        "intercept": a0,
        "slope": a1,
        "sigma_base": sigma_base,
        "sigma_intercept": sigma_intercept,
        "sigma_slope": sigma_slope,
        "n_train": len(ttft_s_bulk),
        "nin_min": int(input_lens.min()),
        "nin_max": int(input_lens.max()),
        "summary_stats": {
            "mean_seconds": float(np.mean(ttft_s_bulk)),
            "median_seconds": float(np.median(ttft_s_bulk)),
            "std_seconds": float(np.std(ttft_s_bulk)),
            "min_observed": float(np.min(ttft_s_bulk)),
            "max_observed": float(np.max(ttft_s_bulk)),
        },
    }


def parse_benchmark_file(filepath: str) -> Optional[PerformanceStats]:
    """
    Parse a benchmark JSON file and extract performance statistics.

    Args:
        filepath: Path to benchmark JSON file

    Returns:
        PerformanceStats object or None if parsing failed
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        # Extract model info from filename
        model_info = extract_model_info(filepath)
        if not model_info:
            print(f"Could not parse model info from {filepath}")
            return None

        model_name, model_size_b, hardware, tp, qps, date = model_info

        # Extract basic statistics (convert from ms to seconds)
        ttft_mean = data.get("mean_ttft_ms", 0) / 1000.0
        ttft_median = data.get("median_ttft_ms", 0) / 1000.0
        ttft_std = data.get("std_ttft_ms", 0) / 1000.0
        ttft_p99 = data.get("p99_ttft_ms", 0) / 1000.0

        tpot_mean = data.get("mean_tpot_ms", 0) / 1000.0
        tpot_median = data.get("median_tpot_ms", 0) / 1000.0
        tpot_std = data.get("std_tpot_ms", 0) / 1000.0
        tpot_p99 = data.get("p99_tpot_ms", 0) / 1000.0

        # Get number of requests
        num_requests = len(data.get("ttfts", []))

        # Collect raw data for fitting
        ttft_values_s = data.get("ttfts", [])
        input_lens = data.get("input_lens", [])

        # Fit log-linear model for TTFT
        if ttft_values_s and input_lens and len(ttft_values_s) == len(input_lens):
            ttft_seconds = [v for v in ttft_values_s]
            ttft_model = fit_ttft_heteroskedastic(input_lens, ttft_seconds)
        else:
            # Fallback when data is incomplete
            ttft_model = {
                "type": "log_linear",
                "intercept": np.log(ttft_mean) if ttft_mean > 0 else 0.0,
                "slope": 0.5,
                "sigma_log": 0.1,
                "n_train": 0,
                "nin_min": 1,
                "nin_max": 1024,
                "summary_stats": {
                    "mean_seconds": ttft_mean,
                    "median_seconds": ttft_median,
                    "std_seconds": ttft_std,
                    "min_observed": 0.0,
                    "max_observed": ttft_p99,
                },
            }

        # TPOT is modeled as Gaussian
        tpot_model = {
            "type": "gaussian",
            "mean": tpot_mean,
            "std": max(tpot_std, 0.0001),
            "summary_stats": {
                "mean_seconds": tpot_mean,
                "median_seconds": tpot_median,
                "std_seconds": tpot_std,
                "min_observed": 0.0,
                "max_observed": tpot_p99,
            },
        }

        return PerformanceStats(
            model_name=model_name,
            model_size_b=model_size_b,
            hardware=hardware,
            tensor_parallelism=tp,
            ttft_mean=ttft_mean,
            ttft_median=ttft_median,
            ttft_std=ttft_std,
            ttft_p99=ttft_p99,
            tpot_mean=tpot_mean,
            tpot_median=tpot_median,
            tpot_std=tpot_std,
            tpot_p99=tpot_p99,
            ttft_gamma_shape=ttft_model.get("slope", 1.0),  # Store model params
            ttft_gamma_scale=ttft_model.get("intercept", 1.0),
            tpot_gaussian_mean=tpot_model["mean"],
            tpot_gaussian_std=tpot_model["std"],
            num_requests=num_requests,
            benchmark_date=date,
            request_rate_qps=qps,
        )

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def discover_benchmark_files(data_root_dir: str) -> List[str]:
    """
    Discover all benchmark JSON files in the data directory.

    Args:
        data_root_dir: Root directory to search

    Returns:
        List of paths to benchmark JSON files
    """
    patterns = ["**/vllm-*qps-tp*-DeepSeek-*.json", "**/vllm-*qps-tp*-Llama-*.json"]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(data_root_dir, pattern), recursive=True))

    return sorted(files)


def create_performance_database(data_root_dir: str, output_file: str):
    """
    Create a comprehensive performance database from benchmark files.
    Uses new log-linear model for TTFT and Gaussian for TPOT.

    Args:
        data_root_dir: Root directory containing benchmark data
        output_file: Output JSON file path
    """
    print(f"Searching for benchmark files in {data_root_dir}...")
    benchmark_files = discover_benchmark_files(data_root_dir)
    print(f"Found {len(benchmark_files)} benchmark files")

    if not benchmark_files:
        print("No benchmark files found. Check your data directory path.")
        return

    # Organize raw data by configuration key
    # We'll collect ALL raw samples and fit once per configuration
    raw_data_by_config = {}

    failed_files = []
    for filepath in benchmark_files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract model info
            model_info = extract_model_info(filepath)
            if not model_info:
                failed_files.append(filepath)
                continue

            model_name, model_size_b, hardware, tp, qps, date = model_info
            key = f"{model_name}_{model_size_b}b_{hardware.lower()}_tp{tp}"

            if key not in raw_data_by_config:
                raw_data_by_config[key] = {
                    "model_name": model_name,
                    "model_size_b": model_size_b,
                    "hardware": hardware,
                    "tensor_parallelism": tp,
                    "input_lens": [],
                    "ttft_values_s": [],
                    "tpot_values_s": [],
                    "num_experiments": 0,
                }

            # Collect raw samples from this file
            ttft_s = data.get("ttfts", [])
            input_lens = data.get("input_lens", [])

            # Add TTFT samples (already in seconds)
            if ttft_s and input_lens and len(ttft_s) == len(input_lens):
                raw_data_by_config[key]["input_lens"].extend(input_lens)
                raw_data_by_config[key]["ttft_values_s"].extend(ttft_s)

            # For TPOT: use ITLs (inter-token latencies) which are per-token decode times
            # ITLs are lists of token-by-token latencies (already in seconds)
            itls = data.get("itls", [])
            if itls:
                # Flatten all ITLs from all requests into one list
                for request_itls in itls:
                    if request_itls:  # Skip empty lists
                        raw_data_by_config[key]["tpot_values_s"].extend(request_itls)

            # Fallback: if no ITLs, use mean TPOT as proxy
            elif "mean_tpot_ms" in data:
                tpot_s = data["mean_tpot_ms"] / 1000.0
                # Add as repeated samples
                n_samples = len(ttft_s) if ttft_s else 1
                raw_data_by_config[key]["tpot_values_s"].extend([tpot_s] * n_samples)

            raw_data_by_config[key]["num_experiments"] += 1

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            failed_files.append(filepath)

    print(f"Successfully processed {len(benchmark_files) - len(failed_files)} files")
    if failed_files:
        print(f"Failed to process {len(failed_files)} files")

    # Now fit models for each configuration
    final_database = {}

    for key, config_data in raw_data_by_config.items():
        input_lens = config_data["input_lens"]
        ttft_values_s = config_data["ttft_values_s"]
        tpot_values_s = config_data["tpot_values_s"]

        if not ttft_values_s:
            print(f"Skipping {key}: no TTFT data")
            continue

        # Fit log-linear TTFT model
        ttft_model = fit_ttft_heteroskedastic(input_lens, ttft_values_s)

        # Fit Gaussian TPOT model
        if tpot_values_s:
            tpot_array = np.array(tpot_values_s)
            tpot_model = {
                "type": "gaussian",
                "mean": float(np.mean(tpot_array)),
                "std": float(np.std(tpot_array)),
                "summary_stats": {
                    "mean_seconds": float(np.mean(tpot_array)),
                    "median_seconds": float(np.median(tpot_array)),
                    "std_seconds": float(np.std(tpot_array)),
                    "min_observed": float(np.min(tpot_array)),
                    "max_observed": float(np.max(tpot_array)),
                },
            }
        else:
            tpot_model = {
                "type": "gaussian",
                "mean": 0.02,
                "std": 0.001,
                "summary_stats": {
                    "mean_seconds": 0.02,
                    "median_seconds": 0.02,
                    "std_seconds": 0.001,
                    "min_observed": 0.0,
                    "max_observed": 0.1,
                },
            }

        final_database[key] = {
            "model_name": config_data["model_name"],
            "model_size_b": config_data["model_size_b"],
            "hardware": config_data["hardware"],
            "tensor_parallelism": config_data["tensor_parallelism"],
            "num_experiments": config_data["num_experiments"],
            "total_requests": len(ttft_values_s),
            "ttft_model": ttft_model,  # NEW FORMAT
            "tpot_distribution": tpot_model
        }

    # Save to JSON
    print(f"Saving performance database to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(final_database, f, indent=2)

    # Print summary
    print(f"\n=== Performance Database Summary ===")
    print(f"Total configurations: {len(final_database)}")

    print("\nConfigurations found:")
    for key, config in final_database.items():
        ttft_stats = config["ttft_model"]["summary_stats"]
        tpot_stats = config["tpot_distribution"]["summary_stats"]
        print(
            f"  {key}: TTFT={ttft_stats['mean_seconds']:.3f}s, TPOT={tpot_stats['mean_seconds']:.4f}s ({config['num_experiments']} experiments)"
        )


def load_performance_database(filepath: str) -> Dict:
    """Load the performance database from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def get_performance_params(
    database: Dict,
    model_name: str,
    model_size_b: int,
    hardware: str,
    tensor_parallelism: int,
) -> Optional[Dict]:
    """
    Get performance parameters for a specific configuration.

    Args:
        database: Performance database loaded from JSON
        model_name: Model name (e.g., "llama-3.1", "deepseek-r1-distill")
        model_size_b: Model size in billions
        hardware: Hardware type ("A100", "H100")
        tensor_parallelism: Tensor parallelism value

    Returns:
        Dictionary with TTFT and TPOT distribution parameters, or None if not found
    """
    key = f"{model_name}_{model_size_b}b_{hardware.lower()}_tp{tensor_parallelism}"
    return database.get(key)


def sample_ttft_tpot(
    database: Dict,
    model_name: str,
    model_size_b: int,
    hardware: str,
    tensor_parallelism: int,
    size: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample TTFT and TPOT values from the database distributions.

    Returns:
        Tuple of (ttft_samples, tpot_samples) in seconds
    """
    params = get_performance_params(
        database, model_name, model_size_b, hardware, tensor_parallelism
    )
    if not params:
        raise ValueError(
            f"No performance data found for {model_name}-{model_size_b}B on {hardware} TP{tensor_parallelism}"
        )

    # Sample TTFT from gamma distribution
    ttft_shape = params["ttft_distribution"]["shape"]
    ttft_scale = params["ttft_distribution"]["scale"]
    ttft_samples = stats.gamma.rvs(ttft_shape, scale=ttft_scale, size=size)

    # Sample TPOT from Gaussian distribution
    tpot_mean = params["tpot_distribution"]["mean"]
    tpot_std = params["tpot_distribution"]["std"]
    tpot_samples = stats.norm.rvs(tpot_mean, tpot_std, size=size)

    # Ensure positive values
    ttft_samples = np.maximum(ttft_samples, 0.01)  # Minimum 10ms TTFT
    tpot_samples = np.maximum(tpot_samples, 0.001)  # Minimum 1ms TPOT

    return ttft_samples, tpot_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract performance statistics from benchmark data"
    )
    parser.add_argument(
        "--data_root_dir",
        type=str,
        required=True,
        help="Root directory containing benchmark JSON files",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/Users/grantwilkins/powertrace-sim/model/config/performance_database.json",
        help="Output JSON file for performance database",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run a test of the sampling functionality"
    )

    args = parser.parse_args()

    if args.test and os.path.exists(args.output_file):
        # Test sampling functionality
        print("Testing sampling functionality...")
        database = load_performance_database(args.output_file)

        # Try sampling from an available configuration
        for key, config in database.items():
            print(f"\nTesting {key}:")
            try:
                ttft_samples, tpot_samples = sample_ttft_tpot(
                    database,
                    config["model_name"],
                    config["model_size_b"],
                    config["hardware"],
                    config["tensor_parallelism"],
                    size=10,
                )

                print(f"  TTFT samples (s): {ttft_samples}")
                print(f"  TPOT samples (s): {tpot_samples}")
                print(
                    f"  TTFT mean: {np.mean(ttft_samples):.3f}s (expected: {config['ttft_distribution']['summary_stats']['mean_seconds']:.3f}s)"
                )
                print(
                    f"  TPOT mean: {np.mean(tpot_samples):.4f}s (expected: {config['tpot_distribution']['summary_stats']['mean_seconds']:.4f}s)"
                )
                break
            except Exception as e:
                print(f"  Error: {e}")
    else:
        # Create the database
        create_performance_database(args.data_root_dir, args.output_file)

        if os.path.exists(args.output_file):
            print(f"\nPerformance database created successfully!")
            print(f"Usage example:")
            print(
                f"  from utils.extract_performance_stats import load_performance_database, sample_ttft_tpot"
            )
            print(f"  database = load_performance_database('{args.output_file}')")
            print(
                f"  ttft, tpot = sample_ttft_tpot(database, 'llama-3.1', 8, 'A100', 1)"
            )
