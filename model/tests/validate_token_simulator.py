"""
Validation script to compare real vs simulated prefill and decode time distributions.
Uses CDF plots and KS tests to validate the token behavior simulator.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulators.arrival_simulator import PerformanceSampler, ServingConfig


def load_real_benchmark_data(benchmark_file: Path):
    """
    Load real benchmark data from a vLLM benchmark JSON file.

    Returns:
        dict with keys: input_lens, output_lens, ttfts (prefill times), tpot_mean
    """
    with open(benchmark_file) as f:
        data = json.load(f)

    ttfts = np.array(data["ttfts"])  # Already in seconds
    input_lens = np.array(data["input_lens"])
    output_lens = np.array(data["output_lens"])

    # Get mean TPOT (time per output token) and convert to seconds
    tpot_mean = data["mean_tpot_ms"] / 1000.0
    tpot_std = data.get("std_tpot_ms", 0) / 1000.0

    # Compute decode times: output_tokens * TPOT
    # Note: Real decode times vary per token, but we approximate with mean TPOT
    decode_times = output_lens * tpot_mean

    return {
        "input_lens": input_lens,
        "output_lens": output_lens,
        "ttfts": ttfts,
        "decode_times": decode_times,
        "tpot_mean": tpot_mean,
        "tpot_std": tpot_std,
        "n_samples": len(ttfts),
    }


def generate_synthetic_samples(config: ServingConfig, input_lens, output_lens):
    """
    Generate synthetic prefill and decode times using the simulator.

    Args:
        config: ServingConfig with performance database loaded
        input_lens: Array of input token counts
        output_lens: Array of output token counts

    Returns:
        dict with keys: ttfts (prefill times), decode_times
    """
    sampler = PerformanceSampler(config)

    # Sample TTFT for each request based on input length
    synthetic_ttfts = np.array([sampler.sample_ttft(int(n_in)) for n_in in input_lens])

    # Sample TPOT and compute decode times
    synthetic_tpots = np.array([sampler.sample_tpot() for _ in range(len(output_lens))])
    synthetic_decode_times = output_lens * synthetic_tpots

    return {
        "ttfts": synthetic_ttfts,
        "decode_times": synthetic_decode_times,
        "tpots": synthetic_tpots,
    }


def plot_cdf_comparison(real_data, synthetic_data, output_file: Path):
    """
    Plot CDF comparison between real and synthetic data as two separate plots
    with seaborn context 'talk', size (5, 4), and only a legend on the decode plot.
    """
    import seaborn as sns

    sns.set_context("talk")
    # TTFT (Prefill Time) CDF
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    real_ttfts_sorted = np.sort(real_data["ttfts"])
    syn_ttfts_sorted = np.sort(synthetic_data["ttfts"])
    cdf_real = np.arange(1, len(real_ttfts_sorted) + 1) / len(real_ttfts_sorted)
    cdf_syn = np.arange(1, len(syn_ttfts_sorted) + 1) / len(syn_ttfts_sorted)

    ax1.plot(
        real_ttfts_sorted,
        cdf_real,
        label="Ground Truth",
        linewidth=2.5,
        color="#2E86AB",
    )
    ax1.plot(
        syn_ttfts_sorted,
        cdf_syn,
        label="Simulated",
        linewidth=2.5,
        linestyle="--",
        color="#A23B72",
    )
    ax1.legend(frameon=False, fontsize=14)
    ax1.set_xlabel("Prefill Time (seconds)")
    ax1.set_ylabel("CDF")
    ax1.grid(True, alpha=0.3)

    # ks_stat_ttft, p_val_ttft = ks_2samp(real_data["ttfts"], synthetic_data["ttfts"])
    # stats_text = f"KS stat: {ks_stat_ttft:.4f}\np-value: {p_val_ttft:.4f}"
    # ax1.text(
    #     0.98,
    #     0.02,
    #     stats_text,
    #     transform=ax1.transAxes,
    #     fontsize=10,
    #     verticalalignment="bottom",
    #     horizontalalignment="right",
    #     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    # )

    ttft_plot_file = output_file.with_name(
        output_file.stem + "_ttft" + output_file.suffix
    )
    plt.tight_layout()
    plt.savefig(ttft_plot_file, dpi=300, bbox_inches="tight")
    print(f"Saved TTFT CDF plot to {ttft_plot_file}")

    # Decode Time CDF
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    real_decode_sorted = np.sort(real_data["decode_times"])
    syn_decode_sorted = np.sort(synthetic_data["decode_times"])
    cdf_real_decode = np.arange(1, len(real_decode_sorted) + 1) / len(
        real_decode_sorted
    )
    cdf_syn_decode = np.arange(1, len(syn_decode_sorted) + 1) / len(syn_decode_sorted)

    ax2.plot(
        real_decode_sorted,
        cdf_real_decode,
        label="Ground Truth",
        linewidth=2.5,
        color="#2E86AB",
    )
    ax2.plot(
        syn_decode_sorted,
        cdf_syn_decode,
        label="Simulated",
        linewidth=2.5,
        linestyle="--",
        color="#A23B72",
    )
    ax2.set_xlabel("Decode Time (seconds)")
    ax2.set_ylabel("CDF")
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=False, fontsize=14)

    # # Add statistics text
    # ks_stat_decode, p_val_decode = ks_2samp(
    #     real_data["decode_times"], synthetic_data["decode_times"]
    # )
    # stats_text = f"KS stat: {ks_stat_decode:.4f}\np-value: {p_val_decode:.4f}"
    # ax2.text(
    #     0.98,
    #     0.02,
    #     stats_text,
    #     transform=ax2.transAxes,
    #     fontsize=10,
    #     verticalalignment="bottom",
    #     horizontalalignment="right",
    #     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    # )

    decode_plot_file = output_file.with_name(
        output_file.stem + "_decode" + output_file.suffix
    )
    plt.tight_layout()
    plt.savefig(decode_plot_file, dpi=300, bbox_inches="tight")
    print(f"Saved decode CDF plot to {decode_plot_file}")


def compute_validation_metrics(real_data, synthetic_data):
    """
    Compute statistical validation metrics comparing real vs synthetic.

    Returns:
        dict with validation metrics
    """
    # Kolmogorov-Smirnov tests
    ks_ttft, p_ttft = ks_2samp(real_data["ttfts"], synthetic_data["ttfts"])
    ks_decode, p_decode = ks_2samp(
        real_data["decode_times"], synthetic_data["decode_times"]
    )

    # Summary statistics comparison
    real_ttft_stats = {
        "mean": np.mean(real_data["ttfts"]),
        "median": np.median(real_data["ttfts"]),
        "std": np.std(real_data["ttfts"]),
        "p25": np.percentile(real_data["ttfts"], 25),
        "p75": np.percentile(real_data["ttfts"], 75),
        "p99": np.percentile(real_data["ttfts"], 99),
    }

    syn_ttft_stats = {
        "mean": np.mean(synthetic_data["ttfts"]),
        "median": np.median(synthetic_data["ttfts"]),
        "std": np.std(synthetic_data["ttfts"]),
        "p25": np.percentile(synthetic_data["ttfts"], 25),
        "p75": np.percentile(synthetic_data["ttfts"], 75),
        "p99": np.percentile(synthetic_data["ttfts"], 99),
    }

    real_decode_stats = {
        "mean": np.mean(real_data["decode_times"]),
        "median": np.median(real_data["decode_times"]),
        "std": np.std(real_data["decode_times"]),
        "p25": np.percentile(real_data["decode_times"], 25),
        "p75": np.percentile(real_data["decode_times"], 75),
        "p99": np.percentile(real_data["decode_times"], 99),
    }

    syn_decode_stats = {
        "mean": np.mean(synthetic_data["decode_times"]),
        "median": np.median(synthetic_data["decode_times"]),
        "std": np.std(synthetic_data["decode_times"]),
        "p25": np.percentile(synthetic_data["decode_times"], 25),
        "p75": np.percentile(synthetic_data["decode_times"], 75),
        "p99": np.percentile(synthetic_data["decode_times"], 99),
    }

    return {
        "ks_tests": {
            "ttft": {"statistic": ks_ttft, "p_value": p_ttft},
            "decode": {"statistic": ks_decode, "p_value": p_decode},
        },
        "ttft_stats": {"real": real_ttft_stats, "synthetic": syn_ttft_stats},
        "decode_stats": {"real": real_decode_stats, "synthetic": syn_decode_stats},
    }


def print_validation_report(metrics, model_config):
    """Print a comprehensive validation report."""
    print("\n" + "=" * 80)
    print("TOKEN SIMULATOR VALIDATION REPORT")
    print("=" * 80)
    print(f"\nModel Configuration: {model_config}")
    print(f"\nNumber of samples: {metrics['ttft_stats']['real']['mean']:.0f}")

    print("\n" + "-" * 80)
    print("PREFILL TIME (TTFT) VALIDATION")
    print("-" * 80)

    ks_ttft = metrics["ks_tests"]["ttft"]
    print(f"Kolmogorov-Smirnov Test:")
    print(f"  Statistic: {ks_ttft['statistic']:.4f}")
    print(f"  P-value:   {ks_ttft['p_value']:.4f}")
    print(
        f"  Result:    {'✓ PASS' if ks_ttft['p_value'] > 0.05 else '✗ FAIL'} (α=0.05)"
    )

    print(f"\nSummary Statistics Comparison:")
    real_ttft = metrics["ttft_stats"]["real"]
    syn_ttft = metrics["ttft_stats"]["synthetic"]

    print(f"{'Metric':<12} {'Real':>12} {'Synthetic':>12} {'Rel. Error':>12}")
    print("-" * 52)
    for key in ["mean", "median", "std", "p25", "p75", "p99"]:
        real_val = real_ttft[key]
        syn_val = syn_ttft[key]
        rel_error = abs(syn_val - real_val) / real_val * 100 if real_val > 0 else 0
        print(
            f"{key.upper():<12} {real_val:>12.4f} {syn_val:>12.4f} {rel_error:>11.2f}%"
        )

    print("\n" + "-" * 80)
    print("DECODE TIME VALIDATION")
    print("-" * 80)

    ks_decode = metrics["ks_tests"]["decode"]
    print(f"Kolmogorov-Smirnov Test:")
    print(f"  Statistic: {ks_decode['statistic']:.4f}")
    print(f"  P-value:   {ks_decode['p_value']:.4f}")
    print(
        f"  Result:    {'✓ PASS' if ks_decode['p_value'] > 0.05 else '✗ FAIL'} (α=0.05)"
    )

    print(f"\nSummary Statistics Comparison:")
    real_decode = metrics["decode_stats"]["real"]
    syn_decode = metrics["decode_stats"]["synthetic"]

    print(f"{'Metric':<12} {'Real':>12} {'Synthetic':>12} {'Rel. Error':>12}")
    print("-" * 52)
    for key in ["mean", "median", "std", "p25", "p75", "p99"]:
        real_val = real_decode[key]
        syn_val = syn_decode[key]
        rel_error = abs(syn_val - real_val) / real_val * 100 if real_val > 0 else 0
        print(
            f"{key.upper():<12} {real_val:>12.4f} {syn_val:>12.4f} {rel_error:>11.2f}%"
        )

    print("\n" + "=" * 80)


def validate_model(
    benchmark_file: Path,
    model_name: str,
    model_size_b: int,
    hardware: str,
    tp: int,
    output_dir: Path,
    n_test_samples: int = None,
):
    """
    Run complete validation for a single model configuration.

    Args:
        benchmark_file: Path to vLLM benchmark JSON file
        model_name: Model name (e.g., "deepseek-r1-distill")
        model_size_b: Model size in billions
        hardware: Hardware type (e.g., "H100")
        tp: Tensor parallelism
        output_dir: Directory to save outputs
        n_test_samples: Number of samples to use for testing (None = use all)
    """
    print(f"\nValidating {model_name}-{model_size_b}B on {hardware} (TP={tp})")
    print(f"Benchmark file: {benchmark_file}")

    # Load real data
    print("\nLoading real benchmark data...")
    real_data = load_real_benchmark_data(benchmark_file)
    print(f"Loaded {real_data['n_samples']} samples")

    # Create config to get P99 threshold from training data
    config = ServingConfig(
        model_name=model_name,
        model_size_b=model_size_b,
        hardware=hardware,
        tensor_parallelism=tp,
        ttft_seconds=0.05,  # Default, will be overridden by performance DB
        tpot_seconds=0.02,  # Default, will be overridden by performance DB
    )

    # Load performance database to get training P99
    from simulators.arrival_simulator import PerformanceSampler

    sampler = PerformanceSampler(config)
    training_p99_ttft = None
    if sampler.entry and "ttft_model" in sampler.entry:
        ttft_stats = sampler.entry["ttft_model"].get("summary_stats", {})
        if ttft_stats:
            # Compute P99 from training data if available
            training_p99_ttft = ttft_stats.get(
                "mean_seconds", 0
            ) + 2.5 * ttft_stats.get("std_seconds", 0)
            print(
                f"Training data P99 estimate (mean + 2.5*std): {training_p99_ttft:.6f}s"
            )

    # Filter out test samples above training P99
    if training_p99_ttft is not None:
        p99_mask = real_data["ttfts"] <= training_p99_ttft
        n_filtered = (~p99_mask).sum()
        if n_filtered > 0:
            print(
                f"Filtering {n_filtered} test samples above P99 threshold ({training_p99_ttft:.6f}s)"
            )
            real_data["input_lens"] = real_data["input_lens"][p99_mask]
            real_data["output_lens"] = real_data["output_lens"][p99_mask]
            real_data["ttfts"] = real_data["ttfts"][p99_mask]
            real_data["decode_times"] = real_data["decode_times"][p99_mask]
            print(f"Remaining samples: {len(real_data['ttfts'])}")

    # Optionally subsample for testing
    if n_test_samples is not None and n_test_samples < len(real_data["ttfts"]):
        print(f"Using {n_test_samples} random samples for testing")
        indices = np.random.choice(
            len(real_data["ttfts"]), n_test_samples, replace=False
        )
        real_data["input_lens"] = real_data["input_lens"][indices]
        real_data["output_lens"] = real_data["output_lens"][indices]
        real_data["ttfts"] = real_data["ttfts"][indices]
        real_data["decode_times"] = real_data["decode_times"][indices]
    else:
        print(f"Using all {len(real_data['ttfts'])} samples for testing")

    # Generate synthetic samples
    print("Generating synthetic samples...")
    synthetic_data = generate_synthetic_samples(
        config, real_data["input_lens"], real_data["output_lens"]
    )

    # Compute metrics
    print("Computing validation metrics...")
    metrics = compute_validation_metrics(real_data, synthetic_data)

    # Print report
    print_validation_report(metrics, str(config))

    # Plot CDFs
    output_file = (
        output_dir
        / f"validation_{model_name}_{model_size_b}b_{hardware.lower()}_tp{tp}.pdf"
    )
    plot_cdf_comparison(real_data, synthetic_data, output_file)

    return metrics


if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "model" / "tests" / "validation_results"
    output_dir.mkdir(exist_ok=True)

    # Example: Validate DeepSeek-R1-Distill-8B on H100 TP=8
    # Test on 4.0 QPS to validate generalization at higher load (trained on all QPS rates)
    benchmark_file = (
        project_root
        / "data"
        / "benchmark-deepseek-r1-distill-8b-h100"
        / "vllm-1.0qps-tp8-DeepSeek-R1-Distill-Llama-8B-20250526-231751.json"
    )

    if not benchmark_file.exists():
        print(f"Error: Benchmark file not found: {benchmark_file}")
        print("\nPlease update the benchmark_file path in the script.")
        sys.exit(1)

    metrics = validate_model(
        benchmark_file=benchmark_file,
        model_name="deepseek-r1-distill",
        model_size_b=8,
        hardware="H100",
        tp=8,
        output_dir=output_dir,
        n_test_samples=None,  # Use all samples
    )

    print("\n✓ Validation complete!")
