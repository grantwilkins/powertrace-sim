#!/usr/bin/env python3
"""
Evaluate all trained models and generate synthetic trace fidelity metrics.
Processes all model/hardware/TP configurations and aggregates results.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp, pearsonr, wasserstein_distance
from statsmodels.tsa.stattools import acf

from model.core.dataset import PowerTraceDataset
from model.core.utils import load_classifier
from model.predictors.smooth_sampler import SmoothingSampler


def compute_autocorrelation_metrics(real_trace, synthetic_trace, max_lags=50):
    """Compare temporal autocorrelation between real and synthetic traces."""
    real_acf = acf(real_trace, nlags=max_lags, fft=True)
    synthetic_acf = acf(synthetic_trace, nlags=max_lags, fft=True)
    acf_correlation, _ = pearsonr(real_acf, synthetic_acf)
    acf_mae = np.mean(np.abs(real_acf - synthetic_acf))

    return {
        "real_acf": real_acf,
        "synthetic_acf": synthetic_acf,
        "acf_r2": acf_correlation,
        "acf_mae": acf_mae,
    }


def evaluate_single_config(
    dataset, classifier, tp, device, smoother, energy_threshold=5.0
):
    """
    Evaluate a single model configuration.

    Returns:
        dict: Metrics if energy error < threshold, None otherwise
    """
    tp_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == tp]

    if len(tp_indices) == 0:
        print(f"  No samples found for TP={tp}")
        return None

    all_original_power = []
    all_sampled_power = []

    # Compute min/max power for clipping
    min_power = float("inf")
    max_power = float("-inf")
    for idx in tp_indices:
        trace_power = dataset.traces[idx]["y"].flatten()
        min_power = min(min_power, np.min(trace_power))
        max_power = max(max_power, np.max(trace_power))

    # Generate synthetic traces
    for idx in tp_indices:
        time, power, states = smoother.sample_power(
            classifier,
            dataset.mu[tp],
            dataset.sigma[tp],
            dataset.traces[idx]["x"],
            dt=0.25,
            smoothing_window=5,
        )
        power = np.clip(power, min_power, max_power)
        original_power = dataset.traces[idx]["y"].flatten()
        sampled_power = power.flatten()

        all_original_power.append(original_power)
        all_sampled_power.append(sampled_power)

    all_original_power = np.concatenate(all_original_power)
    all_sampled_power = np.concatenate(all_sampled_power)

    # Compute energy error
    total_energy_original = np.trapz(all_original_power, dx=0.25)
    total_energy_sampled = np.trapz(all_sampled_power, dx=0.25)
    energy_error_pct = (
        np.abs(total_energy_sampled - total_energy_original)
        / total_energy_original
        * 100
    )

    # Mark if exceeds threshold (but don't skip)
    exceeds_threshold = energy_error_pct > energy_threshold
    if exceeds_threshold:
        print(
            f"  WARNING: Energy error {energy_error_pct:.2f}% > {energy_threshold}% threshold"
        )

    # Compute all metrics
    ks_stat, ks_pvalue = ks_2samp(all_original_power, all_sampled_power)

    autocorr_metrics = compute_autocorrelation_metrics(
        all_original_power, all_sampled_power
    )

    nrmse = np.sqrt(np.mean((all_original_power - all_sampled_power) ** 2)) / (
        np.max(all_original_power) - np.min(all_original_power)
    )

    p95_original = np.percentile(all_original_power, 95)
    p95_sampled = np.percentile(all_sampled_power, 95)
    p95_error_pct = np.abs(p95_original - p95_sampled) / p95_original * 100

    p99_original = np.percentile(all_original_power, 99)
    p99_sampled = np.percentile(all_sampled_power, 99)
    p99_error_pct = np.abs(p99_original - p99_sampled) / p99_original * 100

    metrics = {
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pvalue,
        "acf_r2": autocorr_metrics["acf_r2"],
        "nrmse": nrmse,
        "p95_error_pct": p95_error_pct,
        "p99_error_pct": p99_error_pct,
        "energy_error_pct": energy_error_pct,
        "num_samples": len(tp_indices),
        "exceeds_threshold": exceeds_threshold,
    }

    return metrics


def parse_weight_filename(filename):
    """
    Parse model name, hardware, and TP from weight filename.
    Format: model_hardware_tpX.pt
    """
    basename = os.path.basename(filename).replace(".pt", "")

    # Handle model names with hyphens
    if "deepseek-r1-distill-70b" in basename:
        model = "deepseek-r1-distill-70b"
        rest = basename.replace("deepseek-r1-distill-70b_", "")
    elif "deepseek-r1-distill-8b" in basename:
        model = "deepseek-r1-distill-8b"
        rest = basename.replace("deepseek-r1-distill-8b_", "")
    elif "llama-3-70b" in basename:
        model = "llama-3-70b"
        rest = basename.replace("llama-3-70b_", "")
    elif "llama-3-8b" in basename:
        model = "llama-3-8b"
        rest = basename.replace("llama-3-8b_", "")
    else:
        return None, None, None

    # Parse hardware and TP
    parts = rest.split("_")
    hardware = parts[0]
    tp = int(parts[1].replace("tp", ""))

    return model, hardware, tp


def get_data_file(model, hardware, data_dir):
    """Get the data file path for a given model and hardware."""
    # Map model names to data file patterns
    model_map = {
        "llama-3-8b": "vllm-benchmark_llama-3-8b_{hw}.npz",
        "llama-3-70b": "vllm-benchmark_llama-3-70b_{hw}.npz",
        "deepseek-r1-distill-8b": "vllm-benchmark_deepseek-r1-8b_{hw}.npz",
        "deepseek-r1-distill-70b": "vllm-benchmark_deepseek-r1-70b_{hw}.npz",
    }

    if model not in model_map:
        return None

    filename = model_map[model].format(hw=hardware)
    return os.path.join(data_dir, filename)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all trained models and generate fidelity metrics"
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="./model/best_weights",
        help="Directory containing trained model weights",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./model/training_data",
        help="Directory containing training data files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (cuda, cuda:0, cpu, etc.)",
    )
    parser.add_argument(
        "--energy_threshold",
        type=float,
        default=5.0,
        help="Energy error threshold (%) for including results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results",
    )
    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/metrics", exist_ok=True)

    # Find all weight files
    weight_files = glob.glob(os.path.join(args.weights_dir, "*.pt"))
    print(f"\nFound {len(weight_files)} weight files")

    # Store results: model -> hardware -> tp -> metrics
    all_results = {}

    # Process each weight file
    for weight_file in sorted(weight_files):
        model, hardware, tp = parse_weight_filename(weight_file)

        if model is None:
            print(
                f"\nSkipping {os.path.basename(weight_file)} - could not parse filename"
            )
            continue

        print(f"\n{'=' * 80}")
        print(f"Evaluating: {model} / {hardware} / TP={tp}")
        print(f"{'=' * 80}")

        # Get data file
        data_file = get_data_file(model, hardware, args.data_dir)
        if not os.path.exists(data_file):
            print(f"  Data file not found: {data_file}")
            continue

        # Load dataset
        dataset = PowerTraceDataset(data_file)
        print(f"  Loaded dataset: {data_file}")

        # Determine K from dataset
        tp_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == tp]
        if len(tp_indices) == 0:
            print(f"  No samples found for TP={tp}")
            continue

        # Get K from dataset state labels
        K = dataset.state_labels[tp].n_components
        Dx = dataset.traces[0]["x"].shape[1]

        print(f"  Input dim: {Dx}, Num states: {K}, Num samples: {len(tp_indices)}")

        # Load classifier
        try:
            classifier = load_classifier(
                weight_file,
                device=device,
                Dx=Dx,
                K=K,
            )
            print(f"  Loaded classifier from: {os.path.basename(weight_file)}")
        except Exception as e:
            print(f"  Error loading classifier: {e}")
            continue

        # Create smoother
        smoother = SmoothingSampler(dataset)

        # Evaluate
        metrics = evaluate_single_config(
            dataset, classifier, tp, device, smoother, args.energy_threshold
        )

        if metrics is None:
            continue

        # Store results
        if model not in all_results:
            all_results[model] = {}
        if hardware not in all_results[model]:
            all_results[model][hardware] = {}
        all_results[model][hardware][tp] = metrics

        # Print metrics
        print(f"\n  Metrics:")
        print(f"    KS Statistic: {metrics['ks_statistic']:.4f}")
        print(f"    ACF R²: {metrics['acf_r2']:.4f}")
        print(f"    NRMSE: {metrics['nrmse']:.4f}")
        print(f"    P95 Error: {metrics['p95_error_pct']:.2f}%")
        print(f"    P99 Error: {metrics['p99_error_pct']:.2f}%")
        print(f"    Energy Error: {metrics['energy_error_pct']:.2f}%")

        # Save individual config metrics
        import csv

        config_name = f"{model}_{hardware}_tp{tp}"
        metrics_file = f"{args.output_dir}/metrics/{config_name}_eval.csv"
        with open(metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for key, value in metrics.items():
                writer.writerow([key, value])
        print(f"  Saved metrics to: {metrics_file}")

    # Aggregate results by model (ALL configs and filtered by energy threshold)
    print(f"\n\n{'=' * 80}")
    print("AGGREGATING RESULTS BY MODEL")
    print(f"{'=' * 80}")

    model_stats_all = {}
    model_stats_filtered = {}

    for model in all_results:
        # All configs
        ks_values_all = []
        acf_r2_values_all = []
        nrmse_values_all = []
        p95_error_values_all = []
        p99_error_values_all = []
        energy_error_values_all = []

        # Filtered configs (energy < threshold)
        ks_values_filt = []
        acf_r2_values_filt = []
        nrmse_values_filt = []
        p95_error_values_filt = []
        p99_error_values_filt = []
        energy_error_values_filt = []

        for hardware in all_results[model]:
            for tp in all_results[model][hardware]:
                metrics = all_results[model][hardware][tp]

                # Add to all configs
                ks_values_all.append(metrics["ks_statistic"])
                acf_r2_values_all.append(metrics["acf_r2"])
                nrmse_values_all.append(metrics["nrmse"])
                p95_error_values_all.append(metrics["p95_error_pct"])
                p99_error_values_all.append(metrics["p99_error_pct"])
                energy_error_values_all.append(metrics["energy_error_pct"])

                # Add to filtered if below threshold
                if not metrics["exceeds_threshold"]:
                    ks_values_filt.append(metrics["ks_statistic"])
                    acf_r2_values_filt.append(metrics["acf_r2"])
                    nrmse_values_filt.append(metrics["nrmse"])
                    p95_error_values_filt.append(metrics["p95_error_pct"])
                    p99_error_values_filt.append(metrics["p99_error_pct"])
                    energy_error_values_filt.append(metrics["energy_error_pct"])

        # Store all configs stats
        model_stats_all[model] = {
            "ks_mean": np.mean(ks_values_all),
            "ks_std": np.std(ks_values_all) if len(ks_values_all) > 1 else 0,
            "acf_r2_mean": np.mean(acf_r2_values_all),
            "acf_r2_std": np.std(acf_r2_values_all)
            if len(acf_r2_values_all) > 1
            else 0,
            "nrmse_mean": np.mean(nrmse_values_all),
            "nrmse_std": np.std(nrmse_values_all) if len(nrmse_values_all) > 1 else 0,
            "p95_error_mean": np.mean(p95_error_values_all),
            "p95_error_std": np.std(p95_error_values_all)
            if len(p95_error_values_all) > 1
            else 0,
            "p99_error_mean": np.mean(p99_error_values_all),
            "p99_error_std": np.std(p99_error_values_all)
            if len(p99_error_values_all) > 1
            else 0,
            "energy_error_mean": np.mean(energy_error_values_all),
            "energy_error_std": np.std(energy_error_values_all)
            if len(energy_error_values_all) > 1
            else 0,
            "num_configs": len(ks_values_all),
        }

        # Store filtered stats (only if we have configs that pass threshold)
        if len(ks_values_filt) > 0:
            model_stats_filtered[model] = {
                "ks_mean": np.mean(ks_values_filt),
                "ks_std": np.std(ks_values_filt) if len(ks_values_filt) > 1 else 0,
                "acf_r2_mean": np.mean(acf_r2_values_filt),
                "acf_r2_std": np.std(acf_r2_values_filt)
                if len(acf_r2_values_filt) > 1
                else 0,
                "nrmse_mean": np.mean(nrmse_values_filt),
                "nrmse_std": np.std(nrmse_values_filt)
                if len(nrmse_values_filt) > 1
                else 0,
                "p95_error_mean": np.mean(p95_error_values_filt),
                "p95_error_std": np.std(p95_error_values_filt)
                if len(p95_error_values_filt) > 1
                else 0,
                "p99_error_mean": np.mean(p99_error_values_filt),
                "p99_error_std": np.std(p99_error_values_filt)
                if len(p99_error_values_filt) > 1
                else 0,
                "energy_error_mean": np.mean(energy_error_values_filt),
                "energy_error_std": np.std(energy_error_values_filt)
                if len(energy_error_values_filt) > 1
                else 0,
                "num_configs": len(ks_values_filt),
            }

        print(f"\n{model}:")
        print(f"  Total configs: {model_stats_all[model]['num_configs']}")
        if model in model_stats_filtered:
            print(
                f"  Configs with energy <{args.energy_threshold}%: {model_stats_filtered[model]['num_configs']}"
            )
        else:
            print(
                f"  Configs with energy <{args.energy_threshold}%: 0 (all exceed threshold)"
            )
        print(
            f"  KS: {model_stats_all[model]['ks_mean']:.3f} ± {model_stats_all[model]['ks_std']:.3f}"
        )
        print(
            f"  ACF R²: {model_stats_all[model]['acf_r2_mean']:.3f} ± {model_stats_all[model]['acf_r2_std']:.3f}"
        )
        print(
            f"  NRMSE: {model_stats_all[model]['nrmse_mean']:.3f} ± {model_stats_all[model]['nrmse_std']:.3f}"
        )
        print(
            f"  P95 Err: {model_stats_all[model]['p95_error_mean']:.2f} ± {model_stats_all[model]['p95_error_std']:.2f}%"
        )
        print(
            f"  P99 Err: {model_stats_all[model]['p99_error_mean']:.2f} ± {model_stats_all[model]['p99_error_std']:.2f}%"
        )
        print(
            f"  Energy Err: {model_stats_all[model]['energy_error_mean']:.2f} ± {model_stats_all[model]['energy_error_std']:.2f}%"
        )

    # Save aggregated statistics (all configs)
    import csv

    stats_file_all = f"{args.output_dir}/model_statistics_all.csv"
    with open(stats_file_all, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "num_configs",
                "ks_mean",
                "ks_std",
                "acf_r2_mean",
                "acf_r2_std",
                "nrmse_mean",
                "nrmse_std",
                "p95_error_mean",
                "p95_error_std",
                "p99_error_mean",
                "p99_error_std",
                "energy_error_mean",
                "energy_error_std",
            ]
        )
        for model, stats in sorted(model_stats_all.items()):
            writer.writerow(
                [
                    model,
                    stats["num_configs"],
                    stats["ks_mean"],
                    stats["ks_std"],
                    stats["acf_r2_mean"],
                    stats["acf_r2_std"],
                    stats["nrmse_mean"],
                    stats["nrmse_std"],
                    stats["p95_error_mean"],
                    stats["p95_error_std"],
                    stats["p99_error_mean"],
                    stats["p99_error_std"],
                    stats["energy_error_mean"],
                    stats["energy_error_std"],
                ]
            )
    print(f"\nAll configs statistics saved to: {stats_file_all}")

    # Save filtered statistics (energy < threshold)
    stats_file_filtered = f"{args.output_dir}/model_statistics_filtered.csv"
    with open(stats_file_filtered, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "num_configs",
                "ks_mean",
                "ks_std",
                "acf_r2_mean",
                "acf_r2_std",
                "nrmse_mean",
                "nrmse_std",
                "p95_error_mean",
                "p95_error_std",
                "p99_error_mean",
                "p99_error_std",
                "energy_error_mean",
                "energy_error_std",
            ]
        )
        for model, stats in sorted(model_stats_filtered.items()):
            writer.writerow(
                [
                    model,
                    stats["num_configs"],
                    stats["ks_mean"],
                    stats["ks_std"],
                    stats["acf_r2_mean"],
                    stats["acf_r2_std"],
                    stats["nrmse_mean"],
                    stats["nrmse_std"],
                    stats["p95_error_mean"],
                    stats["p95_error_std"],
                    stats["p99_error_mean"],
                    stats["p99_error_std"],
                    stats["energy_error_mean"],
                    stats["energy_error_std"],
                ]
            )
    print(f"Filtered configs statistics saved to: {stats_file_filtered}")

    # Generate LaTeX tables
    generate_latex_table(
        model_stats_all,
        f"{args.output_dir}/trace_eval_table_all.tex",
        filter_desc="all configurations",
    )
    print(
        f"LaTeX table (all configs) saved to: {args.output_dir}/trace_eval_table_all.tex"
    )

    if model_stats_filtered:
        generate_latex_table(
            model_stats_filtered,
            f"{args.output_dir}/trace_eval_table.tex",
            filter_desc=f"configurations with energy error <{args.energy_threshold}%",
        )
        print(
            f"LaTeX table (filtered) saved to: {args.output_dir}/trace_eval_table.tex"
        )


def generate_latex_table(model_stats, output_file, filter_desc="configurations"):
    """Generate LaTeX table for the paper."""
    model_display = {
        "llama-3-8b": r"\texttt{Llama-3.1 (8B)}",
        "llama-3-70b": r"\texttt{Llama-3.1 (70B)}",
        "deepseek-r1-distill-8b": r"\texttt{DeepSeek-Distill (8B)}",
        "deepseek-r1-distill-70b": r"\texttt{DeepSeek-Distill (70B)}",
    }

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        f"\\caption{{Synthetic trace fidelity metrics on validation data. Results averaged across hardware platforms and tensor parallelism {filter_desc}.}}"
    )
    lines.append(r"\label{tab:trace_eval_model}")
    lines.append(r"\vspace{0.4em}")
    lines.append(r"\resizebox{\linewidth}{!}{")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} &")
    lines.append(r"\textbf{KS $\downarrow$} &")
    lines.append(r"\textbf{ACF $R^2$ $\uparrow$} &")
    lines.append(r"\textbf{NRMSE $\downarrow$} &")
    lines.append(r"\textbf{P95 Err (\%) $\downarrow$} &")
    lines.append(r"\textbf{P99 Err (\%) $\downarrow$} &")
    lines.append(r"\textbf{$\Delta$Energy (\%) $\downarrow$} \\")
    lines.append(r"\midrule")

    model_order = [
        "llama-3-8b",
        "deepseek-r1-distill-8b",
        "llama-3-70b",
        "deepseek-r1-distill-70b",
    ]

    for model in model_order:
        if model not in model_stats:
            continue

        stats = model_stats[model]
        suffix = ""

        # Add markers if only 1 config (no std)
        if stats["num_configs"] == 1:
            if model == "deepseek-r1-distill-8b":
                suffix = r"$^{*}$"
            elif model == "llama-3-70b":
                suffix = r"$^{\dagger}$"

        # Format with or without std
        if stats["num_configs"] == 1:
            ks_str = f"{stats['ks_mean']:.3f}"
            acf_str = f"{stats['acf_r2_mean']:.3f}"
            nrmse_str = f"{stats['nrmse_mean']:.3f}"
            p95_str = f"{stats['p95_error_mean']:.2f}"
            p99_str = f"{stats['p99_error_mean']:.2f}"
            energy_str = f"{stats['energy_error_mean']:.2f}"
        else:
            ks_str = f"{stats['ks_mean']:.3f} $\\pm$ {stats['ks_std']:.3f}"
            acf_str = f"{stats['acf_r2_mean']:.3f} $\\pm$ {stats['acf_r2_std']:.3f}"
            nrmse_str = f"{stats['nrmse_mean']:.3f} $\\pm$ {stats['nrmse_std']:.3f}"
            p95_str = (
                f"{stats['p95_error_mean']:.2f} $\\pm$ {stats['p95_error_std']:.2f}"
            )
            p99_str = (
                f"{stats['p99_error_mean']:.2f} $\\pm$ {stats['p99_error_std']:.2f}"
            )
            energy_str = f"{stats['energy_error_mean']:.2f} $\\pm$ {stats['energy_error_std']:.2f}"

        line = f"{model_display[model]}{suffix} & {ks_str} & {acf_str} & {nrmse_str} & {p95_str} & {p99_str} & {energy_str} \\\\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    with open(output_file, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
