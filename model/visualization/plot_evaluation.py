"""
Visualization scripts for evaluation results.

Generates publication-ready figures:
- Figure 1: Power trace overlays
- Figure 2: Power CDFs by tensor parallelism
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_power_trace_overlay(
    time: np.ndarray,
    power_gt: np.ndarray,
    power_pred: np.ndarray,
    transition_indices: List[int],
    title: str = "",
    output_path: str = None,
    zoom_window_ms: float = 300.0,
    figsize: tuple = (10, 4),
):
    fig, ax1 = plt.figure(figsize=figsize), plt.gca()
    ax1.plot(time, power_gt, label="Ground Truth", linewidth=1.5, alpha=0.8)
    ax1.plot(time, power_pred, label="Model", linewidth=1.5, alpha=0.8, linestyle="--")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("GPU Power (W)")
    ax1.set_title(f"{title} - Full Trace")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_power_cdf_by_tp(
    power_data_by_tp: Dict[int, Tuple[List[float], List[float]]],
    model_hardware: str = "",
    output_path: str = None,
    figsize: tuple = (10, 6),
):
    """
    Plot CDF of power values (GT vs Predicted) split by tensor parallelism.

    Simple, publication-quality single plot.

    Args:
        power_data_by_tp: Dict mapping TP -> (gt_power_list, pred_power_list)
        model_hardware: Model and hardware string for title
        output_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    tp_values = sorted(power_data_by_tp.keys())

    for i, tp in enumerate(tp_values):
        gt_power, pred_power = power_data_by_tp[tp]

        # Plot GT
        if len(gt_power) > 0:
            power_sorted = np.sort(gt_power)
            cdf = np.arange(1, len(power_sorted) + 1) / len(power_sorted)
            mean_power = np.mean(gt_power)

            ax.plot(
                power_sorted, cdf,
                linewidth=2.5,
                color=colors[i % len(colors)],
                linestyle='-',
                label=f"TP={tp} GT (μ={mean_power:.0f}W)"
            )

        # Plot Predicted
        if len(pred_power) > 0:
            power_sorted = np.sort(pred_power)
            cdf = np.arange(1, len(power_sorted) + 1) / len(power_sorted)
            mean_power = np.mean(pred_power)

            ax.plot(
                power_sorted, cdf,
                linewidth=2.5,
                color=colors[i % len(colors)],
                linestyle='--',
                label=f"TP={tp} Pred (μ={mean_power:.0f}W)"
            )

    ax.set_xlabel("GPU Power (W)", fontsize=13)
    ax.set_ylabel("CDF", fontsize=13)

    if model_hardware:
        ax.set_title(f"{model_hardware}", fontsize=14, fontweight='bold')

    ax.legend(loc='best', fontsize=10, framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()

    plt.close()


def generate_all_visualizations(
    evaluation_dir: str, output_dir: str, configs: List[str] = None
):
    os.makedirs(output_dir, exist_ok=True)

    trace_examples_dir = os.path.join(evaluation_dir, "trace_examples")

    # Check if evaluation directory exists
    if not os.path.exists(trace_examples_dir):
        print(f"Error: Trace examples directory not found: {trace_examples_dir}")
        print(f"Please run evaluation first using evaluate.py")
        return

    # Find all configurations
    if configs is None:
        configs = [
            d
            for d in os.listdir(trace_examples_dir)
            if os.path.isdir(os.path.join(trace_examples_dir, d))
        ]

    if len(configs) == 0:
        print(f"No configurations found in {trace_examples_dir}")
        return

    # Group configs by model+hardware for power CDF plots
    power_data_by_model_hw = defaultdict(lambda: defaultdict(list))

    for config in configs:
        print(f"\n{'=' * 60}")
        print(f"Generating visualizations for {config}")
        print(f"{'=' * 60}")

        config_trace_dir = os.path.join(trace_examples_dir, config)
        config_output_dir = os.path.join(output_dir, config)
        os.makedirs(config_output_dir, exist_ok=True)

        # Parse config name: model_hardware_tpX
        parts = config.split("_tp")
        if len(parts) == 2:
            model_hw = parts[0]
            tp = int(parts[1])
        else:
            model_hw = config
            tp = 1

        # Plot power traces
        trace_files = sorted(
            [f for f in os.listdir(config_trace_dir) if f.endswith(".json")]
        )

        for trace_file in trace_files[:3]:  # Plot first 3 examples
            trace_path = os.path.join(config_trace_dir, trace_file)
            with open(trace_path, "r") as f:
                data = json.load(f)

            example_num = trace_file.replace("example_", "").replace(".json", "")
            output_path = os.path.join(
                config_output_dir, f"power_trace_overlay_{example_num}.png"
            )

            plot_power_trace_overlay(
                time=np.array(data["time"]),
                power_gt=np.array(data["power_gt"]),
                power_pred=np.array(data["power_pred"]),
                transition_indices=data["transition_indices"],
                title=f"{config} - Example {example_num}",
                output_path=output_path,
            )

        # Collect all power values for CDF (both GT and predicted)
        all_gt_power = []
        all_pred_power = []
        for trace_file in trace_files:
            trace_path = os.path.join(config_trace_dir, trace_file)
            with open(trace_path, "r") as f:
                data = json.load(f)
            all_gt_power.extend(data["power_gt"])
            all_pred_power.extend(data["power_pred"])

        # Store power data grouped by model+hardware and TP
        power_data_by_model_hw[model_hw][tp] = (all_gt_power, all_pred_power)

    # Generate power CDFs by TP for each model+hardware combination
    print(f"\n{'=' * 60}")
    print("Generating power CDFs by tensor parallelism")
    print(f"{'=' * 60}")

    for model_hw, tp_data in power_data_by_model_hw.items():
        output_path = os.path.join(output_dir, f"{model_hw}_power_cdf_by_tp.png")

        # Format model_hw for display
        display_name = model_hw.replace("_", " ").replace("-", "-").title()

        plot_power_cdf_by_tp(
            power_data_by_tp=tp_data,
            model_hardware=display_name,
            output_path=output_path,
        )

    print(f"\n{'=' * 60}")
    print(f"Visualization complete! Figures saved to {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation visualizations")
    parser.add_argument(
        "--evaluation-dir",
        type=str,
        default="./evaluation_results/",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./figures/",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        help="Configurations to visualize (e.g., llama-3-8b_a100_tp1)",
    )

    args = parser.parse_args()

    generate_all_visualizations(
        evaluation_dir=args.evaluation_dir,
        output_dir=args.output_dir,
        configs=args.configs,
    )
