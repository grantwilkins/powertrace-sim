#!/usr/bin/env python3
"""
Test script to validate preset distributions across all (task, level) combinations.
Generates paper-quality plots in a 5x4 format showing input/output distributions.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_dataset import RealisticRandomDataset
from transformers import AutoTokenizer


# Preset configurations matching benchmark_serving.py
PRESETS = {
    ("conversation", "low"): {
        "in_mu": 5.30, "in_sigma": 0.85, "in_min": 16, "in_max": 2048,
        "tail_p": 0.08, "tail_alpha": 1.4, "tail_xmin": 2048,
        "out_mean": 120, "out_min": 8, "out_max": 1024,
        "implicit_burstiness": 0.87,
    },
    ("conversation", "medium"): {
        "in_mu": 6.22, "in_sigma": 1.08, "in_min": 32, "in_max": 4096,
        "tail_p": 0.08, "tail_alpha": 1.4, "tail_xmin": 4096,
        "out_mean": 250, "out_min": 16, "out_max": 2048,
        "implicit_burstiness": 0.76,
    },
    ("conversation", "high"): {
        "in_mu": 7.31, "in_sigma": 1.08, "in_min": 64, "in_max": 8192,
        "tail_p": 0.08, "tail_alpha": 1.4, "tail_xmin": 4096,
        "out_mean": 600, "out_min": 32, "out_max": 4096,
        "implicit_burstiness": 0.60,
    },
    ("conversation", "ultra"): {
        "in_mu": 8.01, "in_sigma": 1.08, "in_min": 128, "in_max": 16384,
        "tail_p": 0.08, "tail_alpha": 1.4, "tail_xmin": 8192,
        "out_mean": 1000, "out_min": 64, "out_max": 8192,
        "implicit_burstiness": 0.55,
    },
    ("coding", "low"): {
        "in_mu": 5.70, "in_sigma": 0.95, "in_min": 16, "in_max": 4096,
        "tail_p": 0.08, "tail_alpha": 1.4, "tail_xmin": 4096,
        "out_mean": 180, "out_min": 16, "out_max": 2048,
        "implicit_burstiness": 0.87,
    },
    ("coding", "medium"): {
        "in_mu": 6.55, "in_sigma": 1.10, "in_min": 32, "in_max": 8192,
        "tail_p": 0.08, "tail_alpha": 1.4, "tail_xmin": 4096,
        "out_mean": 350, "out_min": 32, "out_max": 4096,
        "implicit_burstiness": 0.76,
    },
    ("coding", "high"): {
        "in_mu": 7.60, "in_sigma": 1.10, "in_min": 64, "in_max": 16384,
        "tail_p": 0.08, "tail_alpha": 1.4, "tail_xmin": 8192,
        "out_mean": 800, "out_min": 64, "out_max": 8192,
        "implicit_burstiness": 0.60,
    },
    ("coding", "ultra"): {
        "in_mu": 8.30, "in_sigma": 1.10, "in_min": 128, "in_max": 24576,
        "tail_p": 0.08, "tail_alpha": 1.4, "tail_xmin": 8192,
        "out_mean": 1400, "out_min": 128, "out_max": 16384,
        "implicit_burstiness": 0.55,
    },
}


def test_preset(task, level, preset, tokenizer, num_requests=20000, seed=42):
    """Test a single preset and return statistics."""
    np.random.seed(seed)

    dataset = RealisticRandomDataset(random_seed=seed)
    requests = dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_requests,
        input_log_mean=preset["in_mu"],
        input_log_std=preset["in_sigma"],
        input_min=preset["in_min"],
        input_max=preset["in_max"],
        tail_probability=preset["tail_p"],
        tail_alpha=preset["tail_alpha"],
        tail_xmin=preset["tail_xmin"],
        use_exponential_output=True,
        exp_output_mean=preset["out_mean"],
        output_min=preset["out_min"],
        output_max=preset["out_max"],
    )

    input_lens = np.array([req.prompt_len for req in requests])
    output_lens = np.array([req.expected_output_len for req in requests])

    stats = {
        "task": task,
        "level": level,
        "input": {
            "min": int(input_lens.min()),
            "median": int(np.median(input_lens)),
            "mean": float(input_lens.mean()),
            "p90": int(np.percentile(input_lens, 90)),
            "max": int(input_lens.max()),
            "target_median": int(np.exp(preset["in_mu"])),
        },
        "output": {
            "min": int(output_lens.min()),
            "median": int(np.median(output_lens)),
            "mean": float(output_lens.mean()),
            "p90": int(np.percentile(output_lens, 90)),
            "max": int(output_lens.max()),
            "target_mean": preset["out_mean"],
        },
        "input_lens": input_lens.tolist(),
        "output_lens": output_lens.tolist(),
    }

    return stats


def create_paper_plots(all_stats, output_path="preset_distributions.png"):
    """Create 5x4 grid of plots showing all distributions."""
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    fig.suptitle("Workload Preset Distributions: Paper Validation", fontsize=16, fontweight='bold')

    tasks = ["conversation", "coding"]
    levels = ["low", "medium", "high", "ultra"]

    for row, task in enumerate(tasks):
        for col, level in enumerate(levels):
            ax = axes[row * 2, col]  # Input distribution
            stats = all_stats[(task, level)]

            # Plot input distribution histogram
            input_lens = np.array(stats["input_lens"])
            ax.hist(input_lens, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(stats["input"]["median"], color='red', linestyle='--', linewidth=2, label=f'Median: {stats["input"]["median"]}')
            ax.axvline(stats["input"]["p90"], color='orange', linestyle='--', linewidth=2, label=f'P90: {stats["input"]["p90"]}')

            ax.set_xlabel('Input Tokens', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{task.capitalize()} {level.upper()}: Inputs\nμ={stats["input"]["in_mu"]:.2f}, σ={stats["input"]["in_sigma"]:.2f}',
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Plot output distribution histogram
            ax = axes[row * 2 + 1, col]
            output_lens = np.array(stats["output_lens"])
            ax.hist(output_lens, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax.axvline(stats["output"]["median"], color='red', linestyle='--', linewidth=2, label=f'Median: {stats["output"]["median"]}')
            ax.axvline(stats["output"]["mean"], color='purple', linestyle='--', linewidth=2, label=f'Mean: {stats["output"]["mean"]:.0f}')

            ax.set_xlabel('Output Tokens', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{task.capitalize()} {level.upper()}: Outputs (Exponential)\nTarget Mean: {stats["output"]["target_mean"]}',
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    # Add summary statistics box
    summary_text = "Distribution Summary:\n"
    summary_text += "• Inputs: Log-normal body + 8% Pareto tail\n"
    summary_text += "• Outputs: Exponential (memoryless)\n"
    summary_text += "• Validation: 20K samples per preset\n"
    summary_text += "• Target: Mean within ±10% of preset"

    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_path}")

    return fig


def print_validation_report(all_stats):
    """Print comprehensive validation report."""
    print("\n" + "="*80)
    print("PRESET VALIDATION REPORT")
    print("="*80)

    tasks = ["conversation", "coding"]
    levels = ["low", "medium", "high", "ultra"]

    for task in tasks:
        print(f"\n{task.upper()} TASK")
        print("-"*80)
        for level in levels:
            stats = all_stats[(task, level)]

            # Calculate deviations
            output_mean_dev = 100 * (stats["output"]["mean"] - stats["output"]["target_mean"]) / stats["output"]["target_mean"]

            print(f"\n  {level.upper()}:")
            print(f"    Input:  median={stats['input']['median']:5d} (target≈{stats['input']['target_median']:5d}), "
                  f"P90={stats['input']['p90']:5d}, range=[{stats['input']['min']}, {stats['input']['max']}]")
            print(f"    Output: mean={stats['output']['mean']:6.1f} (target={stats['output']['target_mean']:4d}, "
                  f"dev={output_mean_dev:+.1f}%), median={stats['output']['median']:5d}")

            # Validation checks
            checks = []
            if abs(output_mean_dev) <= 10:
                checks.append("✓ Output mean within ±10%")
            else:
                checks.append("✗ Output mean deviation > 10%")

            if stats["input"]["p90"] >= 3 * stats["input"]["median"]:
                checks.append("✓ P90/median ratio ≥ 3")
            else:
                checks.append("✗ P90/median ratio < 3")

            print(f"    Validation: {' | '.join(checks)}")


def main():
    """Run validation across all presets."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("\nTesting all 8 (task, level) combinations with 20K samples each...")

    all_stats = {}
    tasks = ["conversation", "coding"]
    levels = ["low", "medium", "high", "ultra"]

    for task in tasks:
        for level in levels:
            print(f"\n  Testing: {task} / {level}...")
            preset_key = (task, level)
            preset = PRESETS[preset_key]

            # Add preset params to stats for plotting
            stats = test_preset(task, level, preset, tokenizer)
            stats["input"]["in_mu"] = preset["in_mu"]
            stats["input"]["in_sigma"] = preset["in_sigma"]

            all_stats[preset_key] = stats

    # Print validation report
    print_validation_report(all_stats)

    # Create plots
    print("\nGenerating paper-quality plots...")
    create_paper_plots(all_stats)

    # Save detailed statistics to JSON
    output_json = "preset_validation_stats.json"
    with open(output_json, "w") as f:
        # Convert tuples to strings for JSON serialization
        json_stats = {f"{k[0]}_{k[1]}": v for k, v in all_stats.items()}
        json.dump(json_stats, f, indent=2)
    print(f"✓ Saved detailed statistics to {output_json}")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
