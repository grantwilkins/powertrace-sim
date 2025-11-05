#!/usr/bin/env python3
"""
Plot training losses for all models, aggregated by model type.
Creates a publication-ready figure similar to the reference plot.
"""

import argparse
import csv
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_filename(filename):
    """Parse model, hardware, and TP from losses CSV filename."""
    basename = os.path.basename(filename).replace("_losses.csv", "")

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


def load_losses_from_csv(csv_file):
    """Load training losses from CSV file."""
    epochs = []
    losses = []

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["train_loss"]))

    return np.array(epochs), np.array(losses)


def main():
    parser = argparse.ArgumentParser(description="Plot training losses for all models")
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="./training_results/metrics",
        help="Directory containing loss CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./training_results/training_loss_curves.pdf",
        help="Output PDF file",
    )
    parser.add_argument("--title", type=str, default=None, help="Plot title (optional)")
    args = parser.parse_args()

    # Find all loss files
    loss_files = glob.glob(os.path.join(args.metrics_dir, "*_losses.csv"))
    print(f"Found {len(loss_files)} loss files")

    if len(loss_files) == 0:
        print("Error: No loss files found!")
        return

    # Group losses by model
    model_losses = {
        "llama-3-8b": [],
        "llama-3-70b": [],
        "deepseek-r1-distill-8b": [],
        "deepseek-r1-distill-70b": [],
    }

    # Load all losses
    for loss_file in sorted(loss_files):
        model, hardware, tp = parse_filename(loss_file)

        if model is None:
            print(f"Warning: Could not parse {os.path.basename(loss_file)}")
            continue

        epochs, losses = load_losses_from_csv(loss_file)

        if len(epochs) > 0:
            model_losses[model].append(
                {
                    "epochs": epochs,
                    "losses": losses,
                    "hardware": hardware,
                    "tp": tp,
                }
            )
            print(f"Loaded {model}_{hardware}_tp{tp}: {len(epochs)} epochs")

    # Set up plotting style
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.5)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "Times New Roman"

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 5))

    # Color palette
    colors = {
        "llama-3-8b": "#1f77b4",  # Blue
        "llama-3-70b": "#ff7f0e",  # Orange
        "deepseek-r1-distill-8b": "#2ca02c",  # Green
        "deepseek-r1-distill-70b": "#d62728",  # Red
    }

    # Display names
    display_names = {
        "llama-3-8b": "Llama-3.1 (8B)",
        "llama-3-70b": "Llama-3.1 (70B)",
        "deepseek-r1-distill-8b": "DeepSeek-Distill (8B)",
        "deepseek-r1-distill-70b": "DeepSeek-Distill (70B)",
    }

    # Plot order for legend
    plot_order = [
        "llama-3-8b",
        "llama-3-70b",
        "deepseek-r1-distill-8b",
        "deepseek-r1-distill-70b",
    ]

    # Plot each model
    for model in plot_order:
        if model not in model_losses or len(model_losses[model]) == 0:
            continue

        configs = model_losses[model]

        # Get max epoch count to align arrays
        max_epochs = max(len(c["epochs"]) for c in configs)

        # Collect all loss curves for this model
        all_losses = []
        for config in configs:
            epochs = config["epochs"]
            losses = config["losses"]

            # Pad if needed (in case some runs ended early)
            if len(losses) < max_epochs:
                # Pad with the last value
                padded_losses = np.pad(
                    losses, (0, max_epochs - len(losses)), mode="edge"
                )
                all_losses.append(padded_losses)
            else:
                all_losses.append(losses)

        all_losses = np.array(all_losses)

        # Compute mean and std
        mean_loss = np.mean(all_losses, axis=0)
        std_loss = np.std(all_losses, axis=0)

        # Use epochs from first config (they should all be the same)
        epochs = np.arange(1, max_epochs + 1)

        # Plot mean line
        ax.plot(
            epochs,
            mean_loss,
            color=colors[model],
            linewidth=2.5,
            label=display_names[model],
            zorder=3,
        )

        # Plot std as shaded region
        ax.fill_between(
            epochs,
            mean_loss - std_loss,
            mean_loss + std_loss,
            color=colors[model],
            alpha=0.2,
            zorder=2,
        )

    # Formatting
    ax.set_xlabel("Epoch", fontsize=20)
    ax.set_ylabel("Training Loss", fontsize=20)
    if args.title:
        ax.set_title(args.title, fontsize=22)

    ax.set_xlim(0, max_epochs)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=False, fontsize=16)

    # Set tick label sizes
    ax.tick_params(axis="both", which="major", labelsize=16)

    plt.tight_layout()

    # Save figure
    plt.savefig(args.output, bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to: {args.output}")

    # Also save as PNG for quick preview
    png_output = args.output.replace(".pdf", ".png")
    plt.savefig(png_output, bbox_inches="tight", dpi=300)
    print(f"PNG version saved to: {png_output}")

    plt.close()

    # Print summary statistics
    print("\nFinal Loss Statistics (mean ± std):")
    for model in plot_order:
        if model not in model_losses or len(model_losses[model]) == 0:
            continue

        configs = model_losses[model]
        final_losses = [c["losses"][-1] for c in configs]
        mean_final = np.mean(final_losses)
        std_final = np.std(final_losses)

        print(
            f"  {display_names[model]}: {mean_final:.4f} ± {std_final:.4f} ({len(configs)} configs)"
        )


if __name__ == "__main__":
    main()
