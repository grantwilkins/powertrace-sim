#!/usr/bin/env python3
"""
Script to train all model configurations and compute comprehensive evaluation metrics.
Generates CSV files with training stats for paper table generation.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

# Configuration mapping: model -> (data_file_pattern, tp_values)
CONFIGS = {

    "llama-3-70b": {
        "tp_values": [4, 8],
        "hardware": ["a100", "h100"],
        "data_pattern": "vllm-benchmark_llama-3-70b_{hw}.npz",
    },
    "deepseek-r1-distill-70b": {
        "tp_values": [4, 8],
        "hardware": ["a100", "h100"],
        "data_pattern": "vllm-benchmark_deepseek-r1-70b_{hw}.npz",
    },
}


def generate_wandb_job_name(model, hardware, tp):
    """Generate a logical wandb job name."""
    return f"{model}_{hardware}_tp{tp}"


def run_training_job(model, hardware, tp, data_dir, lr, num_epochs, dry_run=False):
    """Run a single training job."""
    config = CONFIGS[model]
    data_file = os.path.join(data_dir, config["data_pattern"].format(hw=hardware))

    if not os.path.exists(data_file):
        print(f"WARNING: Data file not found: {data_file}")
        return False

    job_name = generate_wandb_job_name(model, hardware, tp)

    cmd = [
        "python3",
        "-m",
        "model.train_entry",
        "--data_file",
        data_file,
        "--model",
        model,
        "--tp",
        str(tp),
        "--hardware_accelerator",
        hardware,
        "--lr",
        str(lr),
        "--num_epochs",
        str(num_epochs),
        "--weights_path",
        "./model/new_weights/",
    ]

    print(f"\n{'=' * 80}")
    print(f"Training: {job_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    if dry_run:
        print("DRY RUN - would execute above command")
        return True

    # Set WANDB_NAME environment variable
    env = os.environ.copy()
    env["WANDB_NAME"] = job_name

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"ERROR: Training failed for {job_name}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train all model configurations and generate evaluation metrics"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./model/training_data",
        help="Directory containing training data files",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=500, help="Number of training epochs"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(CONFIGS.keys()),
        default=list(CONFIGS.keys()),
        help="Models to train (default: all)",
    )
    parser.add_argument(
        "--hardware",
        nargs="+",
        choices=["a100", "h100"],
        default=["a100", "h100"],
        help="Hardware accelerators to train on (default: all)",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print commands without executing"
    )
    args = parser.parse_args()

    # Create output directory for results
    os.makedirs("./training_results", exist_ok=True)

    # Track all training jobs
    jobs_info = []

    # Run all training jobs
    for model in args.models:
        config = CONFIGS[model]
        for hardware in args.hardware:
            if hardware not in config["hardware"]:
                continue
            for tp in config["tp_values"]:
                job_info = {
                    "model": model,
                    "hardware": hardware,
                    "tp": tp,
                    "job_name": generate_wandb_job_name(model, hardware, tp),
                }
                jobs_info.append(job_info)

                success = run_training_job(
                    model,
                    hardware,
                    tp,
                    args.data_dir,
                    args.lr,
                    args.num_epochs,
                    args.dry_run,
                )
                job_info["success"] = success

    # Save job summary
    summary_file = "./training_results/jobs_summary.json"
    with open(summary_file, "w") as f:
        json.dump(jobs_info, f, indent=2)

    print(f"\n{'=' * 80}")
    print("Training Summary")
    print(f"{'=' * 80}")
    print(f"Total jobs: {len(jobs_info)}")
    print(f"Successful: {sum(1 for j in jobs_info if j.get('success', False))}")
    print(f"Failed: {sum(1 for j in jobs_info if not j.get('success', False))}")
    print(f"\nJob summary saved to: {summary_file}")
    print(f"\nNext steps:")
    print(f"  1. Run evaluation script to compute metrics")
    print(f"  2. Generate LaTeX table from CSV outputs")


if __name__ == "__main__":
    main()
