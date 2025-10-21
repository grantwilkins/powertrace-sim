#!/usr/bin/env python3
"""
Add power state statistics to performance_database.json.
Extracts stats from model_summary JSON files and merges them into the unified database.
"""

import json
import os
import sys
from glob import glob

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


def add_power_stats_to_performance_db(
    summary_data_dir: str = "model/summary_data",
    performance_db_path: str = "model/performance_database.json",
    output_path: str = "model/performance_database.json",
):
    """
    Add power state statistics to performance database.

    Args:
        summary_data_dir: Directory with model_summary_*.json files
        performance_db_path: Path to existing performance database
        output_path: Where to save updated database
    """
    # Load existing performance database
    if os.path.exists(performance_db_path):
        with open(performance_db_path, "r") as f:
            perf_db = json.load(f)
    else:
        perf_db = {}

    # Process all summary files
    pattern = os.path.join(summary_data_dir, "model_summary_*.json")
    summary_files = glob(pattern)

    stats_added = 0
    entries_updated = 0

    for summary_file in sorted(summary_files):
        # Extract model and hardware from filename
        # Format: model_summary_{model_name}_{hardware}.json
        basename = os.path.basename(summary_file)
        parts = (
            basename.replace("model_summary_", "").replace(".json", "").rsplit("_", 1)
        )

        if len(parts) != 2:
            print(f"Skipping {basename} - unexpected format")
            continue

        model_name, hardware = parts

        # Load summary data
        with open(summary_file, "r") as f:
            summary_data = json.load(f)

        # Add power stats for each TP configuration
        for tp_key, tp_data in summary_data.items():
            tp = int(tp_key)

            # Construct database key matching performance_database format
            # Map to standard naming convention
            model_map = {
                "llama-3-8b": "llama-3.1_8b",
                "llama-3-70b": "llama-3.1_70b",
                "llama-3-405b": "llama-3.1_405b",
                "deepseek-r1-8b": "deepseek-r1-distill_8b",
                "deepseek-r1-70b": "deepseek-r1-distill_70b",
                "deepseek-r1-distill-8b": "deepseek-r1-distill_8b",
                "deepseek-r1-distill-70b": "deepseek-r1-distill_70b",
            }

            db_model_name = model_map.get(model_name, model_name)
            db_key = f"{db_model_name}_{hardware}_tp{tp}"

            # Create entry if it doesn't exist
            if db_key not in perf_db:
                perf_db[db_key] = {
                    "model_name": db_model_name.rsplit("_", 1)[0],
                    "model_size_b": int(db_model_name.split("_")[-1].replace("b", "")),
                    "hardware": hardware.upper(),
                    "tensor_parallelism": tp,
                }

            # Add power state statistics
            perf_db[db_key]["power_states"] = {
                "num_states": 6,
                "state_means": tp_data["mu_values"],
                "state_stds": tp_data["sigma_values"],
                "clustering_method": "gaussian_mixture",
            }

            stats_added += 1
            if "ttft_model" in perf_db[db_key]:
                entries_updated += 1

            print(f"Added power stats to {db_key}")

    # Save updated database
    with open(output_path, "w") as f:
        json.dump(perf_db, f, indent=2)

    print(f"\n✓ Updated {output_path}")
    print(f"  - {stats_added} power stat entries added")
    print(f"  - {entries_updated} existing entries updated")
    print(f"  - {stats_added - entries_updated} new entries created")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add power statistics to performance database"
    )
    parser.add_argument(
        "--summary-dir",
        type=str,
        default="model/summary_data",
        help="Directory with model_summary_*.json files",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="model/performance_database.json",
        help="Path to performance database",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model/performance_database.json",
        help="Output path (defaults to updating in place)",
    )
    args = parser.parse_args()

    add_power_stats_to_performance_db(
        summary_data_dir=args.summary_dir,
        performance_db_path=args.db_path,
        output_path=args.output,
    )
