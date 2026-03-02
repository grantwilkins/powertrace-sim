#!/usr/bin/env python3
"""
Generate LaTeX table data for trace fidelity evaluation results.

This script processes raw per-seed evaluation metrics and generates formatted
LaTeX table rows for inclusion in academic papers.

Features:
    - Loads raw per-seed metrics from CSV files
    - Aggregates across seeds, traces, and hardware/TP configurations
    - Supports multiple generation modes (i.i.d., AR(1), AR(1) thresholded)
    - Custom model name mappings and architecture overrides
    - Multiple output formats (LaTeX, CSV)
    - Proper handling of missing standard deviations

Usage:
    # Basic usage (prints to stdout)
    python scripts/eval/generate_trace_fidelity_table.py

    # Save to file
    python scripts/eval/generate_trace_fidelity_table.py --output table.tex

    # CSV format
    python scripts/eval/generate_trace_fidelity_table.py --format csv

Dependencies:
    - pandas
    - numpy

Input:
    results/continuous_v1_gmm_bigru/k10_f2*/eval_metrics/per_seed_metrics.csv

Output:
    LaTeX table rows or CSV data (stdout or file)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow running via: python3 scripts/eval/*.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ============================================================================
# Configuration
# ============================================================================

# Model display name mapping
MODEL_NAME_MAP = {
    "llama-3": "Llama-3.1",
    "deepseek-r1-distill": "DeepSeek-R1-Distill",
    "gpt-oss": "gpt-oss",
}

# Architecture overrides: (model_family, model_size) -> arch_type
# Keep empty by default; infer_arch_type encodes current paper assumptions.
ARCH_OVERRIDE = {}

# Expected generation mode per architecture
EXPECTED_GEN_MODE = {
    "dense": "iid",
    "moe": "ar1",
}

# Hardware/TP configurations to exclude from aggregation
# These configs have systematic issues with certain models
EXCLUDED_HARDWARE_TP = [
    ("A100", 8),  # 100% negative ACF R² for llama-3-70b
    ("A100", 4),  # 67% negative ACF R² for llama-3-70b
    ("A100", 2),  # 100% negative ACF R² for llama-3-8b
    ("H100", 2),  # 87% negative ACF R² for llama-3-8b
]

# Default input directories
# These can be overridden via CLI or by pointing to kauto_max20_f2 runs
DEFAULT_INPUT_DIRS = {
    "iid": "results/continuous_v1_gmm_bigru/k10_f2/eval_metrics",
    "ar1": "results/continuous_v1_gmm_bigru/k10_f2_ar1/eval_metrics",
    "ar1_thresh": "results/continuous_v1_gmm_bigru/k10_f2_ar1_thresh/eval_metrics",
}

# Alternative auto-K directories (use these when running with --auto-k results)
AUTOK_INPUT_DIRS = {
    "iid": "results/continuous_v1_gmm_bigru/kauto_max20_f2/eval_metrics",
    "ar1": "results/continuous_v1_gmm_bigru/kauto_max20_f2_ar1/eval_metrics",
    "ar1_thresh": "results/continuous_v1_gmm_bigru/kauto_max20_f2_ar1_thresh/eval_metrics",
}

# Default inventory used to determine whether pair_key has request timestamps.
DEFAULT_TIMESTAMP_INVENTORY = "results/stage0/data_inventory.json"

# ============================================================================
# Model Parsing Functions (adapted from collect_results.py)
# ============================================================================

def parse_config_id(config_id: str) -> Dict[str, str]:
    """
    Parse config_id string into components.

    Args:
        config_id: Format "{model}-{size}b_{hardware}_tp{digits}"
                   Examples:
                   - "deepseek-r1-distill-70b_H100_tp4"
                   - "llama-3-8b_A100_tp1"
                   - "gpt-oss-120b_H100_tp8"

    Returns:
        Dict with keys:
            - model_family: Model name without size
            - model_size: Parameter count in billions as string
            - hardware: GPU type ("A100" or "H100")
            - tp: Tensor parallelism factor as string

    Raises:
        ValueError: If config_id format is invalid

    Example:
        >>> parse_config_id("deepseek-r1-distill-70b_H100_tp4")
        {'model_family': 'deepseek-r1-distill', 'model_size': '70',
         'hardware': 'H100', 'tp': '4'}
    """
    pattern = r"^(.+)-(\d+)b_(A100|H100)_tp(\d+)$"
    match = re.match(pattern, config_id)

    if not match:
        raise ValueError(
            f"Invalid config_id format: '{config_id}'. "
            f"Expected format: '{{model}}-{{size}}b_{{hardware}}_tp{{digits}}'"
        )

    model_family, model_size, hardware, tp = match.groups()

    return {
        "model_family": model_family,
        "model_size": model_size,
        "hardware": hardware,
        "tp": tp,
    }


def infer_arch_type(model_family: str, model_size: int) -> str:
    """
    Classify model architecture type based on family and size.

    Args:
        model_family: Model name
        model_size: Model parameter count in billions

    Returns:
        "dense" for standard transformers
        "moe" for mixture-of-experts models

    Classification rules:
        - DeepSeek-R1-Distill models are treated as dense
        - "gpt-oss" with size >= 20B → "moe"
        - All others → "dense"
    """
    model_family_lower = model_family.lower()

    if "deepseek-r1-distill" in model_family_lower:
        return "dense"

    if "gpt-oss" in model_family_lower and model_size >= 20:
        return "moe"

    return "dense"


# ============================================================================
# Data Loading & Aggregation
# ============================================================================


def load_k_values_from_manifest(eval_metrics_dir: str) -> Dict[str, int]:
    """
    Load K values for each config from the run_manifest.json.

    Args:
        eval_metrics_dir: Path to eval_metrics directory
            (e.g., "results/continuous_v1_gmm_bigru/kauto_max20_f2/eval_metrics")

    Returns:
        Dict mapping config_id -> K value
    """
    # Navigate up one level to find run_manifest.json
    parent_dir = os.path.dirname(eval_metrics_dir.rstrip("/"))
    manifest_path = os.path.join(parent_dir, "run_manifest.json")

    if not os.path.exists(manifest_path):
        return {}

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        k_values = {}
        configs = manifest.get("configs", {})
        for config_id, config_data in configs.items():
            if isinstance(config_data, dict):
                k_val = config_data.get("k")
                if k_val is not None:
                    k_values[config_id] = int(k_val)

        return k_values
    except Exception:
        return {}


def load_per_seed_csv(
    csv_path: str, generation_mode: str, require_pair_key: bool = False
) -> pd.DataFrame:
    """
    Load per-seed metrics CSV and add generation_mode column.

    Args:
        csv_path: Path to per_seed_metrics.csv file
        generation_mode: One of "iid", "ar1", "ar1_thresh"

    Returns:
        DataFrame with columns: config_id, trace_idx, seed,
                               ks_stat, acf_r2, nrmse, delta_energy_pct,
                               generation_mode, status

    Raises:
        FileNotFoundError: If csv_path doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = {
        "config_id",
        "trace_idx",
        "seed",
        "status",
        "ks_stat",
        "acf_r2",
        "nrmse",
        "delta_energy_pct",
    }
    if require_pair_key:
        required_cols.add("pair_key")
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"CSV missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    # Add generation_mode column
    df["generation_mode"] = generation_mode

    # Filter to successfully evaluated rows only (status can be "ok" or "evaluated")
    df = df[df["status"].isin(["ok", "evaluated"])].copy()

    return df


def load_request_timestamp_availability_map(
    inventory_json_path: str,
) -> Dict[str, bool]:
    """
    Build mapping: pair_key -> has_request_timestamps.

    Uses results/stage0/data_inventory.json paired_runs[*].request_extraction_stats
    and treats pair_key as having request timestamps iff
    missing_request_timestamps_array == False.
    """
    if not os.path.exists(inventory_json_path):
        raise FileNotFoundError(
            f"Timestamp inventory JSON not found: {inventory_json_path}"
        )

    with open(inventory_json_path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {inventory_json_path}")

    paired_runs = payload.get("paired_runs")
    if not isinstance(paired_runs, list):
        raise ValueError(
            f"Invalid inventory format (missing list 'paired_runs'): {inventory_json_path}"
        )

    out: Dict[str, bool] = {}
    for row in paired_runs:
        if not isinstance(row, dict):
            continue
        pair_key = str(row.get("pair_key", "")).strip()
        if pair_key == "":
            continue

        req_stats = row.get("request_extraction_stats", {})
        if not isinstance(req_stats, dict):
            req_stats = {}
        missing_ts = bool(req_stats.get("missing_request_timestamps_array", False))
        has_ts = not missing_ts

        if pair_key in out:
            # Conservative union: if any source indicates timestamps exist, keep True.
            out[pair_key] = bool(out[pair_key] or has_ts)
        else:
            out[pair_key] = bool(has_ts)

    if len(out) == 0:
        raise ValueError(
            f"No pair_key entries found in timestamp inventory: {inventory_json_path}"
        )
    return out


def filter_per_seed_rows_by_request_timestamps(
    df: pd.DataFrame, pair_key_has_timestamps: Dict[str, bool]
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Keep only rows whose pair_key is known and has request timestamps.
    """
    if "pair_key" not in df.columns:
        raise ValueError(
            "Cannot filter by request timestamp availability: missing 'pair_key' column."
        )

    pair_key_series = df["pair_key"].astype(str).str.strip()
    mapped = pair_key_series.map(pair_key_has_timestamps)
    known_mask = mapped.notna()
    has_ts_mask = known_mask & mapped.astype(bool)
    filtered = df.loc[has_ts_mask].copy()

    stats = {
        "rows_before": int(len(df)),
        "rows_after": int(len(filtered)),
        "rows_dropped_no_timestamps": int((known_mask & (~mapped.astype(bool))).sum()),
        "rows_dropped_unmapped_pair_key": int((~known_mask).sum()),
        "unique_pair_keys_before": int(pair_key_series.nunique()),
        "unique_pair_keys_after": int(filtered["pair_key"].astype(str).str.strip().nunique()),
    }
    return filtered, stats


def filter_excluded_hardware_tp(
    df: pd.DataFrame, excluded: List[Tuple[str, int]]
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Filter out configs with excluded hardware/TP combinations.

    Args:
        df: DataFrame with per-config metrics (must have config_id column)
        excluded: List of (hardware, tp) tuples to exclude

    Returns:
        Tuple of (filtered DataFrame, stats dict)
    """
    if not excluded:
        return df, {"rows_before": len(df), "rows_after": len(df), "rows_excluded": 0}

    rows_before = len(df)
    excluded_set = set(excluded)

    # Parse config_ids and check against exclusion list
    def should_exclude(config_id: str) -> bool:
        try:
            parsed = parse_config_id(config_id)
            hw_tp = (parsed["hardware"], int(parsed["tp"]))
            return hw_tp in excluded_set
        except ValueError:
            return False

    mask = ~df["config_id"].apply(should_exclude)
    filtered = df[mask].copy()

    stats = {
        "rows_before": rows_before,
        "rows_after": len(filtered),
        "rows_excluded": rows_before - len(filtered),
        "excluded_configs": list(df[~mask]["config_id"].unique()) if (~mask).any() else [],
    }
    return filtered, stats


def aggregate_seed_to_trace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-seed metrics to per-trace (median across seeds).

    Args:
        df: DataFrame with per-seed metrics

    Returns:
        DataFrame with one row per (config_id, trace_idx, generation_mode)
        Columns: config_id, trace_idx, generation_mode,
                ks_stat_median, acf_r2_median, nrmse_median,
                delta_energy_pct_median, num_seeds
    """
    grouped = df.groupby(["config_id", "trace_idx", "generation_mode"])

    agg_dict = {
        "ks_stat": "median",
        "acf_r2": "median",
        "nrmse": "median",
        "delta_energy_pct": "median",
        "seed": "count",  # Count number of seeds
    }

    result = grouped.agg(agg_dict).reset_index()

    # Rename columns
    result = result.rename(
        columns={
            "ks_stat": "ks_stat_median",
            "acf_r2": "acf_r2_median",
            "nrmse": "nrmse_median",
            "delta_energy_pct": "delta_energy_pct_median",
            "seed": "num_seeds",
        }
    )

    return result


def aggregate_trace_to_config(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-trace metrics to per-config (median across traces).

    Args:
        df: DataFrame with per-trace metrics

    Returns:
        DataFrame with one row per (config_id, generation_mode)
        Columns: config_id, generation_mode, ks_stat, acf_r2, nrmse,
                delta_energy_pct, num_traces
    """
    grouped = df.groupby(["config_id", "generation_mode"])

    agg_dict = {
        "ks_stat_median": "median",
        "acf_r2_median": "median",
        "nrmse_median": "median",
        "delta_energy_pct_median": "median",
        "trace_idx": "count",  # Count number of traces
    }

    result = grouped.agg(agg_dict).reset_index()

    # Rename columns (remove _median suffix for config level)
    result = result.rename(
        columns={
            "ks_stat_median": "ks_stat",
            "acf_r2_median": "acf_r2",
            "nrmse_median": "nrmse",
            "delta_energy_pct_median": "delta_energy_pct",
            "trace_idx": "num_traces",
        }
    )

    return result


def aggregate_config_to_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-config metrics to per-model (mean±std across hw/tp).

    Args:
        df: DataFrame with per-config metrics

    Returns:
        DataFrame with columns:
            model, model_size, arch_type, generation_mode, n_configs,
            KS_mean, KS_std, ACF_R2_mean, ACF_R2_std,
            NRMSE_mean, NRMSE_std, energy_err_mean, energy_err_std
    """
    # Parse config_ids
    parsed_rows = []
    skipped_count = 0

    for _, row in df.iterrows():
        try:
            parsed = parse_config_id(row["config_id"])
            parsed_rows.append(
                {
                    **row.to_dict(),
                    "model_family": parsed["model_family"],
                    "model_size": int(parsed["model_size"]),
                    "hardware": parsed["hardware"],
                    "tp": int(parsed["tp"]),
                }
            )
        except ValueError as e:
            print(f"Warning: Skipping config_id '{row['config_id']}': {e}")
            skipped_count += 1

    if skipped_count > 0:
        print(f"Skipped {skipped_count} rows due to parsing errors")

    if not parsed_rows:
        raise ValueError("No valid config_ids found in input data")

    df_parsed = pd.DataFrame(parsed_rows)

    # Infer architecture type
    df_parsed["arch_type"] = df_parsed.apply(
        lambda row: infer_arch_type(row["model_family"], row["model_size"]), axis=1
    )

    # Apply architecture overrides
    for (model_family, model_size), arch_type in ARCH_OVERRIDE.items():
        mask = (df_parsed["model_family"] == model_family) & (
            df_parsed["model_size"] == model_size
        )
        df_parsed.loc[mask, "arch_type"] = arch_type

    # Group by (model_family, model_size, generation_mode)
    grouped = df_parsed.groupby(["model_family", "model_size", "generation_mode", "arch_type"])

    aggregated = []
    for (model_family, model_size, generation_mode, arch_type), group in grouped:
        # Count configs
        n_configs = len(group)

        # Compute statistics across hardware/TP variants
        ks_mean = group["ks_stat"].mean()
        ks_std = group["ks_stat"].std() if n_configs > 1 else np.nan
        acf_r2_mean = group["acf_r2"].mean()
        acf_r2_std = group["acf_r2"].std() if n_configs > 1 else np.nan
        nrmse_mean = group["nrmse"].mean()
        nrmse_std = group["nrmse"].std() if n_configs > 1 else np.nan
        energy_err_mean = group["delta_energy_pct"].mean()
        energy_err_std = group["delta_energy_pct"].std() if n_configs > 1 else np.nan

        aggregated.append(
            {
                "model": model_family,
                "model_size": model_size,
                "arch_type": arch_type,
                "generation_mode": generation_mode,
                "n_configs": n_configs,
                "KS_mean": ks_mean,
                "KS_std": ks_std,
                "ACF_R2_mean": acf_r2_mean,
                "ACF_R2_std": acf_r2_std,
                "NRMSE_mean": nrmse_mean,
                "NRMSE_std": nrmse_std,
                "energy_err_mean": energy_err_mean,
                "energy_err_std": energy_err_std,
            }
        )

    result_df = pd.DataFrame(aggregated)

    return result_df


def select_best_generation_mode(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model, size), select generation mode with lowest KS_mean.

    Args:
        df: DataFrame with model-level aggregated metrics

    Returns:
        DataFrame with one row per (model, size), best generation_mode selected
    """
    # Group by (model, model_size)
    grouped = df.groupby(["model", "model_size"])

    best_rows = []
    for (model, model_size), group in grouped:
        # Find row with minimum KS_mean
        best_idx = group["KS_mean"].idxmin()
        best_row = group.loc[best_idx]
        best_rows.append(best_row)

    result = pd.DataFrame(best_rows).reset_index(drop=True)

    return result


# ============================================================================
# Formatting Functions
# ============================================================================

def format_display_name(model_family: str, model_size: int) -> str:
    """
    Format model name for display.

    Args:
        model_family: Model name (e.g., "llama-3", "deepseek-r1-distill")
        model_size: Model parameter count in billions

    Returns:
        Formatted display name (e.g., "Llama-3.1 (8B)")

    Example:
        >>> format_display_name("llama-3", 8)
        'Llama-3.1 (8B)'
        >>> format_display_name("deepseek-r1-distill", 70)
        'DeepSeek-R1-Distill (70B)'
    """
    display_name = MODEL_NAME_MAP.get(model_family, model_family)
    return f"{display_name} ({model_size}B)"


def format_metric_with_std(
    mean: float, std: float, decimals: int = 2, omit_std_if_nan: bool = True
) -> str:
    """
    Format metric with standard deviation for LaTeX.

    Args:
        mean: Metric mean value
        std: Metric standard deviation (may be NaN)
        decimals: Number of decimal places
        omit_std_if_nan: If True, omit ± when std is NaN

    Returns:
        Formatted string (e.g., "0.19~$\\pm$~0.09" or "0.09")

    Examples:
        >>> format_metric_with_std(0.193, 0.093, 2)
        '0.19~$\\\\pm$~0.09'
        >>> format_metric_with_std(0.094, np.nan, 2)
        '0.09'
        >>> format_metric_with_std(-1.3, 3.6, 1)
        '-1.3~$\\\\pm$~3.6'
    """
    mean_str = f"{mean:.{decimals}f}"

    if omit_std_if_nan and (pd.isna(std) or np.isnan(std)):
        return mean_str

    std_str = f"{std:.{decimals}f}"
    return f"{mean_str}~$\\pm$~{std_str}"


def format_latex_row(
    model_name: str,
    arch_type: str,
    ks_mean: float,
    ks_std: float,
    acf_mean: float,
    acf_std: float,
    nrmse_mean: float,
    nrmse_std: float,
    energy_mean: float,
    energy_std: float,
) -> str:
    """
    Generate LaTeX table row.

    Args:
        model_name: Display model name
        arch_type: "dense" or "moe" (capitalized for display)
        ks_mean, ks_std: KS statistic mean and std
        acf_mean, acf_std: ACF R² mean and std
        nrmse_mean, nrmse_std: NRMSE mean and std
        energy_mean, energy_std: Energy error % mean and std

    Returns:
        LaTeX row string (e.g., "Model & Arch & KS & ACF & NRMSE & Energy \\\\")
    """
    # Capitalize architecture type for display
    arch_display = arch_type.capitalize() if arch_type == "dense" else "MoE"

    # Format metrics
    ks_str = format_metric_with_std(ks_mean, ks_std, decimals=2)
    acf_str = format_metric_with_std(acf_mean, acf_std, decimals=2)
    nrmse_str = format_metric_with_std(nrmse_mean, nrmse_std, decimals=2)
    energy_str = format_metric_with_std(energy_mean, energy_std, decimals=1)

    # Construct row
    # Pad model name to align columns
    model_name_padded = f"{model_name:<26s}"

    row = (
        f"{model_name_padded} & {arch_display:5s} & "
        f"{ks_str:25s} & {acf_str:25s} & {nrmse_str:25s} & {energy_str:20s} \\\\"
    )

    return row


def format_csv_row(
    model_name: str,
    arch_type: str,
    ks_mean: float,
    ks_std: float,
    acf_mean: float,
    acf_std: float,
    nrmse_mean: float,
    nrmse_std: float,
    energy_mean: float,
    energy_std: float,
) -> List[str]:
    """
    Generate CSV row.

    Returns:
        List of strings for CSV writer
    """
    ks_str = format_metric_with_std(ks_mean, ks_std, decimals=2)
    acf_str = format_metric_with_std(acf_mean, acf_std, decimals=2)
    nrmse_str = format_metric_with_std(nrmse_mean, nrmse_std, decimals=2)
    energy_str = format_metric_with_std(energy_mean, energy_std, decimals=1)

    return [model_name, arch_type.capitalize(), ks_str, acf_str, nrmse_str, energy_str]


# ============================================================================
# Output Generation
# ============================================================================

def generate_latex_table(df: pd.DataFrame) -> List[str]:
    """
    Generate LaTeX table rows.

    Processing:
        1. Sort by architecture (dense first) and model_size (descending)
        2. Generate formatted rows
        3. Insert \\midrule separator between Dense and MoE

    Args:
        df: DataFrame with model-level aggregated metrics (one row per model)

    Returns:
        List of LaTeX row strings
    """
    # Sort: Dense first (by size desc), then MoE (by size desc)
    df_sorted = df.sort_values(
        by=["arch_type", "model_size"], ascending=[True, False]
    ).reset_index(drop=True)

    rows = []
    prev_arch = None

    for _, row in df_sorted.iterrows():
        # Insert separator between Dense and MoE
        if prev_arch == "dense" and row["arch_type"] == "moe":
            rows.append("\\midrule")

        # Format model name
        model_name = format_display_name(row["model"], row["model_size"])

        # Generate row
        latex_row = format_latex_row(
            model_name=model_name,
            arch_type=row["arch_type"],
            ks_mean=row["KS_mean"],
            ks_std=row["KS_std"],
            acf_mean=row["ACF_R2_mean"],
            acf_std=row["ACF_R2_std"],
            nrmse_mean=row["NRMSE_mean"],
            nrmse_std=row["NRMSE_std"],
            energy_mean=row["energy_err_mean"],
            energy_std=row["energy_err_std"],
        )

        rows.append(latex_row)
        prev_arch = row["arch_type"]

    return rows


def generate_csv_output(df: pd.DataFrame) -> List[List[str]]:
    """
    Generate CSV output rows.

    Args:
        df: DataFrame with model-level aggregated metrics

    Returns:
        List of lists (including header row)
    """
    # Sort: Dense first (by size desc), then MoE (by size desc)
    df_sorted = df.sort_values(
        by=["arch_type", "model_size"], ascending=[True, False]
    ).reset_index(drop=True)

    # Header
    rows = [["Model", "Architecture", "KS", "ACF_R2", "NRMSE", "Energy_Error"]]

    for _, row in df_sorted.iterrows():
        model_name = format_display_name(row["model"], row["model_size"])

        csv_row = format_csv_row(
            model_name=model_name,
            arch_type=row["arch_type"],
            ks_mean=row["KS_mean"],
            ks_std=row["KS_std"],
            acf_mean=row["ACF_R2_mean"],
            acf_std=row["ACF_R2_std"],
            nrmse_mean=row["NRMSE_mean"],
            nrmse_std=row["NRMSE_std"],
            energy_mean=row["energy_err_mean"],
            energy_std=row["energy_err_std"],
        )

        rows.append(csv_row)

    return rows


def print_summary_stats(df: pd.DataFrame, verbose: bool = False) -> None:
    """
    Print summary statistics for validation.

    Args:
        df: DataFrame with model-level aggregated metrics
        verbose: If True, print additional details
    """
    print("=" * 80)
    print("Trace Fidelity Table Summary")
    print("=" * 80)
    print(f"\nTotal models: {len(df)}")

    # Architecture breakdown
    arch_counts = df["arch_type"].value_counts()
    print(f"\nArchitecture breakdown:")
    for arch, count in arch_counts.items():
        print(f"  {arch.capitalize()}: {count}")

    # Generation mode breakdown
    gen_mode_counts = df["generation_mode"].value_counts()
    print(f"\nGeneration mode breakdown:")
    for mode, count in gen_mode_counts.items():
        print(f"  {mode}: {count}")

    # Config count range
    print(f"\nConfigs per model:")
    print(f"  Min: {df['n_configs'].min()}")
    print(f"  Max: {df['n_configs'].max()}")
    print(f"  Mean: {df['n_configs'].mean():.1f}")

    if verbose:
        print(f"\n{'=' * 80}")
        print("Per-Model Details:")
        print(f"{'=' * 80}")
        for _, row in df.iterrows():
            model_name = format_display_name(row["model"], row["model_size"])
            print(
                f"\n{model_name} ({row['arch_type'].capitalize()}, {row['generation_mode']}): "
                f"{row['n_configs']} configs"
            )
            print(f"  KS:     {row['KS_mean']:.3f} ± {row['KS_std']:.3f}")
            print(f"  ACF R²: {row['ACF_R2_mean']:.3f} ± {row['ACF_R2_std']:.3f}")
            print(f"  NRMSE:  {row['NRMSE_mean']:.3f} ± {row['NRMSE_std']:.3f}")
            print(f"  Energy: {row['energy_err_mean']:.2f}% ± {row['energy_err_std']:.2f}%")

    print(f"\n{'=' * 80}\n")


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate trace fidelity table from raw evaluation metrics"
    )

    # Input paths
    parser.add_argument(
        "--iid-dir",
        type=str,
        default=None,
        help=f"Directory with i.i.d. per_seed_metrics.csv (default: {DEFAULT_INPUT_DIRS['iid']})",
    )
    parser.add_argument(
        "--ar1-dir",
        type=str,
        default=None,
        help=f"Directory with AR(1) per_seed_metrics.csv (default: {DEFAULT_INPUT_DIRS['ar1']})",
    )
    parser.add_argument(
        "--ar1-thresh-dir",
        type=str,
        default=None,
        help=f"Directory with AR(1) thresholded per_seed_metrics.csv (default: {DEFAULT_INPUT_DIRS['ar1_thresh']})",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["latex", "csv"],
        default="latex",
        help="Output format (default: latex)",
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed statistics",
    )
    parser.add_argument(
        "--timestamp-inventory-json",
        type=str,
        default=os.path.join(REPO_ROOT, DEFAULT_TIMESTAMP_INVENTORY),
        help=(
            "Path to stage0 data inventory JSON used for pair_key request timestamp "
            "availability filtering."
        ),
    )
    parser.add_argument(
        "--allow-missing-request-timestamps",
        action="store_true",
        help=(
            "Disable request timestamp filtering and include all traces. "
            "Default behavior is to exclude traces without stored request timestamps."
        ),
    )
    parser.add_argument(
        "--use-auto-k",
        action="store_true",
        help=(
            "Use auto-K evaluation directories (kauto_max20_f2) instead of fixed K. "
            "Reports K values used per model in the summary."
        ),
    )

    args = parser.parse_args()

    # Select input directories based on auto-k flag
    if args.use_auto_k:
        default_dirs = AUTOK_INPUT_DIRS
        print("Using AUTO-K evaluation directories")
    else:
        default_dirs = DEFAULT_INPUT_DIRS

    # Resolve input directories
    input_dirs = {
        "iid": args.iid_dir or os.path.join(REPO_ROOT, default_dirs["iid"]),
        "ar1": args.ar1_dir or os.path.join(REPO_ROOT, default_dirs["ar1"]),
        "ar1_thresh": args.ar1_thresh_dir
        or os.path.join(REPO_ROOT, default_dirs["ar1_thresh"]),
    }

    print(f"Loading per-seed metrics from:")
    for mode, dir_path in input_dirs.items():
        csv_path = os.path.join(dir_path, "per_seed_metrics.csv")
        exists_str = "✓" if os.path.exists(csv_path) else "✗"
        print(f"  {mode:12s}: {csv_path} {exists_str}")

    # Load per-seed metrics
    print(f"\n{'=' * 80}")
    print("Loading per-seed metrics...")
    dfs_per_seed = []
    enforce_request_timestamp_filter = not bool(args.allow_missing_request_timestamps)

    for mode, dir_path in input_dirs.items():
        csv_path = os.path.join(dir_path, "per_seed_metrics.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: Skipping {mode}, file not found: {csv_path}")
            continue

        df = load_per_seed_csv(
            csv_path,
            generation_mode=mode,
            require_pair_key=bool(enforce_request_timestamp_filter),
        )
        print(f"  Loaded {len(df)} per-seed rows from {mode}")
        dfs_per_seed.append(df)

    if not dfs_per_seed:
        print("Error: No input files found!")
        sys.exit(1)

    # Concatenate all per-seed data
    df_all_seeds = pd.concat(dfs_per_seed, ignore_index=True)
    print(f"\nTotal per-seed rows: {len(df_all_seeds)}")

    # Load K values from manifests if using auto-k
    k_values_map: Dict[str, int] = {}
    if args.use_auto_k:
        print("\nLoading K values from run manifests...")
        for mode, dir_path in input_dirs.items():
            mode_k_values = load_k_values_from_manifest(dir_path)
            for config_id, k_val in mode_k_values.items():
                if config_id not in k_values_map:
                    k_values_map[config_id] = k_val
        if k_values_map:
            k_vals = list(k_values_map.values())
            print(f"  Loaded K values for {len(k_values_map)} configs")
            print(f"  K range: {min(k_vals)} - {max(k_vals)}")
            print(f"  K distribution: {dict(sorted(pd.Series(k_vals).value_counts().items()))}")

    if enforce_request_timestamp_filter:
        print(
            "\nApplying request timestamp availability filter "
            "(keeping only pair_keys with stored request_timestamps)..."
        )
        pair_key_has_timestamps = load_request_timestamp_availability_map(
            args.timestamp_inventory_json
        )
        df_all_seeds, filter_stats = filter_per_seed_rows_by_request_timestamps(
            df_all_seeds, pair_key_has_timestamps
        )
        print(f"  rows_before                 : {filter_stats['rows_before']}")
        print(f"  rows_after                  : {filter_stats['rows_after']}")
        print(
            f"  rows_dropped_no_timestamps  : "
            f"{filter_stats['rows_dropped_no_timestamps']}"
        )
        print(
            f"  rows_dropped_unmapped_key   : "
            f"{filter_stats['rows_dropped_unmapped_pair_key']}"
        )
        print(
            f"  unique_pair_keys_before/after: "
            f"{filter_stats['unique_pair_keys_before']} -> "
            f"{filter_stats['unique_pair_keys_after']}"
        )
        if len(df_all_seeds) == 0:
            print(
                "Error: No per-seed rows remain after request timestamp filtering."
            )
            sys.exit(1)
    else:
        print(
            "\nRequest timestamp filter disabled "
            "(including rows without stored request_timestamps)."
        )

    # Aggregation pipeline
    print(f"\n{'=' * 80}")
    print("Aggregation pipeline...")

    print("  1. Aggregating seeds → traces (median across seeds)...")
    df_per_trace = aggregate_seed_to_trace(df_all_seeds)
    print(f"     Result: {len(df_per_trace)} per-trace rows")

    print("  2. Aggregating traces → configs (median across traces)...")
    df_per_config = aggregate_trace_to_config(df_per_trace)
    print(f"     Result: {len(df_per_config)} per-config rows")

    # Filter out problematic hardware/TP configurations
    if EXCLUDED_HARDWARE_TP:
        print(f"\n  2b. Filtering excluded hardware/TP configs: {EXCLUDED_HARDWARE_TP}...")
        df_per_config, exclude_stats = filter_excluded_hardware_tp(
            df_per_config, EXCLUDED_HARDWARE_TP
        )
        print(f"      Excluded {exclude_stats['rows_excluded']} configs:")
        for cfg in exclude_stats.get("excluded_configs", []):
            print(f"        - {cfg}")
        print(f"      Result: {len(df_per_config)} per-config rows")

    print("  3. Aggregating configs → models (mean±std across hw/tp)...")
    df_per_model = aggregate_config_to_model(df_per_config)
    print(f"     Result: {len(df_per_model)} per-model rows")

    print("  4. Selecting best generation mode per model...")
    df_final = select_best_generation_mode(df_per_model)
    print(f"     Result: {len(df_final)} final model rows")

    # Print summary statistics
    print_summary_stats(df_final, verbose=args.verbose)

    # Generate output
    print(f"{'=' * 80}")
    print(f"Generating {args.format.upper()} output...")

    if args.format == "latex":
        output_lines = generate_latex_table(df_final)
    else:  # csv
        output_rows = generate_csv_output(df_final)
        # Convert to lines
        import csv
        import io

        output_buffer = io.StringIO()
        writer = csv.writer(output_buffer)
        writer.writerows(output_rows)
        output_lines = output_buffer.getvalue().strip().split("\n")

    # Write output
    if args.output:
        with open(args.output, "w") as f:
            f.write("\n".join(output_lines) + "\n")
        print(f"Output written to: {args.output}")
    else:
        print(f"\n{'=' * 80}")
        print("Output:")
        print(f"{'=' * 80}\n")
        for line in output_lines:
            print(line)

    print(f"\n{'=' * 80}")
    print("Done!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
