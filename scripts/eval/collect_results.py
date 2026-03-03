"""
Aggregate node-level evaluation results for paper figures.

Policy:
    - Dense configs always come from i.i.d. results
    - MoE configs prefer AR(1) results when available, else fall back to i.i.d.
"""

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


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
            - model_family: Model name without size (e.g., "deepseek-r1-distill", "llama-3")
            - model_size: Parameter count in billions as string (e.g., "70", "8")
            - hardware: GPU type ("A100" or "H100")
            - tp: Tensor parallelism factor as string (e.g., "1", "4", "8")

    Raises:
        ValueError: If config_id format is invalid or cannot be parsed

    Example:
        >>> parse_config_id("deepseek-r1-distill-70b_H100_tp4")
        {'model_family': 'deepseek-r1-distill', 'model_size': '70',
         'hardware': 'H100', 'tp': '4'}
    """
    # Pattern: {model_name}-{size}b_{hardware}_tp{digits}
    # The model name ends with -{size}b, then hardware and TP follow
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
        model_family: Model name (e.g., "llama-3", "deepseek-r1-distill", "gpt-oss")
        model_size: Model parameter count in billions

    Returns:
        "dense" for standard transformers
        "moe" for mixture-of-experts models

    Classification rules:
        - DeepSeek-R1-Distill models are treated as dense
        - "gpt-oss" with size >= 20B → "moe"
        - All others → "dense"

    Example:
        >>> infer_arch_type("llama-3", 8)
        'dense'
        >>> infer_arch_type("deepseek-r1-distill", 70)
        'dense'
        >>> infer_arch_type("gpt-oss", 120)
        'moe'
    """
    model_family_lower = model_family.lower()

    # DeepSeek distill models are treated as dense.
    if "deepseek-r1-distill" in model_family_lower:
        return "dense"

    # Large GPT-OSS models are MoE
    if "gpt-oss" in model_family_lower and model_size >= 20:
        return "moe"

    # Default to dense
    return "dense"


def load_result_csv(csv_path: str, generation_mode: str) -> pd.DataFrame:
    """
    Load a single config_summary.csv and add generation_mode column.

    Args:
        csv_path: Path to config_summary.csv file
        generation_mode: One of "iid", "ar1", "ar1_thresh"

    Returns:
        DataFrame with all columns from CSV plus generation_mode

    Raises:
        FileNotFoundError: If csv_path doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = {
        "config_id",
        "ks_stat_median",
        "acf_r2_median",
        "nrmse_median",
        "delta_energy_pct_median",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"CSV missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    # Track where each row came from.
    df["generation_mode"] = generation_mode

    return df


def _parse_and_annotate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse config_id fields and attach model metadata columns.

    Args:
        df: DataFrame containing config_id and metric columns

    Returns:
        Parsed DataFrame with model_family, model_size, hardware, tp, arch_type columns.
    """
    parsed_rows: List[Dict[str, object]] = []
    skipped_count = 0

    for _, row in df.iterrows():
        try:
            parsed = parse_config_id(str(row["config_id"]))
            model_family = parsed["model_family"]
            model_size = int(parsed["model_size"])
            parsed_rows.append(
                {
                    **row.to_dict(),
                    "model_family": model_family,
                    "model_size": model_size,
                    "hardware": parsed["hardware"],
                    "tp": int(parsed["tp"]),
                    "arch_type": infer_arch_type(model_family, model_size),
                }
            )
        except ValueError as e:
            print(f"Warning: Skipping config_id '{row['config_id']}': {e}")
            skipped_count += 1

    if skipped_count > 0:
        print(f"Skipped {skipped_count} rows due to parsing errors")
    if not parsed_rows:
        raise ValueError("No valid config_ids found in input data")

    return pd.DataFrame(parsed_rows)


def select_generation_rows(
    iid_df: pd.DataFrame, ar1_df: Optional[pd.DataFrame]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply paper merge policy:
      - dense: i.i.d.
      - MoE: AR(1) when available, else i.i.d. fallback.

    Args:
        iid_df: i.i.d. config_summary rows
        ar1_df: AR(1) config_summary rows (can be None or empty)

    Returns:
        (selected_rows, moe_fallback_config_ids)
    """
    metric_cols: Sequence[str] = (
        "ks_stat_median",
        "acf_r2_median",
        "nrmse_median",
        "delta_energy_pct_median",
    )

    selected = _parse_and_annotate_rows(iid_df.copy())
    selected["source_mode"] = "iid"

    if ar1_df is None or len(ar1_df) == 0:
        fallback_configs = sorted(
            selected.loc[selected["arch_type"] == "moe", "config_id"].astype(str).tolist()
        )
        return selected, fallback_configs

    ar1_parsed = _parse_and_annotate_rows(ar1_df.copy())
    ar1_moe = ar1_parsed.loc[ar1_parsed["arch_type"] == "moe"].copy()

    if ar1_moe.empty:
        fallback_configs = sorted(
            selected.loc[selected["arch_type"] == "moe", "config_id"].astype(str).tolist()
        )
        return selected, fallback_configs

    # Keep one AR(1) row per config_id; prefer lowest KS if duplicates exist.
    ar1_moe = ar1_moe.sort_values(by="ks_stat_median", ascending=True)
    ar1_moe = ar1_moe.drop_duplicates(subset=["config_id"], keep="first")
    ar1_by_config = ar1_moe.set_index("config_id")

    moe_rows = selected["arch_type"] == "moe"
    has_ar1 = selected["config_id"].isin(ar1_by_config.index)
    replace_mask = moe_rows & has_ar1

    for col in metric_cols:
        selected.loc[replace_mask, col] = selected.loc[replace_mask, "config_id"].map(
            ar1_by_config[col]
        )
    selected.loc[replace_mask, "source_mode"] = "ar1"

    fallback_configs = sorted(
        selected.loc[moe_rows & (~has_ar1), "config_id"].astype(str).tolist()
    )
    return selected, fallback_configs


def aggregate_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group selected config rows by model family/size and compute statistics.

    Expected input is the output from select_generation_rows(). If parsing columns
    are absent, this function will parse config_id values directly.

    Args:
        df: Selected config rows with metrics and model metadata

    Returns:
        DataFrame with columns:
            model, model_size, arch_type, generation_mode, n_configs,
            KS_mean, KS_std, ACF_R2_mean, ACF_R2_std,
            NRMSE_mean, NRMSE_std, energy_err_mean, energy_err_std

    Example:
        >>> df_in = pd.DataFrame({
        ...     'config_id': ['llama-3-8b_H100_tp1', 'llama-3-8b_H100_tp4'],
        ...     'ks_stat_median': [0.5, 0.6],
        ...     'acf_r2_median': [0.8, 0.75],
        ...     'nrmse_median': [0.3, 0.35],
        ...     'delta_energy_pct_median': [5.0, 6.0],
        ...     'generation_mode': ['iid', 'iid']
        ... })
        >>> result = aggregate_by_model(df_in)
        >>> result.columns
        Index(['model', 'model_size', 'arch_type', 'generation_mode', 'n_configs',
               'KS_mean', 'KS_std', 'ACF_R2_mean', 'ACF_R2_std',
               'NRMSE_mean', 'NRMSE_std', 'energy_err_mean', 'energy_err_std'],
              dtype='object')
    """
    if "model_family" in df.columns and "model_size" in df.columns and "arch_type" in df.columns:
        df_parsed = df.copy()
    else:
        df_parsed = _parse_and_annotate_rows(df)

    # Group by model family/size/arch
    grouped = df_parsed.groupby(["model_family", "model_size", "arch_type"])

    aggregated = []
    for (model_family, model_size, arch_type), group in grouped:
        # Count configs
        n_configs = len(group)

        # Compute statistics across hardware/TP variants
        ks_mean = group["ks_stat_median"].mean()
        ks_std = group["ks_stat_median"].std()
        acf_r2_mean = group["acf_r2_median"].mean()
        acf_r2_std = group["acf_r2_median"].std()
        nrmse_mean = group["nrmse_median"].mean()
        nrmse_std = group["nrmse_median"].std()
        energy_err_mean = group["delta_energy_pct_median"].mean()
        energy_err_std = group["delta_energy_pct_median"].std()

        if arch_type == "dense":
            generation_mode = "iid"
        else:
            generation_mode = "ar1_with_iid_fallback"

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

    # Sort by arch_type (dense first, then moe), then by model_size descending
    result_df = result_df.sort_values(
        by=["arch_type", "model_size"], ascending=[True, False]
    ).reset_index(drop=True)

    return result_df


def save_summary(df: pd.DataFrame, output_path: str) -> None:
    """
    Save aggregated results to CSV with directory creation.

    Args:
        df: Aggregated summary DataFrame
        output_path: Target CSV path

    Side effects:
        Creates parent directories if needed
        Overwrites existing file at output_path
    """
    # Create parent directories
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"Saved aggregated results to: {output_path}")


def main():
    """
    Main execution: load CSVs → select dense/iid + moe/ar1(fallback) → aggregate → save.
    """
    print("=" * 70)
    print("Collecting Evaluation Results for Paper Figures")
    print("=" * 70)

    # Configuration
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results",
        "continuous_v1_gmm_bigru_sharegpt_all",
    )

    input_files = {
        "iid": os.path.join(base_dir, "kauto_max12_f2", "eval_metrics", "config_summary.csv"),
        "ar1": os.path.join(base_dir, "kauto_max12_f2_ar1", "eval_metrics", "config_summary.csv"),
    }

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results",
        "eval_paper",
        "node_level_summary.csv",
    )

    print(f"\nInput files:")
    for mode, path in input_files.items():
        print(f"  {mode:12s}: {path}")
    print(f"\nOutput file:")
    print(f"  {output_path}")

    # Load CSVs
    print(f"\n{'=' * 70}")
    print("Loading results CSVs...")
    iid_path = input_files["iid"]
    ar1_path = input_files["ar1"]

    if not os.path.exists(iid_path):
        raise FileNotFoundError(f"Required i.i.d. CSV not found: {iid_path}")

    iid_df = load_result_csv(iid_path, generation_mode="iid")
    print(
        f"  Loaded {len(iid_df)} configs from iid ({os.path.basename(os.path.dirname(iid_path))})"
    )

    ar1_df: Optional[pd.DataFrame]
    if os.path.exists(ar1_path):
        ar1_df = load_result_csv(ar1_path, generation_mode="ar1")
        print(
            f"  Loaded {len(ar1_df)} configs from ar1 ({os.path.basename(os.path.dirname(ar1_path))})"
        )
    else:
        print(f"Warning: AR(1) file not found; using i.i.d. fallback for all MoE: {ar1_path}")
        ar1_df = None

    # Apply merge policy before aggregation
    df_selected, fallback_moe_configs = select_generation_rows(iid_df=iid_df, ar1_df=ar1_df)
    print(f"\nSelected configs for aggregation: {len(df_selected)}")
    if fallback_moe_configs:
        print(
            "MoE configs using i.i.d. fallback (missing AR(1)): "
            + ", ".join(fallback_moe_configs)
        )

    # Aggregate
    print(f"\n{'=' * 70}")
    print("Aggregating by model architecture...")
    df_summary = aggregate_by_model(df_selected)

    print(f"\nAggregated to {len(df_summary)} model groups:")
    for _, row in df_summary.iterrows():
        print(
            f"  {row['model']:25s} {row['model_size']:3d}B ({row['arch_type']:5s}): "
            f"{row['n_configs']} configs, KS={row['KS_mean']:.3f}±{row['KS_std']:.3f}, "
            f"mode={row['generation_mode']}"
        )

    # Validate results
    print(f"\n{'=' * 70}")
    print("Validating output...")
    validation_passed = True

    # Check for unexpected NaN/inf
    # Std columns may be NaN when n_configs == 1.
    allowed_nan_cols = {"KS_std", "ACF_R2_std", "NRMSE_std", "energy_err_std"}
    nan_cols = [
        col
        for col in df_summary.columns
        if df_summary[col].isnull().any() and col not in allowed_nan_cols
    ]
    if nan_cols:
        print(f"  ⚠ Warning: Found unexpected NaN values in columns: {nan_cols}")
        validation_passed = False

    if np.isinf(df_summary.select_dtypes(include=[np.number])).any().any():
        print("  ⚠ Warning: Found inf values in output")
        validation_passed = False

    # Check KS range
    if (df_summary["KS_mean"] < 0).any() or (df_summary["KS_mean"] > 1).any():
        print("  ⚠ Warning: KS_mean values outside [0, 1] range")
        validation_passed = False

    # Check n_configs > 0
    if (df_summary["n_configs"] <= 0).any():
        print("  ⚠ Warning: Found groups with n_configs <= 0")
        validation_passed = False

    if validation_passed:
        print("  ✓ All validation checks passed")

    # Save
    print(f"\n{'=' * 70}")
    save_summary(df_summary, output_path)

    print(f"\n{'=' * 70}")
    print(f"Aggregation complete!")
    print(f"Output: {output_path}")
    print(f"Rows: {len(df_summary)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
