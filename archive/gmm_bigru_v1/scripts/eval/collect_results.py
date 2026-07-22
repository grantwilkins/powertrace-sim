"""
Aggregate node-level evaluation results for paper figures.

Policy:
    - All configs (dense and MoE) come from i.i.d. results.
      AR(1) generation was removed.
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from model.utils.config import parse_config_id


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


def load_result_csv(csv_path: str, generation_mode: str = "iid") -> pd.DataFrame:
    """
    Load a single config_summary.csv and add generation_mode column.

    Args:
        csv_path: Path to config_summary.csv file
        generation_mode: Always "iid" (AR(1) generation was removed)

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

    requested_mode = str(generation_mode).strip().lower()
    if "generation_mode" not in df.columns:
        raise ValueError("CSV is missing required generation_mode provenance")
    else:
        recorded_modes = {
            str(value).strip().lower()
            for value in df["generation_mode"].dropna().tolist()
            if str(value).strip()
        }
        if recorded_modes != {requested_mode}:
            raise ValueError(
                "CSV generation_mode does not match the requested source mode: "
                f"recorded={sorted(recorded_modes)}, requested={requested_mode!r}"
            )
        df["generation_mode"] = df["generation_mode"].astype(str).str.strip().str.lower()

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
        except ValueError as exc:
            raise ValueError(f"Invalid config_id {row['config_id']!r}") from exc
    if not parsed_rows:
        raise ValueError("No valid config_ids found in input data")

    return pd.DataFrame(parsed_rows)


def select_generation_rows(iid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare i.i.d. config rows for aggregation.

    All configs (dense and MoE) use i.i.d. generation; AR(1) was removed.

    Args:
        iid_df: i.i.d. config_summary rows

    Returns:
        Annotated rows with source_mode="iid"
    """
    modes = {
        str(value).strip().lower()
        for value in iid_df["generation_mode"].dropna().tolist()
    }
    if modes != {"iid"}:
        raise ValueError(f"IID selection requires only IID source rows; found {sorted(modes)}")
    selected = _parse_and_annotate_rows(iid_df.copy())
    selected["source_mode"] = "iid"
    return selected


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

        generation_modes = {
            str(value).strip().lower()
            for value in group["generation_mode"].dropna().tolist()
        }
        if len(generation_modes) != 1:
            raise ValueError(
                f"Cannot aggregate mixed generation modes for {model_family}-{model_size}: "
                f"{sorted(generation_modes)}"
            )
        generation_mode = next(iter(generation_modes))

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
    Main execution: load i.i.d. CSV → annotate → aggregate → save.
    """
    print("=" * 70)
    print("Collecting Evaluation Results for Paper Figures")
    print("=" * 70)

    # Configuration
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results",
        "continuous_v1_gmm_bigru",
    )

    input_files = {
        "iid": os.path.join(base_dir, "k10_f2", "eval_metrics", "config_summary.csv"),
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

    if not os.path.exists(iid_path):
        raise FileNotFoundError(f"Required i.i.d. CSV not found: {iid_path}")

    iid_df = load_result_csv(iid_path, generation_mode="iid")
    print(
        f"  Loaded {len(iid_df)} configs from iid ({os.path.basename(os.path.dirname(iid_path))})"
    )

    df_selected = select_generation_rows(iid_df)
    print(f"\nSelected configs for aggregation: {len(df_selected)}")

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
