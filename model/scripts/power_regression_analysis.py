#!/usr/bin/env python3
import argparse
import glob
import os
import csv
from typing import Dict, List, Optional, Tuple

import numpy as np


FEATURE_KEYS = ("prefill_tokens", "decode_tokens", "active_requests")
STRESS_METRIC_LABELS = {
    "mean_power": "Mean power (W)",
    "p95_power": "P95 power (W)",
    "mean_power_frac": "Mean power / global max power",
    "p95_power_frac": "P95 power / global max power",
    "mean_active_requests": "Mean active requests",
    "mean_prefill_tokens": "Mean prefill tokens",
    "mean_decode_tokens": "Mean decode tokens",
    "mean_total_tokens": "Mean total tokens (prefill+decode)",
}
DEFAULT_CONFIG_KEYS = (
    "model_sizes",
    "hardware",
    "tensor_parallelism",
    "poisson_rate",
)


def _format_config_value(value) -> str:
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.6g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def _config_label(npz: np.lib.npyio.NpzFile, idx: int, keys: Tuple[str, ...]) -> str:
    parts = []
    for key in keys:
        if key not in npz:
            raise KeyError(f"Missing config key '{key}' in npz")
        parts.append(f"{key}={_format_config_value(npz[key][idx])}")
    return "|".join(parts)


def _ols_r2(X: np.ndarray, y: np.ndarray, fit_intercept: bool = True) -> float:
    if fit_intercept:
        X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    sse = np.sum((y - y_hat) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    if sst == 0:
        return float("nan")
    return 1.0 - (sse / sst)


def _fixed_effects_r2(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    groups = groups.astype(int)
    n_groups = groups.max() + 1
    counts = np.bincount(groups, minlength=n_groups).astype(np.float64)
    y_sum = np.bincount(groups, weights=y, minlength=n_groups).astype(np.float64)
    x_sums = np.zeros((n_groups, X.shape[1]), dtype=np.float64)
    for col in range(X.shape[1]):
        x_sums[:, col] = np.bincount(
            groups, weights=X[:, col], minlength=n_groups
        ).astype(np.float64)

    y_mean = y_sum / counts
    x_mean = x_sums / counts[:, None]

    y_dm = y - y_mean[groups]
    X_dm = X - x_mean[groups]

    beta, *_ = np.linalg.lstsq(X_dm, y_dm, rcond=None)
    alpha = y_mean - x_mean @ beta
    y_hat = (X @ beta) + alpha[groups]

    sse = np.sum((y - y_hat) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    if sst == 0:
        return float("nan")
    return 1.0 - (sse / sst)


def _per_config_stats(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    group_map: Dict[str, int],
) -> List[Dict[str, float]]:
    inverse_map = {idx: label for label, idx in group_map.items()}
    order = np.argsort(groups)
    groups_sorted = groups[order]
    X_sorted = X[order]
    y_sorted = y[order]

    boundaries = np.flatnonzero(np.diff(groups_sorted)) + 1
    splits = np.concatenate(([0], boundaries, [len(groups_sorted)]))

    results: List[Dict[str, float]] = []
    max_power = float(np.nanmax(y_sorted)) if len(y_sorted) else float("nan")
    feature_idx = {k: i for i, k in enumerate(FEATURE_KEYS)}
    for start, end in zip(splits[:-1], splits[1:]):
        group_idx = int(groups_sorted[start])
        label = inverse_map.get(group_idx, f"group_{group_idx}")
        X_slice = X_sorted[start:end]
        y_slice = y_sorted[start:end]
        if len(y_slice) < 2 or np.allclose(y_slice, y_slice[0]):
            r2 = float("nan")
        else:
            r2 = _ols_r2(X_slice, y_slice, fit_intercept=True)
        mean_power = float(np.mean(y_slice)) if len(y_slice) else float("nan")
        p95_power = float(np.percentile(y_slice, 95)) if len(y_slice) else float("nan")
        mean_active_requests = float(
            np.mean(X_slice[:, feature_idx["active_requests"]])
        )
        mean_prefill_tokens = float(
            np.mean(X_slice[:, feature_idx["prefill_tokens"]])
        )
        mean_decode_tokens = float(
            np.mean(X_slice[:, feature_idx["decode_tokens"]])
        )
        mean_total_tokens = mean_prefill_tokens + mean_decode_tokens
        results.append(
            {
                "label": label,
                "group_idx": float(group_idx),
                "n": float(len(y_slice)),
                "r2": r2,
                "mean_power": mean_power,
                "p95_power": p95_power,
                "mean_power_frac": mean_power / max_power if max_power > 0 else float("nan"),
                "p95_power_frac": p95_power / max_power if max_power > 0 else float("nan"),
                "mean_active_requests": mean_active_requests,
                "mean_prefill_tokens": mean_prefill_tokens,
                "mean_decode_tokens": mean_decode_tokens,
                "mean_total_tokens": mean_total_tokens,
            }
        )
    return results


def _collect_npz_paths(
    npz_paths: List[str], npz_dir: Optional[str], pattern: str
) -> List[str]:
    paths: List[str] = []
    if npz_paths:
        paths.extend(npz_paths)
    if npz_dir:
        paths.extend(sorted(glob.glob(os.path.join(npz_dir, pattern))))
    deduped = []
    seen = set()
    for path in paths:
        path = os.path.expanduser(path)
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def _extract_arrays(
    npz_paths: List[str],
    config_keys: Tuple[str, ...],
    use_lag: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[str, int]]:
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    lag_parts: List[np.ndarray] = []
    group_parts: List[np.ndarray] = []
    group_map: Dict[str, int] = {}

    skipped_traces = 0
    total_traces = 0

    for path in npz_paths:
        npz = np.load(path, allow_pickle=True)
        n_traces = npz["timestamps"].shape[0]
        total_traces += n_traces
        for idx in range(n_traces):
            mask = npz["timestamps"][idx] > 0
            if not np.any(mask):
                skipped_traces += 1
                continue
            y = npz["power_traces"][idx][mask].astype(np.float64)
            X = np.stack(
                [
                    npz[key][idx][mask].astype(np.float64)
                    for key in FEATURE_KEYS
                ],
                axis=1,
            )
            if use_lag:
                if len(y) < 2:
                    skipped_traces += 1
                    continue
                lag = y[:-1]
                y = y[1:]
                X = X[1:]
            label = _config_label(npz, idx, config_keys)
            group_idx = group_map.setdefault(label, len(group_map))

            X_parts.append(X)
            y_parts.append(y)
            if use_lag:
                lag_parts.append(lag)
            group_parts.append(np.full(len(y), group_idx, dtype=np.int32))

    if not X_parts:
        raise ValueError("No usable traces found in provided NPZ files.")

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    groups_all = np.concatenate(group_parts, axis=0)
    if use_lag:
        lag_all = np.concatenate(lag_parts, axis=0)
    else:
        lag_all = np.empty((0,), dtype=np.float64)

    meta = {
        "total_traces": total_traces,
        "skipped_traces": skipped_traces,
    }
    return X_all, y_all, lag_all, groups_all, group_map, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate variance explained by instantaneous workload, config fixed effects, "
            "and lagged power on existing NPZ training data."
        )
    )
    parser.add_argument(
        "--npz",
        action="append",
        default=[],
        help="Path to an NPZ file (repeatable).",
    )
    parser.add_argument(
        "--npz-dir",
        default=None,
        help="Directory to search for NPZ files (defaults to model/training_data if no --npz provided).",
    )
    parser.add_argument(
        "--pattern",
        default="*.npz",
        help="Glob pattern for NPZ files inside --npz-dir.",
    )
    parser.add_argument(
        "--config-keys",
        default=",".join(DEFAULT_CONFIG_KEYS),
        help="Comma-separated NPZ keys to define configuration fixed effects.",
    )
    parser.add_argument(
        "--no-per-config",
        action="store_false",
        dest="per_config",
        help="Disable per-config R2 reporting for Model 1.",
    )
    parser.set_defaults(per_config=True)
    parser.add_argument(
        "--per-config-top",
        type=int,
        default=5,
        help="Number of highest/lowest per-config R2 entries to show.",
    )
    parser.add_argument(
        "--per-config-min-n",
        type=int,
        default=0,
        help="Minimum observations required to include a config in per-config summary.",
    )
    parser.add_argument(
        "--per-config-csv",
        default=None,
        help="Optional path to write per-config stats as CSV.",
    )
    parser.add_argument(
        "--stress-metric",
        default="mean_power",
        choices=sorted(STRESS_METRIC_LABELS.keys()),
        help="Metric for per-config stress/utilization plotting.",
    )
    parser.add_argument(
        "--plot-path",
        default=None,
        help="If set, write a scatter plot of per-config R2 vs stress metric.",
    )
    parser.add_argument(
        "--plot-title",
        default=None,
        help="Optional title for the plot.",
    )
    parser.add_argument(
        "--no-lag",
        action="store_true",
        help="Skip the AR(1) model and use full traces without lagging.",
    )

    args = parser.parse_args()
    config_keys = tuple(k.strip() for k in args.config_keys.split(",") if k.strip())
    npz_dir = args.npz_dir
    if not args.npz and npz_dir is None:
        npz_dir = "model/training_data"

    npz_paths = _collect_npz_paths(args.npz, npz_dir, args.pattern)
    if not npz_paths:
        raise SystemExit("No NPZ files found. Provide --npz or --npz-dir.")

    X, y, y_lag, groups, group_map, meta = _extract_arrays(
        npz_paths=npz_paths,
        config_keys=config_keys,
        use_lag=not args.no_lag,
    )

    if args.no_lag:
        r2_base = _ols_r2(X, y, fit_intercept=True)
        r2_fe = _fixed_effects_r2(X, y, groups)
        r2_fe_lag = float("nan")
    else:
        r2_base = _ols_r2(X, y, fit_intercept=True)
        r2_fe = _fixed_effects_r2(X, y, groups)
        X_lag = np.column_stack([X, y_lag])
        r2_fe_lag = _fixed_effects_r2(X_lag, y, groups)

    print("Loaded NPZ files:")
    for path in npz_paths:
        print(f"  - {path}")
    print(
        f"Traces: {meta['total_traces']} total, {meta['skipped_traces']} skipped"
    )
    print(f"Observations used: {len(y):,}")
    print(f"Configs: {len(group_map)} ({', '.join(config_keys)})")
    print("Features: " + ", ".join(FEATURE_KEYS))
    if args.no_lag:
        print("Lagged power: disabled")
    else:
        print("Lagged power: enabled (AR(1), dropping first timestep per trace)")

    print("\nR2 summary:")
    print(
        "  Model 1: power ~ prefill_tokens + decode_tokens + active_requests"
        f"\n    R2 = {r2_base:.4f}"
    )
    print(
        "  Model 2: Model 1 + config fixed effects"
        f"\n    R2 = {r2_fe:.4f} (delta {r2_fe - r2_base:+.4f})"
    )
    if not args.no_lag:
        print(
            "  Model 3: Model 2 + lagged power (AR1)"
            f"\n    R2 = {r2_fe_lag:.4f} (delta {r2_fe_lag - r2_fe:+.4f})"
        )

    if args.per_config:
        per_config = _per_config_stats(X, y, groups, group_map)
        per_config = [
            row for row in per_config if row["n"] >= args.per_config_min_n
        ]
        r2_values = np.array(
            [row["r2"] for row in per_config if np.isfinite(row["r2"])],
            dtype=np.float64,
        )
        weight_values = np.array(
            [row["n"] for row in per_config if np.isfinite(row["r2"])],
            dtype=np.float64,
        )

        print("\nPer-config Model 1 R2:")
        print(
            f"  Configs reported: {len(per_config)} "
            f"(min_n={args.per_config_min_n})"
        )
        if len(r2_values) == 0:
            print("  No valid per-config R2 values.")
        else:
            weighted_mean = float(np.average(r2_values, weights=weight_values))
            print(
                "  R2 stats: "
                f"mean={r2_values.mean():.4f}, "
                f"weighted_mean={weighted_mean:.4f}, "
                f"median={np.median(r2_values):.4f}, "
                f"p25={np.percentile(r2_values, 25):.4f}, "
                f"p75={np.percentile(r2_values, 75):.4f}, "
                f"min={r2_values.min():.4f}, "
                f"max={r2_values.max():.4f}"
            )

            top_n = max(0, args.per_config_top)
            if top_n > 0:
                per_config_sorted = sorted(
                    per_config, key=lambda row: row["r2"]
                )
                print(f"  Lowest {top_n} configs:")
                for row in per_config_sorted[:top_n]:
                    print(
                        f"    r2={row['r2']:.4f} n={int(row['n'])} {row['label']}"
                    )
                print(f"  Highest {top_n} configs:")
                for row in per_config_sorted[-top_n:][::-1]:
                    print(
                        f"    r2={row['r2']:.4f} n={int(row['n'])} {row['label']}"
                    )

        if args.per_config_csv:
            os.makedirs(os.path.dirname(args.per_config_csv) or ".", exist_ok=True)
            fieldnames = [
                "label",
                "group_idx",
                "n",
                "r2",
                "mean_power",
                "p95_power",
                "mean_power_frac",
                "p95_power_frac",
                "mean_active_requests",
                "mean_prefill_tokens",
                "mean_decode_tokens",
                "mean_total_tokens",
            ]
            with open(args.per_config_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in per_config:
                    writer.writerow({k: row.get(k) for k in fieldnames})
            print(f"Wrote per-config CSV to {args.per_config_csv}")

        if args.plot_path:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("matplotlib is not available; skipping plot.")
            else:
                xs = []
                ys = []
                for row in per_config:
                    x_val = row.get(args.stress_metric)
                    y_val = row.get("r2")
                    if (
                        x_val is None
                        or y_val is None
                        or not np.isfinite(x_val)
                        or not np.isfinite(y_val)
                    ):
                        continue
                    xs.append(float(x_val))
                    ys.append(float(y_val))

                if len(xs) == 0:
                    print("No valid points to plot.")
                else:
                    fig, ax = plt.subplots(figsize=(6.5, 4.0))
                    ax.scatter(xs, ys, alpha=0.7, edgecolors="none")
                    if len(xs) >= 2:
                        coeffs = np.polyfit(xs, ys, 1)
                        x_line = np.linspace(min(xs), max(xs), 100)
                        y_line = coeffs[0] * x_line + coeffs[1]
                        ax.plot(x_line, y_line, color="tab:red", linewidth=1)
                        corr = np.corrcoef(xs, ys)[0, 1]
                        ax.text(
                            0.02,
                            0.98,
                            f"Pearson r = {corr:.2f}",
                            transform=ax.transAxes,
                            ha="left",
                            va="top",
                        )

                    ax.set_xlabel(STRESS_METRIC_LABELS[args.stress_metric])
                    ax.set_ylabel("Per-config R2 (Model 1)")
                    if args.plot_title:
                        ax.set_title(args.plot_title)
                    fig.tight_layout()
                    os.makedirs(os.path.dirname(args.plot_path) or ".", exist_ok=True)
                    fig.savefig(args.plot_path, dpi=200)
                    print(f"Wrote plot to {args.plot_path}")


if __name__ == "__main__":
    main()
