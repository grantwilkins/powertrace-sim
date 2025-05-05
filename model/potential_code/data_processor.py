#!/usr/bin/env python3
"""
Clean-room rewrite of the original “data preparation” script.

Highlights
----------
* **Pair discovery** unchanged except for case-insensitive regex flags.
* **Power logs**
    – still assume exactly 8 rows per time-stamp sweep.
    – still “sum the first <tp> GPUs”; no memory-based heuristic added.
    – much more permissive column normalisation.
* **Results logs**
    – ISO-8601 **or** epoch-second “Request Time” handled transparently.
* **Timeline alignment**
    – pick the *first* power-log time-stamp as the common `t = 0 s`.
    – never drop pairs just because the first request occurs a few seconds
      after (or before) logging started.
    – power rows after the very last completion are trimmed; nothing else.
* **Token distribution**
    – unchanged mathematically, but now uses the common zero-point so windows
      line up for every vintage of results file.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1.  Pair discovery
# ---------------------------------------------------------------------------


def discover_experiment_pairs(data_root_dir: str) -> List[Tuple[str, str]]:
    """
    Match power-trace CSVs to results CSVs.

        {MODEL}_tp{TP}_p{RATE}_d{DATE}.csv             (power)
        results_{MODEL}_{RATE}_{TP}_d{DATE}_final.csv  (results)
        results_{MODEL}_{RATE}_{TP}_final.csv          (results, no date)

    Returns
    -------
    list[(power_path, results_path)]
    """
    all_csvs = glob.glob(os.path.join(data_root_dir, "**", "*.csv"), recursive=True)

    power_files, results_files = [], []
    for p in all_csvs:
        base = os.path.basename(p)
        if base.lower().startswith("results_"):
            results_files.append(p)
        elif re.search(r"_tp\d+_p[\d\.]+", base, flags=re.I):
            power_files.append(p)

    def _pinfo(fname: str):
        m = re.match(
            r"(.*)_tp(\d+)_p([\d\.]+)_d(.*)\.csv$", Path(fname).name, flags=re.I
        )
        return None if m is None else m.groups()  # (model, tp, rate, date)

    def _rinfo(fname: str):
        pat1 = r"results_(.*)_([\d\.]+)_(\d+)_d(.*)_final\.csv$"
        pat2 = r"results_(.*)_([\d\.]+)_(\d+)_final\.csv$"
        for pat in (pat1, pat2):
            m = re.match(pat, Path(fname).name, flags=re.I)
            if m:
                groups = m.groups()
                return groups if len(groups) == 4 else (*groups, None)
        return None

    pairs: list[tuple[str, str]] = []
    for pfile in power_files:
        p_model, p_tp, p_rate, p_date = _pinfo(pfile) or (None, None, None, None)
        for rfile in results_files:
            rm = _rinfo(rfile)
            if not rm:
                continue
            r_model, r_rate, r_tp, r_date = rm
            if (p_model, p_tp, p_rate) == (r_model, r_tp, r_rate):
                if p_date and r_date and p_date != r_date:
                    continue
                pairs.append((pfile, rfile))
                break

    return pairs


# ---------------------------------------------------------------------------
# 2.  CSV helpers
# ---------------------------------------------------------------------------

_FLOAT_RX = re.compile(r"[^\d\.]+")


def _col_numeric(series: pd.Series) -> pd.Series:
    """Remove stray chars like ‘ W’, ‘ MiB’, ‘%’, then cast to float."""
    if np.issubdtype(series.dtype, np.number):
        return series.astype(float)
    return series.replace(_FLOAT_RX, "", regex=True).astype(float)


def parse_power_csv(csv_path: str) -> pd.DataFrame:
    """
    Read an nvidia-smi log exported with eight GPUs per node.

    Returns
    -------
    DataFrame with absolute-epoch seconds under column ``timestamp`` and the
    *summed* power of the first <tp> GPUs under column ``power``.
    """
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]

    # Canonical names
    df.rename(
        columns={
            "power.draw [w]": "power",
            "power.draw [w] ": "power",
            "memory.used [mib]": "memory",
            "memory.used [mib] ": "memory",
        },
        inplace=True,
    )

    # Normalise numeric columns
    df["power"] = _col_numeric(df["power"])
    if "memory" in df.columns:
        df["memory"] = _col_numeric(df["memory"])

    # detect timestamp column (“time”, “timestamp”, etc.)
    tcol = next((c for c in df.columns if "time" in c), None)
    if tcol is None:
        raise ValueError(f"{csv_path}: no timestamp column detected")
    df[tcol] = pd.to_datetime(df[tcol])

    # group rows into blocks of 8 (one per GPU)
    tp = int(re.search(r"_tp(\d+)", csv_path, flags=re.I).group(1))
    groups = df.iloc[: len(df) // 8 * 8].groupby(np.arange(len(df)) // 8)

    out = groups.apply(
        lambda g: pd.Series(
            {
                "timestamp": g[tcol].iloc[0].timestamp(),  # epoch seconds
                "power": g.iloc[:tp]["power"].sum(),
            }
        )
    ).reset_index(drop=True)

    return out


def _to_datetime_any(col: pd.Series) -> pd.Series:
    """Accept POSIX seconds *or* ISO-8601 strings."""
    if np.issubdtype(col.dtype, np.number):
        return pd.to_datetime(col, unit="s", origin="unix")
    return pd.to_datetime(col)  # assume ISO / RFC-3339


def parse_results_csv(csv_path: str) -> pd.DataFrame:
    """
    Parse a “results_*.csv” and supply *Completion Time* + *Model Size*.
    """
    df = pd.read_csv(csv_path)

    df["Request Time"] = _to_datetime_any(df["Request Time"])
    if "E2E Latency" not in df.columns:
        raise ValueError(f"{csv_path} missing 'E2E Latency'")
    df["Completion Time"] = df["Request Time"] + pd.to_timedelta(
        df["E2E Latency"], unit="s"
    )

    def _size(model_str: str) -> int:
        m = re.search(r"(\d+(?:\.\d+)?)b", str(model_str).lower())
        return int(float(m.group(1))) if m else np.nan

    df["Model Size"] = df["Model"].apply(_size)

    df.sort_values("Request Time", inplace=True, ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# 3.  Token distribution
# ---------------------------------------------------------------------------


def distribute_tokens(
    power_df: pd.DataFrame,
    results_df: pd.DataFrame,
    *,
    window_s: float = 0.25,
) -> pd.DataFrame:
    """
    Populate ``prefill_tokens`` and ``decode_tokens`` columns in *power_df*
    using a fixed‐width sliding window.

    All time stamps must already be *relative to the same zero-point*.
    """
    r = results_df.copy()
    using_prefill = {"Prefill Tokens", "Decode Tokens"} <= set(r.columns)

    power_df = power_df.copy()
    power_df[["prefill_tokens", "decode_tokens"]] = 0.0

    for i, t in enumerate(power_df["timestamp"]):
        w_start, w_end = t, t + window_s
        active = (r["end_s"] > w_start) & (r["start_s"] < w_end)
        if not active.any():
            continue

        seg = r.loc[active]

        if using_prefill:
            power_df.at[i, "prefill_tokens"] = seg["Prefill Tokens"].sum()
            power_df.at[i, "decode_tokens"] = seg["Decode Tokens"].sum()
        else:
            overlap = (
                np.minimum(seg["end_s"], w_end) - np.maximum(seg["start_s"], w_start)
            ).clip(lower=0.0)
            duration = (seg["end_s"] - seg["start_s"]).replace(0, np.nan)
            frac = (overlap / duration).fillna(0.0)

            power_df.at[i, "prefill_tokens"] = (seg["Input Tokens"] * frac).sum()
            power_df.at[i, "decode_tokens"] = (seg["Output Tokens"] * frac).sum()

    return power_df


# ---------------------------------------------------------------------------
# 4.  Orchestrator
# ---------------------------------------------------------------------------


def load_and_process(
    data_root: str,
    *,
    exclude: Optional[list[str]] = None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[dict]]:
    """
    Parse *all* matched experiment pairs under *data_root*.

    Returns
    -------
    (power_dfs, result_dfs, meta)
    """
    pairs = discover_experiment_pairs(data_root)
    if exclude:
        pairs = [
            p
            for p in pairs
            if not any(x.lower() in (p[0] + p[1]).lower() for x in exclude)
        ]

    power_dfs, result_dfs, meta = [], [], []

    for p_csv, r_csv in pairs:
        try:
            p_df = parse_power_csv(p_csv)
            r_df = parse_results_csv(r_csv)
        except Exception as e:
            print(f"[WARN] skipping ({p_csv}, {r_csv}): {e}", file=sys.stderr)
            continue

        # ------------------------------------------------------------------
        # Common zero-point = first power-log time-stamp
        # ------------------------------------------------------------------
        t0 = p_df["timestamp"].iloc[0]

        p_df["timestamp"] = p_df["timestamp"] - t0
        r_df["start_s"] = r_df["Request Time"].apply(lambda ts: ts.timestamp() - t0)
        r_df["end_s"] = r_df["Completion Time"].apply(lambda ts: ts.timestamp() - t0)

        # Trim power rows after the final completion only (keep any leading idle)
        last_completion_abs = r_df["Completion Time"].max().timestamp()
        p_df = p_df.loc[p_df["timestamp"] + t0 <= last_completion_abs].reset_index(
            drop=True
        )

        if p_df.empty:
            print(
                f"[WARN] {Path(p_csv).name}: no power rows within experiment window — skipped",
                file=sys.stderr,
            )
            continue

        # ------------------------------------------------------------------
        # Token accounting
        # ------------------------------------------------------------------
        p_df = distribute_tokens(p_df, r_df)

        power_dfs.append(p_df)
        result_dfs.append(r_df)

        meta.append(
            dict(
                model=r_df["Model"].iloc[0],
                tp=int(r_df["Tensor Parallel Size"].iloc[0]),
                rate=float(r_df["Poisson Arrival Rate"].iloc[0]),
            )
        )

    return power_dfs, result_dfs, meta


def stack_and_save(
    power_dfs: list[pd.DataFrame],
    result_dfs: list[pd.DataFrame],
    out_path: str,
):
    """Pad to equal length, stack, and dump *.npz*."""
    max_len = max(len(df) for df in power_dfs)

    def _pad(a, fill=0.0):
        return np.pad(a, (0, max_len - len(a)), constant_values=fill)

    power = np.vstack(
        [_pad(df["power"].values, df["power"].values[-1]) for df in power_dfs]
    )
    pre = np.vstack([_pad(df["prefill_tokens"].values) for df in power_dfs])
    dec = np.vstack([_pad(df["decode_tokens"].values) for df in power_dfs])

    tp = np.array([df["Tensor Parallel Size"].iloc[0] for df in result_dfs])
    rate = np.array([df["Poisson Arrival Rate"].iloc[0] for df in result_dfs])
    msize = np.array([df["Model Size"].iloc[0] for df in result_dfs])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        power_traces=power.astype(np.float32),
        prefill_tokens=pre.astype(np.float32),
        decode_tokens=dec.astype(np.float32),
        tensor_parallelism=tp.astype(np.int32),
        poisson_rate=rate.astype(np.float32),
        model_sizes=msize.astype(np.float32),
    )
    print(f"[OK] saved {out_path}")


# ---------------------------------------------------------------------------
# 5.  CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root", required=True, help="root directory to search for CSVs"
    )
    ap.add_argument(
        "--out",
        default="processed_data/power_trace_data.npz",
        help="output *.npz* path",
    )
    ap.add_argument(
        "--exclude", nargs="*", default=[], help="case-insensitive substrings to skip"
    )
    args = ap.parse_args()

    power_dfs, result_dfs, _ = load_and_process(args.data_root, exclude=args.exclude)
    if not power_dfs:
        print("No experiment pairs processed – nothing to save.", file=sys.stderr)
        sys.exit(1)

    stack_and_save(power_dfs, result_dfs, args.out)


if __name__ == "__main__":
    main()
