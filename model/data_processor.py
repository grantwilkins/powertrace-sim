import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


def discover_experiment_pairs(data_root_dir: str) -> List[Tuple[str, str]]:
    """
    Match power-trace CSVs to results CSVs by
      {MODEL}_tp{TP}_p{RATE}_d{DATE}.csv             (power)
      results_{MODEL}_{RATE}_{TP}_d{DATE}_final.csv  (results)

    Returns
    -------
    list[(power_path, results_path)]
    """
    all_csvs = glob.glob(os.path.join(data_root_dir, "**", "*.csv"), recursive=True)
    power_files, results_files = [], []

    for p in all_csvs:
        base = os.path.basename(p)
        if base.startswith("results_"):
            results_files.append(p)
        elif "_tp" in base and "_p" in base:
            power_files.append(p)

    def pinfo(fname: str):
        m = re.match(r"(.*)_tp(\d+)_p([\d\.]+)_d(.*)\.csv$", Path(fname).name)
        return None if m is None else m.groups()  # (model, tp, rate, date)

    def rinfo(fname: str):
        # allow files w/ or w/o explicit date stamp
        pat1 = r"results_(.*)_([\d\.]+)_(\d+)_d(.*)_final\.csv$"
        pat2 = r"results_(.*)_([\d\.]+)_(\d+)_final\.csv$"
        for pat in (pat1, pat2):
            m = re.match(pat, Path(fname).name)
            if m:
                groups = m.groups()
                return groups if len(groups) == 4 else (*groups, None)
        return None

    pairs: list[tuple[str, str]] = []
    for pfile in power_files:
        p_model, p_tp, p_rate, p_date = pinfo(pfile) or (None, None, None, None)
        for rfile in results_files:
            rm = rinfo(rfile)
            if not rm:
                continue
            r_model, r_rate, r_tp, r_date = rm
            if (p_model, p_tp, p_rate) == (r_model, r_tp, r_rate):
                # if both have dates, they must line up
                if p_date and r_date and p_date != r_date:
                    continue
                pairs.append((pfile, rfile))
                break

    return pairs


def parse_results_csv(csv_path: str) -> pd.DataFrame:
    """
    Parse results CSV and create:
      - Request Time  (datetime64)
      - Completion Time (datetime64)
      - model-size integer in B
    Handles both epoch-second floats and ISO timestamps.
    """
    df = pd.read_csv(csv_path)

    def to_datetime(col: pd.Series) -> pd.Series:
        if np.issubdtype(col.dtype, np.number):
            return pd.to_datetime(col, unit="s")
        return pd.to_datetime(col)  # assume ISO / RFC3339

    df["Request Time"] = to_datetime(df["Request Time"])
    if "E2E Latency" not in df.columns:
        raise ValueError(f"{csv_path} missing 'E2E Latency'")
    df["Completion Time"] = df["Request Time"] + pd.to_timedelta(
        df["E2E Latency"], unit="s"
    )

    def _size(model_str: str) -> int:
        m = re.search(r"(\d+(?:\.\d+)?)B", model_str.lower())  # e.g. 70B or 8B
        return int(float(m.group(1))) if m else np.nan

    df["Model Size"] = df["Model"].apply(_size)
    df.sort_values("Request Time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def parse_power_csv(csv_path: str) -> pd.DataFrame:
    """
    Read nvidia-smi power log, down-sample 8-row groups → one sample,
    and sum wattage across the first *tensor_parallelism* GPUs.
    """
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]

    if "power.draw [w]" in df.columns:
        df.rename(columns={"power.draw [w]": "power"}, inplace=True)
    if "memory.used [mib]" in df.columns:
        df.rename(columns={"memory.used [mib]": "memory"}, inplace=True)

    df["power"] = (
        df["power"].replace(r"[^\d.]", "", regex=True).astype(float)
        if df["power"].dtype == object
        else df["power"].astype(float)
    )

    # timestamp column (first containing 'time')
    tcol = next((c for c in df.columns if "time" in c), None)
    if tcol is None:
        raise ValueError(f"{csv_path}: no timestamp column detected")
    df[tcol] = pd.to_datetime(df[tcol])

    tp = int(re.search(r"_tp(\d+)", csv_path).group(1))
    # down-sample 8 rows (8 GPUs per node)
    groups = df.iloc[: len(df) // 8 * 8].groupby(np.arange(len(df)) // 8)
    out = (
        groups.apply(
            lambda g: pd.Series(
                {"timestamp": g[tcol].min(), "power": g.iloc[:tp]["power"].sum()}
            )
        )
        .reset_index(drop=True)
        .assign(timestamp=lambda d: d["timestamp"].view("int64") / 1e9)
    )
    return out


def distribute_tokens(
    power_df: pd.DataFrame,
    results_df: pd.DataFrame,
    *,
    window_s: float = 0.25,
) -> pd.DataFrame:
    start = results_df["Request Time"].min()

    r = results_df.copy()
    r["start_s"] = (r["Request Time"] - start).dt.total_seconds()
    r["end_s"] = (r["Completion Time"] - start).dt.total_seconds()

    using_prefill = {"Prefill Tokens", "Decode Tokens"} <= set(r.columns)

    # initialise output columns
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


def load_and_process(
    data_root: str, *, exclude: Optional[list[str]] = None
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[dict]]:
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

        p_df = p_df.loc[
            (p_df["timestamp"] >= r_df["Request Time"].min().timestamp())
            & (p_df["timestamp"] <= r_df["Completion Time"].max().timestamp())
        ].reset_index(drop=True)
        p_df["timestamp"] -= p_df["timestamp"].iloc[0]

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
    max_len = max(len(df) for df in power_dfs)

    def pad(a, fill=0.0):
        return np.pad(a, (0, max_len - len(a)), "constant", constant_values=fill)

    power = np.vstack(
        [pad(df["power"].values, df["power"].values[-1]) for df in power_dfs]
    )
    pre = np.vstack([pad(df["prefill_tokens"].values) for df in power_dfs])
    dec = np.vstack([pad(df["decode_tokens"].values) for df in power_dfs])

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root", required=True, help="root directory to search for CSVs"
    )
    ap.add_argument("--out", default="processed_data/power_trace_data.npz")
    ap.add_argument(
        "--exclude", nargs="*", default=[], help="case-insensitive substrings"
    )
    args = ap.parse_args()

    power_dfs, result_dfs, meta = load_and_process(args.data_root, exclude=args.exclude)
    if not power_dfs:
        print("No experiment pairs processed – nothing to save.", file=sys.stderr)
        sys.exit(1)

    stack_and_save(power_dfs, result_dfs, args.out)


if __name__ == "__main__":
    main()
