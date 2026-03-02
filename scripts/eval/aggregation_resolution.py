#!/usr/bin/env python3
"""
Experiment 3b: Resolution-dependent PAR for Azure site traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESOLUTIONS: Tuple[float, ...] = (0.25, 1.0, 10.0, 60.0, 300.0, 900.0)
SUPPORTED_RESOLUTIONS = set(DEFAULT_RESOLUTIONS)


def _ensure_dir_for_file(path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Union[str, Path], payload: Mapping[str, object]) -> None:
    _ensure_dir_for_file(path)
    with open(path, "w") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True)


def _build_default_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "aggregated_dir": str(repo_root / "results" / "azure_facility" / "aggregated"),
        "out_plot": str(repo_root / "figures" / "azure_par_vs_resolution.pdf"),
        "out_csv": str(repo_root / "results" / "eval_paper" / "azure_par_vs_resolution.csv"),
        "out_json": str(repo_root / "results" / "eval_paper" / "azure_par_vs_resolution.json"),
    }


def _parse_resolutions(raw: str) -> Tuple[float, ...]:
    out: List[float] = []
    for token in str(raw).split(","):
        t = token.strip()
        if t == "":
            continue
        try:
            value = float(t)
        except Exception as exc:
            raise ValueError(f"Invalid resolution token: '{t}'") from exc
        out.append(float(value))
    if len(out) == 0:
        raise ValueError("resolutions cannot be empty")
    if len(set(out)) != len(out):
        raise ValueError(f"resolutions must be unique, got: {out}")
    if any(float(x) <= 0.0 for x in out):
        raise ValueError(f"resolutions must be positive, got: {out}")
    return tuple(float(x) for x in out)


def _load_array(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required array not found: {path}")
    arr = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        raise ValueError(f"Empty array: {path}")
    return arr


def _downsample_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    f = int(factor)
    if f <= 0:
        raise ValueError("downsample factor must be >= 1")
    if x.size % f != 0:
        raise ValueError(f"Array length {x.size} not divisible by factor {f}")
    return np.mean(x.reshape(-1, f), axis=1).astype(np.float64)


def _compute_par(arr: np.ndarray) -> Dict[str, float]:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    if x.size <= 0:
        raise ValueError("Cannot compute PAR on empty array")
    mean = float(np.mean(x))
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError(f"Mean power must be positive and finite, got {mean}")
    peak = float(np.max(x))
    par = float(peak / mean)
    return {
        "n_samples": int(x.size),
        "peak_w": float(peak),
        "mean_w": float(mean),
        "par": float(par),
    }


def compute_resolution_par(
    *,
    aggregated_dir: str,
    out_plot: str,
    out_csv: str,
    out_json: str,
    resolutions: Sequence[float] = DEFAULT_RESOLUTIONS,
    allow_nonmonotonic: bool = False,
) -> Dict[str, object]:
    res = tuple(float(x) for x in resolutions)
    if len(res) == 0:
        raise ValueError("resolutions must be non-empty")
    if len(set(res)) != len(res):
        raise ValueError(f"resolutions must be unique, got: {res}")
    if any(float(x) <= 0.0 for x in res):
        raise ValueError(f"resolutions must be positive, got: {res}")
    unsupported = [float(x) for x in res if float(x) not in SUPPORTED_RESOLUTIONS]
    if unsupported:
        raise ValueError(
            f"Unsupported resolutions requested: {unsupported}. Supported: {sorted(SUPPORTED_RESOLUTIONS)}"
        )

    need_250 = 0.25 in set(res)
    need_1s = any(float(x) in {1.0, 10.0, 300.0} for x in res)
    need_60 = 60.0 in set(res)
    need_900 = 900.0 in set(res)

    site_250 = _load_array(os.path.join(aggregated_dir, "site_250ms.npy")) if need_250 else None
    site_1s = _load_array(os.path.join(aggregated_dir, "site_1s.npy")) if need_1s else None
    site_60 = _load_array(os.path.join(aggregated_dir, "site_1min.npy")) if need_60 else None
    site_900 = _load_array(os.path.join(aggregated_dir, "site_15min.npy")) if need_900 else None

    by_resolution: Dict[float, np.ndarray] = {}
    for r in res:
        if float(r) == 0.25:
            assert site_250 is not None
            by_resolution[float(r)] = np.asarray(site_250, dtype=np.float64)
        elif float(r) == 1.0:
            assert site_1s is not None
            by_resolution[float(r)] = np.asarray(site_1s, dtype=np.float64)
        elif float(r) == 10.0:
            assert site_1s is not None
            by_resolution[float(r)] = _downsample_mean(site_1s, 10)
        elif float(r) == 60.0:
            assert site_60 is not None
            by_resolution[float(r)] = np.asarray(site_60, dtype=np.float64)
        elif float(r) == 300.0:
            assert site_1s is not None
            by_resolution[float(r)] = _downsample_mean(site_1s, 300)
        elif float(r) == 900.0:
            assert site_900 is not None
            by_resolution[float(r)] = np.asarray(site_900, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported resolution: {r}")

    rows: List[Dict[str, float]] = []
    for r in sorted(res):
        m = _compute_par(by_resolution[float(r)])
        rows.append(
            {
                "resolution_s": float(r),
                "n_samples": int(m["n_samples"]),
                "peak_w": float(m["peak_w"]),
                "mean_w": float(m["mean_w"]),
                "par": float(m["par"]),
            }
        )

    violations: List[Dict[str, float]] = []
    monotonic = True
    tol = 1e-12
    for i in range(1, len(rows)):
        prev = rows[i - 1]
        cur = rows[i]
        if float(cur["par"]) > float(prev["par"]) + tol:
            monotonic = False
            violations.append(
                {
                    "from_resolution_s": float(prev["resolution_s"]),
                    "to_resolution_s": float(cur["resolution_s"]),
                    "from_par": float(prev["par"]),
                    "to_par": float(cur["par"]),
                    "delta": float(cur["par"] - prev["par"]),
                }
            )

    if (not monotonic) and (not bool(allow_nonmonotonic)):
        raise ValueError(
            f"PAR is non-monotonic under coarsening. Violations: {violations}"
        )

    _ensure_dir_for_file(out_csv)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["resolution_s", "n_samples", "peak_w", "mean_w", "par"],
        )
        writer.writeheader()
        writer.writerows(rows)

    x = np.asarray([float(r["resolution_s"]) for r in rows], dtype=np.float64)
    y = np.asarray([float(r["par"]) for r in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(
        x,
        y,
        marker="o",
        linewidth=2.0,
        color="#2c3e50",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Resolution (seconds)")
    ax.set_ylabel("PAR (peak/mean)")
    ax.set_title("Resolution-Dependent PAR (Azure 240-node site)")
    ax.grid(True, which="both", linestyle=":", alpha=0.35)

    by_r = {float(r["resolution_s"]): r for r in rows}
    if 1.0 in by_r:
        par_1s = float(by_r[1.0]["par"])
        ax.annotate(
            f"UPS engineer (1s): PAR = {par_1s:.3f}",
            xy=(1.0, par_1s),
            xytext=(10, 12),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "linewidth": 0.8},
            fontsize=9,
        )
    if 900.0 in by_r:
        par_900 = float(by_r[900.0]["par"])
        ax.annotate(
            f"Utility (15min): PAR = {par_900:.3f}",
            xy=(900.0, par_900),
            xytext=(-160, -18),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "linewidth": 0.8},
            fontsize=9,
        )

    _ensure_dir_for_file(out_plot)
    fig.savefig(out_plot, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "status": "ok",
        "aggregated_dir": str(aggregated_dir),
        "resolutions_s": [float(x) for x in sorted(res)],
        "monotonic_nonincreasing": bool(monotonic),
        "allow_nonmonotonic": bool(allow_nonmonotonic),
        "violations": violations,
        "rows": rows,
        "annotations": {
            "ups_1s_par": float(by_r[1.0]["par"]) if 1.0 in by_r else None,
            "utility_15min_par": float(by_r[900.0]["par"]) if 900.0 in by_r else None,
        },
        "outputs": {
            "plot_pdf": str(out_plot),
            "csv": str(out_csv),
            "json": str(out_json),
        },
    }
    _write_json(out_json, payload)
    return payload


def main() -> None:
    defaults = _build_default_paths()
    parser = argparse.ArgumentParser(
        description="Experiment 3b: PAR vs temporal resolution for Azure site trace."
    )
    parser.add_argument("--aggregated-dir", default=defaults["aggregated_dir"])
    parser.add_argument("--out-plot", default=defaults["out_plot"])
    parser.add_argument("--out-csv", default=defaults["out_csv"])
    parser.add_argument("--out-json", default=defaults["out_json"])
    parser.add_argument(
        "--resolutions",
        default="0.25,1,10,60,300,900",
        help="Comma-separated resolutions in seconds.",
    )
    parser.add_argument(
        "--allow-nonmonotonic",
        action="store_true",
        help="Allow non-monotonic PAR under coarsening instead of raising.",
    )
    args = parser.parse_args()

    resolutions = _parse_resolutions(str(args.resolutions))
    result = compute_resolution_par(
        aggregated_dir=str(args.aggregated_dir),
        out_plot=str(args.out_plot),
        out_csv=str(args.out_csv),
        out_json=str(args.out_json),
        resolutions=resolutions,
        allow_nonmonotonic=bool(args.allow_nonmonotonic),
    )

    print("=" * 72)
    print("Experiment 3b: Resolution-Dependent PAR")
    print("=" * 72)
    print(f"Aggregated dir       : {result['aggregated_dir']}")
    print(f"Resolutions (s)      : {result['resolutions_s']}")
    print(f"Monotonic nonincrease: {result['monotonic_nonincreasing']}")
    ann = result["annotations"]
    if ann["ups_1s_par"] is not None:
        print(f"UPS engineer (1s) PAR: {float(ann['ups_1s_par']):.4f}")
    if ann["utility_15min_par"] is not None:
        print(f"Utility (15m) PAR    : {float(ann['utility_15min_par']):.4f}")
    print(f"Figure               : {result['outputs']['plot_pdf']}")
    print(f"CSV                  : {result['outputs']['csv']}")
    print(f"JSON                 : {result['outputs']['json']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
