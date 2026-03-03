#!/usr/bin/env python3
"""
Appendix B1: Interconnection-planning ramp-rate CDF figure.

Builds a single-panel 15-minute ramp-magnitude CDF for Ours vs Splitwise by
pooling ramps across precomputed seed exports from run_baselines_facility.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OURS_METHOD = "ours"
SPLITWISE_METHOD = "splitwise_lut"
REQUIRED_METHODS = (OURS_METHOD, SPLITWISE_METHOD)
OPTIONAL_ZERO_METHODS = ("tdp", "mean")

COLOR_OURS = "#1f77b4"
COLOR_SPLITWISE = "#ff7f0e"
COLOR_GRAY = "#7f8c8d"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, payload: Mapping[str, object]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _resolve_path(raw_path: str, base_dir: str) -> str:
    p = Path(raw_path)
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        return str(p)
    local = Path(base_dir) / p
    if not local.exists():
        raise FileNotFoundError(f"Missing file: {local}")
    return str(local.resolve())


def _ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    n = int(x.size)
    if n <= 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    y = np.arange(1, n + 1, dtype=np.float64) / float(n)
    return x, y


def _downsample_non_overlapping_mean(values: np.ndarray, factor: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if int(factor) <= 0:
        raise ValueError("factor must be >= 1")
    if arr.size <= 0:
        raise ValueError("cannot downsample an empty array")
    if int(arr.size) % int(factor) != 0:
        raise ValueError(
            f"array length {arr.size} is not divisible by downsample factor {factor}"
        )
    return np.mean(arr.reshape(-1, int(factor)), axis=1).astype(np.float64)


def _list_seed_dirs(seed_trace_root: str, seed_glob: str) -> List[Path]:
    root = Path(seed_trace_root).resolve()
    if not root.exists() or (not root.is_dir()):
        raise FileNotFoundError(f"Seed trace root not found or not a directory: {root}")
    dirs = sorted([p for p in root.glob(seed_glob) if p.is_dir()])
    return dirs


def _load_seed_manifest(seed_dir: Path) -> Dict[str, object]:
    manifest_path = seed_dir / "facility_trace_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing facility_trace_manifest.json in {seed_dir}")
    payload = _load_json(str(manifest_path))
    return payload


def _safe_float(value: object, field: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"Unable to parse float for {field}: {value}") from exc
    if not np.isfinite(out):
        raise ValueError(f"Non-finite float for {field}: {value}")
    return float(out)


def _safe_int(value: object, field: str) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"Unable to parse int for {field}: {value}") from exc
    return int(out)


def _extract_shared_config(seed_payloads: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if len(seed_payloads) <= 0:
        raise ValueError("seed payload list is empty")

    first = seed_payloads[0]
    config_id = str(first.get("config_id", "")).strip()
    dt = _safe_float(first.get("dt", ""), "dt")
    duration_s = _safe_float(first.get("duration_s", ""), "duration_s")
    n_nodes = _safe_int(first.get("n_nodes", ""), "n_nodes")

    for idx, payload in enumerate(seed_payloads[1:], start=1):
        cfg = str(payload.get("config_id", "")).strip()
        if cfg != config_id:
            raise ValueError(
                f"Inconsistent config_id across seeds: seed0={config_id}, seed{idx}={cfg}"
            )
        dt_i = _safe_float(payload.get("dt", ""), "dt")
        dur_i = _safe_float(payload.get("duration_s", ""), "duration_s")
        n_nodes_i = _safe_int(payload.get("n_nodes", ""), "n_nodes")
        if not np.isclose(dt_i, dt, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Inconsistent dt across seeds: seed0={dt}, seed{idx}={dt_i}"
            )
        if not np.isclose(dur_i, duration_s, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"Inconsistent duration_s across seeds: seed0={duration_s}, seed{idx}={dur_i}"
            )
        if int(n_nodes_i) != int(n_nodes):
            raise ValueError(
                f"Inconsistent n_nodes across seeds: seed0={n_nodes}, seed{idx}={n_nodes_i}"
            )
    return {
        "config_id": config_id,
        "dt": float(dt),
        "duration_s": float(duration_s),
        "n_nodes": int(n_nodes),
    }


def _load_seed_method_trace_w(
    *,
    seed_dir: Path,
    payload: Dict[str, object],
    method: str,
) -> np.ndarray:
    files_raw = payload.get("files", {})
    if not isinstance(files_raw, dict):
        raise ValueError(f"Invalid files map in manifest: {seed_dir}")
    file_raw = files_raw.get(method)
    if file_raw is None:
        raise FileNotFoundError(
            f"Missing method '{method}' trace in manifest for seed dir {seed_dir}"
        )
    path = _resolve_path(str(file_raw), base_dir=str(seed_dir))
    arr = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if arr.size <= 1:
        raise ValueError(f"Method '{method}' trace is too short in {path}")
    return arr


def _compute_abs_15min_ramps_mw(values_w: np.ndarray, dt: float) -> np.ndarray:
    dt_s = float(dt)
    if dt_s <= 0.0:
        raise ValueError(f"Invalid dt={dt_s}")
    factor_float = 900.0 / dt_s
    factor = int(round(factor_float))
    if not np.isclose(float(factor), float(factor_float), rtol=0.0, atol=1e-9):
        raise ValueError(
            f"dt={dt_s} does not map cleanly to 15-minute bins (factor={factor_float})."
        )
    series_15min_w = _downsample_non_overlapping_mean(values=values_w, factor=factor)
    ramps_mw = np.diff(series_15min_w) / 1_000_000.0
    return np.abs(np.asarray(ramps_mw, dtype=np.float64).reshape(-1))


def _build_default_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "seed_trace_root": str(repo_root / "results" / "eval_paper" / "appendix_b1_seeds"),
        "out_figure_pdf": str(repo_root / "figures" / "appendix_b1_ramp_rate_cdf_15min.pdf"),
        "out_cdf_csv": str(repo_root / "results" / "eval_paper" / "appendix_b1_ramp_cdf_points.csv"),
        "out_summary_csv": str(repo_root / "results" / "eval_paper" / "appendix_b1_ramp_summary.csv"),
        "out_manifest_json": str(repo_root / "results" / "eval_paper" / "appendix_b1_manifest.json"),
    }


def _plot_ramp_cdf(
    *,
    out_figure_pdf: str,
    ours_x: np.ndarray,
    ours_y: np.ndarray,
    split_x: np.ndarray,
    split_y: np.ndarray,
    p95_ours: float,
    p95_split: float,
) -> None:
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    ax.plot(ours_x, ours_y, color=COLOR_OURS, linestyle="-", linewidth=2.0, label="Ours")
    ax.plot(
        split_x,
        split_y,
        color=COLOR_SPLITWISE,
        linestyle="--",
        linewidth=2.0,
        label="Splitwise",
    )

    x_max = float(max(np.max(ours_x), np.max(split_x))) if (ours_x.size > 0 and split_x.size > 0) else 0.0
    if x_max <= 0.0:
        x_max = 1e-6
    ax.set_xlim(0.0, x_max * 1.02)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("15-minute ramp magnitude |ΔP| (MW)")
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.25)

    tick_dx = max(1e-6, x_max * 0.012)
    for x, color, label in [
        (float(p95_ours), COLOR_OURS, "Ours p95"),
        (float(p95_split), COLOR_SPLITWISE, "Splitwise p95"),
    ]:
        ax.plot([x - tick_dx, x + tick_dx], [0.95, 0.95], color=color, linewidth=2.0)
        ax.text(
            x + (0.6 * tick_dx),
            0.91,
            f"{label}: {x:.4f} MW",
            color=color,
            fontsize=9,
            ha="left",
            va="top",
        )

    ax.axvline(0.0, color=COLOR_GRAY, linestyle=":", linewidth=1.2, alpha=0.9)
    ax.text(
        0.012,
        0.08,
        "TDP/Mean: zero ramp by construction",
        color=COLOR_GRAY,
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
    )
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(out_figure_pdf) or ".")
    fig.savefig(out_figure_pdf, bbox_inches="tight")
    plt.close(fig)


def generate_appendix_interconnection_b1(
    *,
    seed_trace_root: str,
    seed_glob: str = "seed_*",
    expected_seeds: int = 5,
    out_figure_pdf: str,
    out_cdf_csv: str,
    out_summary_csv: str,
    out_manifest_json: str,
    dry_run: bool = False,
) -> Dict[str, object]:
    if int(expected_seeds) <= 0:
        raise ValueError("expected_seeds must be >= 1")

    seed_dirs = _list_seed_dirs(seed_trace_root=seed_trace_root, seed_glob=seed_glob)
    if len(seed_dirs) != int(expected_seeds):
        raise ValueError(
            f"Expected {expected_seeds} seed directories, found {len(seed_dirs)} under {seed_trace_root} with glob '{seed_glob}'."
        )

    payloads = [_load_seed_manifest(seed_dir) for seed_dir in seed_dirs]
    shared = _extract_shared_config(payloads)

    pooled_abs_ramps: Dict[str, List[np.ndarray]] = {k: [] for k in REQUIRED_METHODS}
    zero_ramp_max_by_method: Dict[str, float] = {}
    n_samples_per_seed: List[int] = []

    for seed_dir, payload in zip(seed_dirs, payloads):
        for method in REQUIRED_METHODS:
            series_w = _load_seed_method_trace_w(
                seed_dir=seed_dir, payload=payload, method=method
            )
            n_samples_per_seed.append(int(series_w.size))
            abs_ramps = _compute_abs_15min_ramps_mw(series_w, dt=float(shared["dt"]))
            pooled_abs_ramps[method].append(abs_ramps)

        for method in OPTIONAL_ZERO_METHODS:
            files_map = payload.get("files", {})
            if not isinstance(files_map, dict):
                continue
            if method not in files_map:
                continue
            zero_series_w = _load_seed_method_trace_w(
                seed_dir=seed_dir,
                payload=payload,
                method=method,
            )
            abs_ramps = _compute_abs_15min_ramps_mw(
                zero_series_w, dt=float(shared["dt"])
            )
            prior = zero_ramp_max_by_method.get(method, 0.0)
            zero_ramp_max_by_method[method] = float(
                max(float(prior), float(np.max(abs_ramps)) if abs_ramps.size > 0 else 0.0)
            )

    pooled = {
        k: np.concatenate(v, axis=0).astype(np.float64)
        for k, v in pooled_abs_ramps.items()
    }
    for method in REQUIRED_METHODS:
        if pooled[method].size <= 0:
            raise ValueError(f"No pooled ramps for required method '{method}'")

    ours_x, ours_y = _ecdf(pooled[OURS_METHOD])
    split_x, split_y = _ecdf(pooled[SPLITWISE_METHOD])
    p95_ours = float(np.percentile(pooled[OURS_METHOD], 95))
    p95_split = float(np.percentile(pooled[SPLITWISE_METHOD], 95))

    cdf_rows: List[Dict[str, object]] = []
    for x, y in zip(ours_x, ours_y):
        cdf_rows.append(
            {
                "method": OURS_METHOD,
                "series_label": "Ours",
                "ramp_mw": float(x),
                "cdf": float(y),
            }
        )
    for x, y in zip(split_x, split_y):
        cdf_rows.append(
            {
                "method": SPLITWISE_METHOD,
                "series_label": "Splitwise",
                "ramp_mw": float(x),
                "cdf": float(y),
            }
        )

    summary_rows: List[Dict[str, object]] = []
    for method, label in [(OURS_METHOD, "Ours"), (SPLITWISE_METHOD, "Splitwise")]:
        arr = pooled[method]
        summary_rows.append(
            {
                "method": method,
                "series_label": label,
                "n_seeds": int(len(seed_dirs)),
                "n_ramps": int(arr.size),
                "p50_ramp_mw": float(np.percentile(arr, 50)),
                "p95_ramp_mw": float(np.percentile(arr, 95)),
                "p99_ramp_mw": float(np.percentile(arr, 99)),
                "max_ramp_mw": float(np.max(arr)),
            }
        )
    for method in OPTIONAL_ZERO_METHODS:
        if method in zero_ramp_max_by_method:
            summary_rows.append(
                {
                    "method": method,
                    "series_label": ("TDP" if method == "tdp" else "Mean"),
                    "n_seeds": int(len(seed_dirs)),
                    "n_ramps": int(0),
                    "p50_ramp_mw": float(0.0),
                    "p95_ramp_mw": float(0.0),
                    "p99_ramp_mw": float(0.0),
                    "max_ramp_mw": float(zero_ramp_max_by_method[method]),
                }
            )

    if not bool(dry_run):
        _plot_ramp_cdf(
            out_figure_pdf=out_figure_pdf,
            ours_x=ours_x,
            ours_y=ours_y,
            split_x=split_x,
            split_y=split_y,
            p95_ours=p95_ours,
            p95_split=p95_split,
        )
        _write_csv(
            out_cdf_csv,
            cdf_rows,
            fieldnames=["method", "series_label", "ramp_mw", "cdf"],
        )
        _write_csv(
            out_summary_csv,
            summary_rows,
            fieldnames=[
                "method",
                "series_label",
                "n_seeds",
                "n_ramps",
                "p50_ramp_mw",
                "p95_ramp_mw",
                "p99_ramp_mw",
                "max_ramp_mw",
            ],
        )

    manifest = {
        "schema_version": "appendix-b1-ramp-cdf-v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "seed_trace_root": str(Path(seed_trace_root).resolve()),
            "seed_glob": str(seed_glob),
            "expected_seeds": int(expected_seeds),
            "seed_dirs": [str(p.resolve()) for p in seed_dirs],
        },
        "config": {
            "config_id": str(shared["config_id"]),
            "dt": float(shared["dt"]),
            "duration_s": float(shared["duration_s"]),
            "n_nodes": int(shared["n_nodes"]),
            "ramp_basis": "absolute_15min_magnitude_mw",
        },
        "checks": {
            "n_samples_per_seed_min": int(min(n_samples_per_seed)),
            "n_samples_per_seed_max": int(max(n_samples_per_seed)),
            "zero_ramp_max_mw": {
                key: float(val) for key, val in zero_ramp_max_by_method.items()
            },
        },
        "stats": {
            "ours": {
                "n_ramps": int(pooled[OURS_METHOD].size),
                "p95_ramp_mw": float(p95_ours),
            },
            "splitwise_lut": {
                "n_ramps": int(pooled[SPLITWISE_METHOD].size),
                "p95_ramp_mw": float(p95_split),
            },
        },
        "output_paths": {
            "figure_pdf": str(out_figure_pdf),
            "cdf_csv": str(out_cdf_csv),
            "summary_csv": str(out_summary_csv),
            "manifest_json": str(out_manifest_json),
        },
        "dry_run": bool(dry_run),
    }
    _write_json(out_manifest_json, manifest)
    return manifest


def main() -> None:
    defaults = _build_default_paths()
    parser = argparse.ArgumentParser(
        description="Appendix B1: 15-minute ramp-magnitude CDF for Ours vs Splitwise."
    )
    parser.add_argument("--seed-trace-root", default=defaults["seed_trace_root"])
    parser.add_argument("--seed-glob", default="seed_*")
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--out-figure-pdf", default=defaults["out_figure_pdf"])
    parser.add_argument("--out-cdf-csv", default=defaults["out_cdf_csv"])
    parser.add_argument("--out-summary-csv", default=defaults["out_summary_csv"])
    parser.add_argument("--out-manifest-json", default=defaults["out_manifest_json"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run = generate_appendix_interconnection_b1(
        seed_trace_root=str(args.seed_trace_root),
        seed_glob=str(args.seed_glob),
        expected_seeds=int(args.expected_seeds),
        out_figure_pdf=str(args.out_figure_pdf),
        out_cdf_csv=str(args.out_cdf_csv),
        out_summary_csv=str(args.out_summary_csv),
        out_manifest_json=str(args.out_manifest_json),
        dry_run=bool(args.dry_run),
    )

    if bool(args.dry_run):
        print("[appendix_interconnection_b1] Dry run complete")
    else:
        print("[appendix_interconnection_b1] Done")
    print(f"  manifest: {run['output_paths']['manifest_json']}")
    if not bool(args.dry_run):
        print(f"  figure  : {run['output_paths']['figure_pdf']}")
        print(f"  cdf_csv : {run['output_paths']['cdf_csv']}")
        print(f"  summary : {run['output_paths']['summary_csv']}")


if __name__ == "__main__":
    main()
