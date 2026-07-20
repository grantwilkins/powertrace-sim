"""Reproduce the large-checkpoint rate-4 timing diagnostics.

The report separates request concurrency from token-latency interference and
conditions each measured/predicted ITL on arrivals in the preceding window.

Usage:
    uv run python timing-test/rate4_diagnostic.py \
        --fitted /tmp/powertrace_fitted_efficiencies_v3.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent))

from evaluate_timing import MAX_NUM_SEQS  # noqa: E402
from scheduler_sim import EngineConfig  # noqa: E402
from simulated_ledger import simulate_run  # noqa: E402


def overlap_concurrency(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Mean concurrent decode population over each positive decode interval."""
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    duration = ends - starts
    overlap = np.clip(
        np.minimum(ends[:, None], ends[None, :])
        - np.maximum(starts[:, None], starts[None, :]),
        0.0,
        None,
    )
    return np.divide(
        overlap.sum(axis=1),
        duration,
        out=np.full(duration.shape, np.nan),
        where=duration > 0.0,
    )


def condition_itls(
    arrivals: np.ndarray,
    first_token_times: np.ndarray,
    itl_lists: list[np.ndarray],
    *,
    window_s: float,
) -> dict[str, np.ndarray]:
    """Group ITLs by arrivals preceding each interval's start.

    Measured and predicted token clocks are classified independently. This is
    a retrospective association diagnostic, not paired causal evidence that a
    particular prompt was admitted into the same engine iteration.
    """
    arrivals = np.sort(np.asarray(arrivals, dtype=np.float64))
    grouped = {"0": [], "1": [], "2+": []}
    for first, itls in zip(first_token_times, itl_lists):
        values = np.asarray(itls, dtype=np.float64)
        starts = float(first) + np.r_[0.0, np.cumsum(values[:-1])]
        recent = (
            np.searchsorted(arrivals, starts, side="left")
            - np.searchsorted(arrivals, starts - window_s, side="left")
        )
        for count, value in zip(recent, values):
            grouped["0" if count == 0 else "1" if count == 1 else "2+"].append(
                float(value)
            )
    return {key: np.asarray(values, dtype=np.float64) for key, values in grouped.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_identity(model: str) -> dict:
    paths = [
        Path(__file__).resolve(),
        BASE / "simulated_ledger.py",
        BASE / "scheduler_sim.py",
        BASE / "iteration_time.py",
        BASE / "evaluate_timing.py",
        BASE.parent / "model" / "training_data" / "arch.py",
    ]
    engine = EngineConfig(max_num_seqs=MAX_NUM_SEQS.get(model, 256))
    return {
        "code_sha256": {
            str(path.relative_to(BASE.parent)): _sha256(path) for path in paths
        },
        "engine_config": {
            "max_num_seqs": engine.max_num_seqs,
            "chunk_budget_tokens": engine.chunk_budget_tokens,
            "kv_capacity_tokens": engine.kv_capacity_tokens,
            "gpu_memory_utilization": engine.gpu_memory_utilization,
        },
    }


def _quantiles(values: np.ndarray, qs=(0.5, 0.9, 0.99)) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        f"p{int(100 * q)}": float(np.quantile(finite, q))
        for q in qs
    }


def _itl_summary(grouped: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "count": int(values.size),
            "mean_ms": float(1e3 * values.mean()) if values.size else float("nan"),
        }
        for key, values in grouped.items()
    }


def _measured_run(data: dict, rid: int) -> tuple[np.ndarray, ...]:
    idx = np.flatnonzero(data["req_run_id"] == rid)
    order = idx[np.argsort(data["arrival_time_s"][idx], kind="stable")]
    arrivals = data["arrival_time_s"][order].astype(np.float64)
    first = arrivals + data["ttft_s"][order].astype(np.float64)
    ends = first + data["decode_duration_s"][order].astype(np.float64)
    itls = [
        data["itl_values"][
            int(data["itl_offsets"][i]):int(data["itl_offsets"][i + 1])
        ].astype(np.float64)
        for i in order
    ]
    return order, arrivals, first, ends, itls


def build_report(
    data: dict,
    fitted: dict,
    *,
    model: str,
    hardware: str,
    tp: int,
    window_s: float,
) -> dict:
    cells = []
    for rate in (1.0, 2.0, 4.0):
        run_ids = [
            rid
            for rid in range(len(data["run_model"]))
            if str(data["run_model"][rid]) == model
            and str(data["run_hardware"][rid]) == hardware
            and int(data["run_tp"][rid]) == tp
            and float(data["run_rate"][rid]) == rate
        ]
        measured_groups = {"0": [], "1": [], "2+": []}
        predicted_groups = {"0": [], "1": [], "2+": []}
        measured_all = []
        predicted_all = []
        runs = []
        for rid in run_ids:
            order, arrivals, first, ends, measured_itls = _measured_run(data, rid)
            _, predicted, _, _, _ = simulate_run(data, rid, fitted)
            pred_first_raw = arrivals + np.asarray(
                [row["ttft_s"] for row in predicted], dtype=np.float64
            )
            first_token_overhead = float(fitted[hardware]["first_token_overhead_s"])
            pred_first = pred_first_raw + first_token_overhead
            pred_ends = arrivals + first_token_overhead + np.asarray(
                [row["e2e_s"] for row in predicted], dtype=np.float64
            )
            predicted_itls = [
                np.asarray(row["itl_s"], dtype=np.float64) for row in predicted
            ]
            measured_conditioned = condition_itls(
                arrivals, first, measured_itls, window_s=window_s
            )
            predicted_conditioned = condition_itls(
                arrivals, pred_first_raw, predicted_itls, window_s=window_s
            )
            for key in measured_groups:
                measured_groups[key].append(measured_conditioned[key])
                predicted_groups[key].append(predicted_conditioned[key])
            measured_all.extend(measured_itls)
            predicted_all.extend(predicted_itls)

            measured_e2e = (
                data["ttft_s"][order] + data["decode_duration_s"][order]
            ).astype(np.float64)
            predicted_e2e = first_token_overhead + np.asarray(
                [row["e2e_s"] for row in predicted], dtype=np.float64
            )
            signed_pct = 100.0 * (predicted_e2e - measured_e2e) / measured_e2e
            runs.append(
                {
                    "run_id": rid,
                    "requests": int(order.size),
                    "measured_decode_concurrency": _quantiles(
                        overlap_concurrency(first, ends)
                    ),
                    "predicted_decode_concurrency": _quantiles(
                        overlap_concurrency(pred_first, pred_ends)
                    ),
                    "e2e_median_abs_pct": float(np.median(np.abs(signed_pct))),
                    "e2e_median_signed_pct": float(np.median(signed_pct)),
                }
            )

        measured_group_arrays = {
            key: np.concatenate(values) for key, values in measured_groups.items()
        }
        predicted_group_arrays = {
            key: np.concatenate(values) for key, values in predicted_groups.items()
        }
        measured_values = np.concatenate(measured_all)
        predicted_values = np.concatenate(predicted_all)
        cells.append(
            {
                "rate": rate,
                "runs": runs,
                "measured_itl_ms": {
                    key: 1e3 * value for key, value in _quantiles(
                        measured_values, (0.5, 0.95, 0.99)
                    ).items()
                },
                "predicted_itl_ms": {
                    key: 1e3 * value for key, value in _quantiles(
                        predicted_values, (0.5, 0.95, 0.99)
                    ).items()
                },
                "measured_conditioned_itl": _itl_summary(measured_group_arrays),
                "predicted_conditioned_itl": _itl_summary(predicted_group_arrays),
            }
        )
    return {
        "model": model,
        "hardware": hardware,
        "tp": tp,
        "arrival_window_s": window_s,
        "conditioning": (
            "arrivals in [ITL start - window, ITL start); measured and "
            "predicted raw-engine token clocks classified independently; "
            "formal E2E and concurrency include first-token overhead"
        ),
        "cells": cells,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=BASE / "timing_dataset.npz"
    )
    parser.add_argument(
        "--fitted", type=Path, default=BASE / "fitted_efficiencies.json"
    )
    parser.add_argument("--model", default="llama-3-405b")
    parser.add_argument("--hardware", default="H100")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--window-ms", type=float, default=100.0)
    args = parser.parse_args(argv)

    data = dict(np.load(args.dataset, allow_pickle=False))
    fitted = json.loads(args.fitted.read_text())
    report = build_report(
        data,
        fitted,
        model=args.model,
        hardware=args.hardware,
        tp=args.tp,
        window_s=args.window_ms / 1000.0,
    )
    report["inputs"] = {
        "dataset": str(args.dataset),
        "dataset_sha256": _sha256(args.dataset),
        "fitted": str(args.fitted),
        "fitted_sha256": _sha256(args.fitted),
        **_runtime_identity(args.model),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
