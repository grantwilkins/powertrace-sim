"""Plot the held-out disaggregated prefill and decode power traces."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_phase_traces(
    trace_rows: list[dict],
    cell_metrics: list[dict],
    phase_calibration: dict[str, dict[str, object]],
    output: Path,
    *,
    cell: str,
) -> None:
    traces = [row for row in trace_rows if row["cell"] == cell]
    metrics = next(
        row for row in cell_metrics
        if row["cell"] == cell
        and row["candidate"] == "phase_soft_dtw_calibrated"
    )
    if not traces:
        raise ValueError(f"no representative trace rows for {cell}")
    time_s = np.asarray([row["time_s"] for row in traces])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for axis, role, title in zip(
        axes, ("prefill", "decode"), ("Prefill GPU", "Decode GPU")
    ):
        measured = np.asarray([row[f"measured_{role}_w"] for row in traces])
        predicted = np.asarray([row[f"predicted_{role}_w"] for row in traces])
        correlation = float(np.corrcoef(measured, predicted)[0, 1])
        axis.plot(
            time_s, measured, color="#222222", linewidth=0.65,
            label="Measured 250 ms sample",
        )
        axis.plot(
            time_s, predicted, color="#D55E00", linewidth=0.7,
            label="PowerTrace 250 ms bin",
        )
        diagonal = metrics[f"{role}_trace_soft_dtw_diagonal_divergence"]
        scale = float(phase_calibration[role]["accepted_scale"])
        calibration_label = (
            "no active gain" if scale == 1.0 else f"{scale:.3f}× dynamic gain"
        )
        axis.set_title(
            f"{title} — {calibration_label}\n"
            f"250 ms diagonal soft-DTW {diagonal:.4f}; r={correlation:.3f}"
        )
        axis.set_xlabel("Time since cell start (s)")
        axis.set_ylabel("GPU power (W)")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Disaggregated phase power: held-out 4 requests/s, repeat 2 "
        "(native 250 ms)",
        fontsize=14,
    )
    fig.text(
        0.5, 0.01,
        "Recorded samples only: no smoothing, interpolation, or temporal "
        "warping. Decoder gain uses only 2 requests/s repeat 1.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(output, dpi=180)
    plt.close(fig)
