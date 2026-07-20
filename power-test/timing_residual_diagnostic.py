"""No-dose timing diagnostic for the held-out H100 70B TP8 trace.

The figure removes the same causal 60-second power trend from measured and
predicted power, then compares measured and simulated request state and
completion timing. Engine iteration coordinates are simulation-only.

Usage: uv run python power-test/timing_residual_diagnostic.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from changepoint_ablation import build_records
from fit_power_surface import load_cache

BASE = Path(__file__).resolve().parent
MEASURED_LEDGER = BASE.parent / "feature-test" / "ledger_cache_250ms.npz"
PLOT_PATH = BASE / "h100_tp8_timing_diagnostic.png"
RUN_ID = 375
TREND_S = 60.0
DISPLAY_SMOOTH_S = 5.0


def causal_detrend(values: np.ndarray, window_bins: int) -> np.ndarray:
    """Subtract the trailing mean including the current bin."""
    signal = np.asarray(values, float).reshape(-1)
    if window_bins <= 0:
        raise ValueError("window_bins must be positive")
    sums = np.r_[0.0, np.cumsum(signal)]
    end = np.arange(1, signal.size + 1)
    start = np.maximum(end - window_bins, 0)
    means = (sums[end] - sums[start]) / (end - start)
    return signal - means


def request_completion_rate(
    arrivals_per_s: np.ndarray, delta_active: np.ndarray, dt_s: float
) -> np.ndarray:
    """Request completions/s from arrivals - change in active requests."""
    arrivals = np.asarray(arrivals_per_s, float).reshape(-1)
    delta = np.asarray(delta_active, float).reshape(-1)
    if arrivals.shape != delta.shape:
        raise ValueError("arrivals_per_s and delta_active must align")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    completions = arrivals - delta / dt_s
    if np.any(completions < -1e-9):
        raise ValueError("Derived request completions cannot be negative")
    return completions


def _moving_average(values: np.ndarray, bins: int) -> np.ndarray:
    left = (bins - 1) // 2
    right = bins // 2
    padded = np.pad(np.asarray(values, float), (left, right), mode="edge")
    return np.convolve(padded, np.full(bins, 1.0 / bins), mode="valid")


def _run_data(cache: dict, run_id: int) -> dict:
    selected = cache["run_id"] == run_id
    if not np.any(selected):
        raise ValueError(f"Run {run_id} is absent from the ledger")
    return {
        key: value[selected]
        for key, value in cache.items()
        if isinstance(value, np.ndarray) and value.shape == selected.shape
    }


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def save_plot(path: Path = PLOT_PATH) -> None:
    import matplotlib.pyplot as plt

    simulated_cache = load_cache()
    measured_cache = _load_npz(MEASURED_LEDGER)
    record = next(
        record for record in build_records(simulated_cache)
        if record["run_id"] == RUN_ID
    )
    simulated = _run_data(simulated_cache, RUN_ID)
    measured = _run_data(measured_cache, RUN_ID)
    dt_s = float(simulated_cache["dt_s"])
    if float(measured_cache["dt_s"]) != dt_s:
        raise ValueError("Measured and simulated ledger intervals differ")
    n = min(
        len(record["measured"]),
        len(simulated["run_id"]),
        len(measured["run_id"]),
    )
    time_s = np.arange(n) * dt_s
    tp = record["tp"]
    trend_bins = int(round(TREND_S / dt_s))
    smooth_bins = int(round(DISPLAY_SMOOTH_S / dt_s))

    measured_power = causal_detrend(record["measured"][:n] / tp, trend_bins)
    predicted_power = causal_detrend(record["baseline"][:n] / tp, trend_bins)
    measured_completions = request_completion_rate(
        measured["arrivals"][:n], measured["delta_A_t"][:n], dt_s
    )
    simulated_completions = request_completion_rate(
        simulated["arrivals"][:n], simulated["delta_A_t"][:n], dt_s
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(
        time_s, _moving_average(measured_power, smooth_bins),
        color="black", linewidth=1.5, label="Measured power",
    )
    axes[0].plot(
        time_s, _moving_average(predicted_power, smooth_bins),
        color="#0072B2", linewidth=1.5, label="No-dose model",
    )
    axes[0].set_ylabel("Power residual\n(W/GPU)")
    axes[0].set_title(
        "H100 · Llama-3-70B · TP8 · 4 req/s · held-out run 375\n"
        "Power after subtracting a causal 60 s trend"
    )
    axes[0].legend(ncol=2)

    axes[1].plot(
        time_s, _moving_average(measured["running_requests"][:n], smooth_bins),
        color="black", linewidth=1.4, label="Measured running",
    )
    axes[1].plot(
        time_s, _moving_average(simulated["running_requests"][:n], smooth_bins),
        color="#0072B2", linewidth=1.4, label="Simulated running",
    )
    axes[1].plot(
        time_s, _moving_average(measured["waiting_requests"][:n], smooth_bins),
        color="0.35", linestyle=":", linewidth=1.2, label="Measured waiting",
    )
    axes[1].plot(
        time_s, _moving_average(simulated["waiting_requests"][:n], smooth_bins),
        color="#D55E00", linestyle=":", linewidth=1.2,
        label="Simulated waiting",
    )
    axes[1].set_ylabel("Requests")
    axes[1].legend(ncol=4, fontsize=8)

    axes[2].plot(
        time_s, _moving_average(measured_completions, smooth_bins),
        color="black", linewidth=1.4, label="Measured completions",
    )
    axes[2].plot(
        time_s, _moving_average(simulated_completions, smooth_bins),
        color="#0072B2", linewidth=1.4, label="Simulated completions",
    )
    axes[2].set_ylabel("Request\ncompletions/s")
    axes[2].legend(ncol=2)

    axes[3].plot(
        time_s,
        _moving_average(simulated["engine_iterations_rate"][:n], smooth_bins),
        color="#009E73", linewidth=1.4, label="Simulated iterations/s",
    )
    batch_axis = axes[3].twinx()
    batch_axis.plot(
        time_s,
        _moving_average(simulated["batch"][:n], smooth_bins),
        color="#CC79A7", linewidth=1.2, label="Simulated decode batch",
    )
    axes[3].set_ylabel("Iterations/s")
    batch_axis.set_ylabel("Decode batch")
    lines = axes[3].lines + batch_axis.lines
    axes[3].legend(lines, [line.get_label() for line in lines], ncol=2)

    for axis in axes:
        axis.axvline(
            record["split_s"], color="#D55E00", linestyle="--",
            linewidth=1.1,
        )
        axis.grid(alpha=0.2)
    axes[0].text(
        record["split_s"] + 4.0,
        0.86,
        f"Observed power transition\n{record['split_s']:.1f} s",
        color="#D55E00",
        transform=axes[0].get_xaxis_transform(),
    )
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    save_plot()
    print(f"Saved no-dose timing diagnostic to {PLOT_PATH}")


if __name__ == "__main__":
    main()
