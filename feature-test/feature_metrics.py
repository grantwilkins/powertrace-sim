"""Secondary per-run metrics shared by feature-test candidates and baselines."""

import numpy as np


def _means(values: np.ndarray, factor: int) -> np.ndarray:
    n = values.size // factor * factor
    return values[:n].reshape(-1, factor).mean(1) if n else np.zeros(0)


def _ldc(values: np.ndarray, fraction: float) -> float:
    ordered = np.sort(values)[::-1]
    return float(ordered[min(int(np.floor(fraction * ordered.size)), ordered.size - 1)])


def secondary_trace_metrics(measured, predicted, *, dt_s: float, cap_w: float | None = None) -> dict:
    """Return window-energy, tail, ramp, load-duration, and cap metrics for one run."""
    y, p = np.asarray(measured, float).reshape(-1), np.asarray(predicted, float).reshape(-1)
    if y.size != p.size or y.size == 0 or dt_s <= 0:
        raise ValueError("Secondary metrics require aligned nonempty traces and positive dt_s")
    error = p - y
    out = {
        "measured_mean_w": float(np.mean(y)),
        "power_p95_abs_error_w": float(np.percentile(np.abs(error), 95)),
        "power_p99_abs_error_w": float(np.percentile(np.abs(error), 99)),
    }
    for seconds in (1, 5, 30):
        factor = int(round(seconds / dt_s))
        if not np.isclose(factor * dt_s, seconds):
            raise ValueError("Window seconds must be an integer multiple of dt_s")
        yw, pw = _means(y, factor), _means(p, factor)
        values = 100 * np.abs(pw - yw) / np.maximum(yw, 1e-12)
        out[f"window_energy_{seconds}s_median_error_pct"] = float(np.median(values))
        out[f"window_energy_{seconds}s_p90_error_pct"] = float(np.percentile(values, 90))
    for seconds in (dt_s, 1.0):
        factor = int(round(seconds / dt_s))
        yr, pr = np.diff(_means(y, factor)), np.diff(_means(p, factor))
        delta = pr - yr
        label = "250ms" if np.isclose(seconds, 0.25) else "1s"
        out[f"ramp_{label}_p95_abs_error_w"] = float(np.percentile(np.abs(delta), 95))
        out[f"ramp_{label}_max_up_error_w"] = float(np.max(pr) - np.max(yr))
        out[f"ramp_{label}_max_down_error_w"] = float(np.min(pr) - np.min(yr))
    for fraction in (0.01, 0.05, 0.50):
        label = f"{fraction:.2f}"
        out[f"ldc_measured_{label}_w"] = _ldc(y, fraction)
        out[f"ldc_error_{label}_w"] = _ldc(p, fraction) - out[f"ldc_measured_{label}_w"]
    out["cap_hit_fraction"] = (
        float(np.mean(p >= float(cap_w) - 1e-9)) if cap_w is not None else 0.0
    )
    return out
