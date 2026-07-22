"""Shared causal feature and reporting primitives for the frozen evaluator."""

import csv
import math

import numpy as np

SOFT_DTW_GAMMA = 0.01
SOFT_DTW_BAND_S = 10


def soft_dtw_cost(
    x: np.ndarray, y: np.ndarray, *, gamma: float, band: int
) -> float:
    """Squared-error soft-DTW cost within a Sakoe-Chiba band."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if gamma <= 0.0 or band < 0:
        raise ValueError("soft-DTW gamma must be positive and band non-negative")
    if abs(x.size - y.size) > band:
        return float("inf")
    previous = np.full(y.size + 1, np.inf)
    previous[0] = 0.0
    for i, x_value in enumerate(x, start=1):
        current = np.full(y.size + 1, np.inf)
        for j in range(max(1, i - band), min(y.size, i + band) + 1):
            candidates = (previous[j - 1], previous[j], current[j - 1])
            minimum = min(candidates)
            soft_minimum = minimum - gamma * math.log(sum(
                math.exp(-(value - minimum) / gamma)
                for value in candidates if math.isfinite(value)
            ))
            current[j] = (x_value - y[j - 1]) ** 2 + soft_minimum
        previous = current
    return float(previous[-1])


def soft_dtw_divergence(
    measured: np.ndarray,
    predicted: np.ndarray,
    *,
    gamma: float = SOFT_DTW_GAMMA,
    band: int = SOFT_DTW_BAND_S,
) -> float:
    """Non-negative, symmetric soft-DTW divergence; zero for equal traces."""
    measured = np.asarray(measured, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if measured.size != predicted.size:
        raise ValueError("soft-DTW traces must have equal length")
    cross = soft_dtw_cost(measured, predicted, gamma=gamma, band=band)
    self_measured = soft_dtw_cost(measured, measured, gamma=gamma, band=band)
    self_predicted = soft_dtw_cost(predicted, predicted, gamma=gamma, band=band)
    return max(cross - 0.5 * (self_measured + self_predicted), 0.0)


def normalized_soft_dtw_diagnostics(
    measured: np.ndarray,
    predicted: np.ndarray,
    *,
    band: int = SOFT_DTW_BAND_S,
) -> dict[str, float]:
    """Range-normalized divergence and signed effect of the declared band.

    The effect can be negative because widening the band changes both the
    cross-cost and the two entropic self-costs in soft-DTW divergence.
    """
    measured = np.asarray(measured, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if measured.size != predicted.size or measured.size == 0:
        raise ValueError("soft-DTW diagnostic traces must have equal nonzero length")
    power_range = np.ptp(measured)
    if power_range <= 0.0:
        return {
            "soft_dtw_divergence": float("nan"),
            "soft_dtw_diagonal_divergence": float("nan"),
            "soft_dtw_band_effect": float("nan"),
            "soft_dtw_band_effect_fraction": float("nan"),
        }
    normalized_measured = (measured - measured.mean()) / power_range
    normalized_predicted = (predicted - measured.mean()) / power_range
    warped = soft_dtw_divergence(
        normalized_measured, normalized_predicted, band=band
    ) / measured.size
    diagonal = soft_dtw_divergence(
        normalized_measured, normalized_predicted, band=0
    ) / measured.size
    band_effect = diagonal - warped
    return {
        "soft_dtw_divergence": warped,
        "soft_dtw_diagonal_divergence": diagonal,
        "soft_dtw_band_effect": band_effect,
        "soft_dtw_band_effect_fraction": (
            band_effect / diagonal if diagonal > 0.0 else 0.0
        ),
    }


def raw_channels(d, candidate: str) -> tuple[np.ndarray, list[str], set[int]]:
    exact_state = "A_t" in d and "delta_A_t" in d
    active = d["A_t"] if exact_state else d["batch"] + d["pre_active"]
    if exact_state:
        delta = d["delta_A_t"]
    else:
        delta = np.zeros_like(active)
        for run in np.unique(d["run_id"]):
            at = np.flatnonzero(d["run_id"] == run)
            delta[at[1:]] = np.diff(active[at])
    suffix = "" if exact_state else "_proxy"
    values = [np.log1p(active), delta]
    names, levels = [f"log1p_A_t{suffix}", f"delta_A_t{suffix}"], {0}
    if candidate in ("M2", "M3"):
        exact_marks = all(
            key in d for key in ("arrivals", "input_tokens_arriving", "output_tokens_requested")
        )
        marks = (
            (d["arrivals"], d["input_tokens_arriving"], d["output_tokens_requested"])
            if exact_marks else (d["pre_active"] / 0.25, d["pre_tok"], d["dec_tok"])
        )
        values += [np.log1p(x) for x in marks]
        mark_suffix = "" if exact_marks else "_proxy"
        names += [f"log1p_arrivals_per_s{mark_suffix}",
                  f"log1p_input_tokens_arriving_per_s{mark_suffix}",
                  f"log1p_output_tokens_requested_per_s{mark_suffix}"]
        levels |= {2, 3, 4}
    if candidate == "M3":
        tp = d["tp"].astype(float)
        peak = np.where(d["hw_idx"] == 0, 312e12, 990e12)
        bw = np.where(d["hw_idx"] == 0, 2e12, 3.35e12)
        pre_f, dec_f = 2 * d["n_active"] * d["pre_tok"], 2 * d["n_active"] * d["dec_tok"]
        total_f = pre_f + dec_f
        values += [total_f / (tp * peak),
                   (d["w_read"] + d["kv_read"] + d["kv_write"]) / (tp * bw),
                   (total_f > 0).astype(float),
                   np.divide(pre_f, total_f, out=np.zeros_like(pre_f), where=total_f > 0)]
        names += ["u_compute", "u_memory", "busy", "prefill_share"]
        levels |= {5, 6, 7, 8}
    return np.column_stack(values), names, levels


def causal_design(d, candidate: str, fit_runs: set[int], scale=None):
    """Causal physical-time histories; level channels use a zero-DC basis."""
    raw, names, levels = raw_channels(d, candidate)
    fit = np.isin(d["run_id"], list(fit_runs))
    mean = raw[fit].mean(axis=0) if scale is None else scale[0]
    std = raw[fit].std(axis=0) if scale is None else scale[1]
    std = np.where(std > 1e-12, std, 1.0)
    raw = (raw - mean) / std
    taps_s = (0, 1, 2, 4, 8)
    taps = [int(round(t / float(d["dt_s"]))) for t in taps_s]
    columns, out_names = [], []
    starts = np.r_[0, np.flatnonzero(np.diff(d["run_id"])) + 1]
    if np.unique(d["run_id"]).size != starts.size:
        raise ValueError("Each run must occupy one contiguous ledger segment")
    for channel, name in enumerate(names):
        histories = []
        for tap in taps:
            history = raw[:, channel].copy() if tap == 0 else np.r_[np.zeros(tap), raw[:-tap, channel]]
            if tap:
                invalid = (starts[:, None] + np.arange(tap)).ravel()
                history[invalid[invalid < history.size]] = 0.0
            histories.append(history)
        if channel in levels:
            columns += [histories[j] - histories[0] for j in range(1, len(taps))]
            out_names += [f"{name}@{taps_s[j]}s-minus-0s" for j in range(1, len(taps))]
        else:
            columns += histories
            out_names += [f"{name}@{tap}s" for tap in taps_s]
    return np.column_stack(columns), out_names, (mean, std)


def trace_metrics(measured, predicted, native_dt=0.25) -> dict:
    """Frozen per-run metrics after four-bin mean aggregation."""
    measured, predicted = np.asarray(measured), np.asarray(predicted)
    factor = int(round(1.0 / native_dt))
    n = min(measured.size, predicted.size) // factor * factor
    if n < 62 * factor:
        return {key: float("nan") for key in
                ("energy_error_pct", "acf_r2", "acf_mae", "nrmse_range",
                 "nrmse_mean", "soft_dtw_divergence",
                 "soft_dtw_diagonal_divergence", "soft_dtw_band_effect",
                 "soft_dtw_band_effect_fraction")}
    y = measured[:n].reshape(-1, factor).mean(1)
    p = predicted[:n].reshape(-1, factor).mean(1)
    yc, pc = y - y.mean(), p - p.mean()

    def acf(x):
        denom = x @ x
        return np.array([x[:-lag] @ x[lag:] / denom if denom > 0 else 0.0
                         for lag in range(1, 61)])

    ay, ap = acf(yc), acf(pc)
    tss = np.sum((ay - ay.mean()) ** 2)
    rmse = np.sqrt(np.mean((p - y) ** 2))
    power_range = np.ptp(y)
    soft_dtw = normalized_soft_dtw_diagnostics(y, p)
    return {"energy_error_pct": 100 * abs(p.sum() - y.sum()) / y.sum(),
            "acf_r2": 1 - np.sum((ap - ay) ** 2) / tss if tss > 0 else np.nan,
            "acf_mae": float(np.mean(np.abs(ap - ay))),
            "nrmse_range": rmse / power_range if power_range > 0 else np.nan,
            "nrmse_mean": rmse / y.mean(),
            **soft_dtw,
            "mean_bias_pct": 100 * (p.mean() - y.mean()) / y.mean()}


def summaries(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["split"], row["candidate"]), []).append(row)
    out = []
    metadata = {"split", "candidate", "run_id", "source_id", "hardware", "model", "tp", "rate"}
    for (split, candidate), values in groups.items():
        record = {"split": split, "candidate": candidate, "runs": len(values)}
        for metric in (key for key in values[0] if key not in metadata):
            x = np.asarray([value[metric] for value in values], float)
            record |= {f"{metric}_median": float(np.nanmedian(x)),
                       f"{metric}_p90": float(np.nanpercentile(x, 90)),
                       f"{metric}_worst": float(np.nanmin(x) if metric == "acf_r2" else np.nanmax(x))}
        out.append(record)
    return out


def stratified_summaries(rows: list[dict]) -> list[dict]:
    """Summarize primary metrics by every frozen explanatory stratum."""
    run_load = {}
    for row in rows:
        run_load[row["run_id"]] = row["measured_mean_w"]
    by_hardware = {}
    for row in rows:
        by_hardware.setdefault(row["hardware"], {})[row["run_id"]] = run_load[row["run_id"]]
    cutoffs = {hardware: np.quantile(list(values.values()), (1 / 3, 2 / 3))
               for hardware, values in by_hardware.items()}
    groups = {}
    for row in rows:
        low, high = cutoffs[row["hardware"]]
        load = "low" if row["measured_mean_w"] <= low else (
            "high" if row["measured_mean_w"] > high else "middle")
        key = (row["candidate"], row["hardware"], row["split"], row["model"],
               row["tp"], row["rate"], load)
        groups.setdefault(key, []).append(row)
    output = []
    for key, values in groups.items():
        record = dict(zip(("candidate", "hardware", "split", "family", "tp", "rate", "load"), key))
        record["runs"] = len(values)
        for metric in (
            "energy_error_pct",
            "acf_r2",
            "acf_mae",
            "nrmse_range",
            "soft_dtw_divergence",
        ):
            data = np.asarray([value[metric] for value in values])
            record[f"{metric}_median"] = float(np.nanmedian(data))
        output.append(record)
    return output


def bootstrap_intervals(rows: list[dict], *, seed=20260710, samples=1000) -> list[dict]:
    """Fixed-seed run-level bootstrap intervals; bins are never resampled."""
    groups = {}
    for row in rows:
        groups.setdefault((row["split"], row["candidate"]), []).append(row)
    rng, output = np.random.default_rng(seed), []
    for (split, candidate), values in groups.items():
        record = {"split": split, "candidate": candidate, "runs": len(values),
                  "seed": seed, "samples": samples}
        for metric in (
            "energy_error_pct",
            "acf_r2",
            "acf_mae",
            "nrmse_range",
            "soft_dtw_divergence",
        ):
            data = np.asarray([value[metric] for value in values])
            draws = np.nanmedian(data[rng.integers(0, data.size, (samples, data.size))], axis=1)
            record[f"{metric}_median_ci_low"] = float(np.nanpercentile(draws, 2.5))
            record[f"{metric}_median_ci_high"] = float(np.nanpercentile(draws, 97.5))
        output.append(record)
    return output


def model_scalar_count(fit: dict) -> int:
    if fit.get("candidate") == "B2":
        return fit["parameter_count"]
    if fit.get("candidate") == "B4":
        return sum(len(item["coefficients"]) + 1 for item in fit["configurations"].values())
    base = fit.get("physics", fit.get("coefficients", fit.get("mean", [])))
    count = (len(np.atleast_1d(base)) + len(fit.get("residual", []))
             + int("cap_w_per_gpu" in fit) + int("lag_alpha" in fit))
    if "history_mean" in fit:
        count += np.asarray(fit["history_mean"]).size
        count += np.asarray(fit["history_scale"]).size + int("intercept" in fit)
    count += 2 if fit.get("mean_kind") == "M4A" else 0
    if "scale" in fit:
        count += sum(np.asarray(value).size for value in fit["scale"])
    return count


def write_csv(path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
