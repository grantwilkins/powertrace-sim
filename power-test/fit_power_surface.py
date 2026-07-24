"""Fit the static power surface per hardware and score it (DESIGN.md 4).

For each hardware, NNLS on train bins with finite power, with the design
columns filtered per run through the response chain for each delay in the
plan-fixed grid (0.0, 0.25, 0.5, 0.75) s; the delay is picked by train RMSE
and the full grid is reported. Evaluation runs capped predictions per run
through the chain and scores them with the frozen trace_metrics against
joined measured power, on roles train and test_indomain only — the holdout
roles are plan item 3's matrix and are not burned here. NaN power bins are
linearly interpolated per run before scoring (fraction reported); runs
shorter than trace_metrics' 62 s floor come back all-NaN and are counted as
skipped, never dropped silently.

Usage: uv run python power-test/fit_power_surface.py
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "feature-test"))

from evaluation_core import trace_metrics  # noqa: E402
from build_probe_power_calibration import balanced_level_weights  # noqa: E402
from power_surface import (HARDWARE, UTIL_HINGE, predict,  # noqa: E402
                           surface_design)
from response_chain import MOVING_AVERAGE_S, apply_chain  # noqa: E402
from model.training_data.power_parsing import (  # noqa: E402
    parse_power_csv_per_gpu,
    tp_sum_power,
)

BASE = Path(__file__).resolve().parent
CACHE = BASE / "sim_ledger_power_250ms.npz"
PROBE_CACHE = BASE / "probe_power_calibration_250ms.npz"
DELAY_GRID_S = (0.0, 0.25, 0.5, 0.75)  # plan-fixed
SCORED_ROLES = ("train", "test_indomain")
PRIMARY_METRICS = (
    "energy_error_pct",
    "acf_mae",
    "acf_r2",
    "soft_dtw_divergence",
    "nrmse_range",
)
PROBE_SELECTION_METRICS = (
    "energy_error_pct",
    "acf_mae",
    "acf_r2",
    "nrmse_range",
)


def load_cache(path=CACHE) -> dict:
    path = Path(path)
    if not path.exists():
        sys.exit(f"{path} not found: run power-test/join_power.py first to "
                 "join measured power onto the simulated ledger cache.")
    with np.load(path, allow_pickle=True) as data:
        d = {key: data[key] for key in data.files}
    for key in ("power_valid", "run_id", "role_idx", "role_names", "hw_idx",
                "hw_names"):
        if key not in d:
            sys.exit(f"{path} is missing column {key!r}; it does not match "
                     "the join_power.py output schema.")
    return d


def load_probe_calibration(path=PROBE_CACHE) -> dict:
    path = Path(path)
    if not path.exists():
        raise ValueError(
            f"{path} not found: run power-test/build_probe_power_calibration.py"
        )
    with np.load(path, allow_pickle=False) as data:
        calibration = {key: data[key] for key in data.files}
    required = {
        "hardware", "level_id", "power", "pre_tok", "dec_tok", "busy",
        "w_read", "kv_read", "kv_write", "engine_iterations_rate", "tp",
        "n_active", "transformer_active_params", "output_head_params",
        "logit_tokens_rate", "w_bytes", "fp8", "fp8_flop_frac",
    }
    missing = sorted(required - set(calibration))
    if missing:
        raise ValueError(f"Probe calibration is missing columns: {missing}")
    return calibration


def loaded_idle_anchors(root: Path) -> dict[str, float]:
    """Median model-loaded idle power per GPU from dedicated source probes."""
    values: dict[str, list[float]] = {}
    for manifest_path in sorted(root.glob("*/*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        probe = manifest.get("probe") or {}
        if probe.get("type") != "idle_hold":
            continue
        levels = probe.get("levels") or []
        if len(levels) != 1:
            raise ValueError(f"{manifest_path} idle hold must have one level")
        level = levels[0]
        tp = int(manifest["tp"])
        parsed = parse_power_csv_per_gpu(
            str(manifest_path.parent / "power.csv"),
            gpus_per_node=int(manifest["gpus_per_node"]),
            strict_topology=True,
            local_utc_offset_s=float((manifest.get("clock") or {}).get(
                "local_utc_offset_s", 0.0)),
        )
        if parsed is None:
            raise ValueError(f"Could not parse idle power for {manifest_path.parent}")
        timestamps = np.asarray(parsed["timestamps"], float)
        keep = (
            (timestamps >= float(level["t_start_epoch"]))
            & (timestamps <= float(level["t_end_epoch"]))
        )
        if not np.any(keep):
            raise ValueError(f"Idle level has no power samples: {manifest_path.parent}")
        node_power = tp_sum_power(parsed["power_per_gpu"][keep], tp)
        values.setdefault(str(manifest["hardware"]), []).append(
            float(np.median(node_power) / tp)
        )
    return {
        hardware: float(np.median(samples))
        for hardware, samples in values.items()
    }


def run_slices(run_id: np.ndarray) -> list[tuple[int, int, int]]:
    starts = np.r_[0, np.flatnonzero(run_id[1:] != run_id[:-1]) + 1, run_id.size]
    slices = [(int(run_id[lo]), int(lo), int(hi))
              for lo, hi in zip(starts[:-1], starts[1:])]
    if len({run for run, _, _ in slices}) != len(slices):
        raise ValueError("Each run must occupy one contiguous cache segment")
    return slices


def filtered_design(design, slices, dt, hardware, delay_s) -> np.ndarray:
    out = np.empty_like(design)
    for _, lo, hi in slices:
        out[lo:hi] = apply_chain(design[lo:hi], dt, hardware, delay_s)
    return out


def nnls_rms_scaled(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Same RMS conditioning as feature-test/evaluate_candidates.py.
    scale = np.sqrt(np.mean(x ** 2, axis=0))
    keep = scale > 0
    scaled, _ = nnls(x[:, keep] / scale[keep], y)
    coefficients = np.zeros(x.shape[1])
    coefficients[keep] = scaled / scale[keep]
    return coefficients


def interpolate_nan(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Linear interpolation over NaN gaps; edges clamp to nearest sample."""
    bad = np.isnan(values)
    if not bad.any():
        return values, 0
    if bad.all():
        return values, int(bad.sum())
    out = values.copy()
    good = np.flatnonzero(~bad)
    out[bad] = np.interp(np.flatnonzero(bad), good, values[good])
    return out, int(bad.sum())


def fit_hardware(
    d, hardware: str, *, loaded_idle_w_per_gpu: float | None = None,
    probe_calibration: dict | None = None,
    probe_level_weight_scale: float = 1.0,
) -> tuple[dict, np.ndarray, list[str], list]:
    hw = d["hw_idx"] == list(d["hw_names"]).index(hardware)
    sub = {key: value[hw] for key, value in d.items()
           if isinstance(value, np.ndarray) and value.shape == hw.shape}
    design, names = surface_design(sub, hardware)
    slices = run_slices(sub["run_id"])
    dt = float(d["dt_s"])
    train_role = list(d["role_names"]).index("train")
    y = sub["power"].astype(float)
    families = np.asarray(d["family_names"])[sub["family_idx"]].astype(str)
    dense = ~np.char.startswith(families, "moe")
    train = (sub["role_idx"] == train_role) & dense & np.isfinite(y)
    probe = None
    if probe_calibration is not None:
        selected = np.asarray(probe_calibration["hardware"]).astype(str) == hardware
        if np.any(selected):
            probe = {
                key: value[selected]
                for key, value in probe_calibration.items()
                if isinstance(value, np.ndarray) and value.shape == selected.shape
            }
            probe_design, probe_names = surface_design(probe, hardware)
            if probe_names != names:
                raise ValueError("Probe and workload surface columns disagree")
            level_ids = np.asarray(probe["level_id"]).astype(str)
            starts = np.r_[0, np.flatnonzero(level_ids[1:] != level_ids[:-1]) + 1,
                           level_ids.size]
            probe_slices = [
                (index, int(lo), int(hi))
                for index, (lo, hi) in enumerate(zip(starts[:-1], starts[1:]))
            ]
            base_run_bins = [
                np.sum(train & (sub["run_id"] == run))
                for run in np.unique(sub["run_id"][train])
            ]
            probe_tp = np.unique(np.asarray(probe["tp"], float))
            if probe_tp.size != 1:
                raise ValueError("Probe calibration must bind one TP per hardware")
            if probe_level_weight_scale <= 0.0:
                raise ValueError("Probe level weight scale must be positive")
            target_level_bins = (
                float(np.median(base_run_bins))
                / probe_tp[0] ** 2
                * probe_level_weight_scale
            )
            probe_weights = balanced_level_weights(level_ids, target_level_bins)
    grid = {}
    for delay_s in DELAY_GRID_S:
        x = filtered_design(design, slices, dt, hardware, delay_s)
        fit_x, fit_y = x[train], y[train]
        fit_weights = np.ones(int(train.sum()))
        if probe is not None:
            probe_x = filtered_design(
                probe_design, probe_slices, dt, hardware, delay_s
            )
            fit_x = np.vstack([fit_x, probe_x])
            fit_y = np.concatenate([fit_y, np.asarray(probe["power"], float)])
            fit_weights = np.concatenate([fit_weights, probe_weights])
        root_weight = np.sqrt(fit_weights)
        weighted_x = fit_x * root_weight[:, None]
        weighted_y = fit_y * root_weight
        if loaded_idle_w_per_gpu is None:
            coefficients = nnls_rms_scaled(weighted_x, weighted_y)
        else:
            coefficients = np.zeros(fit_x.shape[1])
            coefficients[0] = float(loaded_idle_w_per_gpu)
            residual_target = fit_y - fit_x[:, 0] * coefficients[0]
            coefficients[3:] = nnls_rms_scaled(
                weighted_x[:, 3:], residual_target * root_weight
            )
        residual = fit_x @ coefficients - fit_y
        rmse = float(np.sqrt(np.average(residual ** 2, weights=fit_weights)))
        base_rmse = float(np.sqrt(np.mean((x[train] @ coefficients - y[train]) ** 2)))
        probe_rmse = (
            float(np.sqrt(np.mean((probe_x @ coefficients - probe["power"]) ** 2)))
            if probe is not None else None
        )
        grid[delay_s] = (rmse, coefficients, base_rmse, probe_rmse)
    best = min(DELAY_GRID_S, key=lambda delay_s: grid[delay_s][0])
    fit = {
        "hardware": hardware,
        "coefficients": dict(zip(names, grid[best][1].tolist())),
        "delay_s": best,
        "delay_grid_rmse": {f"{delay_s:g}": grid[delay_s][0]
                            for delay_s in DELAY_GRID_S},
        "cap_w_per_gpu": None,
        "source_idle_anchor_w_per_gpu": loaded_idle_w_per_gpu,
        "loaded_idle_contract": {
            "checkpoint_dependence": "none under observed vLLM allocation policy",
            "scheduling_policy": "unrecorded in source bundle",
            "power_state": "fixed application clocks; pstate and cap unrecorded",
            "support": (
                "checkpoint transfer only under matched engine/power state"
                if loaded_idle_w_per_gpu is not None
                else "unanchored diagnostic fit only"
            ),
            "unsupported": [
                "unseen scheduling policy",
                "unseen pstate or application clocks",
                "unseen power limit",
            ],
        },
        "moving_average_s": MOVING_AVERAGE_S[hardware],
        "n_train_bins": int(train.sum()),
        "n_probe_bins": 0 if probe is None else int(probe["power"].size),
        "n_probe_levels": 0 if probe is None else int(np.unique(level_ids).size),
        "probe_level_target_bins": None if probe is None else target_level_bins,
        "n_moe_train_bins_excluded": int(
            np.sum((sub["role_idx"] == train_role) & ~dense & np.isfinite(y))),
        "provenance": {
            "coefficients": "fitted-here",
            "fit_population": "dense training bins only; MoE routing blocked",
            "controlled_probe_calibration": (
                "A100 source Llama-70B levels; equal per-GPU-squared total "
                "weight per level"
                if probe is not None else "none for this hardware"
            ),
            "delay_s": "fitted-here (selected from plan-fixed grid by train RMSE)",
            "delay_grid_s": "plan-fixed",
            "source_idle_anchor_w_per_gpu": (
                "fixed from dedicated source idle-hold probe before dynamic fit"
                if loaded_idle_w_per_gpu is not None
                else "unavailable; intercept fitted jointly for diagnostic only"
            ),
            "cap_w_per_gpu": "unbound; no cap applied without run telemetry",
            "moving_average_s": "cited (arXiv:2312.02741, probe-confirmed)",
            "compute_peak_flops_s": f"cited ({HARDWARE[hardware]['compute_peak_flops_s']:g})",
            "hbm_bandwidth_bytes_s": f"cited ({HARDWARE[hardware]['hbm_bandwidth_bytes_s']:g})",
            "utilization_hinge": f"plan-fixed {UTIL_HINGE}",
            "fp8_compute_scale": "plan-fixed (1 - 0.5*clip(fp8_flop_frac, 0, 1))",
            "iter_rate_scale": "conditioning constant only (1000/s)",
        },
    }
    return fit, design, names, [sub, slices]


def coefficients_in_design_order(fit: dict, names: list[str]) -> np.ndarray:
    coefficients = fit["coefficients"]
    if set(coefficients) != set(names):
        raise ValueError("Fitted coefficient names do not match the surface design")
    return np.asarray([coefficients[name] for name in names], dtype=float)


def score_hardware(d, hardware, fit, design, names, context) -> tuple[list[dict], dict]:
    sub, slices = context
    dt = float(d["dt_s"])
    role_names = list(d["role_names"])
    coefficients = coefficients_in_design_order(fit, names)
    rows, counters = [], {"interpolated_bins": 0, "scored_bins": 0,
                          "skipped_runs": []}
    scored_roles = {role_names.index(role) for role in SCORED_ROLES}
    for run, lo, hi in slices:
        role = int(sub["role_idx"][lo])
        if role not in scored_roles:
            continue
        measured, n_bad = interpolate_nan(sub["power"][lo:hi].astype(float))
        raw = predict(design[lo:hi], coefficients, sub["tp"][lo:hi], hardware)
        pred = apply_chain(raw, dt, hardware, fit["delay_s"])
        metrics = (trace_metrics(measured, pred, native_dt=dt)
                   if n_bad < hi - lo else {})
        # trace_metrics' short-run path omits mean_bias_pct; keep rows uniform.
        metrics = {key: metrics.get(key, float("nan"))
                   for key in PRIMARY_METRICS + ("mean_bias_pct", "nrmse_mean")}
        counters["interpolated_bins"] += n_bad
        counters["scored_bins"] += hi - lo
        row = {"hardware": hardware, "role": role_names[role], "run_id": run,
               "rate": float(sub["rate"][lo]),
               "model": str(d["model_names"][sub["model_idx"][lo]]),
               "dense": not str(d["family_names"][sub["family_idx"][lo]]
                                ).startswith("moe"), **metrics}
        if all(np.isnan(row[key]) for key in PRIMARY_METRICS):
            counters["skipped_runs"].append(run)
        rows.append(row)
    return rows, counters


def role_table(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["hardware"], row["role"]), []).append(row)
    tables = []
    for (hardware, role), values in sorted(groups.items()):
        record = {"hardware": hardware, "role": role, "runs": len(values),
                  "skipped_runs": sum(
                      all(np.isnan(v[key]) for key in PRIMARY_METRICS)
                      for v in values)}
        with warnings.catch_warnings():
            # Skipped (< 62 s) runs are reported explicitly; the all-NaN
            # reduction warning adds nothing.
            warnings.simplefilter("ignore", RuntimeWarning)
            for metric in PRIMARY_METRICS:
                data = np.asarray([value[metric] for value in values], float)
                worst = np.nanmin(data) if metric == "acf_r2" else np.nanmax(data)
                record |= {f"{metric}_median": float(np.nanmedian(data)),
                           f"{metric}_p90": float(np.nanpercentile(data, 90)),
                           f"{metric}_worst": float(worst)}
            by_rate = {}
            for value in values:
                by_rate.setdefault(value["rate"], []).append(value["mean_bias_pct"])
            record["mean_bias_pct_by_rate"] = {
                f"{rate:g}": float(np.nanmedian(np.asarray(biases, float)))
                for rate, biases in sorted(by_rate.items())}
        tables.append(record)
    return tables


def development_medians(rows: list[dict]) -> dict[str, float]:
    selected = [
        row for row in rows
        if row["dense"] and row["role"] == "test_indomain"
    ]
    if not selected:
        raise ValueError("Probe selection requires dense source-development rows")
    return {
        metric: float(np.nanmedian([row[metric] for row in selected]))
        for metric in PRIMARY_METRICS
    }


def select_probe_candidate(baseline: dict, candidate: dict) -> bool:
    """Require a source-development Pareto improvement; Qwen is never consulted."""
    nonworse = []
    improved = []
    for metric in PROBE_SELECTION_METRICS:
        direction = 1.0 if metric == "acf_r2" else -1.0
        delta = direction * (candidate[metric] - baseline[metric])
        nonworse.append(delta >= 0.0)
        improved.append(delta > 0.0)
    return all(nonworse) and any(improved)


def print_tables(title: str, tables: list[dict]) -> None:
    print(f"\n== {title} ==")
    for record in tables:
        print(f"{record['hardware']} {record['role']}: runs={record['runs']} "
              f"skipped={record['skipped_runs']}")
        for metric in PRIMARY_METRICS:
            print(f"  {metric}: median={record[f'{metric}_median']:.4g} "
                  f"p90={record[f'{metric}_p90']:.4g} "
                  f"worst={record[f'{metric}_worst']:.4g}")
        biases = " ".join(f"{rate}:{bias:+.2f}%" for rate, bias
                          in record["mean_bias_pct_by_rate"].items())
        print(f"  mean_bias_pct by rate: {biases}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(CACHE))
    parser.add_argument("--surface-out", default=str(BASE / "fitted_surface.json"))
    parser.add_argument("--report-out", default=str(BASE / "fit_report.json"))
    parser.add_argument("--idle-root", default="data/runs")
    parser.add_argument("--probe-calibration", default=str(PROBE_CACHE))
    args = parser.parse_args(argv)
    d = load_cache(args.cache)
    probe_calibration = load_probe_calibration(args.probe_calibration)
    idle = loaded_idle_anchors(Path(args.idle_root))
    missing_idle = sorted(set(map(str, d["hw_names"])) - set(idle))
    if missing_idle:
        raise ValueError(
            f"missing dedicated loaded-idle source anchors: {missing_idle}"
        )
    fits, rows, counters, probe_selection = {}, [], {}, {}
    for hardware in d["hw_names"]:
        hardware = str(hardware)
        baseline_fit, baseline_design, baseline_names, baseline_context = fit_hardware(
            d,
            hardware,
            loaded_idle_w_per_gpu=idle.get(hardware),
        )
        baseline_rows, baseline_counters = score_hardware(
            d, hardware, baseline_fit, baseline_design, baseline_names,
            baseline_context,
        )
        has_probe = np.any(
            np.asarray(probe_calibration["hardware"]).astype(str) == hardware
        )
        fit, design, names, context = (
            baseline_fit, baseline_design, baseline_names, baseline_context
        )
        selected_rows, selected_counters = baseline_rows, baseline_counters
        if has_probe:
            candidate_fit, candidate_design, candidate_names, candidate_context = (
                fit_hardware(
                    d,
                    hardware,
                    loaded_idle_w_per_gpu=idle.get(hardware),
                    probe_calibration=probe_calibration,
                )
            )
            candidate_rows, candidate_counters = score_hardware(
                d, hardware, candidate_fit, candidate_design, candidate_names,
                candidate_context,
            )
            baseline_metrics = development_medians(baseline_rows)
            candidate_metrics = development_medians(candidate_rows)
            selected = select_probe_candidate(baseline_metrics, candidate_metrics)
            probe_selection[hardware] = {
                "selected": selected,
                "rule": "source-development Pareto improvement on energy, "
                        "ACF-MAE, ACF-R2, and range NRMSE",
                "report_only_metrics": ["soft_dtw_divergence"],
                "target_or_holdout_used": False,
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "candidate_fit": candidate_fit,
            }
            if selected:
                fit, design, names, context = (
                    candidate_fit, candidate_design, candidate_names,
                    candidate_context,
                )
                selected_rows, selected_counters = candidate_rows, candidate_counters
        fits[hardware] = fit
        rows += selected_rows
        counters[hardware] = selected_counters
    Path(args.surface_out).write_text(
        json.dumps({"schema_version": "power-test-surface-v3",
                    "per_hardware": fits}, indent=2))
    tables = role_table(rows)
    dense_tables = role_table([row for row in rows if row["dense"]])
    report = {
        "schema_version": "power-test-fit-report-v1",
        "scored_roles": list(SCORED_ROLES),
        "unscored_holdout_note": "holdout_twin/holdout_rate/holdout_model/"
                                 "dtype_calibration reserved for plan item 3",
        "interpolated_power_bin_fraction": {
            hardware: (c["interpolated_bins"] / c["scored_bins"]
                       if c["scored_bins"] else 0.0)
            for hardware, c in counters.items()},
        "skipped_runs": {hardware: c["skipped_runs"]
                         for hardware, c in counters.items()},
        "tables": tables,
        "tables_dense_only": dense_tables,
        "moe_blocked_models": sorted({row["model"] for row in rows
                                      if not row["dense"]}),
        "moe_note": "gpt-oss cells fitted and reported but carry the MoE "
                    "routing block (TODO.md item 1)",
        "probe_calibration_selection": probe_selection,
        "per_run": rows,
    }
    Path(args.report_out).write_text(json.dumps(report, indent=2))
    for hardware, fit in fits.items():
        print(f"{hardware}: delay_s={fit['delay_s']} "
              f"grid_rmse={fit['delay_grid_rmse']} "
              f"train_bins={fit['n_train_bins']}")
        frac = report["interpolated_power_bin_fraction"][hardware]
        print(f"  interpolated power bins: {100 * frac:.2f}% "
              f"skipped runs: {len(counters[hardware]['skipped_runs'])}")
    print_tables("all models", tables)
    print_tables("dense models only (gpt-oss MoE-blocked)", dense_tables)


if __name__ == "__main__":
    main()
