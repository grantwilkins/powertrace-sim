"""Fit and audit separate clean dense and expanded MoE power surfaces."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path[:0] = [str(BASE), str(ROOT / "feature-test")]

from fit_power_surface import (  # noqa: E402
    interpolate_nan,
    nnls_rms_scaled,
    run_slices,
)
from clean_dense_surface import (  # noqa: E402
    DENSE_FEATURES,
    dense_design,
    fit_dense_hardware,
    phase_work_utilization,
    resident_fraction,
)
from moe_surface import baseline_prediction  # noqa: E402
from moe_surface_core import (  # noqa: E402
    apply_by_run,
    predict as predict_moe_v3,
    surface_design as moe_design_node,
)
from plot_power_metric_audit import (  # noqa: E402
    METRICS,
    _plot_metric,
    _plot_signed_bias,
    diagnostic_metrics,
)
from power_surface import HARDWARE  # noqa: E402

CACHE_PATH = BASE / "sim_ledger_power_uniform_current_250ms.npz"
LEGACY_DENSE_ARTIFACT = BASE / "fitted_surface.json"
LEGACY_MOE_ARTIFACT = BASE / "fitted_moe_surface_v3.json"
TIMING_ARTIFACT = ROOT / "timing-test" / "fitted_efficiencies.json"
ARTIFACT_PATH = BASE / "clean_power_surfaces.json"
REPORT_PATH = BASE / "clean_power_report.json"
RUN_CSV_PATH = BASE / "clean_power_per_run.csv"
CELL_CSV_PATH = BASE / "clean_power_cells.csv"

MOE_FEATURES = (
    "idle",
    "multi_gpu_floor",
    "logical_memory_util_lag_250ms",
    "duty_sqrt_exact_compute_util",
    "engine_iterations_rate",
    "log_decode_batch",
)
SOFT_DTW_DIAGNOSTICS = (
    "soft_dtw_diagonal_divergence",
    "soft_dtw_band_effect",
    "soft_dtw_band_effect_fraction",
)


def load_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_metadata(cache: dict) -> dict[int, dict]:
    models = list(map(str, cache["model_names"]))
    hardware = list(map(str, cache["hw_names"]))
    roles = list(map(str, cache["role_names"]))
    families = list(map(str, cache["family_names"]))
    output = {}
    for run, lo, hi in run_slices(cache["run_id"]):
        output[run] = {
            "model": models[int(cache["model_idx"][lo])],
            "hardware": hardware[int(cache["hw_idx"][lo])],
            "family": families[int(cache["family_idx"][lo])],
            "tp": int(cache["tp"][lo]),
            "rate": float(cache["rate"][lo]),
            "legacy_role": roles[int(cache["role_idx"][lo])],
        }
        for key in ("model_idx", "hw_idx", "family_idx", "tp", "rate", "role_idx"):
            if np.unique(cache[key][lo:hi]).size != 1:
                raise ValueError(f"Run {run} has non-constant {key}")
    return output


def source_roles(metadata: dict[int, dict]) -> dict[int, str]:
    """Keep repeated workloads together; external traces provide validation."""
    roles = {}
    for run, row in metadata.items():
        if row["legacy_role"] == "holdout_twin":
            roles[run] = "transfer_twin"
        elif row["rate"] == 4.0:
            roles[run] = "stress_rate4"
        elif row["rate"] < 4.0:
            roles[run] = "train_source"
        else:
            raise ValueError(f"Unexpected source rate {row['rate']:g}")
    return roles


def training_run_sets(metadata: dict[int, dict], roles: dict[int, str]) -> dict[str, set[int]]:
    dense = {
        run for run, row in metadata.items()
        if row["family"].startswith("dense") and roles[run] == "train_source"
    }
    moe = {
        run for run, row in metadata.items()
        if row["family"].startswith("moe") and roles[run] == "train_source"
    }
    if dense & moe:
        raise AssertionError("Dense and MoE training populations overlap")
    return {"dense": dense, "moe": moe}


def fit_run_balanced(design_pg, power_pg, run_id, fit_runs: set[int]) -> np.ndarray:
    """NNLS with equal total squared-error weight per run."""
    design_pg = np.asarray(design_pg, float)
    power_pg = np.asarray(power_pg, float)
    run_id = np.asarray(run_id)
    weights = np.zeros(power_pg.size)
    for run in sorted(fit_runs):
        selected = (run_id == run) & np.isfinite(power_pg)
        if not selected.any():
            raise ValueError(f"Training run {run} has no finite power")
        weights[selected] = 1.0 / selected.sum()
    selected = weights > 0.0
    root = np.sqrt(weights[selected])
    return nnls_rms_scaled(
        design_pg[selected] * root[:, None], power_pg[selected] * root
    )


def per_gpu(node_design, node_power, tp) -> tuple[np.ndarray, np.ndarray]:
    tp = np.asarray(tp, float)
    if np.any(tp <= 0.0):
        raise ValueError("TP must be positive")
    return np.asarray(node_design, float) / tp[:, None], np.asarray(node_power, float) / tp


def moe_feature_basis(design, tp) -> tuple[np.ndarray, list[str]]:
    design = np.asarray(design, float)
    names = list(MOE_FEATURES)
    if np.all(np.asarray(tp) > 1):
        design = np.delete(design, 1, axis=1)
        names.pop(1)
    return design, names


def moe_compute_coordinate(sub: dict) -> np.ndarray:
    compute_util = (
        np.asarray(sub["gemm_flops_rate"], float)
        + np.asarray(sub["attn_flops_rate"], float)
    ) / (
        np.asarray(sub["tp"], float)
        * HARDWARE["A100"]["compute_peak_flops_s"]
    )
    return np.sqrt(
        np.clip(np.asarray(sub["busy"], float), 0.0, 1.0)
        * np.clip(compute_util, 0.0, None)
    )


def fit_surfaces(cache: dict, metadata: dict, roles: dict[int, str]) -> tuple[dict, np.ndarray]:
    fit_runs = training_run_sets(metadata, roles)
    prediction_pg = np.full(cache["run_id"].shape, np.nan)
    dense_fits = {}
    for hardware in map(str, cache["hw_names"]):
        hardware_index = list(map(str, cache["hw_names"])).index(hardware)
        selected = cache["hw_idx"] == hardware_index
        sub_runs = cache["run_id"][selected]
        hardware_fit_runs = fit_runs["dense"] & set(map(int, np.unique(sub_runs)))
        fit, hardware_prediction, fitted_selected = fit_dense_hardware(
            cache, hardware, hardware_fit_runs
        )
        if not np.array_equal(selected, fitted_selected):
            raise AssertionError("Dense hardware selection changed during fitting")
        prediction_pg[selected] = hardware_prediction
        dense_fits[hardware] = fit

    models = np.asarray(cache["model_names"])[cache["model_idx"]].astype(str)
    moe_fits = {}
    for model in ("gpt-oss-20b", "gpt-oss-120b"):
        moe_selected = models == model
        moe_tp = cache["tp"][moe_selected]
        moe_design_pg, moe_target_pg = per_gpu(
            moe_design_node(cache)[moe_selected],
            cache["power"][moe_selected],
            moe_tp,
        )
        moe_sub = {
            key: np.asarray(cache[key])[moe_selected]
            for key in ("gemm_flops_rate", "attn_flops_rate", "tp", "busy")
        }
        moe_design_pg = np.insert(
            moe_design_pg, 3, moe_compute_coordinate(moe_sub), axis=1,
        )
        moe_design_pg, feature_names = moe_feature_basis(
            moe_design_pg, moe_tp
        )
        moe_runs = cache["run_id"][moe_selected]
        model_fit_runs = fit_runs["moe"] & set(map(int, np.unique(moe_runs)))
        coefficients = fit_run_balanced(
            moe_design_pg, moe_target_pg, moe_runs, model_fit_runs
        )
        prediction_pg[moe_selected] = moe_design_pg @ coefficients
        moe_fits[model] = {
            "feature_names": feature_names,
            "coefficients": coefficients.tolist(),
            "fit_run_ids": sorted(model_fit_runs),
            "supported_tp": sorted(set(map(int, moe_tp))),
        }
    artifact = {
        "schema_version": "clean-separated-power-surfaces-v4",
        "training_policy": (
            "per-GPU one-second run/p90-tail-balanced NNLS for dense power; "
            "hardware idle floors fixed from sustained source-idle gaps; repeated "
            "legacy workloads never cross "
            "roles; every source repetition at rates <=2 fits; rate 4 remains "
            "stress-only; dense uses exact timing-roofline work with "
            "phase-resolved diagnostic channels; "
            "architecture-calibrated MoE laws remain separate"
        ),
        "dense": dense_fits,
        "moe": {
            "hardware": "A100",
            "models": ["gpt-oss-20b", "gpt-oss-120b"],
            "per_model": moe_fits,
        },
    }
    if not np.isfinite(prediction_pg).all():
        raise AssertionError("Composite surface left runs without predictions")
    return artifact, prediction_pg


def predict_dense_node_power(cache: dict, artifact: dict) -> np.ndarray:
    """Apply a clean dense artifact and return TP-summed node watts."""
    prediction = np.full(cache["run_id"].shape, np.nan)
    for hardware in map(str, cache["hw_names"]):
        fit = artifact["dense"][hardware]
        if fit["feature_names"] != list(DENSE_FEATURES):
            raise ValueError(f"Unexpected clean dense basis for {hardware}")
        design_pg, selected = dense_design(
            cache, hardware, float(fit["delay_s"])
        )
        prediction_pg = design_pg @ np.asarray(fit["coefficients"], float)
        prediction[selected] = prediction_pg * cache["tp"][selected]
    if not np.isfinite(prediction).all():
        raise AssertionError("Clean dense artifact left rows without predictions")
    return prediction


def previous_predictions(cache: dict) -> tuple[np.ndarray, np.ndarray]:
    dense_artifact = json.loads(LEGACY_DENSE_ARTIFACT.read_text())
    prediction = baseline_prediction(cache, dense_artifact)
    supported = np.ones(prediction.size, dtype=bool)

    models = np.asarray(cache["model_names"])[cache["model_idx"]].astype(str)
    gpt20 = models == "gpt-oss-20b"
    moe_artifact = json.loads(LEGACY_MOE_ARTIFACT.read_text())
    raw = predict_moe_v3(
        moe_design_node(cache), np.asarray(moe_artifact["coefficients"], float),
        cache["tp"],
    )
    moe_prediction = apply_by_run(
        raw, cache["run_id"], float(cache["dt_s"]), "A100", 0.0
    )
    prediction[gpt20] = moe_prediction[gpt20]
    supported[models == "gpt-oss-120b"] = False
    return prediction, supported


def score_runs(cache: dict, metadata: dict, roles: dict[int, str], prediction_pg) -> list[dict]:
    previous, previous_supported = previous_predictions(cache)
    dt_s = float(cache["dt_s"])
    rows = []
    for run, lo, hi in run_slices(cache["run_id"]):
        meta = metadata[run]
        tp = meta["tp"]
        measured, interpolated = interpolate_nan(cache["power"][lo:hi].astype(float))
        candidate = prediction_pg[lo:hi] * tp
        candidate_metrics = diagnostic_metrics(
            measured, candidate, tp=tp, native_dt=dt_s
        )
        previous_metrics = diagnostic_metrics(
            measured, previous[lo:hi], tp=tp, native_dt=dt_s
        )
        row = {
            "run_id": run,
            **meta,
            "role": roles[run],
            "surface": "moe" if meta["family"].startswith("moe") else "dense",
            "surface_supported": True,
            "previous_supported": bool(previous_supported[lo]),
            "interpolated_power_bins": interpolated,
        }
        for metric in METRICS:
            row[metric] = float(candidate_metrics[metric])
            row[f"previous_{metric}"] = float(previous_metrics[metric])
        for metric in SOFT_DTW_DIAGNOSTICS:
            row[metric] = float(candidate_metrics[metric])
            row[f"previous_{metric}"] = float(previous_metrics[metric])
        row["signed_energy_bias_pct"] = float(candidate_metrics["mean_bias_pct"])
        row["previous_signed_energy_bias_pct"] = float(previous_metrics["mean_bias_pct"])
        rows.append(row)
    return rows


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for key, values in sorted(groups.items()):
        record = dict(zip(keys, key)) | {"runs": len(values)}
        record["previous_supported_runs"] = sum(
            row["previous_supported"] for row in values
        )
        for prefix in ("", "previous_"):
            for metric in (
                *METRICS, *SOFT_DTW_DIAGNOSTICS, "signed_energy_bias_pct"
            ):
                record[f"{prefix}{metric}_median"] = float(np.nanmedian([
                    row[f"{prefix}{metric}"] for row in values
                ]))
        output.append(record)
    return output


def aggregate_cells(rows: list[dict]) -> list[dict]:
    keys = ("hardware", "model", "family", "tp", "rate")
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for key, values in sorted(groups.items()):
        record = dict(zip(keys, key))
        record["runs"] = len(values)
        record["roles"] = ",".join(sorted({row["role"] for row in values}))
        record["surface_supported"] = True
        record["previous_supported"] = all(
            row["previous_supported"] for row in values
        )
        for metric in (*METRICS, *SOFT_DTW_DIAGNOSTICS,
                       "signed_energy_bias_pct"):
            record[metric] = float(np.nanmedian([
                row[metric] for row in values
            ]))
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cache = load_cache(CACHE_PATH)
    metadata = run_metadata(cache)
    roles = source_roles(metadata)
    artifact, prediction_pg = fit_surfaces(cache, metadata, roles)
    artifact["inputs"] = {
        "cache": str(CACHE_PATH),
        "cache_sha256": sha256_file(CACHE_PATH),
        "upstream_timing_artifact": str(TIMING_ARTIFACT),
        "upstream_timing_artifact_sha256": sha256_file(TIMING_ARTIFACT),
    }
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")

    rows = score_runs(cache, metadata, roles, prediction_pg)
    cells = aggregate_cells(rows)
    write_csv(RUN_CSV_PATH, rows)
    write_csv(CELL_CSV_PATH, cells)
    figures = {}
    for metric in METRICS:
        path = BASE / f"clean_power_metric_{metric}.pdf"
        _plot_metric(cells, metric, path, include_comparator_legend=False)
        figures[metric] = str(path)
    bias_path = BASE / "clean_power_signed_bias_by_cell.pdf"
    _plot_signed_bias(cells, bias_path)
    figures["signed_bias_by_cell"] = str(bias_path)

    report = {
        "schema_version": "clean-separated-power-report-v4",
        "status": (
            "source fit plus retrospective stress/architecture transfer; the "
            "legacy repetitions replay overlapping request sequences and are "
            "not development; external request-trace transfer remains pending"
        ),
        "comparison_note": (
            "Previous GPT-OSS-120B values are unsupported dense-surface "
            "comparators, not valid baseline predictions."
        ),
        "run_counts_by_role": {
            role: sum(row["role"] == role for row in rows)
            for role in sorted(set(roles.values()))
        },
        "training_runs": {
            key: sorted(value)
            for key, value in training_run_sets(metadata, roles).items()
        },
        "figures": figures,
        "by_surface_role": aggregate(rows, ("surface", "role")),
        "by_model_role": aggregate(rows, ("hardware", "model", "tp", "role")),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    for path in (*figures.values(), ARTIFACT_PATH, REPORT_PATH,
                 RUN_CSV_PATH, CELL_CSV_PATH):
        print(path)


if __name__ == "__main__":
    main()
