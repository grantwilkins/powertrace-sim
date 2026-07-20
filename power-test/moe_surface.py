"""Fit and score the support-limited GPT-OSS MoE power surface.

The model consumes request-derived timing-ledger channels and predicts TP-summed
node power. It never modifies or refits the dense power artifact.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / "feature-test"))

from evaluation_core import trace_metrics  # noqa: E402
from fit_power_surface import (  # noqa: E402
    coefficients_in_design_order,
    interpolate_nan,
    run_slices,
)
from moe_surface_core import (  # noqa: E402
    BIN_S,
    DELAY_S,
    FEATURE_NAMES,
    HARDWARE,
    HBM_BYTES_S,
    MODEL,
    MOE_MODELS,
    ROUTING_MODE,
    SCHEMA,
    SUPPORTED_TP,
    TDP_W_PER_GPU,
    apply_by_run,
    canonical_design_digest,
    fit_coefficients,
    hardware_names_for_rows,
    load_cache,
    load_run_index,
    load_split_roles,
    predict,
    routing_mode,
    run_metadata,
    sha256_file,
    surface_design,
    supported_run,
    target_digest,
    training_runs,
    validate_artifact_contract,
    validate_cache,
)
from power_surface import predict as predict_dense  # noqa: E402
from power_surface import surface_design as dense_design  # noqa: E402

MODEL_CODE_PATHS = (
    BASE / "moe_surface_core.py",
    BASE / "response_chain.py",
)
EVALUATOR_CODE_PATHS = (
    BASE / "moe_surface.py",
    *MODEL_CODE_PATHS,
    BASE / "fit_power_surface.py",
    BASE / "power_surface.py",
    ROOT / "feature-test" / "evaluation_core.py",
)


def code_hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def score_rows(cache: dict, prediction: np.ndarray, baseline: np.ndarray,
               roles: set[str], applied_only: bool | None = None) -> list[dict]:
    metadata = run_metadata(cache)
    dt = float(cache["dt_s"])
    mode = routing_mode(cache)
    rows = []
    for run, lo, hi in run_slices(cache["run_id"]):
        meta = metadata[run]
        if meta["model"] not in MOE_MODELS or meta["role"] not in roles:
            continue
        applied = supported_run(
            meta["model"], meta["hardware"], meta["tp"], mode)
        if applied_only is not None and applied != applied_only:
            continue
        measured, n_bad = interpolate_nan(
            cache["power"][lo:hi].astype(float))
        selected = prediction[lo:hi] if applied else baseline[lo:hi]
        metrics = trace_metrics(measured, selected, native_dt=dt)
        base_metrics = trace_metrics(measured, baseline[lo:hi], native_dt=dt)
        row = {
            "run_id": run,
            **meta,
            "moe_surface_applied": applied,
            "interpolated_power_bins": n_bad,
        }
        for key in ("energy_error_pct", "acf_mae", "acf_r2", "nrmse_range"):
            row[key] = float(metrics[key])
            row[f"baseline_{key}"] = float(base_metrics[key])
            row[f"delta_{key}"] = float(metrics[key] - base_metrics[key])
        rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    keys = sorted({
        (row["model"], row["role"], row["moe_surface_applied"])
        for row in rows
    })
    for model, role, applied in keys:
        selected = [
            row for row in rows
            if (row["model"], row["role"], row["moe_surface_applied"])
            == (model, role, applied)
        ]
        record = {
            "model": model,
            "role": role,
            "moe_surface_applied": applied,
            "runs": len(selected),
        }
        for prefix in ("", "baseline_"):
            for metric in (
                "energy_error_pct", "acf_mae", "acf_r2", "nrmse_range",
            ):
                values = np.asarray(
                    [row[f"{prefix}{metric}"] for row in selected], float)
                record[f"{prefix}{metric}_median"] = float(
                    np.nanmedian(values))
                record[f"{prefix}{metric}_p90"] = float(
                    np.nanpercentile(values, 90))
        output.append(record)
    return output


def baseline_prediction(cache: dict, artifact: dict) -> np.ndarray:
    output = np.full(cache["run_id"].shape, np.nan)
    dt = float(cache["dt_s"])
    hardware_rows = hardware_names_for_rows(cache)
    for hardware in sorted(set(hardware_rows)):
        selected = hardware_rows == hardware
        sub = {
            key: value[selected]
            for key, value in cache.items()
            if isinstance(value, np.ndarray)
            and value.shape == selected.shape
        }
        design, names = dense_design(sub, hardware)
        fit = artifact["per_hardware"][hardware]
        coefficients = coefficients_in_design_order(fit, names)
        raw = predict_dense(design, coefficients, sub["tp"], hardware)
        output[selected] = apply_by_run(
            raw, sub["run_id"], dt, hardware, float(fit["delay_s"]))
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_command(args: argparse.Namespace) -> None:
    cache = load_cache(args.cache)
    index = load_run_index(args.run_index)
    split_roles = load_split_roles(args.split_manifest)
    metadata = validate_cache(cache, index, split_roles)
    fit_runs = training_runs(metadata)
    if len(fit_runs) != 20:
        raise ValueError(f"Expected 20 GPT-OSS-20B training runs, got {len(fit_runs)}")
    for path in (args.artifact, args.selection_manifest, args.dev_report):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite versioned output {path}")
    design = surface_design(cache)
    coefficients = fit_coefficients(
        design, cache["power"], cache["run_id"], fit_runs)
    artifact = {
        "schema_version": SCHEMA,
        "model": MODEL,
        "hardware": HARDWARE,
        "supported_tp": list(SUPPORTED_TP),
        "routing_mode": ROUTING_MODE,
        "target_units": "TP-summed node watts",
        "feature_names": list(FEATURE_NAMES),
        "coefficients": coefficients.tolist(),
        "response_delay_s": DELAY_S,
        "input_bin_s": BIN_S,
        "hbm_bytes_s": HBM_BYTES_S,
        "tdp_w_per_gpu": TDP_W_PER_GPU,
        "fit_role": "train",
        "fit_run_ids": sorted(fit_runs),
        "fit_source_ids": [index[run]["source_id"] for run in sorted(fit_runs)],
        "design_digest": canonical_design_digest(cache, fit_runs),
        "training_target_digest": target_digest(cache, fit_runs),
        "run_index_sha256": sha256_file(args.run_index),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "join_provenance_sha256": sha256_file(args.cache_provenance),
        "dense_comparator_sha256": sha256_file(args.baseline_artifact),
        "provenance": {
            "surface": "MoE-only run-balanced NNLS",
            "features": "causal request/timing-ledger coordinates",
            "feature_selection": (
                "retrospective train/development sweep; five-feature "
                "surface chosen after dropping zero/low-utility coordinates"),
            "memory_alignment": (
                "logical-memory coordinate shifted one 250 ms ledger bin "
                "with a first-value hold at each run boundary; all other "
                "coordinates remain instantaneous"),
            "response_delay": "none",
            "measured_routing": "rejected on source development timing and power",
            "holdout_power_used_for_fit": False,
            "rate4_used_for_model_selection": (
                "yes; retrospective iteration, not a sealed holdout claim"),
        },
        "model_code_sha256": code_hashes(MODEL_CODE_PATHS),
    }
    dense_artifact = json.loads(args.baseline_artifact.read_text())
    raw = predict(design, coefficients, cache["tp"])
    moe_prediction = apply_by_run(
        raw, cache["run_id"], float(cache["dt_s"]), HARDWARE, DELAY_S)
    baseline = baseline_prediction(cache, dense_artifact)
    rows = score_rows(
        cache, moe_prediction, baseline, {"test_indomain"}, applied_only=True)
    report = {
        "schema_version": "moe-power-development-v3",
        "evaluation_target_digest": target_digest(
            cache, {row["run_id"] for row in rows}),
        "evaluation_design_digest": canonical_design_digest(
            cache, {row["run_id"] for row in rows}),
        "join_provenance_sha256": sha256_file(args.cache_provenance),
        "evaluator_code_sha256": code_hashes(EVALUATOR_CODE_PATHS),
        "selection_status": (
            "retrospective model development; rate4 results were already opened"),
        "tables": aggregate(rows),
        "per_run": rows,
    }
    args.artifact.write_text(json.dumps(artifact, indent=2) + "\n")
    args.selection_manifest.write_text(json.dumps({
        "schema_version": "moe-power-selection-v3",
        "artifact_sha256": sha256_file(args.artifact),
        "design_digest": artifact["design_digest"],
        "training_target_digest": artifact["training_target_digest"],
        "split_manifest_sha256": artifact["split_manifest_sha256"],
    }, indent=2) + "\n")
    report["artifact_sha256"] = sha256_file(args.artifact)
    args.dev_report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.artifact} and {args.dev_report}")
    for row in report["tables"]:
        print(
            f"{row['model']} {row['role']}: "
            f"energy={row['energy_error_pct_median']:.2f}% "
            f"acf_r2={row['acf_r2_median']:.3f} "
            f"nrmse={row['nrmse_range_median']:.3f}")


def validate_artifact(artifact: dict, cache: dict, args: argparse.Namespace,
                      metadata: dict[int, dict], index: dict[int, dict]) -> None:
    validate_artifact_contract(artifact)
    if artifact.get("model_code_sha256") != code_hashes(MODEL_CODE_PATHS):
        raise ValueError("MoE artifact does not match the model implementation")
    fit_runs = training_runs(metadata)
    if artifact.get("fit_run_ids") != sorted(fit_runs):
        raise ValueError("MoE artifact training split mismatch")
    if artifact.get("fit_source_ids") != [
            index[run]["source_id"] for run in sorted(fit_runs)]:
        raise ValueError("MoE artifact training sources mismatch")
    if artifact["design_digest"] != canonical_design_digest(cache, fit_runs):
        raise ValueError("MoE artifact does not match the cache design")
    if artifact["training_target_digest"] != target_digest(cache, fit_runs):
        raise ValueError("MoE artifact does not match training targets")
    if artifact["run_index_sha256"] != sha256_file(args.run_index):
        raise ValueError("MoE artifact does not match the run index")
    if artifact["split_manifest_sha256"] != sha256_file(args.split_manifest):
        raise ValueError("MoE artifact does not match the split manifest")
    if artifact["join_provenance_sha256"] != sha256_file(args.cache_provenance):
        raise ValueError("MoE artifact does not match joined-power provenance")
    if artifact["dense_comparator_sha256"] != sha256_file(args.baseline_artifact):
        raise ValueError("MoE artifact does not match the dense comparator")


def score_command(args: argparse.Namespace) -> None:
    cache = load_cache(args.cache)
    index = load_run_index(args.run_index)
    metadata = validate_cache(
        cache, index, load_split_roles(args.split_manifest))
    artifact = json.loads(args.artifact.read_text())
    validate_artifact(artifact, cache, args, metadata, index)
    selection = json.loads(args.selection_manifest.read_text())
    expected_selection = {
        "schema_version": "moe-power-selection-v3",
        "artifact_sha256": sha256_file(args.artifact),
        "design_digest": artifact["design_digest"],
        "training_target_digest": artifact["training_target_digest"],
        "split_manifest_sha256": artifact["split_manifest_sha256"],
    }
    if selection != expected_selection:
        raise ValueError("Selection manifest does not freeze this artifact")
    dense_artifact = json.loads(args.baseline_artifact.read_text())
    design = surface_design(cache)
    coefficients = np.asarray(artifact["coefficients"], float)
    raw = predict(design, coefficients, cache["tp"])
    moe_prediction = apply_by_run(
        raw, cache["run_id"], float(cache["dt_s"]), HARDWARE,
        float(artifact["response_delay_s"]))
    baseline = baseline_prediction(cache, dense_artifact)
    roles = {item.strip() for item in args.roles.split(",") if item.strip()}
    if roles != {"holdout_rate", "holdout_model"}:
        raise ValueError("Score roles are frozen to holdout_rate,holdout_model")
    rows = score_rows(cache, moe_prediction, baseline, roles)
    if args.out_json.exists() or args.out_csv.exists():
        raise FileExistsError("Refusing to overwrite retrospective score outputs")
    applied = [row for row in rows if row["moe_surface_applied"]]
    comparators = [row for row in rows if not row["moe_surface_applied"]]
    report = {
        "schema_version": "moe-power-retrospective-v3",
        "artifact_sha256": sha256_file(args.artifact),
        "evaluation_status": (
            "retrospective stress evaluation; roles retain historical names"),
        "roles": sorted(roles),
        "evaluation_target_digest": target_digest(
            cache, {row["run_id"] for row in rows}),
        "evaluation_design_digest": canonical_design_digest(
            cache, {row["run_id"] for row in rows}),
        "join_provenance_sha256": sha256_file(args.cache_provenance),
        "evaluator_code_sha256": code_hashes(EVALUATOR_CODE_PATHS),
        "applied_runs": len(applied),
        "unsupported_comparator_runs": len(comparators),
        "tables": aggregate(applied),
        "unsupported_baseline_comparators": aggregate(comparators),
        "per_run": rows,
    }
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    write_csv(args.out_csv, rows)
    print(f"Wrote {args.out_json} and {args.out_csv}")
    for row in report["tables"]:
        mode = "surface" if row["moe_surface_applied"] else "unsupported comparator"
        print(
            f"{row['model']} {row['role']} {mode}: "
            f"energy={row['energy_error_pct_median']:.2f}% "
            f"acf_r2={row['acf_r2_median']:.3f} "
            f"nrmse={row['nrmse_range_median']:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--cache", type=Path,
        default=BASE / "sim_ledger_power_uniform_current_250ms.npz")
    common.add_argument(
        "--run-index", type=Path,
        default=ROOT / "timing-test" / "timing_dataset.runs.json")
    common.add_argument(
        "--split-manifest", type=Path,
        default=ROOT / "timing-test" / "split_manifest_fp8.json")
    common.add_argument(
        "--cache-provenance", type=Path,
        default=BASE / "sim_ledger_power_uniform_current_250ms.provenance.json")
    common.add_argument(
        "--baseline-artifact", type=Path,
        default=BASE / "fitted_surface.json")

    fit = subparsers.add_parser("fit", parents=[common])
    fit.add_argument(
        "--artifact", type=Path,
        default=BASE / "fitted_moe_surface_v3.json")
    fit.add_argument(
        "--selection-manifest", type=Path,
        default=BASE / "moe_surface_selection_v3.json")
    fit.add_argument(
        "--dev-report", type=Path,
        default=BASE / "moe_surface_dev_v3.json")
    fit.set_defaults(func=fit_command)

    score = subparsers.add_parser("score", parents=[common])
    score.add_argument(
        "--artifact", type=Path,
        default=BASE / "fitted_moe_surface_v3.json")
    score.add_argument(
        "--selection-manifest", type=Path,
        default=BASE / "moe_surface_selection_v3.json")
    score.add_argument("--roles", default="holdout_rate,holdout_model")
    score.add_argument(
        "--out-json", type=Path,
        default=BASE / "moe_surface_stress_v3.json")
    score.add_argument(
        "--out-csv", type=Path,
        default=BASE / "moe_surface_stress_v3.csv")
    score.set_defaults(func=score_command)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
