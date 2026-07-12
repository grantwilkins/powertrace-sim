"""Build one strict conditional-timing deployment artifact per hardware."""

import resource
import time

from model.classifiers.physics import (
    SELECTED_SCHEMA_VERSION,
    physics_feature_order,
    predict_selected_physics,
)
from model.training_data.arch import get_arch
from gmm_bigru_baseline import predict_s0_gmm_bigru


def selected_provenance_ids(hardware, splits, rows_by_id):
    """Return source-only selection IDs and the S0 production holdout IDs."""
    hardware_splits = [
        split for split in splits
        if rows_by_id[split["train"][0]]["hardware"] == hardware
    ]
    selection = sorted({
        rows_by_id[run]["source_id"]
        for split in hardware_splits for run in split["train"] + split["development"]
    })
    s0 = next(split for split in hardware_splits if split["name"] == f"S0_{hardware}")
    excluded = sorted(rows_by_id[run]["source_id"] for run in s0["test"])
    return selection, excluded


def selected_physics_artifact(
    hardware, candidate, fit, rows, *, selection_source_ids=None,
    excluded_target_source_ids=(), dt_s=0.25,
):
    """Convert an S0 source-selected mean fit to the production schema."""
    modes = {
        "B3": ("M0", False), "M0": ("M0", False),
        "M0b": ("M0b", False), "M0bR": ("M0b", True),
        "M0dR": ("M0d", True), "M0c": ("M0c", True), "M4A": ("M4A", True),
    }
    if candidate not in modes:
        raise ValueError(f"Candidate {candidate} has no deployable physics-only schema")
    mean_kind, residence = modes[candidate]
    state_filters = ([{"name": "A_fast", "ema_alpha": 0.03},
                      {"name": "A_slow", "ema_alpha": 0.5}]
                     if mean_kind == "M4A" else [])
    names = physics_feature_order(
        mean_kind, residence=residence,
        state_filter_names=tuple(item["name"] for item in state_filters),
    )
    scales = {
        "A100": (2e12, 312e12, 600e9),
        "H100": (3.35e12, 990e12, 900e9),
    }
    bandwidth, peak, link = scales[hardware]
    source_ids = list(fit["training_source_ids"])
    selection_ids = source_ids if selection_source_ids is None else list(selection_source_ids)
    models = sorted({row["model"] for row in rows if row["hardware"] == hardware})
    calibrated_tp = sorted({int(row["tp"]) for row in rows if row["hardware"] == hardware})
    return {
        "schema_version": SELECTED_SCHEMA_VERSION,
        "hardware": hardware,
        "timing_contract": "conditional_timing",
        "dt_s": float(dt_s),
        "architectures": {model: get_arch(model) for model in models},
        "parallelism_support": {"kind": "TP_only", "calibrated_tp": calibrated_tp},
        "hardware_profile": {
            "hbm_bandwidth_bytes_s": bandwidth,
            "compute_peak_flops_s": peak,
            "link_bandwidth_bytes_s": link,
            "residence_bytes_per_gpu": 80e9,
        },
        "mode": {
            "mean_kind": mean_kind,
            "residence": residence,
            "state_filters": state_filters,
            "coefficients": dict(zip(names, map(float, fit["physics"]))),
            "lag": {
                "moving_average_s": float(fit.get(
                    "moving_average_s", float(dt_s) * (2 if hardware == "H100" else 1))),
                "ema_alpha": float(fit["lag_alpha"]),
            },
            "cap_w_per_gpu": float(fit["cap_w_per_gpu"]),
        },
        "learned_scalar_count": len(names) + 2 + len(state_filters),
        "provenance": {
            "role": "production_refit_not_transfer_evidence",
            "training_source_ids": source_ids,
            "selection_source_ids": selection_ids,
            "excluded_target_source_ids": list(excluded_target_source_ids),
        },
    }


def benchmark_selected_physics(artifact, data, run_id, *, repeats=3):
    """Measure production-kernel throughput on one unchanged native-grid run."""
    mask = data["run_id"] == run_id
    model = str(data["model_names"][int(data["model_idx"][mask][0])])
    tp = int(data["tp"][mask][0])
    ledger = {key: data[key][mask] for key in (
        "pre_tok", "dec_tok", "w_read", "w_read_dec", "kv_read",
        "w_read_pre", "kv_write", "comm", "A_t",
    )}
    predict_selected_physics(
        ledger, artifact["architectures"][model], tp=tp,
        hardware=artifact["hardware"], artifact=artifact,
        dt_s=float(data["dt_s"]), run_ids=data["run_id"][mask],
    )
    started = time.perf_counter()
    for _ in range(repeats):
        predict_selected_physics(
            ledger, artifact["architectures"][model], tp=tp,
            hardware=artifact["hardware"], artifact=artifact,
            dt_s=float(data["dt_s"]), run_ids=data["run_id"][mask],
        )
    elapsed = time.perf_counter() - started
    return {"bins": int(mask.sum()), "seconds": elapsed / repeats,
            "bins_per_s": float(mask.sum() * repeats / elapsed)}


def benchmark_deployments(specs, data, splits, fits):
    """Compare each hardware-local production artifact with its S0 B2 model."""
    output = []
    config = data["model_idx"].astype(int) * 100 + data["tp"].astype(int)
    for hardware, (candidate, artifact, artifact_path) in specs.items():
        split = next(item for item in splits if item["name"] == f"S0_{hardware}")
        run_id, repeats = split["test"][0], 3
        selected = benchmark_selected_physics(artifact, data, run_id, repeats=repeats)
        b2_fit = fits[(split["name"], "B2")]
        started = time.perf_counter()
        for _ in range(repeats):
            predict_s0_gmm_bigru(
                data["run_id"], config, data["A_t"], data["delta_A_t"],
                (run_id,), b2_fit,
            )
        b2_seconds = (time.perf_counter() - started) / repeats
        artifact_bytes = artifact_path.stat().st_size
        row = {"hardware": hardware, "selected_candidate": candidate, **selected,
               "artifact_bytes": artifact_bytes,
               "b2_bins_per_s": selected["bins"] / b2_seconds,
               "b2_parameter_bytes": b2_fit["parameter_count"] * 4,
               "speedup_over_b2": b2_seconds / selected["seconds"],
               "size_reduction_over_b2": b2_fit["parameter_count"] * 4 / artifact_bytes,
               "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        row["passes_10x_speed_and_size"] = (
            row["speedup_over_b2"] >= 10 and row["size_reduction_over_b2"] >= 10
        )
        output.append(row)
    return output
