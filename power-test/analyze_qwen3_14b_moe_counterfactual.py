"""Apply frozen MoE response laws to the dense Qwen3-14B ledger diagnostically."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path[:0] = [
    str(ROOT), str(ROOT / "timing-test"), str(ROOT / "feature-test"), str(BASE)
]

import analyze_qwen3_30b_moe_transfer as moe_audit  # noqa: E402
import plot_transfer_traces as transfer  # noqa: E402
from response_chain import apply_chain  # noqa: E402

PLOT_PNG = BASE / "qwen3_14b_moe_counterfactual.png"
PLOT_PDF = BASE / "qwen3_14b_moe_counterfactual.pdf"
REPORT = BASE / "qwen3_14b_moe_counterfactual.json"
TRACE_CSV = BASE / "qwen3_14b_moe_counterfactual_1s.csv"
COLORS = ("#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#009E73", "#CC79A7")


def apply_source_law(
    design, coefficients, *, tp: int, hardware: str, dt_s: float
) -> np.ndarray:
    """Apply source coefficients and only the target meter observation model."""
    node_power = np.asarray(design, float) @ np.asarray(coefficients, float) * tp
    return apply_chain(node_power, dt_s, hardware, 0.0)


def candidate_contract(source_model: str, idle_updated: bool) -> dict:
    return {
        "source_model": source_model,
        "source_hardware": "A100",
        "target_family": "dense-14b",
        "surface_supported": False,
        "support_reason": (
            "Frozen architecture-specific MoE response law applied across the "
            "MoE-to-dense family boundary"
        ),
        "coefficient_changes": ["pre-request target idle"] if idle_updated else [],
        "evidence_role": "retrospective unsupported cross-family diagnostic",
    }


def build_candidates(timing_fit: dict, power_fit: dict) -> tuple[dict, dict]:
    manifest, record, ledger, measured, lo, hi, gaps, timing = (
        transfer._request_ledger(transfer.DENSE_BUNDLE, timing_fit)
    )
    idle, devices, samples = moe_audit.target_idle_w_per_gpu(record, manifest)
    dense = transfer.dense_transfer(timing_fit, power_fit)
    candidates = {
        "dense zero-shot": {
            "prediction": dense["predicted"]
            - dense["idle_delta_w_per_gpu"] * record.tp,
            "contract": {
                "surface_supported": True,
                "coefficient_changes": [],
                "evidence_role": "sealed zero-shot dense comparator",
            },
        },
        "dense updated idle": {
            "prediction": dense["predicted"],
            "contract": {
                "surface_supported": True,
                "coefficient_changes": ["pre-request target idle"],
                "evidence_role": "retrospective dense comparator",
            },
        },
    }
    for model, fit in power_fit["moe"]["per_model"].items():
        design, names = moe_audit.design_for_source_fit(ledger, fit)
        base = np.asarray(fit["coefficients"], float)
        for suffix, coefficients, updated in (
            ("zero-shot", base, False),
            ("updated idle", moe_audit.replace_idle(base, names, idle), True),
        ):
            label = f"{model} {suffix}"
            candidates[label] = {
                "prediction": apply_source_law(
                    design, coefficients, tp=record.tp,
                    hardware=record.hardware, dt_s=transfer.DT_S,
                )[lo:hi],
                "contract": candidate_contract(model, updated),
            }
    for row in candidates.values():
        row["metrics"] = transfer._metrics(measured, row["prediction"])
    context = {
        "manifest": manifest,
        "record": record,
        "measured": measured,
        "interpolated_bins": gaps,
        "timing": timing,
        "target_idle_w_per_gpu": idle,
        "target_idle_device_medians_w": devices,
        "target_idle_samples": samples,
    }
    return context, candidates


def _one_second(context: dict, prediction) -> tuple[np.ndarray, np.ndarray]:
    return transfer.per_gpu_one_second_pair(
        context["measured"], prediction, tp=context["record"].tp,
        dt_s=transfer.DT_S,
    )


def save_plot(context: dict, candidates: dict) -> None:
    measured_1s, _ = _one_second(context, next(iter(candidates.values()))["prediction"])
    time_s = np.arange(measured_1s.size)
    views = [_one_second(context, row["prediction"])[1] for row in candidates.values()]
    ymin = 0.95 * min(float(np.min(measured_1s)), *(float(np.min(v)) for v in views))
    ymax = 1.05 * max(float(np.max(measured_1s)), *(float(np.max(v)) for v in views))
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True, sharey=True)
    for axis, (label, row), color in zip(axes.flat, candidates.items(), COLORS):
        predicted_1s = _one_second(context, row["prediction"])[1]
        axis.plot(time_s, measured_1s, color="black", linewidth=1.2)
        axis.plot(time_s, predicted_1s, color=color, linewidth=1.1)
        metric = row["metrics"]
        support = "supported" if row["contract"]["surface_supported"] else "unsupported"
        axis.set_title(
            f"{label} ({support})\nenergy {metric['energy_error_pct']:.1f}% | "
            f"ACF R² {metric['acf_r2']:.3f} | NRMSE {metric['nrmse_range']:.3f}"
        )
        axis.grid(alpha=0.2)
        axis.set_ylim(ymin, ymax)
    for axis in axes[-1]:
        axis.set_xlabel("Seconds from first request")
    for axis in axes[:, 0]:
        axis.set_ylabel("Power per GPU (W)")
    fig.suptitle("Qwen3-14B dense-to-MoE response-law counterfactual", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(PLOT_PNG, dpi=180)
    fig.savefig(PLOT_PDF)
    plt.close(fig)


def main() -> None:
    timing_fit = json.loads(transfer.TIMING_FIT.read_text())
    power_fit = json.loads(transfer.POWER_FIT.read_text())
    context, candidates = build_candidates(timing_fit, power_fit)
    save_plot(context, candidates)

    measured_1s, _ = _one_second(
        context, next(iter(candidates.values()))["prediction"]
    )
    columns = {
        "time_s": np.arange(measured_1s.size),
        "measured_w_per_gpu": measured_1s,
    }
    columns.update({
        label: _one_second(context, row["prediction"])[1]
        for label, row in candidates.items()
    })
    with TRACE_CSV.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))

    report = {
        "schema_version": "qwen3-14b-moe-counterfactual-v1",
        "evidence_role": "retrospective unsupported cross-family diagnostic",
        "run_id": context["manifest"]["run_id"],
        "target_model": context["record"].model,
        "target_family": context["record"].arch["family"],
        "target_hardware": context["record"].hardware,
        "target_tp": context["record"].tp,
        "target_idle_w_per_gpu": context["target_idle_w_per_gpu"],
        "timing": context["timing"],
        "candidates": {
            label: {**row["contract"], **row["metrics"]}
            for label, row in candidates.items()
        },
        "diagnosis": (
            "GPT-OSS-20B improves ACF-profile agreement but creates false deep "
            "power troughs and worse range NRMSE; the MoE response basis is not "
            "a valid replacement for the dense law."
        ),
        "outputs": {
            "plot_png": str(PLOT_PNG.relative_to(ROOT)),
            "plot_pdf": str(PLOT_PDF.relative_to(ROOT)),
            "trace_csv": str(TRACE_CSV.relative_to(ROOT)),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {PLOT_PNG}, {PLOT_PDF}, {REPORT}, and {TRACE_CSV}")


if __name__ == "__main__":
    main()
