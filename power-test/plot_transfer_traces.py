"""Plot paper-style dense and MoE checkpoint-transfer traces."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path[:0] = [
    str(ROOT), str(ROOT / "feature-test")
]

from evaluation_core import trace_metrics  # noqa: E402
from model.paper_replay import sha256_file  # noqa: E402
from scripts.paper import transfer_core  # noqa: E402

DT_S = 0.25
DENSE_BUNDLE = next(
    (ROOT / "data/runs/sealed_qwen3-14b_a100").glob("*/manifest.json")
).parent
MOE_BUNDLE = next(
    (ROOT / "data/runs/sealed_qwen3-30b-a3b_h100").glob("*/manifest.json")
).parent
ARTIFACT = ROOT / "results/clean_model/powertrace_v1.json"
OUT_DIR = ROOT / "results/paper/appendix"
REPORT = OUT_DIR / "transfer_trace_report.json"
TRACE_CSV = OUT_DIR / "transfer_traces_1s.csv"
CAPTIONS = OUT_DIR / "transfer_trace_captions.tex"

CANDIDATE_CONTRACT = {
    "dense": {
        "state_source": "request_simulation",
        "coefficient_changes": ["pre-request target idle"],
        "evidence_role": "retrospective idle-calibrated transfer diagnostic",
    },
    "moe": {
        "state_source": "request_simulation",
        "coefficient_changes": [
            "pre-request target idle",
            "source-only H100/A100 memory energy-per-work ratio",
            "source-only H100/A100 compute energy-per-work ratio",
        ],
        "evidence_role": "retrospective calibrated transfer diagnostic",
    },
}
STANFORD_RED = "#8C1515"
MEASURED_ALPHA_RANGE = (0.35, 0.78)
PREDICTED_ALPHA_RANGE = (0.50, 0.98)


def add_alpha_gradient_line(axis, x, y, *, color, linewidth, alpha_range, label):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba

    points = np.column_stack((np.asarray(x, float), np.asarray(y, float)))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    colors = np.tile(to_rgba(color), (segments.shape[0], 1))
    colors[:, 3] = np.linspace(*alpha_range, segments.shape[0])
    collection = LineCollection(
        segments, colors=colors, linewidths=linewidth, label=label
    )
    axis.add_collection(collection)
    return collection


def per_gpu_one_second_pair(
    measured_node, predicted_node, *, tp: int, dt_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return matched nonoverlapping one-second per-GPU means."""
    if tp < 1:
        raise ValueError("TP must be positive")
    factor = int(round(1.0 / dt_s))
    if factor < 1 or not np.isclose(factor * dt_s, 1.0):
        raise ValueError("Trace timestep must divide one second")
    measured = np.asarray(measured_node, float)
    predicted = np.asarray(predicted_node, float)
    n = min(measured.size, predicted.size) // factor * factor
    if n == 0:
        raise ValueError("Trace has no complete common one-second interval")
    return (
        measured[:n].reshape(-1, factor).mean(1) / tp,
        predicted[:n].reshape(-1, factor).mean(1) / tp,
    )


def _request_ledger(bundle: Path, timing_fit: dict):
    return transfer_core.request_ledger(bundle, timing_fit)


def _metrics(measured, predicted) -> dict:
    values = trace_metrics(measured, predicted, native_dt=DT_S)
    return {
        key: float(values[key])
        for key in (
            "energy_error_pct", "acf_mae", "acf_r2",
            "soft_dtw_divergence", "nrmse_range",
        )
    }


def replace_coefficient(coefficients, names, name: str, value: float) -> np.ndarray:
    output = np.asarray(coefficients, float).copy()
    output[list(names).index(name)] = value
    return output


def predict_dense_node(design, coefficients, tp: int) -> np.ndarray:
    return np.asarray(design, float) @ np.asarray(coefficients, float) * tp


def dense_transfer(timing_fit: dict, power_fit: dict) -> dict:
    manifest, record, ledger, measured, lo, hi, gaps, timing = _request_ledger(
        DENSE_BUNDLE, timing_fit
    )
    fit = power_fit["dense"][record.hardware]
    design = transfer_core.dense_response(record, ledger, fit)
    zero_shot = predict_dense_node(
        design, fit["coefficients"], record.tp
    )[lo:hi]
    idle, idle_devices, idle_samples = transfer_core.target_idle(record, manifest)
    source_idle = float(fit["coefficients"][fit["feature_names"].index(
        "idle_floor"
    )])
    coefficients = replace_coefficient(
        fit["coefficients"], fit["feature_names"], "idle_floor", idle
    )
    predicted = predict_dense_node(design, coefficients, record.tp)[lo:hi]
    return {
        "kind": "dense",
        "manifest": manifest,
        "record": record,
        "measured": measured,
        "predicted": predicted,
        "interpolated_bins": gaps,
        "timing": timing,
        "metrics": _metrics(measured, predicted),
        "pure_zero_shot_metrics": _metrics(measured, zero_shot),
        "source_idle_w_per_gpu": source_idle,
        "target_idle_w_per_gpu": idle,
        "target_idle_device_medians_w": idle_devices,
        "target_idle_samples": idle_samples,
        "idle_delta_w_per_gpu": idle - source_idle,
        **CANDIDATE_CONTRACT["dense"],
    }


def moe_transfer(timing_fit: dict, power_fit: dict) -> dict:
    manifest, record, ledger, measured, lo, hi, gaps, timing = _request_ledger(
        MOE_BUNDLE, timing_fit
    )
    idle, idle_devices, idle_samples = transfer_core.target_idle(record, manifest)
    fit = power_fit["moe"]["per_model"]["gpt-oss-20b"]
    names = list(fit["feature_names"])
    design = transfer_core.source_moe_design(
        ledger, fit, hardware=record.hardware, tp=record.tp
    )
    ratios = transfer_core.hardware_energy_ratios(power_fit)
    zero_shot = transfer_core.predict_moe(
        design, fit["coefficients"], tp=record.tp, hardware=record.hardware
    )[lo:hi]
    coefficients = replace_coefficient(
        fit["coefficients"], names, "idle", idle
    )
    for name, ratio in ratios.items():
        coefficients[names.index(name)] *= ratio
    predicted = transfer_core.predict_moe(
        design, coefficients, tp=record.tp, hardware=record.hardware
    )[lo:hi]
    return {
        "kind": "moe",
        "manifest": manifest,
        "record": record,
        "measured": measured,
        "predicted": predicted,
        "interpolated_bins": gaps,
        "timing": timing,
        "metrics": _metrics(measured, predicted),
        "pure_zero_shot_metrics": _metrics(measured, zero_shot),
        "target_idle_w_per_gpu": idle,
        "target_idle_device_medians_w": idle_devices,
        "target_idle_samples": idle_samples,
        "hardware_energy_per_work_ratios": ratios,
        **CANDIDATE_CONTRACT["moe"],
    }


def output_stem(row: dict) -> Path:
    record = row["record"]
    model = (
        "qwen3_30b_a3b" if row["kind"] == "moe" else "qwen3_14b"
    )
    return OUT_DIR / (
        f"power_trace_{model}_{record.hardware.lower()}_tp{record.tp}_"
        f"{row['kind']}_transfer_1s"
    )


def save_trace(row: dict, ymax_w: float) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns

    measured, predicted = per_gpu_one_second_pair(
        row["measured"], row["predicted"],
        tp=row["record"].tp, dt_s=DT_S,
    )
    time_min = np.arange(measured.size) / 60.0
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.18)
    fig, axis = plt.subplots(figsize=(4.4, 2.5))
    add_alpha_gradient_line(
        axis, time_min, measured, color="black", linewidth=0.65,
        alpha_range=MEASURED_ALPHA_RANGE, label="Measured",
    )
    add_alpha_gradient_line(
        axis, time_min, predicted, color=STANFORD_RED, linewidth=0.9,
        alpha_range=PREDICTED_ALPHA_RANGE, label="PowerTrace-Sim",
    )
    duration_min = measured.size / 60.0
    axis.set(
        xlabel="Time (min)", ylabel="Power per GPU (W)",
        xlim=(0.0, duration_min), ylim=(0.0, ymax_w),
    )
    axis.grid(True, alpha=0.25)
    axis.legend(
        handles=[
            Line2D([], [], color="black", linewidth=0.65, label="Measured"),
            Line2D(
                [], [], color=STANFORD_RED, linewidth=0.9,
                label="PowerTrace-Sim",
            ),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False,
        ncol=2, fontsize=8, handlelength=1.5, columnspacing=0.8,
        borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.35, rect=(0.0, 0.0, 1.0, 0.92))
    stem = output_stem(row)
    pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return {
        "pdf": str(pdf.relative_to(ROOT)),
        "png": str(png.relative_to(ROOT)),
    }


def caption_text(dense: dict, moe: dict) -> str:
    dm, mm = dense["metrics"], moe["metrics"]
    dense_zero_shot = dense["pure_zero_shot_metrics"]
    zero_shot = moe["pure_zero_shot_metrics"]
    ratios = moe["hardware_energy_per_work_ratios"]
    idle = moe["target_idle_w_per_gpu"]
    return rf"""\begin{{figure*}}[t]
  \centering
  \begin{{subfigure}}[t]{{0.48\textwidth}}
    \centering
    \includegraphics[width=\linewidth]{{{dense['outputs']['pdf']}}}
    \caption{{Retrospective idle-calibrated dense transfer to unseen Qwen3-14B
    on A100 TP1. Pure request-only zero-shot transfer has
    {dense_zero_shot['energy_error_pct']:.2f}\% energy error. The shown trace
    keeps the frozen timing law and every dynamic power coefficient, replacing
    only the source idle floor ({dense['source_idle_w_per_gpu']:.1f}~W/GPU)
    with the declared 60-second pre-request target idle
    ({dense['target_idle_w_per_gpu']:.1f}~W/GPU). No loaded-target power sample
    enters the prediction. It reaches
    {dm['energy_error_pct']:.2f}\% energy error, {dm['nrmse_range']:.3f} range
    NRMSE, and {dm['soft_dtw_divergence']:.4f} normalized Soft-DTW. The mean
    transfers, but ACF $R^2={dm['acf_r2']:.2f}$ shows that high-frequency
    event alignment does not. Because the idle update was applied after the
    sealed result was inspected, it is diagnostic rather than a sealed
    zero-shot claim.}}
    \label{{fig:qwen-dense-transfer}}
  \end{{subfigure}}\hfill
  \begin{{subfigure}}[t]{{0.48\textwidth}}
    \centering
    \includegraphics[width=\linewidth]{{{moe['outputs']['pdf']}}}
    \caption{{Retrospective calibrated MoE transfer to unseen Qwen3-30B-A3B on
    H100 TP2. Pure GPT-OSS-20B zero-shot transfer has
    {zero_shot['energy_error_pct']:.2f}\% energy error. The
    shown trace changes only the idle intercept to the pre-request 60-second
    target measurement ({idle:.1f}~W/GPU) and rescales the frozen memory and
    compute terms by source-only H100/A100 energy-per-work ratios
    ({ratios['logical_memory_util_lag_250ms']:.3f} and
    {ratios['duty_sqrt_exact_compute_util']:.3f}). No loaded-target coefficient
    is fit. Energy error falls to {mm['energy_error_pct']:.2f}\%, with
    {mm['nrmse_range']:.3f} NRMSE, {mm['soft_dtw_divergence']:.4f} Soft-DTW,
    and ACF $R^2={mm['acf_r2']:.3f}$. Because this rule was selected after the
    sealed trace was inspected, it is diagnostic rather than a sealed claim.}}
    \label{{fig:qwen-moe-transfer}}
  \end{{subfigure}}
  \caption{{Architecture-aware transfer of PowerTrace-Sim to unseen dense and
  MoE checkpoints. Black is measured power and red is the request-generated
  prediction; both are matched nonoverlapping one-second per-GPU means over
  the complete common observation interval.}}
  \label{{fig:qwen-transfer-traces}}
\end{{figure*}}
"""


def _report_row(row: dict) -> dict:
    record = row["record"]
    return {
        "run_id": row["manifest"]["run_id"],
        "model": record.model,
        "hardware": record.hardware,
        "tp": record.tp,
        "state_source": row["state_source"],
        "coefficient_changes": row["coefficient_changes"],
        "evidence_role": row["evidence_role"],
        "interpolated_bins": row["interpolated_bins"],
        "timing": row["timing"],
        "metrics": row["metrics"],
        "outputs": row["outputs"],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    release = json.loads(ARTIFACT.read_text())
    timing_fit = release["timing"]
    power_fit = release["power"]
    dense, moe = dense_transfer(timing_fit, power_fit), moe_transfer(
        timing_fit, power_fit
    )
    views = [
        per_gpu_one_second_pair(
            row["measured"], row["predicted"],
            tp=row["record"].tp, dt_s=DT_S,
        )
        for row in (dense, moe)
    ]
    ymax = 1.05 * max(float(np.max(series)) for view in views for series in view)
    for row in (dense, moe):
        row["outputs"] = save_trace(row, ymax)
    CAPTIONS.write_text(caption_text(dense, moe))

    dense_view, moe_view = views
    columns = {
        "dense_time_s": np.arange(dense_view[0].size),
        "dense_measured_w_per_gpu": dense_view[0],
        "dense_predicted_w_per_gpu": dense_view[1],
        "moe_time_s": np.arange(moe_view[0].size),
        "moe_measured_w_per_gpu": moe_view[0],
        "moe_predicted_w_per_gpu": moe_view[1],
    }
    length = max(values.size for values in columns.values())
    with TRACE_CSV.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for index in range(length):
            writer.writerow([
                values[index] if index < values.size else ""
                for values in columns.values()
            ])

    report = {
        "schema_version": "checkpoint-transfer-traces-v1",
        "artifact": {"path": str(ARTIFACT.relative_to(ROOT)),
                     "sha256": sha256_file(ARTIFACT)},
        "plot_contract": (
            "request-generated predictions; common observed interval; "
            "nonoverlapping one-second per-GPU means; common y-axis"
        ),
        "dense": _report_row(dense),
        "moe": {
            **_report_row(moe),
            "target_idle_w_per_gpu": moe["target_idle_w_per_gpu"],
            "hardware_energy_per_work_ratios": (
                moe["hardware_energy_per_work_ratios"]
            ),
            "pure_zero_shot_metrics": moe["pure_zero_shot_metrics"],
        },
        "trace_csv": str(TRACE_CSV.relative_to(ROOT)),
        "captions": str(CAPTIONS.relative_to(ROOT)),
    }
    report["dense"].update({
        "pure_zero_shot_metrics": dense["pure_zero_shot_metrics"],
        "source_idle_w_per_gpu": dense["source_idle_w_per_gpu"],
        "target_idle_w_per_gpu": dense["target_idle_w_per_gpu"],
        "idle_delta_w_per_gpu": dense["idle_delta_w_per_gpu"],
    })
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote transfer panels, {REPORT}, {TRACE_CSV}, and {CAPTIONS}")


if __name__ == "__main__":
    main()
