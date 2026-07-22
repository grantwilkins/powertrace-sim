"""Plot idle-calibrated request-generated traces for three sealed BurstGPT strata."""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path[:0] = [
    str(ROOT), str(ROOT / "timing-test"), str(ROOT / "feature-test"), str(BASE)
]

import plot_transfer_traces as transfer  # noqa: E402
from scripts.paper import transfer_core  # noqa: E402
from scripts.paper.render import save_figure  # noqa: E402

MEASURED_ALPHA_RANGE = transfer.MEASURED_ALPHA_RANGE
PREDICTED_ALPHA_RANGE = transfer.PREDICTED_ALPHA_RANGE
STANFORD_RED = transfer.STANFORD_RED
add_alpha_gradient_line = transfer.add_alpha_gradient_line

BUNDLE_ROOT = ROOT / "data/runs/sealed_burstgpt_qwen3-8b_a100"
OUT_DIR = ROOT / "results/paper/appendix"
REPORT = OUT_DIR / "burstgpt_idle_transfer_report.json"
TRACE_CSV = OUT_DIR / "burstgpt_idle_transfer_1s.csv"
CAPTIONS = OUT_DIR / "burstgpt_idle_transfer_captions.tex"


def stratum_index(source_revision: str) -> int:
    match = re.search(r"fano-stratum:(\d+)/3(?:$|;)", source_revision)
    if match is None:
        raise ValueError("BurstGPT source revision lacks a 3-way Fano stratum")
    return int(match.group(1))


def bundle_dirs() -> list[Path]:
    rows = []
    for path in BUNDLE_ROOT.glob("*/manifest.json"):
        manifest = json.loads(path.read_text())
        rows.append((stratum_index(manifest["probe"]["source_revision"]), path.parent))
    if [index for index, _ in sorted(rows)] != [0, 1, 2]:
        raise ValueError("Expected exactly BurstGPT Fano strata 0, 1, and 2")
    return [path for _, path in sorted(rows)]


def evaluate_run(bundle: Path, timing_fit: dict, power_fit: dict) -> dict:
    manifest, record, ledger, measured, lo, hi, gaps, timing = (
        transfer._request_ledger(bundle, timing_fit)
    )
    fit = power_fit["dense"][record.hardware]
    design = transfer_core.dense_response(record, ledger, fit)
    zero_shot = transfer.predict_dense_node(
        design, fit["coefficients"], record.tp
    )[lo:hi]
    idle, devices, samples = transfer_core.target_idle(record, manifest)
    source_idle = float(fit["coefficients"][fit["feature_names"].index(
        "idle_floor"
    )])
    coefficients = transfer.replace_coefficient(
        fit["coefficients"], fit["feature_names"], "idle_floor", idle
    )
    predicted = transfer.predict_dense_node(
        design, coefficients, record.tp
    )[lo:hi]
    return {
        "bundle": bundle,
        "manifest": manifest,
        "record": record,
        "stratum": stratum_index(manifest["probe"]["source_revision"]),
        "requests": int(record.input_lens.size),
        "measured": measured,
        "predicted": predicted,
        "zero_shot_metrics": transfer._metrics(measured, zero_shot),
        "metrics": transfer._metrics(measured, predicted),
        "timing": timing,
        "source_idle_w_per_gpu": source_idle,
        "target_idle_w_per_gpu": idle,
        "target_idle_device_medians_w": devices,
        "target_idle_samples": samples,
        "idle_delta_w_per_gpu": idle - source_idle,
        "interpolated_bins": gaps,
        "state_source": "exact BurstGPT replay marks through request simulation",
        "coefficient_changes": ["pre-request target idle"],
        "evidence_role": "retrospective idle-calibrated sealed-trace diagnostic",
    }


def one_second(row: dict) -> tuple[np.ndarray, np.ndarray]:
    return transfer.per_gpu_one_second_pair(
        row["measured"], row["predicted"], tp=row["record"].tp,
        dt_s=transfer.DT_S,
    )


def output_stem(row: dict) -> Path:
    return OUT_DIR / (
        "power_trace_burstgpt_qwen3_8b_a100_tp1_"
        f"stratum_{row['stratum']}_idle_1s"
    )


def save_trace(row: dict, ymax_w: float) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns

    measured, predicted = one_second(row)
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
    axis.set(
        xlabel="Time (min)", ylabel="Power per GPU (W)",
        xlim=(0.0, measured.size / 60.0), ylim=(0.0, ymax_w),
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
    save_figure(fig, pdf, bbox_inches="tight")
    save_figure(fig, png, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return {
        "pdf": str(pdf.relative_to(ROOT)),
        "png": str(png.relative_to(ROOT)),
    }


def caption_text(rows: list[dict]) -> str:
    subfigures = []
    for row in rows:
        metric, zero = row["metrics"], row["zero_shot_metrics"]
        subfigures.append(rf"""  \begin{{subfigure}}[t]{{0.32\textwidth}}
    \centering
    \includegraphics[width=\linewidth]{{{row['outputs']['pdf']}}}
    \caption{{Fano stratum {row['stratum']} ({row['requests']:,} requests).
    Replacing only the source idle floor
    ({row['source_idle_w_per_gpu']:.1f}~W/GPU) with the run's declared
    pre-request idle ({row['target_idle_w_per_gpu']:.1f}~W/GPU) changes energy
    error from {zero['energy_error_pct']:.2f}\% to
    {metric['energy_error_pct']:.2f}\%, with {metric['nrmse_range']:.3f}
    NRMSE, {metric['soft_dtw_divergence']:.4f} Soft-DTW, and ACF
    $R^2={metric['acf_r2']:.3f}$.}}
    \label{{fig:burstgpt-stratum-{row['stratum']}}}
  \end{{subfigure}}""")
    energy = [row["metrics"]["energy_error_pct"] for row in rows]
    max_nrmse = max(row["metrics"]["nrmse_range"] for row in rows)
    max_dtw = max(row["metrics"]["soft_dtw_divergence"] for row in rows)
    return """\\begin{figure*}[t]
  \\centering
""" + "\\hfill\n".join(subfigures) + rf"""
  \caption{{Transfer to exact BurstGPT arbitrary-arrival replays on Qwen3-8B
  A100 TP1. Black is measured power and red is PowerTrace-Sim from the exact
  replay marks; both are matched one-second per-GPU means over the complete
  common interval. The timing and dynamic power laws remain frozen, and each
  panel changes only its separately measured 60-second pre-request idle
  intercept. Across all three disjoint strata, idle-calibrated energy error is
  {min(energy):.2f}--{max(energy):.2f}\%, range NRMSE is at most
  {max_nrmse:.3f}, and Soft-DTW is at most {max_dtw:.4f}.
  ACF agreement is strong for strata 1--2 but only moderate for stratum 0, where
  the model underestimates the largest power pulses. Because the idle rule was
  applied after these sealed targets were inspected, this is retrospective
  arbitrary-arrival evidence rather than a sealed zero-shot claim.}}
  \label{{fig:burstgpt-idle-transfer}}
\end{{figure*}}
"""


def _report_row(row: dict) -> dict:
    return {
        key: row[key]
        for key in (
            "stratum", "requests", "state_source", "coefficient_changes",
            "evidence_role", "source_idle_w_per_gpu", "target_idle_w_per_gpu",
            "idle_delta_w_per_gpu", "interpolated_bins", "timing",
            "zero_shot_metrics", "metrics", "outputs",
        )
    } | {"run_id": row["manifest"]["run_id"]}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    release = json.loads(transfer.ARTIFACT.read_text())
    timing_fit = release["timing"]
    power_fit = release["power"]
    rows = [evaluate_run(path, timing_fit, power_fit) for path in bundle_dirs()]
    views = [one_second(row) for row in rows]
    ymax = 1.05 * max(float(np.max(series)) for view in views for series in view)
    for row in rows:
        row["outputs"] = save_trace(row, ymax)
    CAPTIONS.write_text(caption_text(rows))

    columns = {}
    for row, (measured, predicted) in zip(rows, views):
        prefix = f"stratum_{row['stratum']}"
        columns[f"{prefix}_time_s"] = np.arange(measured.size)
        columns[f"{prefix}_measured_w_per_gpu"] = measured
        columns[f"{prefix}_predicted_w_per_gpu"] = predicted
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
        "schema_version": "burstgpt-idle-transfer-v1",
        "artifact": {
            "path": str(transfer.ARTIFACT.relative_to(ROOT)),
            "sha256": transfer.sha256_file(transfer.ARTIFACT),
        },
        "evidence_role": "retrospective idle-calibrated sealed-trace diagnostic",
        "claim_boundary": (
            "exact arbitrary-arrival replay marks; frozen timing/dynamic power; "
            "per-run pre-request idle intercept only"
        ),
        "runs": [_report_row(row) for row in rows],
        "outputs": {
            "trace_csv": str(TRACE_CSV.relative_to(ROOT)),
            "captions": str(CAPTIONS.relative_to(ROOT)),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote 3 BurstGPT panels, {REPORT}, {TRACE_CSV}, and {CAPTIONS}")


if __name__ == "__main__":
    main()
