"""Regenerate the retrospective OpenHands platform-calibration appendix panel."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path[:0] = [str(ROOT), str(BASE)]

import plot_transfer_traces as transfer  # noqa: E402
from model.power.response import apply_response  # noqa: E402
from scripts.paper import transfer_core  # noqa: E402
from scripts.paper.render import save_figure  # noqa: E402

BUNDLE_ROOT = ROOT / "data/runs/sealed_openhands_qwen3-8b_a100"
OUT_DIR = ROOT / "results/paper/appendix"
REPORT = OUT_DIR / "openhands_platform_calibration.json"
SAMPLES = OUT_DIR / "openhands_platform_calibration_1s.csv"
PLOT = OUT_DIR / "openhands_platform_calibrated_prediction_overlay.pdf"
PNG = PLOT.with_suffix(".png")

SOURCE_IDLE_W_PER_GPU = 70.1188
TARGET_IDLE_W_PER_GPU = 83.2
ORDINARY_DYNAMIC_GAIN = 1.0425
EXTRA_PREFILL_GAIN = 0.5891


def platform_calibrate(
    source_node_w: np.ndarray, prefill_compute_node_w: np.ndarray, *, tp: int,
) -> np.ndarray:
    """Apply the declared retrospective two-gain calibration in node watts."""
    if tp <= 0:
        raise ValueError("TP must be positive")
    source = np.asarray(source_node_w, dtype=float)
    prefill = np.asarray(prefill_compute_node_w, dtype=float)
    if source.shape != prefill.shape:
        raise ValueError("source and prefill contributions must align")
    return (
        TARGET_IDLE_W_PER_GPU * tp
        + ORDINARY_DYNAMIC_GAIN * (source - SOURCE_IDLE_W_PER_GPU * tp)
        + EXTRA_PREFILL_GAIN * prefill
    )


def bundle_dirs() -> list[Path]:
    rows = []
    for path in BUNDLE_ROOT.glob("*/manifest.json"):
        probe = json.loads(path.read_text())["probe"]
        rows.append((int(probe["pack_index"]), bool(probe["prefix_cache"]), path.parent))
    expected = [(pack, cache) for pack in range(3) for cache in (False, True)]
    if [(pack, cache) for pack, cache, _ in sorted(rows)] != expected:
        raise ValueError("expected one cache-off/on bundle for each OpenHands pack")
    return [path for _, _, path in sorted(rows)]


def evaluate(bundle: Path, timing_fit: dict, power_fit: dict) -> dict:
    manifest, record, ledger, measured, lo, hi, gaps, timing = transfer._request_ledger(
        bundle, timing_fit
    )
    fit = power_fit["dense"][record.hardware]
    design = transfer_core.dense_response(record, ledger, fit)
    coefficients = np.asarray(fit["coefficients"], dtype=float)
    source = (design @ coefficients * record.tp)[lo:hi]
    prefill_util = transfer_core.phase_compute(
        ledger, record.hardware, record.tp
    )
    filtered_prefill = apply_response(
        prefill_util[:, None], dt_s=transfer.DT_S,
        hardware=record.hardware, delay_s=float(fit["delay_s"]),
    )[:, 0]
    compute_index = fit["feature_names"].index("compute_util")
    prefill_node = filtered_prefill * coefficients[compute_index] * record.tp
    calibrated = platform_calibrate(source, prefill_node[lo:hi], tp=record.tp)
    probe = manifest["probe"]
    return {
        "run_id": manifest["run_id"],
        "pack": int(probe["pack_index"]),
        "prefix_cache": bool(probe["prefix_cache"]),
        "tp": record.tp,
        "measured": measured,
        "predicted": calibrated,
        "source_metrics": transfer._metrics(measured, source),
        "calibrated_metrics": transfer._metrics(measured, calibrated),
        "interpolated_bins": gaps,
        "timing": timing,
    }


def one_second(row: dict) -> tuple[np.ndarray, np.ndarray]:
    return transfer.per_gpu_one_second_pair(
        row["measured"], row["predicted"], tp=row["tp"], dt_s=transfer.DT_S
    )


def save_plot(rows: list[dict], views: list[tuple[np.ndarray, np.ndarray]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 7.5), sharey=True)
    for axis, row, (measured, predicted) in zip(axes.flat, rows, views):
        seconds = np.arange(measured.size)
        axis.plot(seconds, measured, color="black", linewidth=0.7, label="Measured")
        axis.plot(seconds, predicted, color="#8C1515", linewidth=0.9,
                  label="Calibrated prediction")
        state = "cache-on" if row["prefix_cache"] else "cache-off"
        metric = row["calibrated_metrics"]
        axis.set_title(
            f"pack {row['pack']} {state} | energy {metric['energy_error_pct']:.1f}% "
            f"| NRMSE {metric['nrmse_range']:.3f}", fontsize=9,
        )
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Power (W/GPU)")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    save_figure(fig, PLOT, bbox_inches="tight")
    save_figure(fig, PNG, bbox_inches="tight", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    release = json.loads(transfer.ARTIFACT.read_text())
    rows = [
        evaluate(bundle, release["timing"], release["power"])
        for bundle in bundle_dirs()
    ]
    views = [one_second(row) for row in rows]
    save_plot(rows, views)
    samples = []
    for row, (measured, predicted) in zip(rows, views):
        samples.extend({
            "run_id": row["run_id"], "pack": row["pack"],
            "prefix_cache": row["prefix_cache"], "time_s": index,
            "measured_w_per_gpu": float(observed),
            "predicted_w_per_gpu": float(estimate),
        } for index, (observed, estimate) in enumerate(zip(measured, predicted)))
    with SAMPLES.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    report_rows = [{key: value for key, value in row.items()
                    if key not in ("measured", "predicted")} for row in rows]
    REPORT.write_text(json.dumps({
        "schema_version": "openhands-platform-calibration-v1",
        "evidence_role": "retrospective few-shot platform calibration",
        "artifact": {"path": str(transfer.ARTIFACT.relative_to(ROOT)),
                     "sha256": transfer.sha256_file(transfer.ARTIFACT)},
        "calibration": {
            "source_idle_w_per_gpu": SOURCE_IDLE_W_PER_GPU,
            "target_idle_w_per_gpu": TARGET_IDLE_W_PER_GPU,
            "ordinary_dynamic_gain": ORDINARY_DYNAMIC_GAIN,
            "extra_prefill_compute_gain": EXTRA_PREFILL_GAIN,
        },
        "runs": report_rows,
        "samples": str(SAMPLES.relative_to(ROOT)),
        "outputs": [str(PLOT.relative_to(ROOT)), str(PNG.relative_to(ROOT))],
    }, indent=2) + "\n")
    print(PLOT)
    print(PNG)
    print(SAMPLES)
    print(REPORT)


if __name__ == "__main__":
    main()
