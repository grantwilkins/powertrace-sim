"""Create standalone paper figures for frozen timing-model evaluation."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TIMING_DIR = REPO / "timing-test"
sys.path.insert(0, str(TIMING_DIR))

from evaluate_timing import EVAL_ROLES, evaluate_run  # noqa: E402
from fit_efficiencies import chunked_prefill_time_s  # noqa: E402
from iteration_time import launch_overhead_s, transformer_bw_scale  # noqa: E402
from model.paper_replay import sha256_file  # noqa: E402
from model.training.timing import probe_points  # noqa: E402
from model.training_data.arch import get_arch  # noqa: E402
from scripts.paper.render import save_figure  # noqa: E402

ARTIFACT_PATH = REPO / "results" / "clean_model" / "powertrace_v1.json"
DATA_PATH = TIMING_DIR / "timing_dataset.npz"
MANIFEST_PATH = TIMING_DIR / "split_manifest_fp8.json"
CALIBRATION_PATH = TIMING_DIR / "probe_calibration.json"
THROUGHPUT_PATH = REPO / "model" / "throughput_database.json"
OUT_DIR = REPO / "results" / "paper"
PREFILL_PATH = OUT_DIR / "timing_prefill_parity.pdf"
DECODE_PATH = OUT_DIR / "timing_decode_parity.pdf"
REPORT_PATH = OUT_DIR / "timing_parity_manifest.json"
POINTS_PER_MODEL_HARDWARE = 120

MODEL_LABELS = {
    "llama-3-8b": "Llama 3 8B",
    "llama-3-70b": "Llama 3 70B",
    "llama-3-405b": "Llama 3 405B",
    "deepseek-r1-distill-8b": "DeepSeek-R1 8B",
    "deepseek-r1-distill-70b": "DeepSeek-R1 70B",
    "gpt-oss-20b": "GPT-OSS 20B",
    "gpt-oss-120b": "GPT-OSS 120B",
}
MODEL_ORDER = tuple(MODEL_LABELS)


def prefill_probe_points(calibration: dict, fit: dict) -> list[dict]:
    """Predict the directly measured queue-free prefill probe levels."""
    points = []
    for point in probe_points(calibration):
        if point["kind"] != "prefill":
            continue
        hardware, model, tp = point["hardware"], point["model"], point["tp"]
        params = fit[hardware]
        arch = dict(get_arch(model))
        launch = launch_overhead_s(
            arch, base_s=params["base_overhead_s"],
            per_message_s=params["per_message_s"][str(tp)],
        )
        predicted = params["first_token_overhead_s"] + chunked_prefill_time_s(
            arch, point["n_in"], hardware=hardware, tp=tp,
            eff_flops=params["eff_flops"], eff_bw=params["eff_bw"],
            t_launch_s=launch,
            transformer_bw_scale=transformer_bw_scale(arch, params, hardware),
        )
        points.append({
            "phase": "prefill", "model": model,
            "model_label": MODEL_LABELS[model], "hardware": hardware,
            "observed_ms": 1e3 * float(point["seconds"]),
            "predicted_ms": 1e3 * predicted,
        })
    return points


def solo_request_indices(data: dict, rid: int) -> np.ndarray:
    """Return stable-TTFT requests with no measured lifetime overlap."""
    idx = np.flatnonzero(data["req_run_id"] == rid)
    if idx.size == 0:
        return idx
    arrival = data["arrival_time_s"][idx]
    finish = arrival + data["ttft_s"][idx] + data["decode_duration_s"][idx]
    order = np.argsort(arrival, kind="stable")
    arrival, finish, idx = arrival[order], finish[order], idx[order]
    previous_finish = np.r_[-np.inf, np.maximum.accumulate(finish)[:-1]]
    next_arrival = np.r_[arrival[1:], np.inf]
    stable = data["output_tokens"][idx] >= 8
    return idx[(arrival >= previous_finish) & (finish <= next_arrival) & stable]


def queue_free_prefill_points(data: dict, roles: dict[int, str],
                              fit: dict) -> list[dict]:
    """Predict individual non-overlapping training-request prefill latency."""
    points = []
    for rid, role in sorted(roles.items()):
        if role != "train":
            continue
        hardware = str(data["run_hardware"][rid])
        model = str(data["run_model"][rid])
        tp = int(data["run_tp"][rid])
        params = fit[hardware]
        arch = dict(get_arch(model))
        launch = launch_overhead_s(
            arch, base_s=params["base_overhead_s"],
            per_message_s=params["per_message_s"][str(tp)],
        )
        bandwidth_scale = transformer_bw_scale(arch, params, hardware)
        for index in solo_request_indices(data, rid):
            predicted = params["first_token_overhead_s"] + chunked_prefill_time_s(
                arch, int(data["input_tokens"][index]), hardware=hardware,
                tp=tp, eff_flops=params["eff_flops"], eff_bw=params["eff_bw"],
                t_launch_s=launch, transformer_bw_scale=bandwidth_scale,
            )
            points.append({
                "phase": "prefill",
                "model": model,
                "model_label": MODEL_LABELS[model],
                "hardware": hardware,
                "observed_ms": 1e3 * float(data["ttft_s"][index]),
                "predicted_ms": 1e3 * predicted,
            })
    return sample_points(points, POINTS_PER_MODEL_HARDWARE)


def request_points(rows: list[dict], *, model: str, hardware: str) -> list[dict]:
    """Convert per-request evaluation rows into phase parity points."""
    points = []
    for row in rows:
        for phase, field in (("prefill", "ttft_s"),
                             ("decode", "decode_duration_s")):
            observed = 1e3 * float(row[f"meas_{field}"])
            predicted = 1e3 * float(row[f"pred_{field}"])
            if observed <= 0 or predicted <= 0:
                continue
            points.append({
                "phase": phase,
                "model": model,
                "model_label": MODEL_LABELS[model],
                "hardware": hardware,
                "observed_ms": observed,
                "predicted_ms": predicted,
            })
    return points


def representative_run_ids(data: dict, roles: dict[int, str]) -> list[int]:
    """Select one repetition from every frozen evaluation cell."""
    cells = {}
    for rid, role in sorted(roles.items()):
        if role not in EVAL_ROLES:
            continue
        key = (role, str(data["run_hardware"][rid]),
               str(data["run_model"][rid]), int(data["run_tp"][rid]),
               float(data["run_rate"][rid]))
        cells.setdefault(key, rid)
    return sorted(cells.values())


def sample_points(points: list[dict], limit: int) -> list[dict]:
    """Deterministically cap each model/hardware/phase stratum."""
    groups = defaultdict(list)
    for point in points:
        groups[(point["phase"], point["model"], point["hardware"])].append(point)
    sampled = []
    for group in groups.values():
        indices = np.linspace(0, len(group) - 1, min(limit, len(group)), dtype=int)
        sampled.extend(group[index] for index in indices)
    return sampled


def evaluation_points(data: dict, roles: dict[int, str], fit: dict,
                      throughput: dict) -> list[dict]:
    points = []
    for rid in representative_run_ids(data, roles):
        hardware = str(data["run_hardware"][rid])
        model = str(data["run_model"][rid])
        rows = evaluate_run(data, rid, fit, throughput)
        points.extend(request_points(rows, model=model, hardware=hardware))
    return sample_points(points, POINTS_PER_MODEL_HARDWARE)


def parity_limits(points: list[dict]) -> tuple[float, float]:
    values = np.asarray([
        value
        for point in points
        for value in (point["observed_ms"], point["predicted_ms"])
    ])
    if values.size == 0 or np.any(values <= 0):
        raise ValueError("Parity plots require positive timing observations")
    return float(values.min() / 1.15), float(values.max() * 1.15)


def draw_phase(axis, points: list[dict], phase: str, *, legend: bool) -> None:
    import seaborn as sns

    selected = [point for point in points if point["phase"] == phase]
    if not selected:
        raise ValueError(f"No {phase} timing points")
    all_labels = [MODEL_LABELS[model] for model in MODEL_ORDER]
    labels = [label for label in all_labels
              if any(point["model_label"] == label for point in selected)]
    colors = sns.color_palette("colorblind", len(all_labels))
    palette = dict(zip(all_labels, colors))
    plot_data = {
        "observed_ms": [point["observed_ms"] for point in selected],
        "predicted_ms": [point["predicted_ms"] for point in selected],
        "Model": [point["model_label"] for point in selected],
        "Hardware": [point["hardware"] for point in selected],
    }
    sns.scatterplot(
        data=plot_data,
        x="observed_ms",
        y="predicted_ms",
        hue="Model",
        hue_order=labels,
        palette=palette,
        style="Hardware",
        style_order=("A100", "H100"),
        markers={"A100": "o", "H100": "X"},
        s=22,
        alpha=0.58,
        edgecolor="white",
        linewidth=0.3,
        legend=legend,
        ax=axis,
    )
    limits = parity_limits(selected)
    axis.plot(limits, limits, linestyle="--", color="0.25", linewidth=1.1,
              zorder=0)
    axis.set(xscale="log", yscale="log", xlim=limits, ylim=limits)
    phase_label = "Prefill latency" if phase == "prefill" else "Decode duration"
    axis.set_xlabel(f"Observed {phase_label.lower()} (ms)")
    axis.set_ylabel(f"Predicted {phase_label.lower()} (ms)")
    axis.set_title("")
    axis.grid(True, which="major", color="0.88", linewidth=0.65)
    axis.grid(True, which="minor", color="0.94", linewidth=0.4)
    axis.set_axisbelow(True)
    if legend:
        axis.legend(title=None, loc="upper left", frameon=False,
                    fontsize=7.1, handletextpad=0.2, borderaxespad=0.3,
                    labelspacing=0.25, ncol=2, columnspacing=0.65)


def save_phase(points: list[dict], phase: str, path: Path, *, legend: bool) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.18)
    fig, axis = plt.subplots(figsize=(3.25, 3.0))
    draw_phase(axis, points, phase, legend=legend)
    fig.tight_layout(pad=0.35)
    save_figure(fig, path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(DATA_PATH, allow_pickle=False))
    roles = {
        int(rid): role
        for rid, role in json.loads(MANIFEST_PATH.read_text())["roles"].items()
    }
    artifact = json.loads(ARTIFACT_PATH.read_text())
    fit = artifact["timing"]
    calibration = json.loads(CALIBRATION_PATH.read_text())
    throughput_raw = json.loads(THROUGHPUT_PATH.read_text())
    throughput = throughput_raw.get("configs", throughput_raw)
    evaluated = evaluation_points(data, roles, fit, throughput)
    points = prefill_probe_points(calibration, fit) + queue_free_prefill_points(
        data, roles, fit
    ) + [
        point for point in evaluated if point["phase"] == "decode"
    ]
    save_phase(points, "prefill", PREFILL_PATH, legend=False)
    save_phase(points, "decode", DECODE_PATH, legend=True)
    counts = {phase: sum(point["phase"] == phase for point in points)
              for phase in ("prefill", "decode")}
    REPORT_PATH.write_text(json.dumps({
        "schema_version": "timing-parity-v1",
        "artifact": {"path": str(ARTIFACT_PATH.relative_to(REPO)),
                     "sha256": sha256_file(ARTIFACT_PATH)},
        "timing_dataset_sha256": sha256_file(DATA_PATH),
        "split_sha256": sha256_file(MANIFEST_PATH),
        "probe_calibration_sha256": sha256_file(CALIBRATION_PATH),
        "point_counts": counts,
        "outputs": [str(PREFILL_PATH), str(DECODE_PATH)],
    }, indent=2) + "\n")
    print(f"Plotted {counts['prefill']} prefill calibration points and "
          f"{counts['decode']} sampled decode requests")
    print(PREFILL_PATH)
    print(DECODE_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
