"""Regenerate selected-release replay tables and representative traces."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from model.paper_replay import (
    evaluate_replay,
    interpolate_power,
    one_second_per_gpu,
    run_slices,
    sha256_file,
)
from scripts.paper.render import save_figure

RATES = (0.125, 1.0, 2.0, 4.0)
METRICS = (
    ("nrmse_range", "Power error (\\%)", 100.0, False, 2),
    ("ks_agreement", "Distribution", 1.0, True, 3),
    ("energy_error_pct", "Energy error (\\%)", 1.0, False, 2),
    ("acf_r2", "ACF $R^2$", 1.0, True, 3),
)
MODEL_LABELS = {
    "deepseek-r1-distill-8b": "DeepSeek-R1-Distill 8B",
    "deepseek-r1-distill-70b": "DeepSeek-R1-Distill 70B",
    "gpt-oss-20b": "GPT-OSS 20B",
    "gpt-oss-120b": "GPT-OSS 120B",
    "llama-3-8b": "Llama 3 8B",
    "llama-3-70b": "Llama 3 70B",
    "llama-3-405b": "Llama 3 405B",
}


def _point_values(rows: list[dict], metric: str) -> list[float]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["hardware"], row["tp"], row["rate"])].append(row[metric])
    return [float(np.median(values)) for _, values in sorted(groups.items())]


def _interval(values: list[float], rng: np.random.Generator, draws=1000) -> dict:
    array = np.asarray(values, dtype=float)
    indices = rng.integers(0, array.size, size=(draws, array.size))
    samples = np.median(array[indices], axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {"median": float(np.median(array)), "low": float(low), "high": float(high)}


def summarize_models(rows: list[dict], seed=20260721) -> list[dict]:
    heldout = [
        row for row in rows
        if row["role"].startswith("heldout") and row["surface_supported"]
    ]
    groups = defaultdict(list)
    for row in heldout:
        groups[row["model"]].append(row)
    rng = np.random.default_rng(seed)
    summary = []
    for model, selected in sorted(groups.items()):
        record = {
            "model": model,
            "label": MODEL_LABELS.get(model, model),
            "traces": len(selected),
            "points": len({(r["hardware"], r["tp"], r["rate"]) for r in selected}),
        }
        for metric, _, scale, _, _ in METRICS:
            record[metric] = _interval(
                [scale * value for value in _point_values(selected, metric)], rng
            )
        summary.append(record)
    return summary


def write_table(summary: list[dict], out_dir: Path, source_csv: Path) -> list[Path]:
    csv_path = out_dir / "selected_model_fidelity_table.csv"
    json_path = out_dir / "selected_model_fidelity_table.json"
    tex_path = out_dir / "selected_model_fidelity_table.tex"
    fields = ["model", "label", "traces", "points"] + [
        f"{metric}_{suffix}"
        for metric, *_ in METRICS for suffix in ("median", "low", "high")
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            flat = {key: row[key] for key in fields[:4]}
            for metric, *_ in METRICS:
                for suffix in ("median", "low", "high"):
                    flat[f"{metric}_{suffix}"] = row[metric][suffix]
            writer.writerow(flat)
    header = " & ".join(["Model", "Traces", "Points", *[label for _, label, *_ in METRICS]])
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Fidelity of the frozen release artifact on held-out measured traces. Medians and 95\% bootstrap intervals are computed over model-local hardware/TP/rate points after collapsing repetitions.}",
        r"\label{tab:selected-model-fidelity}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule", header + " \\\\", r"\midrule",
    ]
    for row in summary:
        values = [row["label"], str(row["traces"]), str(row["points"])]
        for metric, _, _, _, digits in METRICS:
            value = row[metric]
            values.append(
                f"{value['median']:.{digits}f} "
                f"[{value['low']:.{digits}f}, {value['high']:.{digits}f}]"
            )
        lines.append(" & ".join(values) + " \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    tex_path.write_text("\n".join(lines) + "\n")
    json_path.write_text(json.dumps({
        "schema_version": "selected-model-fidelity-table-v1",
        "source": {"path": str(source_csv), "sha256": sha256_file(source_csv)},
        "population": "supported held-out runs scored by the frozen release artifact",
        "bootstrap_seed": 20260721,
        "models": summary,
    }, indent=2) + "\n")
    return [tex_path, csv_path, json_path]


def _rank(values: list[float], maximize=False) -> np.ndarray:
    order = np.argsort(-np.asarray(values) if maximize else np.asarray(values), kind="stable")
    ranks = np.empty(order.size, dtype=int)
    ranks[order] = np.arange(1, order.size + 1)
    return ranks


def select_configuration(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        if row["role"] == "heldout_model" and row["surface_supported"]:
            groups[(row["hardware"], row["model"], row["tp"])].append(row)
    candidates = []
    for key, selected in sorted(groups.items()):
        if not set(RATES).issubset({row["rate"] for row in selected}):
            continue
        candidates.append({
            "hardware": key[0], "model": key[1], "tp": key[2],
            "energy_error_pct": float(np.median([r["energy_error_pct"] for r in selected])),
            "rmse_w_per_gpu": float(np.median([r["rmse_w_per_gpu"] for r in selected])),
            "acf_r2": float(np.median([r["acf_r2"] for r in selected])),
        })
    if not candidates:
        raise ValueError("no supported held-out model covers all showcase rates")
    for metric, maximize in (("energy_error_pct", False), ("rmse_w_per_gpu", False), ("acf_r2", True)):
        for row, rank in zip(candidates, _rank([r[metric] for r in candidates], maximize)):
            row.setdefault("ranks", {})[metric] = int(rank)
    for row in candidates:
        row["rank_sum"] = sum(row["ranks"].values())
    return min(candidates, key=lambda row: (row["rank_sum"], row["model"], row["tp"]))


def _medoid(rows: list[dict]) -> dict:
    values = np.asarray([
        [row["energy_error_pct"], row["rmse_w_per_gpu"], -row["acf_r2"]]
        for row in rows
    ])
    span = np.ptp(values, axis=0)
    span[span == 0.0] = 1.0
    distance = np.sum(np.abs(values - np.median(values, axis=0)) / span, axis=1)
    return rows[min(range(len(rows)), key=lambda index: (distance[index], rows[index]["run_id"]))]


def _series(cache: dict, prediction: np.ndarray, run_id: int):
    selected = cache["run_id"] == run_id
    tp = int(np.unique(cache["tp"][selected])[0])
    measured, _ = interpolate_power(cache["power"][selected])
    return one_second_per_gpu(
        measured, prediction[selected], tp=tp, dt_s=float(cache["dt_s"])
    )


def write_traces(
    rows: list[dict], cache: dict, prediction: np.ndarray, out_dir: Path,
    source_csv: Path, artifact_path: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    selection = select_configuration(rows)
    panels = []
    for rate in RATES:
        candidates = [row for row in rows if (
            row["hardware"] == selection["hardware"]
            and row["model"] == selection["model"]
            and row["tp"] == selection["tp"] and row["rate"] == rate
        )]
        panels.append(_medoid(candidates))
    views = [_series(cache, prediction, row["run_id"]) for row in panels]
    ymax = 1.05 * max(float(np.max(series)) for view in views for series in view)
    outputs = []
    samples = []
    for row, (measured, predicted) in zip(panels, views):
        seconds = min(600, measured.size, predicted.size)
        rate_slug = f"{row['rate']:g}".replace(".", "p")
        path = out_dir / (
            f"power_trace_{row['model'].replace('-', '_')}_{row['hardware'].lower()}_"
            f"tp{row['tp']}_rate_{rate_slug}_release_1s.pdf"
        )
        fig, axis = plt.subplots(figsize=(7.2, 2.8))
        time = np.arange(seconds) / 60.0
        axis.plot(time, measured[:seconds], color="black", linewidth=0.7, label="Measured")
        axis.plot(time, predicted[:seconds], color="#8C1515", linewidth=0.9, label="PowerTrace-Sim")
        axis.set(xlabel="Time (min)", ylabel="Power (W/GPU)", xlim=(0, seconds / 60), ylim=(0, ymax))
        axis.legend(frameon=False, ncol=2, loc="upper center")
        fig.tight_layout()
        save_figure(fig, path, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
        samples.extend({
            "rate_requests_s": row["rate"], "run_id": row["run_id"],
            "time_s": second, "measured_w_per_gpu": float(measured[second]),
            "predicted_w_per_gpu": float(predicted[second]),
        } for second in range(seconds))
    csv_path = out_dir / "selected_representative_traces_1s.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    report_path = out_dir / "selected_representative_traces.json"
    report_path.write_text(json.dumps({
        "schema_version": "selected-representative-traces-v1",
        "artifact": {"path": str(artifact_path), "sha256": sha256_file(artifact_path)},
        "scores": {"path": str(source_csv), "sha256": sha256_file(source_csv)},
        "selection": selection,
        "panels": panels,
        "aggregation": "first 600 matched one-second per-GPU means",
        "samples": str(csv_path),
        "outputs": [str(path) for path in outputs],
    }, indent=2) + "\n")
    return [*outputs, csv_path, report_path]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="power-test/sim_ledger_power_uniform_current_250ms.npz")
    parser.add_argument("--artifact", default="results/clean_model/powertrace_v1.json")
    parser.add_argument("--split", default="timing-test/coverage_split_manifest_fp8.json")
    parser.add_argument("--out-dir", default="results/paper")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_csv = out_dir / "selected_replay_per_run.csv"
    score_json = out_dir / "selected_replay_manifest.json"
    rows, prediction, cache = evaluate_replay(
        cache_path=args.cache, artifact_path=args.artifact, split_path=args.split,
        out_csv=score_csv, out_json=score_json,
    )
    outputs = [score_csv, score_json]
    outputs.extend(write_table(summarize_models(rows), out_dir, score_csv))
    outputs.extend(write_traces(
        rows, cache, prediction, out_dir, score_csv, Path(args.artifact)
    ))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
