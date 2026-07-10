"""Build a power-vs-occupancy roofline from profiling run bundles.

The analyzer consumes the bundle schema emitted by profiling probes and agentic
runs. It reconstructs 5 s prefill/decode token rates from request timestamps,
normalizes them by dedicated prefill/decode capacity probes, and fits the
ramp-plateau roofline:

    P = P_idle + (P_busy - P_idle) * min(ell / k, 1)
    ell = f / F + g / G

Usage:
  uv run -m scripts.eval.occupancy_roofline --run-root data/runs/<campaign_id>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model.training_data.power_parsing import parse_power_csv_per_gpu, tp_sum_power

OUT_ROOT = Path("results") / "occupancy_roofline"
FIG_ROOT = Path("figures") / "occupancy_roofline"
LATENCY_KNEE_FACTOR = 1.25


def _load_requests(path: Path) -> dict:
    data = json.loads(path.read_text())
    n = min(
        len(data.get("input_lens", [])),
        len(data.get("output_lens", [])),
        len(data.get("ttfts", [])),
        len(data.get("itls", [])),
        len(data.get("request_timestamps", [])),
    )
    return {
        "input_lens": np.asarray(data.get("input_lens", [])[:n], dtype=float),
        "output_lens": np.asarray(data.get("output_lens", [])[:n], dtype=float),
        "ttfts": np.asarray(data.get("ttfts", [])[:n], dtype=float),
        "itls": list(data.get("itls", [])[:n]),
        "request_timestamps": np.asarray(data.get("request_timestamps", [])[:n], dtype=float),
    }


def _window_edges(start: float, end: float, window_s: float) -> np.ndarray:
    lo = math.ceil(start / window_s) * window_s
    hi = math.floor(end / window_s) * window_s
    if hi <= lo:
        return np.asarray([], dtype=float)
    return np.arange(lo, hi + 0.5 * window_s, window_s, dtype=float)


def _mean_by_window(edges: np.ndarray, t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_win = len(edges) - 1
    idx = np.searchsorted(edges, t) - 1
    ok = (idx >= 0) & (idx < n_win) & np.isfinite(y)
    sums = np.bincount(idx[ok], weights=y[ok], minlength=n_win)
    counts = np.bincount(idx[ok], minlength=n_win)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / np.maximum(counts, 1), counts > 0


def _add_prefill_tokens(f_tok: np.ndarray, edges: np.ndarray, arrival: float,
                        ttft: float, input_len: float) -> None:
    if input_len <= 0 or ttft <= 0:
        return
    end = arrival + ttft
    lo = max(int((arrival - edges[0]) // (edges[1] - edges[0])), 0)
    hi = min(int((end - edges[0]) // (edges[1] - edges[0])), len(f_tok) - 1)
    if hi < lo:
        return
    for w in range(lo, hi + 1):
        overlap = min(end, edges[w + 1]) - max(arrival, edges[w])
        if overlap > 0:
            f_tok[w] += input_len * overlap / ttft


def _decode_events(arrival: float, ttft: float, itl_value: object,
                   output_len: float) -> tuple[np.ndarray, np.ndarray]:
    n_out = int(max(round(float(output_len)), 0))
    if n_out == 0 or ttft <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    start = arrival + ttft
    if isinstance(itl_value, list):
        deltas = np.asarray(itl_value, dtype=float)
    elif isinstance(itl_value, (int, float)) and np.isfinite(float(itl_value)):
        deltas = np.full(max(n_out - 1, 0), float(itl_value), dtype=float)
    else:
        deltas = np.asarray([], dtype=float)
    times = start + np.concatenate(([0.0], np.cumsum(deltas)))
    itls = np.concatenate(([np.nan], deltas))
    if times.size > n_out:
        times = times[:n_out]
        itls = itls[:n_out]
    return times, itls


def _window_label(manifest: dict, center_epoch_rel: float, power_t0: float) -> str:
    probe = manifest.get("probe", {})
    source = str(probe.get("type", "unknown"))
    levels = probe.get("levels", [])
    for level in levels:
        start = float(level.get("t_start_epoch", 0.0)) - power_t0
        end = float(level.get("t_end_epoch", 0.0)) - power_t0
        if start <= center_epoch_rel <= end:
            return str(level.get("label", source))
    return "long_agentic" if source == "agentic" else source


def window_bundle(run_dir: Path, window_s: float) -> list[dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    clock = manifest.get("clock")
    if not isinstance(clock, dict) or "local_utc_offset_s" not in clock:
        raise ValueError("Bundle manifest requires clock.local_utc_offset_s")
    power = parse_power_csv_per_gpu(
        str(run_dir / "power.csv"),
        gpus_per_node=int(manifest["gpus_per_node"]),
        strict_topology=True,
        local_utc_offset_s=float(clock["local_utc_offset_s"]),
    )
    if power is None:
        raise ValueError(f"could not parse canonical power.csv in {run_dir}")

    req = _load_requests(run_dir / "requests.json")
    p_t = np.asarray(power["timestamps"], dtype=float)
    p_w = tp_sum_power(power["power_per_gpu"], int(manifest["tp"]))
    p_t0 = float(p_t[0])
    p_rel = p_t - p_t0

    probe = manifest.get("probe", {})
    window = probe.get("window", {})
    if req["request_timestamps"].size > 0 and "start_epoch" in window and "end_epoch" in window:
        run_start = float(window["start_epoch"]) - p_t0
        run_end = float(window["end_epoch"]) - p_t0
    else:
        run_start = float(np.nanmin(p_rel))
        run_end = float(np.nanmax(p_rel))
    run_start = max(run_start, float(np.nanmin(p_rel)))
    run_end = min(run_end, float(np.nanmax(p_rel)))

    edges = _window_edges(run_start, run_end, window_s)
    if edges.size < 2:
        return []
    n_win = len(edges) - 1
    mean_power, has_power = _mean_by_window(edges, p_rel, p_w)

    f_tok = np.zeros(n_win, dtype=float)
    g_tok = np.zeros(n_win, dtype=float)
    itl_sum = np.zeros(n_win, dtype=float)
    itl_count = np.zeros(n_win, dtype=float)
    ttft_sum = np.zeros(n_win, dtype=float)
    ttft_count = np.zeros(n_win, dtype=float)

    for i in range(req["input_lens"].size):
        ts = req["request_timestamps"][i]
        if not np.isfinite(ts):
            continue
        arrival = float(ts) - p_t0
        ttft = float(req["ttfts"][i])
        input_len = float(req["input_lens"][i])
        output_len = float(req["output_lens"][i])
        _add_prefill_tokens(f_tok, edges, arrival, ttft, input_len)

        arrival_bin = np.searchsorted(edges, arrival) - 1
        if 0 <= arrival_bin < n_win and input_len > 0 and ttft > 0:
            ttft_sum[arrival_bin] += ttft / input_len * 1000.0
            ttft_count[arrival_bin] += 1.0

        dec_t, dec_itl = _decode_events(arrival, ttft, req["itls"][i], output_len)
        idx = np.searchsorted(edges, dec_t) - 1
        ok = (idx >= 0) & (idx < n_win)
        if np.any(ok):
            g_tok += np.bincount(idx[ok], minlength=n_win)
        itl_ok = ok & np.isfinite(dec_itl)
        if np.any(itl_ok):
            itl_sum += np.bincount(idx[itl_ok], weights=dec_itl[itl_ok] * 1000.0, minlength=n_win)
            itl_count += np.bincount(idx[itl_ok], minlength=n_win)

    rows = []
    source_type = str(probe.get("type", "unknown"))
    for w in range(n_win):
        if not has_power[w]:
            continue
        center = 0.5 * (edges[w] + edges[w + 1])
        rows.append({
            "run_id": str(manifest.get("run_id", run_dir.name)),
            "model": str(manifest.get("model", "")),
            "hardware": str(manifest.get("hardware", "")),
            "tp": int(manifest["tp"]),
            "source_type": source_type,
            "source_label": _window_label(manifest, center, p_t0),
            "window_start_s": float(edges[w]),
            "window_end_s": float(edges[w + 1]),
            "power_w": float(mean_power[w]),
            "f_tps": float(f_tok[w] / window_s),
            "g_tps": float(g_tok[w] / window_s),
            "itl_ms": float(itl_sum[w] / itl_count[w]) if itl_count[w] > 0 else np.nan,
            "ttft_per_tok_ms": (
                float(ttft_sum[w] / ttft_count[w]) if ttft_count[w] > 0 else np.nan
            ),
        })
    return rows


def collect_windows(run_root: Path, window_s: float, *, model: str | None = None,
                    hardware: str | None = None, tp: int | None = None) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        if not (run_dir / "manifest.json").exists():
            continue
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if model and manifest.get("model") != model:
            continue
        if hardware and manifest.get("hardware") != hardware:
            continue
        if tp is not None and int(manifest.get("tp", -1)) != int(tp):
            continue
        rows.extend(window_bundle(run_dir, window_s))
    return rows


def _latency_knee(ell: np.ndarray, itl: np.ndarray, n_bins: int = 20) -> float:
    ok = np.isfinite(ell) & np.isfinite(itl) & (ell > 0)
    if int(ok.sum()) < 20:
        return np.nan
    x, y = ell[ok], itl[ok]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    centers, medians = [], []
    for i in range(len(edges) - 1):
        sel = (x >= edges[i]) & (x <= edges[i + 1])
        if int(sel.sum()) >= 3:
            centers.append(float(np.median(x[sel])))
            medians.append(float(np.median(y[sel])))
    if len(centers) < 4:
        return np.nan
    centers = np.asarray(centers)
    medians = np.asarray(medians)
    best_idx = int(np.argmin(medians))
    bad = np.where(medians[best_idx:] > LATENCY_KNEE_FACTOR * medians[best_idx])[0]
    return float(centers[best_idx + bad[0]]) if bad.size else np.nan


def _fit_power_knee(ell: np.ndarray, power: np.ndarray, p_idle: float,
                    p_busy: float) -> tuple[float, float, np.ndarray]:
    positive = ell[np.isfinite(ell) & (ell > 0)]
    if positive.size == 0:
        raise ValueError("cannot fit roofline without positive occupancy windows")
    candidates = np.linspace(float(np.quantile(positive, 0.01)), float(np.max(positive)), 300)
    best_k, best_sse, best_pred = np.nan, np.inf, np.full_like(power, np.nan)
    for k in candidates:
        pred = p_idle + (p_busy - p_idle) * np.clip(ell / max(k, 1e-12), 0.0, 1.0)
        sse = float(np.sum((power - pred) ** 2))
        if sse < best_sse:
            best_k, best_sse, best_pred = float(k), sse, pred
    denom = float(np.sum((power - np.mean(power)) ** 2))
    r2 = 1.0 - best_sse / denom if denom > 0 else np.nan
    return best_k, r2, best_pred


def analyze_rows(rows: list[dict], label: str) -> tuple[dict, list[dict], np.ndarray]:
    if not rows:
        raise ValueError("no windows found")
    f = np.asarray([r["f_tps"] for r in rows], dtype=float)
    g = np.asarray([r["g_tps"] for r in rows], dtype=float)
    power = np.asarray([r["power_w"] for r in rows], dtype=float)
    itl = np.asarray([r["itl_ms"] for r in rows], dtype=float)
    source = np.asarray([r["source_type"] for r in rows], dtype=object)

    f_cap = f[(source == "prefill_staircase") & (f > 0)]
    g_cap = g[(source == "decode_staircase") & (g > 0)]
    if f_cap.size == 0 or g_cap.size == 0:
        raise ValueError("roofline requires prefill_staircase and decode_staircase capacity windows")
    F = float(np.quantile(f_cap, 0.995))
    G = float(np.quantile(g_cap, 0.995))
    ell = f / F + g / G

    idle_sel = (source == "idle_hold") | ((f + g) == 0)
    p_idle = float(np.median(power[idle_sel])) if np.any(idle_sel) else float(np.min(power))
    top = ell >= np.quantile(ell[np.isfinite(ell)], 0.9)
    p_busy = float(np.median(power[top]))
    k, r2, pred = _fit_power_knee(ell, power, p_idle, p_busy)
    lat_knee = _latency_knee(ell, itl)

    for i, row in enumerate(rows):
        row["F_prefill_tps"] = F
        row["G_decode_tps"] = G
        row["ell"] = float(ell[i])
        row["roofline_pred_w"] = float(pred[i])

    summary = {
        "label": label,
        "n_bundles": len({r["run_id"] for r in rows}),
        "n_windows": len(rows),
        "F_prefill_tps": F,
        "G_decode_tps": G,
        "P_idle_w": p_idle,
        "P_busy_w": p_busy,
        "ell_power_knee": k,
        "r2_ramp": r2,
        "ell_latency_knee": lat_knee,
        "latency_knee_found": bool(np.isfinite(lat_knee)),
    }
    return summary, rows, pred


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_figure(path: Path, rows: list[dict], summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ell = np.asarray([r["ell"] for r in rows], dtype=float)
    power = np.asarray([r["power_w"] for r in rows], dtype=float)
    sources = sorted({r["source_type"] for r in rows})
    colors = dict(zip(sources, plt.cm.tab10(np.linspace(0, 1, max(len(sources), 1)))))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for src in sources:
        sel = np.asarray([r["source_type"] == src for r in rows], dtype=bool)
        ax.scatter(ell[sel], power[sel], s=12, alpha=0.55, label=src,
                   color=colors[src], rasterized=True)

    xs = np.linspace(0, max(float(np.max(ell)) * 1.05, summary["ell_power_knee"] * 1.2), 300)
    ys = summary["P_idle_w"] + (summary["P_busy_w"] - summary["P_idle_w"]) * np.clip(
        xs / max(summary["ell_power_knee"], 1e-12), 0.0, 1.0
    )
    ax.plot(xs, ys, color="black", lw=2.0, label="ramp-plateau roofline")
    ax.axvline(summary["ell_power_knee"], color="black", ls="--", lw=1.2,
               label=f"power knee ell={summary['ell_power_knee']:.2f}")
    if summary["latency_knee_found"]:
        ax.axvline(summary["ell_latency_knee"], color="red", ls=":", lw=1.2,
                   label=f"latency knee ell={summary['ell_latency_knee']:.2f}")
    ax.set_xlabel("occupancy ell = f/F + g/G")
    ax.set_ylabel("measured node power [W]")
    ax.set_title(
        f"{summary['label']} roofline | F={summary['F_prefill_tps']:.0f}, "
        f"G={summary['G_decode_tps']:.0f} tok/s | R2={summary['r2_ramp']:.2f}"
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze_run_root(run_root: Path, *, label: str, model: str | None = None,
                     hardware: str | None = None, tp: int | None = None,
                     window_s: float = 5.0, out_root: Path = OUT_ROOT,
                     fig_root: Path = FIG_ROOT) -> dict:
    rows = collect_windows(run_root, window_s, model=model, hardware=hardware, tp=tp)
    summary, rows, _ = analyze_rows(rows, label)
    out_root.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)
    _write_csv(out_root / f"{label}_windows.csv", rows)
    _write_csv(out_root / f"{label}_summary.csv", [summary])
    _make_figure(fig_root / f"{label}_power_vs_ell.png", rows, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="data/runs")
    parser.add_argument("--label", default="roofline")
    parser.add_argument("--model", default=None)
    parser.add_argument("--hardware", default=None)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--fig-root", default=str(FIG_ROOT))
    args = parser.parse_args()

    summary = analyze_run_root(
        Path(args.run_root),
        label=args.label,
        model=args.model,
        hardware=args.hardware,
        tp=args.tp,
        window_s=args.window_s,
        out_root=Path(args.out_root),
        fig_root=Path(args.fig_root),
    )
    print(
        f"{summary['label']}: {summary['n_windows']} windows, "
        f"F={summary['F_prefill_tps']:.1f}, G={summary['G_decode_tps']:.1f}, "
        f"power knee ell={summary['ell_power_knee']:.3f}"
    )


if __name__ == "__main__":
    main()
