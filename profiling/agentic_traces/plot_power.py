"""Plot the power trace of an agentic run bundle against its engine state.

    python plot_power.py <run_dir> -o out.png

Top panel: GPU power draw (+ SM clock). Bottom panel: engine state
(num_requests_running, KV-cache usage). Per-session spans are shaded and per-turn
arrivals ticked, so the tool-gap idle intervals — the thing the gap model captures —
are visible as the low-power, zero-request troughs between decode bursts.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _power_epoch(ts: str) -> float:
    """nvidia-smi stamps local wall-clock; parse to epoch (tz-corrected later)."""
    ts = ts.strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(f"unparseable power timestamp: {ts!r}")


def _read_csv(path: Path):
    with open(path) as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    return header, rows[1:]


def load_run(run_dir: Path):
    _, prows = _read_csv(run_dir / "power.csv")
    _, erows = _read_csv(run_dir / "engine.csv")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    requests = json.loads((run_dir / "requests.json").read_text())

    eng_t = [float(r[0]) for r in erows]
    # Align power onto the engine's true epoch clock by matching their starts
    # (corrects any nvidia-smi local-vs-UTC timezone fold).
    pwr_raw = [_power_epoch(r[0]) for r in prows]
    shift = eng_t[0] - pwr_raw[0]
    t0 = eng_t[0]

    return {
        "t0": t0,
        "pwr_t": [t + shift - t0 for t in pwr_raw],
        "power": [float(r[1]) for r in prows],
        "sm_clock": [float(r[2]) for r in prows],
        "eng_t": [t - t0 for t in eng_t],
        "running": [float(r[1]) for r in erows],
        "cache": [float(r[3]) * 100 for r in erows],  # fraction -> %
        "manifest": manifest,
        "requests": requests,
    }


def plot(run_dir: Path, out: Path):
    d = load_run(run_dir)
    t0, man = d["t0"], d["manifest"]
    sessions = man["probe"]["sessions"]
    colors = plt.cm.tab10.colors

    fig, (ax_p, ax_e) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                     height_ratios=[2, 1])

    # --- power + SM clock ---
    ax_p.plot(d["pwr_t"], d["power"], lw=1.0, color="#c1121f", label="power.draw")
    ax_p.set_ylabel("power (W)", color="#c1121f")
    ax_p.tick_params(axis="y", labelcolor="#c1121f")
    ax_c = ax_p.twinx()
    ax_c.plot(d["pwr_t"], d["sm_clock"], lw=0.8, color="#457b9d", alpha=0.6)
    ax_c.set_ylabel("SM clock (MHz)", color="#457b9d")
    ax_c.tick_params(axis="y", labelcolor="#457b9d")

    # --- engine state ---
    ax_e.plot(d["eng_t"], d["running"], lw=1.2, color="#2a9d8f", label="requests running")
    ax_e.set_ylabel("# running", color="#2a9d8f")
    ax_e.tick_params(axis="y", labelcolor="#2a9d8f")
    ax_k = ax_e.twinx()
    ax_k.plot(d["eng_t"], d["cache"], lw=0.9, color="#e9c46a")
    ax_k.set_ylabel("KV cache (%)", color="#b08900")
    ax_e.set_xlabel("seconds since run start")

    # --- session spans + turn arrivals ---
    for i, s in enumerate(sessions):
        a, b = s["t_start_epoch"] - t0, s["t_end_epoch"] - t0
        for ax in (ax_p, ax_e):
            ax.axvspan(a, b, color=colors[i % 10], alpha=0.07)
        ax_p.text(a, ax_p.get_ylim()[1], f" {s['session_id']} ({s['n_turns']}t)",
                  va="top", ha="left", fontsize=8, color=colors[i % 10])
    for ts in d["requests"]["request_timestamps"]:
        ax_p.axvline(ts - t0, color="0.6", lw=0.4, alpha=0.5)

    idle = sum(g for g in d["requests"]["post_gap_s"])
    ax_p.set_title(
        f"{man['model']} on {man['hardware']} — agentic replay "
        f"(prefix_cache={man['probe']['prefix_cache']}) | "
        f"{len(d['requests']['input_lens'])} turns, {idle:.0f}s tool-gap idle")
    ax_p.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()
    plot(args.run_dir, args.out)


if __name__ == "__main__":
    main()
