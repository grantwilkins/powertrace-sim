"""Extract measured iteration-timing calibration rows from probe bundles.

Reads manifest.json + levels/level_XXX.json + engine.csv from the local
Tier-1 llama-70b probe bundles and emits timing-test/probe_calibration.json.
Measured numbers only: no model, no fitting.

Per-level timing sources, in priority order:
  (a) detailed level result file (vllm benchmark_serving --save-detailed):
      mean/median TPOT, ITL, TTFT in ms  -> primary "measured" fields;
  (b) engine.csv Prometheus counters iteration_tokens_total_{count,sum}
      deltas within [t_start_epoch, t_end_epoch] -> iterations/s and
      tokens/iteration -> cross-check and effective-batch evidence.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Campaign directory -> served model (architecture-registry name).
# data/runs/a100_iteration_gpt-oss-120b is deliberately ABSENT: gpt-oss-120b
# is the A100 holdout model in the frozen timing split, so its probes are
# quarantined from every fit until the zero-shot scoring is complete.
BUNDLES = {
    REPO / "data/runs/a100_tier1_llama70b": "llama-3-70b",
    REPO / "data/runs/h100_tier1_llama70b": "llama-3-70b",
    REPO / "data/runs/a100_iteration_gpt-oss-20b": "gpt-oss-20b",
}
RELEVANT_PROBES = {
    "decode_staircase",
    "decode_context_grid",
    "prefill_staircase",
    "context_holds",
}
OUTPUT = REPO / "timing-test/probe_calibration.json"


def load_engine(run_dir: Path):
    rows = []
    with open(run_dir / "engine.csv") as f:
        for r in csv.DictReader(f):
            rows.append(
                (
                    float(r["timestamp"]),
                    float(r["num_requests_running"]),
                    float(r["iteration_tokens_total_sum"]),
                    float(r["iteration_tokens_total_count"]),
                    float(r["num_preemptions_total"]),
                )
            )
    return rows


def engine_window_stats(engine, t0, t1):
    """Counter deltas and mean running-batch within [t0, t1]."""
    win = [e for e in engine if t0 <= e[0] <= t1]
    # The level window includes benchmark-client setup/teardown idle time;
    # restrict to the active sub-window so rates are not biased low.
    active = [i for i, e in enumerate(win) if e[1] > 0]
    if active:
        win = win[max(active[0] - 1, 0) : active[-1] + 2]
    if len(win) < 2:
        return None
    dt = win[-1][0] - win[0][0]
    d_count = win[-1][3] - win[0][3]
    d_sum = win[-1][2] - win[0][2]
    d_preemptions = win[-1][4] - win[0][4]
    if dt <= 0 or d_count <= 0:
        return None
    running = [e[1] for e in win if e[1] > 0]
    out = {
        "window_s": dt,
        "iterations": d_count,
        "iterations_per_s": d_count / dt,
        "tokens_per_iteration": d_sum / d_count,
        "iteration_time_ms": 1000.0 * dt / d_count,
        "engine_tokens_per_s": d_sum / dt,
        "preemptions": d_preemptions,
        "num_running_mean": statistics.fmean(running) if running else 0.0,
    }
    # Steady-state sub-window: the benchmark client runs a single-request
    # warm-up before each level, so restrict to samples at >= 80% of the
    # window's peak concurrency for clean per-iteration rates.
    peak = max(e[1] for e in win)
    steady = [e for e in win if e[1] >= 0.8 * peak]
    if len(steady) >= 3:
        sdt = steady[-1][0] - steady[0][0]
        sc = steady[-1][3] - steady[0][3]
        ss = steady[-1][2] - steady[0][2]
        if sdt > 0 and sc > 0:
            out["steady"] = {
                "window_s": sdt,
                "iteration_time_ms": 1000.0 * sdt / sc,
                "tokens_per_iteration": ss / sc,
                "engine_tokens_per_s": ss / sdt,
                "num_running_mean": statistics.fmean(e[1] for e in steady),
            }
    return out


def extract_run(run_dir: Path, skips: list, model: str):
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        if any(p in run_dir.name for p in RELEVANT_PROBES):
            skips.append(f"{run_dir.relative_to(REPO)}: no manifest.json")
        return []
    m = json.loads(manifest_path.read_text())
    probe = m["probe"]["type"]
    if probe not in RELEVANT_PROBES:
        return []
    engine = load_engine(run_dir)
    git_sha = m.get("versions", {}).get("git_sha")
    rows = []
    for lv in m["probe"]["levels"]:
        rel = run_dir.relative_to(REPO)
        level_file = run_dir / "levels" / f"level_{lv['level']:03d}.json"
        if not level_file.exists():
            skips.append(f"{rel} level {lv['level']}: missing level file")
            continue
        det = json.loads(level_file.read_text())
        if not det.get("completed"):
            skips.append(f"{rel} level {lv['level']}: 0 completed requests")
            continue
        p = lv["params"]
        prompt_tokens = p["input_len"] + p["prefix_len"]
        out_len = p["output_len"]
        is_prefill = probe == "prefill_staircase"
        eng = engine_window_stats(engine, lv["t_start_epoch"], lv["t_end_epoch"])
        method = "level_detail(a)" + ("+engine_counters(b)" if eng else "")
        row = {
            "hardware": m["hardware"],
            "tp": m["tp"],
            "model": model,
            "probe": probe,
            "run_id": m["run_id"],
            "label": lv["label"],
            "level": lv["level"],
            "batch": lv["concurrency"],
            "num_prompts": lv["num_prompts"],
            "prompt_tokens": prompt_tokens,
            "output_len": out_len,
            # mean KV length over the decode phase (prompt + mean position)
            "context_tokens_mean": prompt_tokens + (out_len + 1) / 2.0,
            "level_duration_s": lv["summary"]["duration"],
            "output_throughput_tok_s": lv["summary"]["output_throughput"],
            "measured": {
                "mean_tpot_ms": det["mean_tpot_ms"],
                "median_tpot_ms": det["median_tpot_ms"],
                "mean_itl_ms": det["mean_itl_ms"],
                "median_itl_ms": det["median_itl_ms"],
                "mean_ttft_ms": det["mean_ttft_ms"],
                "median_ttft_ms": det["median_ttft_ms"],
                "p99_ttft_ms": det["p99_ttft_ms"],
            },
            "engine_counters": eng,
            "provenance": {
                "bundle_path": str(rel),
                "manifest_git_sha": git_sha,
                "extraction_method": method,
            },
        }
        if is_prefill:
            # Unloaded, concurrency 1: TTFT is prefill time (queue ~ 0).
            # The first request of each level hits the prefix cache (warmup
            # request shares its prompt) and shows near-zero TTFT; exclude
            # such cache-hit samples (< 50% of median) from the mean.
            ttfts_ms = [1000.0 * t for t in det["ttfts"]]
            med = statistics.median(ttfts_ms)
            full = [t for t in ttfts_ms if t >= 0.5 * med]
            row["prefill_time_ms_mean"] = det["mean_ttft_ms"]
            row["prefill_time_ms_median"] = det["median_ttft_ms"]
            row["prefill_time_ms_mean_excl_cache_hits"] = statistics.fmean(full)
            row["n_prefill_samples"] = len(ttfts_ms)
            row["n_cache_hit_samples_excluded"] = len(ttfts_ms) - len(full)
        else:
            row["decode_tokens_per_s_per_request"] = 1000.0 / det["mean_tpot_ms"]
            steady = (eng or {}).get("steady")
            row["effective_decode_batch"] = (
                steady["num_running_mean"]
                if steady
                else (eng["num_running_mean"] if eng else lv["concurrency"])
            )
        rows.append(row)
    return rows


def build():
    skips: list[str] = []
    rows = []
    for bundle, model in BUNDLES.items():
        if not bundle.exists():
            skips.append(f"{bundle.relative_to(REPO)}: bundle dir absent")
            continue
        for run_dir in sorted(bundle.iterdir()):
            if run_dir.is_dir():
                rows.extend(extract_run(run_dir, skips, model))
    doc = {
        "description": "Measured iteration-timing calibration points from "
        "probe bundles (Tier-1 llama-3-70b, iteration gpt-oss-20b). "
        "No fitting. gpt-oss-120b probes are quarantined (holdout model).",
        "n_rows": len(rows),
        "skipped": skips,
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


if __name__ == "__main__":
    doc = build()
    print(f"wrote {OUTPUT} with {doc['n_rows']} rows; {len(doc['skipped'])} skips")
    for s in doc["skipped"]:
        print("  skip:", s)
