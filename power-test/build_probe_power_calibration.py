"""Build source-only A100 power-calibration rows from controlled staircases.

Raw node-power targets stay at their native 250 ms cadence. Engine counters are
conserved over each request-active level because their updates are iteration
bursts, not instantaneous 250 ms work measurements. Only the prefill and decode
staircases are admissible: cached-context probes cannot identify instantaneous
KV work from logical prompt-token counters.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.training_data.arch import get_arch  # noqa: E402
from model.training_data.ledger_view import (  # noqa: E402
    effective_context,
    kv_bytes_per_token,
)
from model.training_data.run_record import load_bundle_run  # noqa: E402

DT_S = 0.25
SOURCE_ROOTS = (REPO / "data/runs/a100_tier1_llama70b",)
MODEL_ARCH = {"meta-llama/Llama-3.1-70B-Instruct": "llama-3-70b"}
PROBE_TYPES = {
    "decode_staircase",
    "prefill_staircase",
}
OUTPUT = REPO / "power-test/probe_power_calibration_250ms.npz"
PROVENANCE_OUTPUT = REPO / "power-test/probe_power_calibration_250ms.json"


def counter_window_rate(
    timestamps: np.ndarray,
    cumulative: np.ndarray,
    start: float,
    end: float,
) -> float:
    """Conserved average counter rate over a completed workload window."""
    timestamps = np.asarray(timestamps, float)
    cumulative = np.asarray(cumulative, float)
    if timestamps.size != cumulative.size or timestamps.size < 2:
        raise ValueError("Counter timestamps and values must have equal nonzero length")
    if np.any(np.diff(timestamps) <= 0.0) or np.any(np.diff(cumulative) < 0.0):
        raise ValueError("Counter timestamps must increase and counters cannot decrease")
    if end <= start or start < timestamps[0] or end > timestamps[-1]:
        raise ValueError("Counter window must lie inside the sampled interval")
    overlap = np.maximum(
        np.minimum(timestamps[1:], end) - np.maximum(timestamps[:-1], start),
        0.0,
    )
    interval_rates = np.diff(cumulative) / np.diff(timestamps)
    return float(overlap @ interval_rates) / (end - start)


def bin_means(
    timestamps: np.ndarray, values: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    bins = np.searchsorted(edges, np.asarray(timestamps, float), side="right") - 1
    keep = (bins >= 0) & (bins < edges.size - 1) & np.isfinite(values)
    sums = np.bincount(bins[keep], weights=values[keep], minlength=edges.size - 1)
    counts = np.bincount(bins[keep], minlength=edges.size - 1)
    return np.divide(
        sums, counts, out=np.full(edges.size - 1, np.nan), where=counts > 0
    )


def balanced_level_weights(level_ids: np.ndarray, target_per_level: float) -> np.ndarray:
    """Give every controlled level the same total regression weight."""
    level_ids = np.asarray(level_ids)
    if target_per_level <= 0.0:
        raise ValueError("Target level weight must be positive")
    weights = np.empty(level_ids.size, dtype=float)
    for level in np.unique(level_ids):
        selected = level_ids == level
        weights[selected] = target_per_level / selected.sum()
    return weights


def _level_rows(record, manifest: dict, level: dict, arch: dict) -> dict:
    level_start = float(level["t_start_epoch"])
    level_end = float(level["t_end_epoch"])
    arrivals = np.asarray(record.request_timestamps, float)
    requests = (arrivals >= level_start) & (arrivals <= level_end)
    if not np.any(requests):
        raise ValueError(f"Probe level {level['label']} has no request rows")
    start = float(np.min(arrivals[requests]))
    end = float(np.max(
        arrivals[requests] + record.ttfts[requests] + record.decode_times[requests]
    ))
    edges = start + np.arange(int(np.floor((end - start) / DT_S)) + 1) * DT_S
    engine_time = np.asarray(record.engine_table["timestamp"], float)
    counter_start = max(start, float(engine_time[0]))
    counter_end = min(end, float(engine_time[-1]))
    if counter_end <= counter_start:
        raise ValueError(f"Probe level {level['label']} has no engine overlap")

    def rate(name: str) -> float:
        return counter_window_rate(
            engine_time,
            np.asarray(record.engine_table[name], float),
            counter_start,
            counter_end,
        )

    pre_tok = rate("prompt_tokens_total")
    dec_tok = rate("generation_tokens_total")
    iterations = rate("iteration_tokens_total_count")
    iteration_tokens = rate("iteration_tokens_total_sum")
    engine_window = (engine_time >= start) & (engine_time <= end)
    running = float(np.mean(
        np.asarray(record.engine_table["num_requests_running"], float)[engine_window]
    ))
    power = bin_means(record.power_timestamps, record.tp_sum_power(), edges)
    active = np.isfinite(power)
    if not np.any(active):
        raise ValueError(f"Probe level {level['label']} has no active power bins")

    params = level["params"]
    context = (
        float(params["input_len"])
        + float(params.get("prefix_len", 0))
        + (float(params["output_len"]) + 1.0) / 2.0
    )
    kv_token_bytes = kv_bytes_per_token(arch)
    count = int(active.sum())
    level_id = f"{manifest['run_id']}:{level['label']}"
    return {
        "pre_tok": np.full(count, pre_tok),
        "dec_tok": np.full(count, dec_tok),
        "batch": np.full(count, running),
        "busy": np.ones(count),
        "w_read": np.full(count, float(arch["w_bytes"]) * iterations),
        "kv_read": (
            np.full(count, dec_tok * effective_context(context, arch) * kv_token_bytes)
        ),
        "kv_write": np.full(count, (pre_tok + dec_tok) * kv_token_bytes),
        "engine_iterations_rate": np.full(count, iterations),
        "engine_iteration_tokens_rate": np.full(count, iteration_tokens),
        "logit_tokens_rate": np.full(count, dec_tok),
        "tp": np.full(count, record.tp, dtype=float),
        "n_active": np.full(count, float(arch["n_active"])),
        "transformer_active_params": np.full(
            count, float(arch["transformer_active_params"])
        ),
        "output_head_params": np.full(count, float(arch["output_head_params"])),
        "w_bytes": np.full(count, float(arch["w_bytes"])),
        "fp8": np.full(count, float(arch.get("fp8", 0))),
        "fp8_flop_frac": np.full(
            count,
            float(arch.get("fp8_flop_frac", 1.0 if arch.get("fp8", 0) else 0.0)),
        ),
        "power": power[active],
        "level_id": np.full(count, level_id),
        "probe_type": np.full(count, manifest["probe"]["type"]),
        "hardware": np.full(count, record.hardware),
    }


def build(source_roots=SOURCE_ROOTS) -> tuple[dict, dict]:
    columns = defaultdict(list)
    levels = []
    for root in source_roots:
        for manifest_path in sorted(Path(root).glob("*/manifest.json")):
            manifest = json.loads(manifest_path.read_text())
            probe_type = (manifest.get("probe") or {}).get("type")
            if manifest.get("hardware") != "A100" or probe_type not in PROBE_TYPES:
                continue
            model = str(manifest["model"])
            if model not in MODEL_ARCH:
                raise ValueError(f"Unapproved probe calibration model {model!r}")
            record = load_bundle_run(manifest_path.parent)
            arch = dict(get_arch(MODEL_ARCH[model]))
            for level in manifest["probe"]["levels"]:
                rows = _level_rows(record, manifest, level, arch)
                for key, values in rows.items():
                    columns[key].append(values)
                levels.append({
                    "level_id": str(rows["level_id"][0]),
                    "probe_type": probe_type,
                    "active_bins": int(rows["power"].size),
                    "bundle": str(manifest_path.parent.relative_to(REPO)),
                    "manifest_sha256": record.provenance["sha256"]["manifest.json"],
                    "power_sha256": record.provenance["sha256"]["power.csv"],
                    "engine_sha256": record.provenance["sha256"]["engine.csv"],
                })
    if not levels:
        raise ValueError("No approved A100 controlled probe levels found")
    data = {key: np.concatenate(values) for key, values in columns.items()}
    data["dt_s"] = np.asarray(DT_S)
    provenance = {
        "schema_version": "probe-power-calibration-v1",
        "selection": {
            "hardware": "A100",
            "models": sorted(MODEL_ARCH),
            "probe_types": sorted(PROBE_TYPES),
            "excluded_probe_types": {
                "decode_context": "logical cached context is not dynamic KV work",
                "context_hold_grid": "logical cached context is not dynamic KV work",
                "mixed_grid": "prefill and decode work are confounded",
                "transient": "not a stationary calibration level",
            },
            "target_or_holdout_bundles_used": False,
        },
        "rows": int(data["power"].size),
        "levels": levels,
    }
    return data, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--provenance-out", type=Path, default=PROVENANCE_OUTPUT)
    args = parser.parse_args()
    data, provenance = build()
    np.savez_compressed(args.out, **data)
    args.provenance_out.write_text(json.dumps(provenance, indent=2) + "\n")
    counts = {
        probe: sum(level["probe_type"] == probe for level in provenance["levels"])
        for probe in sorted(PROBE_TYPES)
    }
    print(
        f"wrote {args.out} with {provenance['rows']} active bins, "
        f"{len(provenance['levels'])} balanced levels: {counts}"
    )


if __name__ == "__main__":
    main()
