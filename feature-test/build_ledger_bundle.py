"""Build the per-bin work ledger from §2 self-describing run bundles.

This is the new (additive) ledger builder for the profiling campaign. It reads a
``data/runs/<campaign_id>/<run_id>/`` bundle (``power.csv`` + ``engine.csv`` +
``requests.json`` + ``manifest.json``) and emits the SAME ``ledger_cache.npz``
schema that ``feature-test/build_ledger_cache.py`` produces, so
``fit_map_priors.py`` / ``peak_and_holdout.py`` / ``final_model.py`` consume it
unchanged.

Legacy pairs and bundles use the same reconstruction implementation in
``model.training_data.ledger_view``. Two source-state paths share its bin-level
work-rate math:

* ``reconstruct_bins`` — the ttft/itl reconstruction path. Its guarded additions are
  guarded branches that are inert when ``n_linear_layers == 0`` and when a
  manifest clock offset is supplied, so existing softmax runs are unchanged.
* ``bins_from_engine_csv`` — the measured hybrid path (stock vLLM ``/metrics``
  plus request timing). Primary for new measured-ledger campaigns.

Run from repo root:
    uv run python feature-test/build_ledger_bundle.py --runs-glob 'data/runs/*/*'
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model.training_data.ledger_view import (  # noqa: E402
    KV_ELEM_BYTES,  # noqa: F401  (re-exported for downstream users)
    bin_work_rates as _bin_work_rates,  # noqa: F401
    reconstruct_bins,
    reconstruct_bins_from_record,
)
from model.training_data.power_parsing import parse_power_csv, parse_request_json  # noqa: E402
from model.training_data.run_record import load_bundle_run  # noqa: E402

GIB = 1024.0**3
HARDWARE_INDEX = {"A100": 0, "H100": 1}
DEFAULT_RUNS_GLOB = "data/runs/*/*"

# Per-bin output arrays (must match build_ledger_cache for schema parity).
BIN_KEYS = (
    "power", "pre_tok", "dec_tok", "batch", "pre_active", "iters",
    "w_read", "w_read_pre", "w_read_dec", "kv_read", "kv_write", "comm",
    "arrivals", "input_tokens_arriving", "output_tokens_requested",
    "A_t", "delta_A_t", "running_requests", "waiting_requests",
)


def state_from_requests(json_path, csv_path, arch, tp, lambda_prefill, dt=1.0,
                        trim_s=5.0):
    """Reconstruction path entry point (parse old-format bundle -> bins)."""
    req = parse_request_json(json_path)
    pw = parse_power_csv(csv_path, tensor_parallelism=tp)
    return reconstruct_bins(req, pw, arch, tp, lambda_prefill, dt=dt, trim_s=trim_s)


MEASURED_ENGINE_FIELDS = (
    "timestamp", "num_requests_running", "num_requests_waiting",
    "gpu_cache_usage_perc", "prompt_tokens_total", "generation_tokens_total",
    "iteration_tokens_total_sum", "iteration_tokens_total_count",
)

MEASURED_LEDGER_KEYS = (
    "engine_iteration_tokens_rate",
    "engine_iterations_rate",
    "engine_tokens_per_iteration",
    "engine_gpu_cache_usage",
)


def ledger_bin_keys(state_source: str) -> tuple[str, ...]:
    if state_source == "reconstruction":
        return BIN_KEYS
    if state_source == "measured_engine":
        return BIN_KEYS + MEASURED_LEDGER_KEYS
    raise ValueError(f"Unknown bundle state source: {state_source!r}")


def _counter_rate(table, name, edges):
    """Cumulative counter -> conserved rate on half-open bin edges."""
    timestamps = np.asarray(table["timestamp"], dtype=np.float64)
    values = np.asarray(table[name], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) < -1e-9):
        raise ValueError(f"engine counter {name!r} must be finite and nondecreasing")
    if timestamps[0] > edges[0] or timestamps[-1] < edges[-1]:
        raise ValueError(f"engine counter {name!r} does not cover the ledger grid")
    return np.diff(np.interp(edges, timestamps, values)) / np.diff(edges)


def _gauge_mean(table, name, edges):
    """Piecewise-linear time mean of a gauge in every half-open bin."""
    timestamps = np.asarray(table["timestamp"], dtype=np.float64)
    values = np.asarray(table[name], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"engine gauge {name!r} must be finite and non-negative")
    if timestamps[0] > edges[0] or timestamps[-1] < edges[-1]:
        raise ValueError(f"engine gauge {name!r} does not cover the ledger grid")
    output = np.empty(edges.size - 1)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        inside = timestamps[(timestamps > left) & (timestamps < right)]
        points = np.r_[left, inside, right]
        output[index] = np.trapezoid(
            np.interp(points, timestamps, values), points
        ) / (right - left)
    return output


def bins_from_engine_csv(record, *, lambda_prefill, dt=1.0, trim_s=5.0):
    """Project measured engine state into the maintained ledger schema.

    Request reconstruction supplies offered marks, phase-specific active state,
    prefill iterations, and context geometry. Stock vLLM counters replace only
    fields with an exact map: actually computed prompt/decode tokens and total
    running/waiting state. Iteration-token histogram counters are retained as
    diagnostics. Per-field lineage makes this hybrid contract explicit.
    """
    table = record.engine_table
    missing = sorted(set(MEASURED_ENGINE_FIELDS) - set(table))
    if missing:
        raise ValueError(f"engine.csv lacks measured-ledger fields: {missing}")
    timestamps = np.asarray(table["timestamp"], dtype=np.float64)
    if not np.all(np.isfinite(timestamps)) or not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("engine timestamps must be finite and strictly increasing")

    base = reconstruct_bins_from_record(
        record, lambda_prefill=lambda_prefill, dt=dt, trim_s=trim_s,
        include_time=True,
    )
    if base is None:
        return None
    time_epoch = np.asarray(base["time_epoch_s"], dtype=np.float64)
    if time_epoch.size > 1 and not np.allclose(np.diff(time_epoch), dt):
        raise ValueError("Measured-state ledger requires a complete uniform power grid")
    edges = np.r_[time_epoch[0] - dt, time_epoch]

    pre_tok = _counter_rate(table, "prompt_tokens_total", edges)
    dec_tok = _counter_rate(table, "generation_tokens_total", edges)
    batch = np.asarray(base["batch"], dtype=np.float64)
    pre_active = np.asarray(base["pre_active"], dtype=np.float64)
    pre_iter = np.asarray(base["w_read_pre"], dtype=np.float64) / record.arch["w_bytes"]
    measured = _bin_work_rates(
        pre_tok, dec_tok, batch, pre_active, pre_iter,
        np.asarray(base["kv_read"], dtype=np.float64),
        record.arch, record.tp, pre_tok.size,
    )

    running = _gauge_mean(table, "num_requests_running", edges)
    waiting = _gauge_mean(table, "num_requests_waiting", edges)
    iteration_tokens = _counter_rate(table, "iteration_tokens_total_sum", edges)
    iterations = _counter_rate(table, "iteration_tokens_total_count", edges)
    measured.update({
        key: base[key] for key in (
            "power", "arrivals", "input_tokens_arriving", "output_tokens_requested"
        )
    })
    measured.update({
        "A_t": running + waiting,
        "delta_A_t": np.r_[0.0, np.diff(running + waiting)],
        "running_requests": running,
        "waiting_requests": waiting,
        "engine_iteration_tokens_rate": iteration_tokens,
        "engine_iterations_rate": iterations,
        "engine_tokens_per_iteration": np.divide(
            iteration_tokens, iterations,
            out=np.zeros_like(iteration_tokens), where=iterations > 0.0,
        ),
        "engine_gpu_cache_usage": _gauge_mean(
            table, "gpu_cache_usage_perc", edges
        ),
        "time_epoch_s": time_epoch,
        "n": int(pre_tok.size),
        "arch": base["arch"],
        "field_sources": {
            "pre_tok": "engine.prompt_tokens_total",
            "dec_tok": "engine.generation_tokens_total",
            "A_t": "engine.num_requests_running+num_requests_waiting",
            "running_requests": "engine.num_requests_running",
            "waiting_requests": "engine.num_requests_waiting",
            "batch": "requests.itls reconstruction",
            "pre_active": "requests.ttft reconstruction",
            "pre_iter": "requests.ttft+lambda_prefill reconstruction",
            "kv_read": "requests.itls+architecture reconstruction",
            "engine_iteration_tokens_rate": "engine.iteration_tokens_total_sum",
            "engine_iterations_rate": "engine.iteration_tokens_total_count",
            "engine_gpu_cache_usage": "engine.gpu_cache_usage_perc",
        },
    })
    return measured


# --------------------------------------------------------------------------- #
# Bundle reading + npz assembly
# --------------------------------------------------------------------------- #

def rate_from_manifest(manifest: dict) -> float:
    """Achieved request throughput (req/s) from the per-level summaries.

    Probes run closed-loop (``--request-rate inf --max-concurrency N``) so no
    offered rate exists; the emitter records per-level ``summary.duration`` /
    ``summary.completed`` and this is their duration-weighted mean. Idle levels
    (all-zero summaries) contribute nothing. 0.0 when no traffic levels exist.
    """
    levels = (manifest.get("probe") or {}).get("levels") or []
    completed = 0.0
    duration = 0.0
    for level in levels:
        summary = level.get("summary") or {}
        completed += float(summary.get("completed") or 0.0)
        duration += float(summary.get("duration") or 0.0)
    return completed / duration if duration > 0 else 0.0


def build_bundle(
    run_dir, *, lambda_prefill, lambda_prefill_source, dt=1.0,
    state_source="reconstruction",
):
    """Build one bundle through an explicitly named state projection."""
    record = load_bundle_run(run_dir)
    if state_source == "reconstruction":
        bins = reconstruct_bins_from_record(
            record, lambda_prefill=lambda_prefill, dt=dt
        )
    elif state_source == "measured_engine":
        bins = bins_from_engine_csv(
            record, lambda_prefill=lambda_prefill, dt=dt
        )
    else:
        raise ValueError(f"Unknown bundle state source: {state_source!r}")
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    manifest["_ledger_source"] = {
        "run_dir": str(Path(run_dir)),
        "sha256": record.provenance["sha256"],
        "lambda_prefill_tok_s": float(lambda_prefill),
        "lambda_prefill_source": str(lambda_prefill_source),
        "state_source": state_source,
        "field_sources": bins.get(
            "field_sources", {"work_and_state": "requests timing reconstruction"}
        ) if bins is not None else {},
    }
    return bins, manifest


def discover_run_dirs(runs_glob: str) -> list[Path]:
    """Find default bundle roots without treating campaign support dirs as runs.

    A manifest marks a directory as a bundle candidate. The default scan skips
    sibling directories such as ``logs`` that do not have one, but keeps a
    manifest-bearing incomplete bundle so normal ingestion reports its error.
    An explicitly supplied glob is always returned unfiltered.
    """
    run_dirs = sorted(path for path in Path().glob(runs_glob) if path.is_dir())
    if runs_glob != DEFAULT_RUNS_GLOB:
        return run_dirs
    return [path for path in run_dirs if (path / "manifest.json").is_file()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", default=DEFAULT_RUNS_GLOB)
    ap.add_argument("--out", default="feature-test/ledger_cache_bundle.npz")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--lambda-prefill", type=float, required=True)
    ap.add_argument("--lambda-prefill-source", required=True)
    ap.add_argument(
        "--state-source", choices=("reconstruction", "measured_engine"),
        default="reconstruction",
    )
    args = ap.parse_args()

    run_dirs = discover_run_dirs(args.runs_glob)
    cols = defaultdict(list)
    model_names, model_arch_json, family_names, hw_names = [], [], [], ["A100", "H100"]
    source_runs = []
    source_config = None
    n_ok = 0
    for rd in run_dirs:
        bins, m = build_bundle(
            rd, lambda_prefill=args.lambda_prefill,
            lambda_prefill_source=args.lambda_prefill_source, dt=args.dt,
            state_source=args.state_source,
        )
        if bins is None:
            continue
        n = bins["n"]
        a = bins["arch"]
        model = m["model"]
        config_key = (model, m["hardware"], int(m["tp"]))
        if source_config is None:
            source_config = config_key
        elif config_key != source_config:
            raise ValueError(
                "One --lambda-prefill value cannot calibrate multiple bundle configs"
            )
        family = m["arch"].get("family", "unknown")
        if model not in model_names:
            model_names.append(model)
            model_arch_json.append(json.dumps(m["arch"], sort_keys=True))
        if family not in family_names:
            family_names.append(family)
        for key in ledger_bin_keys(args.state_source):
            cols[key].append(bins[key])
        cols["n_active"].append(np.full(n, a["n_active"]))
        cols["w_bytes"].append(np.full(n, a["w_bytes"]))
        cols["fp8"].append(np.full(n, a.get("fp8", 0)))
        cols["tp"].append(np.full(n, float(m["tp"])))
        cols["rate"].append(np.full(n, rate_from_manifest(m)))
        cols["run_id"].append(np.full(n, n_ok, dtype=np.int32))
        cols["model_idx"].append(np.full(n, model_names.index(model), dtype=np.int32))
        hw = m["hardware"]
        if hw not in HARDWARE_INDEX:
            raise ValueError(f"Unsupported ledger hardware: {hw!r}")
        cols["hw_idx"].append(np.full(n, HARDWARE_INDEX[hw], dtype=np.int32))
        cols["family_idx"].append(np.full(n, family_names.index(family), dtype=np.int32))
        source_runs.append({
            "run_index": n_ok,
            "run_id": str(m.get("run_id", rd.name)),
            **m["_ledger_source"],
        })
        n_ok += 1

    if not cols:
        print("No bundles parsed.")
        return
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["model_names"] = np.array(model_names)
    out["model_arch_json"] = np.asarray(model_arch_json)
    out["family_names"] = np.array(family_names)
    out["hw_names"] = np.array(hw_names)
    out["lambda_prefill_tok_s"] = np.asarray(float(args.lambda_prefill))
    out["lambda_prefill_source"] = np.asarray(args.lambda_prefill_source)
    out["dt_s"] = np.asarray(float(args.dt))
    np.savez_compressed(args.out, **out)
    sidecar = Path(str(args.out) + ".manifest.json")
    sidecar.write_text(json.dumps({
        "ledger_schema_version": 2,
        "ledger_path": str(args.out),
        "lambda_prefill_tok_s": float(args.lambda_prefill),
        "lambda_prefill_source": args.lambda_prefill_source,
        "state_source": args.state_source,
        "runs": source_runs,
    }, indent=2, sort_keys=True) + "\n")
    print(f"Parsed {n_ok}/{len(run_dirs)} bundles -> {out['power'].size} bins -> {args.out}")


if __name__ == "__main__":
    main()
