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
* ``bins_from_engine_csv`` — the measured-state path (vLLM ``/metrics``). Primary
  for new bundles; compared against reconstruction in Phase-2.

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
)


def state_from_requests(json_path, csv_path, arch, tp, lambda_prefill, dt=1.0,
                        trim_s=5.0):
    """Reconstruction path entry point (parse old-format bundle -> bins)."""
    req = parse_request_json(json_path)
    pw = parse_power_csv(csv_path, tensor_parallelism=tp)
    return reconstruct_bins(req, pw, arch, tp, lambda_prefill, dt=dt, trim_s=trim_s)


def bins_from_engine_csv(*args, **kwargs):
    """Measured-state (engine.csv) consumption — DEFERRED to Phase-2.

    The /metrics scraper already COLLECTS engine.csv; consuming it as the ledger
    state source is intentionally not implemented here. The reconstruction path
    above is the single source for now. A first draft of this function
    was removed because it produced biased data; implement it only against REAL
    bundles, getting each of these right (each was a bug in that draft):

      1. Per-bin token rate: interpolate the cumulative counter onto the bin
         EDGES and diff — ``np.diff(np.interp(edges, t - t0, counter)) / dt``.
         Do NOT use ``(last - first)`` of the samples strictly inside a bin: that
         drops the increment between a bin's last sample and the next bin's first
         sample (~25% undercount at 4 Hz / 1 s bins, ~50% at 2 Hz).
      2. Clock alignment: engine.csv stamps ``time.time()`` (true epoch) while
         power.csv is nvidia-smi local wall time coerced to UTC by
         ``power_timestamp_to_epoch`` — a whole-hour skew off-UTC hosts. Align
         both to one epoch before binning (reconstruction sidesteps this via the
         %1800 fold; the measured path cannot).
      3. Fill ``pre_iter`` / ``kv_read`` from the logged-but-currently-unused
         counters (``request_prefill_time_seconds_sum`` for prefill iterations;
         ``gpu_cache_usage_perc`` — a gauge, bin-MEAN it — for KV occupancy).
         Never emit zeros for these: ``w_read_pre``/``kv_read`` are live fit FEATS,
         and zeros would bias e_w_pre / e_kv and inflate residual variance.
      4. Validate measured-vs-reconstructed agreement before trusting it (Phase-2).
    """
    raise NotImplementedError(bins_from_engine_csv.__doc__)


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


def build_bundle(run_dir, *, lambda_prefill, lambda_prefill_source, dt=1.0):
    """Build per-bin arrays for one bundle via the reconstruction path.

    engine.csv is collected by the scraper but not consumed here yet — measured-
    state consumption is Phase-2 (see ``bins_from_engine_csv``). Reconstruction is
    the shared source.
    """
    record = load_bundle_run(run_dir)
    bins = reconstruct_bins_from_record(
        record, lambda_prefill=lambda_prefill, dt=dt
    )
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    manifest["_ledger_source"] = {
        "run_dir": str(Path(run_dir)),
        "sha256": record.provenance["sha256"],
        "lambda_prefill_tok_s": float(lambda_prefill),
        "lambda_prefill_source": str(lambda_prefill_source),
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
        for key in BIN_KEYS:
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
        "runs": source_runs,
    }, indent=2, sort_keys=True) + "\n")
    print(f"Parsed {n_ok}/{len(run_dirs)} bundles -> {out['power'].size} bins -> {args.out}")


if __name__ == "__main__":
    main()
