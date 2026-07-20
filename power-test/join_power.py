"""Join measured node power onto the simulated ledger's 250 ms bin grid.

power-test DESIGN.md contract 1. For every run in
timing-test/timing_dataset.runs.json: verify the power CSV sha256, parse it
per GPU, TP-sum, and map each power sample onto the run's simulator clock.
The timing clock zero is the earliest validated request timestamp (t0_req,
bit-identical to timing-test/build_timing_dataset.py). A power sample lands
at p_ts - p_ts[0] - delta on that clock, where

    delta = (t0_req - p_ts[0]) - K * 1800,  K = round((t0_req - p_ts[0]) / 1800)

(the fold_1800 alignment policy: 1800 s cancels whole/half-hour clock skew;
constants plan-fixed, shared with model/training_data/alignment.py). The
evidence base says K = 0 on every run, so K != 0 raises, as does the fold
gate (earliest arrival must land in [-2, 600] s of power start).

Run from the repository root:
    uv run python power-test/join_power.py            # full 450-run join
    uv run python power-test/join_power.py --run-id 0 # single-run smoke, no writes

Outputs:
- power-test/sim_ledger_power_250ms.npz — identical to the sim cache with
  `power` filled (per-bin mean of TP-summed samples, NaN where a bin has no
  sample) and a boolean `power_valid` column added.
- power-test/sim_ledger_power_250ms.provenance.json — schema_version,
  constants with provenance class, per-run {run_id, delta_s, k_fold, n_bins,
  n_empty_power_bins, sha256_verified}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model.training_data.power_parsing import (  # noqa: E402
    parse_power_csv_per_gpu,
    parse_request_json,
    tp_sum_power,
)
from model.utils.io import write_json  # noqa: E402

SCHEMA_VERSION = "sim-ledger-power-v1"

# All plan-fixed, shared with model/training_data/alignment.py fold_1800.
FOLD_PERIOD_S = 1800.0
FOLD_GATE_MIN_S = -2.0
FOLD_GATE_MAX_S = 600.0
GPUS_PER_NODE = 8  # plan-fixed: legacy sharegpt nodes are 8-GPU


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_fold(t0_req: float, power_t0: float) -> tuple[float, int]:
    """(delta_s, K) of the fold_1800 alignment for one run."""
    offset = float(t0_req) - float(power_t0)
    k = int(round(offset / FOLD_PERIOD_S))
    return offset - k * FOLD_PERIOD_S, k


def join_run(power_csv: str, requests_json: str, tp: int, n_bins: int,
             dt: float) -> dict:
    """Per-bin mean TP-group power for one run on its simulator grid.

    Returns {"power": (n_bins,) float64 with NaN gaps, "delta_s", "k_fold"}.
    Raises when the parse fails, K != 0, or the fold gate rejects the run.
    """
    per_gpu = parse_power_csv_per_gpu(power_csv, gpus_per_node=GPUS_PER_NODE)
    if per_gpu is None:
        raise ValueError(f"Power CSV is not a raw per-GPU stream: {power_csv}")
    p_ts = np.asarray(per_gpu["timestamps"], dtype=np.float64)
    node_power = tp_sum_power(per_gpu["power_per_gpu"], tp)

    requests = parse_request_json(requests_json)
    if requests is None:
        raise ValueError(f"Request JSON failed validation: {requests_json}")
    t0_req = float(np.min(requests["request_timestamps"]))

    delta, k = compute_fold(t0_req, float(p_ts[0]))
    if k != 0:
        raise ValueError(f"Fold factor K={k} != 0 for {power_csv}")
    # Earliest validated arrival relative to power start (post-fold) is delta.
    if delta < FOLD_GATE_MIN_S or delta > FOLD_GATE_MAX_S:
        raise ValueError(
            f"Fold gate failed: earliest arrival at {delta:.3f} s "
            f"outside [{FOLD_GATE_MIN_S}, {FOLD_GATE_MAX_S}] for {power_csv}"
        )

    bins = np.floor((p_ts - p_ts[0] - delta) / dt).astype(np.int64)
    keep = (bins >= 0) & (bins < n_bins)
    sums = np.bincount(bins[keep], weights=node_power[keep], minlength=n_bins)
    counts = np.bincount(bins[keep], minlength=n_bins)
    power = np.full(n_bins, np.nan, dtype=np.float64)
    filled = counts > 0
    power[filled] = sums[filled] / counts[filled]
    return {"power": power, "delta_s": float(delta), "k_fold": int(k)}


def run_slices(run_id_column: np.ndarray) -> dict[int, slice]:
    """Contiguous cache-row slice per run_id (rows are grid-ordered per run)."""
    rid = np.asarray(run_id_column)
    if np.any(np.diff(rid) < 0):
        raise ValueError("Sim cache run_id column is not grouped by run")
    ids = np.unique(rid)
    starts = np.searchsorted(rid, ids, side="left")
    ends = np.searchsorted(rid, ids, side="right")
    return {int(i): slice(int(a), int(b)) for i, a, b in zip(ids, starts, ends)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default="feature-test/ledger_cache_sim_250ms.npz")
    parser.add_argument("--runs-json", default="timing-test/timing_dataset.runs.json")
    parser.add_argument("--out", default="power-test/sim_ledger_power_250ms.npz")
    parser.add_argument(
        "--provenance-out",
        default="power-test/sim_ledger_power_250ms.provenance.json")
    parser.add_argument(
        "--run-id", type=int, default=None,
        help="Smoke mode: join only this run, print its summary, write nothing")
    args = parser.parse_args()

    cache = np.load(args.cache, allow_pickle=True)
    dt = float(cache["dt_s"])
    run_ids = cache["run_id"]
    slices = run_slices(run_ids)

    index = json.loads(Path(args.runs_json).read_text())
    runs_by_id = {int(r["run_id"]): r for r in index["runs"]}
    if set(runs_by_id) != set(slices):
        raise ValueError("runs.json and sim cache disagree on run_id keys")

    targets = sorted(slices) if args.run_id is None else [args.run_id]
    power_out = np.full(run_ids.shape, np.nan, dtype=np.float64)
    tp_col = cache["tp"]
    per_run_provenance = []

    for rid in targets:
        entry = runs_by_id[rid]
        sl = slices[rid]
        tp_values = np.unique(tp_col[sl])
        if tp_values.size != 1:
            raise ValueError(f"Run {rid} has non-constant tp in the sim cache")
        tp = int(tp_values[0])

        power_csv = entry["paths"]["power_csv"]
        digest = sha256_file(power_csv)
        if digest != entry["sha256"]["power_csv"]:
            raise ValueError(f"sha256 mismatch for {power_csv}")

        n_bins = sl.stop - sl.start
        joined = join_run(power_csv, entry["paths"]["requests_json"], tp,
                          n_bins, dt)
        power_out[sl] = joined["power"]
        n_empty = int(np.sum(~np.isfinite(joined["power"])))
        per_run_provenance.append({
            "run_id": rid,
            "delta_s": joined["delta_s"],
            "k_fold": joined["k_fold"],
            "n_bins": n_bins,
            "n_empty_power_bins": n_empty,
            "sha256_verified": True,
        })
        if args.run_id is not None:
            print(f"run {rid}: tp={tp} n_bins={n_bins} "
                  f"delta_s={joined['delta_s']:.3f} k_fold={joined['k_fold']} "
                  f"empty_bins={n_empty} ({100.0 * n_empty / n_bins:.2f}%)")

    # Coverage summary per role over the runs actually joined.
    role_names = [str(r) for r in cache["role_names"]]
    role_idx = cache["role_idx"]
    print("coverage per role (joined runs):")
    for role_i, role in enumerate(role_names):
        n_bins_role = n_empty_role = n_runs_role = 0
        for rid in targets:
            sl = slices[rid]
            if int(role_idx[sl.start]) != role_i:
                continue
            n_runs_role += 1
            n_bins_role += sl.stop - sl.start
            n_empty_role += int(np.sum(~np.isfinite(power_out[sl])))
        if n_runs_role == 0:
            continue
        print(f"  {role}: runs={n_runs_role} bins={n_bins_role} "
              f"empty={n_empty_role} ({100.0 * n_empty_role / n_bins_role:.2f}%)")

    if args.run_id is not None:
        print("smoke mode: no outputs written")
        return

    out_arrays = {key: cache[key] for key in cache.files}
    out_arrays["power"] = power_out
    out_arrays["power_valid"] = np.isfinite(power_out)
    np.savez_compressed(args.out, **out_arrays)
    write_json(args.provenance_out, {
        "schema_version": SCHEMA_VERSION,
        "constants": {
            "fold_period_s": {"value": FOLD_PERIOD_S, "provenance": "plan-fixed"},
            "fold_gate_s": {"value": [FOLD_GATE_MIN_S, FOLD_GATE_MAX_S],
                            "provenance": "plan-fixed"},
            "gpus_per_node": {"value": GPUS_PER_NODE, "provenance": "plan-fixed"},
            "dt_s": {"value": dt, "provenance": "plan-fixed (sim cache grid)"},
        },
        "runs": per_run_provenance,
    })
    print(f"Wrote {args.out} and {args.provenance_out}")


if __name__ == "__main__":
    main()
