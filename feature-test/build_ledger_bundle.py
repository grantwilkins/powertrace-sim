"""Build the per-bin work ledger from §2 self-describing run bundles.

This is the new (additive) ledger builder for the profiling campaign. It reads a
``data/runs/<run_id>/`` bundle (``power.csv`` + ``engine.csv`` + ``requests.json``
+ ``manifest.json``) and emits the SAME ``ledger_cache.npz`` schema that
``feature-test/build_ledger_cache.py`` produces, so ``fit_map_priors.py`` /
``peak_and_holdout.py`` / ``final_model.py`` consume it unchanged.

It does NOT modify ``build_ledger_cache.py`` (the known-good builder the
equivalence gate measures against). Two state paths share the bin-level work-rate
math:

* ``reconstruct_bins`` — the ttft/itl reconstruction path. This is a faithful
  copy of ``build_run_bins`` (so it reproduces the known-good ledger bit-for-bit
  on existing data — Phase-1 of the equivalence gate). The only additions are
  guarded branches that are inert when ``n_linear_layers == 0`` and when a
  manifest clock offset is supplied, so existing softmax runs are unchanged.
* ``bins_from_engine_csv`` — the measured-state path (vLLM ``/metrics``). Primary
  for new bundles; compared against reconstruction in Phase-2.

Run from repo root:
    uv run python feature-test/build_ledger_bundle.py --runs-glob 'data/runs/*'
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

from model.training_data.power_parsing import parse_power_csv, parse_request_json  # noqa: E402

GIB = 1024.0**3
KV_ELEM_BYTES = 2.0

# Per-bin output arrays (must match build_ledger_cache for schema parity).
BIN_KEYS = (
    "power", "pre_tok", "dec_tok", "batch", "pre_active", "iters",
    "w_read", "w_read_pre", "w_read_dec", "kv_read", "kv_write", "comm",
)


def reconstruct_bins(req, pw, arch, tp, lambda_prefill, dt=1.0, trim_s=5.0):
    """Reconstruct per-bin work rates from request timing + power.

    Faithful re-implementation of ``build_ledger_cache.build_run_bins`` (state
    reconstructed from ttft/itl). With ``arch['n_linear_layers'] in (0, absent)``
    this is bit-identical to the known-good builder.

    Alignment uses the legacy 30-minute fold (not a manifest clock offset): it
    cancels whole/half-hour skews — including the local-vs-UTC offset that
    ``power_timestamp_to_epoch`` introduces by coercing naive nvidia-smi
    timestamps to UTC — without needing a separately-measured offset.
    """
    if req is None or pw is None or not req["has_timestamps"]:
        return None

    p_ts, p_w = pw["timestamps"], pw["power"]
    t0 = float(p_ts[0])
    arr = req["request_timestamps"] - t0
    arr = arr - round(float(np.min(arr)) / 1800.0) * 1800.0
    if float(np.min(arr)) < -2.0 or float(np.min(arr)) > 600.0:
        return None
    ttft, dec = req["ttfts"], req["decode_times"]
    n_in, n_out = req["input_lens"], req["output_lens"]

    pre_e = arr + ttft
    pre_dur = np.minimum(np.maximum(n_in / max(lambda_prefill, 1e-3), 1e-3), ttft)
    pre_s = pre_e - pre_dur
    dec_s, dec_e = pre_e, pre_e + dec
    run_end = min(float(p_ts[-1] - t0), float(np.max(dec_e)))
    run_start = max(trim_s, float(np.min(arr)))
    if run_end - run_start < 10 * dt:
        return None
    edges = np.arange(run_start, run_end, dt)
    if edges.size < 11:
        return None
    nb = edges.size - 1

    p_rel = p_ts - t0
    idx = np.searchsorted(edges, p_rel) - 1
    ok = (idx >= 0) & (idx < nb) & np.isfinite(p_w)
    pow_sum = np.bincount(idx[ok], weights=p_w[ok], minlength=nb)
    pow_cnt = np.bincount(idx[ok], minlength=nb)
    valid = pow_cnt > 0
    power = np.where(valid, pow_sum / np.maximum(pow_cnt, 1), np.nan)

    pre_rate = n_in / np.maximum(pre_dur, 1e-3)
    dec_rate = n_out / np.maximum(dec, 1e-3)
    kv_tok = 2.0 * arch["n_layers"] * arch["n_kv"] * arch["head_dim"] * KV_ELEM_BYTES
    swa = float(arch["swa_window"])
    n_lin = int(arch.get("n_linear_layers", 0) or 0)

    pre_tok = np.zeros(nb)
    dec_tok = np.zeros(nb)
    batch = np.zeros(nb)
    pre_active = np.zeros(nb)
    pre_iter = np.zeros(nb)
    kv_read = np.zeros(nb)

    for j in range(arr.size):
        lo_bin = max(0, int((pre_s[j] - run_start) // dt))
        hi_bin = min(nb, int((dec_e[j] - run_start) // dt) + 1)
        if hi_bin <= lo_bin:
            continue
        b_lo = edges[lo_bin:hi_bin]
        b_hi = b_lo + dt
        ov_pre = np.clip(np.minimum(pre_e[j], b_hi) - np.maximum(pre_s[j], b_lo), 0.0, None)
        ov_dec = np.clip(np.minimum(dec_e[j], b_hi) - np.maximum(dec_s[j], b_lo), 0.0, None)
        sl = slice(lo_bin, hi_bin)
        pre_tok[sl] += pre_rate[j] * ov_pre / dt
        dec_tok[sl] += dec_rate[j] * ov_dec / dt
        batch[sl] += ov_dec / dt
        pre_active[sl] += ov_pre / dt
        pre_iter[sl] += ov_pre / max(float(pre_dur[j]), 1e-3) / dt
        mid = (b_lo + b_hi) / 2.0
        prog = np.clip((mid - dec_s[j]) / max(float(dec[j]), 1e-3), 0.0, 1.0)
        ctx = n_in[j] + prog * n_out[j]
        ctx_eff = ctx if swa <= 0 else 0.5 * ctx + 0.5 * np.minimum(ctx, swa)
        if n_lin <= 0:
            # Softmax KV (identical to build_run_bins).
            kv_read[sl] += dec_rate[j] * (ov_dec / dt) * ctx_eff * kv_tok
        else:
            # Hybrid: softmax layers keep growing KV; linear/lightning layers
            # carry a constant recurrent state (~head_dim), not ctx. Provisional
            # work rate; the linear-attention fit term is downstream follow-up.
            n_lay = max(int(arch["n_layers"]), 1)
            soft_frac = (n_lay - n_lin) / n_lay
            lin_frac = n_lin / n_lay
            ctx_soft = ctx_eff * soft_frac
            ctx_lin = float(arch["head_dim"]) * lin_frac
            kv_read[sl] += dec_rate[j] * (ov_dec / dt) * (ctx_soft + ctx_lin) * kv_tok

    out = _bin_work_rates(
        pre_tok, dec_tok, batch, pre_active, pre_iter, kv_read, arch, tp, nb
    )
    keep = valid & np.isfinite(power)
    n = int(keep.sum())
    if n == 0:
        return None
    out = {k: v[keep] for k, v in out.items()}
    out["power"] = power[keep]
    arch_scalar = {k: float(v) for k, v in arch.items() if k != "family"}
    return dict(out, n=n, arch=arch_scalar)


def _bin_work_rates(pre_tok, dec_tok, batch, pre_active, pre_iter, kv_read,
                    arch, tp, nb):
    """Bin-level work rates shared by both state paths (build_run_bins l.145-165)."""
    dec_iter = dec_tok / np.maximum(batch, 1e-9)
    iters = dec_iter + pre_iter
    if arch["moe_frac"] > 0:
        touched = np.minimum(1.0, batch * arch["top_k"] / arch["n_experts"])
        w_eff_dec = arch["w_bytes"] * ((1 - arch["moe_frac"]) + arch["moe_frac"] * touched)
    else:
        w_eff_dec = np.full(nb, arch["w_bytes"])
    w_read_dec = w_eff_dec * dec_iter
    w_read_pre = arch["w_bytes"] * pre_iter
    w_read = w_read_dec + w_read_pre
    kv_tok = 2.0 * arch["n_layers"] * arch["n_kv"] * arch["head_dim"] * KV_ELEM_BYTES
    kv_write = (pre_tok + dec_tok) * kv_tok
    tok_rate = pre_tok + dec_tok
    comm = (
        tok_rate * arch["n_layers"] * 2.0
        * 2.0 * arch["d_model"] * 2.0 * (tp - 1.0) / max(tp, 1)
    )
    return dict(
        pre_tok=pre_tok, dec_tok=dec_tok, batch=batch, pre_active=pre_active,
        iters=iters, w_read=w_read, w_read_pre=w_read_pre, w_read_dec=w_read_dec,
        kv_read=kv_read, kv_write=kv_write, comm=comm,
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
    above is the equivalence-proven source for now. A first draft of this function
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

def read_bundle(run_dir: Path) -> dict:
    """Load a §2 bundle directory into its components."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return {
        "manifest": manifest,
        "power_csv": str(run_dir / "power.csv"),
        "engine_csv": str(run_dir / "engine.csv"),
        "requests_json": str(run_dir / "requests.json"),
    }


def build_bundle(run_dir, lambda_prefill=5000.0, dt=1.0):
    """Build per-bin arrays for one bundle via the reconstruction path.

    engine.csv is collected by the scraper but not consumed here yet — measured-
    state consumption is Phase-2 (see ``bins_from_engine_csv``). Reconstruction is
    the equivalence-proven source.
    """
    b = read_bundle(run_dir)
    m = b["manifest"]
    bins = state_from_requests(b["requests_json"], b["power_csv"], m["arch"],
                               int(m["tp"]), lambda_prefill, dt=dt)
    return bins, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", default="data/runs/*")
    ap.add_argument("--out", default="feature-test/ledger_cache_bundle.npz")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--lambda-prefill", type=float, default=5000.0)
    args = ap.parse_args()

    run_dirs = sorted(p for p in Path().glob(args.runs_glob) if p.is_dir())
    cols = defaultdict(list)
    model_names, family_names, hw_names = [], [], ["A100", "H100"]
    n_ok = 0
    for i, rd in enumerate(run_dirs):
        try:
            bins, m = build_bundle(rd, lambda_prefill=args.lambda_prefill, dt=args.dt)
        except Exception as e:  # pragma: no cover - defensive
            print(f"  skip {rd}: {e}")
            continue
        if bins is None:
            continue
        n = bins["n"]
        a = bins["arch"]
        model = m["model"]
        family = m["arch"].get("family", "unknown")
        if model not in model_names:
            model_names.append(model)
        if family not in family_names:
            family_names.append(family)
        for key in BIN_KEYS:
            cols[key].append(bins[key])
        cols["n_active"].append(np.full(n, a["n_active"]))
        cols["w_bytes"].append(np.full(n, a["w_bytes"]))
        cols["fp8"].append(np.full(n, a.get("fp8", 0)))
        cols["tp"].append(np.full(n, float(m["tp"])))
        rate = float(m.get("probe", {}).get("params", {}).get("rate", 0.0))
        cols["rate"].append(np.full(n, rate))
        cols["run_id"].append(np.full(n, i, dtype=np.int32))
        cols["model_idx"].append(np.full(n, model_names.index(model), dtype=np.int32))
        hw = m["hardware"]
        cols["hw_idx"].append(np.full(n, 0 if hw == "A100" else 1, dtype=np.int32))
        cols["family_idx"].append(np.full(n, family_names.index(family), dtype=np.int32))
        n_ok += 1

    if not cols:
        print("No bundles parsed.")
        return
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["model_names"] = np.array(model_names)
    out["family_names"] = np.array(family_names)
    out["hw_names"] = np.array(hw_names)
    np.savez_compressed(args.out, **out)
    print(f"Parsed {n_ok}/{len(run_dirs)} bundles -> {out['power'].size} bins -> {args.out}")


if __name__ == "__main__":
    main()
