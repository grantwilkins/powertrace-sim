"""Build the calibration ledger from run bundles via the MEASURED-state path.

This is the implementation the ``feature-test/build_ledger_bundle.bins_from_engine_csv``
stub specified but never wrote. We consume ``engine.csv`` (vLLM /metrics) as the
source of truth for engine state instead of reconstructing it from client-side
TTFT/ITL timing — which is why feature-test mispredicts the closed-loop gemma
staircases (it discards the measured saturation knee and ``dec_tok/batch`` blows
up). The four correctness points from that stub are honored here:

1. Per-bin rates = diff of the cumulative counter INTERPOLATED onto bin edges
   (``np.diff(np.interp(edges, t-t0, counter)) / dt``), not (last-first) inside
   a bin (which undercounts the cross-bin increment).
2. Clock alignment: ``engine.csv`` stamps true epoch ``time.time()`` (matching the
   manifest probe window); ``power.csv`` is nvidia-smi local time coerced to UTC.
   We fold the power stream onto the engine epoch with a whole-half-hour fold.
3. KV read is ANALYTIC (``gpu_cache_usage_perc`` is all-NaN in this campaign);
   prefill iterations come from the manifest level geometry.
4. Idle bins (``num_requests_running ~ 0``) are kept in the ledger so the standing
   power terms (p_idle, p_link) are anchored directly; an explicit idle scalar is
   also emitted for diagnostics.

The ledger stores the raw per-bin STATE (token rates, occupancy, prefill context
length, decode context). The physics work-rates are derived at fit time via
``arch.work_rates`` so ingest and the predict-time chain share one physics.

Run (on a compute node, not the login node):
    python -m powermodel.ingest --runs-glob "$SCRATCH/ptsim/runs/*" --out ledger.npz
"""

from __future__ import annotations

import argparse
import csv
import glob as _glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model.training_data.power_parsing import parse_power_csv  # noqa: E402
from powermodel.arch import normalize_arch  # noqa: E402

# Per-bin state arrays the ledger stores (physics derived later from these).
STATE_KEYS = (
    "power", "pre_tok", "dec_tok", "decode_batch", "pre_active",
    "L_pre", "ctx_dec", "iters_pre", "iters_dec", "busy",
)

ENGINE_COLS = (
    "timestamp", "num_requests_running", "num_requests_waiting",
    "gpu_cache_usage_perc", "prompt_tokens_total", "generation_tokens_total",
    "iteration_tokens_total_sum", "iteration_tokens_total_count",
    "request_prefill_time_seconds_sum", "request_decode_time_seconds_sum",
)
IDLE_RUNNING = 0.5  # num_requests_running below this => idle bin
WINDOW_MARGIN_S = 20.0  # idle margin kept around the probe-activity window


def parse_engine_csv(path: str) -> dict:
    """Load engine.csv into float arrays (NaNs preserved)."""
    cols = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in ENGINE_COLS:
                try:
                    cols[k].append(float(row[k]))
                except (KeyError, ValueError, TypeError):
                    cols[k].append(np.nan)
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def _align_power_to_engine(p_ts, eng_t0):
    """Fold the power timestamps onto the engine epoch (cancel whole-half-hour
    skews from the local->UTC coercion). Returns power timestamps relative to
    ``eng_t0``."""
    p_rel = p_ts - eng_t0
    # Bring the power stream into the engine window by removing half-hour skews.
    p_rel = p_rel - round(float(np.median(p_rel)) / 1800.0) * 1800.0
    return p_rel


def _rate_on_edges(t_rel, counter, edges, dt):
    """Per-bin rate of a cumulative counter via interpolation onto bin edges."""
    finite = np.isfinite(counter)
    if finite.sum() < 2:
        return np.zeros(edges.size - 1)
    c = np.interp(edges, t_rel[finite], counter[finite],
                  left=counter[finite][0], right=counter[finite][-1])
    return np.maximum(np.diff(c) / dt, 0.0)


def _gauge_on_bins(t_rel, gauge, edges, dt):
    """Bin-mean of a gauge signal (e.g. num_requests_running)."""
    nb = edges.size - 1
    idx = np.searchsorted(edges, t_rel) - 1
    ok = (idx >= 0) & (idx < nb) & np.isfinite(gauge)
    s = np.bincount(idx[ok], weights=gauge[ok], minlength=nb)
    n = np.bincount(idx[ok], minlength=nb)
    return np.where(n > 0, s / np.maximum(n, 1), 0.0)


def _level_context(manifest):
    """Build a function bin_mid -> (probe_type, input_len, output_len, conc).

    For staircase probes each manifest level carries (t_start_epoch, t_end_epoch,
    params.input_len/output_len, concurrency). Returns level lookup arrays.
    """
    probe = manifest.get("probe", {})
    ptype = probe.get("type", "unknown")
    levels = probe.get("levels", [])
    starts, ends, inlen, outlen, conc = [], [], [], [], []
    for lv in levels:
        starts.append(float(lv.get("t_start_epoch", np.nan)))
        ends.append(float(lv.get("t_end_epoch", np.nan)))
        p = lv.get("params", {})
        inlen.append(float(p.get("input_len", 0)))
        outlen.append(float(p.get("output_len", 0)))
        conc.append(float(lv.get("concurrency", 0)))
    return ptype, (np.asarray(starts), np.asarray(ends), np.asarray(inlen),
                   np.asarray(outlen), np.asarray(conc))


def bins_from_engine(run_dir: Path, dt: float = 1.0, trim_s: float = 3.0) -> dict | None:
    """Per-bin measured state for one bundle, or None if unusable."""
    run_dir = Path(run_dir)
    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        return None
    manifest = json.loads(mpath.read_text())
    arch = normalize_arch(manifest["arch"])
    tp = float(manifest["tp"])
    hw = manifest["hardware"]
    gpn = int(manifest.get("gpus_per_node", 8))

    eng = parse_engine_csv(str(run_dir / "engine.csv"))
    if eng["timestamp"].size < 10:
        return None
    pw = parse_power_csv(str(run_dir / "power.csv"), tensor_parallelism=int(tp),
                         gpus_per_node=gpn)
    if pw is None:
        return None

    eng_t0 = float(eng["timestamp"][0])
    eng_rel = eng["timestamp"] - eng_t0
    p_rel = _align_power_to_engine(pw["timestamps"], eng_t0)

    # Bin window = overlap of engine and (folded) power streams, trimmed.
    lo = max(float(eng_rel[0]), float(p_rel[0])) + trim_s
    hi = min(float(eng_rel[-1]), float(p_rel[-1])) - trim_s
    # Clamp to the probe-activity window (+ idle margin) when known: a run killed
    # at the time limit can sit idle for many minutes after its last level, which
    # would flood the ledger with idle bins and skew the standing/dynamic split.
    win = manifest.get("probe", {}).get("window", {})
    if win.get("start_epoch") and win.get("end_epoch"):
        ws = float(win["start_epoch"]) - eng_t0 - WINDOW_MARGIN_S
        we = float(win["end_epoch"]) - eng_t0 + WINDOW_MARGIN_S
        lo, hi = max(lo, ws), min(hi, we)
    if hi - lo < 10 * dt:
        return None
    edges = np.arange(lo, hi, dt)
    if edges.size < 11:
        return None
    nb = edges.size - 1
    mids = (edges[:-1] + edges[1:]) / 2.0

    # ---- measured power per bin (mean of samples in bin)
    pidx = np.searchsorted(edges, p_rel) - 1
    pok = (pidx >= 0) & (pidx < nb) & np.isfinite(pw["power"])
    psum = np.bincount(pidx[pok], weights=pw["power"][pok], minlength=nb)
    pcnt = np.bincount(pidx[pok], minlength=nb)
    power = np.where(pcnt > 0, psum / np.maximum(pcnt, 1), np.nan)

    # ---- measured token / iteration rates (point 1: diff of interp on edges)
    pre_tok = _rate_on_edges(eng_rel, eng["prompt_tokens_total"], edges, dt)
    dec_tok = _rate_on_edges(eng_rel, eng["generation_tokens_total"], edges, dt)
    iters = _rate_on_edges(eng_rel, eng["iteration_tokens_total_count"], edges, dt)
    running = _gauge_on_bins(eng_rel, eng["num_requests_running"], edges, dt)

    busy = ((pre_tok + dec_tok) > 0).astype(np.float64)

    # ---- phase split (decode batch vs prefill active occupancy)
    ptype, (l_start, l_end, l_in, l_out, l_conc) = _level_context(manifest)
    L_pre = np.zeros(nb)
    ctx_dec = np.zeros(nb)
    decode_batch = np.zeros(nb)
    pre_active = np.zeros(nb)
    iters_pre = np.zeros(nb)
    iters_dec = np.zeros(nb)

    abs_mids = mids + eng_t0  # back to absolute epoch for level lookup

    is_prefill_probe = "prefill" in ptype
    is_decode_probe = "decode" in ptype

    if l_start.size and (is_prefill_probe or is_decode_probe):
        # Pure staircase: each bin belongs to one level with known geometry.
        for k in range(l_start.size):
            sel = (abs_mids >= l_start[k]) & (abs_mids < l_end[k])
            if not sel.any():
                continue
            if is_prefill_probe:
                L_pre[sel] = l_in[k]
                pre_active[sel] = running[sel]
                iters_pre[sel] = iters[sel]
            else:  # decode staircase: pure decode, context grows toward output_len
                decode_batch[sel] = running[sel]
                iters_dec[sel] = iters[sel]
                # mean generated tokens/seq so far in this level (wrapped by out_len)
                gen0 = np.interp(l_start[k] - eng_t0, eng_rel,
                                 eng["generation_tokens_total"])
                gen = np.interp(abs_mids[sel] - eng_t0, eng_rel,
                                eng["generation_tokens_total"])
                per_seq = (gen - gen0) / max(l_conc[k], 1.0)
                ctx_dec[sel] = l_in[k] + np.minimum(per_seq % max(l_out[k], 1.0),
                                                    l_out[k])
    else:
        # Interleaved regime (mixed_grid probe, or validate/agentic serving):
        # prefill chunks share iterations with decode. Weights are read ONCE per
        # iteration serving the whole running batch, so attribute all weight reads
        # to the iteration count (single count). Double-counting prefill+decode
        # weight reads is what inflated held-out HBM power.
        with np.errstate(divide="ignore", invalid="ignore"):
            db = np.where(iters > 0, dec_tok / iters, 0.0)
        decode_batch = np.clip(db, 0.0, running + 1.0)
        pre_active = np.maximum(running - decode_batch, 0.0)
        iters_dec = iters.copy()          # weights read once per iteration
        iters_pre = np.zeros(nb)          # no extra weight reads for prefill chunks
        if l_start.size:
            # mixed_grid: per-level known geometry (input/output length).
            for k in range(l_start.size):
                sel = (abs_mids >= l_start[k]) & (abs_mids < l_end[k])
                if not sel.any():
                    continue
                L_pre[sel] = np.where(pre_tok[sel] > 0, l_in[k], 0.0)
                ctx_dec[sel] = np.where(decode_batch[sel] > 0,
                                        l_in[k] + l_out[k] / 2.0, 0.0)
        else:
            # validate/agentic: use the run's mean request input length.
            rj = run_dir / "requests.json"
            mean_in = 0.0
            if rj.exists():
                try:
                    data = json.loads(rj.read_text())
                    ins = np.asarray(data.get("input_lens", []), dtype=np.float64)
                    mean_in = float(np.mean(ins)) if ins.size else 0.0
                except Exception:
                    mean_in = 0.0
            L_pre = np.where(pre_tok > 0, max(mean_in, 1.0), 0.0)
            ctx_dec = np.where(decode_batch > 0, mean_in + 128.0, 0.0)

    keep = np.isfinite(power)
    if keep.sum() < 10:
        return None

    state = dict(
        power=power, pre_tok=pre_tok, dec_tok=dec_tok, decode_batch=decode_batch,
        pre_active=pre_active, L_pre=L_pre, ctx_dec=ctx_dec,
        iters_pre=iters_pre, iters_dec=iters_dec, busy=busy,
    )
    state = {k: v[keep] for k, v in state.items()}
    idle_mask = running[keep] < IDLE_RUNNING
    idle_power = float(np.mean(power[keep][idle_mask])) if idle_mask.any() else np.nan

    arch_scalar = {k: float(v) for k, v in arch.items()
                   if isinstance(v, (int, float))}
    return dict(
        state=state, n=int(keep.sum()), arch=arch_scalar, tp=tp, hw=hw,
        family=manifest["arch"].get("family", "unknown"),
        model=manifest.get("model", "unknown"), probe=ptype,
        run_id=run_dir.name, idle_power=idle_power,
    )


def build_ledger(run_dirs, out_path, dt=1.0):
    """Concatenate per-bundle measured state into one ledger .npz."""
    cols = defaultdict(list)
    arch_list, meta = [], []
    families, hws, probes, run_names = [], [], [], []
    n_ok = 0
    for rd in sorted(run_dirs):
        try:
            b = bins_from_engine(Path(rd), dt=dt)
        except Exception as e:  # pragma: no cover - defensive
            print(f"  skip {rd}: {e}")
            continue
        if b is None:
            print(f"  skip {Path(rd).name}: unusable")
            continue
        rid = n_ok
        n = b["n"]
        for k in STATE_KEYS:
            cols[k].append(b["state"][k])
        cols["run_id"].append(np.full(n, rid, dtype=np.int32))
        cols["tp"].append(np.full(n, b["tp"]))
        if b["family"] not in families:
            families.append(b["family"])
        if b["hw"] not in hws:
            hws.append(b["hw"])
        if b["probe"] not in probes:
            probes.append(b["probe"])
        cols["family_idx"].append(np.full(n, families.index(b["family"]), dtype=np.int32))
        cols["hw_idx"].append(np.full(n, hws.index(b["hw"]), dtype=np.int32))
        cols["probe_idx"].append(np.full(n, probes.index(b["probe"]), dtype=np.int32))
        arch_list.append(b["arch"])
        run_names.append(b["run_id"])
        meta.append(dict(run_id=rid, name=b["run_id"], family=b["family"],
                         hw=b["hw"], tp=b["tp"], probe=b["probe"],
                         model=b["model"], idle_power=b["idle_power"], n=n))
        print(f"  ok {b['run_id']}: {n} bins, idle~{b['idle_power']:.0f}W")
        n_ok += 1

    if not cols:
        print("No bundles parsed.")
        return
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["family_names"] = np.array(families)
    out["hw_names"] = np.array(hws)
    out["probe_names"] = np.array(probes)
    out["run_names"] = np.array(run_names)
    # store arch per run as a JSON blob (ragged); keys are stable across runs
    out["arch_json"] = np.array([json.dumps(a) for a in arch_list])
    out["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(out_path, **out)
    print(f"Parsed {n_ok} bundles -> {out['power'].size} bins -> {out_path}")


def load_ledger(path):
    d = dict(np.load(path, allow_pickle=False))
    d["arch"] = [json.loads(s) for s in d["arch_json"]]
    d["meta"] = json.loads(str(d["meta_json"]))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", default="data/runs/*")
    ap.add_argument("--out", default="powermodel/ledger.npz")
    ap.add_argument("--dt", type=float, default=1.0)
    args = ap.parse_args()
    run_dirs = [Path(p) for p in _glob.glob(args.runs_glob) if Path(p).is_dir()]
    build_ledger(run_dirs, args.out, dt=args.dt)


if __name__ == "__main__":
    main()
