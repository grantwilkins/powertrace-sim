"""Data loader for the §2 profiling bundles, built for the *measured-state* path.

Why this exists (the diagnosis of the "bad results"):
    feature-test/build_ledger_bundle.py feeds every bundle through the
    ttft/itl *reconstruction* path -- the code path designed for ShareGPT-style
    validate runs. The new gemma data are closed-loop **staircase probes**: the
    engine is pinned at a known concurrency for ~100-700 s and vLLM /metrics
    records the *true* state (``num_requests_running``, token counters) the whole
    time. Reconstructing that state from client-side request timing is both
    unnecessary and biased -- in particular it never sees the real saturation
    knee (the engine sustains ~82 seqs when 128 are requested, ~105 when 256
    are), which is the single most informative point on the decode curve.

    This loader therefore reads each probe level as one clean steady-state
    ``(measured state -> measured power)`` sample, and reads the validate runs
    (the held-out realistic workload) as per-second bins via the SAME analytic
    work-rate physics. Every work rate is *computed from first principles*
    (arch descriptors + measured throughput/batch); none is fit or memorized.

Outputs a list of "points", each a dict of scalar work rates + power + metadata.
``physics.py`` turns these into model features; nothing here is model-specific.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

KV_ELEM_BYTES = 2.0  # bf16 KV cache

# ---- robust clock alignment ------------------------------------------------
# power.csv carries naive local wall-clock strings; engine.csv and the manifest
# carry true unix epoch. We recover the (whole-half-hour) offset by snapping the
# first power sample onto the manifest window start -- the same %1800 fold the
# known-good reconstruction builder uses, applied once per run.


def _parse_power_csv(path):
    """Return (naive_epoch[N], power_per_gpu[N]) ; naive_epoch treats the local
    string as if UTC (offset corrected later by the fold)."""
    ts, pw = [], []
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            try:
                t = dt.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                t = t.replace(tzinfo=dt.timezone.utc).timestamp()
                p = float(row[1])
            except (ValueError, IndexError):
                continue
            ts.append(t)
            pw.append(p)
    return np.asarray(ts), np.asarray(pw)


def _power_aligned(path, window_start_epoch):
    """power.csv timestamps folded onto the engine/manifest epoch."""
    ts, pw = _parse_power_csv(path)
    if ts.size == 0:
        return ts, pw
    off = round((window_start_epoch - ts[0]) / 1800.0) * 1800.0
    return ts + off, pw


def _read_engine(path):
    cols = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for k in r.fieldnames:
            cols[k] = []
        for row in r:
            for k in r.fieldnames:
                try:
                    cols[k].append(float(row[k]))
                except (ValueError, TypeError):
                    cols[k].append(np.nan)
    return {k: np.asarray(v) for k, v in cols.items()}


def _node_power_in_window(p_ts, p_w, tp, t_lo, t_hi):
    """Mean per-GPU power x tp = node power over active GPUs, in [t_lo, t_hi]."""
    m = (p_ts >= t_lo) & (p_ts <= t_hi) & np.isfinite(p_w)
    if m.sum() < 3:
        return np.nan
    return float(np.mean(p_w[m]) * tp)


def _rate_from_counter(e_ts, counter, t_lo, t_hi):
    """d(cumulative counter)/dt across a window (tokens/s)."""
    m = (e_ts >= t_lo) & (e_ts <= t_hi) & np.isfinite(counter)
    if m.sum() < 2:
        return np.nan
    t = e_ts[m]
    c = counter[m]
    dt_ = t[-1] - t[0]
    return float((c[-1] - c[0]) / dt_) if dt_ > 0 else np.nan


def _mean_in_window(e_ts, vals, t_lo, t_hi):
    m = (e_ts >= t_lo) & (e_ts <= t_hi) & np.isfinite(vals)
    return float(np.mean(vals[m])) if m.sum() else np.nan


def idle_power(run_dir, run_thresh=0.5, min_samp=10):
    """True idle node power: mean power while num_requests_running ~ 0.

    The single most important transferability anchor -- it pins the standing
    (idle) term, which neither staircase identifies (their lowest point, N=1,
    is already busy). Returns node W or NaN.
    """
    run_dir = Path(run_dir)
    man = json.loads((run_dir / "manifest.json").read_text())
    tp = int(man["tp"])
    win = man["probe"]["window"]
    p_ts, p_w = _power_aligned(run_dir / "power.csv", win["start_epoch"])
    eng = _read_engine(run_dir / "engine.csv")
    run_at_p = np.interp(p_ts, eng["timestamp"], eng["num_requests_running"])
    idle = (run_at_p < run_thresh) & np.isfinite(p_w)
    if idle.sum() < min_samp:
        return np.nan, man
    return float(np.mean(p_w[idle]) * tp), man


def idle_point(run_dir):
    """A zero-work fit point anchoring standing power for this hardware."""
    p, man = idle_power(run_dir)
    if not np.isfinite(p):
        return None
    arch = man["arch"]
    zero = work_rates(arch, int(man["tp"]), batch=0.0, iters_s=0.0,
                      dec_tok_s=0.0, pre_tok_s=0.0, ctx_decode=0.0, ctx_prefill=0.0)
    zero["busy"] = 0.0
    return dict(power=p, tp=float(man["tp"]), hw=man["hardware"], model=man["model"],
                family=arch.get("family", "unknown"), probe="idle", level=None,
                concurrency=0, source="idle", run_id=Path(run_dir).name + "_idle",
                weight=1.0, arch=_arch_scalars(arch), **zero)


# ---- analytic work rates (first principles, shared by all model variants) ---


TOK_PER_PASS = 8192.0  # max_num_batched_tokens (recorded in every manifest)


def work_rates(arch, tp, batch, iters_s, dec_tok_s, pre_tok_s,
               ctx_decode, ctx_prefill):
    """Compute per-second physical work rates from MEASURED state + arch.

    Key signal: ``iters_s`` = measured engine forward passes/s
    (d(iteration_tokens_total_count)/dt). Every forward pass streams the weights
    once, so weight traffic = w_eff * iters_s -- fully measured, no fragile
    division by batch. All quantities are physical and transferable; the only
    fit happens later.

    batch       : mean concurrent sequences (measured num_requests_running)
    iters_s     : measured engine steps/s (forward passes)
    dec_tok_s   : decode tokens/s (measured generation throughput)
    pre_tok_s   : prefill tokens/s (measured prompt throughput)
    ctx_decode  : mean KV context length resident during decode (tokens)
    ctx_prefill : mean prompt length being prefilled (tokens)
    """
    n_layers = arch["n_layers"]
    n_kv = arch["n_kv"]
    head_dim = arch["head_dim"]
    d_model = arch["d_model"]
    w_bytes = arch["w_bytes"]
    moe_frac = arch["moe_frac"]
    swa = float(arch.get("swa_window", 0) or 0)
    swa_ratio = float(arch.get("swa_global_ratio", 0) or 0)  # 1 global per N layers

    kv_tok = 2.0 * n_layers * n_kv * head_dim * KV_ELEM_BYTES  # bytes/token resident

    # split measured iters into prefill passes (chunked, ~pre_tok/chunk) and
    # the remainder (decode steps). Bounded by the measured total -> no blowup.
    pre_iters_s = min(pre_tok_s / TOK_PER_PASS, iters_s) if pre_tok_s > 0 else 0.0
    dec_iters_s = max(iters_s - pre_iters_s, 0.0)

    # weight bytes read per forward pass: full for dense; for MoE decode only the
    # experts the batch touches (+ always-on shared/attn weights); prefill passes
    # carry thousands of tokens, so all experts are read.
    if moe_frac > 0:
        touched = min(1.0, max(batch, 1.0) * arch["top_k"] / arch["n_experts"])
        w_eff_dec = w_bytes * ((1 - moe_frac) + moe_frac * touched)
    else:
        w_eff_dec = w_bytes
    w_read_dec = w_eff_dec * dec_iters_s
    w_read_pre = w_bytes * pre_iters_s

    # effective KV context with sliding-window attention: a fraction of layers
    # are global (full ctx), the rest windowed (min(ctx, swa)).
    def ctx_eff(ctx):
        if swa <= 0 or ctx <= 0:
            return ctx
        g = 1.0 / swa_ratio if (swa_ratio and swa_ratio > 1) else 0.5
        return g * ctx + (1 - g) * min(ctx, swa)

    kv_read = dec_tok_s * ctx_eff(ctx_decode) * kv_tok    # decode reads resident KV
    kv_write = (dec_tok_s + pre_tok_s) * kv_tok           # every new token writes KV

    flops_pre = 2.0 * arch["n_active"] * pre_tok_s
    flops_dec = 2.0 * arch["n_active"] * dec_tok_s

    # TP all-reduce: 2 reduces/layer, each moves 2*(tp-1)/tp of the d_model
    # activation (2 bytes) per token across NVLink.
    tok_rate = dec_tok_s + pre_tok_s
    comm = tok_rate * n_layers * 2.0 * 2.0 * d_model * 2.0 * (tp - 1.0) / max(tp, 1)

    return dict(
        batch=float(batch), iters=float(iters_s),
        pre_tok=float(pre_tok_s), dec_tok=float(dec_tok_s),
        flops_pre=float(flops_pre), flops_dec=float(flops_dec),
        w_read_pre=float(w_read_pre), w_read_dec=float(w_read_dec),
        w_read=float(w_read_pre + w_read_dec),
        kv_read=float(kv_read), kv_write=float(kv_write), comm=float(comm),
        busy=1.0,
    )


def _arch_scalars(arch):
    return {k: float(v) for k, v in arch.items()
            if k not in ("family",) and isinstance(v, (int, float))}


# ---- bundle readers --------------------------------------------------------


def load_staircase(run_dir, steady_frac=0.45, min_tail_s=20.0):
    """One steady-state point per probe level (measured state -> power)."""
    run_dir = Path(run_dir)
    man = json.loads((run_dir / "manifest.json").read_text())
    probe = man["probe"]
    ptype = probe["type"]
    arch = man["arch"]
    tp = int(man["tp"])
    win = probe["window"]
    p_ts, p_w = _power_aligned(run_dir / "power.csv", win["start_epoch"])
    eng = _read_engine(run_dir / "engine.csv")
    e_ts = eng["timestamp"]

    pts = []
    for L in probe["levels"]:
        s, e = L["t_start_epoch"], L["t_end_epoch"]
        dur = e - s
        # steady window: drop the leading ramp/KV-fill; keep a clean tail.
        lo = min(s + steady_frac * dur, e - min_tail_s)
        lo = max(lo, s)
        power = _node_power_in_window(p_ts, p_w, tp, lo, e)
        batch = _mean_in_window(e_ts, eng["num_requests_running"], lo, e)
        # token + iteration rates measured over the steady window (counter diff).
        dec_s = _rate_from_counter(e_ts, eng["generation_tokens_total"], lo, e)
        pre_s = _rate_from_counter(e_ts, eng["prompt_tokens_total"], lo, e)
        iters_s = _rate_from_counter(e_ts, eng["iteration_tokens_total_count"], lo, e)
        params = L.get("params", {})
        in_len = float(params.get("input_len", 0))
        out_len = float(params.get("output_len", 0))

        if ptype == "decode_staircase":
            pre_s = 0.0
            # mean KV context over the level: outputs sweep [in, in+out] roughly
            # uniformly under ignore_eos, so the time-average context ~ in+out/2.
            ctx_dec = in_len + 0.5 * out_len
            ctx_pre = 0.0
            if not np.isfinite(batch) or batch < 0.5:
                continue
        elif ptype == "prefill_staircase":
            ctx_pre = in_len
            ctx_dec = in_len  # the single emitted token reads the full prompt KV
            if not np.isfinite(pre_s) or pre_s <= 0:
                continue
        else:
            continue

        if not np.isfinite(power) or not np.isfinite(iters_s):
            continue
        wr = work_rates(arch, tp, batch if np.isfinite(batch) else 1.0,
                        iters_s,
                        dec_s if np.isfinite(dec_s) else 0.0,
                        pre_s if np.isfinite(pre_s) else 0.0,
                        ctx_dec, ctx_pre)
        pts.append(dict(
            power=power, tp=float(tp), hw=man["hardware"], model=man["model"],
            family=arch.get("family", "unknown"),
            probe=ptype, level=L.get("level"), concurrency=L.get("concurrency"),
            source="probe", run_id=run_dir.name, weight=1.0,
            arch=_arch_scalars(arch), **wr,
        ))
    return pts


def load_validate_bins(run_dir, dt_bin=1.0, lambda_prefill=5000.0, trim_s=5.0,
                       steady_only=False):
    """Per-second measured-state bins for a validate (ShareGPT-like) run.

    State is taken from engine.csv (measured num_requests_running + token
    counters interpolated onto bin edges), NOT reconstructed from ttft/itl.
    This is the honest grading workload (held out from fitting).
    """
    run_dir = Path(run_dir)
    man = json.loads((run_dir / "manifest.json").read_text())
    arch = man["arch"]
    tp = int(man["tp"])
    win = man["probe"]["window"]
    p_ts, p_w = _power_aligned(run_dir / "power.csv", win["start_epoch"])
    eng = _read_engine(run_dir / "engine.csv")
    e_ts = eng["timestamp"]
    if e_ts.size < 10 or p_ts.size < 10:
        return []

    t0 = max(win["start_epoch"], e_ts[0]) + trim_s
    t1 = min(win["end_epoch"], e_ts[-1], p_ts[-1])
    if t1 - t0 < 10 * dt_bin:
        return []
    edges = np.arange(t0, t1, dt_bin)
    if edges.size < 11:
        return []
    nb = edges.size - 1
    mids = edges[:-1] + dt_bin / 2.0

    # measured power per bin
    idx = np.searchsorted(edges, p_ts) - 1
    ok = (idx >= 0) & (idx < nb) & np.isfinite(p_w)
    psum = np.bincount(idx[ok], weights=p_w[ok], minlength=nb)
    pcnt = np.bincount(idx[ok], minlength=nb)
    power = np.where(pcnt > 0, psum / np.maximum(pcnt, 1), np.nan) * tp

    # measured state: interpolate cumulative counters onto edges, diff -> rate
    def rate(counter):
        c = np.interp(edges, e_ts, counter)
        return np.diff(c) / dt_bin

    dec_rate = rate(eng["generation_tokens_total"])
    pre_rate = rate(eng["prompt_tokens_total"])
    iters_rate = rate(eng["iteration_tokens_total_count"])
    batch = np.interp(mids, e_ts, eng["num_requests_running"])

    # mean context per bin from request log (input + decoded-so-far): use the
    # request json to estimate average resident context at each bin.
    ctx_dec = _avg_context_track(run_dir, edges, mids, dt_bin)

    pts = []
    for b in range(nb):
        if not np.isfinite(power[b]):
            continue
        bt = batch[b]
        dr = max(dec_rate[b], 0.0)
        pr = max(pre_rate[b], 0.0)
        it = max(iters_rate[b], 0.0)
        if steady_only and (dr + pr) <= 0:
            continue
        wr = work_rates(arch, tp, max(bt, 1e-6), it, dr, pr,
                        ctx_dec[b] if ctx_dec is not None else 0.0,
                        # prefill context: approximate by prompt-token burst size
                        pr * dt_bin if pr > 0 else 0.0)
        wr["busy"] = 1.0 if (dr + pr) > 0 else 0.0
        pts.append(dict(
            power=float(power[b]), tp=float(tp), hw=man["hardware"],
            model=man["model"], family=arch.get("family", "unknown"),
            probe="validate", level=None, concurrency=None,
            source="validate", run_id=run_dir.name, weight=1.0,
            arch=_arch_scalars(arch), **wr,
        ))
    return pts


def _avg_context_track(run_dir, edges, mids, dt_bin):
    """Time-resolved mean resident decode context (tokens) from requests.json.

    Approximates each request as: arrives at request_timestamp, prefills its
    input, then decodes output_len tokens at a steady per-request rate over its
    ITL span. At each bin we average (input + decoded-so-far) over live requests.
    Falls back to None if timing is unavailable.
    """
    try:
        req = json.loads((run_dir / "requests.json").read_text())
        arr = np.asarray(req["request_timestamps"], dtype=float)
        ttft = np.asarray(req["ttfts"], dtype=float)
        n_in = np.asarray(req["input_lens"], dtype=float)
        n_out = np.asarray(req["output_lens"], dtype=float)
        itls = req.get("itls")
    except (KeyError, FileNotFoundError, ValueError):
        return None
    if arr.size == 0:
        return None
    # align request epoch to power/engine epoch via the same %1800 fold
    off = round((edges[0] - float(np.min(arr))) / 1800.0) * 1800.0
    arr = arr + off
    dec_dur = np.array([sum(x) if isinstance(x, list) else 0.0 for x in itls]) \
        if itls is not None else n_out * 0.02
    dec_dur = np.maximum(dec_dur, 1e-3)
    dec_start = arr + ttft
    dec_end = dec_start + dec_dur

    ctx = np.zeros(mids.size)
    for b, m in enumerate(mids):
        live = (dec_start <= m) & (dec_end >= m)
        if not live.any():
            continue
        prog = np.clip((m - dec_start[live]) / dec_dur[live], 0.0, 1.0)
        ctx[b] = float(np.mean(n_in[live] + prog * n_out[live]))
    return ctx
