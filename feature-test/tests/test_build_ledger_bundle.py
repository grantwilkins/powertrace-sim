"""Equivalence gate for the new bundle ledger builder (CAMPAIGN.md §6).

Phase-1 (this file, no GPU): the new reconstruction path must reproduce the
known-good ``build_ledger_cache.build_run_bins`` arrays bit-for-bit on old-format
data, guaranteeing the new pipeline does not regress training-data quality.

Phase-2 wiring: a synthesized full bundle (with engine.csv) is exercised through
the measured-state path so its code path is covered offline; the empirical
measured-vs-reconstructed comparison runs post-launch on real data.
"""

import json

import numpy as np
import pytest

import build_ledger_bundle as blb
from build_ledger_cache import ARCH, build_run_bins

LAMBDA_PREFILL = 5000.0
DT = 1.0
EPOCH = 1_780_000_000.0  # fixed reference epoch (deterministic; no Date.now)


def _ts_text(epoch_s):
    import datetime as _dt
    return _dt.datetime.utcfromtimestamp(epoch_s).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]


def _write_old_format_bundle(tmp_path, *, n_req=40, gpus=8, tp=8, watts=300.0,
                             dur_s=80.0, seed=0):
    """Deterministic old-format (power.csv, requests.json) pair."""
    rng = np.random.default_rng(seed)
    # power.csv: per-GPU rows at 4 Hz
    header = (
        "timestamp, power.draw [W], utilization.gpu [%], memory.used [MiB]"
    )
    lines = [header]
    n_samples = int(dur_s * 4)
    for s in range(n_samples):
        ts = _ts_text(EPOCH + s * 0.25)
        for g in range(gpus):
            w = watts + rng.normal(0, 5)
            lines.append(f"{ts}, {w:.2f}, 95, 70000")
    (tmp_path / "power.csv").write_text("\n".join(lines) + "\n")

    # requests.json: arrivals spread 5..(dur-15)s after power start
    arrivals = np.sort(rng.uniform(5.0, dur_s - 15.0, size=n_req))
    input_lens, output_lens, ttfts, itls, req_ts = [], [], [], [], []
    for a in arrivals:
        n_in = int(rng.integers(256, 1024))
        n_out = int(rng.integers(64, 256))
        ttft = float(rng.uniform(0.2, 0.5))
        itl = float(rng.uniform(0.015, 0.025))
        input_lens.append(n_in)
        output_lens.append(n_out)
        ttfts.append(ttft)
        itls.append([itl] * n_out)          # decode_time = sum(itls)
        req_ts.append(EPOCH + float(a))
    data = dict(
        input_lens=input_lens, output_lens=output_lens, ttfts=ttfts,
        itls=itls, request_timestamps=req_ts,
    )
    (tmp_path / "requests.json").write_text(json.dumps(data))
    return str(tmp_path / "requests.json"), str(tmp_path / "power.csv")


def test_phase1_matches_old_builder(tmp_path):
    jp, cp = _write_old_format_bundle(tmp_path)
    model = "llama-3-70b"
    tp = 8

    ref = build_run_bins(jp, cp, model, tp, LAMBDA_PREFILL, dt=DT)
    new = blb.state_from_requests(jp, cp, ARCH[model], tp, LAMBDA_PREFILL, dt=DT)

    assert ref is not None and new is not None
    assert ref["n"] == new["n"]
    for key in blb.BIN_KEYS:
        np.testing.assert_allclose(
            new[key], ref[key], rtol=1e-9, atol=1e-6,
            err_msg=f"reconstruction diverged from build_run_bins for '{key}'",
        )


def test_output_schema_identical(tmp_path):
    """New builder's per-bin keys match the known-good builder's output."""
    jp, cp = _write_old_format_bundle(tmp_path)
    ref = build_run_bins(jp, cp, "llama-3-70b", 8, LAMBDA_PREFILL, dt=DT)
    new = blb.state_from_requests(jp, cp, ARCH["llama-3-70b"], 8, LAMBDA_PREFILL, dt=DT)
    ref_keys = {k for k in ref if k not in ("n", "arch")}
    new_keys = {k for k in new if k not in ("n", "arch")}
    assert ref_keys == new_keys == set(blb.BIN_KEYS)


def test_linear_attention_inert_when_zero(tmp_path):
    """The linear-attention branch must not perturb softmax models."""
    jp, cp = _write_old_format_bundle(tmp_path)
    arch0 = dict(ARCH["llama-3-70b"])           # no n_linear_layers
    arch_explicit0 = dict(arch0, n_linear_layers=0)
    a = blb.state_from_requests(jp, cp, arch0, 8, LAMBDA_PREFILL, dt=DT)
    b = blb.state_from_requests(jp, cp, arch_explicit0, 8, LAMBDA_PREFILL, dt=DT)
    np.testing.assert_array_equal(a["kv_read"], b["kv_read"])


def test_linear_attention_reduces_kv_read(tmp_path):
    """With linear layers, growing-KV read drops vs the all-softmax baseline."""
    jp, cp = _write_old_format_bundle(tmp_path)
    base = dict(ARCH["llama-3-70b"])
    hybrid = dict(base, n_linear_layers=base["n_layers"] // 2)
    soft = blb.state_from_requests(jp, cp, base, 8, LAMBDA_PREFILL, dt=DT)
    hyb = blb.state_from_requests(jp, cp, hybrid, 8, LAMBDA_PREFILL, dt=DT)
    assert hyb["kv_read"].sum() < soft["kv_read"].sum()


# --------------------------------------------------------------------------- #
# build_bundle goes through the proven reconstruction path; the measured
# (engine.csv) consumer is deferred to Phase-2.
# --------------------------------------------------------------------------- #

def test_build_bundle_uses_reconstruction(tmp_path):
    jp, cp = _write_old_format_bundle(tmp_path)
    arch = dict(ARCH["llama-3-70b"], family="dense-70b")
    manifest = dict(
        manifest_version=1, run_id="synthetic", model="meta-llama/Llama-3.1-70B-Instruct",
        hardware="H100", tp=8, gpus_per_node=8, arch=arch,
        probe={"type": "decode_staircase", "level": 16, "params": {"rate": 0.0}},
        server={}, versions={}, clock={},
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    bins, m = blb.build_bundle(tmp_path, lambda_prefill=LAMBDA_PREFILL, dt=DT)
    assert bins is not None and bins["n"] > 10
    assert np.nanmedian(bins["dec_tok"]) > 0.0
    # reconstruction fills the prefill/KV columns (engine path zeroed them)
    assert bins["w_read_pre"].sum() > 0.0
    assert bins["kv_read"].sum() > 0.0
    assert m["model"].endswith("Llama-3.1-70B-Instruct")


def test_engine_path_deferred_to_phase2():
    """The measured-state consumer is intentionally not implemented yet."""
    with pytest.raises(NotImplementedError):
        blb.bins_from_engine_csv()
