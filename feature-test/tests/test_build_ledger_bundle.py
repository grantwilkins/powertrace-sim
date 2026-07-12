"""
Claim:
The shared ledger view preserves work accounting, and default bundle discovery
selects only bundle roots while explicit selection remains fail-fast.

Plausible wrong implementations:
- Treat campaign support directories as malformed bundles during default scans.
- Filter explicitly requested malformed directories before ingestion can fail.
- Change a conservation equation while keeping output arrays well formed.
- Drop legacy fit columns when adding inspectable request-state columns.
"""

import json

import numpy as np
import pytest

import build_ledger_bundle as blb
from model.training_data.arch import ARCH
from model.training_data.ledger_view import bin_work_rates

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
        "timestamp, index, uuid, power.draw [W], utilization.gpu [%], memory.used [MiB]"
    )
    lines = [header]
    n_samples = int(dur_s * 4)
    for s in range(n_samples):
        ts = _ts_text(EPOCH + s * 0.25)
        for g in range(gpus):
            w = watts + rng.normal(0, 5)
            lines.append(f"{ts}, {g}, GPU-{g}, {w:.2f}, 95, 70000")
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
        itls.append([itl] * (n_out - 1))
        req_ts.append(EPOCH + float(a))
    data = dict(
        input_lens=input_lens, output_lens=output_lens, ttfts=ttfts,
        itls=itls, request_timestamps=req_ts,
    )
    (tmp_path / "requests.json").write_text(json.dumps(data))
    return str(tmp_path / "requests.json"), str(tmp_path / "power.csv")


def test_bin_work_rates_satisfy_hand_worked_physics():
    """One dense bin: decode iterations, bytes, KV writes, and TP comm conserve work."""
    arch = dict(ARCH["llama-3-70b"], w_bytes=100.0, n_layers=1, n_kv=1,
                head_dim=2, d_model=4, moe_frac=0.0)
    out = bin_work_rates(
        pre_tok=np.array([2.0]), dec_tok=np.array([6.0]),
        batch=np.array([2.0]), pre_active=np.array([1.0]),
        pre_iter=np.array([1.0]), kv_read=np.array([5.0]),
        arch=arch, tp=2, nb=1,
    )
    np.testing.assert_array_equal(out["iters"], [4.0])
    np.testing.assert_array_equal(out["w_read_pre"], [100.0])
    np.testing.assert_array_equal(out["w_read_dec"], [300.0])
    np.testing.assert_array_equal(out["w_read"], [400.0])
    np.testing.assert_array_equal(out["kv_write"], [64.0])
    np.testing.assert_array_equal(out["comm"], [128.0])


def test_moe_weight_traffic_uses_expected_unique_experts():
    """Two draws from four experts touch 4*(1-(3/4)^2)=1.75 in expectation."""
    arch = dict(ARCH["gpt-oss-20b"], w_bytes=100.0, moe_frac=1.0,
                n_experts=4, top_k=1, n_layers=1, n_kv=1, head_dim=1,
                d_model=1)
    out = bin_work_rates(
        pre_tok=np.array([0.0]), dec_tok=np.array([2.0]),
        batch=np.array([2.0]), pre_active=np.array([0.0]),
        pre_iter=np.array([0.0]), kv_read=np.array([0.0]),
        arch=arch, tp=1, nb=1,
    )
    np.testing.assert_allclose(out["w_read_dec"], [43.75])


def test_output_schema_identical(tmp_path):
    """New request state extends rather than replaces the legacy fit schema."""
    jp, cp = _write_old_format_bundle(tmp_path)
    new = blb.state_from_requests(jp, cp, ARCH["llama-3-70b"], 8, LAMBDA_PREFILL, dt=DT)
    new_keys = {k for k in new if k not in ("n", "arch")}
    state_keys = {
        "arrivals", "input_tokens_arriving", "output_tokens_requested",
        "A_t", "delta_A_t", "running_requests", "waiting_requests",
    }
    assert set(blb.BIN_KEYS) | state_keys == new_keys
    assert all(new[key].shape == (new["n"],) for key in new_keys)
    assert np.all(np.isfinite(new["power"]))
    assert np.all(new["power"] >= 0.0)


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
# build_bundle defaults to the proven reconstruction path. The measured engine
# projection has a separate explicit contract and equivalence gate.
# --------------------------------------------------------------------------- #

def test_build_bundle_uses_reconstruction(tmp_path):
    jp, cp = _write_old_format_bundle(tmp_path)
    arch = dict(ARCH["llama-3-70b"], family="dense-70b")
    manifest = dict(
        manifest_version=1, run_id="synthetic", model="meta-llama/Llama-3.1-70B-Instruct",
        hardware="H100", tp=8, gpus_per_node=8, arch=arch,
        probe={"type": "decode_staircase",
               "window": {"start_epoch": EPOCH, "end_epoch": EPOCH + 80.0},
               "levels": []},
        server={}, versions={}, clock={"local_utc_offset_s": 0.0},
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "engine.csv").write_text(
        "timestamp,num_requests_running\n1704103200.0,1\n"
    )
    bins, m = blb.build_bundle(
        tmp_path, lambda_prefill=LAMBDA_PREFILL,
        lambda_prefill_source="unit measured prefill probe", dt=DT,
    )
    assert bins is not None and bins["n"] > 10
    assert np.nanmedian(bins["dec_tok"]) > 0.0
    # reconstruction fills the prefill/KV columns (engine path zeroed them)
    assert bins["w_read_pre"].sum() > 0.0
    assert bins["kv_read"].sum() > 0.0
    assert m["model"].endswith("Llama-3.1-70B-Instruct")
    assert m["_ledger_source"]["lambda_prefill_source"] == "unit measured prefill probe"
    assert set(m["_ledger_source"]["sha256"]) == {
        "manifest.json", "power.csv", "engine.csv", "requests.json",
    }


def test_bundle_measured_engine_state_reaches_maintained_ledger(tmp_path):
    import csv
    import sys
    from pathlib import Path

    _write_old_format_bundle(tmp_path)
    client = Path(__file__).resolve().parents[2] / "profiling" / "client"
    sys.path.insert(0, str(client))
    from metrics_logger import ENGINE_HEADER

    with (tmp_path / "engine.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ENGINE_HEADER)
        writer.writeheader()
        for sample in range(321):
            elapsed = sample * 0.25
            row = {name: 0.0 for name in ENGINE_HEADER}
            row.update({
                "timestamp": EPOCH + elapsed,
                "num_requests_running": 3.0,
                "gpu_cache_usage_perc": 0.25,
                "prompt_tokens_total": 4.0 * elapsed,
                "generation_tokens_total": 8.0 * elapsed,
                "iteration_tokens_total_sum": 12.0 * elapsed,
                "iteration_tokens_total_count": elapsed,
            })
            writer.writerow(row)

    arch = dict(ARCH["llama-3-70b"], family="dense-70b")
    manifest = {
        "manifest_version": 3,
        "run_id": "measured-synthetic",
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "hardware": "H100", "tp": 8, "gpus_per_node": 8,
        "arch": arch, "probe": {"type": "decode_staircase", "levels": []},
        "server": {}, "versions": {}, "clock": {"local_utc_offset_s": 0.0},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    bins, emitted = blb.build_bundle(
        tmp_path, lambda_prefill=LAMBDA_PREFILL,
        lambda_prefill_source="unit measured prefill probe", dt=1.0,
        state_source="measured_engine",
    )

    np.testing.assert_allclose(bins["pre_tok"], 4.0)
    np.testing.assert_allclose(bins["dec_tok"], 8.0)
    np.testing.assert_allclose(bins["A_t"], 3.0)
    np.testing.assert_allclose(bins["engine_tokens_per_iteration"], 12.0)
    np.testing.assert_allclose(bins["engine_gpu_cache_usage"], 0.25)
    assert np.all(np.isfinite(bins["batch"]))
    assert bins["field_sources"]["batch"].startswith("requests.itls")
    assert emitted["_ledger_source"]["state_source"] == "measured_engine"


def test_build_bundle_requires_prefill_rate_provenance(tmp_path):
    with pytest.raises(TypeError):
        blb.build_bundle(tmp_path, lambda_prefill=LAMBDA_PREFILL)


def test_engine_counter_rates_use_bin_edges_and_conserve_work():
    table = {
        "timestamp": np.asarray([0.0, 0.5, 1.0, 1.5, 2.0]),
        "tokens": np.asarray([0.0, 1.0, 2.0, 4.0, 6.0]),
    }
    rates = blb._counter_rate(table, "tokens", np.asarray([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(rates, [2.0, 4.0])
    assert rates.sum() == 6.0


def test_engine_gauge_is_time_averaged_not_sample_count_averaged():
    table = {
        "timestamp": np.asarray([0.0, 0.25, 1.0]),
        "running": np.asarray([0.0, 2.0, 2.0]),
    }
    value = blb._gauge_mean(table, "running", np.asarray([0.0, 1.0]))
    # Linear 0->2 over 0.25 s contributes 0.25 request-seconds, then 2 for
    # 0.75 s contributes 1.5: exact one-second mean = 1.75.
    np.testing.assert_allclose(value, [1.75])


def test_measured_cache_persists_only_available_stock_diagnostics():
    measured = set(blb.ledger_bin_keys("measured_engine"))
    assert set(blb.MEASURED_LEDGER_KEYS) <= measured
    assert not {"collective_time", "nvlink_bytes", "expert_touches"} & measured
    assert blb.ledger_bin_keys("reconstruction") == blb.BIN_KEYS


def test_measured_engine_projection_populates_maintained_ledger(monkeypatch):
    from types import SimpleNamespace

    table = {
        "timestamp": np.asarray([0.0, 1.0, 2.0]),
        "num_requests_waiting": np.zeros(3),
        "num_requests_running": np.asarray([1.0, 1.0, 2.0]),
        "gpu_cache_usage_perc": np.asarray([0.0, 0.5, 1.0]),
        "prompt_tokens_total": np.asarray([0.0, 2.0, 2.0]),
        "generation_tokens_total": np.asarray([0.0, 0.0, 2.0]),
        "iteration_tokens_total_sum": np.asarray([0.0, 2.0, 4.0]),
        "iteration_tokens_total_count": np.asarray([0.0, 1.0, 2.0]),
    }
    record = SimpleNamespace(
        engine_table=table,
        arch={"n_layers": 1, "n_kv": 1, "head_dim": 1, "w_bytes": 100.0,
              "d_model": 1, "moe_frac": 0.0},
        tp=1,
    )
    base = {
        "time_epoch_s": np.asarray([1.0, 2.0]),
        "power": np.asarray([100.0, 200.0]),
        "arrivals": np.asarray([1.0, 0.0]),
        "input_tokens_arriving": np.asarray([2.0, 0.0]),
        "output_tokens_requested": np.asarray([0.0, 2.0]),
        "batch": np.asarray([0.0, 1.0]),
        "pre_active": np.asarray([1.0, 0.0]),
        "w_read_pre": np.asarray([100.0, 0.0]),
        "kv_read": np.asarray([0.0, 40.0]),
        "arch": {key: float(value) for key, value in record.arch.items()},
    }
    monkeypatch.setattr(blb, "reconstruct_bins_from_record", lambda *_a, **_k: base)

    measured = blb.bins_from_engine_csv(record, lambda_prefill=1.0, dt=1.0, trim_s=0.0)

    np.testing.assert_array_equal(measured["pre_tok"], [2.0, 0.0])
    np.testing.assert_array_equal(measured["dec_tok"], [0.0, 2.0])
    np.testing.assert_array_equal(measured["kv_read"], [0.0, 40.0])
    np.testing.assert_allclose(measured["running_requests"], [1.0, 1.5])
    np.testing.assert_allclose(measured["A_t"], [1.0, 1.5])
    np.testing.assert_allclose(measured["engine_tokens_per_iteration"], [2.0, 2.0])
    np.testing.assert_allclose(measured["engine_gpu_cache_usage"], [0.25, 0.75])


# --------------------------------------------------------------------------- #
# Emitter <-> consumer manifest contract: the rate column must come from what
# probe_runner actually writes (per-level summaries), not a key that never
# exists (the old probe.params.rate silently read 0.0 for every bundle).
# --------------------------------------------------------------------------- #

def _emitter_modules():
    import sys as _sys
    from pathlib import Path as _Path
    root = _Path(__file__).resolve().parents[2]
    for sub in ("probes", "client"):
        p = str(root / "profiling" / sub)
        if p not in _sys.path:
            _sys.path.insert(0, p)
    import probe_runner
    import run_manifest
    return probe_runner, run_manifest


def _emitted_manifest(level_windows):
    """Build a manifest through the REAL emitter helpers (not a hand-rolled dict)."""
    _, run_manifest = _emitter_modules()
    return run_manifest.build_manifest(
        run_id="h100_mixed_grid_tp8_TEST",
        probe={"type": "mixed_grid",
               "window": {"start_epoch": EPOCH, "end_epoch": EPOCH + 100.0},
               "levels": level_windows},
        model="unit/model", arch={}, hardware="H100", tp=8, gpus_per_node=8,
        server={}, versions={}, clock={"power_epoch_offset_s": 0.0,
                                       "engine_epoch_offset_s": 0.0,
                                       "monotonic_start": 0.0},
    )


def _level_window(level, t0, t1, summary):
    probe_runner, _ = _emitter_modules()
    from types import SimpleNamespace
    lvl = SimpleNamespace(
        level=level, label=f"lvl_{level}", concurrency=8, num_prompts=32,
        request=SimpleNamespace(input_len=8, output_len=128, prefix_len=0,
                                ignore_eos=True))
    return probe_runner.build_level_window(lvl, t0, t1, ["cmd"], summary)


def test_rate_from_manifest_is_duration_weighted_throughput():
    """Hand-worked: 80 req / 40 s + 40 req / 20 s -> 120/60 = 2.0 req/s.

    Idle levels (all-zero summary, as empty_level_result emits) contribute
    nothing to either numerator or denominator.
    """
    windows = [
        _level_window(0, EPOCH, EPOCH + 40.0,
                      {"duration": 40.0, "completed": 80,
                       "request_throughput": 2.0}),
        _level_window(1, EPOCH + 40.0, EPOCH + 60.0,
                      {"duration": 0, "completed": 0,
                       "request_throughput": 0}),  # idle level
        _level_window(2, EPOCH + 60.0, EPOCH + 80.0,
                      {"duration": 20.0, "completed": 40,
                       "request_throughput": 2.0}),
    ]
    manifest = _emitted_manifest(windows)
    assert blb.rate_from_manifest(manifest) == pytest.approx(120.0 / 60.0)


def test_rate_from_manifest_zero_when_no_levels():
    assert blb.rate_from_manifest(_emitted_manifest([])) == 0.0
    assert blb.rate_from_manifest({}) == 0.0


def test_ledger_view_from_record_matches_old_builder(tmp_path):
    """Legacy parsing and direct parsing feed the same shared ledger contract."""
    from model.training_data.ledger_view import reconstruct_bins_from_record
    from model.training_data.run_record import load_legacy_run

    jp, cp = _write_old_format_bundle(tmp_path)
    direct = blb.state_from_requests(
        jp, cp, ARCH["llama-3-70b"], 8, LAMBDA_PREFILL, dt=DT
    )

    record = load_legacy_run(
        {
            "model_name": "llama-3-70b",
            "hardware": "H100",
            "tensor_parallelism": "8",
            "power_csv_path": cp,
            "json_path": jp,
            "pair_key": "synthetic",
            "rate": "1.0",
        }
    )
    assert record is not None
    new = reconstruct_bins_from_record(record, lambda_prefill=LAMBDA_PREFILL, dt=DT)

    assert direct is not None and new is not None
    assert direct["n"] == new["n"]
    for key in blb.BIN_KEYS:
        np.testing.assert_array_equal(
            new[key], direct[key],
            err_msg=f"legacy RunRecord changed ledger column '{key}'",
        )


def test_default_discovery_ignores_campaign_logs_but_keeps_bundle_candidates(
    tmp_path, monkeypatch
):
    """A campaign's logs are not bundles; manifest-bearing directories are."""
    monkeypatch.chdir(tmp_path)
    campaign = tmp_path / "data" / "runs" / "campaign"
    logs = campaign / "logs"
    complete = campaign / "run-1"
    incomplete = campaign / "run-2"
    logs.mkdir(parents=True)
    complete.mkdir()
    incomplete.mkdir()
    (complete / "manifest.json").write_text("{}")
    (incomplete / "manifest.json").write_text("{}")

    assert blb.discover_run_dirs(blb.DEFAULT_RUNS_GLOB) == [
        complete.relative_to(tmp_path),
        incomplete.relative_to(tmp_path),
    ]


def test_explicit_discovery_keeps_non_bundle_dirs_for_fail_fast_ingestion(
    tmp_path, monkeypatch
):
    """An explicit malformed path must reach the normal bundle reader and fail."""
    monkeypatch.chdir(tmp_path)
    logs = tmp_path / "data" / "runs" / "campaign" / "logs"
    logs.mkdir(parents=True)
    selected = blb.discover_run_dirs("data/runs/campaign/logs")

    assert selected == [logs.relative_to(tmp_path)]
    with pytest.raises(ValueError, match="Incomplete bundle"):
        blb.build_bundle(
            selected[0],
            lambda_prefill=LAMBDA_PREFILL,
            lambda_prefill_source="unit test",
        )
