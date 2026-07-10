"""
Claim:
Roofline windows compare canonical bundle power, requests, and probe windows
on one exact UTC epoch basis.

Plausible wrong implementations:
- Ignore the bundle's local-wall-time UTC offset.
- Apply the offset with the wrong sign.
- Reapply the legacy 30-minute timestamp fold to bundle timestamps.
- Convert probe labels on a clock basis different from the window data.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "profiling" / "jobs"))

import campaign_config as cc  # noqa: E402
from scripts.eval.occupancy_roofline import analyze_run_root, window_bundle  # noqa: E402

BASE_DT = datetime(2026, 1, 1, tzinfo=timezone.utc)
BASE_EPOCH = BASE_DT.timestamp()


def _stamp(seconds: int) -> str:
    return (BASE_DT + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S.%f")


def _requests(arrivals, input_len, output_len, ttft, itl, *, epoch_base=BASE_EPOCH):
    n = len(arrivals)
    return {
        "request_timestamps": [epoch_base + float(a) for a in arrivals],
        "input_lens": [input_len] * n,
        "output_lens": [output_len] * n,
        "ttfts": [ttft] * n,
        "itls": [list(itl) for _ in range(n)],
    }


def _write_bundle(root: Path, run_id: str, probe_type: str, requests: dict,
                  power_w: float, duration_s: int = 40,
                  local_utc_offset_s: float = 0.0) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    with open(run_dir / "power.csv", "w") as fh:
        fh.write("timestamp,index,uuid,power.draw\n")
        for t in range(duration_s + 1):
            for gpu in range(8):
                fh.write(f"{_stamp(t)},{gpu},GPU-{gpu},{power_w / 2.0}\n")
    (run_dir / "requests.json").write_text(json.dumps(requests))
    (run_dir / "engine.csv").write_text("timestamp,num_requests_running\n")
    epoch_start = BASE_EPOCH - local_utc_offset_s
    manifest = {
        "manifest_version": 1,
        "run_id": run_id,
        "model": "google/gemma-4-26B-A4B-it",
        "arch": {"family": "unit"},
        "hardware": "A100",
        "tp": 2,
        "gpus_per_node": 8,
        "clock": {"local_utc_offset_s": local_utc_offset_s},
        "probe": {
            "type": probe_type,
            "window": {
                "start_epoch": epoch_start,
                "end_epoch": epoch_start + duration_s,
            },
            "levels": [{
                "label": probe_type,
                "t_start_epoch": epoch_start,
                "t_end_epoch": epoch_start + duration_s,
            }],
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))


def test_roofline_campaign_commands(monkeypatch):
    monkeypatch.setenv("RUNS", "/tmp/powertrace-runs")
    c = cc.load_campaign(ROOT / "profiling/campaigns/roofline_gemma-4-26b-a4b_a100.json")

    assert c["campaign_type"] == "roofline"
    assert cc.tp_degrees(c) == [2]
    assert cc.probes_for_tp(c, 2) == [
        "idle_hold",
        "prefill_staircase",
        "decode_staircase",
        "context_holds",
        "mixed_grid",
    ]

    commands = dict(zip(cc.probes_for_tp(c, 2), cc.probe_commands(c, 2)))
    assert "--contexts 2048 8192 32768 65536" in commands["context_holds"]
    assert "--n-points 24" in commands["mixed_grid"]
    assert "--prefill-min 512" in commands["mixed_grid"]
    assert "--prefill-max 65536" in commands["mixed_grid"]

    serve = cc.probe_serve_command(c, "context_holds", 2)
    assert serve.startswith("env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve")

    agentic = cc.roofline_agentic_command(c, 2)
    assert "--prefix-tokens 4096" in agentic
    assert "--user-tokens-mean 1024" in agentic
    assert "--prefix-cache" not in agentic

    analyze = cc.roofline_analyze_command(c, 2)
    assert "--run-root /tmp/powertrace-runs" in analyze
    assert "--window-s 5.0" in analyze


def test_occupancy_roofline_analyzes_synthetic_bundles(tmp_path):
    run_root = tmp_path / "runs"
    _write_bundle(run_root, "a100_idle_hold_tp2_1", "idle_hold",
                  {k: [] for k in ("request_timestamps", "input_lens", "output_lens", "ttfts", "itls")},
                  power_w=100.0)
    arrivals = [5, 10, 15, 20, 25, 30]
    _write_bundle(run_root, "a100_prefill_staircase_tp2_1", "prefill_staircase",
                  _requests(arrivals, 5000, 1, 5.0, []), power_w=300.0)
    _write_bundle(run_root, "a100_decode_staircase_tp2_1", "decode_staircase",
                  _requests(arrivals, 10, 5000, 0.001, [0.001] * 4999),
                  power_w=400.0)
    _write_bundle(run_root, "a100_agentic_tp2_1", "agentic",
                  _requests([5, 15, 25], 2500, 2500, 2.5, [0.002] * 2499),
                  power_w=250.0)

    summary = analyze_run_root(
        run_root,
        label="unit_roofline",
        model="google/gemma-4-26B-A4B-it",
        hardware="A100",
        tp=2,
        window_s=5.0,
        out_root=tmp_path / "results",
        fig_root=tmp_path / "figures",
    )

    assert summary["n_bundles"] == 4
    assert summary["F_prefill_tps"] == pytest.approx(1000.0)
    assert summary["G_decode_tps"] == pytest.approx(1000.0, rel=0.01)
    assert summary["P_idle_w"] == pytest.approx(100.0)
    assert summary["ell_power_knee"] > 0
    assert (tmp_path / "results/unit_roofline_summary.csv").exists()
    assert (tmp_path / "results/unit_roofline_windows.csv").exists()
    assert (tmp_path / "figures/unit_roofline_power_vs_ell.png").exists()


def test_occupancy_roofline_requires_capacity_probes(tmp_path):
    run_root = tmp_path / "runs"
    _write_bundle(run_root, "a100_idle_hold_tp2_1", "idle_hold",
                  {k: [] for k in ("request_timestamps", "input_lens", "output_lens", "ttfts", "itls")},
                  power_w=100.0)

    with pytest.raises(ValueError, match="prefill_staircase and decode_staircase"):
        analyze_run_root(
            run_root,
            label="missing_capacity",
            window_s=5.0,
            out_root=tmp_path / "results",
            fig_root=tmp_path / "figures",
        )


def test_roofline_uses_exact_bundle_epoch_with_nonstandard_utc_offset(tmp_path):
    """A 660 s local offset must not be rounded away by legacy timestamp folding."""
    offset_s = 660.0
    epoch_start = BASE_EPOCH - offset_s
    _write_bundle(
        tmp_path,
        "offset_prefill",
        "prefill_staircase",
        _requests([1.0], 50, 1, 1.0, [], epoch_base=epoch_start),
        power_w=200.0,
        duration_s=10,
        local_utc_offset_s=offset_s,
    )

    rows = window_bundle(tmp_path / "offset_prefill", window_s=5.0)

    assert len(rows) == 2
    assert rows[0]["window_start_s"] == pytest.approx(0.0)
    assert rows[0]["window_end_s"] == pytest.approx(5.0)
    assert rows[0]["f_tps"] == pytest.approx(10.0)
    assert rows[0]["source_label"] == "prefill_staircase"
