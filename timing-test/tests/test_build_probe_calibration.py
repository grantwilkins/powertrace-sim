"""Tests for timing-test/build_probe_calibration.py.

The hand-checked row is pinned from
data/runs/h100_tier1_llama70b/h100_decode_staircase_tp8_1783963433:
manifest.json probe.levels[0] (label decode_N1, concurrency 1,
summary.duration 49.2991554196924, output_throughput 83.08458765936314)
and levels/level_000.json (mean_itl_ms 12.079044153737062,
mean_tpot_ms 12.031658261179064, mean_ttft_ms 20.143055822700262).
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "timing-test"))

import build_probe_calibration as bpc  # noqa: E402
import fit_efficiencies as fit  # noqa: E402

OUTPUT = REPO / "timing-test/probe_calibration.json"

REQUIRED_FIELDS = {
    "hardware", "tp", "probe", "run_id", "label", "level", "batch",
    "prompt_tokens", "output_len", "context_tokens_mean",
    "level_duration_s", "output_throughput_tok_s", "measured",
    "engine_counters", "provenance",
}


@pytest.fixture(scope="module")
def doc():
    if not OUTPUT.exists():
        bpc.build()
    return json.loads(OUTPUT.read_text())


def test_schema_round_trip(doc):
    """Every row carries the required fields; JSON round-trips losslessly."""
    assert doc["n_rows"] == len(doc["rows"]) > 0
    for row in doc["rows"]:
        assert REQUIRED_FIELDS <= set(row), row["run_id"]
        assert row["measured"]["mean_ttft_ms"] > 0
        if row["probe"] != "prefill_staircase":
            # prefill levels emit 1 output token, so ITL/TPOT are 0 there
            assert row["measured"]["mean_itl_ms"] > 0
            assert row["measured"]["median_tpot_ms"] > 0
        prov = row["provenance"]
        assert prov["manifest_git_sha"]
        assert (REPO / prov["bundle_path"] / "manifest.json").exists()
        assert prov["extraction_method"].startswith("level_detail(a)")
        if row["probe"] == "prefill_staircase":
            assert row["prefill_time_ms_mean_excl_cache_hits"] > 0
        else:
            assert row["decode_tokens_per_s_per_request"] > 0
            assert row["effective_decode_batch"] > 0
    assert json.loads(json.dumps(doc)) == doc


def test_hand_checked_row_h100_tp8_decode_n1(doc):
    """Pin decode_N1 of h100_decode_staircase_tp8_1783963433 (see docstring)."""
    (row,) = [
        r for r in doc["rows"]
        if r["run_id"] == "h100_decode_staircase_tp8_1783963433"
        and r["level"] == 0
    ]
    assert row["hardware"] == "H100" and row["tp"] == 8
    assert row["probe"] == "decode_staircase" and row["label"] == "decode_N1"
    assert row["batch"] == 1
    # params: input_len 8, prefix_len 0, output_len 2048
    assert row["prompt_tokens"] == 8 and row["output_len"] == 2048
    assert row["context_tokens_mean"] == 8 + (2048 + 1) / 2  # 1032.5
    assert row["level_duration_s"] == pytest.approx(49.2991554196924)
    assert row["output_throughput_tok_s"] == pytest.approx(83.08458765936314)
    assert row["measured"]["mean_itl_ms"] == pytest.approx(12.079044153737062)
    assert row["measured"]["mean_ttft_ms"] == pytest.approx(20.143055822700262)
    assert row["decode_tokens_per_s_per_request"] == pytest.approx(
        1000.0 / 12.031658261179064
    )
    # engine-counter cross-check agrees with client ITL within 2%
    steady = row["engine_counters"]["steady"]
    assert steady["iteration_time_ms"] == pytest.approx(
        row["measured"]["mean_itl_ms"], rel=0.02
    )
    assert steady["tokens_per_iteration"] == pytest.approx(1.0, rel=0.01)


def test_engine_window_stats_synthetic():
    """Counter math on a hand-built engine trace: 4 samples at 1 s cadence,
    100 iterations/s and 3 tokens/iteration during the active span."""
    engine = [
        (0.0, 0.0, 0.0, 0.0, 2.0),      # idle (client setup)
        (1.0, 3.0, 0.0, 0.0, 2.0),
        (2.0, 3.0, 300.0, 100.0, 2.0),
        (3.0, 3.0, 600.0, 200.0, 3.0),
        (4.0, 0.0, 600.0, 200.0, 3.0),  # idle (client teardown)
    ]
    s = bpc.engine_window_stats(engine, 0.0, 4.0)
    assert s["iterations"] == 200.0
    assert s["tokens_per_iteration"] == pytest.approx(3.0)
    assert s["preemptions"] == 1.0
    assert s["steady"]["iteration_time_ms"] == pytest.approx(10.0)
    assert s["steady"]["num_running_mean"] == pytest.approx(3.0)
    assert bpc.engine_window_stats(engine, 0.0, 0.5) is None


def test_preempted_probe_level_cannot_fit_nonpreemptive_simulator():
    row = {
        "hardware": "A100", "tp": 4, "model": "llama-3-70b",
        "run_id": "probe", "probe": "decode_staircase", "label": "N256",
        "batch": 256, "effective_decode_batch": 200.0,
        "context_tokens_mean": 1024.0,
        "measured": {"median_itl_ms": 50.0},
        "engine_counters": {"preemptions": 58.0},
    }

    assert fit.probe_points({"rows": [row]}) == []
