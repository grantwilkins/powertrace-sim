"""Unit tests for the run manifest assembler (CAMPAIGN.md §2 / §5-E)."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # profiling/client

from run_manifest import build_manifest, capture_clock, write_manifest  # noqa: E402


def _sample_manifest():
    return build_manifest(
        run_id="h100_decode_staircase_tp8_n16_20260615-2210",
        probe={"type": "decode_staircase", "level": 16, "params": {"concurrency": 16}},
        model="meta-llama/Llama-3.1-70B-Instruct",
        arch={"n_active": 7.05e10, "w_bytes": 1.41e11, "n_layers": 80,
              "linear_attention": 0, "swa_global_ratio": 0.0},
        hardware="H100",
        tp=8,
        gpus_per_node=8,
        server={"max_num_seqs": 256, "max_num_batched_tokens": 8192,
                "enable_chunked_prefill": True, "enable_prefix_caching": False,
                "kv_cache_dtype": "auto", "max_model_len": 131072},
        versions={"vllm": "0.x.y", "git_sha": "deadbeef", "gpu_driver": "560.0"},
        clock=capture_clock(),
    )


def test_manifest_schema():
    m = _sample_manifest()
    for key in ("manifest_version", "run_id", "probe", "model", "arch", "hardware",
                "tp", "gpus_per_node", "server", "versions", "clock"):
        assert key in m
    assert m["probe"]["type"] == "decode_staircase"
    assert m["probe"]["level"] == 16
    for k in ("max_num_seqs", "enable_chunked_prefill", "enable_prefix_caching",
              "kv_cache_dtype", "max_model_len"):
        assert k in m["server"]
    for k in ("vllm", "git_sha", "gpu_driver"):
        assert k in m["versions"]
    for k in ("power_epoch_offset_s", "engine_epoch_offset_s", "monotonic_start"):
        assert k in m["clock"]


def test_clock_offsets_finite_and_monotonic():
    c = capture_clock()
    assert math.isfinite(c["power_epoch_offset_s"])
    assert math.isfinite(c["engine_epoch_offset_s"])
    assert c["monotonic_start"] > 0.0


def test_clock_offset_uses_provided_epochs():
    import time
    # an engine clock 2.5 s ahead of "now" should yield ~+2.5 s offset
    c = capture_clock(engine_clock_epoch=time.time() + 2.5)
    assert abs(c["engine_epoch_offset_s"] - 2.5) < 0.5


def test_write_manifest_roundtrip(tmp_path):
    import json
    p = tmp_path / "manifest.json"
    m = _sample_manifest()
    write_manifest(str(p), m)
    assert json.loads(p.read_text())["run_id"] == m["run_id"]
