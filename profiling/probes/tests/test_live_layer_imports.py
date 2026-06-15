"""Import-safety for the live probe layer (not behavior-tested; runs on GPU box).

The live layer (runner/send/CLIs) is validated end-to-end on first launch per the
acceptance gate. Here we only guarantee the modules import cleanly (so pytest
collection and `python -m profiling.probes.<probe>` don't crash before reaching
the server) and that the thin CLI plumbing assembles the server config.
"""

import argparse
import importlib

import pytest


@pytest.mark.parametrize("mod", [
    "bench_driver", "probe_runner", "_cli",
    "decode_staircase", "prefill_staircase", "context_holds",
    "transients", "mixed_grid",
    "agentic", "session_driver", "session_runner", "agentic_run",
])
def test_live_modules_import(mod):
    importlib.import_module(mod)


def test_cli_server_cfg_shape():
    import _cli

    ns = argparse.Namespace(
        max_num_seqs=256, max_num_batched_tokens=8192,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        kv_cache_dtype="auto", max_model_len=131072,
    )
    cfg = _cli.server_cfg(ns)
    assert cfg["max_num_seqs"] == 256
    assert cfg["enable_chunked_prefill"] is True
    for k in ("max_num_batched_tokens", "enable_prefix_caching",
              "kv_cache_dtype", "max_model_len"):
        assert k in cfg


def test_bench_driver_build_command_has_alignment_flags():
    """The benchmark invocation must request the epoch-timestamped detailed output."""
    import bench_driver
    from schedule import build_decode_staircase

    level = build_decode_staircase(8).levels[2]  # concurrency 4
    cmd = bench_driver.build_command(
        "meta-llama/Llama-3.1-70B-Instruct", "http://localhost:8000/v1", 8,
        level, "/tmp/level.json")
    s = " ".join(cmd)
    assert "--save-detailed" in s and "--save-result" in s
    assert "--request-rate inf" in s
    assert "--max-concurrency 4" in s
    assert "--ignore-eos" in s
    assert "--num-prompts" in s
    assert "--result-filename /tmp/level.json" in s


def test_merge_request_arrays_concatenates_levels():
    import bench_driver
    lr = [
        {"arrays": {k: [1, 2] for k in bench_driver.REQUEST_ARRAY_KEYS},
         "summary": {}},
        {"arrays": {k: [3] for k in bench_driver.REQUEST_ARRAY_KEYS}, "summary": {}},
    ]
    merged = bench_driver.merge_request_arrays(lr)
    for k in ("input_lens", "output_lens", "ttfts", "itls", "request_timestamps"):
        assert merged[k] == [1, 2, 3]


def test_build_level_window_records_epoch_bounds():
    import probe_runner
    from schedule import build_decode_staircase

    level = build_decode_staircase(8).levels[0]
    w = probe_runner.build_level_window(level, 1000.0, 1045.0, ["cmd"], {"completed": 4})
    assert w["t_start_epoch"] == 1000.0 and w["t_end_epoch"] == 1045.0
    assert w["concurrency"] == level.concurrency
    assert w["num_prompts"] == level.num_prompts
    assert w["params"]["output_len"] == 2048


def test_requests_json_writer_roundtrip(tmp_path):
    import json
    import probe_runner
    import bench_driver
    lr = [{"arrays": {k: [7] for k in bench_driver.REQUEST_ARRAY_KEYS}, "summary": {}}]
    merged = probe_runner.assemble_requests_json(lr)
    (tmp_path / "requests.json").write_text(json.dumps(merged))
    data = json.loads((tmp_path / "requests.json").read_text())
    for k in ("input_lens", "output_lens", "ttfts", "itls", "request_timestamps"):
        assert data[k] == [7]
