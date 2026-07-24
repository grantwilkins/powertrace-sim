"""
Claim:
The disaggregated campaign measures one queue-free, one target, and one stress
rate for three ten-minute repetitions, while its smoke mode measures only the
target cell. A run is accepted only when every measured request has all timing
arrays and a joinable proxy timeline; the benchmark's preflight request is not
mistaken for measured traffic.

Plausible wrong implementations:
- Profile only 2 requests/s and mistake one operating point for a load curve.
- Compute prompt counts in minutes rather than seconds.
- Accept a benchmark summary whose detailed timing arrays are incomplete.
- Mutate the decode request while constructing the one-token prefill request.
- Continue without the KV handoff metadata produced by the prefiller.
- Count the benchmark preflight request as part of the measured workload.
"""
import json

import pytest

from profiling.disaggregated_prefill.campaign import (
    cells,
    run_metadata,
    validate_events,
    validate_result,
)
from profiling.disaggregated_prefill.proxy import prefill_payload, transferred_payload
from profiling.disaggregated_prefill.telemetry import NIXL_COLUMNS, role_row


def test_campaign_cells_cover_intrinsic_target_and_stress_loads():
    plan = cells()
    assert [(rate, repeat) for rate, repeat, _ in plan] == [
        (rate, repeat) for rate in (0.25, 2.0, 4.0) for repeat in range(3)
    ]
    assert [prompts for _, _, prompts in plan] == [150] * 3 + [1200] * 3 + [2400] * 3
    assert cells(smoke=True) == [(2.0, 0, 1200)]


def test_result_gate_requires_complete_request_level_timing(tmp_path):
    path = tmp_path / "requests.json"
    result = {"completed": 2, "input_lens": [1, 2], "output_lens": [3, 4],
              "ttfts": [0.1, 0.2], "itls": [[0.1], [0.2]],
              "request_timestamps": [10.0, 11.0], "request_ids": ["a", "b"]}
    path.write_text(json.dumps(result))
    validate_result(path, 2)
    result["itls"].pop()
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="1/2 itls"):
        validate_result(path, 2)


def test_prefill_handoff_preserves_the_original_decode_request():
    original = {"prompt": "x", "max_tokens": 8, "stream": True,
                "min_tokens": 8, "stream_options": {"include_usage": True}}
    prefill = prefill_payload(original)
    assert prefill["max_tokens"] == 1 and prefill["stream"] is False
    assert "min_tokens" not in prefill and "stream_options" not in prefill
    assert original["max_tokens"] == 8 and original["stream"] is True
    params = {"remote_host": "127.0.0.1", "remote_block_ids": [1]}
    decode = transferred_payload(original, {"kv_transfer_params": params})
    assert decode["kv_transfer_params"] == params
    assert decode["max_tokens"] == 8
    with pytest.raises(ValueError, match="omitted kv_transfer_params"):
        transferred_payload(original, {})


def test_stage_gate_requires_one_ordered_timeline_per_request(tmp_path):
    path = tmp_path / "events.jsonl"
    sequence = ("proxy_received", "prefill_sent", "prefill_completed", "decode_sent",
                "decode_first_byte", "decode_completed")
    old = {"request_id": "benchmark-preflight", "event": "ignored", "wall_ns": 0}
    rows = [old] + [
        {"request_id": request_id, "event": event, "wall_ns": base + index}
        for request_id, base in (("a", 10), ("b", 20))
        for index, event in enumerate(sequence)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    validate_events(path, 0, ["a", "b"])
    rows[-1]["event"] = "decode_first_byte"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="incomplete stage events"):
        validate_events(path, 0, ["a", "b"])


def test_role_telemetry_retains_nixl_transfer_work():
    parsed = {
        "vllm:nixl_xfer_time_seconds_sum": [0.25],
        "vllm:nixl_xfer_time_seconds_count": [2.0],
        "vllm:nixl_bytes_transferred_sum": [1024.0, 2048.0],
    }
    row = role_row(parsed, 10.0)
    tail = dict(zip(NIXL_COLUMNS, row[-len(NIXL_COLUMNS):]))
    assert tail["nixl_xfer_time_seconds_sum"] == 0.25
    assert tail["nixl_xfer_time_seconds_count"] == 2.0
    assert tail["nixl_bytes_transferred_sum"] == 3072.0


def test_run_metadata_binds_each_role_to_one_distinct_gpu():
    metadata = run_metadata("GPU-prefill", "GPU-decode", "/image.sif", "/share.json")
    assert metadata["roles"] == {
        "prefill": {"tp": 1, "gpu_uuid": "GPU-prefill"},
        "decode": {"tp": 1, "gpu_uuid": "GPU-decode"},
    }
    assert metadata["workload"]["arrival_process"] == "poisson"
    assert metadata["clock"]["power_timestamp_basis"] == "local_wall_time"
    assert metadata["clock"]["request_timestamp_basis"] == "unix_epoch"
    with pytest.raises(ValueError, match="distinct GPU UUIDs"):
        run_metadata("GPU-same", "GPU-same", "/image.sif", "/share.json")
