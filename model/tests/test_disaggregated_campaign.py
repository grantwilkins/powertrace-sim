"""
Claim:
The confirmatory campaign measures cache-disabled long-context prefill and
decode at two observable, nonsaturated loads. One cell calibrates role idles
and gains; independently seeded repeated cells remain held out. A run is
accepted only when requests, proxy stages, NIXL transfers, and uncached prompt
work agree.

Plausible wrong implementations:
- Reuse the calibration seed or fit on one of the repeated evaluation cells.
- Keep the unsupported 8192-token scheduler budget or prefix caching enabled.
- Let short prefills fall below one 250 ms measurement interval.
- Accept a benchmark summary whose detailed timing arrays are incomplete.
- Mutate the decode request while constructing the one-token prefill request.
- Continue without the KV handoff metadata produced by the prefiller.
- Count the benchmark preflight request as part of the measured workload.
"""
import json
from datetime import datetime
from pathlib import Path

import pytest

from profiling.disaggregated_prefill.campaign import (
    cells,
    run_metadata,
    validate_events,
    validate_prefill_observability,
    validate_power,
    validate_result,
    validate_uncached_metrics,
    validate_workload,
)
from profiling.disaggregated_prefill.proxy import create_app, prefill_payload, transferred_payload
from profiling.disaggregated_prefill.telemetry import (
    NIXL_COLUMNS,
    REQUIRED_COLUMNS,
    missing_required,
    role_row,
)

ROOT = Path(__file__).resolve().parents[2]


def test_campaign_cells_freeze_one_calibration_and_repeated_holdouts():
    assert cells() == [
        (2.0, 0, 600, 0, "calibration"),
        (1.0, 0, 300, 2, "evaluation"),
        (2.0, 1, 600, 1, "evaluation"),
        (1.0, 1, 300, 2, "evaluation"),
        (2.0, 2, 600, 1, "evaluation"),
    ]
    assert cells(smoke=True) == [(2.0, 0, 120, 0, "smoke")]


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


def test_workload_gate_enforces_the_declared_long_context_ranges(tmp_path):
    path = tmp_path / "requests.json"
    path.write_text(json.dumps({
        "input_lens": [6144, 10240],
        "output_lens": [48, 80],
    }))
    validate_workload(path)
    path.write_text(json.dumps({
        "input_lens": [6143],
        "output_lens": [64],
    }))
    with pytest.raises(ValueError, match="input_lens outside"):
        validate_workload(path)


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

def test_proxy_route_injects_the_http_request(tmp_path):
    TestClient = pytest.importorskip("fastapi.testclient").TestClient

    app = create_app("http://prefill", "http://decode", tmp_path / "events.jsonl")
    route = next(route for route in app.routes if route.path == "/v1/completions")
    assert route.dependant.request_param_name == "request"
    assert not route.dependant.query_params
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200



def test_stage_gate_requires_complete_observable_prefill_timelines(tmp_path):
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
    timelines = validate_events(path, 0, ["a", "b"])
    with pytest.raises(ValueError, match="median prefill duration"):
        validate_prefill_observability(timelines)
    for row in rows:
        if row["event"] == "prefill_completed":
            row["wall_ns"] += 300_000_000
        elif row["event"] in ("decode_sent", "decode_first_byte", "decode_completed"):
            row["wall_ns"] += 300_000_000
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    timelines = validate_events(path, 0, ["a", "b"])
    validate_prefill_observability(timelines)
    rows[-1]["event"] = "decode_first_byte"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="incomplete stage events"):
        validate_events(path, 0, ["a", "b"])


def _write_metrics(
    path, *, prompt_tokens, cache_hits, transfers, failed_transfers=0
):
    path.write_text(
        "prompt_tokens_total,prefix_cache_hits_total,"
        "nixl_xfer_time_seconds_count,nixl_num_failed_transfers,"
        "nixl_num_failed_notifications,nixl_num_kv_expired_reqs,"
        "num_preemptions_total\n"
        "0,0,0,0,0,0,0\n"
        f"{prompt_tokens},{cache_hits},{transfers},{failed_transfers},0,0,0\n"
    )


def test_uncached_gate_rejects_cache_work_and_hidden_requests(tmp_path):
    request_path = tmp_path / "requests.json"
    prefill_path = tmp_path / "prefill.csv"
    decode_path = tmp_path / "decode.csv"
    request_path.write_text(json.dumps({
        "completed": 2,
        "input_lens": [7000, 8000],
    }))
    _write_metrics(
        prefill_path, prompt_tokens=15000, cache_hits=0, transfers=0
    )
    _write_metrics(decode_path, prompt_tokens=0, cache_hits=0, transfers=2)
    validate_uncached_metrics(request_path, prefill_path, decode_path)

    _write_metrics(
        prefill_path, prompt_tokens=15000, cache_hits=1, transfers=0
    )
    with pytest.raises(ValueError, match="cached prompt tokens"):
        validate_uncached_metrics(request_path, prefill_path, decode_path)

    _write_metrics(
        prefill_path, prompt_tokens=15000, cache_hits=0, transfers=0
    )
    _write_metrics(decode_path, prompt_tokens=0, cache_hits=0, transfers=3)
    with pytest.raises(ValueError, match="3 transfers for 2 measured"):
        validate_uncached_metrics(request_path, prefill_path, decode_path)

    _write_metrics(decode_path, prompt_tokens=0, cache_hits=0, transfers=2)
    _write_metrics(
        prefill_path,
        prompt_tokens=15000,
        cache_hits=0,
        transfers=0,
        failed_transfers=1,
    )
    with pytest.raises(
        ValueError, match="prefill.csv recorded 1 nixl_num_failed_transfers"
    ):
        validate_uncached_metrics(request_path, prefill_path, decode_path)


def test_power_gate_requires_complete_timed_samples_and_idle_restoration(tmp_path):
    path = tmp_path / "power.csv"
    base = datetime.now().replace(microsecond=0).timestamp()
    lines = [
        "timestamp,query.start,query.end,index,uuid,power.draw [W],"
        "temperature.gpu"
    ]
    for sample in range(361):
        epoch = base + 0.25 * sample
        timestamp = datetime.fromtimestamp(epoch).strftime(
            "%Y/%m/%d %H:%M:%S.%f"
        )[:-3]
        start = datetime.fromtimestamp(epoch - 0.01).strftime(
            "%Y/%m/%d %H:%M:%S.%f"
        )[:-3]
        end = datetime.fromtimestamp(epoch + 0.01).strftime(
            "%Y/%m/%d %H:%M:%S.%f"
        )[:-3]
        active = base + 30.0 <= epoch <= base + 60.0
        for index, uuid in enumerate(("GPU-prefill", "GPU-decode")):
            lines.append(
                f"{timestamp},{start},{end},{index},{uuid},"
                f"{150 if active else 90},{70 if active else 50}"
            )
    path.write_text("\n".join(lines) + "\n")
    validate_power(
        path,
        gpu_uuids=("GPU-prefill", "GPU-decode"),
        workload_start=base + 30.0,
        workload_end=base + 60.0,
    )

    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="incomplete GPU sample"):
        validate_power(
            path,
            gpu_uuids=("GPU-prefill", "GPU-decode"),
            workload_start=base + 30.0,
            workload_end=base + 60.0,
        )


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


def test_telemetry_preflight_requires_every_final_cache_and_nixl_gate():
    parsed = {
        f"vllm:{column}": [0.0]
        for column in REQUIRED_COLUMNS
    }
    assert missing_required(parsed) == []
    del parsed["vllm:nixl_num_failed_transfers"]
    assert missing_required(parsed) == ["nixl_num_failed_transfers"]


def test_run_metadata_binds_each_role_to_one_distinct_gpu():
    metadata = run_metadata("GPU-prefill", "GPU-decode", "/image.sif")
    assert metadata["roles"] == {
        "prefill": {"tp": 1, "gpu_uuid": "GPU-prefill"},
        "decode": {"tp": 1, "gpu_uuid": "GPU-decode"},
    }
    assert metadata["server"]["max_num_batched_tokens"] == 2048
    assert metadata["server"]["enable_prefix_caching"] is False
    assert metadata["workload"]["dataset"] == "random"
    assert metadata["workload"]["input_tokens"] == 8192
    assert metadata["workload"]["output_tokens"] == 64
    assert metadata["workload"]["arrival_process"] == "poisson"
    assert metadata["power_measurement"] == {
        "source": "nvidia-smi power.draw",
        "interval_ms": 250,
        "profile": "core_timed",
        "sample_timestamp": "host query start/end midpoint",
        "raw_samples": True,
    }
    assert metadata["analysis_protocol"]["calibration_cell"] == "rate-2-repeat-0"
    assert metadata["analysis_protocol"]["heldout_role_acceptance"] == {
        "correlation_min": 0.8,
        "std_ratio_min": 0.8,
        "std_ratio_max": 1.25,
        "p95_error_pct_max": 10.0,
        "model_to_duty_null_loss_ratio_max": 0.95,
    }
    assert "never average duplicate bins" in metadata[
        "analysis_protocol"
    ]["sample_mapping"]
    assert metadata["clock"]["power_timestamp_basis"] == (
        "local query midpoint wall time"
    )
    assert metadata["clock"]["request_timestamp_basis"] == "unix_epoch"
    smoke = run_metadata("GPU-prefill", "GPU-decode", "/image.sif", True)
    assert smoke["mode"] == "smoke"
    assert smoke["workload"]["cells"] == [{
        "rate": 2.0,
        "repeat": 0,
        "prompts": 120,
        "seed": 0,
        "split": "smoke",
    }]
    with pytest.raises(ValueError, match="distinct GPU UUIDs"):
        run_metadata("GPU-same", "GPU-same", "/image.sif")


def test_batch_launch_uses_submitted_checkout_and_exposes_nixl_runtime():
    batch = (
        ROOT / "profiling/jobs/disaggregated_gpt_oss_20b.sbatch"
    ).read_text()
    runner = (
        ROOT / "profiling/jobs/run_disaggregated_gpt_oss_20b.sh"
    ).read_text()
    assert 'REPO="${POWERTRACE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"' in batch
    assert 'export POWERTRACE_REPO="$REPO"' in batch
    assert "python/3.12.1 2>/dev/null || true" not in batch
    assert "sys.version_info >= (3, 10)" in batch
    assert 'ln -sfn "$NIXL_SITE/nixl_cu12" "$NIXL_COMPAT/nixl"' in runner
    assert "NixlWrapper is not None and nixl_agent_config is not None" in runner
    assert '--env "HF_HOME=$ROOT/hf"' in runner
    assert runner.index("METADATA_ARG=") < runner.index("campaign.py metadata")
    assert '${METADATA_ARG:+"$METADATA_ARG"}' in runner
    assert "python3 -m profiling.disaggregated_prefill.telemetry" in runner
    assert "--max-num-batched-tokens 2048" in runner
    assert "--no-enable-prefix-caching" in runner
    assert "--dataset-name random" in runner
    assert "--random-input-len 8192 --random-output-len 64" in runner
    assert "--profile core_timed" in runner
    assert "--ignore-eos --skip-test-prompt" in runner
    assert "confirmatory runs cannot resume across allocations" in runner
    assert '--power "$DIR/power.csv"' in runner
    assert '--workload-start "$DIR/workload_start_epoch_s"' in runner
    assert runner.index('> "$RUN_ROOT/preflight.log"') < runner.index(
        'python3 profiling/client/power_logger.py'
    )
    assert "PLAN_ARGS" not in runner
    assert '${PLAN_ARG:+"$PLAN_ARG"}' in runner
    dataset = (ROOT / "profiling/client/benchmark_dataset.py").read_text()
    imports, burst = dataset.split("class BurstGPTDataset", 1)
    assert "import pandas as pd" not in imports
    assert "import pandas as pd" in burst
