"""Unit tests for the /metrics scraper mapping (CAMPAIGN.md §5-A)."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # profiling/client

from metrics_logger import (  # noqa: E402
    ENGINE_HEADER,
    metrics_row,
    metrics_url,
    parse_prometheus_metrics,
)

SAMPLE_METRICS = """\
# HELP vllm:num_requests_running Number of requests running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="x"} 16.0
vllm:num_requests_waiting{model_name="x"} 3.0
vllm:gpu_cache_usage_perc{model_name="x"} 0.42
vllm:prompt_tokens_total{model_name="x"} 12345.0
vllm:generation_tokens_total{model_name="x"} 67890.0
vllm:iteration_tokens_total_sum{model_name="x"} 5000.0
vllm:iteration_tokens_total_count{model_name="x"} 250.0
vllm:request_prefill_time_seconds_sum{model_name="x"} 1.5
vllm:request_decode_time_seconds_sum{model_name="x"} 9.0
"""


def test_metrics_row_mapping():
    parsed = parse_prometheus_metrics(SAMPLE_METRICS)
    row = metrics_row(parsed, t=1000.0)
    record = dict(zip(ENGINE_HEADER, row))
    assert record["timestamp"] == 1000.0
    assert record["num_requests_running"] == 16.0
    assert record["num_requests_waiting"] == 3.0
    assert record["gpu_cache_usage_perc"] == 0.42
    assert record["prompt_tokens_total"] == 12345.0
    assert record["generation_tokens_total"] == 67890.0
    assert record["iteration_tokens_total_sum"] == 5000.0
    assert record["iteration_tokens_total_count"] == 250.0
    assert record["request_prefill_time_seconds_sum"] == 1.5
    assert record["request_decode_time_seconds_sum"] == 9.0


def test_missing_metric_maps_to_nan():
    parsed = parse_prometheus_metrics(
        "vllm:num_requests_running 4.0\n"  # only one metric present
    )
    record = dict(zip(ENGINE_HEADER, metrics_row(parsed, t=0.0)))
    assert record["num_requests_running"] == 4.0
    assert math.isnan(record["generation_tokens_total"])


def test_header_starts_with_timestamp():
    assert ENGINE_HEADER[0] == "timestamp"
    # the state + token-counter fields the ledger needs are all present
    for col in ("num_requests_running", "prompt_tokens_total",
                "generation_tokens_total", "gpu_cache_usage_perc"):
        assert col in ENGINE_HEADER


def test_metrics_url_derivation():
    assert metrics_url("http://localhost:8000/v1") == "http://localhost:8000/metrics"
    assert metrics_url("http://h:9/v1/") == "http://h:9/metrics"
