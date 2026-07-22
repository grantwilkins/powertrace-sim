"""Transfer timing summaries must ignore invalid measurements, not predictions.

A wrong implementation could divide by zero or silently replace invalid target
latencies, changing the retrospective appendix metrics.
"""
from scripts.paper.transfer_core import _timing_summary


def test_timing_summary_filters_zero_measured_latency() -> None:
    rows = [
        {
            "measured_ttft_s": 0.0, "predicted_ttft_s": 9.0,
            "measured_decode_s": 2.0, "predicted_decode_s": 2.2,
            "measured_e2e_s": 2.0, "predicted_e2e_s": 2.2,
        },
        {
            "measured_ttft_s": 1.0, "predicted_ttft_s": 1.1,
            "measured_decode_s": 4.0, "predicted_decode_s": 4.4,
            "measured_e2e_s": 5.0, "predicted_e2e_s": 5.5,
        },
    ]

    report = _timing_summary(rows)

    assert round(report["ttft_s_medabs_pct"], 8) == 10.0
    assert round(report["decode_s_medabs_pct"], 8) == 10.0
    assert round(report["e2e_s_medabs_pct"], 8) == 10.0
