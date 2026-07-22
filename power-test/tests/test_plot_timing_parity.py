"""
Claim:
The timing figures place observed phase timing on x and model-predicted timing
on y, in milliseconds, with a common model-color mapping and a y=x reference.

Plausible wrong implementations:
- Swap observed and predicted axes while retaining a plausible parity cloud.
- Mix seconds and milliseconds or convert only one axis.
- Mix prefill and decode points in one panel.
- Use the same marker shape for A100 and H100 observations.
- Give the standalone prefill panel a legend or title needed only in decode.
"""
import matplotlib.pyplot as plt
import numpy as np
import plot_timing_parity as timing_plot

from plot_timing_parity import (
    draw_phase,
    prefill_probe_points,
    representative_run_ids,
    request_points,
    solo_request_indices,
)


def test_prefill_uses_queue_free_probe_points_with_hardware_identity(monkeypatch):
    calibration = {"rows": [{
        "hardware": "A100", "tp": 1, "model": "llama-3-8b",
        "probe": "prefill_staircase", "label": "prompt_8",
        "prompt_tokens": 8, "measured": {"median_ttft_ms": 20.0},
        "engine_counters": {"preemptions": 0},
    }]}
    fit = {
        "A100": {
            "base_overhead_s": 0.0, "per_message_s": {"1": 0.0},
            "first_token_overhead_s": 0.01, "eff_flops": 1.0,
            "eff_bw": 1.0,
        },
    }
    monkeypatch.setattr(timing_plot, "chunked_prefill_time_s", lambda *a, **k: 0.02)

    points = prefill_probe_points(calibration, fit)

    assert points == [{
        "phase": "prefill",
        "model": "llama-3-8b",
        "model_label": "Llama 3 8B",
        "hardware": "A100",
        "observed_ms": 20.0,
        "predicted_ms": 30.0,
    }]


def test_solo_prefill_excludes_requests_on_either_side_of_an_overlap():
    data = {
        "req_run_id": np.asarray([0, 0, 0]),
        "arrival_time_s": np.asarray([0.0, 5.0, 10.0]),
        "ttft_s": np.asarray([1.0, 1.0, 1.0]),
        "decode_duration_s": np.asarray([5.0, 1.0, 1.0]),
        "output_tokens": np.asarray([16, 16, 16]),
    }

    np.testing.assert_array_equal(solo_request_indices(data, 0), [2])


def test_request_points_preserve_phase_axis_meaning_and_units():
    points = request_points([{
        "meas_ttft_s": 0.002,
        "pred_ttft_s": 0.003,
        "meas_decode_duration_s": 0.4,
        "pred_decode_duration_s": 0.5,
    }], model="gpt-oss-20b", hardware="H100")

    assert points[0]["phase"] == "prefill"
    assert points[0]["observed_ms"] == 2.0
    assert points[0]["predicted_ms"] == 3.0
    assert points[1]["phase"] == "decode"
    assert points[1]["observed_ms"] == 400.0
    assert points[1]["predicted_ms"] == 500.0
    assert {point["hardware"] for point in points} == {"H100"}


def test_representative_runs_keep_one_repetition_per_evaluation_cell():
    data = {
        "run_hardware": np.asarray(["A100", "A100", "H100", "A100"]),
        "run_model": np.asarray(["llama-3-8b"] * 4),
        "run_tp": np.asarray([1, 1, 1, 1]),
        "run_rate": np.asarray([1.0, 1.0, 1.0, 1.0]),
    }
    roles = {0: "test_indomain", 1: "test_indomain",
             2: "test_indomain", 3: "train"}

    assert representative_run_ids(data, roles) == [0, 2]


def test_panels_use_y_equals_x_and_only_decode_has_a_legend():
    points = [
        {"phase": phase, "model": model, "model_label": label,
         "hardware": hardware, "observed_ms": observed,
         "predicted_ms": predicted}
        for phase, model, label, hardware, observed, predicted in (
            ("prefill", "llama-3-8b", "Llama 3 8B", "A100", 10.0, 12.0),
            ("decode", "llama-3-8b", "Llama 3 8B", "A100", 2.0, 3.0),
            ("decode", "gpt-oss-20b", "GPT-OSS 20B", "H100", 4.0, 5.0),
        )
    ]
    prefill_figure, prefill_axis = plt.subplots()
    decode_figure, decode_axis = plt.subplots()

    draw_phase(prefill_axis, points, "prefill", legend=False)
    draw_phase(decode_axis, points, "decode", legend=True)

    np.testing.assert_allclose(prefill_axis.lines[0].get_xdata(),
                               prefill_axis.lines[0].get_ydata())
    assert prefill_axis.get_title() == decode_axis.get_title() == ""
    assert prefill_axis.get_legend() is None
    legend_labels = {
        text.get_text() for text in decode_axis.get_legend().get_texts()
    }
    assert {"Llama 3 8B", "GPT-OSS 20B", "A100", "H100"} <= legend_labels
    plt.close(prefill_figure)
    plt.close(decode_figure)
