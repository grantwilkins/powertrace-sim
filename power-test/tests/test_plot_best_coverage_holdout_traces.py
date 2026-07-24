"""
Claim:
The showcase uses one held-out configuration that covers every requested rate,
then selects a representative repetition separately within each rate.

Plausible wrong implementations:
- Pick a different model or TP at each rate.
- Let an incomplete configuration win because its available trace is excellent.
- Minimize temporal similarity instead of maximizing it.
- Cherry-pick the single lowest-error repetition rather than the metric medoid.
"""

from plot_best_coverage_holdout_traces import (
    FIGSIZE,
    FONT_SCALE,
    LEGEND_ANCHOR,
    MEASURED_LINEWIDTH,
    PREDICTED_LABEL,
    PREDICTED_LINEWIDTH,
    SEABORN_CONTEXT,
    Y_LABEL,
    select_configuration,
    select_representative_rows,
)


RATES = (0.125, 1.0, 2.0, 4.0)


def _configuration(hardware, model, tp, energy, rmse, acf, rates=RATES):
    return [{
        "run_id": run_id,
        "hardware": hardware,
        "model": model,
        "tp": tp,
        "rate": rate,
        "energy_error_pct": energy,
        "rmse_w_per_gpu": rmse,
        "acf_r2": acf,
    } for run_id, rate in enumerate(rates)]


def test_selection_uses_one_complete_configuration_and_metric_directions():
    rows = (
        _configuration("H100", "two-metric-winner", 2, 1.0, 1.0, 0.8)
        + _configuration("H100", "acf-winner", 4, 2.0, 2.0, 0.99)
        + _configuration(
            "A100", "incomplete", 1, 0.0, 0.0, 1.0, rates=(0.125,)
        )
    )

    selected = select_configuration(rows)

    assert selected["model"] == "two-metric-winner"
    assert selected["runs"] == 4
    assert selected["ranks"] == {
        "energy_error_pct": 1,
        "rmse_w_per_gpu": 1,
        "acf_r2": 2,
    }


def test_repetition_selection_keeps_all_rates_and_uses_metric_medoid():
    rows = []
    for rate in RATES:
        rows.extend([
            {"run_id": int(rate * 1000) + 1, "hardware": "H100",
             "model": "model", "tp": 2, "rate": rate,
             "energy_error_pct": 0.0, "rmse_w_per_gpu": 0.0, "acf_r2": 1.0},
            {"run_id": int(rate * 1000) + 2, "hardware": "H100",
             "model": "model", "tp": 2, "rate": rate,
             "energy_error_pct": 2.0, "rmse_w_per_gpu": 2.0, "acf_r2": 0.8},
            {"run_id": int(rate * 1000) + 3, "hardware": "H100",
             "model": "model", "tp": 2, "rate": rate,
             "energy_error_pct": 100.0, "rmse_w_per_gpu": 100.0,
             "acf_r2": -1.0},
        ])

    selected = select_representative_rows(
        rows, {"hardware": "H100", "model": "model", "tp": 2}
    )

    assert [row["rate"] for row in selected] == list(RATES)
    assert [row["run_id"] for row in selected] == [
        int(rate * 1000) + 2 for rate in RATES
    ]


def test_plot_presentation_matches_paper_style_request():
    assert FIGSIZE == (11.0, 4.0)
    assert PREDICTED_LABEL == "Our Simulator"
    assert LEGEND_ANCHOR[1] < 0.0
    assert SEABORN_CONTEXT == "talk"
    assert FONT_SCALE == 1.2
    assert MEASURED_LINEWIDTH > 1.0
    assert PREDICTED_LINEWIDTH > MEASURED_LINEWIDTH
    assert Y_LABEL == "Power (W/GPU)"
