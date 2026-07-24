"""
Claim:
The model-wise fidelity table summarizes held-out supported rows by first
collapsing repeated runs within each model/hardware/TP/rate point, then
bootstrapping model medians over those points.

Plausible wrong implementations:
- Bootstrap or take medians over repeated traces, over-weighting cells with more repeats.
- Treat energy_error_pct as a fraction and multiply it by 100.
- Report nrmse_range as a fraction instead of a percent.
- Report raw Soft-DTW instead of square-root percentage temporal error.
- Include training or unsupported rows in a held-out table.
"""

import csv

import numpy as np

from coverage_model_fidelity_table import load_heldout_rows, summarize_by_model


def _row(role, supported, model="toy", tp=1, rate=1.0, nrmse=0.01,
         ks=0.8, energy=1.0, soft=0.01):
    return {
        "role": role,
        "surface_supported": supported,
        "model": model,
        "hardware": "H100",
        "tp": str(tp),
        "rate": str(rate),
        "nrmse_range": str(nrmse),
        "ks_agreement": str(ks),
        "energy_error_pct": str(energy),
        "soft_dtw_divergence": str(soft),
    }


def test_loader_keeps_only_supported_heldout_rows(tmp_path):
    path = tmp_path / "rows.csv"
    rows = [
        _row("train_source", "True"),
        _row("heldout_rate", "False"),
        _row("heldout_rate", "True"),
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    loaded = load_heldout_rows(path)

    assert len(loaded) == 1
    assert loaded[0]["model"] == "toy"


def test_model_summary_uses_point_medians_and_metric_units():
    rows = [
        *[
            {
                "model": "toy", "hardware": "H100", "tp": 1, "rate": 1.0,
                "nrmse_range": 0.01, "ks_agreement": 0.8,
                "energy_error_pct": 1.0, "soft_dtw_divergence": 0.01,
            }
            for _ in range(5)
        ],
        {
            "model": "toy", "hardware": "H100", "tp": 2, "rate": 1.0,
            "nrmse_range": 0.09, "ks_agreement": 0.6,
            "energy_error_pct": 9.0, "soft_dtw_divergence": 0.03,
        },
    ]

    summary = summarize_by_model(rows, draws=20, seed=1)[0]

    assert summary["traces"] == 6
    assert summary["points"] == 2
    assert summary["nrmse_range"]["median"] == 5.0
    assert summary["ks_agreement"]["median"] == 0.7
    assert summary["energy_error_pct"]["median"] == 5.0
    np.testing.assert_allclose(
        summary["temporal_error_pct"]["median"],
        np.median([100.0 * np.sqrt(0.01), 100.0 * np.sqrt(0.03)]),
    )
