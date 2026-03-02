"""
Tests for scripts/eval/aggregation_resolution.py.
"""

import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/eval"))

from aggregation_resolution import compute_resolution_par  # noqa: E402


def _downsample_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    assert x.size % factor == 0
    return np.mean(x.reshape(-1, factor), axis=1)


def _write_base_fixture(
    aggregated_dir: Path,
    *,
    custom_site_15min: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    site_1s = np.full((3600,), 100.0, dtype=np.float64)
    site_1s[123] = 400.0
    site_250ms = np.repeat(site_1s, 4).astype(np.float64)
    site_1min = _downsample_mean(site_1s, 60)
    if custom_site_15min is None:
        site_15min = _downsample_mean(site_1s, 900)
    else:
        site_15min = np.asarray(custom_site_15min, dtype=np.float64).reshape(-1)

    np.save(aggregated_dir / "site_250ms.npy", np.asarray(site_250ms, dtype=np.float32))
    np.save(aggregated_dir / "site_1s.npy", np.asarray(site_1s, dtype=np.float32))
    np.save(aggregated_dir / "site_1min.npy", np.asarray(site_1min, dtype=np.float32))
    np.save(aggregated_dir / "site_15min.npy", np.asarray(site_15min, dtype=np.float32))
    return {
        "par_1s": float(np.max(site_1s) / np.mean(site_1s)),
        "par_15min": float(np.max(site_15min) / np.mean(site_15min)),
    }


def test_resolution_par_outputs_and_monotonicity():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        aggregated_dir = root / "aggregated"
        expected = _write_base_fixture(aggregated_dir)

        out_plot = root / "par_plot.pdf"
        out_csv = root / "par_rows.csv"
        out_json = root / "par_summary.json"
        result = compute_resolution_par(
            aggregated_dir=str(aggregated_dir),
            out_plot=str(out_plot),
            out_csv=str(out_csv),
            out_json=str(out_json),
            resolutions=(0.25, 1.0, 10.0, 60.0, 300.0, 900.0),
            allow_nonmonotonic=False,
        )

        assert result["status"] == "ok"
        assert bool(result["monotonic_nonincreasing"]) is True
        assert out_plot.exists()
        assert out_csv.exists()
        assert out_json.exists()

        with open(out_csv, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6

        by_res = {float(r["resolution_s"]): r for r in rows}
        assert np.isclose(float(by_res[1.0]["par"]), expected["par_1s"])
        assert np.isclose(float(by_res[900.0]["par"]), expected["par_15min"])

        pars = [float(r["par"]) for r in sorted(rows, key=lambda r: float(r["resolution_s"]))]
        for i in range(1, len(pars)):
            assert pars[i] <= pars[i - 1] + 1e-12


def test_nonmonotonic_rejected_unless_allowed():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        aggregated_dir = root / "aggregated"
        _write_base_fixture(
            aggregated_dir,
            custom_site_15min=np.asarray([500.0, 100.0, 100.0, 100.0], dtype=np.float64),
        )

        out_plot = root / "par_plot.pdf"
        out_csv = root / "par_rows.csv"
        out_json = root / "par_summary.json"

        with pytest.raises(ValueError, match="non-monotonic"):
            compute_resolution_par(
                aggregated_dir=str(aggregated_dir),
                out_plot=str(out_plot),
                out_csv=str(out_csv),
                out_json=str(out_json),
                resolutions=(0.25, 1.0, 10.0, 60.0, 300.0, 900.0),
                allow_nonmonotonic=False,
            )

        result = compute_resolution_par(
            aggregated_dir=str(aggregated_dir),
            out_plot=str(out_plot),
            out_csv=str(out_csv),
            out_json=str(out_json),
            resolutions=(0.25, 1.0, 10.0, 60.0, 300.0, 900.0),
            allow_nonmonotonic=True,
        )
        assert bool(result["monotonic_nonincreasing"]) is False
        assert len(result["violations"]) > 0
