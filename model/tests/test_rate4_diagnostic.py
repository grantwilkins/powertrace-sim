import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE = Path(__file__).resolve().parents[2] / "timing-test" / "rate4_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("rate4_diagnostic", MODULE)
rate4_diagnostic = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(rate4_diagnostic)


def test_overlap_concurrency_is_time_weighted():
    concurrency = rate4_diagnostic.overlap_concurrency(
        np.array([0.0, 1.0, 4.0]),
        np.array([2.0, 3.0, 4.0]),
    )
    assert concurrency[:2] == pytest.approx([1.5, 1.5])
    assert np.isnan(concurrency[2])


def test_condition_itls_counts_arrivals_before_interval_start():
    grouped = rate4_diagnostic.condition_itls(
        np.array([0.0, 0.15, 0.19]),
        np.array([0.10, 0.30, 0.40]),
        [np.array([0.02, 0.10]), np.array([0.01]), np.array([])],
        window_s=0.10,
    )
    assert grouped["0"] == pytest.approx([0.10, 0.01])
    assert grouped["1"] == pytest.approx([0.02])
    assert grouped["2+"] == pytest.approx([])
