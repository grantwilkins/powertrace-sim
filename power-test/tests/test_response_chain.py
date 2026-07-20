"""Hand-worked checks of the delay, the H100 window, and run isolation."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from response_chain import apply_chain, apply_chain_by_run


def test_delay_shifts_exact_bins():
    out = apply_chain([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.25, "A100", 0.5)
    np.testing.assert_allclose(out, [1.0, 1.0, 1.0, 2.0, 3.0, 4.0])


def test_h100_window_is_four_bin_trailing_mean():
    out = apply_chain([4.0, 8.0, 12.0, 16.0, 20.0], 0.25, "H100", 0.0)
    # prefix: 4, (4+8)/2, (4+8+12)/3; full: (4+8+12+16)/4, (8+12+16+20)/4
    np.testing.assert_allclose(out, [4.0, 6.0, 8.0, 10.0, 14.0])


def test_runs_never_bleed():
    values = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
    run_id = [0, 0, 0, 1, 1, 1]
    out = apply_chain_by_run(values, run_id, 0.25, "A100", 0.25)
    np.testing.assert_allclose(out, [1.0, 1.0, 2.0, 10.0, 10.0, 20.0])
