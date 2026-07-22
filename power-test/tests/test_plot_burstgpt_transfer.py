"""
Claim:
Each BurstGPT panel applies only its own pre-request idle intercept to a frozen
request-generated trace and retains the complete common arbitrary-arrival
window for matched one-second per-GPU comparison.

Plausible wrong implementations:
- Pool idle measurements across the three replay strata.
- Use loaded-run power to calibrate the displayed prediction.
- Alter dynamic coefficients while describing an intercept-only update.
- Misidentify or silently omit one of the registered Fano strata.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "plot_burstgpt_transfer.py"
SPEC = importlib.util.spec_from_file_location("burstgpt_transfer", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_stratum_index_requires_registered_three_way_identity(index):
    revision = f"hash;window:1-2;fano-stratum:{index}/3"
    assert MODULE.stratum_index(revision) == index


def test_stratum_index_rejects_missing_or_different_partition():
    with pytest.raises(ValueError):
        MODULE.stratum_index("hash;window:1-2")
    with pytest.raises(ValueError):
        MODULE.stratum_index("hash;window:1-2;fano-stratum:0/4")
