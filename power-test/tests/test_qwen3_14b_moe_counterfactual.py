"""
Claim:
The Qwen3-14B counterfactual applies frozen MoE coefficients to an unchanged
dense request ledger with the A100 meter model, while marking every MoE result
unsupported across the architecture-family boundary.

Plausible wrong implementations:
- Apply the H100 trailing meter average to the A100 target.
- Refit or select coefficients using the opened target power.
- Label the cross-family candidate supported because its metrics improve.
- Hide an idle update inside the nominal zero-shot candidate.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "analyze_qwen3_14b_moe_counterfactual.py"
SPEC = importlib.util.spec_from_file_location("qwen14_moe", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_a100_source_law_has_no_h100_meter_smoothing():
    design = np.asarray([[1.0], [3.0], [5.0], [7.0]])
    prediction = MODULE.apply_source_law(
        design, [2.0], tp=2, hardware="A100", dt_s=0.25
    )
    np.testing.assert_allclose(prediction, [4.0, 12.0, 20.0, 28.0])


def test_cross_family_contract_never_claims_support():
    zero = MODULE.candidate_contract("gpt-oss-20b", False)
    idle = MODULE.candidate_contract("gpt-oss-20b", True)

    assert zero["surface_supported"] is False
    assert idle["surface_supported"] is False
    assert zero["coefficient_changes"] == []
    assert idle["coefficient_changes"] == ["pre-request target idle"]
    assert "unsupported" in zero["evidence_role"]
