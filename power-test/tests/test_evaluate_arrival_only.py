"""
Claim:
Frozen arrival-only evaluation maps legacy references by exact source identity
and uses transfer M4A rows for the two held-out scale models while treating B2
only as a same-configuration S0 reference.

Plausible wrong implementations:
- Compare rows by model/rate and silently pair the wrong repeat.
- Use S0 M4A for a held-out model, erasing the transfer contract.
- Present B2 as a transfer baseline even though it is configuration-local.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_arrival_only import reference_split


def test_reference_contract_is_model_and_candidate_specific():
    gpt = {"hardware": "A100", "model": "gpt-oss-120b"}
    fp8 = {"hardware": "H100", "model": "llama-3-405b"}
    twin = {"hardware": "H100", "model": "deepseek-r1-distill-8b"}
    assert reference_split(gpt, "M4A") == "S1_A100_gpt_oss"
    assert reference_split(fp8, "M4A") == "S2b_H100_llama405"
    assert reference_split(twin, "M4A") == "S0_H100"
    assert reference_split(gpt, "B2") == "S0_A100"
    assert reference_split(fp8, "B2") == "S0_H100"
