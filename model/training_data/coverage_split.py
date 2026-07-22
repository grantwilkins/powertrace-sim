"""Small coverage-basis split for the 450-run legacy corpus."""
from __future__ import annotations

RATE_SET_A = frozenset((0.125, 0.5, 2.0))
RATE_SET_B = frozenset((0.25, 1.0, 4.0))
ALL_RATES = RATE_SET_A | RATE_SET_B

# Set B deliberately places rate 4 in training on both hardware types, dense
# and MoE models, BF16/MXFP4/FP8, and TP1/2/4/8.
TRAINING_RATE_SETS = {
    ("deepseek-r1-distill-8b", "A100", 1): RATE_SET_B,
    ("deepseek-r1-distill-8b", "A100", 4): RATE_SET_A,
    ("llama-3-70b", "A100", 4): RATE_SET_A,
    ("llama-3-70b", "A100", 8): RATE_SET_B,
    ("gpt-oss-20b", "A100", 1): RATE_SET_A,
    ("gpt-oss-20b", "A100", 2): RATE_SET_B,
    ("gpt-oss-120b", "A100", 4): RATE_SET_B,
    ("llama-3-8b", "H100", 1): RATE_SET_B,
    ("llama-3-8b", "H100", 2): RATE_SET_A,
    ("llama-3-70b", "H100", 4): RATE_SET_A,
    ("llama-3-70b", "H100", 8): RATE_SET_B,
    ("deepseek-r1-distill-70b", "H100", 4): RATE_SET_A,
    ("llama-3-405b", "H100", 8): RATE_SET_B,
}

HELDOUT_TP_SETUPS = frozenset({
    ("deepseek-r1-distill-8b", "A100", 2),
    ("deepseek-r1-distill-8b", "A100", 8),
    ("gpt-oss-120b", "A100", 8),
    ("llama-3-8b", "H100", 4),
    ("llama-3-8b", "H100", 8),
    ("deepseek-r1-distill-70b", "H100", 8),
})

HELDOUT_MODEL_SETUPS = frozenset({
    ("deepseek-r1-distill-70b", "A100", 4),
    ("deepseek-r1-distill-70b", "A100", 8),
    ("deepseek-r1-distill-8b", "H100", 1),
    ("deepseek-r1-distill-8b", "H100", 2),
    ("deepseek-r1-distill-8b", "H100", 4),
    ("deepseek-r1-distill-8b", "H100", 8),
})

ROLES = ("train", "heldout_rate", "heldout_tp", "heldout_model")


def setup_key(model: str, hardware: str, tp: int) -> tuple[str, str, int]:
    return str(model), str(hardware), int(tp)


def assign_role(model: str, hardware: str, tp: int, rate: float) -> str:
    """Assign a whole configuration/rate cell to exactly one role."""
    setup = setup_key(model, hardware, tp)
    rate = float(rate)
    if rate not in ALL_RATES:
        raise ValueError(f"Unknown request rate {rate:g}")
    if setup in HELDOUT_MODEL_SETUPS:
        return "heldout_model"
    if setup in HELDOUT_TP_SETUPS:
        return "heldout_tp"
    if setup not in TRAINING_RATE_SETS:
        raise ValueError(f"Setup is absent from the coverage split: {setup}")
    return "train" if rate in TRAINING_RATE_SETS[setup] else "heldout_rate"


def assign_runs(models, hardware, tp, rates) -> dict[int, str]:
    lengths = {len(models), len(hardware), len(tp), len(rates)}
    if len(lengths) != 1:
        raise ValueError("Run-level descriptor arrays must have equal length")
    return {
        rid: assign_role(models[rid], hardware[rid], tp[rid], rates[rid])
        for rid in range(len(models))
    }
