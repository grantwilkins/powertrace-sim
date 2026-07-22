"""
Claim:
The coverage split trains on 117 of 450 runs while covering every measured
rate, hardware, TP degree, model class, and precision regime. Whole repeated
cells stay together, and the other runs test unseen rates, TPs, or models.

Plausible wrong implementations:
- Split repetitions from one configuration/rate cell across roles.
- Accidentally leave rate 4 or a TP degree out of training.
- Assign one setup to both TP and model holdouts.
- Drop or duplicate runs so the four roles no longer partition all 450.
"""
from collections import Counter, defaultdict

from model.training_data.coverage_split import (
    ALL_RATES,
    HELDOUT_MODEL_SETUPS,
    HELDOUT_TP_SETUPS,
    TRAINING_RATE_SETS,
    assign_role,
)


def full_grid():
    setups = set(TRAINING_RATE_SETS) | set(HELDOUT_TP_SETUPS) | set(HELDOUT_MODEL_SETUPS)
    return [
        (model, hardware, tp, rate, repeat)
        for model, hardware, tp in setups
        for rate in ALL_RATES
        for repeat in range(3)
    ]


def test_split_has_exact_counts_and_keeps_repeated_cells_together():
    rows = full_grid()
    roles = [assign_role(*row[:4]) for row in rows]
    assert Counter(roles) == {
        "train": 117,
        "heldout_rate": 117,
        "heldout_tp": 108,
        "heldout_model": 108,
    }
    by_cell = defaultdict(set)
    for row, role in zip(rows, roles):
        by_cell[row[:4]].add(role)
    assert all(len(values) == 1 for values in by_cell.values())


def test_training_basis_contains_every_rate_hardware_and_tp_degree():
    training = [row for row in full_grid() if assign_role(*row[:4]) == "train"]
    assert {row[1] for row in training} == {"A100", "H100"}
    for hardware in ("A100", "H100"):
        assert {row[2] for row in training if row[1] == hardware} == {1, 2, 4, 8}
    assert {row[3] for row in training} == set(ALL_RATES)
    assert any(row[3] == 4.0 and row[1] == "A100" for row in training)
    assert any(row[3] == 4.0 and row[1] == "H100" for row in training)


def test_setup_roles_are_disjoint():
    assert not (set(TRAINING_RATE_SETS) & set(HELDOUT_TP_SETUPS))
    assert not (set(TRAINING_RATE_SETS) & set(HELDOUT_MODEL_SETUPS))
    assert not (set(HELDOUT_TP_SETUPS) & set(HELDOUT_MODEL_SETUPS))
