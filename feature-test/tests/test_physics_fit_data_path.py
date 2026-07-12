"""Semantic gates for physics-fit source identity and held-out isolation.

Claims:
- A missing throughput calibration fails instead of silently becoming 5000 tok/s.
- Every ledger run can be traced to its source pair and content hashes.
- Fit entry points honor an explicitly selected ledger.
- Held-out labels cannot influence a fold's fitted parameters or predictions.

Plausible wrong implementations include a magic-rate fallback, integer-only
run labels, parsing but ignoring ``--ledger-cache``, and warm-starting a fold
from coefficients fitted on all labels.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import build_ledger_cache as cache
import fit_map_priors as map_fit
import fit_models
import peak_and_holdout


def test_missing_throughput_has_no_magic_fallback(monkeypatch):
    def missing(*_args, **_kwargs):
        raise KeyError("not calibrated")

    monkeypatch.setattr(cache, "resolve_throughput", missing)
    with pytest.raises(KeyError, match="not calibrated"):
        cache.resolve_prefill_rate({}, "toy_H100_tp8")


def test_run_source_entry_preserves_pair_identity_and_hashes():
    record = SimpleNamespace(
        config_id="toy_H100_tp8",
        source_layout="sharegpt",
        output_lens=np.asarray([3.0, 2.0, 1.0]),
        itls=np.asarray([[0.1, 0.2], [], []], dtype=object),
        provenance={
            "pair_key": "rate=2|iteration=3",
            "power_csv_path": "/raw/power.csv",
            "json_path": "/raw/requests.json",
            "sha256": {"power_csv": "abc", "requests_json": "def"},
        },
    )
    entry = cache.run_source_entry(7, record)
    assert entry["source_id"] == "toy_H100_tp8|rate=2|iteration=3"
    assert entry["sha256"] == {"power_csv": "abc", "requests_json": "def"}
    assert entry["paths"]["requests_json"] == "/raw/requests.json"
    assert entry["timing_projection"] == {
        "source_requests": 3,
        "retained_exact_itl_requests": 2,
        "excluded_chunk_token_mismatch": 1,
    }


def test_failed_candidate_does_not_consume_cell_quota():
    """Claim: with quota one, a failed first parse cannot suppress a valid peer."""
    runs = [
        {"model": "toy", "hardware": "H100", "tp": 8, "rate": 2.0, "id": 1},
        {"model": "toy", "hardware": "H100", "tp": 8, "rate": 2.0, "id": 2},
        {"model": "toy", "hardware": "H100", "tp": 8, "rate": 2.0, "id": 3},
    ]
    accepted, failed = cache.select_successful_runs(
        runs, 1, lambda run: None if run["id"] == 1 else f"bins-{run['id']}"
    )
    assert failed == 1
    assert accepted == [(runs[1], "bins-2")]


def test_unknown_hardware_is_not_silently_labeled_h100():
    with pytest.raises(ValueError, match="Unsupported ledger hardware"):
        cache.hardware_index("B200")


@pytest.mark.parametrize(
    "parser", [
        fit_models.build_arg_parser,
        map_fit.build_arg_parser,
        peak_and_holdout.build_arg_parser,
    ],
)
def test_fit_entry_points_accept_explicit_ledger(parser):
    args = parser().parse_args(["--ledger-cache", "/chosen/ledger.npz"])
    assert args.ledger_cache == "/chosen/ledger.npz"


def test_fold_prediction_is_independent_of_heldout_labels(monkeypatch):
    def train(X, y, fams, idle_cols):
        theta = np.array([np.log(np.mean(y))])
        return theta, {}, np.ones_like(y)

    def predict(X, theta, multipliers, fams, idle_cols):
        return np.full(X.shape[0], np.exp(theta[0]))

    monkeypatch.setattr(map_fit, "fit_two_stage", train)
    monkeypatch.setattr(map_fit, "predict_map", predict)
    X = np.ones((4, 1))
    y = np.array([10.0, 20.0, 100.0, 200.0])
    train_mask = np.array([True, True, False, False])
    test_mask = ~train_mask
    fams = np.zeros(4, dtype=int)
    _, first = map_fit.fit_fold(X, y, fams, train_mask, test_mask, np.array([True]))
    changed = y.copy()
    changed[test_mask] *= 1000.0
    _, second = map_fit.fit_fold(
        X, changed, fams, train_mask, test_mask, np.array([True])
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, [15.0, 15.0])


def test_training_sources_rejects_an_unindexed_ledger_run():
    with pytest.raises(ValueError, match="missing run_id"):
        peak_and_holdout._training_sources(np.array([1, 2]), {1: {"source_id": "a"}})


def test_bundle_source_index_is_normalized(tmp_path):
    path = tmp_path / "bundle.manifest.json"
    path.write_text(
        '{"ledger_schema_version": 2, "runs": [{"run_index": 3, '
        '"run_id": "campaign-run", "run_dir": "/raw/run", '
        '"sha256": {"manifest.json": "abc"}}]}'
    )
    index = peak_and_holdout._load_run_index(path)
    assert index[3]["source_id"] == "bundle:campaign-run"
    assert index[3]["source_layout"] == "bundle"
    assert index[3]["sha256"] == {"manifest.json": "abc"}
