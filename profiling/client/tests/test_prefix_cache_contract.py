import csv

import pytest

from prefix_cache_contract import validate_prefix_cache_mode


def _engine(path, hits):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("timestamp", "prefix_cache_hits_total")
        )
        writer.writeheader()
        for index, value in enumerate(hits):
            writer.writerow({
                "timestamp": 1000.0 + index * 0.25,
                "prefix_cache_hits_total": value,
            })


def test_cache_off_rejects_hit_counter_increase(tmp_path):
    path = tmp_path / "engine.csv"
    _engine(path, [0, 0, 16])
    with pytest.raises(ValueError, match="cache-off run observed 16"):
        validate_prefix_cache_mode(path, False)


def test_cache_modes_require_matching_counter_evidence(tmp_path):
    path = tmp_path / "engine.csv"
    _engine(path, [0, 0, 0])
    assert validate_prefix_cache_mode(path, False)["status"] == "validated"
    with pytest.raises(ValueError, match="cache-on run observed no"):
        validate_prefix_cache_mode(path, True)

    _engine(path, [0, 16, 32])
    result = validate_prefix_cache_mode(path, True)
    assert result["hit_token_increase"] == 32
