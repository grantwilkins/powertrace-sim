"""
Claim:
Canonical bundle power samples preserve a stable UUID-to-index device mapping
while accepting the small per-GPU timestamp skew produced by one nvidia-smi poll.

Plausible wrong implementations:
- Group bundle rows by exact timestamps and reject one valid staggered sample.
- Treat either UUID or index as sufficient for a canonical bundle.
- Accept UUID/index remapping while every sample still has the declared GPU count.
- Merge measurements from distinct 4 Hz polls into one sample.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from model.training_data.power_parsing import (
    MAX_BUNDLE_SAMPLE_SKEW_S,
    parse_power_csv_per_gpu,
)


def _write_power(path: Path, rows, *, include_uuid: bool = True) -> None:
    header = ["timestamp", "index"]
    if include_uuid:
        header.append("uuid")
    header.append("power.draw")
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _rows(timestamp: str, index: int, uuid: str, power: float):
    return [timestamp, index, uuid, power]


def test_canonical_bundle_accepts_staggered_rows_and_keeps_index_order(tmp_path):
    path = tmp_path / "power.csv"
    _write_power(path, [
        _rows("2024/01/01 10:00:00.000", 0, "GPU-0", 10.0),
        _rows("2024/01/01 10:00:00.001", 1, "GPU-1", 20.0),
        _rows("2024/01/01 10:00:00.250", 1, "GPU-1", 40.0),
        _rows("2024/01/01 10:00:00.251", 0, "GPU-0", 30.0),
    ])

    parsed = parse_power_csv_per_gpu(str(path), gpus_per_node=2, strict_topology=True)

    assert parsed["device_ids"] == ("GPU-0", "GPU-1")
    np.testing.assert_array_equal(parsed["power_per_gpu"], [[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_allclose(np.diff(parsed["timestamps"]), [0.25])


def test_canonical_bundle_requires_uuid_and_index(tmp_path):
    path = tmp_path / "power.csv"
    _write_power(
        path,
        [
            ["2024/01/01 10:00:00.000", 0, 10.0],
            ["2024/01/01 10:00:00.250", 0, 11.0],
        ],
        include_uuid=False,
    )

    with pytest.raises(ValueError, match="both GPU index and UUID"):
        parse_power_csv_per_gpu(str(path), gpus_per_node=1, strict_topology=True)


def test_canonical_bundle_rejects_uuid_index_remapping(tmp_path):
    path = tmp_path / "power.csv"
    _write_power(path, [
        _rows("2024/01/01 10:00:00.000", 0, "GPU-0", 10.0),
        _rows("2024/01/01 10:00:00.001", 1, "GPU-1", 20.0),
        _rows("2024/01/01 10:00:00.250", 1, "GPU-0", 30.0),
        _rows("2024/01/01 10:00:00.251", 0, "GPU-1", 40.0),
    ])

    with pytest.raises(ValueError, match="multiple indices"):
        parse_power_csv_per_gpu(str(path), gpus_per_node=2, strict_topology=True)


def test_bundle_sample_skew_boundary_is_not_merged_with_next_poll(tmp_path):
    path = tmp_path / "power.csv"
    seconds = MAX_BUNDLE_SAMPLE_SKEW_S + 0.001
    _write_power(path, [
        _rows("2024/01/01 10:00:00.000", 0, "GPU-0", 10.0),
        _rows(f"2024/01/01 10:00:00.{int(seconds * 1000):03d}", 1, "GPU-1", 20.0),
    ])

    with pytest.raises(ValueError, match="incomplete device sets"):
        parse_power_csv_per_gpu(str(path), gpus_per_node=2, strict_topology=True)
