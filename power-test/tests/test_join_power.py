"""Tests for power-test/join_power.py (DESIGN.md contract 1).

Hand-worked example: power starts at epoch E (2025-01-01 00:00:00 UTC), the
earliest validated request arrives at E + 10, so delta = 10.0 s and K = 0.
Sample times on the timing clock are p_ts - p_ts[0] - delta:
    E + 0.0  -> -10.0 s -> before the grid, dropped
    E + 10.1 ->   0.1 s -> bin 0
    E + 10.6 ->   0.6 s -> bin 2
    E + 11.4 ->   1.4 s -> bin 5
On a 6-bin, 0.25 s grid the joined power is [300, NaN, 320, NaN, NaN, 440]
(TP-2 sums of the two per-GPU readings at each kept sample).
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import join_power  # noqa: E402

E = datetime(2025, 1, 1, tzinfo=timezone.utc)
E_EPOCH = E.timestamp()


def write_power_csv(path, offsets_s, powers_per_gpu):
    """Raw per-GPU CSV: one (index 0, index 1) row pair per sample time."""
    lines = ["timestamp, index, power.draw [W]"]
    for off, (p0, p1) in zip(offsets_s, powers_per_gpu):
        ts = (E + timedelta(seconds=off)).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
        lines.append(f"{ts}, 0, {p0:.2f} W")
        lines.append(f"{ts}, 1, {p1:.2f} W")
    path.write_text("\n".join(lines) + "\n")


def write_requests_json(path, arrival_epochs):
    n = len(arrival_epochs)
    path.write_text(json.dumps({
        "input_lens": [5] * n,
        "output_lens": [1] * n,
        "ttfts": [0.5] * n,
        "itls": [[] for _ in range(n)],
        "request_timestamps": list(arrival_epochs),
    }))


def test_samples_land_in_right_bins_and_gaps_stay_nan(tmp_path):
    csv_path = tmp_path / "power.csv"
    json_path = tmp_path / "requests.json"
    write_power_csv(csv_path,
                    [0.0, 10.1, 10.6, 11.4],
                    [(100.0, 50.0), (200.0, 100.0),
                     (300.0, 20.0), (400.0, 40.0)])
    write_requests_json(json_path, [E_EPOCH + 10.0, E_EPOCH + 12.0])

    out = join_power.join_run(str(csv_path), str(json_path), tp=2,
                              n_bins=6, dt=0.25)
    assert out["delta_s"] == pytest.approx(10.0)
    assert out["k_fold"] == 0
    expected = np.array([300.0, np.nan, 320.0, np.nan, np.nan, 440.0])
    np.testing.assert_allclose(out["power"], expected)


def test_nonzero_fold_factor_raises(tmp_path):
    # Arrival 1810 s after power start: K = round(1810/1800) = 1, must raise.
    csv_path = tmp_path / "power.csv"
    json_path = tmp_path / "requests.json"
    write_power_csv(csv_path, [0.0, 1.0], [(100.0, 50.0), (200.0, 100.0)])
    write_requests_json(json_path, [E_EPOCH + 1810.0])
    with pytest.raises(ValueError, match="K=1"):
        join_power.join_run(str(csv_path), str(json_path), tp=2,
                            n_bins=4, dt=0.25)


def test_fold_gate_rejects_late_first_arrival(tmp_path):
    # Arrival 700 s after power start: K = 0 but 700 > 600 fails the gate.
    csv_path = tmp_path / "power.csv"
    json_path = tmp_path / "requests.json"
    write_power_csv(csv_path, [0.0, 1.0], [(100.0, 50.0), (200.0, 100.0)])
    write_requests_json(json_path, [E_EPOCH + 700.0])
    with pytest.raises(ValueError, match="gate"):
        join_power.join_run(str(csv_path), str(json_path), tp=2,
                            n_bins=4, dt=0.25)
