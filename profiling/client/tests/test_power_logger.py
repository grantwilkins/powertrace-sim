"""Unit tests for the extended nvidia-smi power logger (CAMPAIGN.md §5-A)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # profiling/client
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from power_logger import DEFAULT_INTERVAL_MS, nvidia_smi_command  # noqa: E402
from model.training_data.power_parsing import parse_power_csv  # noqa: E402


def test_query_has_extended_fields():
    cmd = nvidia_smi_command()
    query = next(a for a in cmd if a.startswith("--query-gpu="))
    for field in ("clocks.sm", "clocks.mem", "utilization.memory", "temperature.gpu"):
        assert field in query
    # DVFS-critical: clocks.sm must be present
    assert "clocks.sm" in query
    assert f"-lms={DEFAULT_INTERVAL_MS}" in cmd
    # compatibility: timestamp + power.draw must lead the query
    assert query.split("=", 1)[1].startswith("timestamp,power.draw")


def _write_extended_power_csv(path, n_samples=3, gpus=8, watts=100.0):
    header = (
        "timestamp, power.draw [W], clocks.sm [MHz], clocks.mem [MHz], "
        "utilization.gpu [%], utilization.memory [%], memory.used [MiB], "
        "temperature.gpu"
    )
    lines = [header]
    for s in range(n_samples):
        ts = f"2026/06/15 18:28:{s:02d}.000"
        for _ in range(gpus):
            lines.append(f"{ts}, {watts:.2f}, 1980, 2619, 95, 40, 81000, 65")
    path.write_text("\n".join(lines) + "\n")


def test_header_parses_with_parse_power_csv(tmp_path):
    csv_path = tmp_path / "power.csv"
    _write_extended_power_csv(csv_path, n_samples=3, gpus=8, watts=100.0)
    parsed = parse_power_csv(str(csv_path), tensor_parallelism=8)
    assert parsed is not None
    # 8 GPUs x 100 W summed across the TP=8 group
    assert parsed["power"].shape[0] == 3
    assert all(abs(p - 800.0) < 1e-6 for p in parsed["power"])
