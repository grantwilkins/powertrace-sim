"""Unit tests for the extended nvidia-smi power logger (CAMPAIGN.md §5-A)."""

import csv
import io
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # profiling/client
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

import power_logger  # noqa: E402
from power_logger import (  # noqa: E402
    DEFAULT_INTERVAL_MS,
    TIMED_FIELDS,
    TP8_STATE_FIELDS,
    nvidia_smi_command,
    nvidia_smi_query_command,
    nvidia_smi_snapshot_command,
    write_query_rows,
)
from model.training_data.power_parsing import parse_power_csv  # noqa: E402


def test_query_has_extended_fields():
    cmd = nvidia_smi_query_command()
    query = next(a for a in cmd if a.startswith("--query-gpu="))
    for field in ("index", "uuid", "clocks.sm", "clocks.mem", "utilization.memory", "temperature.gpu"):
        assert field in query
    # DVFS-critical: clocks.sm must be present
    assert "clocks.sm" in query
    assert "--format=csv,nounits,noheader" in cmd
    # The wrapper stamps one timestamp per all-GPU query.
    assert query.split("=", 1)[1].startswith("index,uuid,power.draw")


def test_query_can_be_limited_to_role_gpu_uuids():
    command = nvidia_smi_query_command(gpu_ids="GPU-prefill,GPU-decode")
    assert command[-1] == "--id=GPU-prefill,GPU-decode"


def test_display_header_uses_canonical_clock_names():
    header = power_logger.display_header()
    assert "clocks.sm [MHz]" in header
    assert "clocks.mem [MHz]" in header
    assert all("clocks.current" not in field for field in header)


def test_logger_command_uses_python_wrapper():
    cmd = nvidia_smi_command()
    assert cmd[:2] == [sys.executable, str(Path(power_logger.__file__).resolve())]
    assert cmd[-4:] == [
        "--interval-ms", str(DEFAULT_INTERVAL_MS), "--profile", "core"
    ]


def test_write_query_rows_uses_one_timestamp_for_all_gpus(tmp_path):
    path = tmp_path / "power.csv"
    with path.open("w", newline="") as stream:
        stream.write(", ".join(power_logger.display_header()) + "\n")
        write_query_rows(
            stream,
            "2026/07/13 01:00:00.123",
            "0, GPU-0, 100, 1980, 2619, 95, 40, 81000, 65\n"
            "1, GPU-1, 101, 1980, 2619, 95, 40, 81000, 65\n",
        )
    rows = path.read_text().splitlines()
    assert rows[1].startswith("2026/07/13 01:00:00.123,0,GPU-0")
    assert rows[2].startswith("2026/07/13 01:00:00.123,1,GPU-1")


def test_timed_profile_uses_query_midpoint_without_changing_power(monkeypatch):
    times = iter((1_000_000_000, 1_200_000_000))
    stream = io.StringIO()

    def query(*_args, **_kwargs):
        power_logger._STOP = True
        return SimpleNamespace(
            stdout="0, GPU-0, 100, 1980, 2619, 95, 40, 81000, 65\n"
        )

    power_logger._STOP = False
    monkeypatch.setattr(power_logger.time, "time_ns", lambda: next(times))
    monkeypatch.setattr(power_logger.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(power_logger.subprocess, "run", query)
    monkeypatch.setattr(power_logger.sys, "stdout", stream)
    power_logger.stream_power(interval_ms=250, profile="core_timed")
    power_logger._STOP = False

    rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
    parse = lambda value: datetime.strptime(value, "%Y/%m/%d %H:%M:%S.%f")
    start = parse(rows[0]["query.start"])
    midpoint = parse(rows[0]["timestamp"])
    end = parse(rows[0]["query.end"])
    assert (midpoint - start).total_seconds() == 0.1
    assert (end - midpoint).total_seconds() == 0.1
    assert rows[0]["power.draw [W]"] == "100"
    query_fields = next(
        value for value in nvidia_smi_query_command(TIMED_FIELDS)
        if value.startswith("--query-gpu=")
    )
    assert "query.start" not in query_fields
    assert query_fields.split("=", 1)[1].startswith("index,uuid,power.draw")


def test_tp8_state_profile_requests_clock_cause_not_just_power():
    """Claim: the TP8 diagnostic records the state needed to identify the jump.

    This catches a plausible run that adds P-state but silently omits power caps,
    thermal slowdown, or hardware slowdown and therefore cannot distinguish the
    competing causes.
    """
    command = nvidia_smi_command(profile="tp8_state")
    assert command[-2:] == ["--profile", "tp8_state"]
    query = next(
        value for value in nvidia_smi_query_command(TP8_STATE_FIELDS)
        if value.startswith("--query-gpu=")
    )
    assert tuple(query.split("=", 1)[1].split(",")) == TP8_STATE_FIELDS[1:]
    for field in ("pstate", "power.limit", "clocks_event_reasons.sw_power_cap",
                  "clocks_event_reasons.hw_thermal_slowdown"):
        assert field in query
    snapshot = nvidia_smi_snapshot_command("tp8_state")
    assert "-lms=250" not in snapshot
    assert "--format=csv,noheader,nounits" in snapshot


def _write_extended_power_csv(path, n_samples=3, gpus=8, watts=100.0):
    header = (
        "timestamp, index, uuid, power.draw [W], clocks.sm [MHz], clocks.mem [MHz], "
        "utilization.gpu [%], utilization.memory [%], memory.used [MiB], "
        "temperature.gpu"
    )
    lines = [header]
    for s in range(n_samples):
        ts = f"2026/06/15 18:28:{s:02d}.000"
        for gpu in range(gpus):
            lines.append(
                f"{ts}, {gpu}, GPU-{gpu}, {watts:.2f}, 1980, 2619, 95, 40, 81000, 65"
            )
    path.write_text("\n".join(lines) + "\n")


def test_header_parses_with_parse_power_csv(tmp_path):
    csv_path = tmp_path / "power.csv"
    _write_extended_power_csv(csv_path, n_samples=3, gpus=8, watts=100.0)
    parsed = parse_power_csv(str(csv_path), tensor_parallelism=8)
    assert parsed is not None
    # 8 GPUs x 100 W summed across the TP=8 group
    assert parsed["power"].shape[0] == 3
    assert all(abs(p - 800.0) < 1e-6 for p in parsed["power"])
