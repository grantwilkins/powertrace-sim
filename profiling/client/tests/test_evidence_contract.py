"""
Claim:
An admissible profiling bundle has synchronized complete per-GPU power and the
measured engine fields required by its named evidence profile.

Plausible wrong implementations:
- Accept mostly-NaN required counters because engine.csv has non-empty rows.
- Treat a reset/decrease in a cumulative counter as valid work.
- Combine anonymous or changing GPU identities into node power.
- Accept local-wall-time power and epoch engine streams with a whole-hour skew.
"""

import csv
from datetime import datetime, timezone

import pytest

from evidence_contract import validate_engine_csv, validate_streams
from metrics_logger import ENGINE_HEADER


def _write_engine(path, *, n=6, missing=None, decreasing=None, start=1_781_568_000.0):
    missing = missing or set()
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ENGINE_HEADER)
        writer.writeheader()
        for i in range(n):
            row = {name: float(i + 1) for name in ENGINE_HEADER}
            row["timestamp"] = start + 0.25 * i
            for name in missing:
                row[name] = float("nan")
            if decreasing and i == n - 1:
                row[decreasing] = 0.0
            writer.writerow(row)


def _write_power(path, *, start=1_781_568_000.0, gpus=2, n=6):
    fields = (
        "timestamp,index,uuid,power.draw [W],clocks.sm [MHz],clocks.mem [MHz],"
        "utilization.gpu [%],utilization.memory [%],memory.used [MiB],temperature.gpu"
    )
    lines = [fields]
    for sample in range(n):
        stamp = datetime.fromtimestamp(start + 0.25 * sample, timezone.utc)
        text = stamp.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
        for gpu in range(gpus):
            lines.append(f"{text},{gpu},GPU-{gpu},100,1200,1500,50,40,1000,60")
    path.write_text("\n".join(lines) + "\n")


def test_measured_ledger_profile_rejects_missing_required_state(tmp_path):
    path = tmp_path / "engine.csv"
    _write_engine(path, missing={"iteration_tokens_total_count"})
    with pytest.raises(ValueError, match="iteration_tokens_total_count coverage"):
        validate_engine_csv(path, "measured_ledger")


def test_measured_ledger_rejects_even_one_scrape_gap(tmp_path):
    path = tmp_path / "engine.csv"
    _write_engine(path)
    rows = path.read_text().splitlines()
    column = ENGINE_HEADER.index("prompt_tokens_total")
    cells = rows[3].split(",")
    cells[column] = "nan"
    rows[3] = ",".join(cells)
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="prompt_tokens_total coverage"):
        validate_engine_csv(path, "measured_ledger")


def test_counter_decrease_is_not_valid_work(tmp_path):
    path = tmp_path / "engine.csv"
    _write_engine(path, decreasing="generation_tokens_total")
    with pytest.raises(ValueError, match="counter generation_tokens_total decreased"):
        validate_engine_csv(path, "core")


def test_stream_validation_checks_epoch_alignment_and_gpu_identity(tmp_path):
    _write_engine(tmp_path / "engine.csv")
    _write_power(tmp_path / "power.csv")
    result = validate_streams(
        tmp_path, "measured_ledger", gpus_per_node=2, local_utc_offset_s=0.0
    )
    assert result["status"] == "validated"
    assert result["power"]["device_ids"] == ["GPU-0", "GPU-1"]
    assert abs(result["first_sample_skew_s"]) < 1e-9


def test_stream_validation_rejects_clock_basis_skew(tmp_path):
    _write_engine(tmp_path / "engine.csv", start=1_781_571_600.0)
    _write_power(tmp_path / "power.csv", start=1_781_568_000.0)
    with pytest.raises(ValueError, match="first-sample skew"):
        validate_streams(
            tmp_path, "measured_ledger", gpus_per_node=2, local_utc_offset_s=0.0
        )
