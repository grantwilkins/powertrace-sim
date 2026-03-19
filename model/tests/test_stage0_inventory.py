import csv
from pathlib import Path

from model.training_data.inventory import (
    discover_dataset_dirs,
    parse_dataset_dir_metadata,
)
from model.training_data.throughput import inspect_json_schema, inspect_power_csv


def test_parse_dataset_dir_metadata_valid():
    parsed = parse_dataset_dir_metadata(
        "sharegpt-benchmark-llama-3-8b-H100", "sharegpt-benchmark"
    )
    assert parsed == ("llama-3-8b", "H100")


def test_inspect_power_csv_valid(tmp_path: Path):
    csv_path = tmp_path / "power.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "power.draw [W]"])
        writer.writerow(["2025/01/01 00:00:00.000", "100.0 W"])
        writer.writerow(["2025/01/01 00:00:00.000", "101.0 W"])
        writer.writerow(["2025/01/01 00:00:00.250", "102.0 W"])
        writer.writerow(["2025/01/01 00:00:00.250", "103.0 W"])

    result = inspect_power_csv(str(csv_path))
    assert result["power_parseable"] is True
    assert "power_header" in result
    assert "rows_per_sample_estimate" in result
    assert int(result["sampled_rows"]) == 4


def test_inspect_json_schema_valid():
    payload = {
        "input_lens": [16, 32],
        "output_lens": [8, 12],
        "ttfts": [0.1, 0.2],
        "itls": [[0.01] * 8, [0.01] * 12],
        "request_timestamps": [1000.0, 1000.5],
    }
    result = inspect_json_schema(payload)
    assert result["aligned_request_count"] == 2
    assert result["itls_format"] == "list"
    assert result["mismatched_length_fields"] == []


def test_discover_dataset_dirs_filters_by_family(tmp_path: Path):
    (tmp_path / "sharegpt-benchmark-llama-3-8b-h100").mkdir()
    (tmp_path / "benchmark-gpt-oss-20b-h100").mkdir()
    (tmp_path / "other-family-foo").mkdir()

    discovered = discover_dataset_dirs(
        str(tmp_path), include_families=["sharegpt-benchmark"]
    )

    assert len(discovered) == 1
    assert discovered[0][0] == "sharegpt-benchmark"
    assert discovered[0][1].endswith("sharegpt-benchmark-llama-3-8b-h100")
