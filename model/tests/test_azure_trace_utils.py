"""
Tests for Azure trace utilities.
"""

import csv
import os
import tempfile
from datetime import datetime, timezone

import pytest

# Add scripts/eval to path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/eval"))

from azure_trace_utils import Request, compute_trace_statistics, load_and_parse_azure_csv


class TestLoadAndParseAzureCSV:
    """Tests for load_and_parse_azure_csv function."""

    def create_temp_csv(self, rows):
        """Helper to create a temporary CSV file."""
        f = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
        writer = csv.DictWriter(f, fieldnames=["TIMESTAMP", "ContextTokens", "GeneratedTokens"])
        writer.writeheader()
        writer.writerows(rows)
        f.close()
        return f.name

    def test_load_valid_trace(self):
        """Test loading a valid Azure trace CSV."""
        rows = [
            {
                "TIMESTAMP": "2024-05-16 07:00:00.000000+00:00",
                "ContextTokens": "100",
                "GeneratedTokens": "50",
            },
            {
                "TIMESTAMP": "2024-05-16 07:00:01.000000+00:00",
                "ContextTokens": "200",
                "GeneratedTokens": "75",
            },
            {
                "TIMESTAMP": "2024-05-16 07:00:02.500000+00:00",
                "ContextTokens": "150",
                "GeneratedTokens": "100",
            },
        ]

        csv_path = self.create_temp_csv(rows)
        try:
            requests = load_and_parse_azure_csv(csv_path, scale_factor=1.0, seed=42)

            assert len(requests) == 3
            assert requests[0].request_id == 0
            assert requests[0].arrival_time == 0.0
            assert requests[0].input_tokens == 100
            assert requests[0].output_tokens == 50

            assert requests[1].arrival_time == 1.0
            assert requests[1].input_tokens == 200
            assert requests[1].output_tokens == 75

            assert requests[2].arrival_time == 2.5
            assert requests[2].input_tokens == 150
            assert requests[2].output_tokens == 100
        finally:
            os.unlink(csv_path)

    def test_load_trace_without_microseconds(self):
        """Test loading trace with timestamps without microseconds."""
        rows = [
            {
                "TIMESTAMP": "2024-05-16 07:00:00+00:00",
                "ContextTokens": "100",
                "GeneratedTokens": "50",
            },
            {
                "TIMESTAMP": "2024-05-16 07:00:05+00:00",
                "ContextTokens": "200",
                "GeneratedTokens": "75",
            },
        ]

        csv_path = self.create_temp_csv(rows)
        try:
            requests = load_and_parse_azure_csv(csv_path)

            assert len(requests) == 2
            assert requests[0].arrival_time == 0.0
            assert requests[1].arrival_time == 5.0
        finally:
            os.unlink(csv_path)

    def test_scale_factor_compression(self):
        """Test that scale_factor compresses time correctly."""
        rows = [
            {
                "TIMESTAMP": "2024-05-16 07:00:00+00:00",
                "ContextTokens": "100",
                "GeneratedTokens": "50",
            },
            {
                "TIMESTAMP": "2024-05-16 07:00:10+00:00",
                "ContextTokens": "200",
                "GeneratedTokens": "75",
            },
        ]

        csv_path = self.create_temp_csv(rows)
        try:
            # Scale by 2x should compress 10s to 5s
            requests = load_and_parse_azure_csv(csv_path, scale_factor=2.0)

            assert len(requests) == 2
            assert requests[0].arrival_time == 0.0
            assert requests[1].arrival_time == pytest.approx(5.0)
        finally:
            os.unlink(csv_path)

    def test_scale_factor_expansion(self):
        """Test that scale_factor < 1.0 expands time."""
        rows = [
            {
                "TIMESTAMP": "2024-05-16 07:00:00+00:00",
                "ContextTokens": "100",
                "GeneratedTokens": "50",
            },
            {
                "TIMESTAMP": "2024-05-16 07:00:10+00:00",
                "ContextTokens": "200",
                "GeneratedTokens": "75",
            },
        ]

        csv_path = self.create_temp_csv(rows)
        try:
            # Scale by 0.5x should expand 10s to 20s
            requests = load_and_parse_azure_csv(csv_path, scale_factor=0.5)

            assert len(requests) == 2
            assert requests[0].arrival_time == 0.0
            assert requests[1].arrival_time == pytest.approx(20.0)
        finally:
            os.unlink(csv_path)

    def test_requests_sorted_by_time(self):
        """Test that requests are sorted by arrival time."""
        # Insert out of order
        rows = [
            {
                "TIMESTAMP": "2024-05-16 07:00:05+00:00",
                "ContextTokens": "300",
                "GeneratedTokens": "50",
            },
            {
                "TIMESTAMP": "2024-05-16 07:00:00+00:00",
                "ContextTokens": "100",
                "GeneratedTokens": "75",
            },
            {
                "TIMESTAMP": "2024-05-16 07:00:03+00:00",
                "ContextTokens": "200",
                "GeneratedTokens": "100",
            },
        ]

        csv_path = self.create_temp_csv(rows)
        try:
            requests = load_and_parse_azure_csv(csv_path)

            # Should be sorted with request_ids reassigned
            assert len(requests) == 3
            assert requests[0].arrival_time == 0.0
            assert requests[0].input_tokens == 100  # Originally second row

            assert requests[1].arrival_time == 3.0
            assert requests[1].input_tokens == 200  # Originally third row

            assert requests[2].arrival_time == 5.0
            assert requests[2].input_tokens == 300  # Originally first row
        finally:
            os.unlink(csv_path)

    def test_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Azure trace file not found"):
            load_and_parse_azure_csv("/nonexistent/path.csv")

    def test_invalid_scale_factor(self):
        """Test error handling for invalid scale factor."""
        rows = [
            {
                "TIMESTAMP": "2024-05-16 07:00:00+00:00",
                "ContextTokens": "100",
                "GeneratedTokens": "50",
            }
        ]
        csv_path = self.create_temp_csv(rows)
        try:
            with pytest.raises(ValueError, match="scale_factor must be positive"):
                load_and_parse_azure_csv(csv_path, scale_factor=0.0)

            with pytest.raises(ValueError, match="scale_factor must be positive"):
                load_and_parse_azure_csv(csv_path, scale_factor=-1.0)
        finally:
            os.unlink(csv_path)

    def test_missing_columns(self):
        """Test error handling for CSV with missing columns."""
        f = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
        writer = csv.DictWriter(f, fieldnames=["TIMESTAMP", "ContextTokens"])
        writer.writeheader()
        writer.writerow({"TIMESTAMP": "2024-05-16 07:00:00+00:00", "ContextTokens": "100"})
        f.close()

        try:
            with pytest.raises(ValueError, match="CSV missing required columns"):
                load_and_parse_azure_csv(f.name)
        finally:
            os.unlink(f.name)

    def test_empty_csv(self):
        """Test error handling for empty CSV."""
        f = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
        writer = csv.DictWriter(f, fieldnames=["TIMESTAMP", "ContextTokens", "GeneratedTokens"])
        writer.writeheader()
        f.close()

        try:
            with pytest.raises(ValueError, match="Azure trace CSV is empty"):
                load_and_parse_azure_csv(f.name)
        finally:
            os.unlink(f.name)

    def test_invalid_token_values(self):
        """Test error handling for non-integer token values."""
        rows = [
            {
                "TIMESTAMP": "2024-05-16 07:00:00+00:00",
                "ContextTokens": "not_a_number",
                "GeneratedTokens": "50",
            }
        ]
        csv_path = self.create_temp_csv(rows)
        try:
            with pytest.raises(ValueError, match="Error parsing row"):
                load_and_parse_azure_csv(csv_path)
        finally:
            os.unlink(csv_path)


class TestComputeTraceStatistics:
    """Tests for compute_trace_statistics function."""

    def test_basic_statistics(self):
        """Test basic statistics computation."""
        requests = [
            Request(request_id=0, arrival_time=0.0, input_tokens=100, output_tokens=50),
            Request(request_id=1, arrival_time=1.0, input_tokens=200, output_tokens=75),
            Request(request_id=2, arrival_time=3.0, input_tokens=150, output_tokens=100),
        ]

        stats = compute_trace_statistics(requests)

        assert stats["num_requests"] == 3
        assert stats["duration_s"] == pytest.approx(3.0)
        assert stats["avg_rate_req_per_s"] == pytest.approx(1.0)  # 3 requests / 3 seconds
        assert stats["avg_input_tokens"] == pytest.approx(150.0)  # (100 + 200 + 150) / 3
        assert stats["avg_output_tokens"] == pytest.approx(75.0)  # (50 + 75 + 100) / 3

    def test_interarrival_percentiles(self):
        """Test inter-arrival time percentile computation."""
        # Create requests with known inter-arrival times: 1s, 2s, 3s, 4s
        requests = [
            Request(request_id=0, arrival_time=0.0, input_tokens=100, output_tokens=50),
            Request(request_id=1, arrival_time=1.0, input_tokens=100, output_tokens=50),
            Request(request_id=2, arrival_time=3.0, input_tokens=100, output_tokens=50),
            Request(request_id=3, arrival_time=6.0, input_tokens=100, output_tokens=50),
            Request(request_id=4, arrival_time=10.0, input_tokens=100, output_tokens=50),
        ]

        stats = compute_trace_statistics(requests)

        # Inter-arrivals: 1000ms, 2000ms, 3000ms, 4000ms
        assert stats["p50_interarrival_ms"] == pytest.approx(2500.0)  # Median of [1000, 2000, 3000, 4000]

    def test_empty_trace(self):
        """Test statistics for empty trace."""
        stats = compute_trace_statistics([])

        assert stats["num_requests"] == 0
        assert stats["duration_s"] == 0.0
        assert stats["avg_rate_req_per_s"] == 0.0
        assert stats["avg_input_tokens"] == 0.0
        assert stats["avg_output_tokens"] == 0.0
        assert stats["p50_interarrival_ms"] == 0.0

    def test_single_request(self):
        """Test statistics for single request."""
        requests = [
            Request(request_id=0, arrival_time=0.0, input_tokens=100, output_tokens=50)
        ]

        stats = compute_trace_statistics(requests)

        assert stats["num_requests"] == 1
        assert stats["duration_s"] == 0.0
        assert stats["avg_rate_req_per_s"] == 0.0  # Avoid division by zero
        assert stats["avg_input_tokens"] == 100.0
        assert stats["avg_output_tokens"] == 50.0
