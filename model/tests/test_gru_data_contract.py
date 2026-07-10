"""
Claim:
The GRU dataset is fit only on its training split, evaluation features and
targets refer to the same sampled time, standalone schedules are strict but
retain prefill-only requests, and projected traces retain compact source
lineage and row-drop accounting. Explicit canonical bundles reach the same
dataset builder without filesystem auto-discovery.

Plausible wrong implementations:
- Compute normalization or clamp extrema before splitting, leaking test power.
- Generate evaluation features from t=0 while comparing them with power at t=dt.
- Treat a zero-token decode as a failed request and discard its prefill work.
- Silently clamp negative tokens or skip malformed standalone schedule rows.
- Save pair labels without source paths, hashes, or projection/drop counts.
- Accept bundle CLI inputs but never route them into a written GRU dataset.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from model.classifiers.features import build_features_from_active
from model.pipeline.evaluation import _build_evaluation_rollout_features
from model.pipeline.evaluation import _request_json_from_lineage
from model.utils.provenance import sha256_file
from model.pipeline.request_builder import load_request_schedule
from model.training_data.manifest import run_prepare_experimental_manifest
from model.training_data.alignment import (
    align_trace_to_grid,
    compute_active_requests,
    resample_trace_to_grid,
)


def _make_projected_manifest(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    traces = []
    powers = ([10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0], [900.0, 901.0, 902.0])
    for idx, power in enumerate(powers):
        power_path = source / f"power-{idx}.csv"
        request_path = source / f"requests-{idx}.json"
        power_path.write_text(f"power-{idx}")
        request_path.write_text(f"requests-{idx}")
        pair_key = f"run-{idx}"
        rows.append(
            {
                "status": "matched",
                "model_name": "toy",
                "hardware": "H100",
                "tensor_parallelism": "1",
                "rate": str(idx),
                "pair_key": pair_key,
                "power_csv_path": str(power_path),
                "json_path": str(request_path),
            }
        )
        traces.append(
            {
                "power": np.asarray(power, dtype=np.float64),
                "active_requests": np.asarray([0.0, 1.0, 0.0]),
                "t_arrive_log": np.zeros(3),
                "dt": 0.25,
                "power_start_epoch_s": 1000.0 + idx,
                "input_lens": np.asarray([100.0]),
                "output_lens": np.asarray([10.0]),
                "ttfts": np.asarray([0.5]),
                "decode_times": np.asarray([0.2]),
            }
        )

    pair_manifest = tmp_path / "pair_manifest.csv"
    with pair_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    records = {}
    for idx, row in enumerate(rows):
        records[row["pair_key"]] = SimpleNamespace(
            view=traces[idx],
            source_layout="sharegpt",
            provenance={
                "pair_key": row["pair_key"],
                "power_csv_path": row["power_csv_path"],
                "json_path": row["json_path"],
                "sha256": {
                    "power_csv": f"power-sha-{idx}",
                    "requests_json": f"request-sha-{idx}",
                },
                "request_rows": {
                    "source": 3,
                    "aligned": 3,
                    "retained": 2,
                    "dropped": {"invalid_timing": 1},
                },
                "request_projection_indices": [0, 2],
                "request_projection": {
                    "num_requests_raw": 3,
                    "num_requests_aligned": 3,
                    "num_requests_used": 2,
                    "num_requests_dropped_invalid_fields": 0,
                    "num_requests_dropped_decode_time": 1,
                    "num_requests_dropped_timestamp": 0,
                },
            },
        )

    out_dir = tmp_path / "prepared"
    with (
        patch(
            "model.training_data.manifest.load_legacy_run",
            side_effect=lambda row, **_: records[row["pair_key"]],
        ),
        patch(
            "model.training_data.manifest.gru_view_from_record",
            side_effect=lambda record: record.view,
        ),
    ):
        manifest = run_prepare_experimental_manifest(
            pair_manifest_csv=str(pair_manifest),
            out_dir=str(out_dir),
            train_ratio=0.5,
            val_ratio=0.25,
            seed=7,
            min_traces_per_config=2,
        )
    return manifest, traces


def test_manifest_norm_and_clamp_use_training_indices_only(tmp_path):
    manifest, traces = _make_projected_manifest(tmp_path)
    config = manifest["configs"]["toy_H100_tp1"]
    split = json.loads(Path(config["split_json"]).read_text())
    norm = json.loads(Path(config["norm_params_json"]).read_text())

    train_power = np.concatenate([traces[i]["power"] for i in split["train_indices"]])
    assert norm["fit_split"] == "train"
    assert norm["power_mean"] == pytest.approx(float(np.mean(train_power)))
    assert norm["power_min"] == pytest.approx(float(np.min(train_power)))
    assert norm["power_max"] == pytest.approx(float(np.max(train_power)))


def test_manifest_rejects_mixed_trace_timesteps(tmp_path):
    with patch(
        "model.training_data.manifest._load_pair_manifest_csv", return_value=[]
    ), patch(
        "model.training_data.manifest.load_bundle_run"
    ) as load_bundle, patch(
        "model.training_data.manifest.gru_view_from_record"
    ) as view, patch(
        "model.training_data.manifest._lineage_entry", return_value={}
    ):
        records = [
            SimpleNamespace(config_id="toy_H100_tp1", provenance={"run_id": str(i)})
            for i in range(3)
        ]
        load_bundle.side_effect = records
        view.side_effect = [
            {
                "power": np.asarray([1.0, 2.0, 3.0]),
                "active_requests": np.zeros(3),
                "t_arrive_log": np.zeros(3),
                "dt": dt,
                "power_start_epoch_s": 0.0,
                "input_lens": np.asarray([100.0]),
                "output_lens": np.asarray([10.0]),
                "ttfts": np.asarray([0.5]),
                "decode_times": np.asarray([0.2]),
            }
            for dt in (0.25, 0.25, 1.0)
        ]
        with pytest.raises(ValueError, match="incompatible sampling intervals"):
            run_prepare_experimental_manifest(
                pair_manifest_csv=str(tmp_path / "unused.csv"),
                bundle_dirs=["a", "b", "c"],
                out_dir=str(tmp_path / "out"),
            )


def test_lineage_sidecar_recovers_hashed_sources_and_drop_counts(tmp_path):
    manifest, _ = _make_projected_manifest(tmp_path)
    config = manifest["configs"]["toy_H100_tp1"]
    lineage = json.loads(Path(config["lineage_json"]).read_text())

    assert lineage["projection"]["stored_fields"] == [
        "power",
        "active_requests",
        "t_arrive_log",
        "input_lens",
        "output_lens",
        "ttfts",
        "decode_times",
    ]
    assert len(lineage["traces"]) == 4
    first = lineage["traces"][0]
    assert Path(first["source_paths"]["power_csv"]).read_text() == "power-0"
    assert first["source_sha256"]["requests_json"] == "request-sha-0"
    assert first["request_rows"] == {
        "source": 3,
        "aligned": 3,
        "retained": 2,
        "dropped": {"invalid_timing": 1},
    }
    assert first["request_projection_indices"] == [0, 2]
    assert first["request_projection"]["num_requests_dropped_decode_time"] == 1


def test_off_grid_arrival_has_same_training_and_evaluation_time_axis():
    dt = 0.25
    norm = {
        "active_mean": 0.0,
        "active_std": 1.0,
        "t_arrive_log_mean": 0.0,
        "t_arrive_log_std": 1.0,
        "delta_A_mean": 0.0,
        "delta_A_std": 1.0,
    }
    requests = [{"arrival_time": 0.10, "input_tokens": 1.0, "output_tokens": 0.0}]
    # The request is active at sampled times 0.25 and 0.50, then complete by 0.75.
    measured_active = np.asarray([0.0, 1.0, 1.0, 0.0])
    training = build_features_from_active(measured_active, None, norm, max_length=3)
    evaluation = _build_evaluation_rollout_features(
        requests=requests,
        throughput={"lambda_prefill": 2.0, "lambda_decode": 10.0},
        norm=norm,
        num_points=3,
        dt=dt,
        feature_set="f2",
    )
    np.testing.assert_array_equal(
        evaluation["features_norm"], training["features_norm"]
    )


def test_request_is_inactive_at_exact_completion_boundary():
    active = compute_active_requests(
        np.asarray([0.0, 0.25, 0.5]),
        np.asarray([0.0]),
        np.asarray([0.25]),
        np.asarray([0.25]),
    )
    np.testing.assert_array_equal(active, [1.0, 1.0, 0.0])


def test_single_missing_power_sample_is_resampled_on_the_median_cadence():
    aligned = align_trace_to_grid(
        {
            "timestamps": np.asarray([0.0, 0.25, 0.75, 1.0]),
            "power": np.asarray([0.0, 10.0, 30.0, 40.0]),
        },
        {
            "request_timestamps": np.asarray([]),
            "ttfts": np.asarray([]),
            "decode_times": np.asarray([]),
            "input_lens": np.asarray([]),
            "output_lens": np.asarray([]),
            "has_timestamps": True,
        },
    )

    assert aligned is not None
    np.testing.assert_array_equal(aligned["timestamps"], [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_array_equal(aligned["power"], [0.0, 10.0, 20.0, 30.0, 40.0])
    assert aligned["power_resampled_to_dt"] is True


def test_three_missing_power_samples_are_resampled_but_a_longer_hole_is_rejected():
    request_data = {
        "request_timestamps": np.asarray([]),
        "ttfts": np.asarray([]),
        "decode_times": np.asarray([]),
        "input_lens": np.asarray([]),
        "output_lens": np.asarray([]),
        "has_timestamps": True,
    }
    aligned = align_trace_to_grid(
        {
            "timestamps": np.asarray([0.0, 0.25, 1.25, 1.5]),
            "power": np.asarray([0.0, 10.0, 50.0, 60.0]),
        },
        request_data,
    )

    assert aligned is not None
    np.testing.assert_array_equal(
        aligned["timestamps"], [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    )
    np.testing.assert_array_equal(
        aligned["power"], [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    )
    assert (
        align_trace_to_grid(
            {
                "timestamps": np.asarray([0.0, 0.25, 1.5, 1.75]),
                "power": np.asarray([0.0, 10.0, 60.0, 70.0]),
            },
            request_data,
        )
        is None
    )


def test_per_config_grid_resampling_recomputes_request_features():
    trace = {
        "timestamps": np.asarray([0.0, 0.251, 0.502, 0.753]),
        "power": np.asarray([0.0, 10.0, 20.0, 30.0]),
        "request_timestamps": np.asarray([0.25]),
        "ttfts": np.asarray([0.25]),
        "decode_times": np.asarray([0.0]),
        "input_lens": np.asarray([8.0]),
        "output_lens": np.asarray([1.0]),
        "dt": 0.251,
        "power_start_epoch_s": 0.0,
        "num_points": 4,
    }

    resampled = resample_trace_to_grid(trace, dt=0.25)

    np.testing.assert_array_equal(resampled["timestamps"], [0.0, 0.25, 0.5, 0.75])
    np.testing.assert_array_equal(resampled["active_requests"], [0.0, 1.0, 0.0, 0.0])
    assert resampled["dt"] == 0.25


def test_standalone_schedule_is_strict_and_keeps_prefill_only(tmp_path):
    path = tmp_path / "requests.json"
    path.write_text(
        json.dumps(
            [
                {"arrival_time": 0.1, "input_tokens": 32, "output_tokens": 0},
                {"arrival_time": 0.2, "input_tokens": 16, "output_tokens": 1},
            ]
        )
    )
    requests = load_request_schedule(str(path))
    assert requests == [
        {"arrival_time": 0.1, "input_tokens": 32.0, "output_tokens": 0.0},
        {"arrival_time": 0.2, "input_tokens": 16.0, "output_tokens": 1.0},
    ]

    path.write_text(
        json.dumps([{"arrival_time": 0.1, "input_tokens": -1, "output_tokens": 1}])
    )
    with pytest.raises(ValueError, match="non-negative"):
        load_request_schedule(str(path))


def test_explicit_bundles_route_into_gru_dataset(tmp_path):
    pair_manifest = tmp_path / "pair_manifest.csv"
    pair_manifest.write_text(
        "status,model_name,hardware,tensor_parallelism,rate,pair_key,power_csv_path,json_path\n"
    )
    view = {
        "power": np.asarray([100.0, 120.0, 110.0]),
        "active_requests": np.asarray([0.0, 1.0, 0.0]),
        "t_arrive_log": np.zeros(3),
        "dt": 0.25,
        "power_start_epoch_s": 1000.0,
        "input_lens": np.asarray([100.0]),
        "output_lens": np.asarray([10.0]),
        "ttfts": np.asarray([0.5]),
        "decode_times": np.asarray([0.2]),
    }

    def record(run_id):
        return SimpleNamespace(
            config_id="toy_H100_tp1",
            source_layout="bundle",
            view=view,
            provenance={
                "run_id": run_id,
                "paths": {
                    "manifest.json": f"/{run_id}/manifest.json",
                    "power.csv": f"/{run_id}/power.csv",
                    "engine.csv": f"/{run_id}/engine.csv",
                    "requests.json": f"/{run_id}/requests.json",
                },
                "sha256": {"manifest.json": "a", "power.csv": "b", "engine.csv": "c", "requests.json": "d"},
                "request_rows": {"source": 1, "aligned": 1, "retained": 1, "dropped": {}},
                "request_projection_indices": [0],
                "request_projection": {
                    "num_requests_raw": 1,
                    "num_requests_aligned": 1,
                    "num_requests_used": 1,
                    "num_requests_dropped_invalid_fields": 0,
                    "num_requests_dropped_decode_time": 0,
                    "num_requests_dropped_timestamp": 0,
                },
            },
        )

    records = {
        "/run-a": record("run-a"),
        "/run-b": record("run-b"),
        "/run-c": record("run-c"),
    }
    with (
        patch(
            "model.training_data.manifest.load_bundle_run",
            side_effect=lambda path: records[path],
        ),
        patch(
            "model.training_data.manifest.gru_view_from_record",
            side_effect=lambda value: value.view,
        ),
    ):
        manifest = run_prepare_experimental_manifest(
            pair_manifest_csv=str(pair_manifest),
            bundle_dirs=list(records),
            out_dir=str(tmp_path / "prepared"),
            min_traces_per_config=3,
        )

    config = manifest["configs"]["toy_H100_tp1"]
    assert config["written"] is True
    with np.load(config["dataset_npz"], allow_pickle=True) as data:
        assert data["pair_key"].tolist() == ["run-a", "run-b", "run-c"]


def test_explicit_bundle_failure_is_not_silently_skipped(tmp_path):
    pair_manifest = tmp_path / "pair_manifest.csv"
    pair_manifest.write_text(
        "status,model_name,hardware,tensor_parallelism,rate,pair_key,power_csv_path,json_path\n"
    )
    with patch(
        "model.training_data.manifest.load_bundle_run",
        side_effect=ValueError("bundle topology mismatch"),
    ):
        with pytest.raises(ValueError, match="bundle topology mismatch"):
            run_prepare_experimental_manifest(
                pair_manifest_csv=str(pair_manifest),
                bundle_dirs=["/bad-bundle"],
                out_dir=str(tmp_path / "prepared"),
            )


def test_bundle_request_source_resolves_from_hash_bound_lineage(tmp_path):
    requests = tmp_path / "requests.json"
    requests.write_text('{"source": "bundle"}')
    lineage = {
        "source_paths": {"requests.json": str(requests)},
        "source_sha256": {"requests.json": sha256_file(requests)},
    }
    assert _request_json_from_lineage(
        lineage, experimental_base=str(tmp_path)
    ) == str(requests)
    lineage["source_sha256"]["requests.json"] = "bad"
    with pytest.raises(ValueError, match="hash mismatch"):
        _request_json_from_lineage(lineage, experimental_base=str(tmp_path))
