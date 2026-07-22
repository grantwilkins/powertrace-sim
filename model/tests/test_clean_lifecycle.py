"""
Claim:
Prepared datasets bind one consistent run population, and evaluation computes
energy, amplitude, and temporal metrics on aligned traces without pooling or
changing the 250 ms power domain.

Plausible wrong implementations:
- accept timing, split, index, and power payloads with different run IDs;
- omit a large input from the provenance hash contract;
- use mean-power error in place of integrated energy error;
- compute ACF on native bins instead of the declared one-second means.
"""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from model.evaluation import evaluate_power_csv
from model.preparation import prepare_dataset


def _write_trace(path, values):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["node_gpu_power_w"])
        writer.writeheader()
        writer.writerows({"node_gpu_power_w": value} for value in values)


def test_identical_trace_has_exact_energy_amplitude_and_acf_scores(tmp_path):
    seconds = np.arange(64, dtype=float)
    one_second = 100.0 + 10.0 * np.sin(seconds / 5.0)
    native = np.repeat(one_second, 4)
    measured = tmp_path / "measured.csv"
    predicted = tmp_path / "predicted.csv"
    _write_trace(measured, native)
    _write_trace(predicted, native)

    metrics = evaluate_power_csv(measured, predicted)

    assert metrics["energy_error_pct"] == 0.0
    assert metrics["signed_bias_pct"] == 0.0
    assert metrics["rmse_w"] == 0.0
    assert metrics["nrmse_range"] == 0.0
    assert metrics["acf_mae"] == 0.0
    assert metrics["acf_r2"] == 1.0


def _prepared_inputs(tmp_path, power_run=0):
    timing = tmp_path / "timing.npz"
    np.savez(
        timing,
        req_run_id=np.asarray([0]), arrival_time_s=np.asarray([0.0]),
        input_tokens=np.asarray([8]), output_tokens=np.asarray([2]),
        ttft_s=np.asarray([0.1]), decode_duration_s=np.asarray([0.1]),
        run_model=np.asarray(["m"]), run_hardware=np.asarray(["A100"]),
        run_tp=np.asarray([1]),
    )
    power = tmp_path / "power.npz"
    np.savez(
        power, run_id=np.asarray([power_run]), power=np.asarray([100.0]),
        power_valid=np.asarray([True]), dt_s=np.asarray(0.25),
    )
    index = tmp_path / "runs.json"
    index.write_text(json.dumps({"runs": [{"run_id": 0}]}) + "\n")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"roles": {"0": "train"}}) + "\n")
    return timing, index, split, power


def test_prepare_dataset_hash_binds_one_consistent_run_population(tmp_path):
    timing, index, split, power = _prepared_inputs(tmp_path)
    out = tmp_path / "manifest.json"

    manifest = prepare_dataset(
        timing_dataset=timing, run_index=index, split_manifest=split,
        base_split_manifest=split, probe_calibration=split,
        power_cache=power, out_manifest=out,
    )

    assert manifest["run_ids"] == [0]
    assert manifest["inputs"]["timing_dataset"]["sha256"]
    assert manifest["inputs"]["power_cache"]["sha256"]
    assert json.loads(out.read_text()) == manifest


def test_prepare_dataset_rejects_run_identity_mismatch(tmp_path):
    timing, index, split, power = _prepared_inputs(tmp_path, power_run=1)
    with pytest.raises(ValueError, match="disagree on run IDs"):
        prepare_dataset(
            timing_dataset=timing, run_index=index, split_manifest=split,
            base_split_manifest=split, probe_calibration=split,
            power_cache=power, out_manifest=tmp_path / "manifest.json",
        )
