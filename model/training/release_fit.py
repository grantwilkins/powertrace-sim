"""Fit the frozen selected equations and emit one compact release artifact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from model.release import DEFAULT_ARTIFACT, load_artifact
from model.training.fp8 import calibrate
from model.training.power import fit_power_surfaces, load_power_cache
from model.training.timing import (
    CHUNK_BUDGET_TOKENS,
    fit_hardware,
    loaded_request_points,
    probe_points,
    solo_request_points,
)

TIMING_FIELDS = (
    "base_overhead_s", "eff_bw", "eff_flops", "first_token_overhead_s",
    "per_message_s", "per_token_sample_s", "fp8_stream_scale",
    "fp8_stream_support", "provenance_notes",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(identity: dict[str, object], root: Path) -> Path:
    path = Path(str(identity["path"]))
    resolved = path if path.is_absolute() else root / path
    if _sha256(resolved) != identity["sha256"]:
        raise ValueError(f"prepared input identity mismatch: {path}")
    return resolved


def fit_timing(
    dataset_path: Path, split_path: Path, calibration_path: Path,
) -> dict[str, object]:
    data = dict(np.load(dataset_path, allow_pickle=False))
    split = json.loads(split_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    roles = {int(key): value for key, value in split["roles"].items()}
    points = (
        probe_points(calibration)
        + solo_request_points(data, roles)
        + loaded_request_points(data, roles)
    )
    held = {
        (hardware, model)
        for source in (split["holdout_model"], split["holdout_twin"])
        for hardware, model in source.items()
    }
    if any((point["hardware"], point["model"]) in held for point in points):
        raise AssertionError("holdout model leaked into timing fitting points")
    fitted = {
        "procedure": "frozen architecture-aware timing fit",
        "chunk_budget_tokens": CHUNK_BUDGET_TOKENS,
    }
    for hardware in ("A100", "H100"):
        fitted[hardware] = fit_hardware(points, hardware)
    calibrated, _, _ = calibrate(data, split, fitted)
    return calibrated


def fit_release(
    prepared_manifest: str | Path, out_artifact: str | Path, *,
    template_artifact: str | Path = DEFAULT_ARTIFACT,
) -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    prepared_path = Path(prepared_manifest)
    prepared = json.loads(prepared_path.read_text())
    if prepared.get("schema_version") != "powertrace-prepared-dataset-v1":
        raise ValueError("unsupported prepared dataset manifest")
    inputs = prepared["inputs"]
    timing_dataset = _resolve(inputs["timing_dataset"], root)
    base_split = _resolve(inputs["base_split_manifest"], root)
    probe_calibration = _resolve(inputs["probe_calibration"], root)
    power_cache = _resolve(inputs["power_cache"], root)
    timing = fit_timing(timing_dataset, base_split, probe_calibration)
    power = fit_power_surfaces(load_power_cache(power_cache))
    template = load_artifact(template_artifact)
    compact_timing = {
        hardware: {
            key: value for key, value in timing[hardware].items()
            if key in TIMING_FIELDS
        }
        for hardware in ("A100", "H100")
    }
    release = {
        **template,
        "release_status": "pre_sealed",
        "timing": compact_timing,
        "power": power,
        "provenance": {
            "prepared_dataset_manifest": str(prepared_path),
            "prepared_dataset_manifest_sha256": _sha256(prepared_path),
            "training_policy": "frozen timing calibration and clean v4 power equations",
        },
    }
    output = Path(out_artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
    return release

