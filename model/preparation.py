"""Bind selected-model training payloads into one validated dataset manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _identity(path: str | Path, root: Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    try:
        display = str(resolved.relative_to(root))
    except ValueError:
        display = str(resolved)
    return {
        "path": display,
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def prepare_dataset(
    *, timing_dataset: str | Path, run_index: str | Path,
    split_manifest: str | Path, base_split_manifest: str | Path,
    probe_calibration: str | Path, power_cache: str | Path,
    out_manifest: str | Path,
) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    with np.load(timing_dataset, allow_pickle=False) as source:
        required = {
            "req_run_id", "arrival_time_s", "input_tokens", "output_tokens",
            "ttft_s", "decode_duration_s", "run_model", "run_hardware", "run_tp",
        }
        missing = sorted(required - set(source.files))
        if missing:
            raise ValueError(f"timing dataset missing fields: {missing}")
        request_count = int(source["req_run_id"].size)
        timing_runs = set(map(int, np.unique(source["req_run_id"])))
    with np.load(power_cache, allow_pickle=True) as cache:
        required = {"run_id", "power", "power_valid", "dt_s"}
        missing = sorted(required - set(cache.files))
        if missing:
            raise ValueError(f"power cache missing fields: {missing}")
        if float(cache["dt_s"]) != 0.25:
            raise ValueError("selected power fitting requires 250 ms bins")
        power_runs = set(map(int, np.unique(cache["run_id"])))
        power_bins = int(cache["run_id"].size)
        finite_power_bins = int(np.isfinite(cache["power"]).sum())
    if timing_runs != power_runs:
        raise ValueError("timing dataset and power cache disagree on run IDs")
    index = json.loads(Path(run_index).read_text())
    indexed_runs = {int(row["run_id"]) for row in index["runs"]}
    if indexed_runs != timing_runs:
        raise ValueError("run index and prepared payloads disagree on run IDs")
    split = json.loads(Path(split_manifest).read_text())
    roles = {int(run) for run in split.get("roles", {})}
    if roles and roles != timing_runs:
        raise ValueError("split manifest and prepared payloads disagree on run IDs")
    payload = {
        "schema_version": "powertrace-prepared-dataset-v1",
        "native_dt_s": 0.25,
        "request_count": request_count,
        "run_count": len(timing_runs),
        "power_bin_count": power_bins,
        "finite_power_bin_count": finite_power_bins,
        "run_ids": sorted(timing_runs),
        "inputs": {
            "timing_dataset": _identity(timing_dataset, root),
            "run_index": _identity(run_index, root),
            "split_manifest": _identity(split_manifest, root),
            "base_split_manifest": _identity(base_split_manifest, root),
            "probe_calibration": _identity(probe_calibration, root),
            "power_cache": _identity(power_cache, root),
        },
    }
    output = Path(out_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
