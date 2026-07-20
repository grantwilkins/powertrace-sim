"""Fail-fast evidence requirements for profiling bundles.

Profiles name the minimum measured state needed for a scientific claim. A run
that lacks a required counter is invalid; it is never silently downgraded to a
request-timing reconstruction.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from model.training_data.power_parsing import parse_power_csv_per_gpu
from power_logger import POWER_PROFILES, canonical_field_name

CORE_GAUGES = (
    "num_requests_running",
    "num_requests_waiting",
    "gpu_cache_usage_perc",
)
CORE_COUNTERS = (
    "prompt_tokens_total",
    "generation_tokens_total",
    "iteration_tokens_total_sum",
    "iteration_tokens_total_count",
)
PROFILE_REQUIREMENTS = {
    "core": CORE_GAUGES + CORE_COUNTERS,
    "measured_ledger": CORE_GAUGES + CORE_COUNTERS,
}

PROFILE_PURPOSE = {
    "core": "stock vLLM scheduler/token evidence retained for reconstruction",
    "measured_ledger": "stock counters replace token rates and total A_t in the hybrid ledger",
}

COUNTER_COLUMNS = set(CORE_COUNTERS)
GAUGE_COLUMNS = set(CORE_GAUGES)


def requirements(profile: str) -> tuple[str, ...]:
    try:
        return PROFILE_REQUIREMENTS[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown evidence profile: {profile!r}") from exc


def _read_engine_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("engine.csv has no header")
        columns = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in columns:
                value = row.get(name, "")
                columns[name].append(float(value) if value not in (None, "") else np.nan)
    return {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}


def validate_engine_csv(path: str | Path, profile: str, *, max_missing=0.0) -> dict:
    """Validate timestamp, coverage, sign, and counter monotonicity."""
    columns = _read_engine_csv(Path(path))
    required = requirements(profile)
    missing_columns = sorted(set(("timestamp",) + required) - set(columns))
    if missing_columns:
        raise ValueError(f"engine.csv missing required columns: {missing_columns}")
    timestamps = columns["timestamp"]
    if timestamps.size < 4 or not np.all(np.isfinite(timestamps)):
        raise ValueError("engine.csv requires at least four finite samples")
    if not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("engine.csv timestamps must be strictly increasing")

    coverage = {}
    for name in required:
        values = columns[name]
        finite = np.isfinite(values)
        coverage[name] = float(np.mean(finite))
        if coverage[name] < 1.0 - max_missing:
            raise ValueError(
                f"engine.csv {name} coverage {coverage[name]:.3f} is below "
                f"{1.0 - max_missing:.3f}"
            )
        observed = values[finite]
        if np.any(observed < 0.0):
            raise ValueError(f"engine.csv {name} must be non-negative")
        if name in COUNTER_COLUMNS and np.any(np.diff(observed) < -1e-9):
            raise ValueError(f"engine.csv counter {name} decreased within one run")

    return {
        "samples": int(timestamps.size),
        "median_cadence_s": float(np.median(np.diff(timestamps))),
        "coverage": coverage,
    }


def validate_streams(
    run_dir: str | Path,
    profile: str,
    *,
    gpus_per_node: int,
    local_utc_offset_s: float,
    power_profile: str = "core",
) -> dict:
    """Validate both logger streams and their common epoch alignment."""
    root = Path(run_dir)
    if power_profile not in POWER_PROFILES:
        raise ValueError(f"unknown power telemetry profile: {power_profile!r}")
    with (root / "power.csv").open(newline="") as stream:
        header = next(csv.reader(stream), [])
    normalized_header = {canonical_field_name(value) for value in header}
    missing_power = sorted(set(POWER_PROFILES[power_profile]) - normalized_header)
    if missing_power:
        raise ValueError(
            f"power.csv missing {power_profile} telemetry columns: {missing_power}"
        )
    engine = validate_engine_csv(root / "engine.csv", profile)
    power = parse_power_csv_per_gpu(
        str(root / "power.csv"),
        gpus_per_node=int(gpus_per_node),
        strict_topology=True,
        local_utc_offset_s=float(local_utc_offset_s),
    )
    if power is None or power["timestamps"].size < 4:
        raise ValueError("power.csv requires at least four complete per-GPU samples")
    power_cadence = float(np.median(np.diff(power["timestamps"])))
    with (root / "engine.csv").open(newline="") as stream:
        first_engine = float(next(csv.DictReader(stream))["timestamp"])
    start_skew = first_engine - float(power["timestamps"][0])
    if abs(start_skew) > 2.0:
        raise ValueError(
            f"power/engine first-sample skew {start_skew:.3f}s exceeds 2.0s"
        )
    for name, cadence in (("engine", engine["median_cadence_s"]), ("power", power_cadence)):
        if not 0.15 <= cadence <= 0.50:
            raise ValueError(f"{name} median cadence {cadence:.3f}s is outside 2-6.7 Hz")
    return {
        "profile": profile,
        "purpose": PROFILE_PURPOSE[profile],
        "required_engine_columns": list(requirements(profile)),
        "engine": engine,
        "power": {
            "profile": power_profile,
            "required_columns": list(POWER_PROFILES[power_profile]),
            "samples": int(power["timestamps"].size),
            "median_cadence_s": power_cadence,
            "device_ids": list(power["device_ids"]),
        },
        "first_sample_skew_s": float(start_skew),
        "status": "validated",
    }


def expected_instrumentation(profile: str, power_profile: str = "core") -> dict:
    """Manifest contract for a dry run before observed stream stats exist."""
    if power_profile not in POWER_PROFILES:
        raise ValueError(f"unknown power telemetry profile: {power_profile!r}")
    return {
        "profile": profile,
        "purpose": PROFILE_PURPOSE[profile],
        "required_engine_columns": list(requirements(profile)),
        "power_profile": power_profile,
        "required_power_columns": list(POWER_PROFILES[power_profile]),
        "status": "expected_dry_run",
    }
