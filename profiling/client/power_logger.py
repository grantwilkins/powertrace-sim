"""Extended ``nvidia-smi`` power logger (Tier-0 instrumentation, CAMPAIGN.md §5-A).

Each row includes stable ``index`` and ``uuid`` identity plus ``clocks.sm``
(DVFS is the largest unmodeled term and is a free field),
``clocks.mem``, ``utilization.memory`` and ``temperature.gpu``, per GPU at 4 Hz.

The bundle parser groups rows by a bounded capture window and UUID, validates a
stable UUID-to-index mapping, and rejects topology drift; it never infers samples
from anonymous row blocks.

Only the command/argv construction lives here (pure, unit-testable). The actual
process is spawned by ``probe_runner`` / the bash logger, redirecting stdout to
``power.csv`` — there is nothing GPU-specific to test offline.
"""

from __future__ import annotations

# Stable identity is part of every row; ingestion groups bounded capture windows.
EXTENDED_FIELDS = (
    "timestamp",
    "index",
    "uuid",
    "power.draw",
    "clocks.sm",
    "clocks.mem",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "temperature.gpu",
)

TP8_STATE_FIELDS = EXTENDED_FIELDS + (
    "pstate",
    "power.limit",
    "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
)

POWER_PROFILES = {
    "core": EXTENDED_FIELDS,
    "tp8_state": TP8_STATE_FIELDS,
}

DEFAULT_INTERVAL_MS = 250  # 4 Hz, aligned to the engine /metrics scraper


def nvidia_smi_command(
    fields=None, interval_ms: int = DEFAULT_INTERVAL_MS, profile: str = "core"
) -> list[str]:
    """Return the ``nvidia-smi`` argv that streams the extended per-GPU fields."""
    if fields is None:
        try:
            fields = POWER_PROFILES[profile]
        except KeyError as exc:
            raise ValueError(f"unknown power telemetry profile: {profile!r}") from exc
    return [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,nounits",
        f"-lms={int(interval_ms)}",
    ]


def nvidia_smi_snapshot_command(profile: str = "core") -> list[str]:
    """One-shot query used to reject unsupported fields before a live workload."""
    try:
        fields = POWER_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown power telemetry profile: {profile!r}") from exc
    return [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
