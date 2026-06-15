"""Extended ``nvidia-smi`` power logger (Tier-0 instrumentation, CAMPAIGN.md §5-A).

The current pipeline logs only ``timestamp,power.draw,utilization.gpu,memory.used``.
This adds ``clocks.sm`` (DVFS is the largest unmodeled term and is a free field),
``clocks.mem``, ``utilization.memory`` and ``temperature.gpu``, per GPU at 4 Hz.

Compatibility constraint: the emitted CSV header must keep a ``timestamp`` column
and a ``power.draw [W]`` column so the existing
``model/training_data/power_parsing.parse_power_csv`` header sniff
(``"time" in h`` and ``"power" in h and "draw" in h``) and its per-GPU grouping
continue to work unchanged. The query string below preserves that ordering.

Only the command/argv construction lives here (pure, unit-testable). The actual
process is spawned by ``probe_runner`` / the bash logger, redirecting stdout to
``power.csv`` — there is nothing GPU-specific to test offline.
"""

from __future__ import annotations

# Order matters: timestamp first, power.draw second (parse_power_csv compat).
EXTENDED_FIELDS = (
    "timestamp",
    "power.draw",
    "clocks.sm",
    "clocks.mem",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "temperature.gpu",
)

DEFAULT_INTERVAL_MS = 250  # 4 Hz, aligned to the engine /metrics scraper


def nvidia_smi_command(
    fields=EXTENDED_FIELDS, interval_ms: int = DEFAULT_INTERVAL_MS
) -> list[str]:
    """Return the ``nvidia-smi`` argv that streams the extended per-GPU fields."""
    return [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,nounits",
        f"-lms={int(interval_ms)}",
    ]
