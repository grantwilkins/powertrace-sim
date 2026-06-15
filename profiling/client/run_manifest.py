"""Run manifest emitter (Tier-0 instrumentation, CAMPAIGN.md §2 / §5-E).

Each probe run emits a self-describing ``manifest.json`` so the ledger builder
reads one directory -> aligned ``(power, state, work_rate)`` rows, with no
cross-file timestamp inference. ``build_manifest`` is a pure assembler (unit
tested); ``capture_clock`` and ``collect_versions`` are the thin live helpers.
"""

from __future__ import annotations

import json
import time
from typing import Optional

MANIFEST_VERSION = 1


def capture_clock(
    power_clock_epoch: Optional[float] = None,
    engine_clock_epoch: Optional[float] = None,
) -> dict:
    """Capture the one-time clock offsets used to align power and engine logs.

    Both loggers stamp wall time, so on a single host the offsets are ~0; we still
    record the measured offset of each logger's first-sample epoch against a
    common reference read here, plus a monotonic anchor for ordering.
    """
    ref = time.time()
    mono = time.perf_counter()
    return {
        "power_epoch_offset_s": (
            float(power_clock_epoch - ref) if power_clock_epoch is not None else 0.0
        ),
        "engine_epoch_offset_s": (
            float(engine_clock_epoch - ref) if engine_clock_epoch is not None else 0.0
        ),
        "monotonic_start": float(mono),
    }


def build_manifest(
    *,
    run_id: str,
    probe: dict,
    model: str,
    arch: dict,
    hardware: str,
    tp: int,
    gpus_per_node: int,
    server: dict,
    versions: dict,
    clock: dict,
) -> dict:
    """Assemble the §2 manifest dict. Pure: all inputs are passed in."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "run_id": run_id,
        "probe": probe,            # {"type": ..., "level": ..., "params": {...}}
        "model": model,
        "arch": arch,              # from arch_extract.extract_arch
        "hardware": hardware,
        "tp": int(tp),
        "gpus_per_node": int(gpus_per_node),
        "server": server,          # max_num_seqs, chunked-prefill, prefix-caching, ...
        "versions": versions,      # {"vllm": ..., "git_sha": ..., "gpu_driver": ...}
        "clock": clock,            # from capture_clock
    }


def write_manifest(path: str, manifest: dict) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def collect_versions() -> dict:
    """Best-effort version collection (live helper, not unit-tested)."""
    import subprocess

    def _safe(fn, default="unknown"):
        try:
            return fn()
        except Exception:
            return default

    def _git_sha():
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()

    def _vllm():
        import vllm  # type: ignore
        return getattr(vllm, "__version__", "unknown")

    def _driver():
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        )
        return out.splitlines()[0].strip()

    return {
        "vllm": _safe(_vllm),
        "git_sha": _safe(_git_sha),
        "gpu_driver": _safe(_driver),
    }
