"""Role-aware inference for one-prefiller, one-decoder deployments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from model.power import predict_power
from model.release import (
    DEFAULT_ARTIFACT,
    load_artifact,
    resolve_deployment,
    support_violations,
)
from model.request_schedule import realize_requests
from model.timing.iteration import launch_overhead_s, transformer_bw_scale
from model.timing.ledger import NATIVE_DT_S, emit_bins
from model.timing.scheduler import EngineConfig, simulate_requests


@dataclass(frozen=True)
class RoleSimulation:
    timed: list[dict[str, object]]
    trace: list[tuple[float, ...]]
    ledger: dict[str, object]
    power: dict[str, object]


@dataclass(frozen=True)
class DisaggregatedSimulation:
    requests: list[dict[str, object]]
    roles: dict[str, RoleSimulation]
    node_gpu_power_w: np.ndarray
    deployment: dict[str, object]
    support_status: str
    support_violations: list[str]
    release_status: str
    source_idle_w_per_gpu: float


def _timing_arguments(
    arch: Mapping[str, object],
    config: Mapping[str, object],
    params: Mapping[str, object],
) -> dict[str, object]:
    hardware = str(config["hardware"])
    tp = int(config["tp"])
    return {
        "arch": arch,
        "hardware": hardware,
        "tp": tp,
        "eff_flops": float(params["eff_flops"]),
        "eff_bw": float(params["eff_bw"]),
        "t_launch_s": launch_overhead_s(
            arch,
            base_s=float(params["base_overhead_s"]),
            per_message_s=float(params["per_message_s"][str(tp)]),
        ),
        "t_sample_s": float(params.get("per_token_sample_s", 0.0)),
        "transformer_bw_scale": transformer_bw_scale(arch, params, hardware),
        "engine": EngineConfig(
            max_num_seqs=int(config["max_num_seqs"]),
            chunk_budget_tokens=int(config["max_num_batched_tokens"]),
            gpu_memory_utilization=float(config["gpu_memory_utilization"]),
        ),
    }


def _source_idle(release: Mapping[str, object], model: str) -> float:
    fit = release["power"]["moe"]["per_model"][model]
    coefficients = dict(zip(fit["feature_names"], fit["coefficients"]))
    return float(coefficients["idle"])


def simulate_disaggregated(
    requests: Sequence[Mapping[str, object]], *,
    deployment: str | Path | Mapping[str, object],
    artifact: str | Path | Mapping[str, object] = DEFAULT_ARTIFACT,
    seed: int | None = None,
    horizon_s: float | None = None,
    allow_unsupported: bool = False,
) -> DisaggregatedSimulation:
    """Simulate serial prefill and decode engines without calling them TP2."""
    release = dict(artifact) if isinstance(artifact, Mapping) else load_artifact(artifact)
    config, overrides = resolve_deployment(deployment, release)
    violations = support_violations(config, release)
    violations.extend(f"calibrated override: {key}" for key in sorted(overrides))
    if violations and not allow_unsupported:
        raise ValueError("unsupported deployment: " + "; ".join(violations))

    realized, _ = realize_requests(requests, seed=seed)
    model = str(config["model"])
    hardware = str(config["hardware"])
    tp = int(config["tp"])
    arch = dict(release["architectures"][model])
    params = release["timing"][hardware]
    timing = _timing_arguments(arch, config, params)
    first_token_overhead = float(params["first_token_overhead_s"])

    prefill_input = [
        (
            float(row["arrival_time"]),
            int(row["executed_input_tokens"]),
            1,
            int(row["cached_prefix_tokens"]),
        )
        for row in realized
    ]
    prefill_trace: list[tuple[float, ...]] = []
    prefill_timed = simulate_requests(
        prefill_input, iteration_trace=prefill_trace, **timing
    )

    decode_input = [
        (
            float(prefill["arrival_s"] + prefill["e2e_s"] + first_token_overhead),
            0,
            int(row["output_tokens"]),
            int(row["input_tokens"]),
        )
        for row, prefill in zip(realized, prefill_timed)
    ]
    decode_trace: list[tuple[float, ...]] = []
    decode_timed = simulate_requests(
        decode_input, iteration_trace=decode_trace, **timing
    )

    request_rows = []
    for row, prefill, decode in zip(realized, prefill_timed, decode_timed):
        arrival = float(row["arrival_time"])
        prefill_s = float(prefill["e2e_s"] + first_token_overhead)
        decode_ttft = float(decode["ttft_s"] + first_token_overhead)
        request_rows.append({
            **row,
            "prefill_s": prefill_s,
            "decode_admitted_time": float(decode["admitted_s"]),
            "decode_ttft_s": decode_ttft,
            "decode_duration_s": float(decode["decode_duration_s"]),
            "ttft_s": float(decode["arrival_s"] - arrival + decode_ttft),
            "e2e_s": float(
                decode["arrival_s"] - arrival
                + decode["e2e_s"]
                + first_token_overhead
            ),
        })

    resolved_horizon = horizon_s
    if resolved_horizon is None:
        resolved_horizon = max(
            float(row["arrival_time"]) + float(row["e2e_s"])
            for row in request_rows
        )
    roles = {}
    for role, timed, trace in (
        ("prefill", prefill_timed, prefill_trace),
        ("decode", decode_timed, decode_trace),
    ):
        ledger = emit_bins(
            trace, timed, arch=arch, tp=tp, dt=NATIVE_DT_S,
            horizon_s=resolved_horizon,
        )
        power = predict_power(
            ledger, arch=arch, model=model, hardware=hardware, tp=tp,
            artifact=release, dt_s=NATIVE_DT_S,
        )
        roles[role] = RoleSimulation(timed, trace, ledger, power)

    node_power = sum(
        np.asarray(role.power["node_gpu_power_w"], dtype=float)
        for role in roles.values()
    )
    return DisaggregatedSimulation(
        requests=request_rows,
        roles=roles,
        node_gpu_power_w=node_power,
        deployment=config,
        support_status="unsupported_extrapolation" if violations else "supported",
        support_violations=violations,
        release_status=str(release["release_status"]),
        source_idle_w_per_gpu=_source_idle(release, model),
    )


def apply_shared_idle_calibration(
    result: DisaggregatedSimulation, target_idle_w_per_gpu: float,
) -> dict[str, np.ndarray]:
    """Replace one shared per-GPU idle scalar while preserving dynamic power."""
    target = float(target_idle_w_per_gpu)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("target idle power must be positive and finite")
    tp = int(result.deployment["tp"])
    shift = (target - result.source_idle_w_per_gpu) * tp
    roles = {
        name: np.asarray(role.power["node_gpu_power_w"], dtype=float) + shift
        for name, role in result.roles.items()
    }
    return {**roles, "node_gpu_power_w": sum(roles.values())}
