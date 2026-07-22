"""End-to-end selected-model request timing and power inference."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from model.power import predict_power
from model.power.predictor import dense_design, moe_design
from model.power.response import MOVING_AVERAGE_S
from model.release import (
    DEFAULT_ARTIFACT,
    load_artifact,
    resolve_deployment,
    support_violations,
)
from model.request_schedule import realize_requests
from model.timing.iteration import launch_overhead_s, transformer_bw_scale
from model.timing.ledger import NATIVE_DT_S, emit_bins, iter_bins
from model.timing.scheduler import EngineConfig, simulate_requests
from model.utils.provenance import git_state


@dataclass(frozen=True)
class SimulationResult:
    requests: list[dict[str, object]]
    power: dict[str, object]
    ledger: dict[str, object]
    deployment: dict[str, object]
    support_status: str
    support_violations: list[str]
    release_status: str
    seed: int | None
    sampled_output_lengths: bool


@dataclass(frozen=True)
class PreparedSimulation:
    requests: list[dict[str, object]]
    timed: list[dict[str, object]]
    trace: list[tuple[float, ...]]
    release: dict[str, object]
    arch: dict[str, object]
    deployment: dict[str, object]
    support_status: str
    support_violations: list[str]
    seed: int | None
    sampled_output_lengths: bool


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    root = Path(__file__).resolve().parents[1]
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def prepare_simulation(
    requests: Sequence[Mapping[str, object]], *,
    deployment: str | Path | Mapping[str, object],
    artifact: str | Path | Mapping[str, object] = DEFAULT_ARTIFACT,
    seed: int | None = None,
    allow_unsupported: bool = False,
) -> PreparedSimulation:
    release = dict(artifact) if isinstance(artifact, Mapping) else load_artifact(artifact)
    config, overrides = resolve_deployment(deployment, release)
    violations = support_violations(config, release)
    violations.extend(f"calibrated override: {key}" for key in sorted(overrides))
    if violations and not allow_unsupported:
        raise ValueError("unsupported deployment: " + "; ".join(violations))
    realized, sampled = realize_requests(requests, seed=seed)
    model = str(config["model"])
    hardware = str(config["hardware"])
    tp = int(config["tp"])
    arch = dict(release["architectures"][model])
    params = release["timing"][hardware]
    timing_input = [
        (
            float(row["arrival_time"]),
            int(row["executed_input_tokens"]),
            int(row["output_tokens"]),
            int(row["cached_prefix_tokens"]),
        )
        for row in realized
    ]
    trace: list[tuple[float, ...]] = []
    timed = simulate_requests(
        timing_input,
        arch=arch,
        hardware=hardware,
        tp=tp,
        eff_flops=float(params["eff_flops"]),
        eff_bw=float(params["eff_bw"]),
        t_launch_s=launch_overhead_s(
            arch,
            base_s=float(params["base_overhead_s"]),
            per_message_s=float(params["per_message_s"][str(tp)]),
        ),
        t_sample_s=float(params.get("per_token_sample_s", 0.0)),
        transformer_bw_scale=transformer_bw_scale(arch, params, hardware),
        engine=EngineConfig(
            max_num_seqs=int(config["max_num_seqs"]),
            chunk_budget_tokens=int(config["max_num_batched_tokens"]),
            gpu_memory_utilization=float(config["gpu_memory_utilization"]),
        ),
        iteration_trace=trace,
    )
    first_token_overhead = float(params["first_token_overhead_s"])
    request_rows = []
    for source, predicted in zip(realized, timed):
        request_rows.append({
            **source,
            "admitted_time": float(predicted["admitted_s"]),
            "ttft_s": float(predicted["ttft_s"] + first_token_overhead),
            "decode_duration_s": float(predicted["decode_duration_s"]),
            "e2e_s": float(predicted["e2e_s"] + first_token_overhead),
        })
    status = "unsupported_extrapolation" if violations else "supported"
    return PreparedSimulation(
        requests=request_rows,
        timed=timed,
        trace=trace,
        release=release,
        arch=arch,
        deployment=config,
        support_status=status,
        support_violations=violations,
        seed=seed,
        sampled_output_lengths=sampled,
    )


def simulate(
    requests: Sequence[Mapping[str, object]], *,
    deployment: str | Path | Mapping[str, object],
    artifact: str | Path | Mapping[str, object] = DEFAULT_ARTIFACT,
    seed: int | None = None,
    allow_unsupported: bool = False,
) -> SimulationResult:
    prepared = prepare_simulation(
        requests, deployment=deployment, artifact=artifact, seed=seed,
        allow_unsupported=allow_unsupported,
    )
    tp = int(prepared.deployment["tp"])
    hardware = str(prepared.deployment["hardware"])
    model = str(prepared.deployment["model"])
    ledger = emit_bins(
        prepared.trace, prepared.timed, arch=prepared.arch, tp=tp,
        dt=NATIVE_DT_S,
    )
    power = predict_power(
        ledger, arch=prepared.arch, model=model, hardware=hardware, tp=tp,
        artifact=prepared.release, dt_s=NATIVE_DT_S,
    )
    return SimulationResult(
        requests=prepared.requests,
        power=power,
        ledger=ledger,
        deployment=prepared.deployment,
        support_status=prepared.support_status,
        support_violations=prepared.support_violations,
        release_status=str(prepared.release["release_status"]),
        seed=prepared.seed,
        sampled_output_lengths=prepared.sampled_output_lengths,
    )


def iter_prepared_power_bins(
    prepared: PreparedSimulation, *, horizon_s: float | None = None,
) -> Iterator[dict[str, object]]:
    """Yield power rows while retaining only meter-response history."""
    config = prepared.deployment
    model = str(config["model"])
    hardware = str(config["hardware"])
    tp = int(config["tp"])
    family = str(prepared.arch["family"])
    if family.startswith("dense"):
        fit = prepared.release["power"]["dense"][hardware]
        surface = "dense"
    else:
        fit = prepared.release["power"]["moe"]["per_model"][model]
        surface = f"moe:{model}"
    names = list(fit["feature_names"])
    coefficients = np.asarray(fit["coefficients"], dtype=float)
    shift = int(round(float(fit.get("delay_s", 0.0)) / NATIVE_DT_S))
    window = max(1, int(round(MOVING_AVERAGE_S[hardware] / NATIVE_DT_S)))
    raw_history: deque[np.ndarray] = deque(maxlen=shift + 1)
    response_history: deque[np.ndarray] = deque(maxlen=window)
    previous_ledger: dict[str, float] | None = None
    rows = iter_bins(
        prepared.trace, prepared.timed, arch=prepared.arch, tp=tp,
        dt=NATIVE_DT_S, horizon_s=horizon_s,
    )
    for index, ledger in enumerate(rows):
        current = {key: np.asarray([value], dtype=float)
                   for key, value in ledger.items()}
        if family.startswith("dense"):
            raw = dense_design(
                current, arch=prepared.arch, hardware=hardware, tp=tp,
            )[0]
            raw_history.append(raw)
            delayed = raw_history[0] if len(raw_history) > shift else raw_history[0]
            response_history.append(delayed)
            design = np.mean(np.stack(response_history), axis=0)
        else:
            prior = ledger if previous_ledger is None else previous_ledger
            paired = {
                key: np.asarray([prior[key], value], dtype=float)
                for key, value in ledger.items()
            }
            design = moe_design(
                paired, hardware=hardware, tp=tp, feature_names=names,
            )[-1]
        previous_ledger = ledger
        contributions = design * coefficients * tp
        mean_gpu = float(np.sum(design * coefficients))
        row = {
            "time_s": (index + 1) * NATIVE_DT_S,
            "node_gpu_power_w": mean_gpu * tp,
            "mean_active_gpu_power_w": mean_gpu,
            "busy_fraction": float(ledger["busy"]),
            "surface": surface,
            "support_status": prepared.support_status,
        }
        row.update({
            f"component_{name}_node_w": float(value)
            for name, value in zip(names, contributions)
        })
        yield row


def iter_power_bins(result: SimulationResult) -> Iterator[dict[str, object]]:
    names = list(result.power["feature_names"])
    contributions = np.asarray(result.power["contributions_node_w"])
    for index, (node, mean_gpu, busy) in enumerate(zip(
        result.power["node_gpu_power_w"],
        result.power["mean_gpu_power_w"],
        result.ledger["busy"],
    )):
        row = {
            "time_s": (index + 1) * NATIVE_DT_S,
            "node_gpu_power_w": float(node),
            "mean_active_gpu_power_w": float(mean_gpu),
            "busy_fraction": float(busy),
            "surface": result.power["surface"],
            "support_status": result.support_status,
        }
        row.update({
            f"component_{name}_node_w": float(value)
            for name, value in zip(names, contributions[index])
        })
        yield row


def write_result(
    result: SimulationResult, out_dir: str | Path, *,
    requests_path: str | Path | None = None,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
) -> dict[str, str]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    power_path = output / "power.csv"
    request_path = output / "requests.csv"
    manifest_path = output / "manifest.json"
    with power_path.open("w", newline="") as stream:
        fieldnames = [
            "time_s", "node_gpu_power_w", "mean_active_gpu_power_w",
            "busy_fraction", "surface", "support_status",
            *(f"component_{name}_node_w" for name in result.power["feature_names"]),
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(iter_power_bins(result))
    with request_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result.requests[0]))
        writer.writeheader()
        writer.writerows(result.requests)
    artifact = Path(artifact_path)
    manifest = {
        "schema_version": "powertrace-inference-result-v1",
        "release_status": result.release_status,
        "support_status": result.support_status,
        "support_violations": result.support_violations,
        "native_dt_s": NATIVE_DT_S,
        "deployment": result.deployment,
        "seed": result.seed,
        "sampling_used": result.sampled_output_lengths,
        "source_revision": git_state(),
        "inputs": {
            "artifact": {"path": _portable_path(artifact), "sha256": _sha256(artifact)},
            "requests": (
                {"path": _portable_path(requests_path), "sha256": _sha256(requests_path)}
                if requests_path is not None else None
            ),
        },
        "outputs": {"power": power_path.name, "requests": request_path.name},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"power": str(power_path), "requests": str(request_path), "manifest": str(manifest_path)}


def write_prepared_result(
    prepared: PreparedSimulation, out_dir: str | Path, *,
    requests_path: str | Path | None = None,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
) -> dict[str, str]:
    """Write a prepared run while consuming power bins exactly once."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    power_path = output / "power.csv"
    request_path = output / "requests.csv"
    manifest_path = output / "manifest.json"
    model = str(prepared.deployment["model"])
    hardware = str(prepared.deployment["hardware"])
    family = str(prepared.arch["family"])
    fit = (
        prepared.release["power"]["dense"][hardware]
        if family.startswith("dense")
        else prepared.release["power"]["moe"]["per_model"][model]
    )
    fieldnames = [
        "time_s", "node_gpu_power_w", "mean_active_gpu_power_w",
        "busy_fraction", "surface", "support_status",
        *(f"component_{name}_node_w" for name in fit["feature_names"]),
    ]
    with power_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(iter_prepared_power_bins(prepared))
    with request_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prepared.requests[0]))
        writer.writeheader()
        writer.writerows(prepared.requests)
    artifact = Path(artifact_path)
    manifest = {
        "schema_version": "powertrace-inference-result-v1",
        "release_status": str(prepared.release["release_status"]),
        "support_status": prepared.support_status,
        "support_violations": prepared.support_violations,
        "native_dt_s": NATIVE_DT_S,
        "deployment": prepared.deployment,
        "seed": prepared.seed,
        "sampling_used": prepared.sampled_output_lengths,
        "source_revision": git_state(),
        "inputs": {
            "artifact": {"path": _portable_path(artifact), "sha256": _sha256(artifact)},
            "requests": (
                {"path": _portable_path(requests_path), "sha256": _sha256(requests_path)}
                if requests_path is not None else None
            ),
        },
        "outputs": {"power": power_path.name, "requests": request_path.name},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"power": str(power_path), "requests": str(request_path), "manifest": str(manifest_path)}


def simulate_file(
    requests_path: str | Path, *, deployment: str | Path,
    out_dir: str | Path, artifact_path: str | Path = DEFAULT_ARTIFACT,
    seed: int | None = None, allow_unsupported: bool = False,
) -> dict[str, str]:
    payload = json.loads(Path(requests_path).read_text())
    rows = payload if isinstance(payload, list) else payload.get("requests")
    if not isinstance(rows, list):
        raise ValueError("requests JSON must be a list or contain a requests list")
    prepared = prepare_simulation(
        rows, deployment=deployment, artifact=artifact_path, seed=seed,
        allow_unsupported=allow_unsupported,
    )
    return write_prepared_result(
        prepared, out_dir, requests_path=requests_path, artifact_path=artifact_path,
    )
