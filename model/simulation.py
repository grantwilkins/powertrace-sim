"""End-to-end selected-model request timing and power inference."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

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


def simulate(
    requests: Sequence[Mapping[str, object]], *,
    deployment: str | Path | Mapping[str, object],
    artifact: str | Path | Mapping[str, object] = DEFAULT_ARTIFACT,
    seed: int | None = None,
    allow_unsupported: bool = False,
) -> SimulationResult:
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
    ledger = emit_bins(
        trace, timed, arch=arch, tp=tp, dt=NATIVE_DT_S,
    )
    power = predict_power(
        ledger, arch=arch, model=model, hardware=hardware, tp=tp,
        artifact=release, dt_s=NATIVE_DT_S,
    )
    status = "unsupported_extrapolation" if violations else "supported"
    return SimulationResult(
        requests=request_rows,
        power=power,
        ledger=ledger,
        deployment=config,
        support_status=status,
        support_violations=violations,
        release_status=str(release["release_status"]),
        seed=seed,
        sampled_output_lengths=sampled,
    )


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


def simulate_file(
    requests_path: str | Path, *, deployment: str | Path,
    out_dir: str | Path, artifact_path: str | Path = DEFAULT_ARTIFACT,
    seed: int | None = None, allow_unsupported: bool = False,
) -> dict[str, str]:
    payload = json.loads(Path(requests_path).read_text())
    rows = payload if isinstance(payload, list) else payload.get("requests")
    if not isinstance(rows, list):
        raise ValueError("requests JSON must be a list or contain a requests list")
    result = simulate(
        rows, deployment=deployment, artifact=artifact_path, seed=seed,
        allow_unsupported=allow_unsupported,
    )
    return write_result(
        result, out_dir, requests_path=requests_path, artifact_path=artifact_path,
    )
