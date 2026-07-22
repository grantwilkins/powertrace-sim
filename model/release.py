"""Release-artifact and deployment-contract loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

DEFAULT_ARTIFACT = Path(__file__).with_name("artifacts") / "powertrace_v1.json"
CALIBRATED_FIELDS = frozenset((
    "model", "hardware", "tp", "dtype", "max_num_seqs",
    "max_num_batched_tokens", "gpu_memory_utilization", "scheduler",
    "moe_routing",
))


def load_artifact(path: str | Path = DEFAULT_ARTIFACT) -> dict:
    artifact = json.loads(Path(path).read_text())
    if artifact.get("schema_version") != "powertrace-release-v1":
        raise ValueError("unsupported PowerTrace release artifact")
    if artifact.get("native_dt_s") != 0.25:
        raise ValueError("selected release requires a 250 ms native grid")
    if artifact.get("release_status") not in ("pre_sealed", "sealed"):
        raise ValueError("invalid release status")
    return artifact


def resolve_deployment(value: str | Path | Mapping[str, object], artifact: dict):
    overrides = {}
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        path = Path(value)
        payload = json.loads(path.read_text()) if path.is_file() else {"preset": str(value)}
    preset_name = payload.get("preset")
    if preset_name not in artifact["presets"]:
        raise ValueError(f"unknown deployment preset {preset_name!r}")
    deployment = dict(artifact["presets"][preset_name])
    supplied = payload.get("overrides", {})
    if not isinstance(supplied, Mapping):
        raise ValueError("deployment overrides must be an object")
    for key, value in supplied.items():
        if key not in CALIBRATED_FIELDS:
            raise ValueError(f"unknown deployment override {key!r}")
        if deployment.get(key) != value:
            overrides[key] = {"preset": deployment.get(key), "requested": value}
            deployment[key] = value
    deployment["preset"] = preset_name
    return deployment, overrides


def support_violations(deployment: Mapping[str, object], artifact: dict) -> list[str]:
    model = str(deployment["model"])
    hardware = str(deployment["hardware"])
    tp = int(deployment["tp"])
    violations = []
    if model not in artifact["architectures"]:
        return [f"unknown architecture: {model}"]
    if hardware not in artifact["support"]["hardware"]:
        violations.append(f"unsupported hardware: {hardware}")
    family = str(artifact["architectures"][model]["family"])
    if family.startswith("dense"):
        dense = artifact["support"]["dense"]
        if family not in dense["families"] or tp not in dense["tp"]:
            violations.append(f"dense support excludes family={family}, tp={tp}")
    else:
        support = artifact["support"]["moe"].get(model)
        if support is None or hardware != support["hardware"] or tp not in support["tp"]:
            violations.append(f"MoE support excludes {model}/{hardware}/tp{tp}")
        if deployment.get("moe_routing") != artifact["routing"]["mode"]:
            violations.append("MoE routing mode differs from the release")
    timing = artifact["timing"].get(hardware, {})
    if str(tp) not in timing.get("per_message_s", {}):
        violations.append(f"timing calibration excludes {hardware}/tp{tp}")
    if deployment.get("scheduler") != "vllm_v1_decode_first":
        violations.append("scheduler policy differs from the release")
    return violations

