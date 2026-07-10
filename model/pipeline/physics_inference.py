from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from model.classifiers.physics import load_physics_artifact, predict_mean_node_power
from model.pipeline.artifact_resolution import resolve_throughput
from model.pipeline.request_builder import load_request_schedule
from model.training_data.arch import get_arch
from model.training_data.ledger_view import KV_ELEM_BYTES, bin_work_rates
from model.utils.io import load_json, write_json
from model.utils.provenance import file_identity, git_state

CONFIG_RE = re.compile(r"^(.+)_(A100|H100)_tp(\d+)$")


def build_modeled_work_ledger(
    requests: Sequence[Mapping[str, object]],
    *,
    arch: Mapping[str, object],
    tp: int,
    throughput: Mapping[str, float],
    dt: float = 1.0,
    T: int | None = None,
) -> dict[str, np.ndarray]:
    """Convert an arrivals-only request schedule to the physics work ledger.

    This is an explicit modeled-timing view: prefill and decode are consecutive,
    queue-free intervals derived from the supplied per-config throughput rates.
    """
    dt = float(dt)
    tp = int(tp)
    prefill_rate = float(throughput["lambda_prefill"])
    decode_rate = float(throughput["lambda_decode"])
    if dt <= 0.0 or tp < 1 or prefill_rate <= 0.0 or decode_rate <= 0.0:
        raise ValueError("dt, tp, and throughput rates must be positive")

    parsed: list[tuple[float, float, float, float, float]] = []
    horizon = 0.0
    for index, request in enumerate(requests):
        try:
            arrival = float(request["arrival_time"])
            n_in = float(request["input_tokens"])
            n_out = float(request["output_tokens"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"request[{index}] has invalid required fields") from exc
        if not np.all(np.isfinite([arrival, n_in, n_out])):
            raise ValueError(f"request[{index}] contains non-finite values")
        if arrival < 0.0 or n_in < 0.0 or n_out < 0.0:
            raise ValueError(f"request[{index}] values must be non-negative")
        prefill_end = arrival + n_in / prefill_rate
        decode_end = prefill_end + n_out / decode_rate
        parsed.append((arrival, prefill_end, decode_end, n_in, n_out))
        horizon = max(horizon, decode_end)

    if T is None:
        if not parsed:
            raise ValueError("empty request schedules require explicit T")
        T = int(math.ceil(horizon / dt))
    T = int(T)
    if T < 0:
        raise ValueError("T must be non-negative")

    edges = np.arange(T + 1, dtype=np.float64) * dt
    bin_lo, bin_hi = edges[:-1], edges[1:]
    pre_tok = np.zeros(T, dtype=np.float64)
    dec_tok = np.zeros(T, dtype=np.float64)
    batch = np.zeros(T, dtype=np.float64)
    pre_active = np.zeros(T, dtype=np.float64)
    pre_iter = np.zeros(T, dtype=np.float64)
    kv_read = np.zeros(T, dtype=np.float64)
    kv_tok = (
        2.0
        * float(arch["n_layers"])
        * float(arch["n_kv"])
        * float(arch["head_dim"])
        * KV_ELEM_BYTES
    )
    swa = float(arch.get("swa_window", 0.0))

    for arrival, prefill_end, decode_end, n_in, n_out in parsed:
        prefill_duration = prefill_end - arrival
        decode_duration = decode_end - prefill_end
        overlap_pre = np.clip(
            np.minimum(prefill_end, bin_hi) - np.maximum(arrival, bin_lo),
            0.0,
            None,
        )
        overlap_decode = np.clip(
            np.minimum(decode_end, bin_hi) - np.maximum(prefill_end, bin_lo),
            0.0,
            None,
        )
        if prefill_duration > 0.0:
            pre_tok += prefill_rate * overlap_pre / dt
            pre_active += overlap_pre / dt
            pre_iter += overlap_pre / prefill_duration / dt
        if decode_duration > 0.0:
            dec_tok += decode_rate * overlap_decode / dt
            batch += overlap_decode / dt
            progress = np.clip(
                ((bin_lo + bin_hi) / 2.0 - prefill_end) / decode_duration,
                0.0,
                1.0,
            )
            context = n_in + progress * n_out
            context_effective = (
                context
                if swa <= 0.0
                else 0.5 * context + 0.5 * np.minimum(context, swa)
            )
            kv_read += (
                decode_rate * (overlap_decode / dt) * context_effective * kv_tok
            )

    return bin_work_rates(
        pre_tok,
        dec_tok,
        batch,
        pre_active,
        pre_iter,
        kv_read,
        arch,
        tp,
        T,
    )


def run_physics_inference(
    *,
    config_id: str,
    requests_json: str,
    physics_artifact: str,
    throughput_db: str,
    out_csv: str,
    dt: float | None = None,
    T: int | None = None,
) -> dict[str, object]:
    match = CONFIG_RE.fullmatch(str(config_id).strip())
    if match is None:
        raise ValueError(f"invalid config_id for physics inference: {config_id!r}")
    model, hardware, tp_text = match.groups()
    tp = int(tp_text)
    artifact = load_physics_artifact(physics_artifact)
    architectures = artifact.get("architectures", {})
    arch = dict(architectures[model]) if model in architectures else get_arch(model)
    throughput_payload = load_json(throughput_db)
    throughput = resolve_throughput(throughput_payload, config_id)
    requests = load_request_schedule(requests_json)
    resolved_dt = float(artifact["dt_s"] if dt is None else dt)
    ledger = build_modeled_work_ledger(
        requests,
        arch=arch,
        tp=tp,
        throughput=throughput,
        dt=resolved_dt,
        T=T,
    )
    power = predict_mean_node_power(
        ledger,
        arch,
        tp=tp,
        hardware=hardware,
        artifact=artifact,
        dt_s=resolved_dt,
    )

    output = Path(out_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["t_bin", "time_s", "power_w", "generation_mode"],
        )
        writer.writeheader()
        for index, value in enumerate(power):
            writer.writerow(
                {
                    "t_bin": index,
                    "time_s": index * resolved_dt,
                    "power_w": float(value),
                    "generation_mode": "physics_modeled_mean",
                }
            )

    manifest_path = f"{out_csv}.manifest.json"
    manifest = {
        "schema_version": "powertrace-physics-inference-v1",
        "source_revision": git_state(),
        "config_id": config_id,
        "generation_mode": "physics_modeled_mean",
        "timing_mode": "arrival_only_modeled_throughput",
        "dt": resolved_dt,
        "T": int(power.size),
        "inputs": {
            "physics_artifact": file_identity(physics_artifact),
            "throughput_db": file_identity(throughput_db),
            "requests": file_identity(requests_json),
        },
        "output": file_identity(out_csv),
    }
    write_json(manifest_path, manifest)
    return {
        "config_id": config_id,
        "generation_mode": manifest["generation_mode"],
        "timing_mode": manifest["timing_mode"],
        "dt": resolved_dt,
        "T": int(power.size),
        "out_csv": out_csv,
        "inference_manifest": manifest_path,
    }
