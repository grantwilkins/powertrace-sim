"""RunRecord: the one normalized ingestion contract (data-path Layer 1).

One reader per raw layout produces the same record; feature views (GRU
active-request features, ledger work rates) consume it. Per-GPU power is the
stored truth (D4); the node/TP-group sum is an accessor. Raw recorded request
timestamps are kept so each view applies its own named alignment policy (D5).

Readers:
- ``load_legacy_run``   — data/sharegpt-benchmark-* pairs (identity from names)
- ``load_bundle_run``   — data/runs/<campaign>/<run_id>/ bundles (Phase D)
"""

from __future__ import annotations

import hashlib
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from model.training_data.alignment import align_trace_to_grid
from model.training_data.arch import arch_from_manifest, get_arch
from model.training_data.power_parsing import (
    parse_power_csv,
    parse_power_csv_per_gpu,
    parse_request_json,
    tp_sum_power,
)


@dataclass
class RunRecord:
    # identity
    config_id: str
    model: str
    hardware: str
    tp: int
    gpus_per_node: int
    source_layout: str  # "sharegpt" | "bundle"
    clock_basis: str

    # device table (native resolution; per-GPU is the stored truth)
    power_timestamps: np.ndarray                  # (n,) epoch seconds
    power_per_gpu: Optional[np.ndarray]           # (n, gpus_per_node) watts
    util_per_gpu: Optional[np.ndarray]            # (n, gpus_per_node) percent or NaN
    mem_per_gpu: Optional[np.ndarray]             # (n, gpus_per_node) MiB or NaN
    node_power: Optional[np.ndarray]              # (n,) only for pre-aggregated sources
    device_ids: Tuple[str, ...]
    device_table: Dict[str, np.ndarray]

    # requests table (validated rows; raw recorded epoch timestamps)
    input_lens: np.ndarray
    output_lens: np.ndarray
    ttfts: np.ndarray
    itls: np.ndarray
    decode_times: np.ndarray
    request_timestamps: np.ndarray
    has_timestamps: bool
    timestamp_source: str  # "recorded" | "missing"
    # Raw list-valued request columns, before the validated model projection.
    # Their source lengths may differ; projection counts live in provenance.
    request_table: Dict[str, np.ndarray]
    engine_table: Dict[str, np.ndarray]

    arch: Dict[str, object]
    provenance: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tp < 1 or self.gpus_per_node < 1 or self.tp > self.gpus_per_node:
            raise ValueError(
                f"Invalid GPU topology: tp={self.tp}, gpus_per_node={self.gpus_per_node}"
            )
        n_power = int(np.asarray(self.power_timestamps).size)
        if n_power < 2:
            raise ValueError("RunRecord requires at least two power samples")
        power_timestamps = np.asarray(self.power_timestamps, dtype=np.float64)
        if not np.all(np.isfinite(power_timestamps)) or not np.all(np.diff(power_timestamps) > 0):
            raise ValueError("RunRecord power timestamps must be finite and strictly increasing")
        if self.power_per_gpu is not None:
            power_per_gpu = np.asarray(self.power_per_gpu, dtype=np.float64)
            shape = power_per_gpu.shape
            if shape != (n_power, self.gpus_per_node):
                raise ValueError(
                    f"power_per_gpu shape {shape} != {(n_power, self.gpus_per_node)}"
                )
            if len(self.device_ids) != self.gpus_per_node:
                raise ValueError("RunRecord device IDs do not match GPU topology")
            if len(set(self.device_ids)) != len(self.device_ids):
                raise ValueError("RunRecord device IDs must be unique")
            if np.any(np.isinf(power_per_gpu)):
                raise ValueError("RunRecord power values must not be infinite")
            for name, values in self.device_table.items():
                if np.asarray(values).shape != shape:
                    raise ValueError(f"Device column {name!r} has wrong shape")
        elif self.node_power is None or np.asarray(self.node_power).size != n_power:
            raise ValueError("RunRecord requires aligned per-GPU or node power")
        elif np.any(np.isinf(np.asarray(self.node_power, dtype=np.float64))):
            raise ValueError("RunRecord power values must not be infinite")
        request_lengths = {
            np.asarray(values).size
            for values in (
                self.input_lens,
                self.output_lens,
                self.ttfts,
                self.itls,
                self.decode_times,
                self.request_timestamps,
            )
        }
        if len(request_lengths) != 1:
            raise ValueError("RunRecord request columns have different lengths")
        engine_lengths = {np.asarray(values).shape[0] for values in self.engine_table.values()}
        if len(engine_lengths) > 1:
            raise ValueError("RunRecord engine columns have different lengths")
        if self.source_layout == "bundle":
            engine_timestamps = np.asarray(
                self.engine_table.get("timestamp", []), dtype=np.float64
            )
            if engine_timestamps.size == 0:
                raise ValueError("Bundle engine.csv requires at least one data row")
            if not np.all(np.isfinite(engine_timestamps)) or not np.all(
                np.diff(engine_timestamps) > 0.0
            ):
                raise ValueError("Bundle engine timestamps must be finite and increasing")

    def tp_sum_power(self) -> np.ndarray:
        """Node-level TP-group power (watts); accessor, never stored truth."""
        if self.power_per_gpu is not None:
            return tp_sum_power(self.power_per_gpu, self.tp)
        if self.node_power is not None:
            return np.asarray(self.node_power, dtype=np.float64)
        raise ValueError("RunRecord has neither per-GPU nor node power")


def load_legacy_run(
    pair_row: Dict[str, str],
    *,
    gpus_per_node: int = 8,
    require_request_timestamps: bool = True,
    require_arch: bool = True,
) -> Optional[RunRecord]:
    """Read one matched stage0 pair (legacy sharegpt layout) into a RunRecord.

    Returns None on any parse failure, mirroring the legacy builders' skip
    behavior; callers keep their own error accounting.
    """
    power_csv = str(pair_row.get("power_csv_path", "")).strip()
    json_path = str(pair_row.get("json_path", "")).strip()
    model = str(pair_row.get("model_name", "")).strip()
    hardware = str(pair_row.get("hardware", "")).strip()
    try:
        tp = int(str(pair_row.get("tensor_parallelism", "")).strip())
    except ValueError:
        return None
    if not (power_csv and json_path and model and hardware and tp >= 1):
        return None
    if not (Path(power_csv).is_file() and Path(json_path).is_file()):
        return None

    per_gpu = parse_power_csv_per_gpu(power_csv, gpus_per_node=gpus_per_node)
    node_power = None
    if per_gpu is None:
        # Pre-aggregated trace (no per-device rows): keep the legacy collapse.
        collapsed = parse_power_csv(
            power_csv, tensor_parallelism=tp, gpus_per_node=gpus_per_node
        )
        if collapsed is None:
            return None
        power_timestamps = collapsed["timestamps"]
        node_power = collapsed["power"]
    else:
        power_timestamps = per_gpu["timestamps"]

    requests = parse_request_json(
        json_path, require_request_timestamps=require_request_timestamps
    )
    if requests is None:
        return None

    arch = get_arch(model) if require_arch else {}

    return RunRecord(
        config_id=f"{model}_{hardware}_tp{tp}",
        model=model,
        hardware=hardware,
        tp=tp,
        gpus_per_node=int(gpus_per_node),
        source_layout="sharegpt",
        clock_basis="naive_local_as_utc",
        power_timestamps=power_timestamps,
        power_per_gpu=None if per_gpu is None else per_gpu["power_per_gpu"],
        util_per_gpu=None if per_gpu is None else per_gpu["util_per_gpu"],
        mem_per_gpu=None if per_gpu is None else per_gpu["mem_per_gpu"],
        node_power=node_power,
        device_ids=() if per_gpu is None else per_gpu["device_ids"],
        device_table={} if per_gpu is None else per_gpu["device_table"],
        input_lens=requests["input_lens"],
        output_lens=requests["output_lens"],
        ttfts=requests["ttfts"],
        itls=requests["itls"],
        decode_times=requests["decode_times"],
        request_timestamps=requests["request_timestamps"],
        has_timestamps=bool(requests["has_timestamps"]),
        timestamp_source="recorded" if requests["has_timestamps"] else "missing",
        request_table=requests["request_table"],
        engine_table={},
        arch=arch,
        provenance={
            "pair_key": str(pair_row.get("pair_key", "")).strip(),
            "power_csv_path": power_csv,
            "json_path": json_path,
            "sha256": {
                "power_csv": _sha256(Path(power_csv)),
                "requests_json": _sha256(Path(json_path)),
            },
            "rate": str(pair_row.get("rate", "")).strip(),
            "request_projection": requests["stats"],
            "request_projection_indices": requests["projection_indices"].tolist(),
            "request_rows": _request_row_lineage(requests["stats"]),
            "arch_source": "registry" if require_arch else "omitted_for_gru_projection",
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_row_lineage(stats: Dict[str, object]) -> Dict[str, object]:
    return {
        "source": int(stats["num_requests_raw"]),
        "source_lengths": dict(stats["source_lengths"]),
        "request_column_lengths": dict(stats["request_column_lengths"]),
        "aligned": int(stats["num_requests_aligned"]),
        "retained": int(stats["num_requests_used"]),
        "dropped": {
            "unaligned": int(stats["num_requests_dropped_unaligned"]),
            "invalid_fields": int(stats["num_requests_dropped_invalid_fields"]),
            "decode_time": int(stats["num_requests_dropped_decode_time"]),
            "timestamp": int(stats["num_requests_dropped_timestamp"]),
        },
    }


def _read_csv_table(path: Path) -> Dict[str, np.ndarray]:
    """Preserve every engine column, using numeric arrays when possible."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        columns = {name.strip(): [] for name in reader.fieldnames}
        for row in reader:
            for raw_name in reader.fieldnames:
                columns[raw_name.strip()].append((row.get(raw_name) or "").strip())
    table = {}
    for name, values in columns.items():
        try:
            table[name] = np.asarray(values, dtype=np.float64)
        except ValueError:
            table[name] = np.asarray(values, dtype=object)
    return table


def load_bundle_run(run_dir: str | Path) -> RunRecord:
    """Read one canonical ``data/runs/<campaign>/<run>`` bundle."""
    root = Path(run_dir)
    paths = {
        name: root / name
        for name in ("manifest.json", "power.csv", "engine.csv", "requests.json")
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"Incomplete bundle {root}: missing {', '.join(missing)}")

    manifest = json.loads(paths["manifest.json"].read_text())
    model = str(manifest.get("model", "")).strip()
    hardware = str(manifest.get("hardware", "")).strip()
    tp = int(manifest["tp"])
    gpus_per_node = int(manifest["gpus_per_node"])
    if not model or not hardware:
        raise ValueError("Bundle manifest requires model and hardware")

    clock = manifest.get("clock")
    if not isinstance(clock, dict) or "local_utc_offset_s" not in clock:
        raise ValueError("Bundle manifest requires clock.local_utc_offset_s")
    local_utc_offset_s = float(clock["local_utc_offset_s"])
    power = parse_power_csv_per_gpu(
        str(paths["power.csv"]), gpus_per_node=gpus_per_node,
        strict_topology=True, local_utc_offset_s=local_utc_offset_s,
    )
    if power is None:
        raise ValueError("Bundle power.csv is not a complete per-GPU stream")
    active_gpu_uuids = (manifest.get("server") or {}).get("active_gpu_uuids")
    if active_gpu_uuids is not None:
        active_gpu_uuids = tuple(str(value) for value in active_gpu_uuids)
        if len(active_gpu_uuids) != tp:
            raise ValueError("Bundle active_gpu_uuids count must equal tp")
        if set(active_gpu_uuids) != set(power["device_ids"][:tp]):
            raise ValueError(
                "Bundle active_gpu_uuids do not match the TP-group power columns"
            )
    requests = parse_request_json(str(paths["requests.json"]))
    if requests is None:
        raise ValueError("Bundle requests.json does not satisfy the request contract")

    engine_table = _read_csv_table(paths["engine.csv"])

    return RunRecord(
        config_id=f"{model}_{hardware}_tp{tp}",
        model=model,
        hardware=hardware,
        tp=tp,
        gpus_per_node=gpus_per_node,
        source_layout="bundle",
        clock_basis="power_local_wall_time_corrected_to_epoch",
        power_timestamps=power["timestamps"],
        power_per_gpu=power["power_per_gpu"],
        util_per_gpu=power["util_per_gpu"],
        mem_per_gpu=power["mem_per_gpu"],
        node_power=None,
        device_ids=power["device_ids"],
        device_table=power["device_table"],
        input_lens=requests["input_lens"],
        output_lens=requests["output_lens"],
        ttfts=requests["ttfts"],
        itls=requests["itls"],
        decode_times=requests["decode_times"],
        request_timestamps=requests["request_timestamps"],
        has_timestamps=True,
        timestamp_source="recorded",
        request_table=requests["request_table"],
        engine_table=engine_table,
        arch=arch_from_manifest(manifest),
        provenance={
            "run_id": str(manifest.get("run_id", root.name)),
            "paths": {name: str(path) for name, path in paths.items()},
            "sha256": {name: _sha256(path) for name, path in paths.items()},
            "probe": manifest.get("probe", {}),
            "clock": clock,
            "request_projection": requests["stats"],
            "request_projection_indices": requests["projection_indices"].tolist(),
            "request_rows": _request_row_lineage(requests["stats"]),
        },
    )


def gru_view_from_record(record: RunRecord) -> Optional[Dict[str, object]]:
    """Build the existing GRU ``A_t/dA_t`` input view from a run record."""
    return align_trace_to_grid(
        {"timestamps": record.power_timestamps, "power": record.tp_sum_power()},
        {
            "request_timestamps": record.request_timestamps,
            "ttfts": record.ttfts,
            "decode_times": record.decode_times,
            "input_lens": record.input_lens,
            "output_lens": record.output_lens,
            "has_timestamps": record.has_timestamps,
        },
    )
