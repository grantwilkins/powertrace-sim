from __future__ import annotations

import concurrent.futures
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from model.simulators.arrival_simulator import (
    ServingConfig,
    create_deepseek_config,
    create_llama_config,
)
from model.simulators.server_power_simulator import ServerPowerSimulator

# Constants
NODES_PER_RACK = 4
GPUS_PER_NODE = 8
ALLOWED_TP = {1, 2, 4, 8}
NODE_TDP_KW = {"A100": 6.5, "H100": 10.2}
RACK_TDP_KW = {
    "A100": NODE_TDP_KW["A100"] * NODES_PER_RACK,
    "H100": NODE_TDP_KW["H100"] * NODES_PER_RACK,
}

# Constant non-GPU overhead per node (e.g., CPU, fans, NICs, etc.)
NODE_OVERHEAD_KW = 1.2


@dataclass
class RowSpec:
    name: str
    capacity_kw: float
    num_racks_a100: int
    num_racks_h100: int


@dataclass
class NodePlacement:
    node_id: int
    row_name: str
    rack_id: str
    hardware: str  # "A100" or "H100"
    job_key: Optional[str] = None  # e.g., "llama-3-8b_h100_tp1"


def parse_job_key(job_key: str) -> Tuple[str, int, str, int]:
    """
    Parse job key like 'llama-3-8b_h100_tp1' to (family_name, size_b, hardware, tp).
    family_name examples: 'llama-3', 'deepseek-r1-distill'
    """
    parts = job_key.split("_")
    if len(parts) != 3 or not parts[2].startswith("tp"):
        raise ValueError(f"Invalid job key: {job_key}")
    model_full = parts[0]
    hardware = parts[1].upper()
    tp = int(parts[2].replace("tp", ""))
    if not model_full.endswith("b"):
        raise ValueError(f"Invalid model size suffix in job key: {job_key}")
    # Extract size at the end "-<N>b"
    try:
        size_b = int(model_full.split("-")[-1].replace("b", ""))
    except Exception as e:
        raise ValueError(f"Invalid size in job key: {job_key}") from e
    # Family is everything before the trailing "-<N>b"
    family = "-".join(model_full.split("-")[:-1])
    return family, size_b, hardware, tp


def model_category_and_config(
    family: str, size_b: int, tp: int, hardware: str
) -> Tuple[str, ServingConfig]:
    """
    Map family to ServeGen category and construct ServingConfig.
    - category: llama -> language; deepseek-r1-distill -> reason
    - model_type: m-mid (fixed)
    """
    if "llama" in family:
        category = "language"
        config = create_llama_config(size_b=size_b, tp=tp, hardware=hardware)
    elif "deepseek" in family:
        category = "reason"
        config = create_deepseek_config(size_b=size_b, tp=tp, hardware=hardware)
    else:
        raise ValueError(f"Unknown family: {family}")
    return category, config


def make_diurnal_rate_fn(
    r_min_qps: float,
    r_max_qps: float,
    start_of_day_sec: float,
    per_replica_scale: float,
    jitter_std_frac: float,
    seed: int,
    *,
    period_seconds: float = 86400.0,
    peak_time_sec: float = 13 * 3600.0,  # Peak near 1pm local time
    sharpness_gamma: float = 1.8,  # >1 sharpens midday peak / deepens troughs
) -> Callable[[float], float]:
    """
    Build a strong diurnal rate function r(t) with configurable peak and shape.

    Normalized diurnal component:
        x = 0.5 * (1 + cos(theta)), where theta is phased to peak at `peak_time_sec`.
        x in [0,1]; use x**gamma (gamma>1) to accentuate peak/valleys.

    Final:
        r(t) = r_min + (r_max - r_min) * (x ** sharpness_gamma) * per_replica_scale * jitter
    """
    rng = np.random.RandomState(seed)
    jitter = float(max(0.1, rng.normal(1.0, jitter_std_frac)))
    period = max(60.0, float(period_seconds))
    # Phase offset so peak occurs at peak_time_sec
    peak_phase = ((peak_time_sec - start_of_day_sec) % period) / period

    def rate_fn(t: float) -> float:
        tod = (start_of_day_sec + max(0.0, t)) % period
        phase = (tod / period) - peak_phase
        theta = 2.0 * math.pi * phase
        x = 0.5 * (1.0 + math.cos(theta))  # [0,1], max at peak
        shaped = x**sharpness_gamma
        base = r_min_qps + (r_max_qps - r_min_qps) * shaped
        return max(0.0, base * per_replica_scale * jitter)

    return rate_fn


class DataCenterSimulator:
    def __init__(self, rows: List[RowSpec]):
        self.rows = rows
        self.nodes: List[NodePlacement] = []
        self._build_topology()

    @staticmethod
    def _simulate_node_worker(
        node: NodePlacement,
        duration: float,
        diurnal_min_qps_per_server: float,
        diurnal_max_qps_per_server: float,
        start_of_day_sec: float,
        output_dt: float,
        base_seed: int,
        fast_workload: bool,
        diurnal_sharpness_gamma: float,
        sample_node_id: Optional[int] = None,
        sample_server_replica_index: Optional[int] = None,
    ) -> Tuple[str, str, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
        """
        Simulate a single node and return
        (row_name, rack_id, node_watts, tokens_in, tokens_out, node_id, sampled_server_watts).
        """
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        # Skip if unassigned
        if not node.job_key:
            z = np.zeros_like(np.arange(0.0, duration + 1e-9, output_dt))
            return node.row_name, node.rack_id, z, z, z, node.node_id, z

        family, size_b, hardware, tp = parse_job_key(node.job_key)
        if tp not in ALLOWED_TP or GPUS_PER_NODE % tp != 0:
            raise ValueError(f"Unsupported TP={tp} for node {node.node_id}")
        z_replicas = GPUS_PER_NODE // tp
        category, serving_config = model_category_and_config(
            family, size_b, tp, hardware
        )
        server_sim = ServerPowerSimulator(serving_config)

        ts_global = np.arange(0.0, duration + 1e-9, output_dt)
        node_watts = np.zeros_like(ts_global)
        node_tok_in = np.zeros_like(ts_global)
        node_tok_out = np.zeros_like(ts_global)

        server_series = np.zeros_like(ts_global)
        for rep in range(z_replicas):
            per_replica_scale = 1.0 / float(z_replicas)
            rate_fn = make_diurnal_rate_fn(
                r_min_qps=diurnal_min_qps_per_server,
                r_max_qps=diurnal_max_qps_per_server,
                start_of_day_sec=start_of_day_sec,
                per_replica_scale=per_replica_scale,
                jitter_std_frac=0.05,
                seed=base_seed + node.node_id * 97 + rep,
                sharpness_gamma=diurnal_sharpness_gamma,
            )
            res = server_sim.simulate_server_power(
                category=category,
                model_type="m-mid",
                duration=duration,
                rate_requests_per_sec=0.0,
                seed=base_seed + node.node_id,
                output_dt=output_dt,
                return_profile=False,
                rate_fn=rate_fn,
                use_fast_workload=fast_workload,
            )
            ts = np.asarray(res["timestamps"], dtype=float)
            watts = np.asarray(res["watts"], dtype=float)
            if len(ts) == 0:
                continue
            ts_rel = ts - ts[0]
            # Hold boundary values at edges to avoid artificial dips at start/end
            left_val = float(watts[0])
            right_val = float(watts[-1])
            watts_interp = np.interp(
                ts_global, ts_rel, watts, left=left_val, right=right_val
            )
            node_watts += watts_interp
            # Capture a single server replica if requested
            if sample_node_id is not None and node.node_id == sample_node_id:
                target_rep = (
                    0
                    if sample_server_replica_index is None
                    else int(max(0, min(z_replicas - 1, sample_server_replica_index)))
                )
                if rep == target_rep:
                    server_series = server_series + watts_interp
            # tokens aggregation
            if "tokens_in" in res and "tokens_out" in res:
                tok_in = np.asarray(res["tokens_in"], dtype=float)
                tok_out = np.asarray(res["tokens_out"], dtype=float)
                cum_in = np.cumsum(tok_in)
                cum_out = np.cumsum(tok_out)
                cum_in_interp = np.interp(
                    ts_global, ts_rel, cum_in, left=0.0, right=float(cum_in[-1])
                )
                cum_out_interp = np.interp(
                    ts_global, ts_rel, cum_out, left=0.0, right=float(cum_out[-1])
                )
                in_series = np.concatenate([[cum_in_interp[0]], np.diff(cum_in_interp)])
                out_series = np.concatenate(
                    [[cum_out_interp[0]], np.diff(cum_out_interp)]
                )
                node_tok_in += in_series
                node_tok_out += out_series

        # Add constant per-node overhead, then clip to node TDP
        node_tdp_w = NODE_TDP_KW[hardware] * 1000.0
        node_watts_with_overhead = node_watts + NODE_OVERHEAD_KW * 1000.0
        node_watts_clipped = np.minimum(node_watts_with_overhead, node_tdp_w)
        return (
            node.row_name,
            node.rack_id,
            node_watts_clipped,
            node_tok_in,
            node_tok_out,
            node.node_id,
            server_series,
        )

    def _build_topology(self) -> None:
        node_id = 0
        for row_idx, row in enumerate(self.rows):
            # A100 racks
            for r in range(row.num_racks_a100):
                rack_id = f"row{row_idx + 1}:A100:{r + 1}"
                for n in range(NODES_PER_RACK):
                    self.nodes.append(
                        NodePlacement(
                            node_id=node_id,
                            row_name=row.name,
                            rack_id=rack_id,
                            hardware="A100",
                        )
                    )
                    node_id += 1
            # H100 racks
            for r in range(row.num_racks_h100):
                rack_id = f"row{row_idx + 1}:H100:{r + 1}"
                for n in range(NODES_PER_RACK):
                    self.nodes.append(
                        NodePlacement(
                            node_id=node_id,
                            row_name=row.name,
                            rack_id=rack_id,
                            hardware="H100",
                        )
                    )
                    node_id += 1

    def assign_jobs(self, job_mix: Dict[str, float]) -> None:
        """Assign node-level jobs according to percentages over job keys.

        job key examples from best_weights directory names:
        - 'llama-3-8b_h100_tp1'
        - 'deepseek-r1-distill-70b_a100_tp8'
        """
        # Normalize mix
        total = sum(max(0.0, v) for v in job_mix.values())
        if total <= 0:
            raise ValueError("job_mix must have positive mass")
        mix = {k: v / total for k, v in job_mix.items()}

        # Nodes by hardware
        a100_nodes = [n for n in self.nodes if n.hardware == "A100"]
        h100_nodes = [n for n in self.nodes if n.hardware == "H100"]

        # Desired counts per job across all nodes (soft - may not fit by hardware)
        desired_counts = {k: int(round(mix[k] * len(self.nodes))) for k in mix}

        # Create per-hardware job lists
        jobs_a100: List[str] = []
        jobs_h100: List[str] = []
        for key, cnt in desired_counts.items():
            family, size_b, hw, tp = parse_job_key(key)
            if hw not in ("A100", "H100"):
                raise ValueError(f"Invalid hardware in job key: {key}")
            target_list = jobs_a100 if hw == "A100" else jobs_h100
            target_list.extend([key] * max(0, cnt))

        # Soft placement: fill node lists; leftover are dropped silently; report counts
        for node, job_key in zip(a100_nodes, jobs_a100):
            node.job_key = job_key
        for node, job_key in zip(h100_nodes, jobs_h100):
            node.job_key = job_key

    def simulate(
        self,
        duration: float,
        diurnal_min_qps_per_server: float,
        diurnal_max_qps_per_server: float,
        start_of_day_sec: float = 0.0,
        output_dt: float = 1.0,
        base_seed: int = 0,
        return_profiles: bool = False,
        fast_workload: bool = True,
        num_workers: Optional[int] = None,
        diurnal_sharpness_gamma: float = 1.8,
        return_per_rack: bool = False,
        sample_row_name: Optional[str] = None,
        sample_rack_id: Optional[str] = None,
        sample_node_id: Optional[int] = None,
        sample_server_replica_index: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run per-node simulations and aggregate to per-row and total DC power.

        Returns dict with keys:
        - timestamps
        - per_row: Dict[row_name] -> watts array
        - datacenter_watts
        - overage: Dict[row_name] -> pre_clip_minus_post_clip array (row capacity)
        - metadata: placements, counts
        """
        ts_global = np.arange(0.0, duration + 1e-9, output_dt)
        rows = {row.name: np.zeros_like(ts_global) for row in self.rows}
        # Optional: per-rack aggregation (unclipped at row capacity)
        per_rack_by_row: Dict[str, Dict[str, np.ndarray]] = (
            {row.name: {} for row in self.rows} if return_per_rack else {}
        )
        rows_tokens_in = {row.name: np.zeros_like(ts_global) for row in self.rows}
        rows_tokens_out = {row.name: np.zeros_like(ts_global) for row in self.rows}
        row_capacity = {row.name: row.capacity_kw * 1000.0 for row in self.rows}  # W
        row_overage = {row.name: np.zeros_like(ts_global) for row in self.rows}

        max_workers = (
            num_workers
            if num_workers and num_workers > 0
            else max(1, min(8, (os.cpu_count() or 2) // 2))
        )

        for row in self.rows:
            nodes_in_row = [n for n in self.nodes if n.row_name == row.name]
            if not nodes_in_row:
                continue

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(
                        DataCenterSimulator._simulate_node_worker,
                        node,
                        duration,
                        diurnal_min_qps_per_server,
                        diurnal_max_qps_per_server,
                        start_of_day_sec,
                        output_dt,
                        base_seed,
                        fast_workload,
                        diurnal_sharpness_gamma,
                        sample_node_id,
                        sample_server_replica_index,
                    )
                    for node in nodes_in_row
                ]

                sampled_node = None
                sampled_server = None
                for fut in concurrent.futures.as_completed(futures):
                    (
                        row_name,
                        rack_id,
                        node_series,
                        in_series,
                        out_series,
                        node_id,
                        server_series,
                    ) = fut.result()
                    rows[row_name] += node_series
                    rows_tokens_in[row_name] += in_series
                    rows_tokens_out[row_name] += out_series
                    if return_per_rack:
                        rack_map = per_rack_by_row[row_name]
                        if rack_id not in rack_map:
                            rack_map[rack_id] = np.zeros_like(ts_global)
                        rack_map[rack_id] += node_series
                    # Capture sampled node and server if requested
                    if sample_node_id is not None and node_id == sample_node_id:
                        sampled_node = node_series
                        sampled_server = server_series

        for row in self.rows:
            w = rows[row.name]
            cap = row_capacity[row.name]
            over = np.clip(w - cap, a_min=0.0, a_max=None)
            rows[row.name] = np.minimum(w, cap)
            row_overage[row.name] = over

        dc_watts = np.zeros_like(ts_global)
        dc_tokens_in = np.zeros_like(ts_global)
        dc_tokens_out = np.zeros_like(ts_global)
        for name in rows:
            dc_watts += rows[name]
            dc_tokens_in += rows_tokens_in[name]
            dc_tokens_out += rows_tokens_out[name]

        result = {
            "timestamps": ts_global,
            "per_row": rows,
            "datacenter_watts": dc_watts,
            "overage": row_overage,
            "per_row_tokens_in": rows_tokens_in,
            "per_row_tokens_out": rows_tokens_out,
            "datacenter_tokens_in": dc_tokens_in,
            "datacenter_tokens_out": dc_tokens_out,
            "metadata": {
                "num_nodes": len(self.nodes),
                "placements": [
                    {
                        "node_id": n.node_id,
                        "row": n.row_name,
                        "rack": n.rack_id,
                        "hardware": n.hardware,
                        "job_key": n.job_key,
                    }
                    for n in self.nodes
                ],
            },
        }
        if return_per_rack:
            result["per_rack_by_row"] = per_rack_by_row
        # Optional sampled outputs
        if sample_row_name and sample_row_name in rows:
            result["sampled_row_watts"] = rows[sample_row_name]
        if sample_rack_id and return_per_rack:
            # Only if per-rack was requested
            for row_name, rack_map in per_rack_by_row.items():
                if sample_rack_id in rack_map:
                    result["sampled_rack_watts"] = rack_map[sample_rack_id]
                    break
        if sample_node_id is not None:
            # sampled_node and sampled_server captured in loop scope; may be None if node unassigned
            if "sampled_node" in locals() and sampled_node is not None:
                result["sampled_node_watts"] = sampled_node
            if "sampled_server" in locals() and sampled_server is not None:
                result["sampled_server_watts"] = sampled_server
        return result
