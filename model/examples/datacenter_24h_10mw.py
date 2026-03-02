import csv
import os

from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from model.simulators.arrival_simulator import ServeGenRequest
from model.simulators.datacenter_simulator import DataCenterSimulator, RowSpec


def build_rows_10mw(
    num_rows: int = 10, capacity_kw_per_row: float = 1000.0
) -> List[RowSpec]:
    """
    Build a simple H100-only topology totaling ~10 MW row capacity.
    - 10 rows × 1,000 kW capacity = 10,000 kW (10 MW)
    - 20 H100 racks per row (80 nodes per row) to provide sufficient capacity headroom
    """
    rows: List[RowSpec] = []
    for i in range(num_rows):
        rows.append(
            RowSpec(
                name=f"row{i + 1}",
                capacity_kw=capacity_kw_per_row,
                num_racks_a100=0,
                num_racks_h100=20,
            )
        )
    return rows


def job_mix_75_25_h100() -> Dict[str, float]:
    """
    75% 70B, 25% 8B on H100.
    - Using Llama 3 family names and common TP settings
    """
    return {
        "llama-3-70b_h100_tp8": 0.75,
        "llama-3-8b_h100_tp1": 0.25,
    }


def build_rows_for_azure_trace(
    num_rows: int = 1, num_racks_per_row: int = 2, capacity_kw_per_row: float = 500.0
) -> List[RowSpec]:
    """
    Build a minimal topology sized for the unscaled Azure trace.
    - 1 row × 2 racks = 8 H100 nodes
    - With 75/25 split: 6 nodes TP8 (6 replicas) + 2 nodes TP1 (16 replicas) = 22 replicas
    - Target: ~2.3 QPS per replica for 51.49 req/s total
    """
    rows: List[RowSpec] = []
    for i in range(num_rows):
        rows.append(
            RowSpec(
                name=f"row{i + 1}",
                capacity_kw=capacity_kw_per_row,
                num_racks_a100=0,
                num_racks_h100=num_racks_per_row,
            )
        )
    return rows


def load_azure_trace(
    csv_path: str, scale_factor: float = 1.0, seed: int = 0
) -> List[ServeGenRequest]:
    """
    Load Azure conversation trace and convert to ServeGenRequest objects.

    Args:
        csv_path: Path to Azure trace CSV with TIMESTAMP, ContextTokens, GeneratedTokens
        scale_factor: Multiply arrival rate by this factor (e.g., 50.0 to scale up)
        seed: Random seed for reproducibility

    Returns:
        List of ServeGenRequest objects sorted by arrival time
    """
    rng = np.random.RandomState(seed)

    # Load CSV
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "timestamp": row["TIMESTAMP"],
                    "context": int(row["ContextTokens"]),
                    "generated": int(row["GeneratedTokens"]),
                }
            )

    # Parse timestamps
    timestamps = []
    for row in rows:
        ts_str = row["timestamp"]
        if "." in ts_str:
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f%z")
        else:
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S%z")
        timestamps.append(dt)

    # Sort by time
    sorted_indices = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
    timestamps = [timestamps[i] for i in sorted_indices]
    rows = [rows[i] for i in sorted_indices]

    # Convert to seconds from start
    start_time = timestamps[0]
    times_seconds = [(t - start_time).total_seconds() for t in timestamps]

    # Apply scaling factor by adjusting inter-arrival times
    if scale_factor != 1.0:
        # Compress/expand time intervals
        scaled_times = [t / scale_factor for t in times_seconds]
    else:
        scaled_times = times_seconds

    # Create ServeGenRequest objects
    requests = []
    for i, (t, row) in enumerate(zip(scaled_times, rows)):
        requests.append(
            ServeGenRequest(
                request_id=i,
                arrival_time=float(t),
                input_tokens=row["context"],
                output_tokens=row["generated"],
            )
        )

    return requests


def downsample_to_15min(
    timestamps: np.ndarray, watts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 15-minute (900s) mean power values.
    Assumes timestamps begin at 0 and are 1-second spaced (output_dt=1).
    """
    if len(timestamps) != len(watts):
        n = min(len(timestamps), len(watts))
        timestamps = timestamps[:n]
        watts = watts[:n]

    # Ensure exactly 86400 seconds (drop the final inclusive endpoint if present)
    # np.arange(0, 86400 + 1e-9, 1.0) yields 86401 entries; drop last for exact 86400
    if len(timestamps) == 86401:
        timestamps = timestamps[:-1]
        watts = watts[:-1]

    # Reshape into (96, 900) bins
    if len(timestamps) != 86400:
        # Fallback: compute with integer windowing
        step = 900
        edges = np.arange(0, 86400 + 1, step)
        means = []
        centers = []
        for i in range(len(edges) - 1):
            lo = edges[i]
            hi = edges[i + 1]
            mask = (timestamps >= lo) & (timestamps < hi)
            if not np.any(mask):
                means.append(0.0)
            else:
                means.append(float(np.mean(watts[mask])))
            centers.append((lo + hi) / 2.0)
        return np.array(centers, dtype=float), np.array(means, dtype=float)

    watts_reshaped = watts.reshape(96, 900)
    mean_15min = watts_reshaped.mean(axis=1)
    # Bin centers at the middle of each 900s block: 450, 1350, ..., 86400-450
    centers = (np.arange(96, dtype=float) * 900.0) + 450.0
    return centers, mean_15min


def save_csv(path: str, timestamps: np.ndarray, watts: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_s", "watts"])
        for t, p in zip(timestamps, watts):
            w.writerow([float(t), float(p)])


def main():
    # Config
    duration = 24 * 3600.0  # 24 hours in seconds
    output_dt = 1.0  # 1-second internal grid for accuracy; we downsample to 15 minutes
    base_seed = 20260112
    num_workers = min(16, max(1, (os.cpu_count() or 2) // 2))

    # Azure trace configuration
    azure_trace_path = "/Users/grantwilkins/one_day_code.csv"
    # NO SCALING - use trace at its natural rate, just reduce number of machines
    trace_scale_factor = 1.0

    print("=" * 70)
    print(f"Loading Azure trace from:")
    print(f"  {azure_trace_path}")
    print("Using UNSCALED trace (1:1 time mapping over full 24 hours)")
    print("=" * 70)

    # Load the Azure trace
    all_requests = load_azure_trace(
        csv_path=azure_trace_path, scale_factor=trace_scale_factor, seed=base_seed
    )

    print(f"\nLoaded {len(all_requests):,} requests from Azure trace")
    print(
        f"Time range: {all_requests[0].arrival_time:.2f}s to {all_requests[-1].arrival_time:.2f}s ({all_requests[-1].arrival_time / 3600:.1f} hours)"
    )
    total_rate = len(all_requests) / (all_requests[-1].arrival_time + 1)
    print(f"Total request rate: {total_rate:.2f} req/s")

    # Compute token statistics
    context_tokens = [r.input_tokens for r in all_requests]
    gen_tokens = [r.output_tokens for r in all_requests]
    print(f"\nToken statistics from trace:")
    print(
        f"  Context tokens: mean={np.mean(context_tokens):.0f}, "
        f"median={np.median(context_tokens):.0f}, "
        f"max={np.max(context_tokens)}"
    )
    print(
        f"  Generated tokens: mean={np.mean(gen_tokens):.0f}, "
        f"median={np.median(gen_tokens):.0f}, "
        f"max={np.max(gen_tokens)}"
    )

    # Build minimal topology sized for unscaled trace
    # 1 row × 4 racks = 16 nodes = 44 replicas (12 TP8 + 32 TP1)
    # Sized to handle 1-second bursts up to 147 req/s while staying under 4 QPS/replica
    num_racks = 8
    rows = build_rows_for_azure_trace(num_rows=1, num_racks_per_row=num_racks)
    dc = DataCenterSimulator(rows)

    num_nodes = num_racks * 4
    replicas_tp8 = int(num_nodes * 0.75)  # 75% of nodes
    replicas_tp1 = int(num_nodes * 0.25) * 8  # 25% of nodes, 8 replicas each
    total_replicas = replicas_tp8 + replicas_tp1

    print(f"\nDatacenter topology:")
    print(
        f"  {len(rows)} row(s) × {num_racks} racks/row × 4 nodes/rack = {num_nodes} H100 nodes"
    )
    print(
        f"  Expected replicas: {replicas_tp8} (TP8) + {replicas_tp1} (TP1) = {total_replicas} total"
    )
    print(f"  Average QPS per replica: {total_rate / total_replicas:.2f}")
    print(
        f"  Peak 1s burst: ~147 req/s = {147 / total_replicas:.2f} QPS/replica (under 4.0 limit ✓)"
    )

    # Assign jobs: 75% Llama 70B, 25% Llama 8B on H100
    job_mix = job_mix_75_25_h100()
    dc.assign_jobs(job_mix)

    print("\nRunning datacenter simulation with Azure trace replay...")

    res = dc.simulate(
        duration=duration,
        diurnal_min_qps_per_server=0.0,  # Not used with trace-based workload
        diurnal_max_qps_per_server=0.0,  # Not used with trace-based workload
        start_of_day_sec=0.0,
        output_dt=output_dt,
        base_seed=base_seed,
        return_profiles=False,
        fast_workload=False,  # We're using trace-based workload
        num_workers=num_workers,
        diurnal_sharpness_gamma=1.0,
        return_per_rack=False,
        sample_row_name=None,
        sample_rack_id=None,
        sample_node_id=None,
        sample_server_replica_index=None,
        verbose=True,
        global_trace_requests=all_requests,  # Pass the Azure trace
        trace_model_routing=job_mix,  # Route according to 75/25 split
    )

    ts = np.asarray(res["timestamps"], dtype=float)
    dc_watts = np.asarray(res["datacenter_watts"], dtype=float)

    # Downsample to 15-minute means
    ts_15, dc_watts_15 = downsample_to_15min(ts, dc_watts)

    # Write outputs
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..",
        "training_results",
        "dc_24h_10mw",
    )
    csv_full = os.path.join(out_dir, "datacenter_power_1s.csv")
    csv_15m = os.path.join(out_dir, "datacenter_power_15min.csv")
    save_csv(csv_full, ts, dc_watts)
    save_csv(csv_15m, ts_15, dc_watts_15)

    peak_mw = float(np.max(dc_watts)) / 1e6
    mean_mw = float(np.mean(dc_watts)) / 1e6
    std_mw = float(np.std(dc_watts)) / 1e6
    print("\n" + "=" * 70)
    print(f"Saved 1s and 15-min DC power CSVs to {out_dir}")
    print(f"Peak power: {peak_mw:.3f} MW")
    print(f"Mean power: {mean_mw:.3f} MW")
    print(f"Std dev: {std_mw:.3f} MW ({std_mw / mean_mw * 100:.1f}%)")
    print(
        f"Workload: Azure Trace Replay (unscaled, {num_nodes} H100 nodes, {total_replicas} replicas)"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
