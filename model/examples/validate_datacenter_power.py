import csv
import os
import sys

import numpy as np

# Ensure package imports resolve when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulators.datacenter_simulator import DataCenterSimulator, RowSpec


def save_row_csvs(base_dir: str, ts: np.ndarray, per_row: dict, dc: np.ndarray):
    os.makedirs(base_dir, exist_ok=True)
    # Per-row
    for row_name, watts in per_row.items():
        path = os.path.join(base_dir, f"{row_name}_power.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_s", "watts"])
            for t, p in zip(ts, watts):
                w.writerow([float(t), float(p)])
    # Data center total
    path = os.path.join(base_dir, "datacenter_power.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_s", "watts"])
        for t, p in zip(ts, dc):
            w.writerow([float(t), float(p)])


def save_row_rack_csvs(
    base_dir: str, ts: np.ndarray, per_rack_by_row: dict, row_name: str
):
    os.makedirs(base_dir, exist_ok=True)
    if row_name not in per_rack_by_row:
        return
    rack_map = per_rack_by_row[row_name]
    for rack_id, watts in rack_map.items():
        safe_rack = rack_id.replace(":", "-")
        path = os.path.join(base_dir, f"{row_name}__{safe_rack}_power.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_s", "watts"])
            for t, p in zip(ts, watts):
                w.writerow([float(t), float(p)])


def save_series(path: str, ts: np.ndarray, series: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_s", "watts"])
        for t, p in zip(ts, series):
            w.writerow([float(t), float(p)])


def save_token_csvs(
    base_dir: str,
    ts: np.ndarray,
    per_row_tokens_in: dict,
    per_row_tokens_out: dict,
    dc_tokens_in: np.ndarray,
    dc_tokens_out: np.ndarray,
):
    os.makedirs(base_dir, exist_ok=True)
    # Per-row tokens in/out
    for row_name, tokens_in in per_row_tokens_in.items():
        # tokens_in
        path_in = os.path.join(base_dir, f"{row_name}_tokens_in.csv")
        with open(path_in, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_s", "tokens_in"])
            for t, v in zip(ts, tokens_in):
                w.writerow([float(t), float(v)])
        # tokens_out
        tokens_out = per_row_tokens_out[row_name]
        path_out = os.path.join(base_dir, f"{row_name}_tokens_out.csv")
        with open(path_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_s", "tokens_out"])
            for t, v in zip(ts, tokens_out):
                w.writerow([float(t), float(v)])

    # Datacenter tokens (both columns)
    dc_path = os.path.join(base_dir, "datacenter_tokens.csv")
    with open(dc_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_s", "tokens_in", "tokens_out"])
        for t, vin, vout in zip(ts, dc_tokens_in, dc_tokens_out):
            w.writerow([float(t), float(vin), float(vout)])


def main():
    # Simple pilot: 1 row with explicit rack counts
    rows = [
        RowSpec(name="row1", capacity_kw=600.0, num_racks_a100=10, num_racks_h100=5),
        # RowSpec(name="row2", capacity_kw=600.0, num_racks_a100=23, num_racks_h100=0),
        # RowSpec(name="row3", capacity_kw=600.0, num_racks_a100=23, num_racks_h100=0),
        # RowSpec(name="row4", capacity_kw=600.0, num_racks_a100=23, num_racks_h100=0),
        # RowSpec(name="row5", capacity_kw=600.0, num_racks_a100=23, num_racks_h100=0),
        # RowSpec(name="row6", capacity_kw=600.0, num_racks_a100=23, num_racks_h100=0),
        # RowSpec(name="row7", capacity_kw=600.0, num_racks_a100=23, num_racks_h100=0),
        # RowSpec(name="row8", capacity_kw=600.0, num_racks_a100=23, num_racks_h100=0),
    ]
    dc = DataCenterSimulator(rows)

    # Equal mix over a subset for demo; replace with your full 16-config percentages
    job_mix = {
        "llama-3-70b_a100_tp8": 0.1,
        "llama-3-8b_a100_tp1": 0.1,
        "llama-3-70b_h100_tp8": 0.1,
        "llama-3-8b_h100_tp1": 0.1,
        "deepseek-r1-distill-70b_a100_tp8": 0.1,
        "deepseek-r1-distill-8b_a100_tp1": 0.1,
        "deepseek-r1-distill-70b_h100_tp8": 0.1,
        "deepseek-r1-distill-8b_h100_tp1": 0.1,
        "llama-3-70b_a100_tp4": 0.1,
        "llama-3-8b_a100_tp2": 0.1,
    }
    dc.assign_jobs(job_mix)

    # Choose sample selectors: first node in row1
    sample_row = "row1"
    sample_node = next((n for n in dc.nodes if n.row_name == sample_row), None)
    sample_rack = sample_node.rack_id if sample_node else None
    sample_node_id = sample_node.node_id if sample_node else None

    res = dc.simulate(
        duration=2 * 7200.0,
        diurnal_min_qps_per_server=0.25,
        diurnal_max_qps_per_server=5.0,
        start_of_day_sec=0,
        output_dt=0.25,
        base_seed=123,
        return_profiles=False,
        fast_workload=True,
        num_workers=16,
        diurnal_sharpness_gamma=10.0,
        return_per_rack=True,
        sample_row_name=sample_row,
        sample_rack_id=sample_rack,
        sample_node_id=sample_node_id,
        sample_server_replica_index=0,
    )

    out_dir = "/Users/grantwilkins/powertrace-sim/training_results/dc_pilot"
    save_row_csvs(out_dir, res["timestamps"], res["per_row"], res["datacenter_watts"])
    if "per_rack_by_row" in res:
        save_row_rack_csvs(
            out_dir, res["timestamps"], res["per_rack_by_row"], sample_row
        )
    # Optional sampled outputs
    if "sampled_server_watts" in res:
        save_series(
            os.path.join(out_dir, f"{sample_row}_sampled_server_power.csv"),
            res["timestamps"],
            res["sampled_server_watts"],
        )
    if "sampled_node_watts" in res:
        save_series(
            os.path.join(out_dir, f"{sample_row}_sampled_node_power.csv"),
            res["timestamps"],
            res["sampled_node_watts"],
        )
    if "sampled_rack_watts" in res:
        # Use safe rack name for file
        safe_rack = sample_rack.replace(":", "-") if sample_rack else "rack"
        save_series(
            os.path.join(out_dir, f"{sample_row}__{safe_rack}_sampled_power.csv"),
            res["timestamps"],
            res["sampled_rack_watts"],
        )
    if "sampled_row_watts" in res:
        save_series(
            os.path.join(out_dir, f"{sample_row}_sampled_row_power.csv"),
            res["timestamps"],
            res["sampled_row_watts"],
        )
    # Save tokens
    if (
        "per_row_tokens_in" in res
        and "per_row_tokens_out" in res
        and "datacenter_tokens_in" in res
        and "datacenter_tokens_out" in res
    ):
        save_token_csvs(
            out_dir,
            res["timestamps"],
            res["per_row_tokens_in"],
            res["per_row_tokens_out"],
            res["datacenter_tokens_in"],
            res["datacenter_tokens_out"],
        )
    print(f"Saved per-row and datacenter CSVs to {out_dir}")


if __name__ == "__main__":
    main()
