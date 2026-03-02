import csv
import os

import numpy as np

from model.simulators.arrival_simulator import create_llama_config
from model.simulators.server_power_simulator import ServerPowerSimulator


def save_csv(path: str, timestamps: np.ndarray, watts: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_s", "watts"])
        for t, p in zip(timestamps, watts):
            w.writerow([float(t), float(p)])


def maybe_plot(path: str, timestamps: np.ndarray, watts: np.ndarray):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.figure(figsize=(12, 4))
        plt.plot(timestamps, watts, linewidth=1.2)
        plt.xlabel("Time (s)")
        plt.ylabel("Power (W)")
        plt.title("Server Power Trace: llama-3-8b H100 TP1")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {path}")
    except Exception as e:
        print(f"Plotting skipped (matplotlib unavailable or error): {e}")


def main():
    # Config: llama-3-8b on H100 TP1
    config = create_llama_config(size_b=8, tp=2, hardware="H100")
    sim = ServerPowerSimulator(config)

    res = sim.simulate_server_power(
        category="language",
        model_type="m-small",
        duration=600.0,
        rate_requests_per_sec=0.25,
        output_dt=0.25,
    )

    ts = res["timestamps"]
    watts = res["watts"]

    out_dir = "/Users/grantwilkins/powertrace-sim/training_results"
    base = "server_power_validation_llama-3-8b_h100_tp1"
    csv_path = os.path.join(out_dir, f"{base}.csv")
    png_path = os.path.join(out_dir, f"{base}.png")
    print(res["profile_seconds"])
    save_csv(csv_path, ts, watts)
    print(f"Saved CSV: {csv_path}")
    maybe_plot(png_path, ts, watts)


if __name__ == "__main__":
    main()
