"""
Example script demonstrating node-level power simulation.
Shows how to use NodePowerSimulator for data center simulation.
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from model.simulators.arrival_simulator import (
    create_deepseek_config,
    create_llama_config,
)
from model.simulators.node_power_simulator import (
    NodeConfig,
    create_node_simulator_from_dataset,
    quick_node_simulation,
)


def example_single_node():
    """Example: Simulate a single node."""
    print("=" * 60)
    print("Example 1: Single Node Simulation")
    print("=" * 60)

    # Quick simulation using convenience function (now uses unified config!)
    power_trace = quick_node_simulation(
        node_id="node-001",
        model="llama-3-8b",
        hardware="a100",
        tp=1,
        workload="language",
        duration=600,  # 10 minutes
        rate=2.0,  # 2 requests/sec
    )

    print(f"\nNode ID: {power_trace.node_id}")
    print(f"Duration: {power_trace.timestamps[-1]:.1f} seconds")
    print(f"Mean Power: {power_trace.metadata['mean_power_watts']:.1f} W")
    print(f"Max Power: {power_trace.metadata['max_power_watts']:.1f} W")
    print(f"Total Energy: {power_trace.metadata['total_energy_joules']:.1f} J")

    return power_trace


def example_multi_node():
    """Example: Simulate multiple nodes with different configurations."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Node Simulation")
    print("=" * 60)

    # Define multiple node configurations (no dataset paths needed!)
    node_configs = [
        {
            "node_id": "llama-8b-node-1",
            "model": "llama-3-8b",
            "hardware": "a100",
            "tp": 1,
            "rate": 1.5,
        },
        {
            "node_id": "llama-8b-node-2",
            "model": "llama-3-8b",
            "hardware": "a100",
            "tp": 2,
            "rate": 2.0,
        },
        {
            "node_id": "llama-70b-node-1",
            "model": "llama-3-70b",
            "hardware": "a100",
            "tp": 4,
            "rate": 0.8,
        },
    ]

    power_traces = []
    for config in node_configs:
        print(f"\nSimulating {config['node_id']}...")
        try:
            trace = quick_node_simulation(
                node_id=config["node_id"],
                model=config["model"],
                hardware=config["hardware"],
                tp=config["tp"],
                workload="language",
                duration=300,  # 5 minutes
                rate=config["rate"],
            )
            power_traces.append(trace)
            print(
                f"  Mean Power: {trace.metadata['mean_power_watts']:.1f} W, "
                f"Total Energy: {trace.metadata['total_energy_joules']:.1f} J"
            )
        except Exception as e:
            print(f"  Error: {e}")

    return power_traces


def example_custom_configuration():
    """Example: Custom configuration with fine-grained control."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)

    # Create custom node config
    node_config = NodeConfig(
        node_id="custom-node-001",
        model_name="llama-3-8b",
        hardware="a100",
        tensor_parallelism=2,
        weights_base_path="../gru_classifier_weights/",
    )

    # Create simulator from dataset
    simulator = create_node_simulator_from_dataset(
        node_config=node_config,
        dataset_path="../data/train_llama-3-8b_a100_tp1-8.npz",
    )

    # Create serving config with custom parameters
    serving_config = create_llama_config(size_b=8, tp=2, hardware="A100")

    # Run simulation with custom parameters
    power_trace = simulator.simulate_from_serving_config(
        serving_config=serving_config,
        workload_category="language",
        model_type="m-large",
        duration=600,
        rate_requests_per_sec=3.0,
        seed=42,
        dt=0.25,
        smoothing_window=7,
    )

    print(f"\nNode ID: {power_trace.node_id}")
    print(f"Configuration: {node_config}")
    print(f"Mean Power: {power_trace.metadata['mean_power_watts']:.1f} W")
    print(f"Total Energy: {power_trace.metadata['total_energy_joules']:.1f} J")

    return power_trace


def plot_power_traces(traces, output_path="power_traces.png"):
    """Plot power traces for visualization."""
    plt.figure(figsize=(12, 6))

    for trace in traces:
        # Downsample for plotting if too long
        if len(trace.timestamps) > 5000:
            step = len(trace.timestamps) // 5000
            timestamps = trace.timestamps[::step]
            power = trace.power_watts[::step]
        else:
            timestamps = trace.timestamps
            power = trace.power_watts

        label = f"{trace.node_id} ({trace.metadata['mean_power_watts']:.0f}W avg)"
        plt.plot(timestamps, power, label=label, alpha=0.8)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Power (Watts)")
    plt.title("Node Power Consumption Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")


def save_traces(traces, output_dir="output"):
    """Save power traces to files."""
    os.makedirs(output_dir, exist_ok=True)

    for trace in traces:
        # Save as NPZ
        npz_path = os.path.join(output_dir, f"{trace.node_id}.npz")
        trace.save_npz(npz_path)

        # Save as JSON
        json_path = os.path.join(output_dir, f"{trace.node_id}.json")
        with open(json_path, "w") as f:
            json.dump(trace.to_dict(), f, indent=2)

        print(f"Saved {trace.node_id} to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate node-level power consumption"
    )
    parser.add_argument(
        "--example",
        type=str,
        choices=["single", "multi", "custom", "all"],
        default="all",
        help="Which example to run",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save traces to output directory",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for traces",
    )
    args = parser.parse_args()

    traces = []

    if args.example in ["single", "all"]:
        trace = example_single_node()
        traces.append(trace)

    if args.example in ["multi", "all"]:
        multi_traces = example_multi_node()
        traces.extend(multi_traces)

    if args.example in ["custom", "all"]:
        trace = example_custom_configuration()
        traces.append(trace)

    # Save traces if requested
    if args.save and traces:
        save_traces(traces, output_dir=args.output_dir)

    # Plot if requested
    if args.plot and traces:
        plot_power_traces(
            traces,
            output_path=os.path.join(args.output_dir, "power_traces.png"),
        )

    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
