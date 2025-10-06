"""
Demonstration of ServeGen-based power simulation.
Shows how to generate realistic workloads and convert them to power traces.
"""

import os
import sys
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulators.arrival_simulator import (
    ServeGenPowerSimulator,
    ServeGenRequest,
    ServingSystemSimulator,
    create_deepseek_config,
    create_llama_config,
)


def create_sample_workload():
    """Create a sample workload with diverse request patterns."""
    print("Creating sample workload...")
    requests = []

    # Bursty period: Many small requests
    for i in range(10):
        requests.append(
            ServeGenRequest(
                request_id=i,
                arrival_time=i * 0.2,  # Every 200ms
                input_tokens=np.random.randint(50, 150),
                output_tokens=np.random.randint(20, 80),
            )
        )

    # Quiet period
    quiet_start = 2.5
    for i in range(3):
        requests.append(
            ServeGenRequest(
                request_id=10 + i,
                arrival_time=quiet_start + i * 1.0,  # Every second
                input_tokens=np.random.randint(200, 400),
                output_tokens=np.random.randint(100, 200),
            )
        )

    # Heavy load period: Large requests
    heavy_start = 6.0
    for i in range(8):
        requests.append(
            ServeGenRequest(
                request_id=13 + i,
                arrival_time=heavy_start + i * 0.3,  # Every 300ms
                input_tokens=np.random.randint(300, 800),
                output_tokens=np.random.randint(150, 400),
            )
        )

    # Cool down period
    cooldown_start = 8.5
    for i in range(4):
        requests.append(
            ServeGenRequest(
                request_id=21 + i,
                arrival_time=cooldown_start + i * 0.8,
                input_tokens=np.random.randint(100, 200),
                output_tokens=np.random.randint(50, 100),
            )
        )

    return sorted(requests, key=lambda x: x.arrival_time)


def analyze_workload_patterns(requests):
    """Analyze and visualize workload characteristics."""
    print(f"Analyzing {len(requests)} requests...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Workload Analysis", fontsize=14)

    # Request size distribution
    ax1 = axes[0, 0]
    input_sizes = [r.input_tokens for r in requests]
    output_sizes = [r.output_tokens for r in requests]

    ax1.scatter(input_sizes, output_sizes, alpha=0.7, s=50)
    ax1.set_xlabel("Input Tokens")
    ax1.set_ylabel("Output Tokens")
    ax1.set_title("Request Size Distribution")
    ax1.grid(True, alpha=0.3)

    # Arrival pattern
    ax2 = axes[0, 1]
    arrival_times = [r.arrival_time for r in requests]
    ax2.hist(arrival_times, bins=20, alpha=0.7, color="orange")
    ax2.set_xlabel("Arrival Time (seconds)")
    ax2.set_ylabel("Request Count")
    ax2.set_title("Request Arrival Pattern")
    ax2.grid(True, alpha=0.3)

    # Token size histograms
    ax3 = axes[1, 0]
    ax3.hist(input_sizes, bins=15, alpha=0.7, label="Input Tokens", color="blue")
    ax3.hist(output_sizes, bins=15, alpha=0.7, label="Output Tokens", color="red")
    ax3.set_xlabel("Token Count")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Token Size Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Request rate over time
    ax4 = axes[1, 1]
    time_bins = np.arange(0, max(arrival_times) + 1, 0.5)
    rates, _ = np.histogram(arrival_times, bins=time_bins)
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
    ax4.bar(
        bin_centers, rates / 0.5, width=0.4, alpha=0.7, color="green"
    )  # Rate per second
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Request Rate (req/s)")
    ax4.set_title("Request Rate Over Time")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "/Users/grantwilkins/powertrace-sim/model/workload_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Print statistics
    print(f"Average input tokens: {np.mean(input_sizes):.1f}")
    print(f"Average output tokens: {np.mean(output_sizes):.1f}")
    print(f"Total duration: {max(arrival_times):.1f}s")
    print(f"Average arrival rate: {len(requests) / max(arrival_times):.2f} req/s")


def compare_serving_configurations():
    """Compare different serving configurations."""
    print("Comparing serving configurations...")

    configs = {
        "Llama-3-8B": create_llama_config(8, tp=1),
        "Llama-3-70B": create_llama_config(70, tp=4),
        "DeepSeek-R1-8B": create_deepseek_config(8, tp=1),
        "DeepSeek-R1-70B": create_deepseek_config(70, tp=4),
    }

    # Create identical workload for comparison
    requests = create_sample_workload()

    results = {}
    for name, config in configs.items():
        simulator = ServingSystemSimulator(config)
        processed = simulator.simulate_request_processing(requests.copy())
        timeline = simulator.create_system_timeline(processed, time_step=0.1)
        results[name] = {"requests": processed, "timeline": timeline, "config": config}

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Configuration Comparison", fontsize=16)

    colors = ["blue", "red", "green", "orange"]

    # Plot 1: Active requests over time
    ax1 = axes[0, 0]
    for i, (name, result) in enumerate(results.items()):
        timeline = result["timeline"]
        ax1.plot(
            timeline["timestamps"],
            timeline["active_requests"],
            color=colors[i],
            linewidth=2,
            label=name,
            alpha=0.8,
        )
        # Show batch size limit
        ax1.axhline(
            y=result["config"].batch_size, color=colors[i], linestyle="--", alpha=0.5
        )

    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Active Requests")
    ax1.set_title("System Concurrency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Processing signals
    ax2 = axes[0, 1]
    for i, (name, result) in enumerate(results.items()):
        timeline = result["timeline"]
        total_processing = timeline["prefill_tokens"] + timeline["decode_tokens"]
        ax2.plot(
            timeline["timestamps"],
            total_processing,
            color=colors[i],
            linewidth=2,
            label=name,
            alpha=0.8,
        )

    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Total Tokens Being Processed")
    ax2.set_title("Processing Load")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: End-to-end latency comparison
    ax3 = axes[1, 0]
    latencies = {}
    for name, result in results.items():
        e2e_latencies = [
            req.decode_end - req.arrival_time for req in result["requests"]
        ]
        latencies[name] = e2e_latencies

    ax3.boxplot(latencies.values(), labels=latencies.keys())
    ax3.set_ylabel("End-to-End Latency (seconds)")
    ax3.set_title("Latency Distribution by Configuration")
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45)

    # Plot 4: Queue times
    ax4 = axes[1, 1]
    queue_stats = {}
    for name, result in results.items():
        queue_times = [
            max(0, req.prefill_start - req.arrival_time) for req in result["requests"]
        ]
        queue_stats[name] = {
            "mean": np.mean(queue_times),
            "max": max(queue_times),
            "p95": np.percentile(queue_times, 95),
        }

    names = list(queue_stats.keys())
    means = [queue_stats[name]["mean"] for name in names]
    maxes = [queue_stats[name]["max"] for name in names]
    p95s = [queue_stats[name]["p95"] for name in names]

    x = np.arange(len(names))
    width = 0.25

    ax4.bar(x - width, means, width, label="Mean", alpha=0.8, color="blue")
    ax4.bar(x, p95s, width, label="P95", alpha=0.8, color="orange")
    ax4.bar(x + width, maxes, width, label="Max", alpha=0.8, color="red")

    ax4.set_xlabel("Configuration")
    ax4.set_ylabel("Queue Time (seconds)")
    ax4.set_title("Queueing Behavior")
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "/Users/grantwilkins/powertrace-sim/model/configuration_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Print comparison statistics
    print("\n=== Configuration Comparison ===")
    for name, result in results.items():
        config = result["config"]
        requests = result["requests"]
        timeline = result["timeline"]

        e2e_latencies = [req.decode_end - req.arrival_time for req in requests]
        queue_times = [max(0, req.prefill_start - req.arrival_time) for req in requests]

        print(f"\n{name}:")
        print(f"  TTFT: {config.ttft_seconds:.3f}s, TPOT: {config.tpot_seconds:.3f}s")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Mean E2E Latency: {np.mean(e2e_latencies):.2f}s")
        print(f"  P95 E2E Latency: {np.percentile(e2e_latencies, 95):.2f}s")
        print(f"  Mean Queue Time: {np.mean(queue_times):.3f}s")
        print(f"  Peak Concurrent Requests: {max(timeline['active_requests'])}")


def demonstrate_feature_extraction():
    """Show how system state is converted to 6D feature vectors."""
    print("Demonstrating feature extraction...")

    # Create workload and process it
    requests = create_sample_workload()[:15]  # Smaller set for clarity
    config = create_llama_config(8)
    simulator = ServingSystemSimulator(config)

    processed_requests = simulator.simulate_request_processing(requests)
    timeline = simulator.create_system_timeline(processed_requests, time_step=0.2)
    feature_matrix = simulator.create_feature_matrix(timeline)

    # Create detailed visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Feature Extraction Process", fontsize=16)

    # Plot 1: Raw timeline signals
    ax1 = axes[0, 0]
    ax1.plot(
        timeline["timestamps"],
        timeline["request_count"],
        "o-",
        label="New Requests",
        linewidth=2,
    )
    ax1.plot(
        timeline["timestamps"],
        timeline["active_requests"],
        "s-",
        label="Active Requests",
        linewidth=2,
    )
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Count")
    ax1.set_title("System State: Request Counts")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Token signals
    ax2 = axes[0, 1]
    ax2.plot(
        timeline["timestamps"],
        timeline["input_tokens"],
        "o-",
        label="Input Tokens (Arriving)",
        linewidth=2,
    )
    ax2.plot(
        timeline["timestamps"],
        timeline["output_tokens"],
        "s-",
        label="Output Tokens (Arriving)",
        linewidth=2,
    )
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Token Count")
    ax2.set_title("System State: Arrival Tokens")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Processing signals
    ax3 = axes[1, 0]
    ax3.plot(
        timeline["timestamps"],
        timeline["prefill_tokens"],
        "o-",
        label="Prefill Tokens",
        color="orange",
        linewidth=2,
    )
    ax3.plot(
        timeline["timestamps"],
        timeline["decode_tokens"],
        "s-",
        label="Decode Tokens",
        color="blue",
        linewidth=2,
    )
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Token Count")
    ax3.set_title("System State: Processing Tokens")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Feature matrix heatmap
    ax4 = axes[1, 1]
    feature_names = [
        "Request Count",
        "Input Tokens",
        "Output Tokens",
        "Active Requests",
        "Prefill Tokens",
        "Decode Tokens",
    ]

    im = ax4.imshow(
        feature_matrix.T, aspect="auto", cmap="RdBu_r", interpolation="nearest"
    )
    ax4.set_xlabel("Time Steps")
    ax4.set_ylabel("Feature Dimension")
    ax4.set_title("6D Feature Matrix (Z-scored)")
    ax4.set_yticks(range(6))
    ax4.set_yticklabels(feature_names, fontsize=10)
    plt.colorbar(im, ax=ax4, label="Normalized Value")

    # Plot 5: Individual feature evolution
    ax5 = axes[2, 0]
    for i, name in enumerate(feature_names):
        ax5.plot(
            timeline["timestamps"],
            feature_matrix[:, i],
            "o-",
            label=name,
            linewidth=2,
            alpha=0.8,
        )
    ax5.set_xlabel("Time (seconds)")
    ax5.set_ylabel("Normalized Feature Value")
    ax5.set_title("Feature Evolution Over Time")
    ax5.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax5.grid(True, alpha=0.3)

    # Plot 6: Request timeline with phases
    ax6 = axes[2, 1]
    for req in processed_requests[:10]:  # Show first 10 requests
        # Prefill phase
        ax6.barh(
            req.request_id,
            req.prefill_end - req.prefill_start,
            left=req.prefill_start,
            color="orange",
            alpha=0.7,
            label=(
                "Prefill" if req.request_id == processed_requests[0].request_id else ""
            ),
        )
        # Decode phase
        ax6.barh(
            req.request_id,
            req.decode_end - req.decode_start,
            left=req.decode_start,
            color="blue",
            alpha=0.7,
            label=(
                "Decode" if req.request_id == processed_requests[0].request_id else ""
            ),
        )
        # Queue time
        if req.prefill_start > req.arrival_time:
            ax6.barh(
                req.request_id,
                req.prefill_start - req.arrival_time,
                left=req.arrival_time,
                color="red",
                alpha=0.7,
                label=(
                    "Queue"
                    if req.request_id == processed_requests[0].request_id
                    else ""
                ),
            )

    ax6.set_xlabel("Time (seconds)")
    ax6.set_ylabel("Request ID")
    ax6.set_title("Request Processing Phases")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "/Users/grantwilkins/powertrace-sim/model/feature_extraction_demo.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Feature statistics:")
    print(f"  Mean (should be ~0): {np.mean(feature_matrix):.4f}")
    print(f"  Std (should be ~1): {np.std(feature_matrix):.4f}")
    print(f"  Feature names: {feature_names}")


def main():
    """Run the complete demonstration."""
    print("ServeGen Power Simulation Demonstration")
    print("=" * 50)

    # Seed for reproducible results
    np.random.seed(42)

    try:
        # Step 1: Analyze workload patterns
        print("\n1. Creating and analyzing sample workload...")
        requests = create_sample_workload()
        analyze_workload_patterns(requests)

        # Step 2: Compare configurations
        print("\n2. Comparing serving configurations...")
        compare_serving_configurations()

        # Step 3: Demonstrate feature extraction
        print("\n3. Demonstrating feature extraction...")
        demonstrate_feature_extraction()

        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("Check the generated PNG files for visualizations:")
        print("- workload_analysis.png")
        print("- configuration_comparison.png")
        print("- feature_extraction_demo.png")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
