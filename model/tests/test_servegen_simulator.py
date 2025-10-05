"""
Unit tests for ServeGen-based arrival simulator.
Tests request generation, system simulation, and feature matrix creation.
"""
import unittest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulators.arrival_simulator import (
    ServingConfig, ServeGenRequest, ServingSystemSimulator,
    ServeGenWorkloadGenerator, ServeGenPowerSimulator,
    create_llama_config, create_deepseek_config, quick_simulate
)


class TestServingConfig(unittest.TestCase):
    """Test ServingConfig dataclass functionality."""

    def test_serving_config_creation(self):
        """Test basic ServingConfig creation and properties."""
        config = ServingConfig(
            model_name="llama-3-8b",
            model_size_b=8,
            hardware="A100",
            tensor_parallelism=2,
            ttft_seconds=0.5,
            tpot_seconds=0.02,
            batch_size=32
        )

        self.assertEqual(config.model_name, "llama-3-8b")
        self.assertEqual(config.model_size_b, 8)
        self.assertEqual(config.hardware, "A100")
        self.assertEqual(config.tensor_parallelism, 2)
        self.assertEqual(config.ttft_seconds, 0.5)
        self.assertEqual(config.tpot_seconds, 0.02)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(str(config), "llama-3-8b-TP2-A100")

    def test_convenience_configs(self):
        """Test convenience configuration functions."""
        llama_config = create_llama_config(8, tp=2, hardware="H100")
        self.assertEqual(llama_config.model_name, "llama-3-8b")
        self.assertEqual(llama_config.tensor_parallelism, 2)
        self.assertEqual(llama_config.hardware, "H100")
        self.assertAlmostEqual(llama_config.ttft_seconds, 0.5)
        self.assertAlmostEqual(llama_config.tpot_seconds, 0.02)

        deepseek_config = create_deepseek_config(70, tp=4)
        self.assertEqual(deepseek_config.model_name, "deepseek-r1-70b")
        self.assertEqual(deepseek_config.tensor_parallelism, 4)
        self.assertGreater(deepseek_config.ttft_seconds, llama_config.ttft_seconds)  # Reasoning models slower


class TestServeGenRequest(unittest.TestCase):
    """Test ServeGenRequest dataclass functionality."""

    def test_request_creation(self):
        """Test ServeGenRequest creation and properties."""
        req = ServeGenRequest(
            request_id=1,
            arrival_time=10.5,
            input_tokens=100,
            output_tokens=50
        )

        self.assertEqual(req.request_id, 1)
        self.assertEqual(req.arrival_time, 10.5)
        self.assertEqual(req.input_tokens, 100)
        self.assertEqual(req.output_tokens, 50)
        self.assertEqual(req.total_tokens, 150)

        # Check that timing fields start as None
        self.assertIsNone(req.prefill_start)
        self.assertIsNone(req.prefill_end)
        self.assertIsNone(req.decode_start)
        self.assertIsNone(req.decode_end)


class TestServingSystemSimulator(unittest.TestCase):
    """Test the serving system simulation logic."""

    def setUp(self):
        """Set up test configuration."""
        self.config = ServingConfig(
            model_name="test-model",
            model_size_b=8,
            hardware="A100",
            ttft_seconds=0.5,      # 500ms TTFT
            tpot_seconds=0.02,     # 20ms per output token
            batch_size=2           # Small batch for testing
        )
        self.simulator = ServingSystemSimulator(self.config)

    def test_single_request_processing(self):
        """Test processing a single request."""
        request = ServeGenRequest(
            request_id=1,
            arrival_time=10.0,
            input_tokens=100,
            output_tokens=50
        )

        processed = self.simulator.simulate_request_processing([request])
        self.assertEqual(len(processed), 1)

        req = processed[0]
        # Check timing calculations
        self.assertEqual(req.prefill_start, 10.0)  # Starts immediately
        self.assertEqual(req.prefill_end, 10.5)    # TTFT = 0.5s
        self.assertEqual(req.decode_start, 10.5)   # Decode starts after prefill
        self.assertEqual(req.decode_end, 11.5)     # 50 tokens * 0.02s = 1.0s decode time

    def test_multiple_requests_no_queueing(self):
        """Test multiple requests within batch size limit."""
        requests = [
            ServeGenRequest(request_id=1, arrival_time=10.0, input_tokens=100, output_tokens=50),
            ServeGenRequest(request_id=2, arrival_time=10.1, input_tokens=80, output_tokens=30),
        ]

        processed = self.simulator.simulate_request_processing(requests)
        self.assertEqual(len(processed), 2)

        # Both should start immediately (within batch size)
        self.assertEqual(processed[0].prefill_start, 10.0)
        self.assertEqual(processed[1].prefill_start, 10.1)

    def test_queueing_behavior(self):
        """Test queueing when batch size is exceeded."""
        requests = [
            ServeGenRequest(request_id=1, arrival_time=10.0, input_tokens=100, output_tokens=50),
            ServeGenRequest(request_id=2, arrival_time=10.1, input_tokens=80, output_tokens=30),
            ServeGenRequest(request_id=3, arrival_time=10.2, input_tokens=120, output_tokens=40),  # Should queue
        ]

        processed = self.simulator.simulate_request_processing(requests)

        # First two requests start immediately
        self.assertEqual(processed[0].prefill_start, 10.0)
        self.assertEqual(processed[1].prefill_start, 10.1)

        # Third request should wait for earliest completion
        earliest_completion = min(processed[0].decode_end, processed[1].decode_end)
        self.assertGreaterEqual(processed[2].prefill_start, earliest_completion)

    def test_timeline_creation(self):
        """Test system timeline generation."""
        requests = [
            ServeGenRequest(request_id=1, arrival_time=10.0, input_tokens=100, output_tokens=50,
                          prefill_start=10.0, prefill_end=10.5, decode_start=10.5, decode_end=11.5),
            ServeGenRequest(request_id=2, arrival_time=10.2, input_tokens=80, output_tokens=30,
                          prefill_start=10.2, prefill_end=10.7, decode_start=10.7, decode_end=11.3),
        ]

        timeline = self.simulator.create_system_timeline(requests, time_step=0.1)

        # Check basic structure
        self.assertIn('timestamps', timeline)
        self.assertIn('active_requests', timeline)
        self.assertIn('prefill_tokens', timeline)
        self.assertIn('decode_tokens', timeline)
        self.assertIn('request_count', timeline)
        self.assertIn('input_tokens', timeline)
        self.assertIn('output_tokens', timeline)

        # Check that we have data points
        self.assertGreater(len(timeline['timestamps']), 0)
        self.assertEqual(len(timeline['active_requests']), len(timeline['timestamps']))

    def test_feature_matrix_creation(self):
        """Test 6D feature matrix creation."""
        # Create a simple timeline
        timeline = {
            'timestamps': np.array([10.0, 10.25, 10.5, 10.75, 11.0]),
            'request_count': np.array([1, 1, 0, 0, 0]),
            'input_tokens': np.array([100, 80, 0, 0, 0]),
            'output_tokens': np.array([50, 30, 0, 0, 0]),
            'active_requests': np.array([1, 2, 2, 1, 0]),
            'prefill_tokens': np.array([100, 80, 0, 0, 0]),
            'decode_tokens': np.array([0, 0, 50, 30, 0]),
            'request_timestamps': np.array([10.0, 10.1]),
            'individual_input_tokens': np.array([100, 80]),
            'individual_output_tokens': np.array([50, 30])
        }

        features = self.simulator.create_feature_matrix(timeline)

        # Should be (T, 6) shape
        self.assertEqual(features.shape[1], 6)
        self.assertEqual(features.shape[0], len(timeline['timestamps']))

        # Features should be z-score normalized
        self.assertAlmostEqual(np.mean(features), 0.0, places=1)


class TestServeGenWorkloadGenerator(unittest.TestCase):
    """Test ServeGen workload generation (mocked)."""

    def setUp(self):
        """Set up mocked ServeGen components."""
        self.mock_servegen_request = MagicMock()
        self.mock_servegen_request.request_id = 1
        self.mock_servegen_request.timestamp = 10.0
        self.mock_servegen_request.data = {'input_tokens': 100, 'output_tokens': 50}

    @patch('simulators.arrival_simulator.SERVEGEN_AVAILABLE', True)
    @patch('simulators.arrival_simulator.ClientPool')
    @patch('simulators.arrival_simulator.generate_workload')
    @patch('simulators.arrival_simulator.get_constant_rate_fn')
    @patch('simulators.arrival_simulator.Category')
    def test_request_generation(self, mock_category, mock_rate_fn, mock_generate, mock_pool):
        """Test request generation with mocked ServeGen."""
        # Setup mocks
        mock_generate.return_value = [self.mock_servegen_request]
        mock_pool_instance = MagicMock()
        mock_pool.return_value = mock_pool_instance
        mock_view = MagicMock()
        mock_pool_instance.span.return_value = mock_view

        generator = ServeGenWorkloadGenerator()
        requests = generator.generate_requests(
            category="language",
            duration=3600,
            rate_requests_per_sec=1.0,
            seed=42
        )

        # Should convert ServeGen requests to our format
        self.assertEqual(len(requests), 1)
        self.assertIsInstance(requests[0], ServeGenRequest)
        self.assertEqual(requests[0].request_id, 1)
        self.assertEqual(requests[0].arrival_time, 10.0)
        self.assertEqual(requests[0].input_tokens, 100)
        self.assertEqual(requests[0].output_tokens, 50)

    def test_time_window_parsing(self):
        """Test time window string parsing."""
        generator = ServeGenWorkloadGenerator()

        # Valid formats
        start, end = generator._parse_time_window("14:30-16:45")
        self.assertEqual(start, 14*3600 + 30*60)  # 14:30 in seconds
        self.assertEqual(end, 16*3600 + 45*60)    # 16:45 in seconds

        # Invalid format should raise error
        with self.assertRaises(ValueError):
            generator._parse_time_window("invalid-format")


class TestEndToEndSimulation(unittest.TestCase):
    """Test complete end-to-end simulation pipeline."""

    @patch('simulators.arrival_simulator.SERVEGEN_AVAILABLE', True)
    @patch('simulators.arrival_simulator.ClientPool')
    @patch('simulators.arrival_simulator.generate_workload')
    @patch('simulators.arrival_simulator.get_constant_rate_fn')
    @patch('simulators.arrival_simulator.Category')
    def test_complete_simulation_pipeline(self, mock_category, mock_rate_fn, mock_generate, mock_pool):
        """Test complete simulation from ServeGen to feature matrix."""
        # Create mock requests with variety of sizes
        mock_requests = []
        for i in range(5):
            mock_req = MagicMock()
            mock_req.request_id = i
            mock_req.timestamp = 10.0 + i * 0.5
            mock_req.data = {
                'input_tokens': 100 + i * 20,
                'output_tokens': 50 + i * 10
            }
            mock_requests.append(mock_req)

        mock_generate.return_value = mock_requests
        mock_pool_instance = MagicMock()
        mock_pool.return_value = mock_pool_instance
        mock_view = MagicMock()
        mock_pool_instance.span.return_value = mock_view

        # Run simulation
        config = create_llama_config(8)
        simulator = ServeGenPowerSimulator(config)

        result = simulator.generate_power_simulation_data(
            duration=60,
            rate_requests_per_sec=1.0
        )

        # Check result structure
        self.assertIn('feature_matrix', result)
        self.assertIn('timeline', result)
        self.assertIn('requests', result)
        self.assertIn('serving_config', result)

        # Check feature matrix properties
        features = result['feature_matrix']
        self.assertEqual(features.shape[1], 6)  # 6D features
        self.assertGreater(features.shape[0], 0)  # Have time steps

        # Check requests were processed
        requests = result['requests']
        self.assertEqual(len(requests), 5)
        for req in requests:
            self.assertIsNotNone(req.prefill_start)
            self.assertIsNotNone(req.decode_end)

    def test_performance_scaling(self):
        """Test that performance parameters scale correctly."""
        small_config = create_llama_config(8)
        large_config = create_llama_config(70)

        # Larger models should have higher TTFT and TPOT
        self.assertGreater(large_config.ttft_seconds, small_config.ttft_seconds)
        self.assertGreater(large_config.tpot_seconds, small_config.tpot_seconds)

    def test_model_type_differences(self):
        """Test differences between model types."""
        llama_config = create_llama_config(8)
        deepseek_config = create_deepseek_config(8)

        # Reasoning models should be slower
        self.assertGreater(deepseek_config.ttft_seconds, llama_config.ttft_seconds)
        self.assertGreater(deepseek_config.tpot_seconds, llama_config.tpot_seconds)

        # Different batch sizes for different workloads
        self.assertLess(deepseek_config.batch_size, llama_config.batch_size)


class TestVisualizationAndValidation(unittest.TestCase):
    """Test visualization and validation methods."""

    def test_timing_consistency(self):
        """Test that timing calculations are consistent."""
        config = ServingConfig(
            model_name="test",
            model_size_b=8,
            hardware="A100",
            ttft_seconds=0.5,
            tpot_seconds=0.02,
            batch_size=32
        )

        request = ServeGenRequest(
            request_id=1,
            arrival_time=10.0,
            input_tokens=100,
            output_tokens=50
        )

        simulator = ServingSystemSimulator(config)
        processed = simulator.simulate_request_processing([request])
        req = processed[0]

        # Verify timing relationships
        self.assertGreaterEqual(req.prefill_start, req.arrival_time)
        self.assertGreater(req.prefill_end, req.prefill_start)
        self.assertGreaterEqual(req.decode_start, req.prefill_end)
        self.assertGreater(req.decode_end, req.decode_start)

        # Verify duration calculations
        prefill_duration = req.prefill_end - req.prefill_start
        decode_duration = req.decode_end - req.decode_start

        self.assertAlmostEqual(prefill_duration, config.ttft_seconds)
        self.assertAlmostEqual(decode_duration, req.output_tokens * config.tpot_seconds)


def create_visualization_demo():
    """Create comprehensive visualization of the simulator behavior."""
    print("Creating visualization demo...")

    # Create a realistic scenario
    config = create_llama_config(8, tp=2, hardware="A100")

    # Create some sample requests with different characteristics
    requests = [
        ServeGenRequest(1, 0.0, 50, 20),    # Small request
        ServeGenRequest(2, 0.5, 200, 100),  # Large request
        ServeGenRequest(3, 1.0, 100, 50),   # Medium request
        ServeGenRequest(4, 1.2, 150, 75),   # Another medium
        ServeGenRequest(5, 2.0, 80, 30),    # Small request
        ServeGenRequest(6, 2.5, 300, 150),  # Very large request
    ]

    simulator = ServingSystemSimulator(config)
    processed_requests = simulator.simulate_request_processing(requests)
    timeline = simulator.create_system_timeline(processed_requests, time_step=0.1)

    # Create comprehensive plots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('ServeGen Simulator: System Behavior Analysis', fontsize=16)

    # Plot 1: Request Timeline
    ax1 = axes[0, 0]
    for req in processed_requests:
        # Prefill phase
        ax1.barh(req.request_id, req.prefill_end - req.prefill_start,
                left=req.prefill_start, color='orange', alpha=0.7, label='Prefill' if req.request_id == 1 else '')
        # Decode phase
        ax1.barh(req.request_id, req.decode_end - req.decode_start,
                left=req.decode_start, color='blue', alpha=0.7, label='Decode' if req.request_id == 1 else '')
        # Queue time (if any)
        if req.prefill_start > req.arrival_time:
            ax1.barh(req.request_id, req.prefill_start - req.arrival_time,
                    left=req.arrival_time, color='red', alpha=0.7, label='Queue' if req.request_id == 1 else '')

    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Request ID')
    ax1.set_title('Request Processing Timeline')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Active Requests Over Time
    ax2 = axes[0, 1]
    ax2.plot(timeline['timestamps'], timeline['active_requests'], 'g-', linewidth=2)
    ax2.axhline(y=config.batch_size, color='r', linestyle='--', alpha=0.7, label=f'Batch Size ({config.batch_size})')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Active Requests')
    ax2.set_title('System Concurrency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Token Processing Signals
    ax3 = axes[1, 0]
    ax3.plot(timeline['timestamps'], timeline['prefill_tokens'], 'orange', linewidth=2, label='Prefill Tokens')
    ax3.plot(timeline['timestamps'], timeline['decode_tokens'], 'blue', linewidth=2, label='Decode Tokens')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Tokens Being Processed')
    ax3.set_title('Prefill vs Decode Token Processing')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Request Arrivals
    ax4 = axes[1, 1]
    ax4.bar(timeline['timestamps'], timeline['request_count'], width=0.08, alpha=0.7, color='purple')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('New Requests')
    ax4.set_title('Request Arrival Pattern')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Token Arrival vs Processing
    ax5 = axes[2, 0]
    ax5.plot(timeline['timestamps'], timeline['input_tokens'], 'orange', alpha=0.7, label='Input Tokens (Arriving)')
    ax5.plot(timeline['timestamps'], timeline['output_tokens'], 'blue', alpha=0.7, label='Output Tokens (Arriving)')
    ax5.plot(timeline['timestamps'], timeline['prefill_tokens'], 'red', linewidth=2, label='Prefill Tokens (Processing)')
    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('Token Count')
    ax5.set_title('Arrival vs Processing Signals')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Feature Matrix Visualization
    ax6 = axes[2, 1]
    feature_matrix = simulator.create_feature_matrix(timeline)
    feature_names = ['Request Count', 'Input Tokens', 'Output Tokens', 'Active Requests', 'Prefill Tokens', 'Decode Tokens']

    # Show normalized features as heatmap
    im = ax6.imshow(feature_matrix.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax6.set_xlabel('Time Steps')
    ax6.set_ylabel('Feature Dimension')
    ax6.set_title('6D Feature Matrix (Z-scored)')
    ax6.set_yticks(range(6))
    ax6.set_yticklabels(feature_names, fontsize=8)
    plt.colorbar(im, ax=ax6, label='Normalized Value')

    plt.tight_layout()
    plt.savefig('/Users/grantwilkins/powertrace-sim/model/servegen_simulator_demo.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print some statistics
    print("\n=== Simulation Statistics ===")
    print(f"Configuration: {config}")
    print(f"Number of requests: {len(processed_requests)}")
    print(f"Simulation duration: {timeline['timestamps'][-1] - timeline['timestamps'][0]:.2f}s")
    print(f"Peak concurrent requests: {max(timeline['active_requests'])}")
    print(f"Feature matrix shape: {feature_matrix.shape}")

    # Timing analysis
    total_prefill_time = sum(req.prefill_end - req.prefill_start for req in processed_requests)
    total_decode_time = sum(req.decode_end - req.decode_start for req in processed_requests)
    queue_times = [max(0, req.prefill_start - req.arrival_time) for req in processed_requests]

    print(f"Average prefill time: {total_prefill_time/len(processed_requests):.3f}s")
    print(f"Average decode time: {total_decode_time/len(processed_requests):.3f}s")
    print(f"Average queue time: {np.mean(queue_times):.3f}s")
    print(f"Max queue time: {max(queue_times):.3f}s")

    return processed_requests, timeline, feature_matrix


if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)

    # Create visualization demo
    print("\n" + "="*50)
    create_visualization_demo()