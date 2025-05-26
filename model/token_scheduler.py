import itertools
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import transformers
from iohmm import histogram_requests


@dataclass
class ModelConfig:
    """Configuration for a specific model deployment."""

    model_size: int
    tensor_parallelism: int
    hardware: str

    def __hash__(self):
        return hash((self.model_size, self.tensor_parallelism, self.hardware))

    def __str__(self):
        return f"{self.model_size}B-TP{self.tensor_parallelism}-{self.hardware}"


@dataclass
class TokenRequest:
    """Represents a single token processing request."""

    arrival_time: float
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TokenDistribution:
    """Manages token distributions and sampling."""

    def __init__(self, input_tokens: np.ndarray, output_tokens: np.ndarray):
        """
        Initialize with token distributions.

        Args:
            input_tokens: Array of input token counts
            output_tokens: Array of output token counts
        """
        self.input_tokens = input_tokens[input_tokens != 0]
        self.output_tokens = output_tokens[output_tokens != 0]

    def sample(self, count: int = 1) -> List[Tuple[int, int]]:
        """
        Sample pairs of input and output token counts.

        Args:
            count: Number of samples to generate

        Returns:
            List of (input_tokens, output_tokens) tuples
        """
        in_samples = np.random.choice(self.input_tokens, size=count)
        out_samples = np.random.choice(self.output_tokens, size=count)
        return [(int(i), int(o)) for i, o in zip(in_samples, out_samples)]

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        input_field: str = "inputs",
        output_field: str = "outputs",
    ):
        """
        Create a distribution from a HuggingFace dataset.

        Args:
            dataset_name: Name of the HuggingFace dataset
            input_field: Field containing input text
            output_field: Field containing output text

        Returns:
            A new TokenDistribution instance
        """
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct"
        )
        data = datasets.load_dataset(dataset_name)

        input_tokens = np.array(
            [len(tokenizer.encode(text)) for text in data[input_field]]
        )

        output_tokens = np.array(
            [len(tokenizer.encode(text)) for text in data[output_field]]
        )

        return cls(input_tokens, output_tokens)


class ThroughputStats:
    """Manages throughput statistics for different model configurations."""

    def __init__(self):
        self.prefill_throughput: Dict[ModelConfig, float] = {}
        self.decode_throughput: Dict[ModelConfig, float] = {}
        self.ttft_constant: Dict[ModelConfig, float] = {}

    def add_config(
        self, config: ModelConfig, prefill_thr: float, decode_thr: float, ttft: float
    ):
        """Add throughput stats for a configuration."""
        self.prefill_throughput[config] = prefill_thr
        self.decode_throughput[config] = decode_thr
        self.ttft_constant[config] = ttft

    def get_prefill_throughput(self, config: ModelConfig) -> float:
        """Get prefill throughput for a configuration."""
        return self.prefill_throughput[config]

    def get_decode_throughput(self, config: ModelConfig) -> float:
        """Get decode throughput for a configuration."""
        return self.decode_throughput[config]


class RequestSimulator:
    """Simulates token processing requests."""

    def __init__(
        self, token_distribution: TokenDistribution, throughput_stats: ThroughputStats
    ):
        self.token_distribution = token_distribution
        self.throughput_stats = throughput_stats

    def generate_poisson_requests(
        self, time_horizon: float, arrival_rate: float
    ) -> List[TokenRequest]:
        """
        Generate requests following a Poisson process.

        Args:
            time_horizon: Maximum time to generate requests for
            arrival_rate: Average number of requests per time unit

        Returns:
            List of TokenRequest objects
        """
        requests = []
        t = 0.0

        while t < time_horizon:
            # Sample next arrival time from exponential distribution
            t += np.random.exponential(1 / arrival_rate)
            if t >= time_horizon:
                break

            # Sample input and output token counts
            in_tok, out_tok = self.token_distribution.sample(1)[0]
            requests.append(TokenRequest(t, in_tok, out_tok))

        return requests

    def estimate_completion_times(
        self, requests: List[TokenRequest], config: ModelConfig
    ) -> pd.DataFrame:
        """
        Estimate completion times for a list of requests.

        Args:
            requests: List of token requests
            config: Model configuration to use

        Returns:
            DataFrame with request timing information
        """
        prefill_thr = self.throughput_stats.get_prefill_throughput(config)
        decode_thr = self.throughput_stats.get_decode_throughput(config)

        data = []
        for req in requests:
            prefill_end = req.arrival_time + (req.input_tokens / prefill_thr)
            decode_end = prefill_end + (req.output_tokens / decode_thr)

            data.append(
                {
                    "arrival_time": req.arrival_time,
                    "input_tokens": req.input_tokens,
                    "output_tokens": req.output_tokens,
                    "prefill_end": prefill_end,
                    "decode_end": decode_end,
                    "total_time": decode_end - req.arrival_time,
                }
            )

        return pd.DataFrame(data)


class TokenScheduler:
    """Schedules and analyzes token processing."""

    def __init__(self, simulator: RequestSimulator):
        self.simulator = simulator

    def analyze_schedule(
        self,
        requests: List[TokenRequest],
        config: ModelConfig,
        time_horizon: float,
        time_step: float,
    ) -> Dict[str, np.ndarray]:
        """
        Analyze a request schedule to get token processing statistics.

        Args:
            requests: List of token requests
            config: Model configuration to use
            time_horizon: Maximum time to analyze
            time_step: Time step for binning

        Returns:
            Dictionary with arrays of statistics over time
        """
        # Calculate completion times
        timing_df = self.simulator.estimate_completion_times(requests, config)

        # Initialize time bins
        timestamps = np.arange(0, time_horizon, time_step)
        prefill_tokens = np.zeros_like(timestamps, dtype=float)
        decode_tokens = np.zeros_like(timestamps, dtype=float)
        active_requests = np.zeros_like(timestamps, dtype=int)

        # Fill bins
        for t_idx, t in enumerate(timestamps):
            # Requests that arrive in this time bin
            new_reqs = timing_df[
                (timing_df["arrival_time"] >= t)
                & (timing_df["arrival_time"] < t + time_step)
            ]
            prefill_tokens[t_idx] = new_reqs["input_tokens"].sum()

            # Requests in decode phase during this time bin
            decoding = timing_df[
                (timing_df["prefill_end"] <= t) & (t < timing_df["decode_end"])
            ]
            decode_tokens[t_idx] = decoding["output_tokens"].sum()

            # Active requests during this time bin
            active = timing_df[
                (timing_df["arrival_time"] <= t) & (t < timing_df["decode_end"])
            ]
            active_requests[t_idx] = len(active)

        # Prepare result arrays
        return {
            "timestamps": timestamps,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "active_requests": active_requests,
            "request_times": np.array([r.arrival_time for r in requests]),
            "input_tokens": np.array([r.input_tokens for r in requests]),
            "output_tokens": np.array([r.output_tokens for r in requests]),
        }

    def prepare_inference_features(
        self, schedule_data: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Format schedule data for inference model input.

        Args:
            schedule_data: Schedule data from analyze_schedule

        Returns:
            Feature matrix ready for inference (z-scored)
        """
        trace_dict = {
            "timestamps": schedule_data["timestamps"],
            "request_ts": schedule_data["request_times"],
            "input_tokens": schedule_data["input_tokens"],
            "output_tokens": schedule_data["output_tokens"],
            "active_requests": schedule_data["active_requests"],
            "prefill_tokens": schedule_data["prefill_tokens"],
            "decode_tokens": schedule_data["decode_tokens"],
        }

        cnt, tok_in, tok_out = histogram_requests(
            bin_ts=trace_dict["timestamps"],
            req_ts=trace_dict["request_ts"],
            in_tok=trace_dict["input_tokens"],
            out_tok=trace_dict["output_tokens"],
        )

        x = np.stack(
            [
                cnt,
                tok_in,
                tok_out,
                trace_dict["active_requests"],
                trace_dict["prefill_tokens"],
                trace_dict["decode_tokens"],
            ],
            axis=-1,
        )

        # Z-score normalize each column
        mu = x.mean(0, keepdims=True)
        sd = x.std(0, keepdims=True) + 1e-6
        return (x - mu) / sd


class TokenSimulator:
    """Main class for token-based LLM simulation."""

    @classmethod
    def from_npz(cls, npz_file: str, hf_dataset: Optional[str] = None):
        """
        Create a simulator from an NPZ file.

        Args:
            npz_file: Path to NPZ file with benchmark data
            hf_dataset: Optional HuggingFace dataset name

        Returns:
            A configured TokenSimulator instance
        """
        # Load data
        data = np.load(npz_file, allow_pickle=True)

        # Set up token distribution
        if hf_dataset:
            token_dist = TokenDistribution.from_huggingface(hf_dataset)
        else:
            token_dist = TokenDistribution(data["input_tokens"], data["output_tokens"])

        # Set up throughput stats
        throughput_stats = ThroughputStats()
        unique_models = np.unique(data["model_sizes"])
        unique_tp = np.unique(data["tensor_parallelism"])
        unique_hw = np.unique(data["hardware"])

        for ms, tp, hw in itertools.product(unique_models, unique_tp, unique_hw):
            mask = (
                (data["model_sizes"] == ms)
                & (data["tensor_parallelism"] == tp)
                & (data["hardware"] == hw)
            )

            if not np.any(mask):
                continue

            config = ModelConfig(ms, tp, hw)
            prefill_thr = np.mean(data["prefill_throughputs"][mask].flatten())
            decode_thr = np.mean(data["decode_throughputs"][mask].flatten())
            ttft = np.mean(data["prefill_times"][mask].flatten())

            throughput_stats.add_config(config, prefill_thr, decode_thr, ttft)

        request_sim = RequestSimulator(token_dist, throughput_stats)
        scheduler = TokenScheduler(request_sim)

        simulator = cls()
        simulator.token_distribution = token_dist
        simulator.throughput_stats = throughput_stats
        simulator.request_simulator = request_sim
        simulator.scheduler = scheduler

        return simulator

    def run_simulation(
        self,
        config: ModelConfig,
        time_horizon: float = 600,
        arrival_rate: float = 1.0,
        time_step: float = 0.25,
    ) -> Dict[str, np.ndarray]:
        """
        Run a complete simulation.

        Args:
            config: Model configuration to use
            time_horizon: Simulation time horizon
            arrival_rate: Request arrival rate (per time unit)
            time_step: Time step for analysis

        Returns:
            Dictionary with simulation results
        """
        requests = self.request_simulator.generate_poisson_requests(
            time_horizon, arrival_rate
        )

        return self.scheduler.analyze_schedule(
            requests, config, time_horizon, time_step
        )

    def prepare_for_inference(
        self, simulation_results: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Convert simulation results to the format expected by inference models.

        Args:
            simulation_results: Results from run_simulation

        Returns:
            Feature matrix ready for inference (z-scored)
        """
        return self.scheduler.prepare_inference_features(simulation_results)


if __name__ == "__main__":
    dataset = TokenSimulator.from_npz(
        npz_file="./training_data/vllm-benchmark-llama-3-8b-power-a100.npz"
    )
    results = dataset.run_simulation(
        config=ModelConfig(model_size=8, tensor_parallelism=1, hardware="A100"),
        time_horizon=600,
        arrival_rate=1.0,
        time_step=0.25,
    )
    plt.plot(results["timestamps"], results["active_requests"])
    plt.xlabel("Time")
    plt.ylabel("Active Requests")
    plt.title("Active Requests over Time")
    plt.show()
