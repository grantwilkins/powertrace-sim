"""
ServeGen-based arrival simulator for generating realistic LLM serving workloads.
Converts ServeGen request patterns into system-level processing timelines.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add ServeGen to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../ServeGen"))

try:
    from servegen import Category, ClientPool
    from servegen.construct import generate_workload
    from servegen.utils import get_constant_rate_fn

    SERVEGEN_AVAILABLE = True
except ImportError as e:
    print(f"ServeGen not available: {e}")
    SERVEGEN_AVAILABLE = False


_PERF_DB_CACHE: Optional[Dict[str, Dict]] = None


def _get_perf_db_path() -> str:
    """Return absolute path to performance_database.json."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    perf_path = os.path.join(this_dir, "../config/performance_database.json")
    return os.path.normpath(perf_path)


def _load_performance_db() -> Dict[str, Dict]:
    """Load and cache the performance database from JSON."""
    global _PERF_DB_CACHE
    if _PERF_DB_CACHE is not None:
        return _PERF_DB_CACHE

    path = _get_perf_db_path()
    if not os.path.exists(path):
        _PERF_DB_CACHE = {}
        return _PERF_DB_CACHE

    with open(path, "r") as f:
        _PERF_DB_CACHE = json.load(f)
    return _PERF_DB_CACHE


def _normalize_hardware_for_key(hardware: str) -> str:
    # Keys use lowercase (e.g., a100, h100)
    return (hardware or "").strip().lower()


def _infer_model_for_key(model_name: str, override: Optional[str]) -> Optional[str]:
    # Use explicit override if provided
    if override:
        return override.strip()

    # Heuristics to map common names to DB families
    name = (model_name or "").strip().lower()
    if "llama" in name:
        # Database uses "llama-3.1"
        return "llama-3.1"
    if "deepseek" in name:
        # Database uses "deepseek-r1-distill"
        if "distill" in name or "r1" in name:
            return "deepseek-r1-distill"
        return "deepseek-r1-distill"
    return None


def _build_db_key(config: ServingConfig) -> Optional[str]:
    model_key = _infer_model_for_key(config.model_name, config.perf_db_model_name)
    if not model_key:
        return None
    size_part = f"{config.model_size_b}b"
    hw_part = _normalize_hardware_for_key(config.hardware)
    tp_part = f"tp{config.tensor_parallelism}"
    return f"{model_key}_{size_part}_{hw_part}_{tp_part}"


class PerformanceSampler:
    """Samples TTFT and TPOT per request from performance database with fallbacks."""

    def __init__(self, config: ServingConfig):
        self.config = config
        self.db = _load_performance_db()
        self.db_key = _build_db_key(config)
        self.entry = self.db.get(self.db_key) if self.db_key else None
        self.use_fallback = self.entry is None

        # Parse distributions if available
        self.ttft_dist = None
        self.tpot_dist = None
        self.ttft_clip = None
        self.tpot_clip = None

        if not self.use_fallback:
            # Try new format first, fallback to old format
            ttft = self.entry.get("ttft_model") or self.entry.get(
                "ttft_distribution", {}
            )
            tpot = self.entry.get("tpot_distribution", {})

            # For clipping to observed range if present
            ttft_stats = ttft.get("summary_stats", {})
            tpot_stats = tpot.get("summary_stats", {})
            if ttft_stats:
                self.ttft_clip = (
                    ttft_stats.get("min_observed"),
                    ttft_stats.get("max_observed"),
                )
            if tpot_stats:
                self.tpot_clip = (
                    tpot_stats.get("min_observed"),
                    tpot_stats.get("max_observed"),
                )

            # TTFT distribution
            ttft_type = (ttft.get("type") or "").lower()
            if ttft_type == "decile_quantile":
                # Decile-conditioned quantile sampler
                bin_edges_z = ttft.get("bin_edges_z", [])
                quantile_levels = ttft.get("quantile_levels", [])
                bin_quantiles = ttft.get("bin_quantiles", [])
                nin_min = int(ttft.get("nin_min", 1))
                nin_max = int(ttft.get("nin_max", 4096))
                self.ttft_dist = (
                    "decile_quantile",
                    bin_edges_z,
                    quantile_levels,
                    bin_quantiles,
                    nin_min,
                    nin_max,
                )
            elif ttft_type == "gamma_glm":
                # Gamma GLM: μ = exp(β0 + β1 * log(n_in + 1)), Var = φ * μ^2
                beta0 = float(ttft.get("beta0", 0.0))
                beta1 = float(ttft.get("beta1", 0.0))
                phi = float(ttft.get("phi", 0.01))
                nin_min = int(ttft.get("nin_min", 1))
                nin_max = int(ttft.get("nin_max", 4096))
                self.ttft_dist = ("gamma_glm", beta0, beta1, phi, nin_min, nin_max)
            elif ttft_type == "heteroskedastic_log_linear":
                # Heteroskedastic model: base variance + input-dependent variance
                intercept = float(ttft.get("intercept", 0.0))
                slope = float(ttft.get("slope", 1.0))
                sigma_base = float(ttft.get("sigma_base", 0.1))  # Unconditional noise
                sigma_intercept = float(ttft.get("sigma_intercept", -2.3))
                sigma_slope = float(ttft.get("sigma_slope", 0.0))
                nin_min = int(ttft.get("nin_min", 1))
                nin_max = int(ttft.get("nin_max", 4096))
                self.ttft_dist = (
                    "heteroskedastic_log_linear",
                    intercept,
                    slope,
                    sigma_base,
                    sigma_intercept,
                    sigma_slope,
                    nin_min,
                    nin_max,
                )
            elif ttft_type == "log_linear":
                # Legacy log-linear model: log(TTFT) = a0 + a1 * log(input_tokens) + N(0, sigma^2)
                intercept = float(ttft.get("intercept", 0.0))
                slope = float(ttft.get("slope", 1.0))
                sigma_log = float(ttft.get("sigma_log", 0.1))
                nin_min = int(ttft.get("nin_min", 1))
                nin_max = int(ttft.get("nin_max", 4096))
                self.ttft_dist = (
                    "log_linear",
                    intercept,
                    slope,
                    sigma_log,
                    nin_min,
                    nin_max,
                )
            elif ttft_type == "gamma":
                shape = float(ttft.get("shape", 1.0))
                scale = float(ttft.get("scale", 0.001))
                self.ttft_dist = ("gamma", shape, scale)
            elif ttft_type in ("gaussian", "normal", "norm"):
                mean = float(ttft.get("mean", self.config.ttft_seconds))
                std = float(ttft.get("std", 1e-3))
                self.ttft_dist = ("gaussian", mean, std)

            # TPOT distribution
            tpot_type = (tpot.get("type") or "").lower()
            if tpot_type in ("gaussian", "normal", "norm"):
                mean = float(tpot.get("mean", self.config.tpot_seconds))
                std = float(tpot.get("std", 1e-4))
                self.tpot_dist = ("gaussian", mean, std)
            elif tpot_type == "gamma":
                shape = float(tpot.get("shape", 1.0))
                scale = float(tpot.get("scale", 1e-3))
                self.tpot_dist = ("gamma", shape, scale)

        # Seed numpy for reproducibility if desired (kept global control outside)

    def _clip(
        self, value: float, clip: Optional[Tuple[Optional[float], Optional[float]]]
    ) -> float:
        if not clip:
            return value
        lo, hi = clip
        if lo is not None and value < lo:
            value = lo
        if hi is not None and value > hi:
            value = hi
        return value

    def sample_ttft(self, input_tokens: Optional[int] = None) -> float:
        """
        Sample TTFT (time to first token) in seconds.

        Args:
            input_tokens: Number of input tokens (required for log_linear model)
        """
        if self.use_fallback or self.ttft_dist is None:
            # Fallback: scale by input length if provided
            if input_tokens is not None:
                return float(self.config.ttft_seconds * (input_tokens / 512))
            return float(self.config.ttft_seconds)

        kind = self.ttft_dist[0]
        if kind == "decile_quantile":
            # Decile-conditioned quantile sampler
            _, bin_edges_z, quantile_levels, bin_quantiles, nin_min, nin_max = (
                self.ttft_dist
            )
            if input_tokens is None:
                input_tokens = (nin_min + nin_max) // 2

            # Transform to z-space: z = log(1 + n_in)
            z = np.log(1 + input_tokens)

            # Find which bin(s) z falls into
            bin_edges_z = np.array(bin_edges_z)
            bin_idx = np.searchsorted(bin_edges_z, z, side="right") - 1
            bin_idx = np.clip(bin_idx, 0, len(bin_quantiles) - 1)

            # Get quantiles for this bin
            quantiles = np.array(bin_quantiles[bin_idx])
            quantile_levels = np.array(quantile_levels)

            # Draw uniform random variable
            u = np.random.uniform(0, 1)

            # Interpolate inverse CDF at probability u
            val = float(np.interp(u, quantile_levels, quantiles))
            val = max(val, 0.001)  # Ensure positive (min 1ms)
        elif kind == "gamma_glm":
            # Gamma GLM: μ = exp(β0 + β1 * log(n_in + 1))
            # Var = φ * μ^2, so shape k = 1/φ, scale θ = μ/k
            _, beta0, beta1, phi, nin_min, nin_max = self.ttft_dist
            if input_tokens is None:
                input_tokens = (nin_min + nin_max) // 2
            nin_clamped = np.clip(input_tokens, nin_min, nin_max)

            # Compute mean: μ = exp(β0 + β1 * log(n_in + 1))
            x = np.log(nin_clamped + 1)
            mu = np.exp(beta0 + beta1 * x)

            # Gamma parameters: k = 1/φ, θ = μ/k = μ*φ
            k = 1.0 / phi  # shape
            theta = mu * phi  # scale

            # Sample from Gamma(k, θ)
            val = float(np.random.gamma(k, theta))
            val = max(val, 0.001)  # Ensure positive (min 1ms)
        elif kind == "heteroskedastic_log_linear":
            # Heteroskedastic: base variance + input-dependent variance
            (
                _,
                intercept,
                slope,
                sigma_base,
                sigma_intercept,
                sigma_slope,
                nin_min,
                nin_max,
            ) = self.ttft_dist
            if input_tokens is None:
                # Fallback to mean if input_tokens not provided
                input_tokens = (nin_min + nin_max) // 2
            # Clamp to observed range
            nin_clamped = np.clip(input_tokens, nin_min, nin_max)
            log_n = np.log(nin_clamped)

            # Mean prediction
            mu = intercept + slope * log_n

            # Total variance = base + input-dependent
            # sigma_total = sqrt(sigma_base^2 + sigma_input^2)
            log_sigma_input = sigma_intercept + sigma_slope * log_n
            sigma_input = np.exp(log_sigma_input)
            sigma_total = np.sqrt(sigma_base**2 + sigma_input**2)

            # Sample with combined noise
            log_ttft = mu + np.random.normal(0, sigma_total)
            val = float(np.exp(log_ttft))
            val = max(val, 0.001)  # Ensure positive (min 1ms)
        elif kind == "log_linear":
            # Legacy log-linear: constant variance
            _, intercept, slope, sigma_log, nin_min, nin_max = self.ttft_dist
            if input_tokens is None:
                # Fallback to mean if input_tokens not provided
                input_tokens = (nin_min + nin_max) // 2
            # Clamp to observed range
            nin_clamped = np.clip(input_tokens, nin_min, nin_max)
            log_ttft = (
                intercept + slope * np.log(nin_clamped) + np.random.normal(0, sigma_log)
            )
            val = float(np.exp(log_ttft))
            val = max(val, 0.001)  # Ensure positive (min 1ms)
        elif kind == "gamma":
            _, shape, scale = self.ttft_dist
            val = float(np.random.gamma(shape, scale))
        elif kind == "gaussian":
            _, mean, std = self.ttft_dist
            val = float(np.random.normal(mean, std))
            if val <= 0:
                val = mean  # ensure positive
        else:
            val = float(self.config.ttft_seconds)
        return self._clip(val, self.ttft_clip)

    def sample_tpot(self) -> float:
        if self.use_fallback or self.tpot_dist is None:
            return float(self.config.tpot_seconds)
        kind = self.tpot_dist[0]
        if kind == "gaussian":
            _, mean, std = self.tpot_dist
            val = float(np.random.normal(mean, std))
            if val <= 0:
                val = mean
        elif kind == "gamma":
            _, shape, scale = self.tpot_dist
            val = float(np.random.gamma(shape, scale))
        else:
            val = float(self.config.tpot_seconds)
        return self._clip(val, self.tpot_clip)


@dataclass
class ServingConfig:
    """Configuration for LLM serving system parameters."""

    # Model and hardware
    model_name: str
    model_size_b: int  # Model size in billions of parameters
    hardware: str  # "A100", "H100", etc.
    tensor_parallelism: int = 1

    # Performance parameters (can be distributions later)
    ttft_seconds: float = 0.5  # Time to first token (includes prefill)
    tpot_seconds: float = 0.02  # Time per output token (1/decode_tps)

    # Optional: explicit performance DB model name (e.g., "llama-3.1", "deepseek-r1-distill")
    perf_db_model_name: Optional[str] = None

    # System constraints
    batch_size: int = 32  # Maximum concurrent requests
    queue_limit: int = 100  # Maximum queued requests

    def __str__(self):
        return f"{self.model_name}-TP{self.tensor_parallelism}-{self.hardware}"


@dataclass
class ServeGenRequest:
    """Request from ServeGen with timing information."""

    request_id: int
    arrival_time: float  # When request arrived
    input_tokens: int  # Context/prompt tokens
    output_tokens: int  # Generated tokens

    # Computed timing (set by simulator)
    prefill_start: Optional[float] = None
    prefill_end: Optional[float] = None
    decode_start: Optional[float] = None
    decode_end: Optional[float] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ServeGenWorkloadGenerator:
    """Generates realistic workloads using ServeGen data."""

    def __init__(self):
        if not SERVEGEN_AVAILABLE:
            raise RuntimeError("ServeGen not available. Please install ServeGen.")

    def generate_requests(
        self,
        category: str = "language",  # "language", "reason", "multimodal"
        model_type: str = "m-large",  # "m-small", "m-mid", "m-large"
        duration: float = 3600,  # Simulation duration in seconds
        time_window: Optional[str] = None,  # "18:00-19:00" or None for random
        rate_requests_per_sec: float = 1.0,  # Target request rate
        seed: int = 0,
    ) -> List[ServeGenRequest]:
        """
        Generate a realistic request stream using ServeGen.

        Args:
            category: Workload category (language/reason/multimodal)
            model_type: Model size category (m-small/m-mid/m-large)
            duration: How long to simulate (seconds)
            time_window: Time window for realistic patterns (e.g., "18:00-19:00")
            rate_requests_per_sec: Target arrival rate
            seed: Random seed for reproducibility

        Returns:
            List of ServeGenRequest objects with realistic token distributions
        """
        np.random.seed(seed)
        category_enum = getattr(Category, category.upper())
        pool = ClientPool(category_enum, model_type)
        if time_window:
            start_sec, end_sec = self._parse_time_window(time_window)
            view = pool.span(start_sec, end_sec)
        else:
            view = pool.span(50400, 64800)  # 2pm-6pm

        rate_fn = get_constant_rate_fn(view, rate_requests_per_sec)
        servegen_requests = generate_workload(
            view, rate_fn, duration=duration, seed=seed
        )
        requests = []
        for req in servegen_requests:
            requests.append(
                ServeGenRequest(
                    request_id=req.request_id,
                    arrival_time=req.timestamp,
                    input_tokens=req.data["input_tokens"],
                    output_tokens=req.data["output_tokens"],
                )
            )

        return sorted(requests, key=lambda x: x.arrival_time)

    def _parse_time_window(self, time_window: str) -> Tuple[int, int]:
        try:
            start_str, end_str = time_window.split("-")
            start_hour, start_min = map(int, start_str.split(":"))
            end_hour, end_min = map(int, end_str.split(":"))

            start_seconds = start_hour * 3600 + start_min * 60
            end_seconds = end_hour * 3600 + end_min * 60

            return start_seconds, end_seconds
        except ValueError as e:
            raise ValueError(
                f"Invalid time window format: {time_window}. Use 'HH:MM-HH:MM'"
            ) from e


class ServingSystemSimulator:
    def __init__(self, config: ServingConfig):
        self.config = config

    def simulate_request_processing(
        self, requests: List[ServeGenRequest]
    ) -> List[ServeGenRequest]:
        requests = sorted(requests, key=lambda x: x.arrival_time)
        processed_requests = []
        active_requests = []  # Currently processing
        sampler = PerformanceSampler(self.config)

        for req in requests:
            if len(active_requests) < self.config.batch_size:
                start_time = req.arrival_time
            else:
                earliest_completion = min(
                    r.decode_end for r in active_requests if r.decode_end
                )
                start_time = max(req.arrival_time, earliest_completion)
            sampled_ttft = sampler.sample_ttft(req.input_tokens)
            sampled_tpot = sampler.sample_tpot()

            prefill_start = start_time
            prefill_end = prefill_start + sampled_ttft

            decode_start = prefill_end
            decode_duration = req.output_tokens * sampled_tpot
            decode_end = decode_start + decode_duration
            req.prefill_start = prefill_start
            req.prefill_end = prefill_end
            req.decode_start = decode_start
            req.decode_end = decode_end

            processed_requests.append(req)
            active_requests = [r for r in active_requests if r.decode_end > start_time]
            active_requests.append(req)

        return processed_requests

    def create_system_timeline(
        self, requests: List[ServeGenRequest], time_step: float = 0.25
    ) -> Dict[str, np.ndarray]:
        if not requests:
            return {}
        start_time = min(req.arrival_time for req in requests)
        end_time = max(req.decode_end for req in requests)
        time_bins = np.arange(start_time, end_time + time_step, time_step)
        timeline = {
            "timestamps": time_bins,
            "request_count": np.zeros(len(time_bins)),  # New requests this bin
            "input_tokens": np.zeros(len(time_bins)),  # Input tokens arriving
            "output_tokens": np.zeros(len(time_bins)),  # Output tokens arriving
            "active_requests": np.zeros(len(time_bins)),  # Concurrent requests
            "prefill_tokens": np.zeros(len(time_bins)),  # Tokens starting prefill
            "decode_tokens": np.zeros(len(time_bins)),  # Tokens being decoded
            "request_timestamps": [],  # Individual request times
            "individual_input_tokens": [],  # Individual request inputs
            "individual_output_tokens": [],  # Individual request outputs
        }

        for i, current_time in enumerate(time_bins):
            if i == 0:
                interval_start = current_time - time_step / 2
            else:
                interval_start = (current_time + time_bins[i - 1]) / 2

            if i == len(time_bins) - 1:
                interval_end = current_time + time_step / 2
            else:
                interval_end = (current_time + time_bins[i + 1]) / 2

            arriving = [
                r for r in requests if interval_start <= r.arrival_time < interval_end
            ]
            timeline["request_count"][i] = len(arriving)
            timeline["input_tokens"][i] = sum(r.input_tokens for r in arriving)
            timeline["output_tokens"][i] = sum(r.output_tokens for r in arriving)
            prefill_starting = [
                r for r in requests if interval_start <= r.prefill_start < interval_end
            ]
            timeline["prefill_tokens"][i] = sum(
                r.input_tokens for r in prefill_starting
            )
            decoding = [
                r for r in requests if r.decode_start <= current_time <= r.decode_end
            ]
            timeline["decode_tokens"][i] = sum(r.output_tokens for r in decoding)
            active = [
                r for r in requests if r.prefill_start <= current_time <= r.decode_end
            ]
            timeline["active_requests"][i] = len(active)
        timeline["request_timestamps"] = np.array([r.arrival_time for r in requests])
        timeline["individual_input_tokens"] = np.array(
            [r.input_tokens for r in requests]
        )
        timeline["individual_output_tokens"] = np.array(
            [r.output_tokens for r in requests]
        )

        return timeline

    def create_feature_matrix(
        self, timeline: Dict[str, np.ndarray], arrival_rate: float = None
    ) -> np.ndarray:
        """
        Create feature matrix using the same preprocessing as training.

        Args:
            timeline: System timeline dictionary
            arrival_rate: Optional arrival rate to include as feature

        Returns:
            Feature matrix (T, Dx) with Dx=10 (or 7 without diff features)
        """
        from model.core.utils import make_schedule_matrix

        trace_dict = {
            "timestamps": timeline["timestamps"],
            "request_ts": timeline["request_timestamps"],
            "input_tokens": timeline["individual_input_tokens"],
            "output_tokens": timeline["individual_output_tokens"],
            "active_requests": timeline["active_requests"],
            "prefill_tokens": timeline["prefill_tokens"],
            "decode_tokens": timeline["decode_tokens"],
        }
        return make_schedule_matrix(
            trace_dict, arrival_rate=arrival_rate, add_diff_features=True
        )


class ServeGenPowerSimulator:
    """End-to-end simulator using ServeGen for power trace generation."""

    def __init__(self, serving_config: ServingConfig):
        self.serving_config = serving_config
        self.workload_generator = ServeGenWorkloadGenerator()
        self.system_simulator = ServingSystemSimulator(serving_config)

    def generate_power_simulation_data(
        self,
        category: str = "language",
        model_type: str = "m-large",
        duration: float = 3600,
        rate_requests_per_sec: float = 1.0,
        time_window: Optional[str] = None,
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Generate complete simulation data for power prediction.

        Returns dictionary with keys matching the training NPZ format:
        - timestamps, request_timestamps
        - input_tokens, output_tokens
        - prefill_tokens, decode_tokens, active_requests
        - And other metadata needed for GRU inference
        """

        # Step 1: Generate realistic requests
        requests = self.workload_generator.generate_requests(
            category=category,
            model_type=model_type,
            duration=duration,
            time_window=time_window,
            rate_requests_per_sec=rate_requests_per_sec,
            seed=seed,
        )

        if not requests:
            raise ValueError("No requests generated. Check ServeGen configuration.")

        # Step 2: Simulate serving system processing
        processed_requests = self.system_simulator.simulate_request_processing(requests)

        # Step 3: Create system timeline
        timeline = self.system_simulator.create_system_timeline(processed_requests)

        # Step 4: Create feature matrix for GRU (with arrival rate)
        feature_matrix = self.system_simulator.create_feature_matrix(
            timeline, arrival_rate=rate_requests_per_sec
        )

        return {
            "feature_matrix": feature_matrix,
            "timeline": timeline,
            "requests": processed_requests,
            "serving_config": self.serving_config,
            "arrival_rate": rate_requests_per_sec,
        }


# Convenience functions for easy usage
def create_llama_config(
    size_b: int, tp: int = 1, hardware: str = "A100"
) -> ServingConfig:
    """Create a standard Llama model configuration."""
    # TTFT varies by model size (larger models take longer for prefill)
    ttft = 0.5 if size_b <= 8 else 0.8 if size_b <= 70 else 1.2
    # TPOT also varies by model size (larger models slower per token)
    tpot = 0.02 if size_b <= 8 else 0.033 if size_b <= 70 else 0.05

    return ServingConfig(
        model_name=f"llama-3-{size_b}b",
        model_size_b=size_b,
        hardware=hardware,
        tensor_parallelism=tp,
        ttft_seconds=ttft,
        tpot_seconds=tpot,
        batch_size=32,
        perf_db_model_name="llama-3.1",
    )


def create_deepseek_config(
    size_b: int, tp: int = 1, hardware: str = "A100"
) -> ServingConfig:
    """Create a standard DeepSeek model configuration."""
    # Reasoning models typically have higher TTFT and TPOT
    ttft = 0.8 if size_b <= 8 else 1.2 if size_b <= 70 else 1.8
    tpot = (
        0.022 if size_b <= 8 else 0.04 if size_b <= 70 else 0.067
    )  # Slightly slower per token

    return ServingConfig(
        model_name=f"deepseek-r1-{size_b}b",
        model_size_b=size_b,
        hardware=hardware,
        tensor_parallelism=tp,
        ttft_seconds=ttft,
        tpot_seconds=tpot,
        batch_size=24,  # Smaller batches for reasoning workloads
        perf_db_model_name="deepseek-r1-distill",
    )


def quick_simulate(
    model: str = "llama-3-8b",
    hardware: str = "A100",
    tp: int = 1,
    workload: str = "language",
    duration: float = 3600,
    rate: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Quick simulation with sensible defaults.

    Args:
        model: Model name (e.g., "llama-3-8b", "deepseek-r1-70b")
        hardware: Hardware type ("A100", "H100")
        tp: Tensor parallelism
        workload: Workload type ("language", "reason", "multimodal")
        duration: Simulation time (seconds)
        rate: Request rate (requests/sec)

    Returns:
        Simulation data ready for GRU inference
    """
    # Parse model configuration
    if "llama" in model.lower():
        size = int(model.split("-")[-1].replace("b", ""))
        config = create_llama_config(size, tp, hardware)
    elif "deepseek" in model.lower():
        size = int(model.split("-")[-1].replace("b", ""))
        config = create_deepseek_config(size, tp, hardware)
    else:
        raise ValueError(f"Unknown model: {model}")

    # Run simulation
    simulator = ServeGenPowerSimulator(config)
    return simulator.generate_power_simulation_data(
        category=workload, duration=duration, rate_requests_per_sec=rate
    )
