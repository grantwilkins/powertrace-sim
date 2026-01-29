from __future__ import annotations

import functools
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from model.core.utils import load_classifier
    from model.predictors.smooth_sampler import SmoothingSampler
    from model.simulators.arrival_simulator import (
        ServeGenPowerSimulator,
        ServeGenRequest,
        ServingConfig,
        _get_perf_db_path,
        _infer_model_for_key,
    )
except ModuleNotFoundError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from core.utils import load_classifier
    from predictors.smooth_sampler import SmoothingSampler
    from simulators.arrival_simulator import (
        ServeGenPowerSimulator,
        ServeGenRequest,
        ServingConfig,
        _get_perf_db_path,
        _infer_model_for_key,
    )


def _resolve_weights_basename(config: ServingConfig) -> str:
    """
    Resolve the model basename used for weight files in `model/best_weights/`.

    Examples:
        llama-3-8b_h100_tp1.pt -> basename: "llama-3-8b"
        deepseek-r1-distill-8b_h100_tp1.pt -> basename: "deepseek-r1-distill-8b"
    """
    model_name = (config.model_name or "").strip().lower()
    size_suffix = f"{config.model_size_b}b"

    if "deepseek" in model_name and "distill" not in model_name:
        if (config.perf_db_model_name or "").startswith("deepseek-r1-distill"):
            return f"deepseek-r1-distill-{size_suffix}"

    if model_name.endswith("b") or f"-{size_suffix}" in model_name:
        return model_name

    return f"{model_name}-{size_suffix}"


def _resolve_weights_path(config: ServingConfig, base_dir: Optional[str] = None) -> str:
    if base_dir is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.normpath(os.path.join(this_dir, "../best_weights"))
    basename = _resolve_weights_basename(config)
    filename = f"{basename}_{config.hardware.lower()}_tp{config.tensor_parallelism}.pt"
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Classifier weights not found: {path}. Expected pattern '<model>-<size>b_<hardware>_tp<tp>.pt'"
        )
    return path


def _resample_series(
    timestamps: np.ndarray, values: np.ndarray, output_dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a regular series to a new time step using linear interpolation.

    Assumes `timestamps` are uniformly spaced and increasing.
    """
    if output_dt is None or output_dt <= 0:
        raise ValueError("output_dt must be positive")
    if len(timestamps) == 0:
        return timestamps, values

    t0 = float(timestamps[0])
    t1 = float(timestamps[-1])
    new_ts = np.arange(t0, t1 + 1e-9, output_dt)
    new_vals = np.interp(new_ts, timestamps, values)
    return new_ts, new_vals


@functools.lru_cache(maxsize=16)
def _cached_load_classifier(weights_path: str, Dx: int, K: int, device_str: str):
    device = torch.device(device_str)
    return load_classifier(weights_path, device=device, Dx=Dx, K=K)


@functools.lru_cache(maxsize=64)
def _cached_sampler(
    perf_db_path: str,
    db_model_name: str,
    size_b: int,
    hardware: str,
    tp: int,
):
    return SmoothingSampler.from_performance_database(
        database_path=perf_db_path,
        model_name=db_model_name,
        model_size_b=size_b,
        hardware=hardware,
        tensor_parallelism=tp,
    )


class ServerPowerSimulator:
    """
    End-to-end server power simulator that composes ServeGen arrivals, system timeline
    feature construction, GRU inference, and state-conditioned smoothing.

    - Inference is performed on a 0.25 s grid matching training.
    - Optional resampling to arbitrary `output_dt` is supported via interpolation.
    - Reported power is total server power for the configured setup (TP across GPUs).
    """

    def __init__(
        self, serving_config: ServingConfig, perf_db_path: Optional[str] = None
    ):
        self.serving_config = serving_config
        self.perf_db_path = perf_db_path or _get_perf_db_path()
        self._servegen_sim = ServeGenPowerSimulator(serving_config)

    def simulate_server_power(
        self,
        category: str = "language",
        model_type: str = "m-large",
        duration: float = 3600.0,
        rate_requests_per_sec: float = 1.0,
        time_window: Optional[str] = None,
        seed: int = 0,
        classifier_weights_path: Optional[str] = None,
        smoothing_window_steps: int = 5,
        output_dt: Optional[float] = None,
        return_profile: bool = True,
        rate_fn: Optional[Callable[[float], float]] = None,
        use_fast_workload: bool = False,
        precomputed_requests: Optional[List[ServeGenRequest]] = None,
    ) -> Dict[str, object]:
        """
        Generate a power time series for the configured server.

        Args:
            category: ServeGen category ("language", "reason", "multimodal")
            model_type: ServeGen model size category ("m-small", "m-mid", "m-large")
            duration: Simulation duration in seconds
            rate_requests_per_sec: Target request arrival rate
            time_window: Optional wall-clock window (e.g., "18:00-19:00")
            seed: Random seed
            classifier_weights_path: Optional explicit path to GRU weights. If None, resolved automatically.
            smoothing_window_steps: Median smoothing window in steps (0.25 s per step)
            output_dt: Optional resampling step (seconds). If None or 0.25, returns inference grid.

        Returns:
            Dict with keys:
                - timestamps: np.ndarray (seconds, absolute timeline)
                - watts: np.ndarray (total server power)
                - states: np.ndarray (argmax state per step)
                - feature_matrix: np.ndarray (z-scored features)
                - timeline: Dict[str, np.ndarray] (system timeline)
                - weights_path: str (resolved classifier weights path)
        """
        import time

        t0 = time.perf_counter()
        if precomputed_requests is not None:
            # Bypass ServeGen workload generation when requests are provided
            processed_requests = (
                self._servegen_sim.system_simulator.simulate_request_processing(
                    precomputed_requests
                )
            )
            timeline = self._servegen_sim.system_simulator.create_system_timeline(
                processed_requests
            )
            feature_matrix = self._servegen_sim.system_simulator.create_feature_matrix(
                timeline
            )
        else:
            sim_data = self._servegen_sim.generate_power_simulation_data(
                category=category,
                model_type=model_type,
                duration=duration,
                rate_requests_per_sec=rate_requests_per_sec,
                time_window=time_window,
                seed=seed,
                rate_fn=rate_fn,
                use_fast_workload=use_fast_workload,
            )
            feature_matrix: np.ndarray = sim_data["feature_matrix"]
            timeline: Dict[str, np.ndarray] = sim_data["timeline"]
        t1 = time.perf_counter()

        if classifier_weights_path is None:
            classifier_weights_path = _resolve_weights_path(self.serving_config)
        Dx = int(feature_matrix.shape[1])
        K = 6  # as confirmed
        classifier = _cached_load_classifier(classifier_weights_path, Dx, K, "cpu")
        t2 = time.perf_counter()

        db_model_name = _infer_model_for_key(
            self.serving_config.model_name, self.serving_config.perf_db_model_name
        )
        if not db_model_name:
            raise ValueError(
                "Unable to map model name to performance DB family. "
                f"Got model_name='{self.serving_config.model_name}'. "
                "Set 'perf_db_model_name' to an explicit value (e.g., 'llama-3.1', 'deepseek-r1-distill')."
            )
        sampler = _cached_sampler(
            self.perf_db_path,
            db_model_name,
            self.serving_config.model_size_b,
            self.serving_config.hardware,
            self.serving_config.tensor_parallelism,
        )
        mu = sampler.mu[self.serving_config.tensor_parallelism]
        sigma = sampler.sigma[self.serving_config.tensor_parallelism]
        t3 = time.perf_counter()

        dt = 0.25
        t_rel, watts, states = sampler.sample_power(
            net=classifier,
            mu=mu,
            sigma=sigma,
            schedule_x=feature_matrix,
            dt=dt,
            smoothing_window=smoothing_window_steps,
        )
        t4 = time.perf_counter()

        ts = timeline["timestamps"].astype(float)
        if len(ts) != len(watts):
            min_len = min(len(ts), len(watts))
            ts = ts[:min_len]
            watts = watts[:min_len]
            states = states[:min_len]

        if output_dt is not None and abs(output_dt - dt) > 1e-9:
            rs_ts, rs_watts = _resample_series(ts, watts, output_dt)
            ts_out, watts_out = rs_ts, rs_watts
        else:
            ts_out, watts_out = ts, watts
        t5 = time.perf_counter()

        # Compute token arrival series (input/output) resampled to ts_out
        tokens_in = None
        tokens_out = None
        try:
            small_ts = np.asarray(timeline["timestamps"], dtype=float)
            in_small = np.asarray(timeline.get("input_tokens", None), dtype=float)
            out_small = np.asarray(timeline.get("output_tokens", None), dtype=float)
            if in_small.size > 0 and out_small.size > 0:
                # Interpolate cumulative tokens to ts_out, then take discrete diff per step
                cum_in_small = np.cumsum(in_small)
                cum_out_small = np.cumsum(out_small)
                cum_in_interp = np.interp(
                    ts_out,
                    small_ts,
                    cum_in_small,
                    left=0.0,
                    right=float(cum_in_small[-1]),
                )
                cum_out_interp = np.interp(
                    ts_out,
                    small_ts,
                    cum_out_small,
                    left=0.0,
                    right=float(cum_out_small[-1]),
                )
                tokens_in = np.concatenate([[cum_in_interp[0]], np.diff(cum_in_interp)])
                tokens_out = np.concatenate(
                    [[cum_out_interp[0]], np.diff(cum_out_interp)]
                )
        except Exception:
            pass
        result = {
            "timestamps": ts_out,
            "watts": watts_out,
            "states": states,
            "feature_matrix": feature_matrix,
            "timeline": timeline,
            "weights_path": classifier_weights_path,
        }
        if tokens_in is not None and tokens_out is not None:
            result["tokens_in"] = tokens_in
            result["tokens_out"] = tokens_out
        if return_profile:
            result["profile_seconds"] = {
                "servegen_and_timeline": t1 - t0,
                "load_classifier": t2 - t1,
                "load_gmm": t3 - t2,
                "gru_inference_and_smoothing": t4 - t3,
                "resample": t5 - t4,
                "total": t5 - t0,
            }
        return result
