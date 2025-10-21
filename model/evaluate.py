"""
Evaluation pipeline for power trace inference models.

Generates metrics and visualization data for:
- Figure 1: Power trace overlays with transition zoom
- Figure 2: Phase duration CDFs (prefill/decode)
- Table 1: Per-configuration metrics (F1, Transition MAE, Power MAPE, Autocorr R²)
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from core.dataset import PowerTraceDataset
from core.utils import load_classifier
from predictors.smooth_sampler import SmoothingSampler
from scipy import stats
from sklearn.metrics import f1_score
from summary_json import load_summary_from_json


@dataclass
class EvaluationMetrics:
    """Metrics for a single configuration."""

    hardware: str
    model: str
    tp: int
    f1_score: float
    transition_mae: float  # Watts
    power_mape: float  # Percentage
    autocorr_r2_mean: float
    autocorr_r2_std: float
    num_samples: int


@dataclass
class PhaseStats:
    """Phase duration statistics for CDF analysis.

    Note: Phase durations are extracted from the workload schedule, which is
    identical for both GT and predictions. These distributions characterize
    the workload, not the prediction accuracy.
    """

    prefill_durations: List[float]  # Prefill phase durations (seconds)
    decode_durations: List[float]  # Decode phase durations (seconds)


@dataclass
class PowerTraceExample:
    """Single power trace example for visualization."""

    time: np.ndarray  # Time values
    power_gt: np.ndarray  # Ground truth power
    power_pred: np.ndarray  # Predicted power
    states_pred: np.ndarray  # Predicted states
    prefill_tokens: np.ndarray  # Prefill token counts over time
    decode_tokens: np.ndarray  # Decode token counts over time
    transition_indices: List[int]  # Indices where transitions occur


def detect_transitions(
    prefill_tokens: np.ndarray,
    decode_tokens: np.ndarray,
    threshold: float = 0.1,
    min_change: int = 1,
) -> List[int]:
    """
    Detect phase transitions (prefill↔decode boundaries).

    Args:
        prefill_tokens: Prefill token counts over time (T,)
        decode_tokens: Decode token counts over time (T,)
        threshold: Relative change threshold
        min_change: Minimum absolute change to consider

    Returns:
        List of indices where transitions occur
    """
    transitions = []

    # Detect prefill drops
    prefill_diff = np.abs(np.diff(prefill_tokens))
    prefill_max = np.max(prefill_tokens) + 1e-6
    prefill_transitions = np.where(
        (prefill_diff > threshold * prefill_max) & (prefill_diff > min_change)
    )[0]

    # Detect decode rises
    decode_diff = np.abs(np.diff(decode_tokens))
    decode_max = np.max(decode_tokens) + 1e-6
    decode_transitions = np.where(
        (decode_diff > threshold * decode_max) & (decode_diff > min_change)
    )[0]

    # Combine and sort (offset by 1 due to diff)
    transitions = sorted(set(list(prefill_transitions + 1) + list(decode_transitions + 1)))

    return transitions


def extract_phase_durations(
    prefill_tokens: np.ndarray,
    decode_tokens: np.ndarray,
    dt: float = 0.25,
) -> Tuple[List[float], List[float]]:
    """
    Extract prefill and decode phase durations from token counts.

    Args:
        prefill_tokens: Prefill token counts over time (T,)
        decode_tokens: Decode token counts over time (T,)
        dt: Time step in seconds

    Returns:
        prefill_durations: List of prefill phase durations (seconds)
        decode_durations: List of decode phase durations (seconds)
    """
    prefill_durations = []
    decode_durations = []

    # Simple heuristic: prefill phase when prefill_tokens > 0, decode when decode_tokens > 0
    in_prefill = False
    in_decode = False
    current_duration = 0

    for t in range(len(prefill_tokens)):
        # Check if we're in prefill phase
        if prefill_tokens[t] > 0:
            if not in_prefill:
                # Start new prefill phase
                if in_decode and current_duration > 0:
                    decode_durations.append(current_duration * dt)
                in_prefill = True
                in_decode = False
                current_duration = 0
            current_duration += 1
        # Check if we're in decode phase
        elif decode_tokens[t] > 0:
            if not in_decode:
                # Start new decode phase
                if in_prefill and current_duration > 0:
                    prefill_durations.append(current_duration * dt)
                in_decode = True
                in_prefill = False
                current_duration = 0
            current_duration += 1
        else:
            # Idle state - close any open phase
            if in_prefill and current_duration > 0:
                prefill_durations.append(current_duration * dt)
            elif in_decode and current_duration > 0:
                decode_durations.append(current_duration * dt)
            in_prefill = False
            in_decode = False
            current_duration = 0

    # Close final phase
    if in_prefill and current_duration > 0:
        prefill_durations.append(current_duration * dt)
    elif in_decode and current_duration > 0:
        decode_durations.append(current_duration * dt)

    return prefill_durations, decode_durations


def compute_power_mape(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error for power predictions.

    Args:
        y_pred: Predicted power values (N, T) or (T,)
        y_true: True power values (N, T) or (T,)

    Returns:
        MAPE as percentage
    """
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()

    # Avoid division by zero
    mask = y_true_flat > 1.0  # Only consider non-negligible power values
    if mask.sum() == 0:
        return 0.0

    mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
    return float(mape)


def compute_autocorr_r2_aggregated(
    y_pred: np.ndarray, y_true: np.ndarray, min_lag: int = 1, max_lag: int = 20
) -> Tuple[float, float]:
    """
    Compute mean and std of R² for autocorrelation across lags 1-20.

    Args:
        y_pred: Predicted power values (N, T)
        y_true: True power values (N, T)
        min_lag: Minimum lag
        max_lag: Maximum lag

    Returns:
        (mean_r2, std_r2)
    """

    def autocorr_at_lag(x, lag):
        """Compute autocorrelation at a specific lag."""
        x_mean = np.mean(x)
        x_std = np.std(x)
        if x_std < 1e-8:
            return 0.0
        x_norm = (x - x_mean) / x_std
        n = len(x_norm)
        if lag >= n:
            return 0.0
        numerator = np.sum(x_norm[:-lag] * x_norm[lag:])
        denominator = n - lag
        return numerator / denominator if denominator > 0 else 0.0

    pred_acf_all = []
    true_acf_all = []

    for lag in range(min_lag, max_lag + 1):
        pred_acf_samples = []
        true_acf_samples = []

        for i in range(len(y_pred)):
            if len(y_pred[i]) >= lag + 4:  # Minimum length requirement
                pred_r = autocorr_at_lag(y_pred[i], lag)
                true_r = autocorr_at_lag(y_true[i], lag)
                pred_acf_samples.append(pred_r)
                true_acf_samples.append(true_r)

        if len(pred_acf_samples) == 0:
            continue

        # Fisher z-transform aggregation
        pred_z = [np.arctanh(np.clip(r, -0.999, 0.999)) for r in pred_acf_samples]
        true_z = [np.arctanh(np.clip(r, -0.999, 0.999)) for r in true_acf_samples]

        pred_z_mean = np.mean(pred_z)
        true_z_mean = np.mean(true_z)

        pred_acf_agg = np.tanh(pred_z_mean)
        true_acf_agg = np.tanh(true_z_mean)

        pred_acf_all.append(pred_acf_agg)
        true_acf_all.append(true_acf_agg)

    # Compute R² across all lags
    if len(pred_acf_all) == 0:
        return 0.0, 0.0

    pred_acf_all = np.array(pred_acf_all)
    true_acf_all = np.array(true_acf_all)

    ss_res = np.sum((true_acf_all - pred_acf_all) ** 2)
    ss_tot = np.sum((true_acf_all - np.mean(true_acf_all)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    r2 = max(0.0, r2)

    # Since we're computing a single R² across lags, std is 0
    # To get std, we'd need to compute per-sample R² values
    # For now, return mean=r2, std=0
    return float(r2), 0.0


def evaluate_configuration(
    dataset: PowerTraceDataset,
    classifier: torch.nn.Module,
    smoother: SmoothingSampler,
    tp: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    min_power: float,
    max_power: float,
    num_samples: int = 100,
    dt: float = 0.25,
    smoothing_window: int = 5,
    device: Optional[torch.device] = None,
) -> Tuple[EvaluationMetrics, PhaseStats, List[PowerTraceExample]]:
    """
    Evaluate a single configuration and collect all metrics.

    Args:
        dataset: PowerTraceDataset
        classifier: Trained GRU classifier
        smoother: SmoothingSampler
        tp: Tensor parallelism value
        mu: State means for this TP
        sigma: State stds for this TP
        min_power: Minimum power for clipping
        max_power: Maximum power for clipping
        num_samples: Number of samples to evaluate
        dt: Time step
        smoothing_window: Smoothing window size
        device: Device for inference

    Returns:
        metrics: EvaluationMetrics
        phase_stats: PhaseStats for CDF analysis
        examples: List of PowerTraceExample for visualization
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier.eval()

    # Get indices for this TP
    tp_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == tp]
    if len(tp_indices) == 0:
        raise ValueError(f"No samples found for TP={tp}")

    # Randomly sample indices
    num_samples = min(num_samples, len(tp_indices))
    sample_indices = np.random.choice(tp_indices, num_samples, replace=False)

    # Collect predictions and ground truth
    all_power_pred = []
    all_power_gt = []
    all_states_pred = []
    all_states_gt = []
    all_x = []

    prefill_durations = []
    decode_durations = []

    examples = []

    for idx in sample_indices:
        trace = dataset.traces[idx]
        x = trace["x"]
        y_gt = trace["y"].flatten()
        z_gt = trace["z"]

        # Run inference
        time_vals, power_pred, states_pred = smoother.sample_power(
            classifier,
            mu,
            sigma,
            schedule_x=x,
            dt=dt,
            smoothing_window=smoothing_window,
        )

        # Clip power
        power_pred = np.clip(power_pred, min_power, max_power)

        # Extract phase durations from the schedule
        # These characterize the workload, not the prediction accuracy
        prefill_dur, decode_dur = extract_phase_durations(x[:, 0], x[:, 1], dt)

        prefill_durations.extend(prefill_dur)
        decode_durations.extend(decode_dur)

        # Store for metrics
        all_power_pred.append(power_pred)
        all_power_gt.append(y_gt)
        all_states_pred.append(states_pred)
        all_states_gt.append(z_gt)
        all_x.append(x)

        # Store example for visualization
        transitions = detect_transitions(x[:, 0], x[:, 1])
        examples.append(
            PowerTraceExample(
                time=time_vals,
                power_gt=y_gt,
                power_pred=power_pred,
                states_pred=states_pred,
                prefill_tokens=x[:, 0],
                decode_tokens=x[:, 1],
                transition_indices=transitions,
            )
        )

    # Compute metrics
    # 1. F1 score (state classification)
    all_states_pred_flat = np.concatenate([s.flatten() for s in all_states_pred])
    all_states_gt_flat = np.concatenate([s.flatten() for s in all_states_gt])
    f1 = f1_score(all_states_gt_flat, all_states_pred_flat, average="macro", zero_division=0)

    # 2. Transition MAE
    from classifiers.train import compute_transition_mae

    # Pad sequences to same length for transition MAE
    max_len = max(len(p) for p in all_power_pred)
    all_x_padded = []
    all_power_pred_padded = []
    all_power_gt_padded = []

    for x, p_pred, p_gt in zip(all_x, all_power_pred, all_power_gt):
        pad_len = max_len - len(p_pred)
        if pad_len > 0:
            x_padded = np.pad(x, ((0, pad_len), (0, 0)), mode='constant')
            p_pred_padded = np.pad(p_pred, (0, pad_len), mode='constant')
            p_gt_padded = np.pad(p_gt, (0, pad_len), mode='constant')
        else:
            x_padded = x
            p_pred_padded = p_pred
            p_gt_padded = p_gt
        all_x_padded.append(x_padded)
        all_power_pred_padded.append(p_pred_padded)
        all_power_gt_padded.append(p_gt_padded)

    all_x_np = np.array(all_x_padded)
    all_power_pred_np = np.array(all_power_pred_padded)
    all_power_gt_np = np.array(all_power_gt_padded)
    transition_mae = compute_transition_mae(all_power_pred_np, all_power_gt_np, all_x_np)

    # 3. Power MAPE
    power_mape = compute_power_mape(all_power_pred_np, all_power_gt_np)

    # 4. Autocorr R² (use original unpadded sequences)
    autocorr_r2_mean, autocorr_r2_std = compute_autocorr_r2_aggregated(
        all_power_pred, all_power_gt, min_lag=1, max_lag=20
    )

    metrics = EvaluationMetrics(
        hardware="",  # Will be filled by caller
        model="",  # Will be filled by caller
        tp=tp,
        f1_score=f1,
        transition_mae=transition_mae,
        power_mape=power_mape,
        autocorr_r2_mean=autocorr_r2_mean,
        autocorr_r2_std=autocorr_r2_std,
        num_samples=num_samples,
    )

    phase_stats = PhaseStats(
        prefill_durations=prefill_durations,
        decode_durations=decode_durations,
    )

    return metrics, phase_stats, examples


def save_metrics_table(metrics_list: List[EvaluationMetrics], output_path: str):
    """Save metrics to CSV table."""
    df = pd.DataFrame([asdict(m) for m in metrics_list])
    df = df[
        [
            "hardware",
            "model",
            "tp",
            "f1_score",
            "transition_mae",
            "power_mape",
            "autocorr_r2_mean",
            "autocorr_r2_std",
            "num_samples",
        ]
    ]
    df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"Saved metrics table to {output_path}")


def save_phase_stats(phase_stats_dict: Dict[str, PhaseStats], output_dir: str):
    """Save phase duration statistics for CDF plotting."""
    os.makedirs(output_dir, exist_ok=True)

    for config_name, stats in phase_stats_dict.items():
        # Save as JSON
        data = {
            "prefill_durations": stats.prefill_durations,
            "decode_durations": stats.decode_durations,
            "num_prefill_phases": len(stats.prefill_durations),
            "num_decode_phases": len(stats.decode_durations),
            "mean_prefill_duration": float(np.mean(stats.prefill_durations)) if stats.prefill_durations else 0.0,
            "mean_decode_duration": float(np.mean(stats.decode_durations)) if stats.decode_durations else 0.0,
            "median_prefill_duration": float(np.median(stats.prefill_durations)) if stats.prefill_durations else 0.0,
            "median_decode_duration": float(np.median(stats.decode_durations)) if stats.decode_durations else 0.0,
        }

        output_path = os.path.join(output_dir, f"phase_stats_{config_name}.json")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    print(f"Saved phase statistics to {output_dir}")


def save_trace_examples(examples_dict: Dict[str, List[PowerTraceExample]], output_dir: str):
    """Save power trace examples for visualization."""
    os.makedirs(output_dir, exist_ok=True)

    for config_name, examples in examples_dict.items():
        config_dir = os.path.join(output_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)

        for i, example in enumerate(examples[:5]):  # Save top 5 examples
            # Convert numpy types to native Python types for JSON serialization
            data = {
                "time": [float(x) for x in example.time],
                "power_gt": [float(x) for x in example.power_gt],
                "power_pred": [float(x) for x in example.power_pred],
                "states_pred": [int(x) for x in example.states_pred],
                "prefill_tokens": [float(x) for x in example.prefill_tokens],
                "decode_tokens": [float(x) for x in example.decode_tokens],
                "transition_indices": [int(x) for x in example.transition_indices],
            }

            output_path = os.path.join(config_dir, f"example_{i}.json")
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

    print(f"Saved trace examples to {output_dir}")


def run_evaluation(
    data_file: str,
    weights_dir: str,
    summary_dir: str,
    output_dir: str,
    configs: List[Tuple[str, str, int]],  # List of (model, hardware, tp)
    num_samples: int = 100,
    device: Optional[torch.device] = None,
):
    """
    Run evaluation pipeline across multiple configurations.

    Args:
        data_file: Path to NPZ data file
        weights_dir: Directory containing model weights
        summary_dir: Directory containing summary JSON files
        output_dir: Output directory for results
        configs: List of (model, hardware, tp) tuples to evaluate
        num_samples: Number of samples per configuration
        device: Device for inference
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    all_metrics = []
    all_phase_stats = {}
    all_examples = {}

    for model, hardware, tp in configs:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {model} on {hardware} with TP={tp}")
        print(f"{'=' * 60}")

        # Load dataset
        dataset = PowerTraceDataset(data_file)

        # Load summary
        summary_path = os.path.join(summary_dir, f"model_summary_{model}_{hardware}.json")
        if not os.path.exists(summary_path):
            print(f"Warning: Summary file not found: {summary_path}")
            continue

        summary = load_summary_from_json(summary_path)
        tp_str = str(tp)

        if tp_str not in summary:
            print(f"Warning: TP={tp} not found in summary")
            continue

        # Load classifier
        weights_path = os.path.join(weights_dir, f"{model}_{hardware}_tp{tp}.pt")
        if not os.path.exists(weights_path):
            print(f"Warning: Weights not found: {weights_path}")
            continue

        classifier = load_classifier(
            weights_path,
            device=device,
            Dx=summary[tp_str]["Dx"],
            K=summary[tp_str]["K"],
        )

        # Create smoother
        smoother = SmoothingSampler(state_stats=summary[tp_str]["state_stats"])

        # Run evaluation
        try:
            metrics, phase_stats, examples = evaluate_configuration(
                dataset=dataset,
                classifier=classifier,
                smoother=smoother,
                tp=tp,
                mu=np.array(summary[tp_str]["mu_values"]),
                sigma=np.array(summary[tp_str]["sigma_values"]),
                min_power=summary[tp_str]["min_power"],
                max_power=summary[tp_str]["max_power"],
                num_samples=num_samples,
                device=device,
            )

            # Fill in metadata
            metrics.hardware = hardware
            metrics.model = model

            all_metrics.append(metrics)
            config_name = f"{model}_{hardware}_tp{tp}"
            all_phase_stats[config_name] = phase_stats
            all_examples[config_name] = examples

            print(f"Results:")
            print(f"  F1: {metrics.f1_score:.4f}")
            print(f"  Transition MAE: {metrics.transition_mae:.2f} W")
            print(f"  Power MAPE: {metrics.power_mape:.2f}%")
            print(
                f"  Autocorr R²: {metrics.autocorr_r2_mean:.4f} ± {metrics.autocorr_r2_std:.4f}"
            )
            print(f"  Num Prefill Phases: {len(phase_stats.prefill_durations)}")
            print(f"  Num Decode Phases: {len(phase_stats.decode_durations)}")
            if len(phase_stats.prefill_durations) > 0:
                print(f"  Mean Prefill Duration: {np.mean(phase_stats.prefill_durations):.2f}s")
            if len(phase_stats.decode_durations) > 0:
                print(f"  Mean Decode Duration: {np.mean(phase_stats.decode_durations):.2f}s")

        except Exception as e:
            print(f"Error evaluating {model} {hardware} TP={tp}: {e}")
            continue

    # Save results
    if len(all_metrics) > 0:
        save_metrics_table(all_metrics, os.path.join(output_dir, "metrics_table.csv"))
        save_phase_stats(all_phase_stats, os.path.join(output_dir, "phase_stats"))
        save_trace_examples(all_examples, os.path.join(output_dir, "trace_examples"))

        print(f"\n{'=' * 60}")
        print(f"Evaluation complete! Results saved to {output_dir}")
        print(f"{'=' * 60}")
    else:
        print("\nNo configurations were successfully evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate power trace inference models")
    parser.add_argument("--data-file", type=str, required=True, help="Path to NPZ data file")
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="./gru_classifier_weights/",
        help="Directory containing model weights",
    )
    parser.add_argument(
        "--summary-dir",
        type=str,
        default="./summary_data/",
        help="Directory containing summary JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of samples to evaluate per config"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to run inference on (cuda, cpu, etc.)"
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        help='Configurations to evaluate in format "model:hardware:tp" (e.g., "llama-3-8b:a100:1")',
    )

    args = parser.parse_args()

    # Parse configs
    configs = []
    if args.configs:
        for config_str in args.configs:
            parts = config_str.split(":")
            if len(parts) != 3:
                raise ValueError(f"Invalid config format: {config_str}. Use model:hardware:tp")
            model, hardware, tp = parts
            configs.append((model, hardware, int(tp)))
    else:
        # Default: evaluate all available configs
        configs = [
            ("llama-3-8b", "a100", 1),
            ("llama-3-8b", "a100", 2),
            ("llama-3-8b", "h100", 1),
            ("llama-3-8b", "h100", 2),
            ("llama-3-70b", "h100", 4),
            ("llama-3-70b", "h100", 8),
        ]

    device = None
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_evaluation(
        data_file=args.data_file,
        weights_dir=args.weights_dir,
        summary_dir=args.summary_dir,
        output_dir=args.output_dir,
        configs=configs,
        num_samples=args.num_samples,
        device=device,
    )
