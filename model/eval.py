import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from core.dataset import PowerTraceDataset
from core.utils import load_classifier
from predictors.smooth_sampler import SmoothingSampler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer model to create power traces")
    parser.add_argument(
        "--data-file", type=str, required=True, help="Path to the NPZ data file"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "llama-3-8b",
            "llama-3-70b",
            "deepseek-r1-distill-8b",
            "deepseek-r1-distill-70b",
        ],
        help="LLM name",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism value to infer on",
    )
    parser.add_argument("--hardware_accelerator", type=str, default="a100")
    parser.add_argument(
        "--weights-path",
        type=str,
        default="./gru_classifier_weights/",
        help="Path to the classifier weights folder",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on on (cuda, cuda:0, cpu, etc.).",
    )
    args = parser.parse_args()

    device = None
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PowerTraceDataset(args.data_file, use_gmm=True)
    tp = args.tp
    classifier = load_classifier(
        args.weights_path + args.model + f"_{args.hardware_accelerator}_tp{tp}.pt",
        device=device,
        Dx=dataset.traces[0]["x"].shape[1],
        K=dataset.state_labels[tp].n_components,
    )
    smoother = SmoothingSampler(dataset)
    tp1_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == tp]

    all_original_power = []
    all_sampled_power = []
    sampled_power_arrs = {}
    import scipy.stats as stats

    min_power = float("inf")
    max_power = float("-inf")
    for idx in tp1_indices:
        trace_power = dataset.traces[idx]["y"].flatten()
        min_power = min(min_power, np.min(trace_power))
        max_power = max(max_power, np.max(trace_power))

    for idx in tp1_indices:
        time, power, states = smoother.sample_power(
            classifier[tp],
            dataset.mu[tp],
            dataset.sigma[tp],
            dataset.traces[idx]["x"],
            dt=0.25,
            smoothing_window=5,
        )
        power = np.clip(power, min_power, max_power)
        original_power = dataset.traces[idx]["y"].flatten()
        sampled_power = power.flatten()
        sampled_power_arrs[idx] = (time, sampled_power)

        all_original_power.append(original_power)
        all_sampled_power.append(sampled_power)
        ks_stat, p_value = stats.ks_2samp(original_power, sampled_power)
        print(f"Trace {idx} - KS Statistic: {ks_stat}, p-value: {p_value}")

    all_original_power = np.concatenate(all_original_power)
    all_sampled_power = np.concatenate(all_sampled_power)
    overall_ks_stat, overall_p_value = stats.ks_2samp(
        all_original_power, all_sampled_power
    )
    print(f"Overall - KS Statistic: {overall_ks_stat}, p-value: {overall_p_value}")
    sorted_original_power = np.sort(all_original_power)
    sorted_sampled_power = np.sort(all_sampled_power)

    cdf_original = np.arange(1, len(sorted_original_power) + 1) / len(
        sorted_original_power
    )
    cdf_sampled = np.arange(1, len(sorted_sampled_power) + 1) / len(
        sorted_sampled_power
    )

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_original_power, cdf_original, label="Original CDF")
    plt.plot(sorted_sampled_power, cdf_sampled, label="Sampled CDF")
    plt.xlabel("Power (W)")
    plt.ylabel("CDF")
    plt.legend()
    plt.title("CDF of All Original and Sampled Power")
    plt.grid(True, alpha=0.3)
    plt.savefig("cdf_power_trace.pdf")

    print(dataset.tp_all)

    total_energy_original = np.trapezoid(all_original_power, dx=0.25)
    total_energy_sampled = np.trapezoid(all_sampled_power, dx=0.25)
    print(f"Total energy consumed (original): {total_energy_original:.2f} J")
    print(f"Total energy consumed (sampled): {total_energy_sampled:.2f} J")
    print(f"Energy difference: {total_energy_sampled - total_energy_original:.2f} J")
