import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import acf, pacf

from model.core.dataset import PowerTraceDataset
from model.core.utils import load_classifier
from model.predictors.smooth_sampler import SmoothingSampler


def compute_autocorrelation_metrics(real_trace, synthetic_trace, max_lags=50):
    """
    Compare temporal autocorrelation between real and synthetic traces
    """
    real_acf = acf(real_trace, nlags=max_lags, fft=True)
    synthetic_acf = acf(synthetic_trace, nlags=max_lags, fft=True)
    acf_correlation, _ = pearsonr(real_acf, synthetic_acf)
    acf_mae = np.mean(np.abs(real_acf - synthetic_acf))

    return {
        "real_acf": real_acf,
        "synthetic_acf": synthetic_acf,
        "acf_correlation": acf_correlation,
        "acf_mae": acf_mae,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer model to create power traces")
    parser.add_argument(
        "--data-file", type=str, required=True, help="Path to the NPZ data file"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["llama-3-8b", "llama-3-70b", "gpt-oss-20b", "gpt-oss-120b"],
        help="LLM name",
    )
    # parser.add_argument(
    #     "--tp",
    #     type=int,
    #     default=1,
    #     help="Tensor parallelism value to infer on",
    # )
    parser.add_argument("--hardware_accelerator", type=str, default="a100")
    parser.add_argument(
        "--weights-path",
        type=str,
        default="model/sharegpt_gru_classifier_weights/",
        help="Path to the classifier weights folder",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on on (cuda, cuda:0, cpu, etc.).",
    )
    args = parser.parse_args()

    model = args.model
    hw = args.hardware_accelerator
    device = None
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PowerTraceDataset(args.data_file)
    # tp = args.tp

    import pandas as pd
    import seaborn as sns

    # Store all CDF data for plotting
    cdf_data = []

    for tp in [1, 2]:
        print(f"\nProcessing TP={tp}")
        classifier = load_classifier(
            args.weights_path + args.model + f"_{args.hardware_accelerator}_tp{tp}.pt",
            device=device,
            Dx=dataset.traces[0]["x"].shape[1],
            K=dataset.state_labels[tp].n_components,
        )
        smoother = SmoothingSampler(dataset)
        tp_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == tp]

        all_original_power = []
        all_sampled_power = []
        sampled_power_arrs = {}
        import scipy.stats as stats

        min_power = float("inf")
        max_power = float("-inf")
        for idx in tp_indices:
            trace_power = dataset.traces[idx]["y"].flatten()
            min_power = min(min_power, np.min(trace_power))
            max_power = max(max_power, np.max(trace_power))

        plt.clf()
        plt.plot(dataset.traces[tp_indices[0]]["z"].flatten())
        plt.xlim(0, 50)
        plt.title("State Sequence for First Trace")
        plt.xlabel("Time Step")
        plt.ylabel("State")
        plt.savefig(f"state_sequence_trace_{tp}.pdf")
        plt.close()
        plt.plot(dataset.traces[tp_indices[0]]["y"].flatten())
        plt.xlim(0, 50)
        plt.title("Power Trace for First Trace")
        plt.xlabel("Time Step")
        plt.ylabel("Power (W)")
        plt.savefig(f"power_trace_{tp}.pdf")
        plt.close()

        for idx in tp_indices:
            time, power, states = smoother.sample_power(
                classifier,
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

        # Add data to cdf_data list
        cdf_data.extend(
            [
                {"Power": p, "CDF": c, "Type": "Original", "TP": tp}
                for p, c in zip(sorted_original_power, cdf_original)
            ]
        )
        cdf_data.extend(
            [
                {"Power": p, "CDF": c, "Type": "Sampled", "TP": tp}
                for p, c in zip(sorted_sampled_power, cdf_sampled)
            ]
        )

        print(f"TP={tp} indices:", dataset.tp_all)

        total_energy_original = np.trapz(all_original_power, dx=0.25)
        total_energy_sampled = np.trapz(all_sampled_power, dx=0.25)
        print(f"Total energy consumed (original): {total_energy_original:.2f} J")
        print(f"Total energy consumed (sampled): {total_energy_sampled:.2f} J")
        print(
            f"Energy difference: {total_energy_sampled - total_energy_original:.2f} J"
        )

        # Compute autocorrelation metrics
        print("Computing autocorrelation metrics...")
        autocorr_metrics = compute_autocorrelation_metrics(
            all_original_power, all_sampled_power
        )
        print("Autocorrelation metrics:")
        print(f"ACF Correlation: {autocorr_metrics['acf_correlation']:.4f}")
        print(f"ACF MAE: {autocorr_metrics['acf_mae']:.4f}")

        # Compute earth mover's distance
        from scipy.stats import wasserstein_distance

        emd = wasserstein_distance(sorted_original_power, sorted_sampled_power)
        print(f"Earth Mover's Distance: {emd:.4f}")

        # Calculate p99 error
        p99_original = np.percentile(all_original_power, 99)
        p99_sampled = np.percentile(all_sampled_power, 99)
        p99_error = np.abs(p99_original - p99_sampled) / p99_original * 100
        print(f"P99 Error: {p99_error:.2f}%")

        # Calculate p95 error
        p95_original = np.percentile(all_original_power, 95)
        p95_sampled = np.percentile(all_sampled_power, 95)
        p95_error = np.abs(p95_original - p95_sampled) / p95_original * 100
        print(f"P95 Error: {p95_error:.2f}%")

        # Compute NRMSE
        nrmse = np.sqrt(np.mean((all_original_power - all_sampled_power) ** 2)) / (
            np.max(all_original_power) - np.min(all_original_power)
        )
        print(f"NRMSE: {nrmse:.4f}")

    df = pd.DataFrame(cdf_data)
    if len(df) > 10000:  # If more than 10k points, sample down
        df = df.sample(n=10000, random_state=42)
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    sns.set_context("paper", font_scale=1.2)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "Times New Roman"

    plt.figure(figsize=(4, 3))
    sns.lineplot(
        data=df,
        x="Power",
        y="CDF",
        style="Type",
        hue="TP",
        lw=1.5,
        palette="colorblind",
    )
    # plt.xlim(0, 1600)
    plt.ylim(0, 1.05)
    plt.xlabel("Active GPU Power (W)")
    plt.ylabel("CDF")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"cdf-power-trace_{model}_{hw}_combined.pdf")
    plt.close()
