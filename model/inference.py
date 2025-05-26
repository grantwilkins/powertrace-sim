import argparse
import itertools
import os
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Dataset


def histogram_requests(
    bin_ts: np.ndarray, req_ts: np.ndarray, in_tok: np.ndarray, out_tok: np.ndarray
):
    dt = np.median(np.diff(bin_ts)) if len(bin_ts) > 1 else 0.25
    if len(bin_ts) > 1:
        if not np.all(np.diff(bin_ts) > 0):
            bin_ts = np.unique(bin_ts)
            if len(bin_ts) <= 1:
                bin_ts = (
                    np.array([0, dt])
                    if len(bin_ts) == 0
                    else np.array([bin_ts[0], bin_ts[0] + dt])
                )
    else:
        bin_ts = (
            np.array([0, dt])
            if len(bin_ts) == 0
            else np.array([bin_ts[0], bin_ts[0] + dt])
        )
    edges = np.append(bin_ts, bin_ts[-1] + dt)
    new_req_cnt, _ = np.histogram(req_ts, edges)
    new_in_tok, _ = np.histogram(req_ts, edges, weights=in_tok)
    new_out_tok, _ = np.histogram(req_ts, edges, weights=out_tok)

    return new_req_cnt.astype("float32"), new_in_tok, new_out_tok


def make_schedule_matrix(trace_dict):
    """
    trace_dict contains 1-D numpy arrays *already cut to true length*.
    Returns x_t  (T × Dx)  where columns are z-scored.
    """

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
        axis=1,
    ).astype("float32")

    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True) + 1e-6
    return (x - mu) / sd


class PowerTraceDataset(Dataset):
    """
    Returns  (x_t  ,  y_t  ,  z_t)  for one whole trace.
              (T,3)   (T,1)   (T,)
    """

    def __init__(self, npz_file: str, K: int = 6, use_gmm: bool = False):
        d = np.load(npz_file, allow_pickle=True)
        self.traces: List[Dict[str, np.ndarray]] = []
        self.tp_all: List[int] = []
        tp_array = d["tensor_parallelism"]

        n_exp = d["timestamps"].shape[0]
        for i in range(n_exp):
            bin_mask = d["timestamps"][i] > 0
            req_mask = d["request_timestamps"][i] > 0

            trace = dict(
                timestamps=d["timestamps"][i][bin_mask],
                power=d["power_traces"][i][bin_mask].astype("float32"),
                prefill_tokens=d["prefill_tokens"][i][bin_mask],
                decode_tokens=d["decode_tokens"][i][bin_mask],
                active_requests=d["active_requests"][i][bin_mask],
                request_ts=d["request_timestamps"][i][req_mask],
                input_tokens=d["input_tokens"][i][req_mask],
                output_tokens=d["output_tokens"][i][req_mask],
            )

            trace["x"] = make_schedule_matrix(trace)
            trace["y"] = trace["power"][:, None]  # (T,1)
            self.traces.append(trace)
            self.tp_all.append(int(tp_array[i]))

        self.state_labels: Dict[int, np.ndarray] = {}
        unique_tp = np.unique(self.tp_all)
        for tp in unique_tp:
            y_concat = np.concatenate(
                [tr["y"] for tr, tp_i in zip(self.traces, self.tp_all) if tp_i == tp]
            ).reshape(-1, 1)

            if use_gmm:
                model = GaussianMixture(
                    n_components=K, covariance_type="diag", n_init=10, random_state=0
                ).fit(y_concat)
            else:
                model = KMeans(n_clusters=K, n_init=10, random_state=0).fit(y_concat)
            self.state_labels[tp] = model

        for tr, tp in zip(self.traces, self.tp_all):
            model = self.state_labels[tp]
            tr["z"] = model.predict(tr["y"]).astype("int64")

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tr = self.traces[idx]
        return (
            torch.from_numpy(tr["x"]),
            torch.from_numpy(tr["y"]),
            torch.from_numpy(tr["z"]),
        )


class GRUClassifier(nn.Module):
    def __init__(self, Dx, K, H=64):
        super().__init__()
        self.gru = nn.GRU(Dx, H, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * H, K)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.fc(h)


def train_classifiers(dataset, hidden_size=64, device=None, classifiers=None):
    """Train separate GRU classifiers for each tensor parallelism value."""
    if classifiers is None:
        classifiers = {}
    training_losses = {}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    for tp in sorted(set(dataset.tp_all)):
        print(f"Training classifier for TP={tp}")
        tp_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == tp]

        class TPDataset(Dataset):
            def __init__(self, parent_dataset, indices):
                self.parent = parent_dataset
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                return self.parent[self.indices[idx]]

        tp_dataset = TPDataset(dataset, tp_indices)
        loader = DataLoader(tp_dataset, batch_size=1, shuffle=True)
        x_sample, y_sample, z_sample = dataset[tp_indices[0]]
        Dx = x_sample.shape[1]
        K = len(torch.unique(z_sample))
        if tp not in classifiers:
            classifier = GRUClassifier(Dx, K, H=hidden_size).to(device)
        else:
            classifier = classifiers[tp].to(device)
        classifier.train()
        classifier.to(device)

        optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-3)
        criterion = nn.CrossEntropyLoss()

        epoch_losses = []

        for epoch in range(500):
            epoch_loss = 0.0
            batch_losses = []
            progress_bar = tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/500")
            for x, y, z in progress_bar:
                x = x.to(device)
                z = z.to(device)

                optimizer.zero_grad()
                logits = classifier(x)
                loss = criterion(logits.view(-1, K), z.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                optimizer.step()

                batch_loss = loss.item()
                batch_losses.append(batch_loss)
                epoch_loss += batch_loss
                progress_bar.set_postfix({"loss": f"{batch_loss:.4f}"})

            avg_loss = epoch_loss / len(loader)
            epoch_losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"TP {tp}, Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        classifiers[tp] = classifier
        training_losses[tp] = epoch_losses
        np.save(f"training_losses_tp{tp}.npy", np.array(epoch_losses))
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_losses)
        plt.title(f"Training Loss for TP={tp}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(f"training_loss_tp{tp}.pdf")
        plt.close()

        torch.save(
            classifier.state_dict(),
            f"classifier_llama3_8b_a100_tp{tp}.pt",
        )

    return classifiers, training_losses


def load_classifier(path, device=None):
    """
    Load a classifier from a file.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading classifier from {path} on device: {device}")
    classifier = GRUClassifier(6, 6).to(device)
    classifier.load_state_dict(torch.load(path, map_location=device))
    return classifier


def sample_power(net, mu, sigma, schedule_x, dt=0.25):
    """
    schedule_x:  (T,Dx)  – already z-scored feature matrix on desired Δt grid
    """
    net.eval()
    with torch.no_grad():
        logits = net(torch.from_numpy(schedule_x[None]).float()).squeeze(0)  # (T,K)
        probs = torch.softmax(logits, -1).cpu().numpy()
    K = probs.shape[1]
    z = np.array([np.random.choice(K, p=p) for p in probs])
    watts = np.random.normal(mu[z], sigma[z])

    t = np.arange(len(z)) * dt
    return t, watts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IOHMM model on power traces")
    parser.add_argument(
        "--data_file", type=str, required=True, help="Path to the NPZ data file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cuda, cuda:0, cpu, etc.). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--use_gmm",
        action="store_true",
        help="Use Gaussian Mixture Model instead of KMeans for clustering",
    )
    args = parser.parse_args()
    from token_scheduler import ModelConfig, TokenSimulator

    token_sim = TokenSimulator.from_npz(npz_file=args.data_file)
    results = token_sim.run_simulation(config=ModelConfig(8, 2, "A100"))
    feature_vec = token_sim.prepare_for_inference(results)

    # Set device

    device = None
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PowerTraceDataset(args.data_file, use_gmm=args.use_gmm)
    classifiers = {
        1: load_classifier("classifier_llama3_8b_a100_tp1_gmm.pt", device=device)
    }
    # classifiers, training_losses = train_classifiers(dataset, device=device)

    K = 6
    mu = {}
    sigma = {}

    for tp in dataset.state_labels:
        model = dataset.state_labels[tp]
        mu[tp] = np.zeros(K)
        sigma[tp] = np.zeros(K)

        if args.use_gmm:
            for k in range(K):
                mu[tp][k] = model.means_[k][0]
                sigma[tp][k] = np.sqrt(model.covariances_[k][0])
        else:
            for k in range(K):
                mu[tp][k] = model.cluster_centers_[k][0]

                cluster_points = []
                for tr, tp_i in zip(dataset.traces, dataset.tp_all):
                    if tp_i == tp:
                        k_indices = np.where(tr["z"] == k)[0]
                        if len(k_indices) > 0:
                            cluster_points.append(tr["y"][k_indices])

                if cluster_points:
                    cluster_points = np.concatenate(cluster_points)
                    centroid = model.cluster_centers_[k]
                    distances = np.sqrt(
                        np.sum((cluster_points - centroid) ** 2, axis=1)
                    )
                    sigma[tp][k] = np.mean(distances)
                else:
                    sigma[tp][k] = 1.0
    tp = 1
    # Get all traces with tp==1
    tp1_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == tp]

    all_original_power = []
    all_sampled_power = []

    import matplotlib.pyplot as plt
    import seaborn as sns

    time, power = sample_power(
        classifiers[1],
        mu[tp],
        sigma[tp] * 0.2,
        feature_vec,
        dt=0.25,
    )

    # import scipy.stats as stats
    # for idx in tp1_indices:
    #     time, power = sample_power(
    #         classifiers[1],
    #         mu[tp],
    #         sigma[tp] * 0.2,
    #         dataset.traces[idx]["x"],
    #         dt=0.25,
    #     )

    #     # Store original and sampled power for this trace
    #     original_power = dataset.traces[idx]["y"].flatten()
    #     sampled_power = power.flatten()

    #     all_original_power.append(original_power)
    #     all_sampled_power.append(sampled_power)

    #     # Run KS test for this individual trace
    #     ks_stat, p_value = stats.ks_2samp(original_power, sampled_power)
    #     print(f"Trace {idx} - KS Statistic: {ks_stat}, p-value: {p_value}")

    # # Concatenate all power traces for overall comparison
    # all_original_power = np.concatenate(all_original_power)
    # all_sampled_power = np.concatenate(all_sampled_power)

    # # Run KS test on all data combined
    # overall_ks_stat, overall_p_value = stats.ks_2samp(
    #     all_original_power, all_sampled_power
    # )
    # print(f"Overall - KS Statistic: {overall_ks_stat}, p-value: {overall_p_value}")

    # # plot cdf of original and sampled power
    # # Use all concatenated power data for CDF
    # sorted_original_power = np.sort(all_original_power)
    # sorted_sampled_power = np.sort(all_sampled_power)

    # cdf_original = np.arange(1, len(sorted_original_power) + 1) / len(
    #     sorted_original_power
    # )
    # cdf_sampled = np.arange(1, len(sorted_sampled_power) + 1) / len(
    #     sorted_sampled_power
    # )

    # plt.figure(figsize=(10, 6))
    # plt.plot(sorted_original_power, cdf_original, label="Original CDF")
    # plt.plot(sorted_sampled_power, cdf_sampled, label="Sampled CDF")
    # plt.xlabel("Power (W)")
    # plt.ylabel("CDF")
    # plt.legend()
    # plt.title("CDF of All Original and Sampled Power")
    # plt.grid(True, alpha=0.3)
    # plt.savefig("cdf_power_trace.pdf")

    # print(dataset.tp_all)

    plt.figure(figsize=(4, 3))
    # plt.plot(time, dataset.traces[1]["y"], label="Original")
    plt.plot(time, power, alpha=0.5, label="Sampled")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    # plt.savefig("sampled_power_trace.pdf")

    # model_type = "gmm" if args.use_gmm else "kmeans"
    # for tp, classifier in classifiers.items():
    #     torch.save(
    #         classifier.state_dict(),
    #         f"classifier_llama3_8b_a100_tp{tp}_{model_type}.pt",
    #     )
    #     with open(f"training_summary_tp{tp}_{model_type}.txt", "w") as f:
    #         f.write(f"Final loss: {training_losses[tp][-1]:.6f}\n")
    #         f.write(f"Best loss: {min(training_losses[tp]):.6f}\n")
    #         f.write(
    #             f"Loss reduction: {training_losses[tp][0] - training_losses[tp][-1]:.6f}\n"
    #         )
