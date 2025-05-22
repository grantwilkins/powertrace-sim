import argparse
import itertools
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
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

    # Ensure bin_ts is sorted and has no duplicates
    if len(bin_ts) > 1:
        # Check if bin_ts is monotonically increasing
        if not np.all(np.diff(bin_ts) > 0):
            # Sort bin_ts and remove duplicates
            bin_ts = np.unique(bin_ts)
            if len(bin_ts) <= 1:
                bin_ts = (
                    np.array([0, dt])
                    if len(bin_ts) == 0
                    else np.array([bin_ts[0], bin_ts[0] + dt])
                )
    else:
        # If bin_ts has 0 or 1 elements, create a simple array
        bin_ts = (
            np.array([0, dt])
            if len(bin_ts) == 0
            else np.array([bin_ts[0], bin_ts[0] + dt])
        )

    # Create edges with guaranteed monotonic increase
    edges = np.append(bin_ts, bin_ts[-1] + dt)

    # Compute histograms
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

    # 2. stack per-bin features
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

    # 3. z-score column-wise (safe small epsilon)
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True) + 1e-6
    return (x - mu) / sd


class PowerTraceDataset(Dataset):
    """
    Returns  (x_t  ,  y_t  ,  z_t)  for one whole trace.
              (T,3)   (T,1)   (T,)
    """

    def __init__(self, npz_file: str, K: int = 6):
        d = np.load(npz_file, allow_pickle=True)

        # -------- Pass 1: build list of trace dicts -------------------
        self.traces: List[Dict[str, np.ndarray]] = []
        self.tp_all: List[int] = []  # keeps TP per trace
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
                request_ts=d["request_timestamps"][i][req_mask],  # not used here
                input_tokens=d["input_tokens"][i][req_mask],  # not used here
                output_tokens=d["output_tokens"][i][req_mask],  # not used here
            )

            trace["x"] = make_schedule_matrix(trace)
            trace["y"] = trace["power"][:, None]  # (T,1)
            self.traces.append(trace)
            self.tp_all.append(int(tp_array[i]))

        self.state_labels: Dict[int, np.ndarray] = {}  # maps TP→ (K,)
        unique_tp = np.unique(self.tp_all)
        for tp in unique_tp:
            y_concat = np.concatenate(
                [tr["y"] for tr, tp_i in zip(self.traces, self.tp_all) if tp_i == tp]
            ).reshape(-1, 1)

            # km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(y_concat)
            # self.state_labels[tp] = km  # store the fitted model
            gmm = GaussianMixture(K, covariance_type="diag", n_init=10, random_state=0)
            gmm.fit(y_concat)
            self.state_labels[tp] = gmm  # store the fitted model
        for tr, tp in zip(self.traces, self.tp_all):
            # km = self.state_labels[tp]
            # tr["z"] = km.predict(tr["y"]).astype("int64")  # (T,)
            gmm = self.state_labels[tp]
            tr["z"] = gmm.predict(tr["y"]).astype("int64")

    # -----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.traces)

    # -----------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tr = self.traces[idx]
        return (
            torch.from_numpy(tr["x"]),  # schedule features (T,3)
            torch.from_numpy(tr["y"]),  # power            (T,1)
            torch.from_numpy(tr["z"]),  # discrete state   (T,)
        )


class GRUClassifier(nn.Module):
    def __init__(self, Dx, K, H=128):
        super().__init__()
        self.gru = nn.GRU(Dx, H, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * H, K)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.fc(h)


def train_classifiers(dataset, hidden_size=128, device=None):
    """Train separate GRU classifiers for each tensor parallelism value."""
    classifiers = {}
    training_losses = {}  # Store losses for each TP

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # for tp in sorted(set(dataset.tp_all)):
    for tp in [1]:
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
        classifier = GRUClassifier(Dx, K, H=hidden_size).to(device)

        optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()

        # Initialize loss tracking for this TP
        epoch_losses = []
        num_epochs = 10
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            batch_losses = []
            progress_bar = tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for x, y, z in progress_bar:
                # Move data to device
                x = x.to(device)
                z = z.to(device)

                optimizer.zero_grad()
                logits = classifier(x)
                loss = criterion(logits.view(-1, K), z.view(-1))
                loss.backward()
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

        # Save the loss history to a file
        np.save(f"training_losses_tp{tp}.npy", np.array(epoch_losses))

        # Plot the training loss
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_losses)
        plt.title(f"Training Loss for TP={tp}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(f"training_loss_tp{tp}.pdf")
        plt.close()

    return classifiers, training_losses


# ------------------------------------------------------------
# 1.  fit β_k  and  ρ_k  once from the dataset  --------------
# ------------------------------------------------------------
def calibrate_amplitude(dataset, K, Dx):
    """
    Returns
        mu      (K,)         unconditional mean  (your gmm centroids)
        beta    (K,Dx)       small linear term per state
        sigma   (K,)         innovation std
        rho     (K,)         AR(1) coefficient  (0.0 → white, 0.95 → very smooth)
    """
    mu = np.zeros(K)
    beta = np.zeros((K, Dx))
    var_innov = np.zeros(K)
    rho = np.zeros(K)

    # gather per-state design matrices and residuals
    buckets = [[] for _ in range(K)]
    for tr in dataset.traces:
        x, y, z = tr["x"], tr["y"].squeeze(), tr["z"]
        for k in range(K):
            idx = z == k
            if idx.sum() == 0:
                continue
            buckets[k].append((x[idx], y[idx]))

    for k in range(K):
        if not buckets[k]:
            mu[k] = 0
            rho[k] = 0.0
            var_innov[k] = 1.0
            continue

        X = np.vstack([b[0] for b in buckets[k]])  # (Nk,Dx)
        Y = np.hstack([b[1] for b in buckets[k]])  # (Nk,)

        mu[k] = Y.mean()
        # simple ridge to avoid singularity
        XtX = X.T @ X + 1e-3 * np.eye(Dx)
        beta[k] = np.linalg.solve(XtX, X.T @ (Y - mu[k]))

        # AR(1) fit on residuals
        res = (Y - mu[k] - X @ beta[k]).astype("float64")
        if len(res) < 2:
            rho[k] = 0.0
            var_innov[k] = np.var(res)
        else:
            rho[k] = np.corrcoef(res[:-1], res[1:])[0, 1]
            rho[k] = np.clip(rho[k], 0.0, 0.95)  # keep stable
            var_innov[k] = np.var(res[1:] - rho[k] * res[:-1]) + 1e-6

    sigma = np.sqrt(var_innov)
    return mu, beta, sigma, rho


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
    schedule_x:  (T,Dx)  – already z-scored feature matrix on desired dt grid
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
    args = parser.parse_args()

    # Set device
    device = None
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PowerTraceDataset(args.data_file, 6)
    # classifiers, training_losses = train_classifiers(dataset, device=device)
    classifiers = {
        1: load_classifier(
            "classifier_models/classifier_llama3_8b_a100_tp4.pt", device=device
        )
    }

    K = 6
    mu = {}
    sigma = {}

    mu, beta, sigma, rho = calibrate_amplitude(dataset, K, 6)
    # print("Mu:", mu)
    # print("Beta:", beta)
    # print("Sigma:", sigma)
    # print("Rho:", rho)

    # for tp in dataset.state_labels:
    #     km = dataset.state_labels[tp]
    #     mu[tp] = km.means_.flatten()
    #     sigma[tp] = km.covariances_.flatten()
    #     sigma[tp] = np.sqrt(sigma[tp])

    print("Mu:", mu)
    print("Sigma:", sigma)
    # Get cluster centers for mean values
    # for k in range(K):
    #     pts = dataset.state_labels[4].cluster_centers_[k]
    #     mu[k] = pts[0]

    # for tp in dataset.state_labels:
    #     km = dataset.state_labels[tp]
    #     for k in range(K):
    #         cluster_points = []
    #         for tr, tp_i in zip(dataset.traces, dataset.tp_all):
    #             if tp_i == tp:
    #                 k_indices = np.where(tr["z"] == k)[0]
    #                 if len(k_indices) > 0:
    #                     cluster_points.append(tr["y"][k_indices])

    #         if cluster_points:
    #             cluster_points = np.concatenate(cluster_points)
    #             centroid = km.cluster_centers_[k]
    #             distances = np.sqrt(np.sum((cluster_points - centroid) ** 2, axis=1))
    #             sigma[k] = np.mean(distances)
    #         else:
    #             sigma[k] = 1.0

    time, power = sample_power(
        classifiers[1],
        mu[1],
        sigma * 0.25,
        dataset.traces[1]["x"],
        dt=0.25,
    )
    # # Create a CDF (Cumulative Distribution Function) of the sampled and true power traces
    # fig, ax = plt.subplots(figsize=(6, 6))

    # # Sort the power values and calculate the cumulative probabilities
    # true_power_sorted = np.sort(dataset.traces[0]["y"].flatten())
    # sampled_power_sorted = np.sort(power)
    # print("True power sorted:", true_power_sorted)

    # # Calculate the cumulative probabilities (0 to 1)
    # true_cdf = np.arange(1, len(true_power_sorted) + 1) / len(true_power_sorted)
    # sampled_cdf = np.arange(1, len(sampled_power_sorted) + 1) / len(
    #     sampled_power_sorted
    # )

    # # Plot the CDFs
    # ax.plot(true_power_sorted, true_cdf, label="Original", linewidth=2)
    # ax.plot(sampled_power_sorted, sampled_cdf, label="Sampled", linewidth=2, alpha=0.7)

    # ax.set_xlabel("Power (W)")
    # ax.set_ylabel("Cumulative Probability")
    # ax.set_title("CDF of TP4 Llama-3-8B")
    # ax.grid(True, linestyle="--", alpha=0.7)
    # ax.legend()

    # plt.tight_layout()
    # plt.savefig("power_trace_cdf_tp4_llama3_8b.pdf")

    # Create histograms based on tensor parallelism using Seaborn
    import seaborn as sns

    # Prepare data for Seaborn
    all_power_data = []
    for tp in sorted(set(dataset.tp_all)):
        for j, tp_j in enumerate(dataset.tp_all):
            if tp_j == tp:
                power_values = dataset.traces[j]["y"].flatten()
                all_power_data.extend([(p, f"TP={tp}") for p in power_values])

    # Convert to DataFrame for easier plotting with Seaborn
    import pandas as pd

    power_df = pd.DataFrame(all_power_data, columns=["Power", "TP"])
    # plt.figure(figsize=(10, 6))
    # sns.histplot(
    #     data=power_df,
    #     x="Power",
    #     hue="TP",
    #     bins=500,
    #     alpha=0.5,
    #     multiple="layer",
    #     stat="density",
    # )
    # plt.xlabel("Power (W)")
    # plt.ylabel("Frequency")
    # plt.title("Power Histgram by Tensor Parallelism")
    # plt.grid(True, linestyle="--", alpha=0.7)
    # plt.tight_layout()
    # plt.savefig("power_histograms_by_tp.pdf")

    # plt.figure(figsize=(10, 6))
    # sns.kdeplot(
    #     data=power_df,
    #     x="Power",
    #     hue="TP",
    #     common_norm=False,
    #     fill=True,
    #     alpha=0.3,
    # )
    # plt.xlabel("Power (W)")
    # plt.ylabel("Frequency")
    # plt.title("Power Distribution by Tensor Parallelism")
    # plt.grid(True, linestyle="--", alpha=0.7)
    # plt.tight_layout()
    # plt.savefig("power_kde_by_tp.pdf")

    # Create plots showing K-means cluster distributions by tensor parallelism
    # plt.figure(figsize=(12, 8))

    # # Get unique tensor parallelism values and K (number of clusters)
    # unique_tps = sorted(set(dataset.tp_all))
    # K = len(dataset.state_labels[unique_tps[0]].means_)

    # # Prepare data for plotting
    # cluster_data = []

    # for tp in unique_tps:
    #     # Get GMM model for this TP
    #     gmm = dataset.state_labels[tp]

    #     for j, tp_j in enumerate(dataset.tp_all):
    #         if tp_j == tp:
    #             trace_clusters = dataset.traces[j]["z"]
    #             power_values = dataset.traces[j]["y"].flatten()
    #             for cluster_id, power in zip(trace_clusters, power_values):
    #                 cluster_data.append(
    #                     {
    #                         "Power": power,
    #                         "Cluster": f"Cluster {cluster_id}",
    #                         "TP": f"TP={tp}",
    #                     }
    #                 )

    # cluster_df = pd.DataFrame(cluster_data)
    # for tp in unique_tps:
    #     tp_data = cluster_df[cluster_df["TP"] == f"TP={tp}"]
    #     plt.figure(figsize=(10, 6))

    #     # Plot actual data distribution by cluster
    #     sns.kdeplot(
    #         data=tp_data,
    #         x="Power",
    #         hue="Cluster",
    #         fill=True,
    #         alpha=0.4,
    #         palette="viridis",
    #         common_norm=False,
    #     )

    #     plt.xlabel("Power (W)")
    #     plt.ylabel("Density")
    #     plt.title(f"Gaussian Mixture Model Power Distributions for TP={tp}")
    #     plt.grid(True, linestyle="--", alpha=0.7)
    #     plt.legend(title="Components")
    #     plt.tight_layout()
    #     plt.savefig(f"gmm_clusters_tp_{tp}.pdf")
    #     plt.close()

    plt.plot(time, dataset.traces[1]["y"], label="Original")
    plt.plot(time, power, alpha=0.5, label="Sampled")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.show()
    plt.savefig("sampled_power_trace.pdf")
