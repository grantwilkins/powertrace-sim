from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from core.dataset import PowerTraceDataset
from scipy.stats import gaussian_kde, norm


def compute_decision_bounds(gmm, data_min, data_max):
    """Compute decision boundaries between GMM components."""
    μ = gmm.means_.ravel()
    σ = np.sqrt(gmm.covariances_.ravel())
    w = gmm.weights_
    roots = []
    # solve for x: w_i * φ(x;μ_i,σ_i) = w_j * φ(x;μ_j,σ_j)
    for i, j in combinations(range(len(w)), 2):
        # log w_i - 0.5*(x-μ_i)^2/σ_i^2 - [same for j] = 0  → quadratic in x
        A = 1 / (2 * σ[j] ** 2) - 1 / (2 * σ[i] ** 2)
        B = μ[i] / (σ[i] ** 2) - μ[j] / (σ[j] ** 2)
        C = (
            (μ[j] ** 2) / (2 * σ[j] ** 2)
            - (μ[i] ** 2) / (2 * σ[i] ** 2)
            + np.log((w[i] * σ[j]) / (w[j] * σ[i]))
        )
        if abs(A) < 1e-10:  # Linear case when variances are equal
            if abs(B) > 1e-10:
                roots.append(-C / B)
        else:  # Quadratic case
            disc = B * B - 4 * A * C
            if disc > 0:
                r1 = (-B + np.sqrt(disc)) / (2 * A)
                r2 = (-B - np.sqrt(disc)) / (2 * A)
                roots.extend([r1, r2])
    return sorted([r for r in roots if data_min <= r <= data_max])


def plot_gmm_pdf(data_file: str, tp: int = 1):
    """Plot PDF of GMM components for a given tensor parallelism value."""
    dataset = PowerTraceDataset(data_file, use_gmm=True)

    # Get indices for specified TP value
    tp_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == tp]

    # Combine all power traces for this TP value
    all_power = np.concatenate(
        [dataset.traces[idx]["y"].flatten() for idx in tp_indices]
    )

    # Get GMM parameters
    gmm = dataset.state_labels[tp]

    # Create figure
    plt.figure(figsize=(10, 6))
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # PDF plot
    xs = np.linspace(all_power.min(), all_power.max(), 1000)

    # Plot individual components with better styling
    for k, (w, μ, σ) in enumerate(
        zip(gmm.weights_, gmm.means_.ravel(), np.sqrt(gmm.covariances_.ravel()))
    ):
        pdf_k = w * norm.pdf(xs, μ, σ)
        plt.plot(
            xs,
            pdf_k,
            color=palette[k],
            lw=2.5,
            label=f"Component {k+1} (μ={μ:.1f}, w={w:.2f})",
        )

    # Plot mixture with thicker line
    mixture_pdf = sum(
        w * norm.pdf(xs, μ, σ)
        for w, μ, σ in zip(
            gmm.weights_, gmm.means_.ravel(), np.sqrt(gmm.covariances_.ravel())
        )
    )
    plt.plot(xs, mixture_pdf, "k-", lw=3, label="GMM Mixture")

    # Add data histogram for reference
    plt.hist(
        all_power, bins=50, density=True, alpha=0.3, color="lightgray", label="Data"
    )

    plt.title(f"GMM Components PDF (TP={tp})")
    plt.xlabel("Power (W)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"gmm_components_pdf_tp{tp}.pdf")
    plt.close()


def plot_gmm_cdf(data_file: str, tp: int = 1):
    """Plot CDF of GMM components for a given tensor parallelism value."""
    dataset = PowerTraceDataset(data_file, use_gmm=True)

    # Get indices for specified TP value
    tp_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == tp]

    # Combine all power traces for this TP value
    all_power = np.concatenate(
        [dataset.traces[idx]["y"].flatten() for idx in tp_indices]
    )

    # Get GMM parameters
    gmm = dataset.state_labels[tp]

    # Create figure
    plt.figure(figsize=(10, 6))
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Compute decision boundaries
    bounds = compute_decision_bounds(gmm, all_power.min(), all_power.max())
    full_bounds = [all_power.min()] + bounds + [all_power.max()]

    print(f"Decision boundaries found: {len(bounds)} boundaries")
    print(f"Boundary values: {[f'{b:.1f}' for b in bounds]}")

    # Track which components we've already labeled
    labeled_components = set()

    # Color regions based on dominant component
    for i in range(len(full_bounds) - 1):
        region_start = full_bounds[i]
        region_end = full_bounds[i + 1]

        # Find dominant component at region midpoint
        mid = 0.5 * (region_start + region_end)
        component_probs = [
            w * norm.pdf(mid, μ, σ)
            for w, μ, σ in zip(
                gmm.weights_,
                gmm.means_.ravel(),
                np.sqrt(gmm.covariances_.ravel()),
            )
        ]
        dominant_comp = np.argmax(component_probs)

        # Only add label if this component hasn't been labeled yet
        label = (
            f"Component {dominant_comp + 1}"
            if dominant_comp not in labeled_components
            else ""
        )
        if dominant_comp not in labeled_components:
            labeled_components.add(dominant_comp)

        # Color the region
        plt.axvspan(
            region_start,
            region_end,
            color=palette[dominant_comp],
            alpha=0.3,
            label=label,
        )

    # Plot empirical CDF
    x_sorted = np.sort(all_power)
    cdf = np.arange(1, len(all_power) + 1) / len(all_power)
    plt.step(x_sorted, cdf, where="post", color="black", lw=2, label="Empirical CDF")

    plt.title(f"GMM Decision Regions CDF (TP={tp})")
    plt.xlabel("Power (W)")
    plt.ylabel("CDF")

    # Sort legend by component number
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_pairs = sorted(
        zip(handles, labels),
        key=lambda x: (
            int(x[1].split()[1]) if x[1].startswith("Component") else float("inf")
        ),
    )
    plt.legend(*zip(*sorted_pairs))

    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"gmm_components_cdf_tp{tp}.pdf")
    plt.close()


def plot_gmm_responsibilities(data_file: str, tp: int = 1):
    """Plot component responsibilities to see which component dominates where."""
    dataset = PowerTraceDataset(data_file, use_gmm=True)

    # Get indices for specified TP value
    tp_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == tp]

    # Combine all power traces for this TP value
    all_power = np.concatenate(
        [dataset.traces[idx]["y"].flatten() for idx in tp_indices]
    )

    # Get GMM parameters
    gmm = dataset.state_labels[tp]

    # Create figure
    plt.figure(figsize=(10, 6))
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    xs = np.linspace(all_power.min(), all_power.max(), 1000)

    # Compute mixture PDF
    mixture_pdf = sum(
        w * norm.pdf(xs, μ, σ)
        for w, μ, σ in zip(
            gmm.weights_, gmm.means_.ravel(), np.sqrt(gmm.covariances_.ravel())
        )
    )

    # Plot responsibilities
    for k, (w, μ, σ) in enumerate(
        zip(gmm.weights_, gmm.means_.ravel(), np.sqrt(gmm.covariances_.ravel()))
    ):
        responsibility = (w * norm.pdf(xs, μ, σ)) / mixture_pdf
        plt.plot(xs, responsibility, color=palette[k], lw=2, label=f"Component {k+1}")

    plt.title(f"GMM Component Responsibilities (TP={tp})")
    plt.xlabel("Power (W)")
    plt.ylabel("Responsibility")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])

    plt.tight_layout()
    plt.savefig(f"gmm_responsibilities_tp{tp}.pdf")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot GMM components from training data"
    )
    parser.add_argument(
        "--data-file", type=str, required=True, help="Path to the NPZ data file"
    )
    parser.add_argument(
        "--tp", type=int, default=1, help="Tensor parallelism value to plot"
    )
    args = parser.parse_args()

    # plot_gmm_pdf(args.data_file, args.tp)
    plot_gmm_cdf(args.data_file, args.tp)
    # plot_gmm_responsibilities(args.data_file, args.tp)
