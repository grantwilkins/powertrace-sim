import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import math


class PowerTraceGP:
    """
    A Gaussian Process approach for power trace modeling with less memory usage.

    Main changes:
    1. Only one RBF kernel for time dimension * RBF kernel for Poisson rate.
    2. Subsampling of training points to avoid huge N x N kernel matrices.
    """

    def __init__(self, device="cpu", max_points_per_trace=2000):
        self.device = device
        self.models = {}
        self.likelihoods = {}
        self.data = {}
        # Control how many points we keep per trace to avoid big kernel matrices
        self.max_points_per_trace = max_points_per_trace

    def prepare_data(
        self,
        power_traces,
        tensor_parallelism,
        poisson_rate,
        model_size,
        input_tokens=None,
        output_tokens=None,
        scale_power=True,
    ):
        """
        Prepare data for GP modeling, grouping by (tp, ms) only.
        """
        # Convert to numpy if needed
        if isinstance(power_traces, torch.Tensor):
            power_traces = power_traces.cpu().numpy()

        N = len(power_traces)
        data_by_config = {}

        for i in range(N):
            tp = tensor_parallelism[i]
            pr = poisson_rate[i]
            ms = model_size[i] if model_size is not None else 0
            config_key = f"tp{tp}_ms{ms}"

            # Normalize power by tensor parallelism if requested
            if scale_power:
                power = power_traces[i] / tp
            else:
                power = power_traces[i]

            if config_key not in data_by_config:
                data_by_config[config_key] = {
                    "config": (tp, ms),
                    "traces": [],
                    "poisson_rates": [],
                    "input_tokens": [],
                    "output_tokens": [],
                }

            data_by_config[config_key]["traces"].append(power)
            data_by_config[config_key]["poisson_rates"].append(pr)

            if input_tokens is not None:
                data_by_config[config_key]["input_tokens"].append(input_tokens[i])

            if output_tokens is not None:
                data_by_config[config_key]["output_tokens"].append(output_tokens[i])

        self.data = data_by_config
        print(f"Processed {N} traces into {len(data_by_config)} unique configs.")
        return data_by_config

    def train_gp_for_config(
        self,
        config_key,
        train_percentage=0.8,
        training_iterations=100,
        lr=0.1,
    ):
        """
        Train a GP model for one config, with Poisson rate as a feature.
        Downsamples each trace to avoid large memory usage.
        """
        dtype = torch.float64
        data = self.data[config_key]
        traces = data["traces"]
        poisson_rates = data["poisson_rates"]

        n_traces = len(traces)
        n_train = max(1, int(n_traces * train_percentage))

        # Separate train/test splits
        if n_traces > 1:
            # Use different traces for train/test
            indices = np.arange(n_traces)
            np.random.shuffle(indices)
            train_indices = indices[:n_train]
            test_indices = indices[n_train:]
        else:
            # Single trace, we'll do time-based split
            train_indices = [0]
            test_indices = []

        # We'll gather all train points in these lists
        train_x_list = []
        train_y_list = []

        # Subsampling logic to avoid huge data
        def subsample_trace(x, y, max_points):
            # If the trace is too large, choose a random subset
            if len(x) > max_points:
                idx = np.random.choice(len(x), size=max_points, replace=False)
                idx = np.sort(idx)
                return x[idx], y[idx]
            return x, y

        # Build train data
        for idx in train_indices:
            trace = traces[idx]
            pr = poisson_rates[idx]

            # Build time indices
            times = torch.linspace(0, len(trace) - 1, len(trace), dtype=dtype)
            pr_tensor = torch.full_like(times, pr)

            trace_x = torch.stack([times, pr_tensor], dim=1)
            trace_y = torch.tensor(trace, dtype=dtype)

            # Subsample to reduce memory usage
            trace_x, trace_y = subsample_trace(
                trace_x, trace_y, self.max_points_per_trace
            )

            train_x_list.append(trace_x)
            train_y_list.append(trace_y)

        # Concatenate
        train_x = torch.cat(train_x_list, dim=0).to(self.device)
        train_y = torch.cat(train_y_list, dim=0).to(self.device)

        # Build test data from the first test trace if available
        if len(test_indices) > 0:
            idx = test_indices[0]
            test_trace = traces[idx]
            pr = poisson_rates[idx]
            times = torch.linspace(0, len(test_trace) - 1, len(test_trace), dtype=dtype)
            pr_tensor = torch.full_like(times, pr)
            test_x = torch.stack([times, pr_tensor], dim=1).to(self.device)
            test_y = torch.tensor(test_trace, dtype=dtype).to(self.device)
        else:
            # If no separate test trace, just hold out last chunk of train data
            cutoff = int(len(train_x) * 0.2)  # 20% for test
            test_x = train_x[-cutoff:]
            test_y = train_y[-cutoff:]
            train_x = train_x[:-cutoff]
            train_y = train_y[:-cutoff]

        # --- Define a simpler kernel to reduce memory usage ---
        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                # We just do RBF for time * RBF for Poisson rate
                self.time_kernel = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(active_dims=[0])
                )
                self.pr_kernel = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(active_dims=[1])
                )
                self.covar_module = self.time_kernel * self.pr_kernel

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
            dtype=dtype, device=self.device
        )
        model = ExactGPModel(train_x, train_y, likelihood).to(
            dtype=dtype, device=self.device
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Training loop
        model.train()
        likelihood.train()
        pbar = tqdm(range(training_iterations), desc=f"Training {config_key}")
        for i in pbar:
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})
            # Reduce LR occasionally
            if (i + 1) % 30 == 0:
                for g in optimizer.param_groups:
                    g["lr"] *= 0.8

        # Evaluate
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = model(test_x)
            rmse = torch.sqrt(torch.mean((preds.mean - test_y) ** 2)).item()
            print(f"[{config_key}] Test RMSE: {rmse:.4f}")

        self.models[config_key] = model
        self.likelihoods[config_key] = likelihood

    def train_all_configs(self, max_configs=None, training_iterations=100):
        """
        Train GPs for all configs, up to max_configs.
        """
        config_keys = list(self.data.keys())
        if max_configs is not None:
            config_keys = config_keys[:max_configs]

        for key in config_keys:
            self.train_gp_for_config(key, training_iterations=training_iterations)

    def predict(
        self,
        config_key,
        seed_sequence,
        poisson_rate,
        steps=50,
        sample=True,
        n_samples=1,
    ):
        """
        Predict future steps, given a seed sequence, with a chosen Poisson rate.
        """
        if config_key not in self.models:
            raise ValueError(f"No trained model for {config_key}")

        model = self.models[config_key]
        likelihood = self.likelihoods[config_key]
        model.eval()
        likelihood.eval()

        dtype = torch.float64
        seed_length = len(seed_sequence)
        predictions = np.zeros((n_samples, steps))

        # We'll simply predict at t = seed_length, seed_length+1, ...
        # ignoring the seed beyond the final point in time
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for s in range(n_samples):
                for t in range(steps):
                    time_val = seed_length + t
                    x = torch.tensor(
                        [[time_val, poisson_rate]], dtype=dtype, device=self.device
                    )
                    dist = likelihood(model(x))
                    if sample:
                        predictions[s, t] = dist.sample().cpu().item()
                    else:
                        predictions[s, t] = dist.mean.cpu().item()
        return predictions


def run_small_memory_example(
    data_path="./processed_data/power_trace_data.npz", output_dir="gp_results"
):
    """
    Example driver that uses PowerTraceGP with simpler kernel and downsampling.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load NPZ data (example)
    data = np.load(data_path)
    power_traces = data["power_traces"]
    tensor_parallelism = data["tensor_parallelism"]
    poisson_rate = data["poisson_rate"]
    model_size = data.get("model_size", np.zeros_like(tensor_parallelism))
    input_tokens = data.get("input_tokens", None)
    output_tokens = data.get("output_tokens", None)

    print("Data loaded:")
    print(f"  Number of traces: {len(power_traces)}")
    print(f"  Example trace shape: {power_traces[0].shape}")

    # Create GP object with a limit on points per trace
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gp_model = PowerTraceGP(device=device, max_points_per_trace=2500)

    # Prepare data
    gp_model.prepare_data(
        power_traces=power_traces,
        tensor_parallelism=tensor_parallelism,
        poisson_rate=poisson_rate,
        model_size=model_size,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        scale_power=True,
    )

    # Train only the first 2 configurations for demonstration
    gp_model.train_all_configs(max_configs=2, training_iterations=100)

    # Example prediction
    config_key = list(gp_model.data.keys())[0]  # pick first config
    print(f"Predicting on config {config_key}")

    # Use half of the first trace as "seed"
    seed_sequence = power_traces[0][:100]  # just 100 points as seed
    pr = poisson_rate[0]

    preds = gp_model.predict(
        config_key=config_key,
        seed_sequence=seed_sequence,
        poisson_rate=pr,
        steps=30,
        sample=True,
        n_samples=3,
    )

    # Visualize
    plt.figure()
    plt.plot(range(len(seed_sequence)), seed_sequence, label="Seed Power")
    for i in range(preds.shape[0]):
        start = len(seed_sequence)
        plt.plot(
            range(start, start + preds.shape[1]),
            preds[i, :],
            alpha=0.5,
            label=f"Prediction {i+1}",
        )
    plt.axvline(x=len(seed_sequence) - 1, color="k", linestyle="--")
    plt.title(f"Predictions for {config_key} (Poisson Rate={pr:.2f})")
    plt.xlabel("Time step")
    plt.ylabel("Power (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{config_key}_prediction_example.png"))
    plt.show()

    # Save your models
    torch.save(
        {
            key: {
                "model_state": gp_model.models[key].state_dict(),
                "likelihood_state": gp_model.likelihoods[key].state_dict(),
            }
            for key in gp_model.models
        },
        os.path.join(output_dir, "models_downsampled.pt"),
    )

    print("Done training and prediction with lower memory usage.")


if __name__ == "__main__":
    run_small_memory_example()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gp_model = PowerTraceGP(device=device, max_points_per_trace=2000)

    # # 2. Load the checkpoint dictionary from disk.
    # #    This dictionary maps config_key -> {"model_state", "likelihood_state"}.
    # checkpoint = torch.load("gp_results/models_downsampled.pt", map_location=device)

    # # 3. For each config_key in the checkpoint, re-create the same model architecture
    # #    and load the saved state dicts.
    # for config_key, state_dicts in checkpoint.items():
    #     model_state = state_dicts["model_state"]
    #     likelihood_state = state_dicts["likelihood_state"]

    #     # --- Re-create the same GP architecture used in train_gp_for_config ---
    #     class ExactGPModel(gpytorch.models.ExactGP):
    #         def __init__(self, train_x, train_y, likelihood):
    #             super().__init__(train_x, train_y, likelihood)
    #             self.mean_module = gpytorch.means.ConstantMean()

    #             self.time_kernel = gpytorch.kernels.ScaleKernel(
    #                 gpytorch.kernels.RBFKernel(active_dims=[0])
    #             )
    #             self.pr_kernel = gpytorch.kernels.ScaleKernel(
    #                 gpytorch.kernels.RBFKernel(active_dims=[1])
    #             )
    #             self.covar_module = self.time_kernel * self.pr_kernel

    #         def forward(self, x):
    #             mean_x = self.mean_module(x)
    #             covar_x = self.covar_module(x)
    #             return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    #     likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
    #         device=device, dtype=torch.float64
    #     )

    #     # We create a "dummy" training set just so the ExactGPModel constructor
    #     # has something to initialize with.  The real parameters will come from state_dict.
    #     dummy_x = torch.zeros(1, 2, dtype=torch.float64, device=device)
    #     dummy_y = torch.zeros(1, dtype=torch.float64, device=device)

    #     model_obj = ExactGPModel(dummy_x, dummy_y, likelihood).to(
    #         device=device, dtype=torch.float64
    #     )

    #     # Load the weights
    #     model_obj.load_state_dict(model_state)
    #     likelihood.load_state_dict(likelihood_state)

    #     # Put in eval mode
    #     model_obj.eval()
    #     likelihood.eval()

    #     # Store in gp_model so we can call gp_model.predict(...)
    #     gp_model.models[config_key] = model_obj
    #     gp_model.likelihoods[config_key] = likelihood

    # # 4. Now do inference/prediction using the existing PowerTraceGP code.
    # #    Suppose we want to predict for config_key="tp1_ms0"
    # #    with a seed sequence of 5 points and poisson_rate=0.5
    # config_key = "tp2_ms0"
    # poisson_rate = 2.0

    # predictions = gp_model.predict(
    #     config_key=config_key,
    #     poisson_rate=poisson_rate,
    #     seed_sequence=np.array([100, 200, 250, 240, 250]),
    #     steps=10,
    #     sample=True,
    #     n_samples=1,
    # )

    # print(f"Predictions for {config_key}:", predictions)
