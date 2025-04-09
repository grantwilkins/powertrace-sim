import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import math


class PowerTraceGP:
    """
    A Gaussian Process approach for power trace modeling.

    This class trains GP models based on tensor parallelism and model size only,
    with poisson arrival rate as a parameter at inference time.
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.models = {}  # Dictionary to store trained GP models by config
        self.likelihoods = {}  # Dictionary to store likelihoods by config

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
        Prepare data for GP modeling.

        Args:
            power_traces: [N, T] array of power traces
            tensor_parallelism: [N] array of tensor parallelism values
            poisson_rate: [N] array of poisson rates
            model_size: [N] array of model sizes
            input_tokens: [N, T] array of input tokens (optional)
            output_tokens: [N, T] array of output tokens (optional)
            scale_power: Whether to normalize power by tensor parallelism

        Returns:
            Processed data as a dictionary
        """
        # Convert to numpy if needed
        if isinstance(power_traces, torch.Tensor):
            power_traces = power_traces.cpu().numpy()

        # Number of traces
        N = len(power_traces)

        # Create a dictionary to store data by configuration (tp, ms only)
        data_by_config = {}

        # Process each trace
        for i in range(N):
            # Get configuration - now using only TP and model size
            tp = tensor_parallelism[i]
            pr = poisson_rate[i]
            ms = model_size[i]
            config_key = f"tp{tp}_ms{ms}"  # Removed poisson rate from config key

            # Normalize power by tensor parallelism if requested
            if scale_power:
                power = power_traces[i] / tp
            else:
                power = power_traces[i]

            # Create data entry
            if config_key not in data_by_config:
                data_by_config[config_key] = {
                    "config": (tp, ms),  # Config now only has tp and ms
                    "traces": [],
                    "poisson_rates": [],  # Store poisson rates separately
                    "input_tokens": [],
                    "output_tokens": [],
                }

            # Add trace and its associated poisson rate
            data_by_config[config_key]["traces"].append(power)
            data_by_config[config_key]["poisson_rates"].append(pr)

            # Add tokens if available
            if input_tokens is not None:
                data_by_config[config_key]["input_tokens"].append(input_tokens[i])

            if output_tokens is not None:
                data_by_config[config_key]["output_tokens"].append(output_tokens[i])

        # Return processed data
        self.data = data_by_config
        print(f"Processed {N} traces into {len(data_by_config)} unique configurations")
        return data_by_config

    def train_gp_for_config(
        self, config_key, train_percentage=0.8, training_iterations=100
    ):
        """
        Train a GP model for a specific configuration with poisson rate as a feature.

        Args:
            config_key: Configuration key (tp_ms)
            train_percentage: Percentage of data to use for training
            training_iterations: Number of training iterations

        Returns:
            Trained model and likelihood
        """
        # Set a consistent dtype (double precision is more stable for GPs)
        dtype = torch.float64

        # Get data for this configuration
        data = self.data[config_key]
        traces = data["traces"]
        poisson_rates = data["poisson_rates"]
        config = data["config"]  # Now (tp, ms)

        # Split traces into train and test
        n_traces = len(traces)
        n_train = max(1, int(n_traces * train_percentage))

        # Prepare training data (now including poisson rate as a feature)
        train_x_list = []
        train_y_list = []
        test_x_list = []
        test_y_list = []

        # Handle different cases based on number of traces
        if n_traces == 1:
            # Single trace, split it into train/test
            trace = traces[0]
            pr = poisson_rates[0]
            train_len = int(len(trace) * train_percentage)

            # Training data: time and poisson rate as features
            times = torch.linspace(
                0, train_len - 1, train_len, dtype=dtype, device=self.device
            )
            pr_tensor = torch.full_like(times, pr)
            train_x = torch.stack(
                [times, pr_tensor], dim=1
            )  # 2D inputs: [time, poisson_rate]
            train_y = torch.tensor(trace[:train_len], dtype=dtype, device=self.device)

            # Test data
            test_times = torch.linspace(
                train_len,
                len(trace) - 1,
                len(trace) - train_len,
                dtype=dtype,
                device=self.device,
            )
            test_pr_tensor = torch.full_like(test_times, pr)
            test_x = torch.stack([test_times, test_pr_tensor], dim=1)
            test_y = torch.tensor(trace[train_len:], dtype=dtype, device=self.device)
        else:
            # Multiple traces available - use different traces for train/test
            train_indices = np.random.choice(n_traces, n_train, replace=False)
            test_indices = np.array(
                [i for i in range(n_traces) if i not in train_indices]
            )

            # Collect training data from all training traces
            for idx in train_indices:
                trace = traces[idx]
                pr = poisson_rates[idx]
                times = torch.linspace(
                    0, len(trace) - 1, len(trace), dtype=dtype, device=self.device
                )
                pr_tensor = torch.full_like(times, pr)
                trace_x = torch.stack([times, pr_tensor], dim=1)
                trace_y = torch.tensor(trace, dtype=dtype, device=self.device)

                train_x_list.append(trace_x)
                train_y_list.append(trace_y)

            # Concatenate all training data
            if train_x_list:
                train_x = torch.cat(train_x_list, dim=0)
                train_y = torch.cat(train_y_list, dim=0)
            else:
                # Fallback - shouldn't happen with proper train_indices
                train_x = torch.zeros((0, 2), dtype=dtype, device=self.device)
                train_y = torch.zeros(0, dtype=dtype, device=self.device)

            # Collect test data if available
            if len(test_indices) > 0:
                for idx in test_indices:
                    test_trace = traces[idx]
                    test_pr = poisson_rates[idx]
                    test_times = torch.linspace(
                        0,
                        len(test_trace) - 1,
                        len(test_trace),
                        dtype=dtype,
                        device=self.device,
                    )
                    test_pr_tensor = torch.full_like(test_times, test_pr)
                    test_trace_x = torch.stack([test_times, test_pr_tensor], dim=1)
                    test_trace_y = torch.tensor(
                        test_trace, dtype=dtype, device=self.device
                    )

                    test_x_list.append(test_trace_x)
                    test_y_list.append(test_trace_y)

                # Use first test trace for evaluation
                test_x = test_x_list[0]
                test_y = test_y_list[0]
            else:
                # If no test traces, use a portion of the training data
                test_len = min(300, len(train_y) // 3)
                test_x = train_x[-test_len:]
                test_y = train_y[-test_len:]

        # Create the modified GP model for 2D inputs
        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()

                # Create separate kernels for each input dimension
                # Dimension 0: Time kernel (RBF + Periodic + Matern)
                self.time_kernel = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(active_dims=0)
                    + gpytorch.kernels.PeriodicKernel(active_dims=0)
                    + gpytorch.kernels.MaternKernel(nu=1.5, active_dims=0)
                )

                # Dimension 1: Poisson rate kernel (RBF)
                self.poisson_kernel = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(active_dims=1)
                )

                # Product kernel combines both dimensions
                self.covar_module = self.time_kernel * self.poisson_kernel

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        # Initialize likelihood and model
        likelihood = (
            gpytorch.likelihoods.GaussianLikelihood().to(dtype=dtype).to(self.device)
        )
        model = (
            ExactGPModel(train_x, train_y, likelihood).to(dtype=dtype).to(self.device)
        )

        # Use the Adam optimizer
        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters()},
            ],
            lr=0.1,
        )

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Training loop
        model.train()
        likelihood.train()

        # Progress bar
        pbar = tqdm(range(training_iterations), desc=f"Training GP for {config_key}")
        train_losses = []

        for i in pbar:
            # Zero gradients
            optimizer.zero_grad()

            # Output from model
            output = model(train_x)

            # Calculate loss and backprop
            loss = -mll(output, train_y)
            loss.backward()

            # Update progress
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})
            train_losses.append(loss.item())

            # Take a step
            optimizer.step()

            # Reduce learning rate
            if (i + 1) % 30 == 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] * 0.8

        # Switch to eval mode
        model.eval()
        likelihood.eval()

        # Store the trained model
        self.models[config_key] = model
        self.likelihoods[config_key] = likelihood

        # Evaluate on test data
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_x))
            test_rmse = torch.sqrt(
                torch.mean((observed_pred.mean - test_y) ** 2)
            ).item()
            print(f"Test RMSE for {config_key}: {test_rmse:.4f}")

        # Return model and likelihood
        return model, likelihood, train_losses

    def train_all_configs(self, max_configs=5, training_iterations=100):
        """
        Train GP models for all configurations (or a subset).

        Args:
            max_configs: Maximum number of configurations to train
            training_iterations: Number of training iterations per config

        Returns:
            Dictionary of trained models
        """
        # Get all configuration keys
        config_keys = list(self.data.keys())

        # Limit to max_configs
        if max_configs is not None and max_configs < len(config_keys):
            config_keys = config_keys[:max_configs]

        # Train a model for each configuration
        for key in config_keys:
            print(f"\nTraining GP for configuration: {key}")
            self.train_gp_for_config(key, training_iterations=training_iterations)

        return self.models

    def predict(
        self,
        config_key,
        seed_sequence,
        poisson_rate,
        steps=300,
        sample=True,
        n_samples=5,
    ):
        """
        Predict future power values with specified poisson rate.

        Args:
            config_key: Configuration key (tp_ms)
            seed_sequence: Seed power sequence
            poisson_rate: Poisson arrival rate to use for prediction
            steps: Number of steps to predict
            sample: Whether to sample or use mean
            n_samples: Number of sample trajectories

        Returns:
            Predicted power values
        """
        # Set consistent dtype
        dtype = torch.float64

        # Ensure seed_sequence is a numpy array
        if isinstance(seed_sequence, torch.Tensor):
            seed_sequence = seed_sequence.cpu().numpy()

        # Check if model exists
        if config_key not in self.models:
            raise ValueError(f"No trained model for configuration {config_key}")

        # Get model and likelihood
        model = self.models[config_key]
        likelihood = self.likelihoods[config_key]

        # Set to evaluation mode
        model.eval()
        likelihood.eval()

        # Create predictions array
        predictions = np.zeros((n_samples, steps))

        # Starting point is the last value in seed sequence
        last_idx = len(seed_sequence) - 1

        # For each sample
        for s in range(n_samples):
            # Current sequence starts with the seed
            current_seq = seed_sequence.copy()

            # Predict step by step
            for t in range(steps):
                # Next time point (with poisson rate)
                next_time = last_idx + t + 1
                next_x = torch.tensor(
                    [[next_time, poisson_rate]], dtype=dtype, device=self.device
                )

                # Predict
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    # Get distribution
                    pred_dist = likelihood(model(next_x))

                    # Sample or mean
                    if sample:
                        # Sample from the predicted distribution
                        next_y = pred_dist.sample().cpu().numpy()[0]
                    else:
                        # Use mean prediction
                        next_y = pred_dist.mean.cpu().numpy()[0]

                # Store prediction
                predictions[s, t] = next_y

                # Update sequence
                current_seq = np.append(current_seq[1:], next_y)

        return predictions

    def visualize_prediction(
        self,
        config_key,
        seed_sequence,
        poisson_rate,
        steps=300,
        n_samples=5,
        actual_future=None,
        input_tokens=None,
        output_tokens=None,
        save_path=None,
    ):
        """
        Visualize a prediction.

        Args:
            config_key: Configuration key (tp_ms)
            seed_sequence: Seed power sequence
            poisson_rate: Poisson arrival rate to use for prediction
            steps: Number of steps to predict
            n_samples: Number of sample trajectories
            actual_future: Actual future values (optional)
            input_tokens: Input token sequence (optional)
            output_tokens: Output token sequence (optional)
            save_path: Path to save the visualization
        """
        # Make predictions with the specified poisson rate
        predictions = self.predict(
            config_key,
            seed_sequence,
            poisson_rate,
            steps,
            sample=True,
            n_samples=n_samples,
        )

        # Get configuration (now just tp and ms)
        tp, ms = self.data[config_key]["config"]

        # Set up the figure
        plt.figure(figsize=(14, 8))

        # Plot the seed sequence
        plt.plot(
            range(len(seed_sequence)),
            seed_sequence,
            "b-",
            linewidth=2,
            label="Seed Sequence",
        )

        # Plot the actual future if available
        if actual_future is not None:
            future_start = len(seed_sequence)
            future_end = future_start + len(actual_future)
            plt.plot(
                range(future_start, future_end),
                actual_future,
                "g-",
                linewidth=2,
                label="Actual Future",
            )

        # Plot the predicted trajectories
        future_start = len(seed_sequence)
        for i in range(n_samples):
            plt.plot(
                range(future_start, future_start + steps),
                predictions[i],
                "r-",
                alpha=0.3,
            )

        # Add a legend entry for predictions
        plt.plot([], [], "r-", alpha=0.7, label=f"{n_samples} Sampled Predictions")

        # Plot tokens if available
        if input_tokens is not None:
            token_len = min(len(input_tokens), future_start + steps)
            plt.plot(
                range(token_len),
                input_tokens[:token_len],
                "c-",
                alpha=0.5,
                label="Input Tokens",
            )

        if output_tokens is not None:
            token_len = min(len(output_tokens), future_start + steps)
            plt.plot(
                range(token_len),
                output_tokens[:token_len],
                "m-",
                alpha=0.5,
                label="Output Tokens",
            )

        # Add vertical line
        plt.axvline(x=future_start - 1, color="k", linestyle="--")

        # Add title and labels with poisson rate included
        plt.title(
            f"Power Prediction with GP (TP={tp}, Model Size={ms}, Poisson Rate={poisson_rate:.1f})",
            fontsize=14,
        )
        plt.xlabel("Time Step", fontsize=12)
        plt.ylabel("Normalized Power", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")

        plt.show()


def run_gp_pipeline(data_path, output_dir="gp_results"):
    """
    Run the GP power prediction pipeline.

    Args:
        data_path: Path to data file
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {data_path}")
    data = np.load(data_path)

    power_traces = data["power_traces"]
    tensor_parallelism = data["tensor_parallelism"]
    poisson_rate = data["poisson_rate"]
    model_size = data.get("model_size", np.zeros_like(tensor_parallelism))
    input_tokens = data.get("input_tokens", None)
    output_tokens = data.get("output_tokens", None)

    print(f"Loaded {len(power_traces)} power traces with shape {power_traces[0].shape}")

    # Create GP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    power_gp = PowerTraceGP(device=device)

    # Prepare data - now grouping by tp and ms only
    data_by_config = power_gp.prepare_data(
        power_traces=power_traces,
        tensor_parallelism=tensor_parallelism,
        poisson_rate=poisson_rate,
        model_size=model_size,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        scale_power=True,
    )

    # Train models (limited to first 3 configs for demonstration)
    models = power_gp.train_all_configs(max_configs=1, training_iterations=100)

    # Visualize a prediction for each configuration with different poisson rates
    for config_key in models.keys():
        # Get first trace for this config
        traces = data_by_config[config_key]["traces"]
        poisson_rates = data_by_config[config_key]["poisson_rates"]

        # Use first 1/3 as seed, middle 1/3 as actual future
        seed_len = len(traces[0]) // 3
        seed_sequence = traces[0][:seed_len]
        actual_future = traces[0][seed_len : 2 * seed_len]

        # Use the poisson rate from the first trace for prediction
        first_poisson_rate = poisson_rates[0]

        # Get tokens if available
        input_tok = None
        if (
            "input_tokens" in data_by_config[config_key]
            and len(data_by_config[config_key]["input_tokens"]) > 0
        ):
            input_tok = data_by_config[config_key]["input_tokens"][0]

        output_tok = None
        if (
            "output_tokens" in data_by_config[config_key]
            and len(data_by_config[config_key]["output_tokens"]) > 0
        ):
            output_tok = data_by_config[config_key]["output_tokens"][0]

        # Visualize with original poisson rate
        save_path = os.path.join(
            output_dir, f"prediction_{config_key}_pr{first_poisson_rate:.1f}.png"
        )
        power_gp.visualize_prediction(
            config_key=config_key,
            seed_sequence=seed_sequence,
            poisson_rate=first_poisson_rate,
            steps=seed_len,
            n_samples=5,
            actual_future=actual_future,
            input_tokens=input_tok,
            output_tokens=output_tok,
            save_path=save_path,
        )

        # Try a different poisson rate to demonstrate parameter flexibility
        different_poisson_rate = first_poisson_rate * 1.5  # 50% higher rate
        save_path = os.path.join(
            output_dir, f"prediction_{config_key}_pr{different_poisson_rate:.1f}.png"
        )
        power_gp.visualize_prediction(
            config_key=config_key,
            seed_sequence=seed_sequence,
            poisson_rate=different_poisson_rate,
            steps=seed_len,
            n_samples=1,
            actual_future=None,  # No ground truth for different rate
            input_tokens=input_tok,
            output_tokens=output_tok,
            save_path=save_path,
        )

    # Save trained models
    model_path = os.path.join(output_dir, "power_gp_models.pt")
    model_dict = {
        key: {
            "model": power_gp.models[key].state_dict(),
            "likelihood": power_gp.likelihoods[key].state_dict(),
            "config": power_gp.data[key]["config"],  # Now just (tp, ms)
        }
        for key in power_gp.models.keys()
    }
    torch.save(model_dict, model_path)
    print(f"Models saved to {model_path}")

    return power_gp


if __name__ == "__main__":
    power_gp = run_gp_pipeline(data_path="./processed_data/power_trace_data.npz")
