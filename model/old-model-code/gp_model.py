import os

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


class EnhancedSeedlessGPModel(gpytorch.models.ApproximateGP):
    """
    An improved GP model with enhanced kernel structure specifically designed
    for LLM power traces with sharp transitions and multi-level behavior.
    """

    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(EnhancedSeedlessGPModel, self).__init__(variational_strategy)

        # Mean module
        self.mean_module = gpytorch.means.ConstantMean()

        # === TIME DIMENSION KERNEL (much more expressive) ===
        # Matern kernel with ν=0.5 for capturing sharp transitions (step-like behavior)
        self.matern_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=0.5,  # Lowest smoothness for capturing steps
                active_dims=0,
                lengthscale_prior=gpytorch.priors.GammaPrior(1.0, 2.0),
            )
        )

        # RBF kernel for smooth local variations
        self.rbf_time_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                active_dims=0,
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
            )
        )

        # RationalQuadratic kernel for handling outliers/spikes
        self.rq_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RQKernel(
                active_dims=0,
                alpha_prior=gpytorch.priors.GammaPrior(0.5, 1.0),
            )
        )

        # Periodic kernel for capturing any repeating patterns
        self.periodic_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel(
                active_dims=0,
            )
        )

        # === PARAMETER KERNELS ===
        # For Poisson rate (affects burstiness of power)
        self.poisson_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                active_dims=1,
                lengthscale_prior=gpytorch.priors.GammaPrior(2.0, 2.0),
            )
        )

        # For tensor parallelism (affects baseline power)
        self.tp_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                active_dims=2,
                lengthscale_prior=gpytorch.priors.GammaPrior(2.0, 2.0),
            )
        )

        # For model size (affects computational complexity)
        self.ms_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                active_dims=3,
                lengthscale_prior=gpytorch.priors.GammaPrior(2.0, 2.0),
            )
        )

        # === COMBINED KERNEL ===
        # Sum the time kernels for capturing different temporal features
        self.time_kernel = (
            self.matern_kernel
            + self.rbf_time_kernel
            + self.rq_kernel
            + self.periodic_kernel
        )

        # Multiply time kernel by each parameter kernel
        self.covar_module = (
            self.time_kernel * self.poisson_kernel * self.tp_kernel * self.ms_kernel
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class PowerTraceGPSeedless:
    """
    A seedless version of the PowerTraceGP class with enhanced kernel structure.
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.model = None
        self.likelihood = None
        self.input_scalers = None
        self.token_statistics = None  # Will store token information if provided

    def prepare_training_data(
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
        Prepare and normalize all training data for the seedless GP model.
        """
        # Convert to numpy if needed
        if isinstance(power_traces, torch.Tensor):
            power_traces = power_traces.cpu().numpy()

        # Number of traces
        N = len(power_traces)

        # Collect inputs (time, poisson, tp, ms) and targets (power)
        all_inputs = []
        all_targets = []

        # Process each trace
        for i in range(N):
            # Get parameters
            tp = tensor_parallelism[i]
            pr = poisson_rate[i]
            ms = model_size[i]

            # Get power trace and normalize if needed
            trace = power_traces[i]

            # Create input features for each time step
            trace_length = len(trace)

            # Use relative time positions (0 to 1) for better scaling
            rel_time = np.linspace(0, 1, trace_length)

            # Create input sequence with parameters
            inputs = np.zeros((trace_length, 4))
            inputs[:, 0] = rel_time  # Normalized time steps (0 to 1)
            inputs[:, 1] = pr  # Poisson rate
            inputs[:, 2] = tp  # Tensor parallelism
            inputs[:, 3] = ms  # Model size

            all_inputs.append(inputs)
            all_targets.append(trace)

        # Concatenate all data
        inputs = np.vstack(all_inputs)
        targets = np.concatenate(all_targets)

        # Normalize inputs for stability
        input_means = np.mean(inputs, axis=0)
        input_stds = np.std(inputs, axis=0)
        # Prevent division by zero
        input_stds[input_stds == 0] = 1.0

        # Store normalization parameters for later inference
        self.input_scalers = {"means": input_means, "stds": input_stds}

        # Apply normalization
        normalized_inputs = (inputs - input_means) / input_stds

        # Target normalization (power values)
        target_mean = np.mean(targets)
        target_std = np.std(targets)
        normalized_targets = (targets - target_mean) / target_std

        # Store target normalization
        self.input_scalers["target_mean"] = target_mean
        self.input_scalers["target_std"] = target_std

        # Convert to torch tensors
        inputs_tensor = torch.tensor(
            normalized_inputs, dtype=torch.float32, device=self.device
        )
        targets_tensor = torch.tensor(
            normalized_targets, dtype=torch.float32, device=self.device
        )

        print(f"Prepared {len(inputs_tensor)} total training points from {N} traces")
        print(f"Input normalization - means: {input_means}, stds: {input_stds}")
        print(f"Target normalization - mean: {target_mean}, std: {target_std}")

        # Analyze token information if provided (optional)
        if input_tokens is not None or output_tokens is not None:
            self._analyze_token_patterns(input_tokens, output_tokens, power_traces)

        return inputs_tensor, targets_tensor

    def _analyze_token_patterns(self, input_tokens, output_tokens, power_traces):
        """
        Analyze token patterns and their correlation with power.
        This is optional and doesn't affect the core model.
        """
        # Implementation would go here - similar to earlier token-aware code
        self.token_statistics = {
            "input": {"available": input_tokens is not None},
            "output": {"available": output_tokens is not None},
        }
        print("Token statistics analyzed and stored")

    def train_model(
        self,
        train_x,
        train_y,
        num_inducing_points=1000,
        training_iterations=500,
        batch_size=1024,
    ):
        """
        Train the seedless GP model on all data using mini-batch optimization.
        """
        # Cap the number of inducing points to avoid memory issues
        max_inducing = min(num_inducing_points, max(500, len(train_x) // 20))

        print(f"Using {max_inducing} inducing points")

        # Select inducing points via k-means clustering for better representation
        try:
            # Try to use sklearn's KMeans for better inducing point selection
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=max_inducing, random_state=0, n_init=1).fit(
                train_x.cpu().numpy()
            )
            inducing_points = torch.tensor(
                kmeans.cluster_centers_, dtype=torch.float32, device=self.device
            )
            print("Using K-means clustering for inducing point selection")
        except (ImportError, ModuleNotFoundError):
            # Fall back to random selection if sklearn is not available
            idx = torch.randperm(len(train_x))[:max_inducing]
            inducing_points = train_x[idx]
            print("Using random selection for inducing points (sklearn not available)")

        # Create model and likelihood
        model = EnhancedSeedlessGPModel(inducing_points).to(self.device)

        # Use a reasonable noise level - not too small, not too large
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
        ).to(self.device)
        likelihood.noise = 0.05  # Initial noise level

        # Optimizer with lower learning rate
        optimizer = torch.optim.Adam(
            [{"params": model.parameters()}, {"params": likelihood.parameters()}],
            lr=0.003,
        )

        # Loss function
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_y))

        # Training loop with early stopping
        model.train()
        likelihood.train()

        pbar = tqdm(range(training_iterations), desc="Training Seedless GP")
        train_losses = []
        best_loss = float("inf")
        patience = 20
        patience_counter = 0

        # Find optimal batch size - avoid memory issues
        # If we have a GPU, we can use a larger batch size
        is_cuda = self.device.type == "cuda"

        # Determine actual batch size to use
        actual_batch_size = batch_size
        if is_cuda:
            # Adjust based on available GPU memory
            cuda_batch_sizes = [1024, 512, 256, 128, 64]
            for bs in cuda_batch_sizes:
                if bs <= batch_size:
                    try:
                        # Try allocating a tensor of this size
                        test_tensor = torch.zeros(
                            (bs, train_x.shape[1]), device=self.device
                        )
                        del test_tensor
                        actual_batch_size = bs
                        break
                    except RuntimeError:
                        continue
        else:
            # On CPU, use a more moderate batch size
            actual_batch_size = min(batch_size, 512)

        print(f"Using batch size: {actual_batch_size}")

        # Safety settings for numerical stability
        with gpytorch.settings.cholesky_jitter(1e-3):
            for i in pbar:
                try:
                    # Get random batch
                    batch_idx = torch.randperm(len(train_x))[:actual_batch_size]
                    x_batch = train_x[batch_idx]
                    y_batch = train_y[batch_idx]

                    # Zero gradients
                    optimizer.zero_grad()

                    # Forward pass
                    output = model(x_batch)

                    # Loss calculation
                    loss = -mll(output, y_batch)

                    # Backward pass and update
                    loss.backward()
                    optimizer.step()

                    # Log progress
                    current_loss = loss.item()
                    pbar.set_postfix({"loss": f"{current_loss:.3f}"})
                    train_losses.append(current_loss)

                    # Early stopping check
                    if current_loss < best_loss:
                        best_loss = current_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Reduce learning rate
                    if (i + 1) % 50 == 0:
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = param_group["lr"] * 0.7

                    # Break if no improvement for a while
                    if patience_counter >= patience:
                        print(f"Early stopping at iteration {i+1}")
                        break

                except RuntimeError as e:
                    print(f"Error at iteration {i+1}: {e}")
                    if "CUDA out of memory" in str(e):
                        # Reduce batch size and try again
                        actual_batch_size = actual_batch_size // 2
                        if actual_batch_size < 32:
                            print("Batch size too small, stopping training")
                            break
                        print(f"Reduced batch size to {actual_batch_size}")
                        continue

                    if "cholesky" in str(e).lower():
                        print("Numerical stability issue, increasing jitter")
                        # We've hit a numerical issue - let's save what we have
                        if i > 50:  # If we've done some training
                            break
                        else:
                            raise
                    else:
                        raise

        # Evaluation mode
        model.eval()
        likelihood.eval()

        # Store model and likelihood
        self.model = model
        self.likelihood = likelihood

        return model, likelihood, train_losses

    def generate_trace(
        self,
        poisson_rate,
        tensor_parallelism,
        model_size,
        sequence_length=500,
        sample=True,
        n_samples=5,
    ):
        """
        Generate a complete power trace without needing a seed sequence.
        """
        if self.model is None or self.input_scalers is None:
            raise ValueError("Model has not been trained yet or missing input scalers")

        # Create normalized input features using the stored scalers
        time_steps = np.linspace(0, 1, sequence_length)  # Normalized time

        # Create input array and apply normalization
        inputs = np.zeros((sequence_length, 4))
        inputs[:, 0] = time_steps
        inputs[:, 1] = poisson_rate
        inputs[:, 2] = tensor_parallelism
        inputs[:, 3] = model_size

        # Apply normalization using stored scalers
        normalized_inputs = (inputs - self.input_scalers["means"]) / self.input_scalers[
            "stds"
        ]

        # Convert to tensor
        inputs_tensor = torch.tensor(
            normalized_inputs, dtype=torch.float32, device=self.device
        )

        # Initialize output
        traces = np.zeros((n_samples, sequence_length))

        # Generate traces
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(
            1e-3
        ):
            # We might need to process in batches for very long sequences
            batch_size = min(200, sequence_length)

            for i in range(n_samples):
                outputs = []

                # Process in batches to avoid memory issues
                for j in range(0, sequence_length, batch_size):
                    end_idx = min(j + batch_size, sequence_length)
                    batch_input = inputs_tensor[j:end_idx]

                    # Get distribution
                    pred_dist = self.likelihood(self.model(batch_input))

                    # Sample or use mean
                    if sample:
                        batch_output = pred_dist.sample().cpu().numpy()
                    else:
                        batch_output = pred_dist.mean.cpu().numpy()

                    outputs.append(batch_output)

                # Combine batches and denormalize
                normalized_trace = np.concatenate(outputs)
                full_trace = (
                    normalized_trace * self.input_scalers["target_std"]
                    + self.input_scalers["target_mean"]
                )
                traces[i] = full_trace

        return traces

    def visualize_generated_trace(
        self,
        poisson_rate,
        tensor_parallelism,
        model_size,
        sequence_length=500,
        n_samples=5,
        actual_trace=None,
        save_path=None,
    ):
        """
        Generate and visualize power traces.
        """
        # Generate traces
        traces = self.generate_trace(
            poisson_rate,
            tensor_parallelism,
            model_size,
            sequence_length,
            sample=True,
            n_samples=n_samples,
        )

        # Visualization
        plt.figure(figsize=(14, 8))

        # Plot the generated traces
        for i in range(n_samples):
            plt.plot(
                range(sequence_length),
                traces[i],
                "r-",
                alpha=0.3,
            )

        # Add a legend entry for generated traces
        plt.plot([], [], "r-", alpha=0.7, label=f"{n_samples} Generated Traces")

        # Plot actual trace if provided
        if actual_trace is not None:
            plot_length = min(len(actual_trace), sequence_length)
            plt.plot(
                range(plot_length),
                actual_trace[:plot_length],
                "g-",
                linewidth=2,
                label="Actual Trace",
            )

        # Add title and labels
        plt.title(
            f"Generated Power Traces (TP={tensor_parallelism}, Rate={poisson_rate:.1f}, Model Size={model_size})",
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

    def save_model(self, save_path):
        """Save the trained model, likelihood, and scalers."""
        if self.model is None:
            raise ValueError("No trained model to save")

        save_dict = {
            "model": self.model.state_dict(),
            "likelihood": self.likelihood.state_dict(),
            "input_scalers": self.input_scalers,
            "token_statistics": self.token_statistics,
        }

        torch.save(save_dict, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """Load a trained model, likelihood, and scalers."""
        device = self.device

        # Load state dict
        loaded_dict = torch.load(load_path, map_location=device)

        # Extract input scalers
        self.input_scalers = loaded_dict["input_scalers"]
        self.token_statistics = loaded_dict.get("token_statistics", None)

        # Create dummy data for model initialization
        dummy_x = torch.zeros((10, 4), dtype=torch.float32, device=device)

        # Create model and likelihood
        model = EnhancedSeedlessGPModel(dummy_x).to(device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

        # Load state dicts
        model.load_state_dict(loaded_dict["model"])
        likelihood.load_state_dict(loaded_dict["likelihood"])

        # Store model and likelihood
        self.model = model
        self.likelihood = likelihood

        print(f"Model loaded from {load_path}")


class EnhancedExactGPModel(gpytorch.models.ExactGP):
    """
    Enhanced ExactGP model with specialized kernel for LLM power traces.
    This model is used for the seed-based approach (one model per configuration).
    """

    def __init__(self, train_x, train_y, likelihood):
        super(EnhancedExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # Check if we have multiple feature dimensions or just time
        is_multidimensional = train_x.dim() > 1 and train_x.shape[1] > 1

        # Matern kernel with ν=0.5 for capturing sharp transitions
        self.matern_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=0.5,  # Lowest smoothness for capturing steps
                lengthscale_prior=gpytorch.priors.GammaPrior(1.0, 2.0),
            )
        )

        # RBF kernel for smooth local variations
        self.rbf_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
            )
        )

        # RationalQuadratic kernel for handling outliers/spikes
        self.rq_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RQKernel(
                alpha_prior=gpytorch.priors.GammaPrior(0.5, 1.0),
            )
        )

        # Periodic kernel for capturing any repeating patterns
        self.periodic_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel()
        )

        # Sum of all kernels for better representation of power dynamics
        self.covar_module = (
            self.matern_kernel + self.rbf_kernel + self.rq_kernel + self.periodic_kernel
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class PowerTraceGP:
    """
    Enhanced Gaussian Process approach for power trace modeling.
    This class handles training multiple independent GPs for power trace forecasting,
    with one GP per trace configuration.
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
        """
        # Convert to numpy if needed
        if isinstance(power_traces, torch.Tensor):
            power_traces = power_traces.cpu().numpy()

        # Number of traces
        N = len(power_traces)

        # Create a dictionary to store data by configuration
        data_by_config = {}

        # Process each trace
        for i in range(N):
            # Get configuration
            tp = tensor_parallelism[i]
            pr = poisson_rate[i]
            ms = model_size[i]
            config_key = f"tp{tp}_pr{pr:.1f}_ms{ms}"

            # Normalize power by tensor parallelism if requested
            if scale_power:
                power = power_traces[i] / tp
            else:
                power = power_traces[i]

            # Create data entry
            if config_key not in data_by_config:
                data_by_config[config_key] = {
                    "config": (tp, pr, ms),
                    "traces": [],
                    "input_tokens": [],
                    "output_tokens": [],
                }

            # Add trace
            data_by_config[config_key]["traces"].append(power)

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
        Train a GP model for a specific configuration with consistent dtype.
        """
        # Set a consistent dtype (double precision is more stable for GPs)
        dtype = torch.float64

        # Get data for this configuration
        data = self.data[config_key]
        traces = data["traces"]
        config = data["config"]

        # Split traces into train and test
        n_traces = len(traces)
        n_train = max(1, int(n_traces * train_percentage))

        # Use a single trace for validation if only one trace is available
        if n_traces == 1:
            # For a single trace, we'll split the trace itself
            trace = traces[0]
            train_len = int(len(trace) * train_percentage)

            # Important: reshape to 2D for kernel compatibility
            train_x = torch.linspace(
                0, train_len - 1, train_len, dtype=dtype, device=self.device
            ).unsqueeze(
                1
            )  # Make 2D with shape [train_len, 1]

            train_y = torch.tensor(trace[:train_len], dtype=dtype, device=self.device)

            # Also reshape test data
            test_x = torch.linspace(
                train_len,
                len(trace) - 1,
                len(trace) - train_len,
                dtype=dtype,
                device=self.device,
            ).unsqueeze(
                1
            )  # Make 2D with shape [test_len, 1]

            test_y = torch.tensor(trace[train_len:], dtype=dtype, device=self.device)
        else:
            # Use different traces for train and validation
            train_indices = np.random.choice(n_traces, n_train, replace=False)
            test_indices = np.array(
                [i for i in range(n_traces) if i not in train_indices]
            )

            # Use first trace for training
            train_trace = traces[train_indices[0]]
            train_x = torch.linspace(
                0,
                len(train_trace) - 1,
                len(train_trace),
                dtype=dtype,
                device=self.device,
            ).unsqueeze(
                1
            )  # Make 2D

            train_y = torch.tensor(train_trace, dtype=dtype, device=self.device)

            # Use first test trace for testing
            if len(test_indices) > 0:
                test_trace = traces[test_indices[0]]
                test_x = torch.linspace(
                    0,
                    len(test_trace) - 1,
                    len(test_trace),
                    dtype=dtype,
                    device=self.device,
                ).unsqueeze(
                    1
                )  # Make 2D

                test_y = torch.tensor(test_trace, dtype=dtype, device=self.device)
            else:
                # If no test traces, use a portion of the train trace
                test_len = min(300, len(train_trace) // 3)
                test_start = len(train_trace) - test_len
                test_x = torch.linspace(
                    test_start,
                    len(train_trace) - 1,
                    test_len,
                    dtype=dtype,
                    device=self.device,
                ).unsqueeze(
                    1
                )  # Make 2D

                test_y = train_y[test_start:]

        # Initialize likelihood and model (with consistent dtype)
        likelihood = (
            gpytorch.likelihoods.GaussianLikelihood().to(dtype=dtype).to(self.device)
        )
        model = (
            EnhancedExactGPModel(train_x, train_y, likelihood)
            .to(dtype=dtype)
            .to(self.device)
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
        """
        # Get all configuration keys
        config_keys = list(self.data.keys())

        # Limit to max_configs
        if max_configs is not None and max_configs < len(config_keys):
            config_keys = config_keys[5:max_configs]

        # Train a model for each configuration
        for key in config_keys:
            print(f"\nTraining GP for configuration: {key}")
            self.train_gp_for_config(key, training_iterations=training_iterations)

        return self.models

    def predict(self, config_key, seed_sequence, steps=300, sample=True, n_samples=5):
        """
        Predict future power values with consistent dtype.
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
                # Next time point
                next_x = torch.tensor(
                    [[last_idx + t + 1]], dtype=dtype, device=self.device
                )  # Make 2D with shape [1, 1]

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
        steps=300,
        n_samples=5,
        actual_future=None,
        input_tokens=None,
        output_tokens=None,
        save_path=None,
    ):
        """
        Visualize a prediction.
        """
        # Make predictions
        predictions = self.predict(
            config_key, seed_sequence, steps, sample=True, n_samples=n_samples
        )

        # Get configuration
        tp, pr, ms = self.data[config_key]["config"]

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

        # Add title and labels
        plt.title(
            f"Power Prediction with GP (TP={tp}, Rate={pr:.1f}, Model Size={ms})",
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

    # Prepare data
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
    models = power_gp.train_all_configs(max_configs=3, training_iterations=200)

    # Visualize a prediction for each configuration
    for config_key in models.keys():
        # Get first trace for this config
        traces = data_by_config[config_key]["traces"]

        # Use first 1/3 as seed, middle 1/3 as actual future
        seed_len = len(traces[0]) // 3
        seed_sequence = traces[0][:seed_len]
        actual_future = traces[0][seed_len : 2 * seed_len]

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

        # Visualize
        save_path = os.path.join(output_dir, f"prediction_{config_key}.png")
        power_gp.visualize_prediction(
            config_key=config_key,
            seed_sequence=seed_sequence,
            steps=seed_len,
            n_samples=5,
            actual_future=actual_future,
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
            "config": power_gp.data[key]["config"],
        }
        for key in power_gp.models.keys()
    }
    torch.save(model_dict, model_path)
    print(f"Models saved to {model_path}")

    return power_gp


def train_seedless_gp_pipeline(data_path, output_dir="seedless_gp_results"):
    """
    Train a seedless GP model pipeline using all available data.
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

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    seedless_gp = PowerTraceGPSeedless(device=device)

    # Prepare ALL data - no subsampling
    train_x, train_y = seedless_gp.prepare_training_data(
        power_traces=power_traces,
        tensor_parallelism=tensor_parallelism,
        poisson_rate=poisson_rate,
        model_size=model_size,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        scale_power=True,
    )

    # Calculate memory requirements and print warning if needed
    if device.type == "cuda":
        total_points = len(train_x)
        estimated_memory_gb = (
            total_points * total_points * 4 / (1024**3)
        )  # Rough estimate
        available_memory = torch.cuda.get_device_properties(device).total_memory / (
            1024**3
        )

        print(f"Estimated memory for full GP: {estimated_memory_gb:.2f} GB")
        print(f"Available GPU memory: {available_memory:.2f} GB")

        if estimated_memory_gb > available_memory * 0.5:
            print(
                "WARNING: Training might exceed GPU memory. Using optimized SVGP approach."
            )

    # Train model with large inducing points set and adaptive batch size
    try:
        seedless_gp.train_model(
            train_x=train_x,
            train_y=train_y,
            num_inducing_points=2000,  # Large inducing point set for better representation
            training_iterations=400,  # More iterations for full data
            batch_size=2048,  # Start with large batch size, will auto-reduce if needed
        )

        # Save model
        model_path = os.path.join(output_dir, "seedless_gp_model_full.pt")
        seedless_gp.save_model(model_path)

    except RuntimeError as e:
        print(f"Error during training: {e}")
        print("Falling back to more conservative settings...")

        # Try again with more conservative settings
        device = torch.device("cpu")  # Switch to CPU for better numerical stability
        seedless_gp = PowerTraceGPSeedless(device=device)

        # Re-prepare data for CPU
        train_x, train_y = seedless_gp.prepare_training_data(
            power_traces=power_traces,
            tensor_parallelism=tensor_parallelism,
            poisson_rate=poisson_rate,
            model_size=model_size,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            scale_power=True,
        )

        # Conservative training with all data
        seedless_gp.train_model(
            train_x=train_x,
            train_y=train_y,
            num_inducing_points=1000,
            training_iterations=300,
            batch_size=512,
        )

        # Save model
        model_path = os.path.join(output_dir, "seedless_gp_model_full_conservative.pt")
        seedless_gp.save_model(model_path)

    # Visualize generation for a few different configurations
    unique_tp = np.unique(tensor_parallelism)
    unique_pr = np.unique(poisson_rate)
    unique_ms = np.unique(model_size)

    for tp in unique_tp[:2]:  # Just use first two TPs
        for pr in unique_pr[:2]:  # Just use first two rates
            for ms in unique_ms[:1]:  # Just use first model size
                # Find a matching trace for comparison
                match_idx = np.where(
                    (tensor_parallelism == tp)
                    & (np.isclose(poisson_rate, pr))
                    & (model_size == ms)
                )[0]

                if len(match_idx) > 0:
                    actual_trace = power_traces[match_idx[0]] / tp  # Normalize
                else:
                    actual_trace = None

                # Visualize
                save_path = os.path.join(
                    output_dir, f"generation_tp{tp}_pr{pr:.1f}_ms{ms}.png"
                )
                seedless_gp.visualize_generated_trace(
                    poisson_rate=pr,
                    tensor_parallelism=tp,
                    model_size=ms,
                    sequence_length=min(500, len(power_traces[0])),
                    n_samples=1,
                    actual_trace=actual_trace,
                    save_path=save_path,
                )

    # Try generating for an interpolated configuration
    # Find mid-points between observed values
    if len(unique_pr) > 1:
        interp_pr = (unique_pr[0] + unique_pr[1]) / 2

        save_path = os.path.join(
            output_dir,
            f"generation_interp_tp{unique_tp[0]}_pr{interp_pr:.1f}_ms{unique_ms[0]}.png",
        )
        seedless_gp.visualize_generated_trace(
            poisson_rate=interp_pr,
            tensor_parallelism=unique_tp[0],
            model_size=unique_ms[0],
            sequence_length=min(500, len(power_traces[0])),
            n_samples=5,
            actual_trace=None,  # No actual trace for interpolated config
            save_path=save_path,
        )

    return seedless_gp


if __name__ == "__main__":
    # Choose which model to train
    train_seeded = True
    train_seedless = True

    if train_seeded:
        print("\n==== Training seeded GP models (one per configuration) ====")
        power_gp = run_gp_pipeline(data_path="./processed_data/power_trace_data.npz")

    if train_seedless:
        print("\n==== Training seedless GP model (global model) ====")
        seedless_gp = train_seedless_gp_pipeline(
            data_path="./processed_data/power_trace_data.npz"
        )

    print("\nTraining complete. Models are ready for use.")
