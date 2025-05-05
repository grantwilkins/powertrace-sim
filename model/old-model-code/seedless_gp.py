import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import math


class SeedlessGPModel(gpytorch.models.ApproximateGP):
    """
    A numerically stable GP model that can generate power traces without requiring a seed sequence.
    Uses a larger number of inducing points and stable kernel configurations.
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
        super(SeedlessGPModel, self).__init__(variational_strategy)

        # Mean module
        self.mean_module = gpytorch.means.ConstantMean()

        # Time kernel (position in sequence)
        self.time_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(active_dims=0),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
        )

        # Parameter kernels with priors for better stability
        self.params_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                active_dims=[1, 2, 3],
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
        )

        # Combined kernel
        self.covar_module = self.time_kernel * self.params_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class PowerTraceGPSeedless:
    """
    A seedless version of the PowerTraceGP class that trains on the full dataset.
    Enhanced to handle token information as auxiliary data.
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.model = None
        self.likelihood = None
        self.input_scalers = None
        self.token_statistics = None  # Will store statistical information about tokens

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
        Also analyzes token patterns if provided, without changing the core model.
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
            if scale_power:
                trace = trace / tp

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

        # Analyze token information if provided
        if input_tokens is not None or output_tokens is not None:
            self.token_statistics = self._analyze_token_patterns(
                input_tokens, output_tokens, power_traces
            )
            print("Token statistics analyzed and stored for simulation")

        return inputs_tensor, targets_tensor

    def _analyze_token_patterns(self, input_tokens, output_tokens, power_traces):
        """
        Analyze token patterns and their correlation with power to enable realistic simulation.
        This doesn't affect the core GP model but enhances trace generation capabilities.
        """
        print("Analyzing token patterns...")

        # Initialize statistics
        stats = {
            "input": {"available": input_tokens is not None},
            "output": {"available": output_tokens is not None},
        }

        # Process input tokens if available
        if input_tokens is not None:
            in_stats = self._compute_token_stats(input_tokens)
            stats["input"].update(in_stats)

            # Compute correlation with power
            power_corr = []
            for i in range(len(input_tokens)):
                # Get min length
                min_len = min(len(input_tokens[i]), len(power_traces[i]))
                if min_len > 10:  # Only compute if we have enough data
                    corr = np.corrcoef(
                        input_tokens[i][:min_len], power_traces[i][:min_len]
                    )[0, 1]
                    power_corr.append(corr)

            if power_corr:
                stats["input"]["power_correlation"] = {
                    "mean": np.mean(power_corr),
                    "std": np.std(power_corr),
                }
                print(
                    f"Input token-power correlation: {np.mean(power_corr):.4f} ± {np.std(power_corr):.4f}"
                )

        # Process output tokens if available
        if output_tokens is not None:
            out_stats = self._compute_token_stats(output_tokens)
            stats["output"].update(out_stats)

            # Compute correlation with power
            power_corr = []
            for i in range(len(output_tokens)):
                # Get min length
                min_len = min(len(output_tokens[i]), len(power_traces[i]))
                if min_len > 10:  # Only compute if we have enough data
                    corr = np.corrcoef(
                        output_tokens[i][:min_len], power_traces[i][:min_len]
                    )[0, 1]
                    power_corr.append(corr)

            if power_corr:
                stats["output"]["power_correlation"] = {
                    "mean": np.mean(power_corr),
                    "std": np.std(power_corr),
                }
                print(
                    f"Output token-power correlation: {np.mean(power_corr):.4f} ± {np.std(power_corr):.4f}"
                )

        # If both input and output tokens are available, compute cross-correlation
        if input_tokens is not None and output_tokens is not None:
            cross_corr = []
            for i in range(len(input_tokens)):
                # Get min length
                min_len = min(len(input_tokens[i]), len(output_tokens[i]))
                if min_len > 10:  # Only compute if we have enough data
                    corr = np.corrcoef(
                        input_tokens[i][:min_len], output_tokens[i][:min_len]
                    )[0, 1]
                    cross_corr.append(corr)

            if cross_corr:
                stats["cross_correlation"] = {
                    "mean": np.mean(cross_corr),
                    "std": np.std(cross_corr),
                }
                print(
                    f"Input-output token correlation: {np.mean(cross_corr):.4f} ± {np.std(cross_corr):.4f}"
                )

        return stats

    def _compute_token_stats(self, token_sequences):
        """Compute statistics for a set of token sequences."""
        # Extract basic stats for each sequence
        rates = []
        burstiness = []
        autocorr = []

        for seq in token_sequences:
            if len(seq) > 0:
                # Mean rate
                mean_rate = np.mean(seq)
                rates.append(mean_rate)

                # Burstiness (coefficient of variation)
                std = np.std(seq)
                burst = std / (mean_rate + 1e-6)
                burstiness.append(burst)

                # Autocorrelation (lag-1)
                if len(seq) > 1:
                    ac = np.corrcoef(seq[:-1], seq[1:])[0, 1]
                    autocorr.append(ac)

        # Compile statistics
        stats = {}

        if rates:
            stats["rate"] = {
                "mean": np.mean(rates),
                "std": np.std(rates),
                "min": np.min(rates),
                "max": np.max(rates),
            }

        if burstiness:
            stats["burstiness"] = {
                "mean": np.mean(burstiness),
                "std": np.std(burstiness),
            }

        if autocorr:
            stats["autocorrelation"] = {
                "mean": np.mean(autocorr),
                "std": np.std(autocorr),
            }

        return stats

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
        Uses a large number of inducing points for better representation.
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
        model = SeedlessGPModel(inducing_points).to(self.device)

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

    def generate_trace_with_tokens(
        self,
        poisson_rate,
        tensor_parallelism,
        model_size,
        sequence_length=500,
        sample=True,
        n_samples=5,
    ):
        """
        Generate a complete power trace along with simulated token patterns.
        This leverages the analyzed token statistics to produce realistic token patterns.
        """
        # Generate power traces
        power_traces = self.generate_trace(
            poisson_rate,
            tensor_parallelism,
            model_size,
            sequence_length,
            sample,
            n_samples,
        )

        # If we don't have token statistics, just return the power traces
        if self.token_statistics is None:
            print("No token statistics available. Returning only power traces.")
            return power_traces, None, None

        # Initialize token arrays
        input_tokens = None
        output_tokens = None

        # Generate input tokens if statistics are available
        if self.token_statistics["input"]["available"]:
            input_tokens = np.zeros((n_samples, sequence_length))
            for i in range(n_samples):
                input_tokens[i] = self._simulate_token_pattern(
                    "input", sequence_length, power_traces[i]
                )

        # Generate output tokens if statistics are available
        if self.token_statistics["output"]["available"]:
            output_tokens = np.zeros((n_samples, sequence_length))
            # If we have input tokens and cross-correlation info, use both
            use_input_for_output = (
                input_tokens is not None
                and "cross_correlation" in self.token_statistics
            )

            for i in range(n_samples):
                # Pass input tokens if we're using them to influence output tokens
                input_seq = input_tokens[i] if use_input_for_output else None
                output_tokens[i] = self._simulate_token_pattern(
                    "output", sequence_length, power_traces[i], input_seq
                )

        return power_traces, input_tokens, output_tokens

    def _simulate_token_pattern(
        self, token_type, sequence_length, power_trace=None, input_tokens=None
    ):
        """
        Simulate a token pattern based on learned statistics.

        Args:
            token_type: Either "input" or "output"
            sequence_length: Length of the token sequence to generate
            power_trace: Power trace to correlate with token pattern
            input_tokens: Input tokens to influence output tokens (only used if token_type is "output")

        Returns:
            A simulated token sequence
        """
        stats = self.token_statistics[token_type]

        # Extract statistics
        if "rate" not in stats:
            # Default values if statistics are missing
            mean_rate = 1.0
            burst = 1.5
        else:
            mean_rate = stats["rate"]["mean"]
            burst = stats["burstiness"]["mean"]

        # Default autocorrelation if not available
        autocorr = 0.7
        if "autocorrelation" in stats and "mean" in stats["autocorrelation"]:
            autocorr = stats["autocorrelation"]["mean"]

        # Generate initial token pattern
        tokens = np.zeros(sequence_length)

        # Several factors can influence the token pattern:
        # 1. Power trace correlation
        # 2. Input tokens (for output tokens)
        # 3. Autocorrelation within tokens

        # Generate base pattern with autocorrelation
        base_pattern = np.zeros(sequence_length)
        base_pattern[0] = np.random.normal(0, 1)
        for i in range(1, sequence_length):
            base_pattern[i] = autocorr * base_pattern[i - 1] + np.sqrt(
                1 - autocorr**2
            ) * np.random.normal(0, 1)

        # Determine how much each factor contributes
        power_weight = 0.0
        input_weight = 0.0
        base_weight = 1.0

        # If power trace is provided and correlation is known, use it
        if (
            power_trace is not None
            and "power_correlation" in stats
            and "mean" in stats["power_correlation"]
        ):
            power_weight = stats["power_correlation"]["mean"]
            # Rescale power trace to mean zero, unit variance
            norm_power = (power_trace - np.mean(power_trace)) / (
                np.std(power_trace) + 1e-6
            )
            base_weight -= power_weight  # Reduce base weight
        else:
            norm_power = np.zeros(sequence_length)

        # For output tokens, use input tokens as additional influence if available
        if token_type == "output" and input_tokens is not None:
            # Get cross-correlation if available
            if (
                "cross_correlation" in self.token_statistics
                and "mean" in self.token_statistics["cross_correlation"]
            ):
                input_weight = self.token_statistics["cross_correlation"]["mean"]
                base_weight -= input_weight  # Reduce base weight further

                # Use slightly delayed input tokens to simulate processing delay
                delay = min(
                    5, sequence_length // 20
                )  # Small delay, about 5% of trace length
                delayed_input = np.zeros(sequence_length)
                delayed_input[delay:] = input_tokens[: sequence_length - delay]

                # Normalize input tokens
                norm_input = (delayed_input - np.mean(delayed_input)) / (
                    np.std(delayed_input) + 1e-6
                )
            else:
                norm_input = np.zeros(sequence_length)
        else:
            norm_input = np.zeros(sequence_length)

        # Ensure weights are non-negative and sum to 1
        base_weight = max(0.3, base_weight)  # At least 30% randomness
        total_weight = base_weight + power_weight + input_weight
        base_weight /= total_weight
        power_weight /= total_weight
        input_weight /= total_weight

        # Combine factors
        tokens = (
            base_weight * base_pattern
            + power_weight * norm_power
            + input_weight * norm_input
        )

        # Scale to match desired statistics
        tokens = (tokens - np.mean(tokens)) / (np.std(tokens) + 1e-6)
        tokens = tokens * mean_rate * burst + mean_rate

        # Ensure non-negative and round to integers
        tokens = np.maximum(tokens, 0)
        tokens = np.round(tokens).astype(int)

        # Add occasional bursts for more realism
        num_bursts = max(1, int(sequence_length / 100))  # 1 burst per 100 timesteps
        for _ in range(num_bursts):
            burst_start = np.random.randint(0, sequence_length - 10)
            burst_length = np.random.randint(5, 15)
            burst_intensity = np.random.randint(2, 5)

            for j in range(burst_length):
                if burst_start + j < sequence_length:
                    tokens[burst_start + j] = max(
                        tokens[burst_start + j],
                        burst_intensity * np.random.randint(1, 4),
                    )

        return tokens

    def visualize_generated_trace_with_tokens(
        self,
        poisson_rate,
        tensor_parallelism,
        model_size,
        sequence_length=500,
        n_samples=1,
        actual_trace=None,
        actual_input_tokens=None,
        actual_output_tokens=None,
        save_path=None,
    ):
        """
        Generate and visualize power traces with accompanying token patterns.
        """
        # Generate traces with tokens
        traces, input_tokens, output_tokens = self.generate_trace_with_tokens(
            poisson_rate,
            tensor_parallelism,
            model_size,
            sequence_length,
            sample=True,
            n_samples=n_samples,
        )

        # Create a figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # Plot power traces on the first subplot
        ax1 = axes[0]
        for i in range(n_samples):
            ax1.plot(
                range(sequence_length),
                traces[i],
                "r-",
                alpha=0.7 if n_samples == 1 else 0.3,
            )

        # Add a legend entry for generated traces
        ax1.plot([], [], "r-", alpha=0.7, label=f"{n_samples} Generated Traces")

        # Plot actual power trace if provided
        if actual_trace is not None:
            plot_length = min(len(actual_trace), sequence_length)
            ax1.plot(
                range(plot_length),
                actual_trace[:plot_length],
                "g-",
                linewidth=2,
                label="Actual Trace",
            )

        # Add title and labels
        ax1.set_title(
            f"Generated Power Traces (TP={tensor_parallelism}, Rate={poisson_rate:.1f}, Model Size={model_size})",
            fontsize=14,
        )
        ax1.set_ylabel("Normalized Power", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)

        # Plot input tokens on the second subplot
        ax2 = axes[1]
        if input_tokens is not None:
            for i in range(n_samples):
                ax2.plot(
                    range(sequence_length),
                    input_tokens[i],
                    "b-",
                    alpha=0.7 if n_samples == 1 else 0.3,
                )

            # Add a legend entry for generated tokens
            ax2.plot(
                [], [], "b-", alpha=0.7, label=f"{n_samples} Generated Input Tokens"
            )

            # Plot actual input tokens if provided
            if actual_input_tokens is not None:
                plot_length = min(len(actual_input_tokens), sequence_length)
                ax2.plot(
                    range(plot_length),
                    actual_input_tokens[:plot_length],
                    "g-",
                    linewidth=2,
                    label="Actual Input Tokens",
                )

            ax2.set_title("Input Token Traces", fontsize=14)
            ax2.set_ylabel("Input Tokens", fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=11)
        else:
            ax2.text(
                0.5,
                0.5,
                "No input token data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax2.transAxes,
                fontsize=14,
            )

        # Plot output tokens on the third subplot
        ax3 = axes[2]
        if output_tokens is not None:
            for i in range(n_samples):
                ax3.plot(
                    range(sequence_length),
                    output_tokens[i],
                    "m-",
                    alpha=0.7 if n_samples == 1 else 0.3,
                )

            # Add a legend entry for generated tokens
            ax3.plot(
                [], [], "m-", alpha=0.7, label=f"{n_samples} Generated Output Tokens"
            )

            # Plot actual output tokens if provided
            if actual_output_tokens is not None:
                plot_length = min(len(actual_output_tokens), sequence_length)
                ax3.plot(
                    range(plot_length),
                    actual_output_tokens[:plot_length],
                    "g-",
                    linewidth=2,
                    label="Actual Output Tokens",
                )

            ax3.set_title("Output Token Traces", fontsize=14)
            ax3.set_ylabel("Output Tokens", fontsize=12)
            ax3.set_xlabel("Time Step", fontsize=12)
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=11)
        else:
            ax3.text(
                0.5,
                0.5,
                "No output token data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax3.transAxes,
                fontsize=14,
            )

        # Adjust layout
        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")

        plt.show()

    def save_model(self, save_path):
        """Save the trained model, likelihood, scalers and token statistics."""
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
        """Load a trained model, likelihood, scalers and token statistics."""
        device = self.device

        # Load state dict
        loaded_dict = torch.load(load_path, map_location=device)

        # Extract input scalers and token statistics
        self.input_scalers = loaded_dict["input_scalers"]
        self.token_statistics = loaded_dict.get("token_statistics", None)

        # Create dummy data for model initialization
        dummy_x = torch.zeros((10, 4), dtype=torch.float32, device=device)

        # Create model and likelihood
        model = SeedlessGPModel(dummy_x).to(device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

        # Load state dicts
        model.load_state_dict(loaded_dict["model"])
        likelihood.load_state_dict(loaded_dict["likelihood"])

        # Store model and likelihood
        self.model = model
        self.likelihood = likelihood

        print(f"Model loaded from {load_path}")
        if self.token_statistics is not None:
            print("Token statistics loaded for token simulation")


def train_seedless_gp_pipeline(data_path, output_dir="seedless_gp_results"):
    """
    Train a seedless GP model pipeline using all available data,
    including token information if available.
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

    # Load token data if available
    input_tokens = data.get("input_tokens", None)
    output_tokens = data.get("output_tokens", None)

    has_token_data = input_tokens is not None or output_tokens is not None
    if has_token_data:
        print(
            f"Found token data: input_tokens={input_tokens is not None}, output_tokens={output_tokens is not None}"
        )

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

                    # Get tokens if available
                    actual_input = None
                    actual_output = None
                    if input_tokens is not None:
                        actual_input = input_tokens[match_idx[0]]
                    if output_tokens is not None:
                        actual_output = output_tokens[match_idx[0]]
                else:
                    actual_trace = None
                    actual_input = None
                    actual_output = None

                # Visualize with tokens
                save_path = os.path.join(
                    output_dir, f"generation_tp{tp}_pr{pr:.1f}_ms{ms}.png"
                )
                seedless_gp.visualize_generated_trace_with_tokens(
                    poisson_rate=pr,
                    tensor_parallelism=tp,
                    model_size=ms,
                    sequence_length=min(500, len(power_traces[0])),
                    n_samples=1,
                    actual_trace=actual_trace,
                    actual_input_tokens=actual_input,
                    actual_output_tokens=actual_output,
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
        seedless_gp.visualize_generated_trace_with_tokens(
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
    # Train on ALL available data including tokens
    seedless_gp = train_seedless_gp_pipeline(
        data_path="./processed_data/power_trace_data.npz"
    )

    # After training, you can use the model for inference with or without token simulation
    print("\nModel training complete. Generate traces for any configuration:")
    print(
        "Example with tokens: seedless_gp.visualize_generated_trace_with_tokens(poisson_rate=0.5, tensor_parallelism=8, model_size=7)"
    )
    print(
        "Example power only: seedless_gp.visualize_generated_trace(poisson_rate=0.5, tensor_parallelism=8, model_size=7)"
    )
