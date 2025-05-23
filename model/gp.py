import argparse
import itertools
import os
from typing import Dict, List, Tuple

import gpytorch
import numpy as np
import torch
import tqdm
from gpytorch.constraints import Interval
from sklearn.cluster import MiniBatchKMeans


class PowerTraceDataset:
    def __init__(self, data_file: str):
        """
        Load and prepare the power trace dataset from NPZ file.

        Args:
            data_file: Path to the NPZ file containing the data
        """
        data = np.load(data_file)

        # Extract data
        self.power_traces = data["power_traces"]
        self.prefill_tokens = data["prefill_tokens"]
        self.decode_tokens = data["decode_tokens"]
        self.poisson_rate = data["poisson_rate"]
        self.tensor_parallelism = data["tensor_parallelism"]
        self.model_size = data["model_sizes"]
        self.hardware = data["hardware"]
        self._store_data_ranges()
        self._normalize_data()
        self.grouped_data = self._group_data()

    def _store_data_ranges(self) -> None:
        """Store min/max ranges for each feature for later denormalization."""
        self.data_ranges = {}

        for tp in np.unique(self.tensor_parallelism):
            mask = self.tensor_parallelism == tp
            self.data_ranges[f"power_{tp}"] = {
                "min": np.min(self.power_traces[mask]),
                "max": np.max(self.power_traces[mask]),
            }
        for tp in np.unique(self.tensor_parallelism):
            for ms in np.unique(self.model_size):
                mask = (self.tensor_parallelism == tp) & (self.model_size == ms)
                self.data_ranges[f"prefill_{tp}_{ms}"] = {
                    "min": np.min(self.prefill_tokens[mask]),
                    "max": np.max(self.prefill_tokens[mask]),
                }
                self.data_ranges[f"decode_{tp}_{ms}"] = {
                    "min": np.min(self.decode_tokens[mask]),
                    "max": np.max(self.decode_tokens[mask]),
                }

        self.data_ranges["poisson"] = {
            "min": np.min(self.poisson_rate),
            "max": np.max(self.poisson_rate),
        }
        self.data_ranges["tp"] = {
            "min": np.min(self.tensor_parallelism),
            "max": np.max(self.tensor_parallelism),
        }
        self.data_ranges["model_size"] = {
            "min": np.min(self.model_size),
            "max": np.max(self.model_size),
        }

    def _normalize_data(self) -> None:
        """Normalize all features to [0, 1] range."""
        # Initialize normalized arrays
        self.norm_power_traces = np.zeros_like(self.power_traces)
        self.norm_prefill = np.zeros_like(self.prefill_tokens)
        self.norm_decode = np.zeros_like(self.decode_tokens)
        self.norm_poisson = np.zeros_like(self.poisson_rate)
        self.norm_tp = np.zeros_like(self.tensor_parallelism)
        self.norm_model_size = np.zeros_like(self.model_size)
        self.norm_hardware = np.zeros_like(self.hardware, dtype=float)

        for tp in np.unique(self.tensor_parallelism):
            mask = self.tensor_parallelism == tp
            min_val = self.data_ranges[f"power_{tp}"]["min"]
            max_val = self.data_ranges[f"power_{tp}"]["max"]
            self.norm_power_traces[mask] = (self.power_traces[mask] - min_val) / (
                max_val - min_val
            )

        for tp in np.unique(self.tensor_parallelism):
            for ms in np.unique(self.model_size):
                mask = (self.tensor_parallelism == tp) & (self.model_size == ms)
                min_val = self.data_ranges[f"prefill_{tp}_{ms}"]["min"]
                max_val = self.data_ranges[f"prefill_{tp}_{ms}"]["max"]
                if max_val > min_val:
                    self.norm_prefill[mask] = (self.prefill_tokens[mask] - min_val) / (
                        max_val - min_val
                    )
                min_val = self.data_ranges[f"decode_{tp}_{ms}"]["min"]
                max_val = self.data_ranges[f"decode_{tp}_{ms}"]["max"]
                if max_val > min_val:
                    self.norm_decode[mask] = (self.decode_tokens[mask] - min_val) / (
                        max_val - min_val
                    )

        tp_min = self.data_ranges["tp"]["min"]
        tp_max = self.data_ranges["tp"]["max"]
        if tp_max > tp_min:
            self.norm_tp = (self.tensor_parallelism - tp_min) / (tp_max - tp_min)

        ms_min = self.data_ranges["model_size"]["min"]
        ms_max = self.data_ranges["model_size"]["max"]
        if ms_max > ms_min:
            self.norm_model_size = (self.model_size - ms_min) / (ms_max - ms_min)
        self.norm_poisson = self.poisson_rate

        # Normalize hardware to binary values (0 for A100, 1 for H100)
        for i, hw in enumerate(np.unique(self.hardware)):
            mask = self.hardware == hw
            self.norm_hardware[mask] = 0.0 if hw == "A100" else 1.0

    def _group_data(self) -> Dict[Tuple, Dict[str, np.ndarray]]:
        """
        Group data by tensor parallelism, model size, and hardware type.

        Returns:
            Dictionary with configuration tuples as keys and data dictionaries as values
        """
        grouped_data = {}

        for tp, ms, hw in itertools.product(
            np.unique(self.tensor_parallelism),
            np.unique(self.model_size),
            np.unique(self.hardware),
        ):
            mask = (
                (self.tensor_parallelism == tp)
                & (self.model_size == ms)
                & (self.hardware == hw)
            )
            if not np.any(mask):
                continue
            grouped_data[(tp, ms, hw)] = {
                "power_traces": self.norm_power_traces[mask],
                "prefill_tokens": self.norm_prefill[mask],
                "decode_tokens": self.norm_decode[mask],
                "poisson_rate": self.norm_poisson[mask],
            }

        return grouped_data

    def _inducing_pts(self, X: torch.Tensor, m: int) -> torch.Tensor:
        """
        Select inducing points from input data using K-means clustering or random selection.

        Args:
            X: Input tensor from which to select inducing points
            m: Number of inducing points to select

        Returns:
            Tensor of selected inducing points
        """
        if len(X) >= m:
            km = MiniBatchKMeans(n_clusters=m, n_init=1, random_state=0).fit(X.cpu())
            return torch.tensor(km.cluster_centers_, dtype=X.dtype, device=X.device)
        idx = torch.randperm(len(X))[:m]
        return X[idx]

    def get_inducing_points(self, tp: int, ms: float, hw: str, m: int) -> torch.Tensor:
        """
        Get inducing points for a specific configuration.

        Args:
            tp: Tensor parallelism value
            ms: Model size in billions of parameters
            hw: Hardware type ("A100" or "H100")
            m: Number of inducing points to select

        Returns:
            Tensor of selected inducing points
        """
        config = (tp, ms, hw)
        if config not in self.grouped_data:
            raise ValueError(f"No data available for configuration: {config}")

        # Get data for this configuration
        config_data = self.grouped_data[config]

        # Prepare training data
        n_samples = len(config_data["power_traces"])
        X = []

        for i in range(n_samples):
            # Stack features for each time step
            sample_features = np.column_stack(
                [
                    config_data["prefill_tokens"][i],
                    config_data["decode_tokens"][i],
                    np.full_like(
                        config_data["prefill_tokens"][i], config_data["poisson_rate"][i]
                    ),
                ]
            )
            X.append(sample_features)
        X_tensor = torch.FloatTensor(np.vstack(X))
        return self._inducing_pts(X_tensor, m)

    def get_train_data(
        self, tp: int, ms: float, hw: str, sequence_length: int = 10, stride: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get training data as sequences to preserve temporal relationships.

        Args:
            tp: Tensor parallelism value
            ms: Model size in billions of parameters
            hw: Hardware type ("A100" or "H100")
            sequence_length: Length of sequences to extract (time steps)
            stride: Step size between sequences

        Returns:
            Tuple of (X, y) tensors where X includes temporal features
        """

        config = (tp, ms, hw)
        if config not in self.grouped_data:
            raise ValueError(f"No data available for configuration: {config}")

        config_data = self.grouped_data[config]
        n_samples = len(config_data["power_traces"])

        X_sequences = []
        y_sequences = []

        for i in range(n_samples):
            prefill = config_data["prefill_tokens"][i]
            decode = config_data["decode_tokens"][i]
            poisson = config_data["poisson_rate"][i]
            power = config_data["power_traces"][i]
            prefill_rate = np.gradient(prefill)
            decode_rate = np.gradient(decode)
            power_lag = np.pad(power[:-1], (1, 0), mode="edge")
            trace_length = len(power)

            for start in range(0, trace_length - sequence_length + 1, stride):
                end = start + sequence_length

                sequence_features = []
                for j in range(start, end):
                    rel_pos = (j - start) / sequence_length
                    time_step_features = [
                        prefill[j],  # Current prefill tokens
                        decode[j],  # Current decode tokens
                        prefill_rate[j],  # Rate of change in prefill
                        decode_rate[j],  # Rate of change in decode
                        power_lag[j],  # Previous power value
                        poisson,  # Poisson rate (constant)
                        rel_pos,  # Relative position in sequence
                    ]
                    sequence_features.append(time_step_features)

                X_sequences.append(sequence_features)
                y_sequences.append(power[start:end])

        X_tensor = torch.FloatTensor(np.array(X_sequences))
        y_tensor = torch.FloatTensor(np.array(y_sequences))

        return X_tensor, y_tensor

    def _prepare_temporal_features(self) -> None:
        """
        Prepare time-series features to capture temporal relationships between power and tokens.
        """
        self.temporal_features = {}

        for config, data in self.grouped_data.items():
            n_samples = len(data["power_traces"])

            prefill_rates = []
            decode_rates = []
            prefill_moving_avg = []
            decode_moving_avg = []
            power_lagged = []

            for i in range(n_samples):
                prefill_rate = np.gradient(data["prefill_tokens"][i])
                decode_rate = np.gradient(data["decode_tokens"][i])
                window = 5
                prefill_ma = np.convolve(
                    data["prefill_tokens"][i], np.ones(window) / window, mode="same"
                )
                decode_ma = np.convolve(
                    data["decode_tokens"][i], np.ones(window) / window, mode="same"
                )

                power_lag = np.pad(data["power_traces"][i][:-1], (1, 0), mode="edge")
                prefill_rates.append(prefill_rate)
                decode_rates.append(decode_rate)
                prefill_moving_avg.append(prefill_ma)
                decode_moving_avg.append(decode_ma)
                power_lagged.append(power_lag)

            self.temporal_features[config] = {
                "prefill_rates": prefill_rates,
                "decode_rates": decode_rates,
                "prefill_moving_avg": prefill_moving_avg,
                "decode_moving_avg": decode_moving_avg,
                "power_lagged": power_lagged,
            }

    def denormalize_power(self, norm_power: np.ndarray, tp: int, hw: str) -> np.ndarray:
        """
        Denormalize power traces to original scale.

        Args:
            norm_power: Normalized power trace
            tp: Tensor parallelism value used for normalization
            hw: Hardware type ("A100" or "H100")

        Returns:
            Denormalized power trace
        """
        min_val = self.data_ranges[f"power_{tp}"]["min"]
        max_val = self.data_ranges[f"power_{tp}"]["max"]

        return norm_power * (max_val - min_val) + min_val

    def get_available_configs(self) -> List[Tuple]:
        """
        Get list of available configurations in the dataset.

        Returns:
            List of (tp, ms, hw) configuration tuples
        """
        return list(self.grouped_data.keys())


class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, input_dim, sequence_length=10):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
            jitter_val=1e-1,
        )
        super(GPModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()

        self.matern_kernel = gpytorch.kernels.MaternKernel(
            nu=1.5, ard_num_dims=input_dim
        )
        self.linear_kernel = gpytorch.kernels.LinearKernel(ard_num_dims=input_dim)
        self.rbf_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        self.periodic_kernel = gpytorch.kernels.PeriodicKernel(ard_num_dims=input_dim)
        self.mixture_kernel = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=4, ard_num_dims=input_dim
        )

        # Properly scale kernels using ScaleKernel
        # self.scaled_linear = gpytorch.kernels.ScaleKernel(self.linear_kernel)
        # self.scaled_rbf = gpytorch.kernels.ScaleKernel(self.rbf_kernel)
        # self.scaled_periodic = gpytorch.kernels.ScaleKernel(self.periodic_kernel)

        # self.scaled_linear.outputscale = 0.5
        # self.scaled_rbf.outputscale = 0.3
        # self.scaled_periodic.outputscale = 0.2

        self.covar_module = (
            self.matern_kernel
            + self.mixture_kernel
            # + self.scaled_linear
            # + self.scaled_rbf
            # + self.scaled_periodic
        )

        self.sequence_length = sequence_length

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class PowerTraceGenerator:
    def __init__(self, models_dir: str = None, data_dir: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.likelihoods = {}
        self.models_dir = models_dir or os.path.join(os.getcwd(), "powergp_models")
        self.dataset = PowerTraceDataset(data_dir) if data_dir else None
        self.sequence_length = 5

        os.makedirs(models_dir, exist_ok=True)

    def _get_model_path(self, tp: int, ms: str, hw: str) -> str:
        return os.path.join(
            self.models_dir,
            f"model_tp{tp}_ms{ms}_{hw}.pth",
        )

    def load_model(self, tp: int, ms: float, hw: str):
        """
        Load a trained GPModel + GaussianLikelihood from disk and register it
        under self.models[(tp,ms,hw)] and self.likelihoods[(tp,ms,hw)].
        """
        model_path = self._get_model_path(tp, ms, hw)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")

        # 1) load the full dict you saved in train_config
        checkpoint = torch.load(model_path, map_location=self.device)
        model_state = checkpoint["model_state"]
        likelihood_state = checkpoint["likelihood_state"]
        config = checkpoint["config"]

        # 2) pull out the saved inducing‐points shape
        #    this lives under the variational strategy in the state dict
        ip_key = "variational_strategy.inducing_points"
        saved_ips = model_state[ip_key]  # a Tensor of shape (n_inducing, input_dim)
        n_inducing, input_dim = saved_ips.shape
        sequence_length = config.get("sequence_length", None)

        # 3) instantiate a fresh GPModel with dummy inducing‐points of the correct shape
        dummy_ips = torch.zeros(n_inducing, input_dim, device=self.device)
        model = GPModel(
            inducing_points=dummy_ips,
            input_dim=input_dim,
            sequence_length=sequence_length,
        ).to(self.device)

        # 4) load its parameters (this will overwrite the dummy ips with your saved ones)
        model.load_state_dict(model_state)

        # 5) instantiate & load the likelihood
        # likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        likelihood = gpytorch.likelihoods.StudentTLikelihood(
            deg_free_constraint=Interval(2.0, 100.0)
        ).to(self.device)
        likelihood.load_state_dict(likelihood_state)

        # 6) switch to eval mode and register
        model.eval()
        likelihood.eval()
        self.models[(tp, ms, hw)] = model
        self.likelihoods[(tp, ms, hw)] = likelihood

        print(f"Loaded GP model + likelihood for TP={tp}, MS={ms}, HW={hw}")
        return model, likelihood

    def train_all_configs(self, n_inducing: int = 2000):
        for tp, ms, hw in self.dataset.grouped_data.keys():
            print(f"Training model for TP={tp}, MS={ms}, HW={hw}")
            self.train_config(
                dataset=self.dataset,
                hw_type=hw,
                tp=tp,
                model_size=ms,
                n_epochs=250,
                batch_size=4096,
                lr=0.01,
                n_inducing=n_inducing,
            )
            print(f"Training complete for TP={tp}, MS={ms}, HW={hw}")
        print("All models trained.")

    def train_config(
        self,
        dataset: PowerTraceDataset,
        hw_type: str,
        tp: int,
        model_size: float,
        n_epochs: int = 20,
        batch_size: int = 512,
        lr: float = 0.01,
        n_inducing: int = 500,
    ):
        print(f"Training model for TP={tp}, MS={model_size}, HW={hw_type}")
        try:
            train_x, train_y = dataset.get_train_data(
                tp=tp,
                ms=model_size,
                hw=hw_type,
                sequence_length=self.sequence_length,
                stride=10,
            )
            n_sequences, seq_len, n_features = train_x.shape
            train_x_flat = train_x.reshape(-1, n_features).to(self.device)
            train_y_flat = train_y.reshape(-1).to(self.device)

        except ValueError as e:
            print(f"Error: {e}")
            return

        print(
            f"Training data shape: sequences={train_x.shape}, flattened={train_x_flat.shape}"
        )

        if train_x_flat.shape[0] > n_inducing:
            inducing_indices = []
            n_points_per_seq = max(1, n_inducing // n_sequences)
            for i in range(n_sequences):
                seq_indices = torch.linspace(
                    0, seq_len - 1, n_points_per_seq, dtype=torch.long
                )
                global_indices = i * seq_len + seq_indices
                inducing_indices.append(global_indices)

            inducing_indices = torch.cat(inducing_indices)[:n_inducing]
            inducing_points = train_x_flat[inducing_indices].clone()
        else:
            inducing_points = train_x_flat.clone()

        # likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        likelihood = gpytorch.likelihoods.StudentTLikelihood(
            deg_free_constraint=Interval(2.0, 100.0)
        ).to(self.device)
        model = GPModel(
            inducing_points, train_x_flat.shape[1], sequence_length=self.sequence_length
        ).to(self.device)

        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters()},
                {"params": likelihood.parameters()},
            ],
            lr=lr,
        )

        mll = gpytorch.mlls.VariationalELBO(
            likelihood, model, num_data=train_y_flat.size(0)
        )

        train_dataset = torch.utils.data.TensorDataset(train_x_flat, train_y_flat)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        model.train()
        likelihood.train()

        for epoch in tqdm.tqdm(range(n_epochs)):
            epoch_loss = 0.0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 1 == 0:
                tqdm.tqdm.write(
                    f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss/len(train_loader):.4f}"
                )

        save_dict = {
            "model_state": model.state_dict(),
            "likelihood_state": likelihood.state_dict(),
            "config": {
                "hw_type": hw_type,
                "tp": tp,
                "model_size": model_size,
                "input_dim": train_x_flat.shape[1],
                "n_inducing": inducing_points.shape[0],
                "sequence_length": self.sequence_length,
            },
        }

        torch.save(save_dict, self._get_model_path(tp, model_size, hw_type))
        print(
            f"Temporal model saved to {self._get_model_path(tp, model_size, hw_type)}"
        )

        self.models[(tp, model_size, hw_type)] = model
        self.likelihoods[(tp, model_size, hw_type)] = likelihood

    def inference_power_trace(
        self,
        tp: int,
        ms: int,
        hw: str,
        prefill_tokens: np.ndarray,
        decode_tokens: np.ndarray,
        poisson_rate: float,
        time_step: float = 0.25,
        duration: float = 600.0,
    ) -> np.ndarray:
        """
        Generate a power trace using the trained AR-GP model:
        1. Normalize inputs per training data ranges
        2. Autoregressively predict normalized power
        3. Denormalize to original scale
        """
        # 1) Ensure model + likelihood are loaded
        if (tp, ms, hw) not in self.models:
            raise ValueError(f"Model for TP={tp}, MS={ms}, HW={hw} not loaded.")
        model = self.models[(tp, ms, hw)].to(self.device)
        likelihood = self.likelihoods[(tp, ms, hw)].to(self.device)
        model.eval()
        likelihood.eval()

        # 2) Truncate to desired length
        trace_length = min(len(prefill_tokens), int(duration / time_step))

        # 3) Fetch normalization ranges
        prange = self.dataset.data_ranges
        p_min, p_max = (
            prange[f"prefill_{tp}_{ms}"]["min"],
            prange[f"prefill_{tp}_{ms}"]["max"],
        )
        d_min, d_max = (
            prange[f"decode_{tp}_{ms}"]["min"],
            prange[f"decode_{tp}_{ms}"]["max"],
        )

        # 4) Normalize input sequences
        norm_prefill = (prefill_tokens[:trace_length] - p_min) / (p_max - p_min)
        norm_decode = (decode_tokens[:trace_length] - d_min) / (d_max - d_min)

        # 5) Prepare containers for normalized power
        norm_power = np.zeros(trace_length, dtype=float)

        # 6) Initialize first timestep to mean normalized power from training
        key = (tp, ms, hw)
        if key in self.dataset.grouped_data:
            norm_power[0] = np.mean(self.dataset.grouped_data[key]["power_traces"])
        else:
            norm_power[0] = 0.5

        # 7) Precompute token-rate features
        prefill_rate = np.gradient(norm_prefill)
        decode_rate = np.gradient(norm_decode)

        seq_len = self.dataset.get_train_data(tp, ms, hw)[0].shape[1]  # sequence_length

        # 8) Warm-up: autoregress until we have enough history
        with torch.no_grad():
            # WARM-UP
            for i in range(1, min(seq_len, trace_length)):
                feat = torch.tensor(
                    [
                        norm_prefill[i],
                        norm_decode[i],
                        prefill_rate[i],
                        decode_rate[i],
                        norm_power[i - 1],
                        poisson_rate,
                        i / seq_len,
                    ],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                post = likelihood(model(feat))
                # post.mean might be a vector → grab the last element
                m = post.mean.cpu().numpy().ravel()  # shape (N,) even if N==1
                norm_power[i] = m[-1]

            # MAIN AR LOOP
            for i in range(seq_len, trace_length):
                seq_feats = []
                for j in range(i - seq_len, i):
                    rel = (j - (i - seq_len)) / seq_len
                    seq_feats.append(
                        [
                            norm_prefill[j],
                            norm_decode[j],
                            prefill_rate[j],
                            decode_rate[j],
                            norm_power[j - 1],
                            poisson_rate,
                            rel,
                        ]
                    )
                x_seq = torch.tensor(seq_feats, dtype=torch.float32, device=self.device)
                post = likelihood(model(x_seq))
                m = post.mean.cpu().numpy().ravel()
                norm_power[i] = m[-1]

        # 10) Denormalize
        denorm = self.dataset.denormalize_power(norm_power, tp, hw)
        return denorm

    def save_model(self, model_dir):
        """
        Save the model to the specified directory.

        Args:
            model_dir: Directory to save the model
        """
        os.makedirs(model_dir, exist_ok=True)
        for (tp, ms, hw), model in self.models.items():
            torch.save(
                model.state_dict(),
                os.path.join(model_dir, f"model_tp{tp}_ms{ms}_{hw}.pth"),
            )
            print(f"Model saved to {model_dir}")


parser = argparse.ArgumentParser(description="Power Trace Generator")
parser.add_argument(
    "--data_dir",
    type=str,
    help="Path to the power trace dataset",
    default="./vllm-benchmark-llama-3-8b-power.npz",
)
parser.add_argument(
    "--model_dir",
    type=str,
    help="Directory to save the models",
    default="./powergp_models",
)

args = parser.parse_args()
generator = PowerTraceGenerator(models_dir=args.model_dir, data_dir=args.data_dir)
generator.train_all_configs()
generator.save_model(args.model_dir)
# loaded_model, loaded_likelihood = generator.load_model(1, 8, "A100")
# # Example usage
# tp = 1
# ms = 8
# hw = "A100"
# # get example prefill and decode tokens from dataset
# trace_id = 1
# prefill_tokens = generator.dataset.prefill_tokens[trace_id]
# decode_tokens = generator.dataset.decode_tokens[trace_id]
# poisson_rate = generator.dataset.poisson_rate[trace_id]
# power_trace_orig = generator.dataset.power_traces[trace_id]
# tensor_parallelism = generator.dataset.tensor_parallelism[trace_id]
# print("Poisson Rate:", poisson_rate)
# print("tp", tensor_parallelism)
# poisson_rate = 1.0
# generated_trace = generator.inference_power_trace(
#     tp, ms, hw, prefill_tokens, decode_tokens, poisson_rate
# )
# import matplotlib.pyplot as plt

# print(generated_trace)
# plt.plot(generated_trace)
# plt.plot(power_trace_orig)
# plt.title("Generated Power Trace")
# plt.savefig("generated_power_trace.pdf")
