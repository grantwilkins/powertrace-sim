import gpytorch
import os
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
import tqdm
import itertools


import numpy as np
import torch
import itertools
from sklearn.cluster import MiniBatchKMeans
from typing import Dict, Tuple, Any, List, Optional


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
        self.model_size = data["model_size"]
        self.hardware = data["hardware"]

        # Store original data ranges for later denormalization
        self._store_data_ranges()

        # Normalize data
        self._normalize_data()

        # Group data by configuration
        self.grouped_data = self._group_data()

    def _store_data_ranges(self) -> None:
        """Store min/max ranges for each feature for later denormalization."""
        self.data_ranges = {}

        # Store ranges by tensor parallelism for power traces
        for tp in np.unique(self.tensor_parallelism):
            mask = self.tensor_parallelism == tp
            self.data_ranges[f"power_{tp}"] = {
                "min": np.min(self.power_traces[mask]),
                "max": np.max(self.power_traces[mask]),
            }

        # Store ranges by tensor parallelism and model size for token counts
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

        # Store global ranges for scalar features
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

        # Normalize power traces by tensor parallelism
        for tp in np.unique(self.tensor_parallelism):
            mask = self.tensor_parallelism == tp
            min_val = self.data_ranges[f"power_{tp}"]["min"]
            max_val = self.data_ranges[f"power_{tp}"]["max"]
            self.norm_power_traces[mask] = (self.power_traces[mask] - min_val) / (
                max_val - min_val
            )

        # Normalize prefill and decode tokens by tensor parallelism and model size
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

        pr_min = self.data_ranges["poisson"]["min"]
        pr_max = self.data_ranges["poisson"]["max"]
        if pr_max > pr_min:
            self.norm_poisson = (self.poisson_rate - pr_min) / (pr_max - pr_min)

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
            # Create mask for current configuration
            mask = (
                (self.tensor_parallelism == tp)
                & (self.model_size == ms)
                & (self.hardware == hw)
            )

            # Skip if no data for this configuration
            if not np.any(mask):
                continue

            # Group the normalized data
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

        # Combine all samples
        X_tensor = torch.FloatTensor(np.vstack(X))

        # Select inducing points
        return self._inducing_pts(X_tensor, m)

    def get_train_data(
        self, tp: int, ms: float, hw: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get training data for a specific configuration.

        Args:
            tp: Tensor parallelism value
            ms: Model size in billions of parameters
            hw: Hardware type ("A100" or "H100")

        Returns:
            Tuple of (X, y) tensors for training
        """
        config = (tp, ms, hw)
        if config not in self.grouped_data:
            raise ValueError(f"No data available for configuration: {config}")

        # Get data for this configuration
        config_data = self.grouped_data[config]

        # Prepare training data
        n_samples = len(config_data["power_traces"])
        X = []
        y = []

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
            y.append(config_data["power_traces"][i])

        # Combine all samples
        X_tensor = torch.FloatTensor(np.vstack(X))
        y_tensor = torch.FloatTensor(np.concatenate(y))

        return X_tensor, y_tensor

    def denormalize_power(self, norm_power: np.ndarray, tp: int) -> np.ndarray:
        """
        Denormalize power traces to original scale.

        Args:
            norm_power: Normalized power trace
            tp: Tensor parallelism value used for normalization

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
    def __init__(self, inducing_points, input_dim):

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
            jitter_val=1e-2,
        )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.matern_kernel = gpytorch.kernels.MaternKernel(
            nu=1.5, ard_num_dims=input_dim
        )
        self.linear_kernel = gpytorch.kernels.LinearKernel(ard_num_dims=input_dim)
        self.rbf_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        self.covar_module = (
            self.matern_kernel + 0.5 * self.linear_kernel + 0.1 * self.rbf_kernel
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class PowerTraceGenerator:
    def __init__(self, models_dir: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.models = {}
        self.likelihoods = {}
        self.models_dir = models_dir or os.path.join(os.getcwd(), "powergp_models")

        os.makedirs(models_dir, exist_ok=True)

    def _get_model_path(self, tp: int, ms: str, hw: str) -> str:
        return os.path.join(
            self.models_dir,
            f"model_tp{tp}_ms{ms}_{hw}.pth",
        )

    def load_model(self, tp: int, ms: str, hw: str):
        model_path = self._get_model_path(tp, ms, hw)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model = GPModel(
                checkpoint["config"]["inducing_points"],
                checkpoint["config"]["input_dim"],
            ).to(self.device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()

            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            likelihood.load_state_dict(checkpoint["likelihood_state"])

            self.models[(tp, ms, hw)] = model
            self.likelihoods[(tp, ms, hw)] = likelihood
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model not found at {model_path}")
            return None

    def train_all_configs(self, data_dir: str, n_inducing: int = 1000):
        dataset = PowerTraceDataset(data_dir)
        for tp, ms, hw in dataset.grouped_data.items():
            print(f"Training model for TP={tp}, MS={ms}, HW={hw}")

            self.train_config(
                dataset=dataset,
                hw=hw,
                tp=tp,
                ms=ms,
                n_epochs=100,
                batch_size=512,
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
        n_epochs: int = 100,
        batch_size: int = 512,
        lr: float = 0.01,
        n_inducing: int = 1000,
    ):
        print(f"Training model for TP={tp}, MS={model_size}, HW={hw_type}")
        try:
            train_x, train_y, _ = dataset.get_train_data(hw_type, tp, model_size)
            train_x, train_y = train_x.to(self.device), train_y.to(self.device)
        except ValueError as e:
            print(f"Error: {e}")
            return
        print(f"Training data shape: {train_x.shape}, {train_y.shape}")

        if train_x.shape[0] > n_inducing:
            inducing_points = dataset.get_inducing_points(
                tp=tp, ms=model_size, hw=hw_type, m=n_inducing
            ).to(self.device)
        else:
            inducing_points = train_x.clone()

        # Initialize model and likelihood
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        model = GPModel(inducing_points, train_x.shape[1]).to(self.device)

        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters()},
                {"params": likelihood.parameters()},
            ],
            lr=lr,
        )
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
        train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
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
                tqdm.tqdm.write(
                    f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}"
                )

        save_dict = {
            "model_state": model.state_dict(),
            "likelihood_state": likelihood.state_dict(),
            "config": {
                "hw_type": hw_type,
                "tp": tp,
                "model_size": model_size,
                "input_dim": train_x.shape[1],
                "n_inducing": inducing_points.shape[0],
            },
        }
        torch.save(save_dict, self._get_model_path(tp, model_size, hw_type))
        print(f"Model saved to {self._get_model_path(tp, model_size, hw_type)}")
        self.models[(tp, model_size, hw_type)] = model
        self.likelihoods[(tp, model_size, hw_type)] = likelihood

    def save_model(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.model.state_dict(), os.path.join(model_dir, "model.pth"))
        torch.save(
            self.likelihood.state_dict(), os.path.join(model_dir, "likelihood.pth")
        )
        print(f"Model saved to {model_dir}")

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
    ):
        if (tp, ms, hw) not in self.models:
            print(f"Model for TP={tp}, MS={ms}, HW={hw} not found.")
            return None

        model = self.models[(tp, ms, hw)].to(self.device)
        likelihood = self.likelihoods[(tp, ms, hw)].to(self.device)
        model.eval()
        likelihood.eval()
        num_steps = np.min(len(prefill_tokens), int(duration / time_step))
        
