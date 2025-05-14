import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, Dataset


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

        for tp, ms, hw in zip(self.tensor_parallelism, self.model_size, self.hardware):
            config = (tp, ms, hw)
            if config not in grouped_data:
                grouped_data[config] = {
                    "power_traces": [],
                    "prefill_tokens": [],
                    "decode_tokens": [],
                    "poisson_rate": [],
                }

            idx = np.where(
                (self.tensor_parallelism == tp)
                & (self.model_size == ms)
                & (self.hardware == hw)
            )[0][
                0
            ]  # Take the first matching index

            grouped_data[config]["power_traces"].append(self.norm_power_traces[idx])
            grouped_data[config]["prefill_tokens"].append(self.norm_prefill[idx])
            grouped_data[config]["decode_tokens"].append(self.norm_decode[idx])
            grouped_data[config]["poisson_rate"].append(self.norm_poisson[idx])

        return grouped_data

    def get_config_data(
        self, tp: int, ms: float, hw: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data for a specific configuration.

        Args:
            tp: Tensor parallelism value
            ms: Model size in billions of parameters
            hw: Hardware type ("A100" or "H100")

        Returns:
            Tuple of (power_traces, prefill_tokens, decode_tokens, poisson_rate)
        """
        config = (tp, ms, hw)
        if config not in self.grouped_data:
            raise ValueError(f"No data available for configuration: {config}")

        return (
            np.array(self.grouped_data[config]["power_traces"]),
            np.array(self.grouped_data[config]["prefill_tokens"]),
            np.array(self.grouped_data[config]["decode_tokens"]),
            np.array(self.grouped_data[config]["poisson_rate"]),
        )

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
        if f"power_{tp}" not in self.data_ranges:
            raise ValueError(f"No data available for tensor parallelism: {tp}")

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


class PowerTraceSequenceDataset(Dataset):
    """PyTorch Dataset for loading power trace sequences."""

    def __init__(
        self,
        power_traces,
        prefill_tokens,
        decode_tokens,
        poisson_rate,
        sequence_length=50,
        stride=10,
        add_features=True,
    ):
        self.power_traces = power_traces
        self.prefill_tokens = prefill_tokens
        self.decode_tokens = decode_tokens
        self.poisson_rate = poisson_rate
        self.sequence_length = sequence_length
        self.stride = stride
        self.add_features = add_features

        self.sequences = self._prepare_sequences()

    def _prepare_sequences(self):
        sequences = []

        for i in range(len(self.power_traces)):
            power = self.power_traces[i]
            prefill = self.prefill_tokens[i]
            decode = self.decode_tokens[i]
            poisson = self.poisson_rate[i]

            # Calculate derivative features if requested
            if self.add_features:
                prefill_rate = np.gradient(prefill)
                decode_rate = np.gradient(decode)
                power_lag = np.pad(power[:-1], (1, 0), mode="edge")

            trace_length = len(power)

            # Create overlapping sequences
            for start in range(0, trace_length - self.sequence_length + 1, self.stride):
                end = start + self.sequence_length

                # Input features
                if self.add_features:
                    # Stack all features for each timestep
                    features = np.stack(
                        [
                            prefill[start:end],
                            decode[start:end],
                            prefill_rate[start:end],
                            decode_rate[start:end],
                            power_lag[start:end],
                            np.full(self.sequence_length, poisson),
                            np.linspace(
                                0, 1, self.sequence_length
                            ),  # Position encoding
                        ],
                        axis=1,
                    )
                else:
                    # Just use basic features
                    features = np.stack(
                        [
                            prefill[start:end],
                            decode[start:end],
                            np.full(self.sequence_length, poisson),
                        ],
                        axis=1,
                    )

                # Target is the power at each timestep
                target = power[start:end]

                sequences.append((features, target))

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        features, target = self.sequences[idx]
        return torch.FloatTensor(features), torch.FloatTensor(target)


class PowerTraceLSTM(nn.Module):
    """LSTM model for power trace prediction."""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super(PowerTraceLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: [batch_size, seq_len, hidden_size]

        # Apply output layer to each timestep
        batch_size, seq_len, _ = lstm_out.shape
        lstm_out_reshaped = lstm_out.contiguous().view(-1, lstm_out.size(-1))
        output = self.output_layer(lstm_out_reshaped)
        output = output.view(batch_size, seq_len)

        return output


class PowerTraceAutoLSTM(nn.Module):
    """Autoregressive LSTM model for power trace prediction."""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super(PowerTraceAutoLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x, hidden=None):
        # x shape: [batch_size, seq_len, features]
        lstm_out, hidden = self.lstm(x, hidden)
        # lstm_out shape: [batch_size, seq_len, hidden_size]

        # Apply output layer to each timestep
        batch_size, seq_len, _ = lstm_out.shape
        lstm_out_reshaped = lstm_out.contiguous().view(-1, lstm_out.size(-1))
        output = self.output_layer(lstm_out_reshaped)
        output = output.view(batch_size, seq_len)

        return output, hidden

    def predict_next(self, x, hidden=None):
        """Predict just the next timestep."""
        # x shape: [batch_size, 1, features]
        lstm_out, hidden = self.lstm(x, hidden)

        # Apply output layer to the last timestep
        output = self.output_layer(lstm_out[:, -1])

        return output.squeeze(-1), hidden


class LSTMPowerTraceGenerator:
    def __init__(self, models_dir: str = None, data_dir: str = None):
        """
        Initialize the LSTM-based power trace generator.

        Args:
            models_dir: Directory to save models
            data_dir: Path to the dataset file
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.models_dir = models_dir or os.path.join(os.getcwd(), "lstm_models")
        self.dataset = PowerTraceDataset(data_dir) if data_dir else None

        os.makedirs(models_dir, exist_ok=True)

    def _get_model_path(self, tp: int, ms: float, hw: str) -> str:
        """Get the path for saving/loading a model."""
        return os.path.join(
            self.models_dir,
            f"lstm_model_tp{tp}_ms{ms}_{hw}.pt",
        )

    def load_model(self, tp: int, ms: float, hw: str):
        """Load a trained model for the specified configuration."""
        model_path = self._get_model_path(tp, ms, hw)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)

            if checkpoint["config"].get("autoregressive", False):
                model = PowerTraceAutoLSTM(
                    input_size=checkpoint["config"]["input_size"],
                    hidden_size=checkpoint["config"]["hidden_size"],
                    num_layers=checkpoint["config"]["num_layers"],
                    dropout=checkpoint["config"]["dropout"],
                ).to(self.device)
            else:
                model = PowerTraceLSTM(
                    input_size=checkpoint["config"]["input_size"],
                    hidden_size=checkpoint["config"]["hidden_size"],
                    num_layers=checkpoint["config"]["num_layers"],
                    dropout=checkpoint["config"]["dropout"],
                ).to(self.device)

            model.load_state_dict(checkpoint["model_state"])
            model.eval()

            self.models[(tp, ms, hw)] = model
            print(f"Model loaded from {model_path}")
            return model
        else:
            print(f"Model not found at {model_path}")
            return None

    def train_all_configs(self, **kwargs):
        """Train models for all available configurations."""
        for tp, ms, hw in self.dataset.get_available_configs():
            print(f"Training model for TP={tp}, MS={ms}, HW={hw}")
            self.train_config(tp=tp, ms=ms, hw=hw, **kwargs)
            print(f"Training complete for TP={tp}, MS={ms}, HW={hw}")
        print("All models trained.")

    def train_config(
        self,
        tp: int,
        ms: float,
        hw: str,
        sequence_length: int = 50,
        stride: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 50,
        autoregressive: bool = True,
        add_features: bool = True,
        validation_split: float = 0.2,
    ):
        """
        Train an LSTM model for a specific configuration.

        Args:
            tp: Tensor parallelism value
            ms: Model size
            hw: Hardware type
            sequence_length: Length of sequence chunks for training
            stride: Step size between sequences
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            autoregressive: Whether to use autoregressive LSTM
            add_features: Whether to add derived features
            validation_split: Fraction of data to use for validation
        """
        try:
            # Get data for this configuration
            power_traces, prefill_tokens, decode_tokens, poisson_rate = (
                self.dataset.get_config_data(tp, ms, hw)
            )

            # Create dataset
            full_dataset = PowerTraceSequenceDataset(
                power_traces,
                prefill_tokens,
                decode_tokens,
                poisson_rate,
                sequence_length=sequence_length,
                stride=stride,
                add_features=add_features,
            )

            # Split into train and validation
            dataset_size = len(full_dataset)
            val_size = int(dataset_size * validation_split)
            train_size = dataset_size - val_size

            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )

            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            print(f"Training data: {train_size} sequences")
            print(f"Validation data: {val_size} sequences")

            # Get the input size from the dataset
            input_size = full_dataset[0][0].shape[1]

            # Create model
            if autoregressive:
                model = PowerTraceAutoLSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                ).to(self.device)
            else:
                model = PowerTraceLSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                ).to(self.device)

            # Optimizer and loss function
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            # Early stopping
            best_val_loss = float("inf")
            patience = 5
            patience_counter = 0

            # Training loop
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0

                for features, targets in tqdm.tqdm(
                    train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"
                ):
                    features = features.to(self.device)
                    targets = targets.to(self.device)

                    optimizer.zero_grad()

                    if autoregressive:
                        outputs, _ = model(features)
                    else:
                        outputs = model(features)

                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * features.size(0)

                train_loss /= train_size

                # Validation
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for features, targets in val_loader:
                        features = features.to(self.device)
                        targets = targets.to(self.device)

                        if autoregressive:
                            outputs, _ = model(features)
                        else:
                            outputs = model(features)

                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * features.size(0)

                val_loss /= val_size

                print(
                    f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
                )

                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    checkpoint = {
                        "model_state": model.state_dict(),
                        "config": {
                            "tp": tp,
                            "ms": ms,
                            "hw": hw,
                            "input_size": input_size,
                            "hidden_size": hidden_size,
                            "num_layers": num_layers,
                            "dropout": dropout,
                            "sequence_length": sequence_length,
                            "autoregressive": autoregressive,
                            "add_features": add_features,
                        },
                    }
                    torch.save(checkpoint, self._get_model_path(tp, ms, hw))
                    print(f"Model saved to {self._get_model_path(tp, ms, hw)}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping after {epoch+1} epochs")
                        break

            # Store the model
            self.models[(tp, ms, hw)] = model

            return model
        except Exception as e:
            print(f"Error training model: {e}")
            return None

    def inference_power_trace(
        self,
        tp: int,
        ms: float,
        hw: str,
        prefill_tokens: np.ndarray,
        decode_tokens: np.ndarray,
        poisson_rate: float,
        time_step: float = 0.25,
        duration: float = 600.0,
        autoregressive_inference: bool = True,
        add_features: bool = True,
    ):
        """
        Generate a power trace for the given inputs.

        Args:
            tp: Tensor parallelism value
            ms: Model size
            hw: Hardware type
            prefill_tokens: Array of prefill token counts over time
            decode_tokens: Array of decode token counts over time
            poisson_rate: Poisson rate parameter
            time_step: Time step between points
            duration: Total duration
            autoregressive_inference: Whether to use autoregressive inference
            add_features: Whether to use derived features

        Returns:
            Generated power trace
        """
        if (tp, ms, hw) not in self.models:
            model = self.load_model(tp, ms, hw)
            if model is None:
                print(f"No model found for TP={tp}, MS={ms}, HW={hw}")
                return None
        else:
            model = self.models[(tp, ms, hw)]

        model.eval()

        # Determine inference length
        trace_length = min(len(prefill_tokens), int(duration / time_step))

        # Initialize power trace
        power_trace = np.zeros(trace_length)

        if autoregressive_inference:
            # Autoregressive inference (one step at a time)
            # Calculate features
            prefill_rate = np.gradient(prefill_tokens[:trace_length])
            decode_rate = np.gradient(decode_tokens[:trace_length])

            # Initialize first power value
            power_trace[0] = 0.5  # Default normalized value

            # Initialize LSTM hidden state
            hidden = None

            with torch.no_grad():
                for i in range(1, trace_length):
                    # Create input features for this timestep
                    if add_features:
                        features = np.array(
                            [
                                prefill_tokens[i],
                                decode_tokens[i],
                                prefill_rate[i],
                                decode_rate[i],
                                power_trace[i - 1],  # Previous power (autoregressive)
                                poisson_rate,
                                i / 100.0,  # Position encoding (normalized)
                            ]
                        ).reshape(
                            1, 1, 7
                        )  # [batch=1, seq_len=1, features=7]
                    else:
                        features = np.array(
                            [prefill_tokens[i], decode_tokens[i], poisson_rate]
                        ).reshape(
                            1, 1, 3
                        )  # [batch=1, seq_len=1, features=3]

                    # Convert to tensor
                    features_tensor = torch.FloatTensor(features).to(self.device)

                    # Predict next power value
                    if isinstance(model, PowerTraceAutoLSTM):
                        output, hidden = model.predict_next(features_tensor, hidden)
                    else:
                        # Handle non-autoregressive model
                        output = model(features_tensor).squeeze(0)

                    # Store prediction
                    power_trace[i] = output.cpu().numpy()
        else:
            # Sliding window inference
            window_size = 50  # Or another appropriate size
            stride = window_size // 2  # 50% overlap

            # Calculate features
            prefill_rate = np.gradient(prefill_tokens[:trace_length])
            decode_rate = np.gradient(decode_tokens[:trace_length])

            # Pad initial power values
            power_lag = np.zeros(trace_length)
            power_lag[0] = 0.5  # Default start value

            with torch.no_grad():
                # Create sliding windows
                for start in range(0, trace_length, stride):
                    end = min(start + window_size, trace_length)
                    actual_size = end - start

                    if actual_size < 5:  # Too small for meaningful prediction
                        continue

                    # Update lagged power values (use what we've predicted so far)
                    if start > 0:
                        power_lag[start:end] = np.pad(
                            power_trace[start - 1 : end - 2], (1, 0), mode="edge"
                        )

                    # Create input features
                    if add_features:
                        features = np.stack(
                            [
                                prefill_tokens[start:end],
                                decode_tokens[start:end],
                                prefill_rate[start:end],
                                decode_rate[start:end],
                                power_lag[start:end],
                                np.full(actual_size, poisson_rate),
                                np.linspace(0, 1, actual_size),  # Position encoding
                            ],
                            axis=1,
                        )
                    else:
                        features = np.stack(
                            [
                                prefill_tokens[start:end],
                                decode_tokens[start:end],
                                np.full(actual_size, poisson_rate),
                            ],
                            axis=1,
                        )

                    # Add batch dimension
                    features = features.reshape(1, actual_size, -1)
                    features_tensor = torch.FloatTensor(features).to(self.device)

                    # Predict power values for this window
                    if isinstance(model, PowerTraceAutoLSTM):
                        outputs, _ = model(features_tensor)
                    else:
                        outputs = model(features_tensor)

                    # Get predictions
                    predictions = outputs.cpu().numpy().reshape(-1)

                    # Blend predictions with existing ones for overlap regions
                    if start > 0:
                        # Create blending weights for smooth transition
                        blend_size = min(stride, actual_size)
                        blend_weights = np.linspace(0, 1, blend_size)

                        # Blend beginning
                        power_trace[start : start + blend_size] = (
                            1 - blend_weights
                        ) * power_trace[
                            start : start + blend_size
                        ] + blend_weights * predictions[
                            :blend_size
                        ]

                        # Use full predictions for the rest
                        if actual_size > blend_size:
                            power_trace[start + blend_size : end] = predictions[
                                blend_size:
                            ]
                    else:
                        # First window - use all predictions
                        power_trace[start:end] = predictions

        # Denormalize if needed
        if hasattr(self.dataset, "denormalize_power"):
            power_trace = self.dataset.denormalize_power(power_trace, tp, hw)

        return power_trace


def main():
    parser = argparse.ArgumentParser(description="LSTM Power Trace Generator")
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
        default="./lstm_models",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference"],
        default="train",
        help="Mode to run the script in",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism for inference/training",
    )
    parser.add_argument(
        "--ms",
        type=float,
        default=8.0,
        help="Model size for inference/training",
    )
    parser.add_argument(
        "--hw",
        type=str,
        default="A100",
        help="Hardware type for inference/training",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="LSTM hidden size",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of LSTM layers",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=50,
        help="Sequence length for training",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Stride between sequences",
    )
    parser.add_argument(
        "--autoregressive",
        action="store_true",
        help="Use autoregressive LSTM model",
    )
    parser.add_argument(
        "--inference_file",
        type=str,
        help="Path to file with token traces for inference",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save inference results",
    )

    args = parser.parse_args()

    # Create the generator
    generator = LSTMPowerTraceGenerator(
        models_dir=args.model_dir, data_dir=args.data_dir
    )

    if args.mode == "train":
        # Train model
        generator.train_config(
            tp=args.tp,
            ms=args.ms,
            hw=args.hw,
            sequence_length=args.sequence_length,
            stride=args.stride,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            autoregressive=args.autoregressive,
        )
    else:
        # Inference mode
        if args.inference_file:
            # Load real token traces
            trace_data = np.load(args.inference_file)

            # Try to find prefill and decode columns
            prefill_key = next(
                (k for k in trace_data.files if "prefill" in k.lower()), None
            )
            decode_key = next(
                (k for k in trace_data.files if "decode" in k.lower()), None
            )

            if prefill_key is None or decode_key is None:
                print("Could not find prefill or decode token arrays in the file")
                print(f"Available keys: {trace_data.files}")
                return

            prefill_tokens = trace_data[prefill_key]
            decode_tokens = trace_data[decode_key]
        else:
            # Generate example token patterns
            trace_length = 2400  # 10 minutes at 250ms per step

            prefill_tokens = np.zeros(trace_length)
            decode_tokens = np.zeros(trace_length)

            # Example: simulate batched inference with periodic prefill spikes
            for i in range(0, trace_length, 200):
                # Add a prefill spike
                prefill_tokens[i : i + 20] = np.linspace(0, 1, 20)  # Ramp up
                prefill_tokens[i + 20 : i + 40] = np.linspace(1, 0, 20)  # Ramp down

                # Add corresponding decode activity
                decode_tokens[i + 40 : i + 120] = 0.7 * np.ones(80)  # Sustained decode

        # Generate power trace
        power_trace = generator.inference_power_trace(
            tp=args.tp,
            ms=args.ms,
            hw=args.hw,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            poisson_rate=0.5,
            autoregressive_inference=args.autoregressive,
        )

        # Save results
        output_file = (
            args.output_file
            or f"lstm_power_trace_tp{args.tp}_ms{args.ms}_{args.hw}.npz"
        )
        np.savez(
            output_file,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            power_trace=power_trace,
            poisson_rate=np.array([0.5]),
        )
        print(f"Results saved to {output_file}")

        # Visualize if matplotlib is available
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))

            plt.subplot(3, 1, 1)
            plt.plot(prefill_tokens)
            plt.title("Prefill Tokens")
            plt.grid(True)

            plt.subplot(3, 1, 2)
            plt.plot(decode_tokens)
            plt.title("Decode Tokens")
            plt.grid(True)

            plt.subplot(3, 1, 3)
            plt.plot(power_trace)
            plt.title("Generated Power Trace")
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(output_file.replace(".npz", ".png"))
            print(f"Plot saved to {output_file.replace('.npz', '.png')}")

        except ImportError:
            print("Matplotlib not available for visualization")


if __name__ == "__main__":
    main()
