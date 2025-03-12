import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import pandas as pd
import math
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class PowerTraceDataset(Dataset):
    """Dataset for power trace data with configuration parameters."""

    def __init__(
        self,
        power_traces: np.ndarray,  # Shape: [num_samples, sequence_length]
        config_params: Dict[str, np.ndarray],
        sequence_length: int = 600,  # 10 minutes at 1Hz sampling = 600 points
        stride: int = 60,
    ):
        """
        Args:
            power_traces: Array of power measurements
            config_params: Dictionary of configuration parameters
                - poisson_rate: Poisson arrival rate of queries
                - tensor_parallelism: Degree of tensor parallelism
                - model_size: Model size in parameter count (in billions)
                - is_reasoning: Boolean indicating if reasoning model (1) or not (0)
                - is_h100: Boolean indicating if H100 (1) or A100 (0)
            sequence_length: Length of output sequences
            stride: Stride for sliding window
        """
        self.power_traces = power_traces
        self.config_params = config_params
        self.sequence_length = sequence_length
        self.stride = stride

        # Calculate valid starting indices
        self.valid_indices = []
        for i in range(len(power_traces)):
            for j in range(0, len(power_traces[i]) - sequence_length + 1, stride):
                self.valid_indices.append((i, j))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        sample_idx, start_idx = self.valid_indices[idx]

        # Get sequence of power values
        sequence = self.power_traces[sample_idx][
            start_idx : start_idx + self.sequence_length
        ]

        # Get configuration parameters for this sample
        config = {
            "poisson_rate": self.config_params["poisson_rate"][sample_idx],
            "tensor_parallelism": self.config_params["tensor_parallelism"][sample_idx],
            "model_size": self.config_params["model_size"][sample_idx],
            "is_reasoning": self.config_params["is_reasoning"][sample_idx],
            "is_h100": self.config_params["is_h100"][sample_idx],
        }

        # Convert to tensors
        sequence_tensor = torch.FloatTensor(sequence)

        # Create config tensor
        config_tensor = torch.FloatTensor(
            [
                config["poisson_rate"],
                config["tensor_parallelism"],
                config["model_size"],
                config["is_reasoning"],
                config["is_h100"],
            ]
        )

        return {
            "config": config_tensor,
            "power_trace": sequence_tensor,
            # During training, input = target shifted by one step for autoregressive training
            "input_trace": torch.cat([torch.zeros(1), sequence_tensor[:-1]]),
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1)]
        return x


class ConfigEncoder(nn.Module):
    """Encoder for configuration parameters."""

    def __init__(self, config_dim: int = 5, embed_dim: int = 128):
        super().__init__()
        self.config_embedding = nn.Sequential(
            nn.Linear(config_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, config):
        """
        Args:
            config: Configuration tensor [batch_size, config_dim]

        Returns:
            config_embedding: Embedded configuration [batch_size, embed_dim]
        """
        return self.config_embedding(config)


class PowerTraceTransformer(nn.Module):
    """Transformer model for power trace generation."""

    def __init__(
        self,
        config_dim: int = 5,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_length: int = 600,
    ):
        super().__init__()

        # Config encoder
        self.config_encoder = ConfigEncoder(config_dim, d_model)

        # Input embedding for power values
        self.input_embedding = nn.Sequential(nn.Linear(1, d_model), nn.ReLU())

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, 1)

        # Save hyperparameters
        self.d_model = d_model
        self.max_seq_length = max_seq_length

    def forward(
        self,
        config: torch.Tensor,  # [batch_size, config_dim]
        input_seq: torch.Tensor,  # [batch_size, seq_len]
        tgt_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through the transformer model.

        Args:
            config: Configuration parameters [batch_size, config_dim]
            input_seq: Input power trace sequence [batch_size, seq_len]
            tgt_mask: Target mask for transformer decoder

        Returns:
            output: Predicted power values [batch_size, seq_len]
        """
        batch_size, seq_len = input_seq.shape

        # Encode configuration
        config_embedding = self.config_encoder(config)  # [batch_size, d_model]

        # Repeat config embedding for each position in sequence to form encoder memory
        memory = config_embedding.unsqueeze(1).repeat(
            1, seq_len, 1
        )  # [batch_size, seq_len, d_model]

        # Embed input sequence
        input_seq = input_seq.unsqueeze(-1)  # [batch_size, seq_len, 1]
        tgt = self.input_embedding(input_seq)  # [batch_size, seq_len, d_model]

        # Add positional encoding
        tgt = self.pos_encoder(tgt)

        # Generate target mask if not provided (for autoregressive generation)
        if tgt_mask is None and self.training:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
                tgt.device
            )

        # Pass through transformer
        output = self.transformer(src=memory, tgt=tgt, tgt_mask=tgt_mask)

        # Project to output dimension
        output = self.output_projection(output)  # [batch_size, seq_len, 1]
        output = output.squeeze(-1)  # [batch_size, seq_len]

        return output

    def generate(
        self,
        config: torch.Tensor,  # [batch_size, config_dim]
        seq_length: int = None,
        temperature: float = 1.0,
        initial_context: Optional[torch.Tensor] = None,
    ):
        """
        Generate power traces autoregressively.

        Args:
            config: Configuration parameters [batch_size, config_dim]
            seq_length: Length of sequence to generate
            temperature: Sampling temperature (higher = more random)
            initial_context: Optional initial context for conditioning [batch_size, context_len]

        Returns:
            generated_trace: Generated power trace [batch_size, seq_length]
        """
        if seq_length is None:
            seq_length = self.max_seq_length

        batch_size = config.shape[0]
        device = next(self.parameters()).device

        # Start with zeros if no initial context provided
        if initial_context is None:
            generated = torch.zeros(batch_size, 1, device=device)
        else:
            generated = initial_context

        # Generate one step at a time
        self.eval()
        with torch.no_grad():
            for _ in range(seq_length - generated.size(1)):
                # Predict next value
                next_val = self(config, generated)

                # Get last predicted value
                next_val = next_val[:, -1].unsqueeze(1)

                # Add noise based on temperature (optional)
                if temperature > 0:
                    noise = torch.randn_like(next_val) * temperature
                    next_val = next_val + noise

                # Append to generated sequence
                generated = torch.cat([generated, next_val], dim=1)

        return generated


def train_model(
    model: PowerTraceTransformer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 50,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "mps",
):
    """Train the power trace transformer model."""

    # Move model to device
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Training loop
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # Get data
            config = batch["config"].to(device)
            input_trace = batch["input_trace"].to(device)
            target_trace = batch["power_trace"].to(device)

            # Forward pass
            output = model(config, input_trace)

            # Calculate loss
            loss = criterion(output, target_trace)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update progress bar
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": train_loss / (progress_bar.n + 1)})

        # Calculate average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    # Get data
                    config = batch["config"].to(device)
                    input_trace = batch["input_trace"].to(device)
                    target_trace = batch["power_trace"].to(device)

                    # Forward pass
                    output = model(config, input_trace)

                    # Calculate loss
                    loss = criterion(output, target_trace)
                    val_loss += loss.item()

            # Calculate average validation loss
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # Update learning rate scheduler
            scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}")

    return train_losses, val_losses


def generate_power_traces(
    model: PowerTraceTransformer,
    poisson_rate: float,
    tensor_parallelism: int,
    model_size: float,  # in billions
    is_reasoning: bool,
    is_h100: bool,
    seq_length: int = 600,
    num_samples: int = 1,
    temperature: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "mps",
):
    """Generate power traces for given configuration."""

    # Create configuration tensor
    config = (
        torch.tensor(
            [
                poisson_rate,
                tensor_parallelism,
                model_size,
                float(is_reasoning),
                float(is_h100),
            ],
            device=device,
        )
        .unsqueeze(0)
        .repeat(num_samples, 1)
    )

    # Generate traces
    model.eval()
    with torch.no_grad():
        traces = model.generate(
            config=config, seq_length=seq_length, temperature=temperature
        )

    return traces.cpu().numpy()


def plot_power_traces(
    traces: np.ndarray,
    title: str = "Generated GPU Power Traces",
    sampling_rate: int = 1,
):  # Hz
    """Plot generated power traces."""
    plt.figure(figsize=(12, 6))

    time_axis = np.arange(traces.shape[1]) / sampling_rate  # Convert to seconds

    for i in range(traces.shape[0]):
        plt.plot(time_axis, traces[i], alpha=0.8, label=f"Sample {i+1}")

    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Power (W)")
    plt.grid(True, alpha=0.3)
    if traces.shape[0] <= 10:  # Only show legend if not too many traces
        plt.legend()
    plt.tight_layout()

    return plt


# Example usage
def example_workflow():
    """Example workflow to demonstrate the full pipeline."""

    # 1. Create synthetic data for demonstration
    def generate_synthetic_data(num_samples=50, seq_length=600):
        # Configuration parameters
        poisson_rates = np.random.uniform(1.0, 10.0, num_samples)
        tensor_parallelism = np.random.choice([1, 2, 4, 8], num_samples)
        model_sizes = np.random.uniform(7.0, 70.0, num_samples)  # 7B to 70B
        is_reasoning = np.random.choice([0, 1], num_samples)
        is_h100 = np.random.choice([0, 1], num_samples)

        config_params = {
            "poisson_rate": poisson_rates,
            "tensor_parallelism": tensor_parallelism,
            "model_size": model_sizes,
            "is_reasoning": is_reasoning,
            "is_h100": is_h100,
        }

        # Generate synthetic power traces
        power_traces = []
        for i in range(num_samples):
            # Base power depends on GPU type
            base_power = 100 if is_h100[i] == 0 else 200  # A100 vs H100

            # Model size impact
            model_factor = model_sizes[i] / 10.0

            # Tensor parallelism efficiency
            tp_factor = 0.7 + 0.3 * (1.0 / tensor_parallelism[i])

            # Reasoning impact
            reasoning_factor = 1.0 + 0.5 * is_reasoning[i]

            # Generate time series with query patterns
            trace = np.zeros(seq_length)

            # Add base load
            trace += base_power * model_factor * tp_factor

            # Add query patterns based on Poisson rate
            query_times = np.random.poisson(
                lam=60.0 / poisson_rates[i], size=int(seq_length / 10)
            )
            query_times = np.cumsum(query_times)
            query_times = query_times[query_times < seq_length]

            # Add power spike for each query
            for t in query_times:
                # Query duration depends on model size and reasoning
                query_duration = int(5 + model_sizes[i] / 10.0 * reasoning_factor)
                power_spike = 50 * model_factor * reasoning_factor

                # Add power spike with decay
                for j in range(min(query_duration, seq_length - t)):
                    decay = 1.0 - (j / query_duration)
                    if t + j < seq_length:
                        trace[t + j] += power_spike * decay

            # Add noise
            trace += np.random.normal(0, 5, seq_length)

            # Ensure power is positive
            trace = np.maximum(trace, 0)

            power_traces.append(trace)

        return np.array(power_traces), config_params

    # Generate synthetic data
    power_traces, config_params = generate_synthetic_data(num_samples=100)

    # 2. Create dataset and dataloaders
    dataset = PowerTraceDataset(power_traces, config_params)

    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 3. Create and train model
    model = PowerTraceTransformer(
        config_dim=5, d_model=128, nhead=8, num_encoder_layers=3, num_decoder_layers=3
    )

    train_losses, val_losses = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader, num_epochs=20
    )

    # 4. Generate traces for different configurations
    # Generate traces for A100 vs H100 comparison
    a100_traces = generate_power_traces(
        model=model,
        poisson_rate=5.0,
        tensor_parallelism=4,
        model_size=13.0,
        is_reasoning=False,
        is_h100=False,
        num_samples=3,
    )

    h100_traces = generate_power_traces(
        model=model,
        poisson_rate=5.0,
        tensor_parallelism=4,
        model_size=13.0,
        is_reasoning=False,
        is_h100=True,
        num_samples=3,
    )

    # Compare reasoning vs non-reasoning
    reasoning_traces = generate_power_traces(
        model=model,
        poisson_rate=5.0,
        tensor_parallelism=4,
        model_size=13.0,
        is_reasoning=True,
        is_h100=True,
        num_samples=3,
    )

    # Plot results
    plot_power_traces(a100_traces, title="A100 GPU Power Traces")
    plot_power_traces(h100_traces, title="H100 GPU Power Traces")
    plot_power_traces(reasoning_traces, title="Reasoning Model Power Traces")

    # Return model for further use
    return model


if __name__ == "__main__":
    # Run example workflow
    model = example_workflow()
