import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import os
import math
from tqdm import tqdm
import pandas as pd


def gaussian_nll_loss(mu, log_sigma, target, reduce=True):
    """
    Gaussian negative log-likelihood loss function.

    Args:
        mu: [B, seq_len] predicted mean
        log_sigma: [B, seq_len] predicted log std
        target: [B, seq_len] target values
        reduce: whether to reduce to a scalar loss

    Returns:
        Loss value
    """
    sigma = torch.exp(log_sigma)  # [B, seq_len]
    nll = 0.5 * (math.log(2 * math.pi) + 2 * log_sigma + ((target - mu) / sigma) ** 2)

    if reduce:
        return nll.mean()
    else:
        return nll


class PowerTraceDataset(Dataset):
    """
    Dataset for power trace modeling.
    """

    def __init__(
        self,
        power_traces,
        tensor_parallelism,
        poisson_rate,
        model_size,
        valid_indices,
        sequence_length,
        input_tokens=None,
        output_tokens=None,
        scale_power=True,
    ):
        self.power_traces = power_traces
        self.tensor_parallelism = tensor_parallelism
        self.poisson_rate = poisson_rate
        self.model_size = model_size
        self.valid_indices = valid_indices
        self.sequence_length = sequence_length
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

        # Scale power values if requested
        if scale_power:
            # Scale the power traces by tensor parallelism to normalize
            self.normalized_power = []
            for i, trace in enumerate(power_traces):
                self.normalized_power.append(trace / (tensor_parallelism[i]))
        else:
            self.normalized_power = power_traces

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get sample index and start position
        sample_idx, start_idx = self.valid_indices[idx]

        # Extract power sequence
        power_seq = self.normalized_power[sample_idx][
            start_idx : start_idx + self.sequence_length
        ]
        power_seq = torch.tensor(power_seq, dtype=torch.float32)

        # Create configuration vector
        config_vec = torch.tensor(
            [
                self.tensor_parallelism[sample_idx],
                self.poisson_rate[sample_idx],
                self.model_size[sample_idx],
            ],
            dtype=torch.float32,
        )

        # Prepare input and target sequences
        # Input is all but the last value, target is all but the first value
        input_power = power_seq[:-1]
        target_power = power_seq[1:]

        result = {
            "config": config_vec,
            "input_seq": input_power,
            "target_seq": target_power,
        }

        # Add token information if available
        if self.input_tokens is not None:
            input_tokens_seq = self.input_tokens[sample_idx][
                start_idx : start_idx + self.sequence_length - 1
            ]
            result["input_tokens"] = torch.tensor(input_tokens_seq, dtype=torch.float32)

        if self.output_tokens is not None:
            output_tokens_seq = self.output_tokens[sample_idx][
                start_idx : start_idx + self.sequence_length - 1
            ]
            result["output_tokens"] = torch.tensor(
                output_tokens_seq, dtype=torch.float32
            )

        return result


def prepare_power_dataset(
    power_traces,
    tensor_parallelism,
    poisson_rate,
    model_size,
    sequence_length=600,
    overlap=0.75,
    batch_size=32,
    num_workers=4,
    input_tokens=None,
    output_tokens=None,
):
    """
    Prepare dataset and dataloaders for power trace modeling.

    Args:
        power_traces: [N, T] array of power traces
        tensor_parallelism: [N] array of tensor parallelism values
        poisson_rate: [N] array of poisson rates
        model_size: [N] array of model sizes
        sequence_length: length of sequences to extract
        overlap: fraction of overlap between sequences
        batch_size: batch size for dataloader
        num_workers: number of workers for dataloader
        input_tokens: [N, T] array of input tokens (optional)
        output_tokens: [N, T] array of output tokens (optional)

    Returns:
        train_loader, val_loader, dataset
    """
    N, T = power_traces.shape

    # Create valid indices with overlap
    valid_indices = []
    step_size = int(sequence_length * (1 - overlap))
    step_size = max(1, step_size)  # Ensure step size is at least 1

    for i in range(N):
        for j in range(0, T - sequence_length, step_size):
            valid_indices.append((i, j))

    valid_indices = np.array(valid_indices)

    # Create dataset
    dataset = PowerTraceDataset(
        power_traces=power_traces,
        tensor_parallelism=tensor_parallelism,
        poisson_rate=poisson_rate,
        model_size=model_size,
        valid_indices=valid_indices,
        sequence_length=sequence_length,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        scale_power=True,
    )

    # Split into train and validation sets
    # Use sample stratification to ensure each power trace is represented in both sets
    sample_indices = np.unique([idx[0] for idx in valid_indices])
    train_size = int(0.8 * len(sample_indices))
    train_samples = np.random.choice(sample_indices, train_size, replace=False)

    train_indices = [
        i for i, idx in enumerate(valid_indices) if idx[0] in train_samples
    ]
    val_indices = [
        i for i, idx in enumerate(valid_indices) if idx[0] not in train_samples
    ]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, dataset


class PowerTraceLSTM(nn.Module):
    """
    LSTM model for power trace prediction with uncertainty estimation.
    """

    def __init__(
        self,
        config_dim=3,
        hidden_dim=128,
        num_layers=3,
        dropout=0.1,
        bidirectional=False,
        use_tokens=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_tokens = use_tokens

        # Input sizes
        self.input_size = 1  # Power value
        if use_tokens:
            self.input_size += 2  # Input and output tokens

        # Config encoder (maps configuration to a vector)
        self.config_encoder = nn.Sequential(
            nn.Linear(config_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size + hidden_dim,  # Power + config embedding
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Output size depends on whether LSTM is bidirectional
        lstm_output_size = hidden_dim * 2 if bidirectional else hidden_dim

        # Output layer for mean and log standard deviation
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, 2),  # Mean and log_std
        )

    def forward(self, power_seq, config, input_tokens=None, output_tokens=None):
        """
        Forward pass.

        Args:
            power_seq: [B, seq_len] input power sequence
            config: [B, config_dim] configuration vector
            input_tokens: [B, seq_len] input token counts (optional)
            output_tokens: [B, seq_len] output token counts (optional)

        Returns:
            mu: [B, seq_len] predicted mean power
            log_sigma: [B, seq_len] predicted log standard deviation
        """
        batch_size, seq_len = power_seq.shape

        # Encode configuration
        config_encoding = self.config_encoder(config)  # [B, hidden_dim]

        # Expand config encoding to match sequence length
        config_encoding = config_encoding.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # [B, seq_len, hidden_dim]

        # Prepare input to LSTM
        power_seq = power_seq.unsqueeze(2)  # [B, seq_len, 1]

        # Create input tensor
        if self.use_tokens and input_tokens is not None and output_tokens is not None:
            # Combine power, input tokens, and output tokens
            input_tokens = input_tokens.unsqueeze(2)  # [B, seq_len, 1]
            output_tokens = output_tokens.unsqueeze(2)  # [B, seq_len, 1]
            lstm_input = torch.cat(
                [power_seq, input_tokens, output_tokens, config_encoding], dim=2
            )
        else:
            # Just power and config
            lstm_input = torch.cat([power_seq, config_encoding], dim=2)

        # Run LSTM
        lstm_output, _ = self.lstm(
            lstm_input
        )  # [B, seq_len, hidden_dim*2] if bidirectional

        # Project to mean and log_std
        output = self.output_layer(lstm_output)  # [B, seq_len, 2]

        # Split into mean and log_std
        mu = output[:, :, 0]  # [B, seq_len]
        log_sigma = output[:, :, 1]  # [B, seq_len]
        log_sigma = torch.clamp(log_sigma, min=-4, max=0.5)  # Stability

        return mu, log_sigma


def train_power_model(
    model,
    train_loader,
    val_loader=None,
    num_epochs=30,
    lr=3e-4,
    weight_decay=1e-5,
    device="cuda",
    patience=5,
):
    """
    Train the power trace LSTM model.

    Args:
        model: model to train
        train_loader: training data loader
        val_loader: validation data loader (optional)
        num_epochs: number of epochs to train for
        lr: learning rate
        weight_decay: weight decay for regularization
        device: device to train on
        patience: patience for early stopping

    Returns:
        train_losses, val_losses
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch in train_pbar:
            # Extract batch data
            config = batch["config"].to(device)
            input_seq = batch["input_seq"].to(device)
            target_seq = batch["target_seq"].to(device)

            # Get token data if available
            input_tokens = batch.get("input_tokens", None)
            if input_tokens is not None:
                input_tokens = input_tokens.to(device)

            output_tokens = batch.get("output_tokens", None)
            if output_tokens is not None:
                output_tokens = output_tokens.to(device)

            # Forward pass
            mu, log_sigma = model(
                input_seq,
                config,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            # Calculate loss
            loss = gaussian_nll_loss(mu, log_sigma, target_seq)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update parameters
            optimizer.step()

            # Track loss
            epoch_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Calculate average training loss
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

            with torch.no_grad():
                for batch in val_pbar:
                    # Extract batch data
                    config = batch["config"].to(device)
                    input_seq = batch["input_seq"].to(device)
                    target_seq = batch["target_seq"].to(device)

                    # Get token data if available
                    input_tokens = batch.get("input_tokens", None)
                    if input_tokens is not None:
                        input_tokens = input_tokens.to(device)

                    output_tokens = batch.get("output_tokens", None)
                    if output_tokens is not None:
                        output_tokens = output_tokens.to(device)

                    # Forward pass
                    mu, log_sigma = model(
                        input_seq,
                        config,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )

                    # Calculate loss
                    loss = gaussian_nll_loss(mu, log_sigma, target_seq)

                    # Track loss
                    val_loss += loss.item()
                    val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Calculate average validation loss
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # Update learning rate scheduler
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

            print(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}")

    # Load best model if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

    return train_losses, val_losses


import torch
import torch.nn as nn
import math


def predict_power_trace(
    model: nn.Module,
    seed_sequence: torch.Tensor,
    config: torch.Tensor,
    steps: int = 300,
    device: str = "cuda",
    sample: bool = True,
    temperature: float = 1.0,
    seed_tokens: dict = None,  # e.g. {"input_tokens": [B, seed_len], "output_tokens": [B, seed_len]}
    n_samples: int = 1,
    all_input_tokens: torch.Tensor = None,  # [B, total_seq_len] if you want real token data for future
    all_output_tokens: torch.Tensor = None,  # [B, total_seq_len] if you want real token data for future
) -> torch.Tensor:
    """
    Autoregressive power prediction with optional token data.

    Args:
        model (nn.Module): Trained LSTM model that expects:
            - Power [B, seq_len]
            - Config [B, config_dim]
            - (optional) input_tokens [B, seq_len]
            - (optional) output_tokens [B, seq_len]
        seed_sequence (torch.Tensor): [B, seed_len]
            The power trace used to initialize the hidden state for prediction.
        config (torch.Tensor): [B, config_dim]
            Configuration vectors (tensor_parallelism, poisson_rate, model_size).
        steps (int): Number of future time steps to predict.
        device (str): Device for inference, e.g. "cpu" or "cuda".
        sample (bool): If True, sample from the predicted Gaussian; else use mean.
        temperature (float): Scales the predicted standard deviation if sampling.
                             E.g., temperature < 1 => less noise, > 1 => more noise.
        seed_tokens (dict): Dictionary holding optional seed token sequences. Each key should be
                            "input_tokens" or "output_tokens", and each value is shape [B, seed_len].
        n_samples (int): How many distinct future trajectories to sample. If > 1, returns shape [B, n_samples, steps].
        all_input_tokens (torch.Tensor): [B, total_seq_len]
            If you want to feed *real* future token data (instead of sampling).
        all_output_tokens (torch.Tensor): [B, total_seq_len]
            If you want to feed *real* future token data (instead of sampling).

    Returns:
        torch.Tensor: The predicted future power values for each sample.
                      Shape is [B, steps] if n_samples=1, or [B, n_samples, steps] if n_samples>1.
    """
    model.eval()

    # 1) Move everything to device
    seed_sequence = seed_sequence.to(device)
    config = config.to(device)

    # If we have "seed tokens," move them to device, too
    current_input_tokens = None
    current_output_tokens = None
    if seed_tokens is not None:
        for k, v in seed_tokens.items():
            seed_tokens[k] = v.to(device)

        current_input_tokens = seed_tokens.get("input_tokens", None)
        current_output_tokens = seed_tokens.get("output_tokens", None)

    # Move all_input_tokens / all_output_tokens to device if they exist
    if all_input_tokens is not None:
        all_input_tokens = all_input_tokens.to(device)
    if all_output_tokens is not None:
        all_output_tokens = all_output_tokens.to(device)

    # Ensure batch dimension
    if seed_sequence.dim() == 1:
        # If shape was [seed_len], make it [1, seed_len]
        seed_sequence = seed_sequence.unsqueeze(0)
    if config.dim() == 1:
        # If shape was [config_dim], make it [1, config_dim]
        config = config.unsqueeze(0)

    # 2) Set up the seed portion
    current_sequence = seed_sequence.clone()  # shape [B, seed_len]
    B, seed_len = current_sequence.shape

    # 3) Prepare to store predictions for each "n_samples" scenario
    # We'll accumulate them in a list, then stack.
    all_trajectories = []

    for sample_idx in range(n_samples):
        # For each sample, we clone the seed so we don't lose it
        seq_clone = current_sequence.clone()
        in_tok_clone = (
            current_input_tokens.clone() if current_input_tokens is not None else None
        )
        out_tok_clone = (
            current_output_tokens.clone() if current_output_tokens is not None else None
        )

        # We'll store the new predictions in [B, steps]
        pred_sequence = torch.zeros(B, steps, device=device)

        # Autoregressive loop
        for t in range(steps):
            with torch.no_grad():
                # Forward pass
                mu, log_sigma = model(
                    power_seq=seq_clone,
                    config=config,
                    input_tokens=in_tok_clone,
                    output_tokens=out_tok_clone,
                )
                # We want the last time step in the sequence
                last_mu = mu[:, -1]  # shape [B]
                last_log_sigma = log_sigma[:, -1]  # shape [B]

                if sample:
                    # sample from Normal(mu, sigma) scaled by 'temperature'
                    eps = torch.randn_like(last_mu)
                    sigma = torch.exp(last_log_sigma)
                    next_val = last_mu + temperature * sigma * eps
                else:
                    # just take the mean
                    next_val = last_mu

                # Store
                pred_sequence[:, t] = next_val

            # Shift the current power sequence to append next_val
            # shape: seq_clone is [B, seed_len + t so far]
            seq_clone = torch.cat([seq_clone[:, 1:], next_val.unsqueeze(1)], dim=1)

            # If we want real tokens from 'all_input_tokens' or 'all_output_tokens'
            # at time step `seed_len + t`, we can index them:
            if in_tok_clone is not None and all_input_tokens is not None:
                # real token index
                real_idx = seed_len + t
                if real_idx < all_input_tokens.shape[1]:
                    new_in_token = all_input_tokens[:, real_idx].unsqueeze(1)  # [B, 1]
                    # Shift while maintaining the same sequence length
                    in_tok_clone = torch.cat([in_tok_clone[:, 1:], new_in_token], dim=1)

            if out_tok_clone is not None and all_output_tokens is not None:
                real_idx = seed_len + t
                if real_idx < all_output_tokens.shape[1]:
                    new_out_token = all_output_tokens[:, real_idx].unsqueeze(
                        1
                    )  # [B, 1]
                    # Shift while maintaining the same sequence length
                    out_tok_clone = torch.cat(
                        [out_tok_clone[:, 1:], new_out_token], dim=1
                    )

        # End steps loop
        all_trajectories.append(pred_sequence)  # shape [B, steps]

    # 4) If n_samples > 1, stack them; else just return the single one
    if n_samples == 1:
        # all_trajectories is a list with one tensor of shape [B, steps]
        return all_trajectories[0]  # shape [B, steps]
    else:
        # stack along dim=1 => shape [B, n_samples, steps]
        return torch.stack(all_trajectories, dim=1)


def visualize_prediction(
    model, val_loader, device="cuda", steps=300, n_samples=5, save_path=None
):
    """
    Visualize a prediction from the model.

    Args:
        model: trained model
        val_loader: validation data loader
        device: device to run prediction on
        steps: number of steps to predict
        n_samples: number of trajectories to sample
        save_path: path to save the visualization
    """
    model.eval()

    # Get a batch
    batch = next(iter(val_loader))

    # Extract data
    config = batch["config"][0].unsqueeze(0).to(device)
    input_seq = batch["input_seq"][0].unsqueeze(0).to(device)
    target_seq = batch["target_seq"][0].unsqueeze(0).to(device)

    # Get tokens if available
    seed_tokens = {}
    if "input_tokens" in batch:
        seed_tokens["input_tokens"] = batch["input_tokens"][0].unsqueeze(0).to(device)

    if "output_tokens" in batch:
        seed_tokens["output_tokens"] = batch["output_tokens"][0].unsqueeze(0).to(device)

    # Get tensor parallelism and poisson rate for reference
    tensor_parallelism = config[0, 0].item()
    poisson_rate = config[0, 1].item()

    # Use part of the sequence as seed
    seed_len = input_seq.shape[1] // 2
    seed_sequence = input_seq[:, :seed_len]

    if "input_tokens" in seed_tokens:
        seed_tokens["input_tokens"] = seed_tokens["input_tokens"][:, :seed_len]

    if "output_tokens" in seed_tokens:
        seed_tokens["output_tokens"] = seed_tokens["output_tokens"][:, :seed_len]

    all_in = batch.get("input_tokens", None)
    if all_in is not None:
        # shape is [32, seq_len], so select the same single row [0]
        all_in = all_in[0].unsqueeze(0).to(device)  # [1, seq_len]
    all_out = batch.get("output_tokens", None)
    if all_out is not None:
        all_out = all_out[0].unsqueeze(0).to(device)  # [1, seq_len]

    # Generate predictions
    predictions = predict_power_trace(
        model,
        seed_sequence,
        config,
        steps=steps,
        device=device,
        sample=True,
        seed_tokens=seed_tokens,
        n_samples=n_samples,
        all_input_tokens=all_in,
        all_output_tokens=all_out,
    )

    # Convert to numpy for plotting
    seed_sequence_np = seed_sequence.cpu().numpy()[0]
    target_full_np = (
        torch.cat([input_seq[0], target_seq[0, -1].unsqueeze(0)]).cpu().numpy()
    )

    # Set up the figure
    plt.figure(figsize=(12, 8))

    # Plot the seed sequence
    plt.plot(
        range(len(seed_sequence_np)), seed_sequence_np, "b-", label="Seed Sequence"
    )

    # Plot the actual future values
    future_start = len(seed_sequence_np)
    future_end = len(target_full_np)
    plt.plot(
        range(future_start, future_end),
        target_full_np[future_start:future_end],
        "g-",
        label="Actual Future",
    )

    # Plot predicted trajectories
    if n_samples > 1:
        # Plot multiple sampled trajectories
        for i in range(n_samples):
            pred_np = predictions[:, i, :].cpu().numpy()[0]
            plt.plot(
                range(future_start, future_start + len(pred_np)),
                pred_np,
                "r-",
                alpha=0.3,
            )

        # Add a single red line for the legend
        plt.plot([], [], "r-", alpha=0.5, label=f"{n_samples} Sampled Trajectories")
    else:
        # Plot a single trajectory
        pred_np = predictions.cpu().numpy()[0]
        plt.plot(
            range(future_start, future_start + len(pred_np)),
            pred_np,
            "r-",
            alpha=0.7,
            label="Predicted Future",
        )

    # Add a vertical line to separate seed from prediction
    plt.axvline(x=future_start, color="k", linestyle="--")

    # Add title and labels
    plt.title(
        f"Power Trace Prediction (TP={tensor_parallelism}, Rate={poisson_rate:.2f})"
    )
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Power")
    plt.legend()
    plt.grid(True)

    # Save if requested
    if save_path:
        plt.savefig(save_path)

    plt.show()


def full_training_pipeline(data_path, output_dir="lstm_model_output"):
    """
    Run the full training pipeline.

    Args:
        data_path: path to the data file
        output_dir: directory to save outputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    # Load data
    data = np.load(data_path)

    power_traces = data["power_traces"]
    tensor_parallelism = data["tensor_parallelism"]
    poisson_rate = data["poisson_rate"]
    model_size = data.get("model_size", np.zeros_like(tensor_parallelism))
    input_tokens = data.get("input_tokens", None)
    output_tokens = data.get("output_tokens", None)

    print(
        f"Loaded data with {len(power_traces)} traces of length {power_traces.shape[1]}"
    )
    print(f"Tensor parallelism values: {tensor_parallelism}")
    print(f"Poisson rates: {poisson_rate}")

    # Prepare datasets and dataloaders
    train_loader, val_loader, dataset = prepare_power_dataset(
        power_traces=power_traces,
        tensor_parallelism=tensor_parallelism,
        poisson_rate=poisson_rate,
        model_size=model_size,
        sequence_length=600,
        overlap=0.75,
        batch_size=32,
        num_workers=4,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")

    # Create model
    model = PowerTraceLSTM(
        config_dim=3,  # tensor_parallelism, poisson_rate, model_size
        hidden_dim=128,
        num_layers=5,
        dropout=0.1,
        bidirectional=True,
        use_tokens=(input_tokens is not None and output_tokens is not None),
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")

    # Train model
    train_losses, val_losses = train_power_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        lr=5e-2,
        weight_decay=1e-5,
        device=device,
        patience=30,
    )

    # Save model
    torch.save(model.state_dict(), "power_lstm_model.pt")

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss.png")

    # Visualize predictions
    visualize_prediction(
        model=model,
        val_loader=val_loader,
        device=device,
        steps=300,
        n_samples=1,
        save_path="prediction_visualization.png",
    )

    return model, train_loader, val_loader


if __name__ == "__main__":
    # Run the full pipeline
    model, train_loader, val_loader = full_training_pipeline(
        data_path="../processed_data/power_trace_data.npz"
    )
