import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import math


def gaussian_nll_loss(mu, log_sigma, target, reduce=True):
    """
    mu:        [B, seq_len] predicted mean
    log_sigma: [B, seq_len] predicted log std
    target:    [B, seq_len]
    """
    sigma = torch.exp(log_sigma)  # [B, seq_len]
    # NLL per element
    nll = 0.5 * (math.log(2 * math.pi) + 2 * log_sigma + ((target - mu) / sigma) ** 2)
    if reduce:
        return nll.mean()
    else:
        return nll


class PowerTraceHSMMDataset(Dataset):
    """
    Enhanced dataset for power trace modeling with HSMM states.

    This dataset:
    1. Identifies discrete states in power traces
    2. Generates state sequences and durations
    3. Returns both power traces and state information
    """

    def __init__(
        self,
        power_traces: np.ndarray,  # [N, T]
        tensor_parallelism: np.ndarray,  # [N]
        poisson_rate: np.ndarray,  # [N]
        model_size: np.ndarray,  # [N] (in billions of parameters)
        valid_indices: np.ndarray,  # [M, 2], each row = (sample_idx, start_idx)
        sequence_length: int,
        n_states: int = 6,
        state_identifier=None,
    ):
        super().__init__()
        self.power_traces = power_traces
        self.tensor_parallelism = tensor_parallelism
        self.poisson_rate = poisson_rate
        self.model_size = model_size
        self.valid_indices = valid_indices
        self.sequence_length = sequence_length
        self.n_states = n_states

        # KMeans-based state labeling (or reuse a pre-fitted model)
        if state_identifier is None:
            print("Identifying states in power traces via KMeans...")
            self.state_identifier = KMeans(n_clusters=n_states, random_state=42)
            flat_data = power_traces.reshape(-1, 1)
            self.state_identifier.fit(flat_data)
        else:
            self.state_identifier = state_identifier

        self.state_sequences = np.zeros_like(power_traces, dtype=np.int32)
        for i in range(len(power_traces)):
            self.state_sequences[i] = self.state_identifier.predict(
                power_traces[i].reshape(-1, 1)
            )

        self.state_centroids = self.state_identifier.cluster_centers_
        self.analyze_states()

    def analyze_states(self):
        """Analyze state transitions and durations in the dataset."""
        # Count occurrences of each state
        state_counts = np.zeros(self.n_states, dtype=np.int32)
        for seq in self.state_sequences:
            unique, counts = np.unique(seq, return_counts=True)
            for st, count in zip(unique, counts):
                state_counts[st] += count

        # Compute average durations for each state
        self.avg_durations = np.zeros(self.n_states, dtype=np.float32)
        duration_counts = np.zeros(self.n_states, dtype=np.int32)

        for seq in self.state_sequences:
            current_state = seq[0]
            current_duration = 1
            for i in range(1, len(seq)):
                if seq[i] == current_state:
                    current_duration += 1
                else:
                    self.avg_durations[current_state] += current_duration
                    duration_counts[current_state] += 1
                    current_state = seq[i]
                    current_duration = 1
            # Handle the last run
            self.avg_durations[current_state] += current_duration
            duration_counts[current_state] += 1

        duration_counts = np.maximum(duration_counts, 1)
        self.avg_durations /= duration_counts

        # Transition matrix
        self.transition_matrix = np.zeros(
            (self.n_states, self.n_states), dtype=np.int32
        )
        for seq in self.state_sequences:
            for i in range(len(seq) - 1):
                self.transition_matrix[seq[i], seq[i + 1]] += 1
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1)
        self.transition_probs = self.transition_matrix / row_sums

        # Print some info
        print("State analysis complete:")
        print(f"  - State counts: {state_counts}")
        print(f"  - Average durations: {self.avg_durations}")
        print(f"  - State centroids: {self.state_centroids.flatten()}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Retrieve which sample & offset
        sample_idx, start_idx = self.valid_indices[idx]

        power_seq = self.power_traces[sample_idx][
            start_idx : start_idx + self.sequence_length
        ]
        power_seq = torch.tensor(power_seq, dtype=torch.float32)

        state_seq = self.state_sequences[sample_idx][
            start_idx : start_idx + self.sequence_length
        ]
        state_seq = torch.tensor(state_seq, dtype=torch.long)

        config_vec = torch.tensor(
            [
                self.tensor_parallelism[sample_idx],
                self.poisson_rate[sample_idx],
                self.model_size[sample_idx],
            ],
            dtype=torch.float32,
        )

        # Shifted input for next-step prediction
        input_seq = torch.cat([torch.zeros(1), power_seq[:-1]], dim=0)

        return {
            "config": config_vec,  # [3]
            "input_seq": input_seq,  # [seq_len]
            "target_seq": power_seq,  # [seq_len]
            "state_seq": state_seq,  # [seq_len]
        }


def prepare_hsmm_dataset(
    power_traces,
    tensor_parallelism,
    poisson_rate,
    model_size,
    sequence_length=550,
    n_states=6,
    batch_size=8,
    num_workers=4,
):
    """
    Prepare DataLoaders for HSMM-Transformer training.

    Args:
        power_traces: [N, T] array of power time-series
        tensor_parallelism: array of shape [N]
        poisson_rate: array of shape [N]
        model_size: array of shape [N]
        sequence_length: how many time steps each sample includes
        n_states: number of discrete states to identify via KMeans
        batch_size: DataLoader batch size
        num_workers: DataLoader num_workers

    Returns:
        train_loader, val_loader, fitted_state_identifier
    """
    N, T = power_traces.shape

    # Create valid (sample_idx, start_idx) pairs
    valid_indices = []
    step_size = sequence_length // 2  # or pick a different overlap
    for i in range(N):
        for j in range(0, T - sequence_length + 1, step_size):
            valid_indices.append((i, j))
    valid_indices = np.array(valid_indices)

    dataset = PowerTraceHSMMDataset(
        power_traces=power_traces,
        tensor_parallelism=tensor_parallelism,
        poisson_rate=poisson_rate,
        model_size=model_size,
        valid_indices=valid_indices,
        sequence_length=sequence_length,
        n_states=n_states,
    )

    # Train / Val Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Build DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, dataset.state_identifier


class HSMMTransitionModel(nn.Module):
    """
    Models state transitions and durations (HSMM).
    """

    def __init__(self, n_states=6, config_dim=2, hidden_dim=64):
        super().__init__()
        self.n_states = n_states

        # Next-state predictor
        self.state_transition = nn.Sequential(
            nn.Linear(config_dim + 1 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_states),
        )

        # Duration predictor (arbitrary design)
        self.duration_predictor = nn.Sequential(
            nn.Linear(config_dim + 1 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 100),
        )

    def forward(self, current_state, duration_so_far, config):
        """
        Predict next state and duration from (current_state, config).

        Args:
            current_state: [B] long
            duration_so_far: [B] float
            config: [B, config_dim]

        Returns:
            next_state_logits: [B, n_states]
            duration_logits:   [B, 100] (example size)
        """
        # Normalize
        norm_state = current_state.float() / max(self.n_states - 1, 1)
        norm_duration = torch.clamp(duration_so_far.float() / 100, 0, 1)

        # Combine as input
        state_inputs = torch.cat(
            [config, norm_state.unsqueeze(1), norm_duration.unsqueeze(1)], dim=1
        )
        next_state_logits = self.state_transition(state_inputs)

        # For the duration, we assume we know next_state. We take argmax for next_state:
        next_state_pred = torch.argmax(next_state_logits, dim=1)
        norm_next_state = next_state_pred.float() / max(self.n_states - 1, 1)

        duration_inputs = torch.cat(
            [config, norm_state.unsqueeze(1), norm_next_state.unsqueeze(1)], dim=1
        )
        duration_logits = self.duration_predictor(duration_inputs)

        return next_state_logits, duration_logits


class HybridHSMMTransformer(nn.Module):
    """
    Hybrid model:
      - A Transformer to predict power usage
      - An HSMM to predict discrete state transitions & durations
    """

    def __init__(
        self,
        n_states=6,
        config_dim=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=600,
    ):
        super().__init__()
        self.hsmm = HSMMTransitionModel(
            n_states=n_states, config_dim=config_dim, hidden_dim=d_model
        )

        self.config_encoder = nn.Sequential(
            nn.Linear(config_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.state_embedding = nn.Embedding(n_states, d_model)
        self.value_embedding = nn.Linear(1, d_model)

        # Simple positional encoder: feed normalized position through MLP
        self.pos_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer: Encoder + Decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Output projection for power
        self.fc_out = nn.Linear(d_model, 2)

        self.d_model = d_model
        self.n_states = n_states

    def forward(self, config, input_seq, state_seq=None, predict_len=1):
        """
        Forward pass:
          - Encodes the full input sequence with the transformer encoder
          - Decodes auto-regressively (with subsequent mask)
          - HSMM is used afterwards to predict next state.

        Args:
            config: [B, config_dim]
            input_seq: [B, seq_len]
            state_seq: [B, seq_len] (optional)
            predict_len: how many steps to predict (currently not used in full)

        Returns:
            power_pred: [B, seq_len]
            state_pred: [B, n_states] distribution (only the next-step, currently)
        """
        device = input_seq.device
        B, seq_len = input_seq.shape

        # Encode config
        config_emb = self.config_encoder(config)

        # Build embeddings for the source
        # positions -> [seq_len, 1], then expand to [B, seq_len, 1]
        positions = torch.arange(seq_len, device=device).unsqueeze(1).float()
        positions = positions / max(seq_len, 1)
        # shape = [seq_len, 1], broadcast to [B, seq_len, 1] in add
        pos_emb = self.pos_encoder(positions)

        power_emb = self.value_embedding(input_seq.unsqueeze(-1))
        if state_seq is not None:
            state_emb = self.state_embedding(state_seq)  # [B, seq_len, d_model]
            # Rely on broadcasting of pos_emb from [seq_len, d_model] to [B, seq_len, d_model]
            input_emb = power_emb + state_emb + pos_emb
        else:
            input_emb = power_emb + pos_emb

        # 1) Encoder (no mask)
        memory = self.transformer_encoder(input_emb)  # [B, seq_len, d_model]

        # 2) Build the decoder input ("teacher forcing" approach)
        #    Shift input to the right by 1 (first token is zeros)
        tgt = torch.cat(
            [
                torch.zeros(B, 1, self.d_model, device=device),
                input_emb[:, :-1, :],
            ],
            dim=1,
        )
        # 3) Mask for the decoder to keep it causal
        tgt_len = tgt.size(1)  # == seq_len
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)

        # 4) Decoder
        output = self.transformer_decoder(tgt, memory, tgt_mask=causal_mask)
        # output shape: [B, seq_len, d_model]

        # 5) Project to power
        # power_pred = self.fc_out(output).squeeze(-1)  # [B, seq_len]

        out = self.fc_out(output)  # [B, seq_len, 2]
        mu = out[..., 0]  # [B, seq_len]
        log_sigma = out[..., 1]  # [B, seq_len]
        log_sigma = torch.clamp(log_sigma, min=-6, max=3)

        # 6) HSMM: if we have states, we use the last state in the sequence
        if state_seq is not None:
            last_state = state_seq[:, -1]
        else:
            last_state = torch.zeros(B, dtype=torch.long, device=device)

        duration_so_far = torch.ones(B, device=device)
        next_state_logits, duration_logits = self.hsmm(
            last_state, duration_so_far, config
        )
        # For training, we return the distribution over next states:
        state_pred = F.softmax(next_state_logits, dim=1)

        return mu, log_sigma, state_pred

    def predict_sequence(self, config, input_seq, state_seq=None, horizon=24):
        """
        Example function to roll out the model for 'horizon' steps.
        Currently does a naive step-by-step approach.
        """
        B, seq_len = input_seq.shape
        device = input_seq.device

        power_preds = torch.zeros(B, horizon, device=device)
        state_preds = torch.zeros(B, horizon, device=device)

        current_power = input_seq.clone()

        if state_seq is None:
            current_state = torch.zeros(B, dtype=torch.long, device=device)
        else:
            current_state = state_seq[:, -1]

        duration_so_far = torch.ones(B, device=device)

        for t in range(horizon):
            # Single forward pass
            power_out, state_distr = self.forward(config, current_power, state_seq)

            # Next power is the last predicted time-step
            next_power = power_out[:, -1]
            # Next state from HSMM
            next_state_logits, _ = self.hsmm(current_state, duration_so_far, config)
            probs = F.softmax(next_state_logits, dim=1)
            next_state = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Store
            power_preds[:, t] = next_power
            state_preds[:, t] = next_state

            # Shift the power input
            current_power = torch.cat(
                [current_power[:, 1:], next_power.unsqueeze(1)], dim=1
            )

            # Track durations for HSMM
            same_state_mask = next_state == current_state
            duration_so_far[same_state_mask] += 1
            duration_so_far[~same_state_mask] = 1

            current_state = next_state

        return power_preds, state_preds


def train_hybrid_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda",
    state_weight: float = 0.2,
):
    """
    Simple training loop for the Hybrid HSMM-Transformer model.

    Args:
        model: The model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        num_epochs: number of epochs
        lr: learning rate
        device: "cuda", "cpu", or "mps"
        state_weight: weighting factor on cross-entropy loss of next-state prediction

    Returns:
        (train_losses, val_losses): Lists of average epoch losses
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    power_criterion = nn.L1Loss()
    state_criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_power_loss = 0.0
        epoch_train_state_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{num_epochs}")
        for batch in train_pbar:
            config = batch["config"].to(device)
            input_seq = batch["input_seq"].to(device)
            target_seq = batch["target_seq"].to(device)

            # States if available
            if "state_seq" in batch:
                state_seq = batch["state_seq"].to(device)
            else:
                state_seq = None

            # power_pred, state_logits = model(config, input_seq, state_seq)
            mu, log_sigma, state_logits = model(config, input_seq, state_seq)
            # power_loss = power_criterion(power_pred, target_seq)
            power_loss = gaussian_nll_loss(mu, log_sigma, target_seq)

            if state_seq is not None:
                state_loss = state_criterion(state_logits, state_seq[:, -1])
            else:
                state_loss = 0.0

            total_loss = power_loss + state_weight * state_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += total_loss.item()
            epoch_train_power_loss += power_loss.item()
            if state_seq is not None:
                epoch_train_state_loss += state_loss.item()

            train_pbar.set_postfix(
                {
                    "total_loss": f"{total_loss.item():.4f}",
                    "power_loss": f"{power_loss.item():.4f}",
                    "state_loss": f"{state_loss if state_seq is not None else 0.0:.4f}",
                }
            )

        # Average over the train set
        epoch_train_loss /= len(train_loader)
        epoch_train_power_loss /= len(train_loader)
        if len(train_loader) > 0 and state_seq is not None:
            epoch_train_state_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        if val_loader is not None:
            model.eval()
            epoch_val_loss = 0.0
            epoch_val_power_loss = 0.0
            epoch_val_state_loss = 0.0

            val_pbar = tqdm(val_loader, desc=f"[Val]   Epoch {epoch+1}/{num_epochs}")
            with torch.no_grad():
                for batch in val_pbar:
                    config = batch["config"].to(device)
                    input_seq = batch["input_seq"].to(device)
                    target_seq = batch["target_seq"].to(device)

                    if "state_seq" in batch:
                        state_seq = batch["state_seq"].to(device)
                    else:
                        state_seq = None

                    # power_pred, state_logits = model(config, input_seq, state_seq)
                    mu, log_sigma, state_logits = model(config, input_seq, state_seq)

                    # power_loss = power_criterion(power_pred, target_seq)
                    power_loss = gaussian_nll_loss(mu, log_sigma, target_seq)

                    if state_seq is not None:
                        state_loss = state_criterion(state_logits, state_seq[:, -1])
                    else:
                        state_loss = 0.0

                    total_loss = power_loss + state_weight * state_loss

                    epoch_val_loss += total_loss.item()
                    epoch_val_power_loss += power_loss.item()
                    epoch_val_state_loss += (
                        state_loss.item() if state_seq is not None else 0.0
                    )

                    val_pbar.set_postfix(
                        {
                            "total_loss": f"{total_loss.item():.4f}",
                            "power_loss": f"{power_loss.item():.4f}",
                            "state_loss": f"{state_loss if state_seq is not None else 0.0:.4f}",
                        }
                    )

            epoch_val_loss /= len(val_loader)
            epoch_val_power_loss /= len(val_loader)
            epoch_val_state_loss /= len(val_loader) if state_seq is not None else 1
            val_losses.append(epoch_val_loss)

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {epoch_train_loss:.4f} (Power: {epoch_train_power_loss:.4f}, State: {epoch_train_state_loss:.4f}) | "
                f"Val Loss: {epoch_val_loss:.4f} (Power: {epoch_val_power_loss:.4f}, State: {epoch_val_state_loss:.4f})"
            )
        else:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {epoch_train_loss:.4f} (Power: {epoch_train_power_loss:.4f}, State: {epoch_train_state_loss:.4f})"
            )

    return train_losses, val_losses


def predict_sequence_closed_loop(
    model, config, init_power, init_states, horizon=100, device="cpu"
):
    """
    Generate a sequence of future predictions in a 'closed-loop':
      - Each newly predicted state is appended to the running state sequence,
      - Each newly predicted power is appended to the running power sequence,
    so the Transformer conditions on its own predicted states in the next step.

    Args:
        model: trained HybridHSMMTransformer
        config: tensor [B, config_dim]
        init_power: initial seed power [B, seed_len]
        init_states: initial seed states [B, seed_len] (if you want the model to see states)
        horizon: how many steps to generate
        device: "cpu", "cuda", or "mps"

    Returns:
        full_power (torch.Tensor): [B, seed_len + horizon] combined seed + predicted power
        full_states (torch.Tensor): [B, seed_len + horizon] combined seed + predicted states
    """
    model.eval()

    B, seed_len = init_power.shape

    # We'll keep "running" sequences that grow with each new step
    current_power = init_power.clone()  # shape [B, seed_len]
    current_states = init_states.clone() if init_states is not None else None

    # Start from the last known state if we have them, or assume zeros
    if current_states is not None:
        last_state = current_states[:, -1]  # shape [B]
    else:
        last_state = torch.zeros(B, dtype=torch.long, device=device)

    duration_so_far = torch.ones(B, device=device)  # HSMM duration tracking

    power_generated = []
    state_generated = []

    for t in range(horizon):
        # Forward pass with the entire current sequence
        # If we have states, pass them in; else pass None
        mu, log_sigma, _ = model(config, current_power, current_states)
        # "power_pred" shape: [B, current_seq_len]

        # The newly predicted power is the last time-step
        next_mu = mu[:, -1]
        next_ls = log_sigma[:, -1]
        # sample from Normal(next_mu, exp(next_ls))
        next_sigma = torch.exp(next_ls)
        eps = torch.randn_like(next_sigma)
        next_power = next_mu + eps * next_sigma

        # Now get next state from HSMM
        next_state_logits, _ = model.hsmm(last_state, duration_so_far, config)
        next_state = torch.argmax(next_state_logits, dim=1)

        power_generated.append(next_power.unsqueeze(1))  # shape [B, 1]
        state_generated.append(next_state.unsqueeze(1))  # shape [B, 1]

        # Append to "running" sequences
        current_power = torch.cat([current_power, next_power.unsqueeze(1)], dim=1)

        if current_states is not None:
            current_states = torch.cat([current_states, next_state.unsqueeze(1)], dim=1)
        else:
            # If we started with no states, we'll build them from scratch
            if t == 0:
                # build with shape [B, 1]
                current_states = next_state.unsqueeze(1)
            else:
                current_states = torch.cat(
                    [current_states, next_state.unsqueeze(1)], dim=1
                )

        # HSMM duration update
        same_state_mask = next_state == last_state
        duration_so_far[same_state_mask] += 1
        duration_so_far[~same_state_mask] = 1
        last_state = next_state

    # Concatenate predicted future steps
    power_generated = torch.cat(power_generated, dim=1)  # [B, horizon]
    state_generated = torch.cat(state_generated, dim=1)  # [B, horizon]

    # Combine seed + predictions
    full_power = torch.cat(
        [init_power, power_generated], dim=1
    )  # [B, seed_len + horizon]
    if init_states is not None:
        full_states = torch.cat(
            [init_states, state_generated], dim=1
        )  # [B, seed_len + horizon]
    else:
        full_states = state_generated  # since we built from scratch

    return full_power, full_states


if __name__ == "__main__":
    # Example usage:
    data_path = os.path.join("processed_data", "power_trace_data.npz")
    data = np.load(data_path)

    power_traces = data["power_traces"]  # shape [N, T]
    tensor_parallelism = data["tensor_parallelism"]  # shape [N]
    poisson_rate = data["poisson_rate"]  # shape [N]
    model_size = data.get("model_size", np.zeros_like(tensor_parallelism))

    sequence_length = 600
    train_loader, val_loader, state_identifier = prepare_hsmm_dataset(
        power_traces,
        tensor_parallelism,
        poisson_rate,
        model_size,
        sequence_length=sequence_length,
        n_states=6,
        batch_size=8,
        num_workers=4,
    )
    print(f"Train dataset samples: {len(train_loader.dataset)}")
    print(f"Val dataset samples:   {len(val_loader.dataset)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridHSMMTransformer(
        n_states=6,
        config_dim=3,  # [tensor_parallel, poisson_rate, model_size]
        d_model=64,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=sequence_length,
    )

    train_losses, val_losses = train_hybrid_model(
        model,
        train_loader,
        val_loader,
        num_epochs=20,
        lr=0.1,
        device=device,
        state_weight=0.5,
    )
    # model.load_state_dict(torch.load("hsmm_transformer_model.pt"))

    # Save the model
    torch.save(model.state_dict(), "hsmm_transformer_model.pt")

    # Plot the losses
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("hsmm_transformer_losses.pdf")

    def simulate_power_trace(model, loader, horizon=100, device="mps"):
        """
        Generate (simulate) a power trace from the HybridHSMMTransformer model.
        """

        # Put the model in eval mode, just in case
        model.eval()

        # Grab a single batch from the loader (or pick a specific sample)
        batch = next(iter(loader))  # get first batch from loader
        config = batch["config"][0].unsqueeze(0).to(device)  # [1, config_dim]

        # Only use first half of sequence as input
        seq_len = batch["input_seq"][0].size(0)
        half_seq_len = seq_len // 2
        input_seq = (
            batch["input_seq"][0][:half_seq_len].unsqueeze(0).to(device)
        )  # [1, seq_len/2]
        state_seq = (
            batch["state_seq"][0][:half_seq_len].unsqueeze(0).to(device)
        )  # [1, seq_len/2] (optional)

        # Use the model's predict_sequence method to roll forward "horizon" steps
        power_preds, state_preds = model.predict_sequence(
            config,
            input_seq,
            state_seq=state_seq,
            horizon=horizon,
        )
        # power_preds.shape = [1, horizon]
        # state_preds.shape = [1, horizon]

        # Let's also reconstruct the "seed" portion plus the predicted future
        # if you want to visualize a continuous trace from t=0 to t=seed_len + horizon
        seed_len = input_seq.size(1)
        # Convert everything to CPU numpy
        seed_power = input_seq[0].cpu().numpy()  # shape [seed_len]
        future_power = power_preds[0].detach().cpu().numpy()  # shape [horizon]
        simulated_power = np.concatenate([seed_power, future_power], axis=0)

        return simulated_power, seed_len

    model.load_state_dict(torch.load("hsmm_transformer_model.pt", map_location=device))
    model.eval()

    # Grab a seed sample from the data loader
    batch = next(iter(train_loader))  # or val_loader
    config = batch["config"][0].unsqueeze(0).to(device)  # shape [1, config_dim]

    # We'll seed with half the input_seq
    full_seq = batch["input_seq"][0]  # shape [seq_len]
    seed_len = full_seq.size(0) // 2
    init_power = full_seq[:seed_len].unsqueeze(0).to(device)  # shape [1, seed_len]

    # For states, do similarly
    full_states = batch["state_seq"][0]
    init_states = full_states[:seed_len].unsqueeze(0).to(device)  # shape [1, seed_len]

    # Now do closed-loop generation for horizon=300 steps
    horizon = 300
    full_power, full_states = predict_sequence_closed_loop(
        model,
        config,
        init_power,
        init_states,
        horizon=horizon,
        device=device,
    )
    # full_power: [1, seed_len + horizon]

    # Convert to numpy for plotting
    full_power_np = full_power[0].detach().cpu().numpy()
    full_states_np = full_states[0].detach().cpu().numpy()

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(full_power_np, label="Simulated Power Trace")
    plt.axvline(
        x=seed_len - 1, color="red", linestyle="--", label="End of seed sequence"
    )
    plt.title("Hybrid HSMM-Transformer: Closed-Loop Simulated Power Trace")
    plt.xlabel("Time Steps")
    plt.ylabel("Power")
    plt.legend()
    plt.show()

    # If you want to visualize states:
    plt.figure(figsize=(10, 2))
    plt.plot(full_states_np, label="Predicted States")
    plt.axvline(
        x=seed_len - 1, color="red", linestyle="--", label="End of seed sequence"
    )
    plt.title("HSMM Predicted States Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("State ID")
    plt.legend()
    plt.show()
