#############################
# train_transformer.py
#############################

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm


########################################
# 1) Dataset / Data Loading
########################################


class PowerTraceDataset(Dataset):
    """
    Minimal dataset for sliding-window power-trace modeling.
    We assume the .npz file includes:
      - power_traces: shape [N, T]
      - tensor_parallelism: shape [N]
      - poisson_rate: shape [N]
      - valid_indices: shape [M, 2] or similar (list of (i, start_idx))
    For each valid index, we return:
      - config vector (tp, rate)
      - input_seq
      - target_seq (same as input_seq for auto-regressive tasks)
    """

    def __init__(
        self,
        power_traces: np.ndarray,  # [N, T]
        tensor_parallelism: np.ndarray,  # [N]
        poisson_rate: np.ndarray,  # [N]
        valid_indices: np.ndarray,  # shape [M, 2], each row = (sample_idx, start_idx)
        sequence_length: int,
    ):
        super().__init__()
        self.power_traces = power_traces
        self.tensor_parallelism = tensor_parallelism
        self.poisson_rate = poisson_rate
        self.valid_indices = valid_indices
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Which sample, and where to start
        sample_idx, start_idx = self.valid_indices[idx]
        # Slice out the power trace
        seq = self.power_traces[sample_idx][
            start_idx : start_idx + self.sequence_length
        ]
        seq = torch.tensor(seq, dtype=torch.float32)  # [seq_len]

        # For autoregressive training, we often provide input_seq and target_seq.
        # Here, input_seq is the same but shifted by 1. We'll do that in the model or
        # you can do it here. For simplicity, let's provide them both:
        input_seq = torch.cat([torch.zeros(1), seq[:-1]], dim=0)  # shift by 1
        target_seq = seq

        # Build config: we do a 2D config for (tensor_parallelism, poisson_rate).
        config_vec = torch.tensor(
            [self.tensor_parallelism[sample_idx], self.poisson_rate[sample_idx]],
            dtype=torch.float32,
        )

        return {
            "config": config_vec,  # shape [2]
            "input_seq": input_seq,  # shape [seq_len]
            "target_seq": target_seq,  # shape [seq_len]
        }


########################################
# 2) A Simple Transformer Model
########################################


class SimplePositionalEncoding(nn.Module):
    """Basic positional encoding for 1D sequences."""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, d_model, 2, dtype=torch.float)
            / d_model
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer so it's not a parameter but moves with device
        self.register_buffer("pe", pe.unsqueeze(0))  # shape [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        # Add positional encoding up to seq_len
        return x + self.pe[:, :seq_len, :]


class ConfigEncoder(nn.Module):
    """Simple MLP to embed 2D config => d_model."""

    def __init__(self, config_dim=2, d_model=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )

    def forward(self, config: torch.Tensor) -> torch.Tensor:
        # config shape: [batch_size, 2]
        return self.net(config)  # [batch_size, d_model]


class PowerTraceTransformer(nn.Module):
    """
    A minimal Transformer to predict next power value given the past,
    plus a config embedding as an additional "context" in the encoder.
    """

    def __init__(
        self,
        config_dim=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=600,
    ):
        super().__init__()

        # 1) Config -> embedding
        self.config_encoder = ConfigEncoder(config_dim, d_model)

        # 2) Input embedding for power values
        self.value_embedding = nn.Linear(1, d_model)

        # 3) Positional encoding
        self.pos_encoder = SimplePositionalEncoding(d_model, max_len=max_seq_len)

        # 4) Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # 5) Final projection from d_model -> 1 (predicted power)
        self.fc_out = nn.Linear(d_model, 1)

        self.d_model = d_model

    def forward(self, config, input_seq):
        """
        Args:
            config: shape [batch_size, 2]
            input_seq: shape [batch_size, seq_len], the AR input

        Returns:
            out: shape [batch_size, seq_len] (predictions)
        """
        bs, seq_len = input_seq.shape

        # 1) Encode config to produce "encoder memory" repeated across positions
        config_emb = self.config_encoder(config)  # [batch_size, d_model]
        # Suppose we want the encoder to produce a single "token." We'll treat that as shape [batch_size, 1, d_model].
        # Another approach is to tile it for each time step. We'll keep it simple: a single token.
        encoder_src = config_emb.unsqueeze(1)  # [batch_size, 1, d_model]
        print(config)
        # 2) Embed input_seq as shape [batch_size, seq_len, d_model]
        x = input_seq.unsqueeze(-1)  # [batch_size, seq_len, 1]
        x = self.value_embedding(x)  # => [batch_size, seq_len, d_model]
        x = self.pos_encoder(x)  # add positional encoding

        # 3) Because we only have a single "encoder token," we won't add more PE there.
        # We'll pass that single token as the "src" to the transformer, the input embedding as "tgt".
        # We also want to apply a casual mask so the model can't peek at future tokens.
        # PyTorch can generate a square subsequent mask:
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
            x.device
        )

        # 4) Run transformer
        # src: [batch_size, src_len=1, d_model]
        # tgt: [batch_size, seq_len, d_model]
        out = self.transformer(
            src=encoder_src, tgt=x, tgt_mask=causal_mask
        )  # => [batch_size, seq_len, d_model]
        print(out.shape)

        # 5) Projection to single value
        out = self.fc_out(out)  # => [batch_size, seq_len, 1]
        out = out.squeeze(-1)  # => [batch_size, seq_len]
        return out


########################################
# 3) Training Function
########################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConfigEncoder(nn.Module):
    """Turn a 2D config (tp, rate) into an embedding vector of size config_embed_dim."""

    def __init__(self, config_dim=2, config_embed_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config_dim, config_embed_dim),
            nn.ReLU(),
            nn.Linear(config_embed_dim, config_embed_dim),
        )

    def forward(self, config):
        # config: [batch_size, 2]
        return self.net(config)  # => [batch_size, config_embed_dim]


class LSTMPowerPredictor(nn.Module):
    """
    A simple LSTM-based model for autoregressive power trace prediction.
    We'll incorporate the config embedding into each time step's input.
    """

    def __init__(
        self,
        config_dim=2,
        config_embed_dim=16,
        input_embed_dim=16,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        # Encode the 2D config into a small embedding
        self.config_encoder = ConfigEncoder(config_dim, config_embed_dim)

        # Embedding for each power value (1D -> input_embed_dim)
        self.value_embedding = nn.Linear(1, input_embed_dim)

        # LSTM that processes each time step
        # We'll feed in (input_embed_dim + config_embed_dim) at each timestep
        self.lstm = nn.LSTM(
            input_size=input_embed_dim + config_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Final projection from LSTM hidden -> 1
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, config, input_seq):
        """
        Args:
            config: shape [batch_size, 2]
            input_seq: shape [batch_size, seq_len], the SHIFTED input for autoregression
        Returns:
            out: [batch_size, seq_len] - predicted power
        """
        batch_size, seq_len = input_seq.shape

        # 1) Encode config: shape [batch_size, config_embed_dim]
        config_emb = self.config_encoder(config)

        # 2) Embed the input power values => [batch_size, seq_len, input_embed_dim]
        x = input_seq.unsqueeze(-1)  # => [batch_size, seq_len, 1]
        x = self.value_embedding(x)  # => [batch_size, seq_len, input_embed_dim]

        # 3) Broadcast config_emb to each time step, then concatenate
        # config_emb: [batch_size, config_embed_dim] => expand to [batch_size, seq_len, config_embed_dim]
        config_emb_expanded = config_emb.unsqueeze(1).expand(-1, seq_len, -1)
        lstm_input = torch.cat([x, config_emb_expanded], dim=2)
        # => shape [batch_size, seq_len, input_embed_dim + config_embed_dim]

        # 4) Pass through LSTM
        lstm_output, _ = self.lstm(lstm_input)
        # => [batch_size, seq_len, hidden_size]

        # 5) Project to a single value per time step
        out = self.fc_out(lstm_output)  # => [batch_size, seq_len, 1]
        out = out.squeeze(-1)  # => [batch_size, seq_len]
        return out


def train_transformer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "mps",
):
    """Simple training loop for MSE regression."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # ---- TRAIN ----
        model.train()
        epoch_train_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_pbar:
            # print(batch)
            config = batch["config"].to(device)  # [batch_size, 2]
            input_seq = batch["power_trace"].to(device)  # [batch_size, seq_len]
            target_seq = batch["input_trace"].to(device)

            # Forward
            pred_seq = model(config, input_seq)

            # Compute loss
            loss = criterion(pred_seq, target_seq)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            epoch_train_loss += loss.item()
            train_pbar.set_postfix({"loss": loss.item()})

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        # ---- VALIDATION (optional) ----
        if val_loader is not None:
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                for batch in val_pbar:
                    config = batch["config"].to(device)
                    input_seq = batch["power_trace"].to(device)
                    target_seq = batch["input_trace"].to(device)

                    pred_seq = model(config, input_seq)
                    loss = criterion(pred_seq, target_seq)
                    epoch_val_loss += loss.item()
                    val_pbar.set_postfix({"loss": loss.item()})

            epoch_val_loss /= len(val_loader)
            val_losses.append(epoch_val_loss)
            print(
                f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}"
            )
        else:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f}")

    return train_losses, val_losses


def variable_length_collate_fn(batch):
    """
    Custom collate function for the PowerTraceDataset to handle
    variable-length sequences by padding them to the same length
    within each batch.

    Each 'item' in the batch is a dict with:
      {
        "config": Tensor([2]),
        "power_trace": Tensor([seq_len]),
        "input_trace": Tensor([seq_len]),
      }

    This function:
      1) Finds the max seq_len among items in the batch.
      2) Pads all sequences to that length.
      3) Stacks them into a single batch of tensors.

    Returns a dict:
      {
        "config": Tensor([batch_size, 2]),
        "power_trace": Tensor([batch_size, max_len]),
        "input_trace": Tensor([batch_size, max_len]),
      }
    """
    # 1) Extract all fields into lists
    # print(batch)
    configs = [item["config"] for item in batch]
    power_traces = [item["input_seq"] for item in batch]
    input_traces = [item["target_seq"] for item in batch]

    # 2) Determine the max sequence length in this batch
    max_len = max(p.shape[0] for p in power_traces)

    # 3) Collate 'config' by stacking (they should already be the same shape, e.g. [2])
    config_batch = torch.stack(configs, dim=0)  # [batch_size, 2]

    # 4) Pad each power_trace and input_trace to 'max_len'
    padded_power = []
    padded_input = []
    for i in range(len(power_traces)):
        seq_len = power_traces[i].shape[0]
        pad_len = max_len - seq_len

        # Pad right side with zeros (for a 1D tensor, pad takes (left_pad, right_pad))
        power_padded = F.pad(power_traces[i], (0, pad_len), value=0.0)
        input_padded = F.pad(input_traces[i], (0, pad_len), value=0.0)

        padded_power.append(power_padded)
        padded_input.append(input_padded)

    # 5) Stack into [batch_size, max_len]
    power_batch = torch.stack(padded_power, dim=0)
    input_batch = torch.stack(padded_input, dim=0)

    return {
        "config": config_batch,
        "power_trace": power_batch,
        "input_trace": input_batch,
    }


########################################
# 4) Demo: Loading Data, Training, Etc.
########################################

if __name__ == "__main__":
    # 1) Load NPZ data
    data = np.load("processed_data/power_trace_data.npz")
    power_traces = data["power_traces"]  # shape [N, T]
    tp = data["tensor_parallelism"]  # shape [N]
    rate = data["poisson_rate"]  # shape [N]
    valid_indices = data["valid_indices"]  # shape [M, 2]
    seq_len = power_traces.shape[
        1
    ]  # or use a smaller subset if you used a shorter sliding window

    # # 2) Create dataset & dataloaders
    dataset = PowerTraceDataset(
        power_traces=power_traces,
        tensor_parallelism=tp,
        poisson_rate=rate,
        valid_indices=valid_indices,
        sequence_length=seq_len,
    )

    # Simple 80/20 train/val split
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=variable_length_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, collate_fn=variable_length_collate_fn
    )

    # print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # # model = PowerTraceTransformer(
    # #     config_dim=2,
    # #     d_model=64,
    # #     nhead=4,
    # #     num_layers=2,
    # #     dim_feedforward=256,
    # #     dropout=0.1,
    # #     max_seq_len=seq_len,
    # # )

    model = LSTMPowerPredictor(
        config_dim=2,
        config_embed_dim=16,
        input_embed_dim=16,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
    )
    # model.load_state_dict(torch.load("model.pth"))
    train_losses, val_losses = train_transformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        lr=1e-1,
    )
    torch.save(model.state_dict(), "model.pth")

    # 4) Train
    # train_losses, val_losses = train_transformer(
    #     model, train_loader, val_loader, num_epochs=10, lr=1e-5
    # )

    # 5) Plot training curves (optional)
    import matplotlib.pyplot as plt

    plt.plot(train_losses, label="Train Loss")
    if val_losses:
        plt.plot(val_losses, label="Val Loss")
    plt.title("Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

    # 6) (Optional) Predict / Generate a few sequences
    # For "generation," let's just pass the same input back to see if it reconstructs it
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))  # get a random batch from val
        config = batch["config"]
        input_seq = batch["power_trace"]
        target_seq = batch["input_trace"]

        # Move to device
        device = next(model.parameters()).device
        config = config.to(device)
        input_seq = input_seq.to(device)

        pred_seq = model(config, input_seq)  # shape [bs, seq_len]
        # Compare pred_seq with target_seq...
        # plt.plot(target_seq[0].cpu(), label="Target")
        print(pred_seq[0].cpu())
        plt.plot(pred_seq[0].cpu(), label="Predicted")
        plt.legend()
        plt.show()
        print("Predicted shape:", pred_seq.shape)

    print("Done!")
