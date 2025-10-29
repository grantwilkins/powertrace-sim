import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import tqdm
import wandb
from classifiers.gru import GRUClassifier
from core.dataset import PowerTraceDataset
from torch.utils.data import DataLoader, Dataset


def train_classifiers(
    dataset: PowerTraceDataset,
    tp: int,
    hidden_size: int = 64,
    num_epochs: int = 500,
    lr: float = 5e-3,
    device: Optional[torch.device] = None,
    use_wandb: bool = True,
    val_split: float = 0.2,
):
    """Train a GRU classifier for a specific tensor parallelism value."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    print(f"Training classifier for TP={tp}")
    tp_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == tp]

    # Split into train and validation
    n_val = int(len(tp_indices) * val_split)
    val_indices = tp_indices[:n_val]
    train_indices = tp_indices[n_val:]

    train_dataset = TPDataset(dataset, train_indices)
    val_dataset = TPDataset(dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    x_sample, y_sample, z_sample = dataset[tp_indices[0]]
    Dx = x_sample.shape[1]
    K = len(torch.unique(z_sample))
    classifier = GRUClassifier(Dx, K, H=hidden_size).to(device)
    classifier.train()
    classifier.to(device)

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="powertrace-classifier-training",
            config={
                "tp": tp,
                "hidden_size": hidden_size,
                "num_epochs": num_epochs,
                "lr": lr,
                "device": str(device),
                "input_dim": Dx,
                "num_states": K,
                "train_samples": len(train_indices),
                "val_samples": len(val_indices),
            },
        )
        wandb.watch(classifier, log="all", log_freq=10)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    epoch_losses = []

    for epoch in range(num_epochs):
        # Training phase
        classifier.train()
        epoch_loss = 0.0
        batch_losses = []
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for x, y, z in progress_bar:
            x = x.to(device)
            z = z.to(device)

            optimizer.zero_grad()
            logits = classifier(x)
            loss = criterion(logits.view(-1, K), z.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()

            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            epoch_loss += batch_loss
            progress_bar.set_postfix({"loss": f"{batch_loss:.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_train_loss)

        # Validation phase
        classifier.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, z in val_loader:
                x = x.to(device)
                z = z.to(device)

                logits = classifier(x)
                loss = criterion(logits.view(-1, K), z.view(-1))
                val_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == z).sum().item()
                total += z.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        # Log to wandb
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy,
                }
            )

        if (epoch + 1) % 10 == 0:
            print(
                f"TP {tp}, Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )

    if use_wandb:
        wandb.finish()

    return classifier, epoch_losses


class TPDataset(Dataset):
    def __init__(self, parent_dataset: PowerTraceDataset, indices: list):
        self.parent = parent_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.parent[self.indices[idx]]
