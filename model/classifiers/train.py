import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import tqdm
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
):
    """Train a GRU classifier for a specific tensor parallelism value."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    print(f"Training classifier for TP={tp}")
    tp_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == tp]

    tp_dataset = TPDataset(dataset, tp_indices)
    loader = DataLoader(tp_dataset, batch_size=1, shuffle=True)
    x_sample, y_sample, z_sample = dataset[tp_indices[0]]
    Dx = x_sample.shape[1]
    K = len(torch.unique(z_sample))
    classifier = GRUClassifier(Dx, K, H=hidden_size).to(device)
    classifier.train()
    classifier.to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    epoch_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_losses = []
        progress_bar = tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
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

        avg_loss = epoch_loss / len(loader)
        epoch_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"TP {tp}, Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    return classifier, epoch_losses


class TPDataset(Dataset):
    def __init__(self, parent_dataset: PowerTraceDataset, indices: list):
        self.parent = parent_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.parent[self.indices[idx]]
