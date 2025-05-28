import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader, Dataset

from model.classifiers.gru import GRUClassifier


def train_classifiers(dataset, hidden_size=64, device=None, classifiers=None):
    """Train separate GRU classifiers for each tensor parallelism value."""
    if classifiers is None:
        classifiers = {}
    training_losses = {}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    for tp in sorted(set(dataset.tp_all)):
        print(f"Training classifier for TP={tp}")
        tp_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == tp]

        class TPDataset(Dataset):
            def __init__(self, parent_dataset, indices):
                self.parent = parent_dataset
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                return self.parent[self.indices[idx]]

        tp_dataset = TPDataset(dataset, tp_indices)
        loader = DataLoader(tp_dataset, batch_size=1, shuffle=True)
        x_sample, y_sample, z_sample = dataset[tp_indices[0]]
        Dx = x_sample.shape[1]
        K = len(torch.unique(z_sample))
        if tp not in classifiers:
            classifier = GRUClassifier(Dx, K, H=hidden_size).to(device)
        else:
            classifier = classifiers[tp].to(device)
        classifier.train()
        classifier.to(device)

        optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-3)
        criterion = nn.CrossEntropyLoss()

        epoch_losses = []

        for epoch in range(500):
            epoch_loss = 0.0
            batch_losses = []
            progress_bar = tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/500")
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

        classifiers[tp] = classifier
        training_losses[tp] = epoch_losses
        np.save(f"training_losses_tp{tp}.npy", np.array(epoch_losses))
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_losses)
        plt.title(f"Training Loss for TP={tp}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(f"training_loss_tp{tp}.pdf")
        plt.close()

        torch.save(
            classifier.state_dict(),
            f"classifier_llama3_8b_a100_tp{tp}.pt",
        )

    return classifiers, training_losses
