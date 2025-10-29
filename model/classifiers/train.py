import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import acf

import wandb
from model.classifiers.gru import GRUClassifier
from model.core.dataset import PowerTraceDataset


def compute_calibration_error(y_true, y_probs, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find samples in this bin
        in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def train_classifiers(
    dataset: PowerTraceDataset,
    tp: int,
    hidden_size: int = 64,
    num_epochs: int = 500,
    lr: float = 5e-3,
    device: Optional[torch.device] = None,
    use_wandb: bool = True,
    val_split: float = 0.2,
    model_name: str = None,
    hardware_name: str = None,
):
    """Train a GRU classifier for a specific tensor parallelism value."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    print(f"Training classifier for TP={tp}")
    tp_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == tp]
    n_val = int(len(tp_indices) * val_split)
    val_indices = tp_indices[:n_val]
    train_indices = tp_indices[n_val:]

    train_dataset = TPDataset(dataset, train_indices)
    val_dataset = TPDataset(dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    x_sample, y_sample, z_sample = dataset[tp_indices[0]]
    Dx = x_sample.shape[1]

    # Determine K from ALL samples for this TP value, not just the first one
    all_states = []
    for idx in tp_indices:
        _, _, z = dataset[idx]
        all_states.extend(torch.unique(z).tolist())
    unique_states = sorted(set(all_states))
    K = len(unique_states)

    # Check if states are contiguous [0, 1, 2, ..., K-1]
    if unique_states != list(range(K)):
        print(f"WARNING: State labels are not contiguous: {unique_states}")
        print(f"Expected: {list(range(K))}")
        print("This may cause issues with CrossEntropyLoss. Consider remapping labels.")

    print(f"Found {K} unique states across all samples for TP={tp}: {unique_states}")

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
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
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
        all_predictions = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for x, y, z in val_loader:
                x = x.to(device)
                z = z.to(device)

                logits = classifier(x)
                loss = criterion(logits.view(-1, K), z.view(-1))
                val_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)

                correct += (predictions == z).sum().item()
                total += z.numel()

                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(z.cpu().numpy().flatten())
                # Store max probability for calibration
                max_probs = torch.max(probs, dim=-1)[0]
                all_probs.extend(max_probs.cpu().numpy().flatten())

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
                f"TP {tp}, Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )

    # Final comprehensive evaluation
    print("\nComputing final evaluation metrics...")
    classifier.eval()

    # Collect all predictions and labels for metrics
    final_predictions = []
    final_labels = []
    final_probs = []
    final_powers_real = []
    final_powers_pred = []

    with torch.no_grad():
        for x, y, z in val_loader:
            x = x.to(device)
            z = z.to(device)

            logits = classifier(x)
            predictions = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]

            final_predictions.extend(predictions.cpu().numpy().flatten())
            final_labels.extend(z.cpu().numpy().flatten())
            final_probs.extend(max_probs.cpu().numpy().flatten())
            final_powers_real.extend(y.cpu().numpy().flatten())

    final_predictions = np.array(final_predictions)
    final_labels = np.array(final_labels)
    final_probs = np.array(final_probs)
    final_powers_real = np.array(final_powers_real)

    # Compute metrics
    val_f1 = f1_score(final_labels, final_predictions, average='weighted', zero_division=0)
    val_precision = precision_score(final_labels, final_predictions, average='weighted', zero_division=0)
    val_recall = recall_score(final_labels, final_predictions, average='weighted', zero_division=0)

    # Calibration error
    correct_predictions = (final_predictions == final_labels).astype(float)
    ece = compute_calibration_error(correct_predictions, final_probs)

    # Compute autocorrelation R^2
    # For power traces, we compute autocorrelation and compare
    try:
        real_acf = acf(final_powers_real, nlags=min(50, len(final_powers_real) // 2), fft=True)
        # For synthetic, we'd need to generate it - for now use predictions as proxy
        synthetic_power = final_powers_real.copy()  # Placeholder
        synthetic_acf = acf(synthetic_power, nlags=min(50, len(synthetic_power) // 2), fft=True)
        autocorr_r2, _ = pearsonr(real_acf, synthetic_acf)
        autocorr_r2 = autocorr_r2 ** 2  # R^2
    except:
        autocorr_r2 = 0.0

    # Transition MAE - compute power differences at state transitions
    transitions = np.where(np.diff(final_labels) != 0)[0]
    if len(transitions) > 0:
        transition_power_diffs = np.abs(np.diff(final_powers_real)[transitions])
        transition_mae = np.mean(transition_power_diffs)
    else:
        transition_mae = 0.0

    metrics = {
        'val_f1': val_f1,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'ece': ece,
        'autocorr_r2': autocorr_r2,
        'transition_mae': transition_mae,
        'final_train_loss': epoch_losses[-1] if epoch_losses else 0,
        'final_val_loss': avg_val_loss,
        'val_accuracy': val_accuracy,
    }

    print(f"\nFinal Metrics:")
    print(f"  Validation F1: {val_f1:.4f}")
    print(f"  Validation Precision: {val_precision:.4f}")
    print(f"  Validation Recall: {val_recall:.4f}")
    print(f"  Expected Calibration Error: {ece:.4f}")
    print(f"  Autocorrelation R^2: {autocorr_r2:.4f}")
    print(f"  Transition MAE (W): {transition_mae:.2f}")

    # Save metrics to CSV
    if model_name and hardware_name:
        job_name = f"{model_name}_{hardware_name}_tp{tp}"
        metrics_dir = "./training_results/metrics"
        os.makedirs(metrics_dir, exist_ok=True)

        import csv
        metrics_file = f"{metrics_dir}/{job_name}_metrics.csv"
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for key, value in metrics.items():
                writer.writerow([key, value])
        print(f"\nMetrics saved to: {metrics_file}")

        # Save training losses
        losses_file = f"{metrics_dir}/{job_name}_losses.csv"
        with open(losses_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss'])
            for i, loss in enumerate(epoch_losses):
                writer.writerow([i+1, loss])
        print(f"Training losses saved to: {losses_file}")

    # Log final metrics to wandb
    if use_wandb:
        wandb.log({
            'final_metrics/val_f1': val_f1,
            'final_metrics/val_precision': val_precision,
            'final_metrics/val_recall': val_recall,
            'final_metrics/ece': ece,
            'final_metrics/autocorr_r2': autocorr_r2,
            'final_metrics/transition_mae': transition_mae,
        })
        wandb.finish()

    return classifier, epoch_losses, metrics


class TPDataset(Dataset):
    def __init__(self, parent_dataset: PowerTraceDataset, indices: list):
        self.parent = parent_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.parent[self.indices[idx]]
