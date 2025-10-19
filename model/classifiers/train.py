import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from classifiers.gru import GRUClassifier
from core.dataset import PowerTraceDataset
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, random_split


def compute_class_weights(
    tp_dataset: Dataset, K: int, device: torch.device
) -> torch.Tensor:
    """Compute inverse frequency class weights for handling class imbalance."""
    all_labels = []
    for i in range(len(tp_dataset)):
        _, _, z = tp_dataset[i]
        all_labels.append(z.numpy())
    all_labels = np.concatenate(all_labels)

    class_counts = np.bincount(all_labels, minlength=K)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * K

    return torch.FloatTensor(weights).to(device)


def evaluate_classifier(
    classifier: nn.Module,
    loader: DataLoader,
    K: int,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float, float, np.ndarray]:
    """Evaluate classifier and return loss, accuracy, macro-F1, and confusion matrix."""
    classifier.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y, z in loader:
            x = x.to(device)
            z = z.to(device)

            logits = classifier(x)
            loss = criterion(logits.view(-1, K), z.view(-1))
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1).cpu().numpy().flatten()
            labels = z.cpu().numpy().flatten()

            all_preds.extend(preds)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    avg_loss = total_loss / len(loader)
    accuracy = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(K)))

    return avg_loss, accuracy, macro_f1, conf_matrix


def lr_sweep(
    dataset: PowerTraceDataset,
    tp: int,
    lr_range: List[float] = None,
    num_epochs: int = 15,
    val_split: float = 0.15,
    hidden_size: int = 64,
    device: Optional[torch.device] = None,
    output_dir: str = "./lr_sweep_results",
    seed: int = 42,
) -> Tuple[float, Dict]:
    """
    Perform LR sweep with constant learning rates.

    Args:
        dataset: PowerTraceDataset
        tp: Tensor parallelism value
        lr_range: List of learning rates to try (defaults to log-space 3e-4 to 3e-3)
        num_epochs: Number of epochs for each LR trial
        val_split: Fraction of data for validation
        hidden_size: Hidden size for GRU
        device: Device to train on
        output_dir: Directory to save results
        seed: Random seed

    Returns:
        best_lr: Best learning rate based on validation macro-F1
        results: Dictionary with detailed results for each LR
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if lr_range is None:
        # Default: 7 values in log-space from 3e-4 to 3e-3
        lr_range = np.logspace(np.log10(3e-4), np.log10(3e-3), 7).tolist()

    os.makedirs(output_dir, exist_ok=True)

    print(f"LR Sweep for TP={tp}")
    print(f"LR range: {[f'{lr:.2e}' for lr in lr_range]}")

    # Prepare dataset
    tp_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == tp]
    tp_dataset = TPDataset(dataset, tp_indices)

    # Get K and input dimension
    x_sample, _, z_sample = dataset[tp_indices[0]]
    Dx = x_sample.shape[1]
    all_z = torch.cat([dataset[i][2] for i in tp_indices])
    K = len(torch.unique(all_z))

    # Compute class weights
    class_weights = compute_class_weights(tp_dataset, K, device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Split into train/val
    torch.manual_seed(seed)
    val_size = int(len(tp_dataset) * val_split)
    train_size = len(tp_dataset) - val_size
    train_dataset, val_dataset = random_split(tp_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    results = {}
    best_lr = None
    best_f1 = -1.0

    for lr in lr_range:
        print(f"\n{'=' * 60}")
        print(f"Testing LR = {lr:.2e}")
        print(f"{'=' * 60}")

        torch.manual_seed(seed)
        classifier = GRUClassifier(Dx, K, H=hidden_size).to(device)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        train_losses = []
        val_losses = []
        val_f1s = []

        for epoch in range(num_epochs):
            # Training
            classifier.train()
            epoch_loss = 0.0
            for x, y, z in train_loader:
                x = x.to(device)
                z = z.to(device)

                optimizer.zero_grad()
                logits = classifier(x)
                loss = criterion(logits.view(-1, K), z.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            train_losses.append(epoch_loss / len(train_loader))

            # Validation
            val_loss, val_acc, val_f1, _ = evaluate_classifier(
                classifier, val_loader, K, device, criterion
            )
            val_losses.append(val_loss)
            val_f1s.append(val_f1)

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch + 1:02d}: train_loss={train_losses[-1]:.4f}, "
                    f"val_loss={val_loss:.4f}, val_f1={val_f1:.4f}"
                )

        final_f1 = val_f1s[-1]
        results[lr] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_f1s": val_f1s,
            "final_f1": final_f1,
        }

        print(f"Final validation F1: {final_f1:.4f}")

        if final_f1 > best_f1:
            best_f1 = final_f1
            best_lr = lr

    print(f"\n{'=' * 60}")
    print(f"Best LR: {best_lr:.2e} (val F1 = {best_f1:.4f})")
    print(f"{'=' * 60}")

    # Save sweep results
    np.save(os.path.join(output_dir, f"lr_sweep_tp{tp}.npy"), results)

    # Save to CSV
    save_lr_sweep_to_csv(results, output_dir, tp)

    return best_lr, results


def save_lr_sweep_to_csv(results: Dict, output_dir: str, tp: int):
    """Save LR sweep results to CSV."""
    # Summary CSV
    summary_data = []
    for lr, metrics in results.items():
        summary_data.append({
            'learning_rate': lr,
            'final_val_f1': metrics['final_f1'],
            'best_val_f1': max(metrics['val_f1s']),
            'final_train_loss': metrics['train_losses'][-1],
            'final_val_loss': metrics['val_losses'][-1],
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(
        os.path.join(output_dir, f'lr_sweep_summary_tp{tp}.csv'),
        index=False
    )

    # Per-epoch CSV for each LR
    for lr, metrics in results.items():
        epoch_data = []
        for epoch in range(len(metrics['train_losses'])):
            epoch_data.append({
                'epoch': epoch + 1,
                'train_loss': metrics['train_losses'][epoch],
                'val_loss': metrics['val_losses'][epoch],
                'val_f1': metrics['val_f1s'][epoch],
            })

        epoch_df = pd.DataFrame(epoch_data)
        lr_str = f"{lr:.2e}".replace('.', 'p').replace('-', 'm')
        epoch_df.to_csv(
            os.path.join(output_dir, f'lr_sweep_tp{tp}_lr{lr_str}_epochs.csv'),
            index=False
        )


def train_classifiers(
    dataset: PowerTraceDataset,
    tp: int,
    hidden_size: int = 64,
    num_epochs: int = 500,
    lr: float = 5e-3,
    device: Optional[torch.device] = None,
    use_scheduler: bool = True,
    scheduler_type: str = "cosine",
    val_split: float = 0.15,
    seed: int = 42,
    output_dir: str = "./training_results",
    use_class_weights: bool = True,
) -> Tuple[nn.Module, Dict]:
    """
    Train a GRU classifier for a specific tensor parallelism value.

    Args:
        dataset: PowerTraceDataset
        tp: Tensor parallelism value
        hidden_size: Hidden size for GRU
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        use_scheduler: Whether to use learning rate scheduler
        scheduler_type: Type of scheduler ('cosine' or 'onecycle')
        val_split: Fraction of data for validation
        seed: Random seed
        output_dir: Directory to save results
        use_class_weights: Whether to use class weighting for imbalanced data

    Returns:
        classifier: Trained classifier
        metrics: Dictionary with training metrics and results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    print(f"Training classifier for TP={tp}")
    tp_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == tp]

    tp_dataset = TPDataset(dataset, tp_indices)
    x_sample, y_sample, z_sample = dataset[tp_indices[0]]
    Dx = x_sample.shape[1]
    all_z = torch.cat([dataset[i][2] for i in tp_indices])
    K = len(torch.unique(all_z))
    print(f"Number of unique states across all TP={tp} samples: {K}")
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(tp_dataset, K, device)
        print(f"Class weights: {class_weights.cpu().numpy()}")
    torch.manual_seed(seed)
    val_size = int(len(tp_dataset) * val_split)
    train_size = len(tp_dataset) - val_size
    train_dataset, val_dataset = random_split(tp_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"Train size: {train_size}, Val size: {val_size}")

    classifier = GRUClassifier(Dx, K, H=hidden_size).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Setup scheduler
    scheduler = None
    if use_scheduler:
        total_steps = num_epochs * len(train_loader)
        if scheduler_type == "cosine":
            warmup_steps = int(0.05 * total_steps)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=total_steps - warmup_steps, eta_min=lr * 0.01
            )
        elif scheduler_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.05
            )

    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []
    learning_rates = []

    for epoch in range(num_epochs):
        classifier.train()
        epoch_loss = 0.0
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

            if scheduler is not None:
                scheduler.step()
                learning_rates.append(optimizer.param_groups[0]["lr"])

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_losses.append(epoch_loss / len(train_loader))

        # Validation
        val_loss, val_acc, val_f1, conf_mat = evaluate_classifier(
            classifier, val_loader, K, device, criterion
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        if (epoch + 1) % 10 == 0:
            print(
                f"TP {tp}, Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
            )

    final_val_loss, final_val_acc, final_val_f1, final_conf_mat = evaluate_classifier(
        classifier, val_loader, K, device, criterion
    )

    print(f"\nFinal Results for TP={tp}:")
    print(f"  Val Loss: {final_val_loss:.4f}")
    print(f"  Val Accuracy: {final_val_acc:.4f}")
    print(f"  Val Macro-F1: {final_val_f1:.4f}")

    # Save training results
    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "val_f1s": val_f1s,
        "final_val_loss": final_val_loss,
        "final_val_acc": final_val_acc,
        "final_val_f1": final_val_f1,
        "confusion_matrix": final_conf_mat,
        "learning_rates": learning_rates if scheduler else None,
    }

    # Save per-epoch metrics to CSV
    save_training_metrics_to_csv(metrics, output_dir, tp, lr, seed)

    return classifier, metrics


def save_training_metrics_to_csv(metrics: Dict, output_dir: str, tp: int, lr: float, seed: int):
    """Save training metrics to CSV file."""
    epoch_data = []
    for epoch in range(len(metrics['train_losses'])):
        row = {
            'epoch': epoch + 1,
            'train_loss': metrics['train_losses'][epoch],
            'val_loss': metrics['val_losses'][epoch],
            'val_acc': metrics['val_accs'][epoch],
            'val_f1': metrics['val_f1s'][epoch],
        }
        # Add learning rate if available
        if metrics['learning_rates'] and len(metrics['learning_rates']) > 0:
            # Average LR for the epoch (multiple steps per epoch)
            epoch_start = epoch * (len(metrics['learning_rates']) // len(metrics['train_losses']))
            epoch_end = (epoch + 1) * (len(metrics['learning_rates']) // len(metrics['train_losses']))
            row['avg_lr'] = np.mean(metrics['learning_rates'][epoch_start:epoch_end])

        epoch_data.append(row)

    epoch_df = pd.DataFrame(epoch_data)
    lr_str = f"{lr:.2e}".replace('.', 'p').replace('-', 'm')
    epoch_df.to_csv(
        os.path.join(output_dir, f'training_tp{tp}_lr{lr_str}_seed{seed}_epochs.csv'),
        index=False
    )

    # Save summary metrics
    summary_data = {
        'tp': tp,
        'learning_rate': lr,
        'seed': seed,
        'final_train_loss': metrics['train_losses'][-1],
        'final_val_loss': metrics['final_val_loss'],
        'final_val_acc': metrics['final_val_acc'],
        'final_val_f1': metrics['final_val_f1'],
        'best_val_f1': max(metrics['val_f1s']),
        'num_epochs': len(metrics['train_losses']),
    }

    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(
        os.path.join(output_dir, f'training_tp{tp}_lr{lr_str}_seed{seed}_summary.csv'),
        index=False
    )


def multi_seed_training(
    dataset: PowerTraceDataset, tp: int, lr: float, num_seeds: int = 3, **kwargs
) -> Dict:
    """
    Train with multiple seeds and report mean ± std.

    Args:
        dataset: PowerTraceDataset
        tp: Tensor parallelism value
        lr: Learning rate to use
        num_seeds: Number of seeds to run
        **kwargs: Additional arguments for train_classifiers

    Returns:
        results: Dictionary with aggregated results across seeds
    """
    all_val_f1s = []
    all_val_accs = []
    all_classifiers = []

    for seed in range(num_seeds):
        print(f"\n{'=' * 60}")
        print(f"Training with seed {seed}")
        print(f"{'=' * 60}")

        classifier, metrics = train_classifiers(dataset, tp, lr=lr, seed=seed, **kwargs)

        all_val_f1s.append(metrics["final_val_f1"])
        all_val_accs.append(metrics["final_val_acc"])
        all_classifiers.append(classifier)

    results = {
        "mean_f1": np.mean(all_val_f1s),
        "std_f1": np.std(all_val_f1s),
        "mean_acc": np.mean(all_val_accs),
        "std_acc": np.std(all_val_accs),
        "all_f1s": all_val_f1s,
        "all_accs": all_val_accs,
        "classifiers": all_classifiers,
    }

    print(f"\n{'=' * 60}")
    print(f"Multi-seed Results (n={num_seeds}):")
    print(f"  Macro-F1: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    print(f"  Accuracy: {results['mean_acc']:.4f} ± {results['std_acc']:.4f}")
    print(f"{'=' * 60}")

    return results


class TPDataset(Dataset):
    def __init__(self, parent_dataset: PowerTraceDataset, indices: list):
        self.parent = parent_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.parent[self.indices[idx]]
