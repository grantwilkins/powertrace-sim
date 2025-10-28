import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, random_split

import wandb
from model.classifiers.gru import GRUClassifier
from model.core.dataset import PowerTraceDataset


def build_cosine_with_warmup(optimizer, total_steps: int, warmup_pct: float = 0.05):
    """
    Build a cosine learning rate scheduler with linear warmup.

    Args:
        optimizer: PyTorch optimizer
        total_steps: Total number of training steps
        warmup_pct: Percentage of total_steps for warmup (default 0.05 = 5%)

    Returns:
        LambdaLR scheduler
    """
    warmup_steps = max(1, int(total_steps * warmup_pct))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_class_weights(
    tp_dataset: Dataset, K: int, device: torch.device
) -> torch.Tensor:
    """Compute inverse frequency class weights for handling class imbalance."""
    all_labels = []
    for i in range(len(tp_dataset)):
        _, _, z = tp_dataset[i]
        all_labels.append(z.numpy())
    all_labels = np.concatenate(all_labels)
    counts = np.bincount(all_labels, minlength=K)
    weights = np.ones(K, dtype=np.float32)
    for i in range(K):
        if counts[i] > 0:
            weights[i] = 1.0 / counts[i]

    present_classes = counts > 0
    num_present = present_classes.sum()
    if num_present > 0:
        sum_present_weights = weights[present_classes].sum()
        weights[present_classes] = weights[present_classes] * (
            num_present / sum_present_weights
        )

    return torch.FloatTensor(weights).to(device)


def compute_transition_mae(
    y_pred: np.ndarray, y_true: np.ndarray, x: np.ndarray, threshold: float = 0.1
) -> float:
    """
    Compute MAE at transition points (prefill→decode boundaries).

    Transitions are detected when prefill_tokens (x[:, 0]) drops significantly
    or decode_tokens (x[:, 1]) rises significantly.

    Args:
        y_pred: Predicted power values (N, T)
        y_true: True power values (N, T)
        x: Input features (N, T, Dx) where x[:,:,0] is prefill_tokens
        threshold: Relative change threshold to detect transitions

    Returns:
        MAE at transition points
    """
    transition_errors = []

    for i in range(len(y_pred)):
        prefill = x[i, :, 0]
        decode = x[i, :, 1]

        prefill_diff = np.abs(np.diff(prefill))
        decode_diff = np.abs(np.diff(decode))
        transition_mask = (prefill_diff > threshold * np.max(prefill + 1e-6)) | (
            decode_diff > threshold * np.max(decode + 1e-6)
        )

        if transition_mask.sum() > 0:
            transition_indices = np.where(transition_mask)[0] + 1
            errors = np.abs(
                y_pred[i, transition_indices] - y_true[i, transition_indices]
            )
            transition_errors.extend(errors)

    return np.mean(transition_errors) if len(transition_errors) > 0 else 0.0


def compute_autocorr_r2(
    y_pred: np.ndarray, y_true: np.ndarray, max_lag: int = 128
) -> float:
    def autocorr_at_lag(x, lag):
        x_mean = np.mean(x)
        x_std = np.std(x)

        if x_std < 1e-8:
            return 0.0
        x_norm = (x - x_mean) / x_std
        n = len(x_norm)
        if lag >= n:
            return 0.0

        numerator = np.sum(x_norm[:-lag] * x_norm[lag:])
        denominator = n - lag

        return numerator / denominator if denominator > 0 else 0.0

    min_length = max_lag + 4
    valid_indices = [i for i in range(len(y_pred)) if len(y_pred[i]) >= min_length]

    if len(valid_indices) == 0:
        return 0.0

    pred_acf_by_lag = {lag: [] for lag in range(1, max_lag + 1)}
    true_acf_by_lag = {lag: [] for lag in range(1, max_lag + 1)}

    for i in valid_indices:
        for lag in range(1, max_lag + 1):
            pred_r = autocorr_at_lag(y_pred[i], lag)
            true_r = autocorr_at_lag(y_true[i], lag)
            pred_acf_by_lag[lag].append(pred_r)
            true_acf_by_lag[lag].append(true_r)

    pred_acf_agg = []
    true_acf_agg = []

    for lag in range(1, max_lag + 1):
        pred_z = [np.arctanh(np.clip(r, -0.999, 0.999)) for r in pred_acf_by_lag[lag]]
        true_z = [np.arctanh(np.clip(r, -0.999, 0.999)) for r in true_acf_by_lag[lag]]
        pred_z_mean = np.mean(pred_z)
        true_z_mean = np.mean(true_z)
        pred_acf_agg.append(np.tanh(pred_z_mean))
        true_acf_agg.append(np.tanh(true_z_mean))

    pred_acf_agg = np.array(pred_acf_agg)
    true_acf_agg = np.array(true_acf_agg)
    ss_res = np.sum((true_acf_agg - pred_acf_agg) ** 2)
    ss_tot = np.sum((true_acf_agg - np.mean(true_acf_agg)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return max(0.0, r2)


def tolerance_power_accuracy(
    pred_z: np.ndarray, true_z: np.ndarray, mu: np.ndarray, tol_w: float = 25.0
) -> float:
    """
    Compute accuracy within a tolerance window in watts.
    More forgiving than exact state ID matching when GMM states overlap.

    Args:
        pred_z: Predicted state IDs (N, T)
        true_z: True state IDs (N, T)
        mu: State power means (K,)
        tol_w: Tolerance in watts
    """
    pw_pred = mu[pred_z.astype(int)]
    pw_true = mu[true_z.astype(int)]
    return float(np.mean(np.abs(pw_pred - pw_true) <= tol_w))


def boundary_f1(pred_z: np.ndarray, true_z: np.ndarray) -> float:
    """
    F1 score for transition boundary detection.
    Evaluates how well the model captures state changes.

    Args:
        pred_z: Predicted state IDs (N, T)
        true_z: True state IDs (N, T)
    """
    if pred_z.shape[1] <= 1:
        return 0.0

    b_pred = np.diff(pred_z, axis=1) != 0
    b_true = np.diff(true_z, axis=1) != 0
    tp = np.logical_and(b_pred, b_true).sum()
    fp = np.logical_and(b_pred, ~b_true).sum()
    fn = np.logical_and(~b_pred, b_true).sum()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(f1)


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Measures how well predicted probabilities match actual accuracy.

    Args:
        probs: Softmax probabilities (N*T, K)
        labels: True labels (N*T,)
        n_bins: Number of bins for calibration curve
    """
    conf = probs.max(1)
    preds = probs.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    m = len(labels)

    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i + 1])
        if mask.any():
            acc = np.mean(preds[mask] == labels[mask])
            avg_conf = np.mean(conf[mask])
            ece_val += np.abs(acc - avg_conf) * np.sum(mask) / m

    return float(ece_val)


def evaluate_classifier(
    classifier: nn.Module,
    loader: DataLoader,
    K: int,
    device: torch.device,
    criterion: nn.Module,
    compute_extra_metrics: bool = False,
    state_power_means: Optional[np.ndarray] = None,
    use_viterbi: bool = False,
    viterbi_penalty: float = 1.0,
) -> Dict[str, any]:
    """
    Evaluate classifier and return comprehensive metrics.

    Args:
        use_viterbi: If True, apply Viterbi decoding to smooth predictions
        viterbi_penalty: Transition penalty for Viterbi (higher = smoother)

    Returns dict with keys: loss, accuracy, balanced_accuracy, macro_f1,
    confusion_matrix, and optionally: transition_mae, autocorr_r2,
    tolerance_power_acc, boundary_f1, ece
    If use_viterbi=True, also includes viterbi_* versions of metrics.
    """
    from model.core.utils import viterbi_decode

    classifier.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_logits = []
    all_x = []
    all_y = []

    with torch.no_grad():
        for x, y, z in loader:
            x = x.to(device)
            z = z.to(device)

            logits = classifier(x)
            loss = criterion(logits.view(-1, K), z.view(-1))
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy().flatten()
            labels = z.cpu().numpy().flatten()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.append(probs.reshape(-1, K))
            all_logits.append(logits.cpu().numpy())  # Store for Viterbi

            if compute_extra_metrics:
                all_x.append(x.cpu().numpy())
                all_y.append(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)
    all_logits_cat = np.concatenate(all_logits, axis=0)  # (B, T, K)

    # Base metrics (independent predictions)
    avg_loss = total_loss / len(loader)
    accuracy = (all_preds == all_labels).mean()
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(K)))
    ece_val = expected_calibration_error(all_probs, all_labels)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "confusion_matrix": conf_matrix,
        "ece": ece_val,
    }

    # Viterbi-decoded metrics (sequence smoothing)
    if use_viterbi:
        # Apply Viterbi decoding
        viterbi_preds = viterbi_decode(
            all_logits_cat, transition_penalty=viterbi_penalty
        )
        viterbi_preds_flat = viterbi_preds.flatten()

        viterbi_acc = (viterbi_preds_flat == all_labels).mean()
        viterbi_balanced_acc = balanced_accuracy_score(all_labels, viterbi_preds_flat)
        viterbi_f1 = f1_score(
            all_labels, viterbi_preds_flat, average="macro", zero_division=0
        )

        metrics["viterbi_accuracy"] = viterbi_acc
        metrics["viterbi_balanced_accuracy"] = viterbi_balanced_acc
        metrics["viterbi_macro_f1"] = viterbi_f1

    # Extended metrics
    if compute_extra_metrics and len(all_x) > 0:
        all_x = np.concatenate(all_x, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        all_preds_2d = all_preds.reshape(all_y.shape).squeeze(-1)
        all_labels_2d = all_labels.reshape(all_y.shape).squeeze(-1)

        # Boundary F1 (sequence-aware)
        metrics["boundary_f1"] = boundary_f1(all_preds_2d, all_labels_2d)

        # Viterbi boundary F1 if using Viterbi
        if use_viterbi:
            viterbi_preds_2d = viterbi_preds.reshape(all_y.shape).squeeze(-1)
            metrics["viterbi_boundary_f1"] = boundary_f1(viterbi_preds_2d, all_labels_2d)

        # Convert state IDs to power values if means are provided
        if state_power_means is not None:

            def states_to_power(states_2d, mu):
                """Convert state IDs to power values using GMM means."""
                return mu[states_2d.astype(int)]

            y_pred_power = states_to_power(all_preds_2d, state_power_means)
            y_true_power = states_to_power(all_labels_2d, state_power_means)

            metrics["transition_mae"] = compute_transition_mae(y_pred_power, y_true_power, all_x)
            metrics["autocorr_r2"] = compute_autocorr_r2(y_pred_power, y_true_power)
            metrics["tolerance_power_acc"] = tolerance_power_accuracy(
                all_preds_2d, all_labels_2d, state_power_means, tol_w=25.0
            )

            # Viterbi versions
            if use_viterbi:
                y_viterbi_power = states_to_power(viterbi_preds_2d, state_power_means)
                metrics["viterbi_transition_mae"] = compute_transition_mae(
                    y_viterbi_power, y_true_power, all_x
                )
                metrics["viterbi_autocorr_r2"] = compute_autocorr_r2(
                    y_viterbi_power, y_true_power
                )
                metrics["viterbi_tolerance_power_acc"] = tolerance_power_accuracy(
                    viterbi_preds_2d, all_labels_2d, state_power_means, tol_w=25.0
                )
        else:
            # Fallback: use state IDs as proxy for power
            metrics["transition_mae"] = compute_transition_mae(all_preds_2d, all_labels_2d, all_x)
            metrics["autocorr_r2"] = compute_autocorr_r2(all_preds_2d, all_labels_2d)

    return metrics


def lr_sweep(
    dataset: PowerTraceDataset,
    tp: int,
    lr_range: List[float] = None,
    num_epochs: int = 100,
    val_split: float = 0.15,
    hidden_size: int = 64,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    output_dir: str = "./lr_sweep_results",
    seed: int = 42,
    wandb_project: Optional[str] = None,
    wandb_entity: str = "grantfwilkins-stanford-university",
    wandb_run_prefix: str = "",
    bidirectional: bool = False,
) -> Tuple[float, Dict]:
    """
    Perform LR sweep with constant learning rates.

    Args:
        dataset: PowerTraceDataset
        tp: Tensor parallelism value
        lr_range: List of learning rates to try (hardcoded default)
        num_epochs: Number of epochs for each LR trial
        val_split: Fraction of data for validation
        hidden_size: Hidden size for GRU
        device: Device to train on
        output_dir: Directory to save results
        seed: Random seed
        wandb_project: WandB project name
        wandb_entity: WandB entity/team name
        wandb_run_prefix: Prefix for wandb run name
        bidirectional: Whether to use bidirectional GRU

    Returns:
        best_lr: Best learning rate based on validation macro-F1
        results: Dictionary with detailed results for each LR
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if lr_range is None:
        # Hardcoded LR range
        lr_range = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

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

    # Verify K matches cached state discovery
    if hasattr(dataset, "K_by_tp") and tp in dataset.K_by_tp:
        K_cached = dataset.K_by_tp[tp]
        assert K == K_cached, (
            f"K mismatch: found {K} unique states in data but cache has {K_cached}"
        )
        method = dataset.state_method_by_tp.get(tp, "unknown")
        print(f"Using {K} states for TP={tp} (method: {method})")
    else:
        print(f"Number of unique states: {K}")

    # Compute class weights
    class_weights = compute_class_weights(tp_dataset, K, device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Split into train/val
    val_size = int(len(tp_dataset) * val_split)
    train_size = len(tp_dataset) - val_size
    g = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        tp_dataset, [train_size, val_size], generator=g
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    results = {}
    best_lr = None
    best_f1 = -1.0

    for lr in lr_range:
        print(f"\n{'=' * 60}")
        print(f"Testing LR = {lr:.2e}")
        print(f"{'=' * 60}")

        # Initialize wandb for this LR trial
        if wandb_project:
            run_name = f"{wandb_run_prefix}_lr{lr:.2e}_tp{tp}"
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=run_name,
                config={
                    "stage": "lr_sweep",
                    "lr": lr,
                    "tp": tp,
                    "hidden_size": hidden_size,
                    "bidirectional": bidirectional,
                    "num_epochs": num_epochs,
                    "seed": seed,
                },
                reinit=True,
            )

        torch.manual_seed(seed)
        classifier = GRUClassifier(
            Dx, K, H=hidden_size, bidirectional=bidirectional
        ).to(device)
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
            val_metrics = evaluate_classifier(
                classifier,
                val_loader,
                K,
                device,
                criterion,
                compute_extra_metrics=False,
            )
            val_losses.append(val_metrics["loss"])
            val_f1s.append(val_metrics["macro_f1"])

            # Log to wandb
            if wandb_project:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_losses[-1],
                        "val_loss": val_metrics["loss"],
                        "val_f1": val_metrics["macro_f1"],
                        "val_acc": val_metrics["accuracy"],
                        "val_balanced_acc": val_metrics["balanced_accuracy"],
                        "val_ece": val_metrics["ece"],
                        "lr": lr,
                    }
                )

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:03d}: train_loss={train_losses[-1]:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, val_f1={val_metrics['macro_f1']:.4f}"
                )

        final_f1 = max(val_f1s)  # Changed: use max F1, not last epoch
        results[lr] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_f1s": val_f1s,
            "final_f1": final_f1,
        }

        print(f"Best validation F1: {final_f1:.4f}")

        if wandb_project:
            wandb.finish()

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
        summary_data.append(
            {
                "learning_rate": lr,
                "final_val_f1": metrics["final_f1"],
                "best_val_f1": max(metrics["val_f1s"]),
                "final_train_loss": metrics["train_losses"][-1],
                "final_val_loss": metrics["val_losses"][-1],
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(
        os.path.join(output_dir, f"lr_sweep_summary_tp{tp}.csv"), index=False
    )

    # Per-epoch CSV for each LR
    for lr, metrics in results.items():
        epoch_data = []
        for epoch in range(len(metrics["train_losses"])):
            epoch_data.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": metrics["train_losses"][epoch],
                    "val_loss": metrics["val_losses"][epoch],
                    "val_f1": metrics["val_f1s"][epoch],
                }
            )

        epoch_df = pd.DataFrame(epoch_data)
        lr_str = f"{lr:.2e}".replace(".", "p").replace("-", "m")
        epoch_df.to_csv(
            os.path.join(output_dir, f"lr_sweep_tp{tp}_lr{lr_str}_epochs.csv"),
            index=False,
        )


def train_classifiers(
    dataset: PowerTraceDataset,
    tp: int,
    hidden_size: int = 64,
    num_epochs: int = 1000,
    lr: float = 5e-3,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    use_scheduler: bool = True,
    scheduler_type: str = "cosine",
    val_split: float = 0.15,
    seed: int = 42,
    output_dir: str = "./training_results",
    use_class_weights: bool = True,
    bidirectional: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: str = "grantfwilkins-stanford-university",
    wandb_run_name: Optional[str] = None,
    save_model: bool = False,
    compute_extra_metrics: bool = True,
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
        bidirectional: Whether to use bidirectional GRU
        wandb_project: WandB project name
        wandb_entity: WandB entity/team name
        wandb_run_name: WandB run name
        save_model: Whether to save the trained model weights
        compute_extra_metrics: Whether to compute transition MAE and autocorr R²

    Returns:
        classifier: Trained classifier
        metrics: Dictionary with training metrics and results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Initialize wandb
    if wandb_project:
        if wandb_run_name is None:
            wandb_run_name = f"tp{tp}_H{hidden_size}_lr{lr:.2e}_{'bi' if bidirectional else 'uni'}GRU"

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config={
                "tp": tp,
                "hidden_size": hidden_size,
                "num_epochs": num_epochs,
                "lr": lr,
                "scheduler": scheduler_type if use_scheduler else "none",
                "seed": seed,
                "bidirectional": bidirectional,
            },
            reinit=True,
        )

    print(f"Training classifier for TP={tp}")
    tp_indices = [i for i, tp_i in enumerate(dataset.tp_all) if tp_i == tp]

    tp_dataset = TPDataset(dataset, tp_indices)
    x_sample, y_sample, z_sample = dataset[tp_indices[0]]
    Dx = x_sample.shape[1]
    all_z = torch.cat([dataset[i][2] for i in tp_indices])
    K = len(torch.unique(all_z))

    # Verify K matches cached state discovery
    if hasattr(dataset, "K_by_tp") and tp in dataset.K_by_tp:
        K_cached = dataset.K_by_tp[tp]
        assert K == K_cached, (
            f"K mismatch: found {K} unique states in data but cache has {K_cached}"
        )
        method = dataset.state_method_by_tp.get(tp, "unknown")
        print(f"Using {K} states for TP={tp} (method: {method})")
    else:
        print(f"Number of unique states across all TP={tp} samples: {K}")

    # Get state power means for this TP (for power-domain metrics)
    state_power_means = dataset.mu.get(tp, None) if hasattr(dataset, "mu") else None
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(tp_dataset, K, device)
        print(f"Class weights: {class_weights.cpu().numpy()}")
    val_size = int(len(tp_dataset) * val_split)
    train_size = len(tp_dataset) - val_size
    g = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        tp_dataset, [train_size, val_size], generator=g
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train size: {train_size}, Val size: {val_size}")

    classifier = GRUClassifier(Dx, K, H=hidden_size, bidirectional=bidirectional).to(
        device
    )
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-2)

    # Use label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # Setup scheduler
    scheduler = None
    scheduler_step_mode = "batch"  # "batch" or "epoch"
    if use_scheduler:
        if scheduler_type == "cosine":
            total_steps = num_epochs * len(train_loader)
            scheduler = build_cosine_with_warmup(
                optimizer, total_steps, warmup_pct=0.05
            )
            scheduler_step_mode = "batch"
        elif scheduler_type == "onecycle":
            total_steps = num_epochs * len(train_loader)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.05
            )
            scheduler_step_mode = "batch"
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3, cooldown=1, verbose=True
            )
            scheduler_step_mode = "epoch"

    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []
    val_transition_maes = []
    val_autocorr_r2s = []
    learning_rates = []
    best_val_f1 = 0.0

    for epoch in range(num_epochs):
        classifier.train()
        epoch_loss = 0.0
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for x, y, z in progress_bar:
            x = x.to(device)
            z = z.to(device)

            # Log pre-step LR
            cur_lr = optimizer.param_groups[0]["lr"]
            learning_rates.append(cur_lr)

            optimizer.zero_grad()
            logits = classifier(x)
            loss = criterion(logits.view(-1, K), z.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()

            # Step scheduler per batch if needed
            if scheduler is not None and scheduler_step_mode == "batch":
                scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_losses.append(epoch_loss / len(train_loader))

        # Validation
        val_metrics = evaluate_classifier(
            classifier,
            val_loader,
            K,
            device,
            criterion,
            compute_extra_metrics=compute_extra_metrics,
            state_power_means=state_power_means,
        )

        val_losses.append(val_metrics["loss"])
        val_accs.append(val_metrics["accuracy"])
        val_f1s.append(val_metrics["macro_f1"])
        if "transition_mae" in val_metrics:
            val_transition_maes.append(val_metrics["transition_mae"])
        if "autocorr_r2" in val_metrics:
            val_autocorr_r2s.append(val_metrics["autocorr_r2"])

        # Track best F1 for model saving
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]

        # Step scheduler per epoch if needed (e.g., ReduceLROnPlateau)
        if scheduler is not None and scheduler_step_mode == "epoch":
            scheduler.step(val_metrics["loss"])

        # Log to wandb
        if wandb_project:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_losses[-1],
                "val_loss": val_metrics["loss"],
                "val_f1": val_metrics["macro_f1"],
                "val_acc": val_metrics["accuracy"],
                "val_balanced_acc": val_metrics["balanced_accuracy"],
                "val_ece": val_metrics["ece"],
            }
            # Add extended metrics if available
            if "transition_mae" in val_metrics:
                log_dict["val_transition_mae"] = val_metrics["transition_mae"]
            if "autocorr_r2" in val_metrics:
                log_dict["val_autocorr_r2"] = val_metrics["autocorr_r2"]
            if "tolerance_power_acc" in val_metrics:
                log_dict["val_tolerance_power_acc"] = val_metrics["tolerance_power_acc"]
            if "boundary_f1" in val_metrics:
                log_dict["val_boundary_f1"] = val_metrics["boundary_f1"]
            if scheduler is not None:
                log_dict["lr"] = optimizer.param_groups[0]["lr"]
            wandb.log(log_dict)

        if (epoch + 1) % 10 == 0:
            extra_str = ""
            if "transition_mae" in val_metrics:
                extra_str += f", Trans MAE: {val_metrics['transition_mae']:.4f}"
            if "autocorr_r2" in val_metrics:
                extra_str += f", AutoCorr R²: {val_metrics['autocorr_r2']:.4f}"
            if "tolerance_power_acc" in val_metrics:
                extra_str += f", Tol.Acc(25W): {val_metrics['tolerance_power_acc']:.3f}"
            if "boundary_f1" in val_metrics:
                extra_str += f", Boundary F1: {val_metrics['boundary_f1']:.3f}"
            print(
                f"TP {tp}, Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['macro_f1']:.4f}{extra_str}"
            )

    final_metrics = evaluate_classifier(
        classifier,
        val_loader,
        K,
        device,
        criterion,
        compute_extra_metrics=compute_extra_metrics,
        state_power_means=state_power_means,
    )

    print(f"\nFinal Results for TP={tp}:")
    print(f"  Val Loss: {final_metrics['loss']:.4f}")
    print(f"  Val Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Val Balanced Accuracy: {final_metrics['balanced_accuracy']:.4f}")
    print(f"  Val Macro-F1: {final_metrics['macro_f1']:.4f}")
    print(f"  Val ECE: {final_metrics['ece']:.4f}")
    if "transition_mae" in final_metrics:
        print(f"  Transition MAE: {final_metrics['transition_mae']:.4f}")
    if "autocorr_r2" in final_metrics:
        print(f"  AutoCorr R²: {final_metrics['autocorr_r2']:.4f}")
    if "tolerance_power_acc" in final_metrics:
        print(f"  Tolerance Power Acc (25W): {final_metrics['tolerance_power_acc']:.3f}")
    if "boundary_f1" in final_metrics:
        print(f"  Boundary F1: {final_metrics['boundary_f1']:.3f}")

    # Save training results
    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "val_f1s": val_f1s,
        "val_transition_maes": val_transition_maes if compute_extra_metrics else None,
        "val_autocorr_r2s": val_autocorr_r2s if compute_extra_metrics else None,
        "final_val_loss": final_metrics["loss"],
        "final_val_acc": final_metrics["accuracy"],
        "final_val_balanced_acc": final_metrics["balanced_accuracy"],
        "final_val_f1": final_metrics["macro_f1"],
        "final_val_ece": final_metrics["ece"],
        "final_transition_mae": final_metrics.get("transition_mae"),
        "final_autocorr_r2": final_metrics.get("autocorr_r2"),
        "final_tolerance_power_acc": final_metrics.get("tolerance_power_acc"),
        "final_boundary_f1": final_metrics.get("boundary_f1"),
        "confusion_matrix": final_metrics["confusion_matrix"],
        "learning_rates": learning_rates if scheduler else None,
    }

    # Save per-epoch metrics to CSV
    save_training_metrics_to_csv(
        metrics, output_dir, tp, lr, seed, hidden_size, bidirectional
    )

    # Save model weights if requested
    if save_model:
        model_path = os.path.join(
            output_dir,
            f"model_tp{tp}_H{hidden_size}_{'bi' if bidirectional else 'uni'}GRU.pt",
        )
        torch.save(classifier.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Finish wandb run
    if wandb_project:
        wandb.finish()

    return classifier, metrics


def save_training_metrics_to_csv(
    metrics: Dict,
    output_dir: str,
    tp: int,
    lr: float,
    seed: int,
    hidden_size: int = 64,
    bidirectional: bool = False,
):
    """Save training metrics to CSV file."""
    epoch_data = []
    for epoch in range(len(metrics["train_losses"])):
        row = {
            "epoch": epoch + 1,
            "train_loss": metrics["train_losses"][epoch],
            "val_loss": metrics["val_losses"][epoch],
            "val_acc": metrics["val_accs"][epoch],
            "val_f1": metrics["val_f1s"][epoch],
        }

        # Add extra metrics if available
        if (
            metrics.get("val_transition_maes")
            and len(metrics["val_transition_maes"]) > epoch
        ):
            row["val_transition_mae"] = metrics["val_transition_maes"][epoch]
        if metrics.get("val_autocorr_r2s") and len(metrics["val_autocorr_r2s"]) > epoch:
            row["val_autocorr_r2"] = metrics["val_autocorr_r2s"][epoch]

        # Add learning rate if available
        if metrics["learning_rates"] and len(metrics["learning_rates"]) > 0:
            # Average LR for the epoch (multiple steps per epoch)
            epoch_start = epoch * (
                len(metrics["learning_rates"]) // len(metrics["train_losses"])
            )
            epoch_end = (epoch + 1) * (
                len(metrics["learning_rates"]) // len(metrics["train_losses"])
            )
            row["avg_lr"] = np.mean(metrics["learning_rates"][epoch_start:epoch_end])

        epoch_data.append(row)

    epoch_df = pd.DataFrame(epoch_data)
    lr_str = f"{lr:.2e}".replace(".", "p").replace("-", "m")
    arch_str = "bi" if bidirectional else "uni"
    epoch_df.to_csv(
        os.path.join(
            output_dir,
            f"training_tp{tp}_H{hidden_size}_{arch_str}_lr{lr_str}_seed{seed}_epochs.csv",
        ),
        index=False,
    )

    # Save summary metrics
    summary_data = {
        "tp": tp,
        "hidden_size": hidden_size,
        "bidirectional": bidirectional,
        "learning_rate": lr,
        "seed": seed,
        "final_train_loss": metrics["train_losses"][-1],
        "final_val_loss": metrics["final_val_loss"],
        "final_val_acc": metrics["final_val_acc"],
        "final_val_balanced_acc": metrics["final_val_balanced_acc"],
        "final_val_f1": metrics["final_val_f1"],
        "final_val_ece": metrics["final_val_ece"],
        "best_val_f1": max(metrics["val_f1s"]),
        "num_epochs": len(metrics["train_losses"]),
    }

    if metrics.get("final_transition_mae") is not None:
        summary_data["final_transition_mae"] = metrics["final_transition_mae"]
    if metrics.get("final_autocorr_r2") is not None:
        summary_data["final_autocorr_r2"] = metrics["final_autocorr_r2"]
    if metrics.get("final_tolerance_power_acc") is not None:
        summary_data["final_tolerance_power_acc"] = metrics["final_tolerance_power_acc"]
    if metrics.get("final_boundary_f1") is not None:
        summary_data["final_boundary_f1"] = metrics["final_boundary_f1"]

    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(
        os.path.join(
            output_dir,
            f"training_tp{tp}_H{hidden_size}_{arch_str}_lr{lr_str}_seed{seed}_summary.csv",
        ),
        index=False,
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
