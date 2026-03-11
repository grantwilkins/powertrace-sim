#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import evaluate_trained_model
from prepare import DEFAULT_CONFIG_ID, DEFAULT_OUT_DIR, build_training_inputs

from model.classifiers.gru import GRUClassifier

DEFAULT_PREPARED_DIR = DEFAULT_OUT_DIR


def _resolve_device(device: Optional[torch.device | str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def _parse_k_candidates(csv_text: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for tok in str(csv_text).split(","):
        tok = tok.strip()
        if tok == "":
            continue
        val = int(tok)
        if val < 1 or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def _train_model(
    *,
    model: GRUClassifier,
    train_data: Sequence[Dict[str, object]],
    val_data: Sequence[Dict[str, object]],
    k: int,
    input_dim: int,
    epochs: int,
    lr: float,
    patience: int,
    scheduler_patience: int,
    scheduler_factor: float,
    device: torch.device,
) -> Dict[str, object]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=int(scheduler_patience),
        factor=float(scheduler_factor),
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    patience_counter = 0
    history: List[Dict[str, float]] = []

    for epoch in range(int(max(1, epochs))):
        model.train()
        train_losses: List[float] = []
        for trace in train_data:
            x = torch.from_numpy(
                np.asarray(trace["features_norm"], dtype=np.float32)
            ).to(device)
            y = torch.from_numpy(np.asarray(trace["state_labels"], dtype=np.int64)).to(
                device
            )
            if x.ndim != 2 or y.ndim != 1 or x.shape[1] != int(input_dim):
                continue
            if x.shape[0] == 0 or y.shape[0] != x.shape[0]:
                continue

            logits = model(x.unsqueeze(0))
            loss = criterion(logits.reshape(-1, int(k)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for trace in val_data:
                x = torch.from_numpy(
                    np.asarray(trace["features_norm"], dtype=np.float32)
                ).to(device)
                y = torch.from_numpy(
                    np.asarray(trace["state_labels"], dtype=np.int64)
                ).to(device)
                if x.ndim != 2 or y.ndim != 1 or x.shape[1] != int(input_dim):
                    continue
                if x.shape[0] == 0 or y.shape[0] != x.shape[0]:
                    continue
                logits = model(x.unsqueeze(0))
                loss = criterion(logits.reshape(-1, int(k)), y.reshape(-1))
                val_losses.append(float(loss.item()))

        mean_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        sched_metric = mean_val_loss if np.isfinite(mean_val_loss) else mean_train_loss
        if np.isfinite(sched_metric):
            scheduler.step(float(sched_metric))

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(mean_train_loss),
                "val_loss": float(mean_val_loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if epoch % 25 == 0 or epoch == int(max(1, epochs)) - 1:
            print(
                f"epoch={epoch:4d} train_loss={mean_train_loss:.5f} "
                f"val_loss={mean_val_loss:.5f} lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if np.isfinite(mean_val_loss) and mean_val_loss < best_val_loss:
            best_val_loss = float(mean_val_loss)
            best_epoch = int(epoch)
            patience_counter = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            patience_counter += 1
            if patience_counter >= int(max(1, patience)):
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train_loss = float(history[-1]["train_loss"]) if history else float("nan")
    final_val_loss = float(history[-1]["val_loss"]) if history else float("nan")
    train_loss_at_best_epoch = float("nan")
    if history and best_epoch >= 0:
        train_loss_at_best_epoch = float(history[best_epoch]["train_loss"])

    return {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
        "train_loss_at_best_epoch": float(train_loss_at_best_epoch),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the isolated gpt-oss-20b_A100_tp1 sandbox model and print heldout metrics."
    )
    parser.add_argument("--prepared-dir", default=str(DEFAULT_PREPARED_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-set", choices=["f2", "f3"], default="f2")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Fixed K override. If omitted, K is selected by BIC over --bic-candidates.",
    )
    parser.add_argument("--bic-candidates", default="6,8,10,12,14,16")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--scheduler-patience", type=int, default=20)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    prepared_dir = Path(args.prepared_dir).resolve()
    feature_set = str(args.feature_set).strip().lower()
    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    training_inputs = build_training_inputs(
        prepared_dir=prepared_dir,
        config_id=DEFAULT_CONFIG_ID,
        feature_set=feature_set,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(max(1, args.num_layers)),
        seed=seed,
        fixed_k=args.k,
        bic_candidates=_parse_k_candidates(args.bic_candidates),
    )
    config_id = str(training_inputs["config_id"])
    input_dim = int(training_inputs["input_dim"])
    selected_k = int(training_inputs["selected_k"])
    k_selection_reason = str(training_inputs["k_selection_reason"])

    device = _resolve_device(args.device)
    model = GRUClassifier(
        Dx=int(input_dim),
        K=int(selected_k),
        H=int(args.hidden_dim),
        num_layers=int(max(1, args.num_layers)),
    ).to(device)

    t0 = time.perf_counter()
    train_result = _train_model(
        model=model,
        train_data=training_inputs["train_data"],
        val_data=training_inputs["val_data"],
        k=int(selected_k),
        input_dim=int(input_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        patience=int(args.patience),
        scheduler_patience=int(args.scheduler_patience),
        scheduler_factor=float(args.scheduler_factor),
        device=device,
    )
    training_time_sec = float(time.perf_counter() - t0)

    heldout_metrics = evaluate_trained_model(
        model=model,
        gmm_params=training_inputs["gmm_fit"],
        norm_payload=training_inputs["norm_payload"],
        prepared_dir=prepared_dir,
        feature_set=feature_set,
        input_dim=int(input_dim),
        device=device,
    )

    architecture = (
        f"GRUClassifier(Dx={input_dim}, K={selected_k}, H={int(args.hidden_dim)}, "
        f"num_layers={int(max(1, args.num_layers))}, bidirectional=True)"
    )

    print(f"config_id: {config_id}")
    print(f"feature_set: {feature_set}")
    print(f"k: {selected_k}")
    print(f"k_selection_reason: {k_selection_reason}")
    print(f"hidden_dim: {int(args.hidden_dim)}")
    print(f"num_layers: {int(max(1, args.num_layers))}")
    print(f"architecture: {architecture}")
    print(f"training_time_sec: {training_time_sec:.4f}")
    print(f"best_val_loss: {train_result['best_val_loss']:.6f}")
    print(f"final_train_loss: {train_result['final_train_loss']:.6f}")
    print(f"final_val_loss: {train_result['final_val_loss']:.6f}")
    print(f"heldout_acf_r2: {heldout_metrics['acf_r2']:.6f}")
    print(f"heldout_ks_stat: {heldout_metrics['ks_stat']:.6f}")
    print(f"heldout_delta_energy_pct: {heldout_metrics['delta_energy_pct']:.6f}")


if __name__ == "__main__":
    main()
