import argparse
import os

import numpy as np
import torch

from model.classifiers.train import train_classifiers
from model.core.dataset import PowerTraceDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRU model on power traces")
    parser.add_argument(
        "--data_file", type=str, required=True, help="Path to the NPZ data file"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "llama-3-8b",
            "llama-3-70b",
            "deepseek-r1-distill-8b",
            "deepseek-r1-distill-70b",
            "gpt-oss-20b",
            "gpt-oss-120b",
        ],
        help="LLM name",
    )
    parser.add_argument(
        "--tp",
        type=int,
        required=True,
        help="Tensor parallelism value to train. Use -1 to train all TP values in the dataset.",
    )
    parser.add_argument("--hardware_accelerator", type=str, required=True)
    parser.add_argument(
        "--weights_path",
        type=str,
        default="./gru_classifier_weights/",
        help="Path to the classifier weights file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cuda, cuda:0, cpu, etc.). Defaults to cuda if available.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for training (default: 1e-3)",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Hidden size for GRU (default: 64)",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use bidirectional GRU",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
        help="Number of training epochs (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project name (e.g., model_hardware). If not specified, wandb is disabled.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="WandB run name (auto-generated if not specified)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./training_results",
        help="Directory to save training logs and plots",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save model weights after training",
    )

    args = parser.parse_args()

    dataset = PowerTraceDataset(
        args.data_file,
        K=6,
    )
    print(f"Loaded dataset with {len(dataset)} traces")
    print(f"Sample trace shape: {dataset.traces[0]['z'].shape}")
    os.makedirs("./training_data/losses", exist_ok=True)
    os.makedirs(args.weights_path, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device) if args.device else None

    if args.tp == -1:
        tps_to_train = sorted(set(dataset.tp_all))
        print(f"Training all TP values: {tps_to_train}")
    else:
        tps_to_train = [args.tp]
        print(f"Training TP={args.tp}")

    data_basename = os.path.basename(args.data_file)
    if data_basename.startswith("random_") and data_basename.endswith(".npz"):
        parts = data_basename[7:-4].rsplit("_", 1)
        if len(parts) == 2:
            inferred_model, inferred_hardware = parts
            print(
                f"Inferred from filename: model={inferred_model}, hardware={inferred_hardware}"
            )
        else:
            inferred_model = args.model
            inferred_hardware = args.hardware_accelerator
    else:
        inferred_model = args.model
        inferred_hardware = args.hardware_accelerator

    for tp in tps_to_train:
        print(f"\n{'=' * 70}")
        print(f"Processing TP={tp}")
        print(f"{'=' * 70}\n")
        wandb_project = args.wandb_project

        print(
            f"\nTraining with LR={args.lr:.2e}, H={args.hidden_size}, {'Bi' if args.bidirectional else 'Uni'}GRU..."
        )
        classifier, metrics = train_classifiers(
            dataset=dataset,
            tp=tp,
            hidden_size=args.hidden_size,
            lr=args.lr,
            num_epochs=args.num_epochs,
            device=device,
            output_dir=args.output_dir,
            seed=args.seed,
            bidirectional=args.bidirectional,
            wandb_project=wandb_project,
            wandb_run_name=args.wandb_run_name,
            save_model=args.save_model,
        )
        if args.save_model:
            classifier.to("cpu")
            torch.save(
                classifier.state_dict(),
                f"{args.weights_path}/{args.model}_{args.hardware_accelerator}_tp{tp}.pt",
            )
            print(
                f"\nSaved weights to: {args.weights_path}/{args.model}_{args.hardware_accelerator}_tp{tp}.pt"
            )

    print(f"\n{'=' * 70}")
    print("Training complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Model weights saved to: {args.weights_path}")
    print(f"{'=' * 70}")
