import argparse
import os

import numpy as np
import torch
from classifiers.train import train_classifiers, lr_sweep, multi_seed_training
from core.dataset import PowerTraceDataset

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

    # New LR sweep and training options
    parser.add_argument(
        "--stage",
        type=str,
        default="train",
        choices=["lr_sweep", "hidden_size", "directionality", "train"],
        help="Training stage (default: train)",
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
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "onecycle", "none"],
        help="LR scheduler type (default: cosine)",
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

    dataset = PowerTraceDataset(args.data_file)
    print(f"Loaded dataset with {len(dataset)} traces")
    print(f"Sample trace shape: {dataset.traces[0]['z'].shape}")

    # Setup directories
    os.makedirs("./training_data/losses", exist_ok=True)
    os.makedirs(args.weights_path, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device) if args.device else None

    # Determine which TPs to train
    if args.tp == -1:
        tps_to_train = sorted(set(dataset.tp_all))
        print(f"Training all TP values: {tps_to_train}")
    else:
        tps_to_train = [args.tp]
        print(f"Training TP={args.tp}")

    # Parse model and hardware from data_file if not using defaults
    # Expected format: random_{model}_{hardware}.npz
    data_basename = os.path.basename(args.data_file)
    if data_basename.startswith("random_") and data_basename.endswith(".npz"):
        parts = data_basename[7:-4].rsplit("_", 1)  # Remove "random_" prefix and ".npz" suffix
        if len(parts) == 2:
            inferred_model, inferred_hardware = parts
            print(f"Inferred from filename: model={inferred_model}, hardware={inferred_hardware}")
        else:
            inferred_model = args.model
            inferred_hardware = args.hardware_accelerator
    else:
        inferred_model = args.model
        inferred_hardware = args.hardware_accelerator

    # Main training loop
    for tp in tps_to_train:
        print(f"\n{'='*70}")
        print(f"Processing TP={tp}")
        print(f"{'='*70}\n")

        # Auto-generate wandb project name if not specified
        wandb_project = args.wandb_project
        if wandb_project is None and args.stage != "train":
            # For ablation stages, automatically create project name
            wandb_project = f"{inferred_model}_{inferred_hardware}"

        if args.stage == "lr_sweep":
            print("Running LR sweep (100 epochs per LR)...")
            best_lr, sweep_results = lr_sweep(
                dataset=dataset,
                tp=tp,
                num_epochs=100,
                hidden_size=args.hidden_size,
                device=device,
                output_dir=args.output_dir,
                seed=args.seed,
                wandb_project=wandb_project,
                wandb_run_prefix=f"stage1_lr_sweep",
                bidirectional=args.bidirectional,
            )
            print(f"\nBest LR from sweep: {best_lr:.2e}")

            # Save best LR to file
            with open(os.path.join(args.output_dir, "best_lr.txt"), "w") as f:
                f.write(f"{best_lr:.2e}\n")
            print(f"Best LR saved to {args.output_dir}/best_lr.txt")

        elif args.stage == "hidden_size":
            print(f"Running hidden size ablation (H ∈ {{16, 32, 64, 128, 256}})...")
            hidden_sizes = [64, 128, 256]
            best_f1 = 0.0
            best_H = 64

            for H in hidden_sizes:
                print(f"\n--- Training with H={H} ---")
                run_name = f"stage2_H{H}_tp{tp}" if args.wandb_run_name is None else args.wandb_run_name
                classifier, metrics = train_classifiers(
                    dataset=dataset,
                    tp=tp,
                    hidden_size=H,
                    lr=args.lr,
                    num_epochs=args.num_epochs,
                    device=device,
                    use_scheduler=(args.scheduler != "none"),
                    scheduler_type=args.scheduler if args.scheduler != "none" else "cosine",
                    output_dir=args.output_dir,
                    seed=args.seed,
                    bidirectional=args.bidirectional,
                    wandb_project=wandb_project,
                    wandb_run_name=run_name,
                    save_model=False,  # We'll save the best one at the end
                )

                if metrics['final_val_f1'] > best_f1:
                    best_f1 = metrics['final_val_f1']
                    best_H = H
                    # Save this model as it's the best so far
                    model_path = os.path.join(args.output_dir, f"model_tp{tp}_H{H}_{'bi' if args.bidirectional else 'uni'}GRU_best.pt")
                    torch.save(classifier.state_dict(), model_path)
                    print(f"New best H={H} with F1={best_f1:.4f}, saved to {model_path}")

            print(f"\nBest hidden size: H={best_H} (F1={best_f1:.4f})")
            with open(os.path.join(args.output_dir, "best_hidden_size.txt"), "w") as f:
                f.write(f"{best_H}\n")

        elif args.stage == "directionality":
            print(f"Running directionality ablation (UniGRU vs BiGRU)...")

            # UniGRU
            print("\n--- Training UniGRU ---")
            run_name_uni = f"stage3_uniGRU_tp{tp}" if args.wandb_run_name is None else f"{args.wandb_run_name}_uni"
            classifier_uni, metrics_uni = train_classifiers(
                dataset=dataset,
                tp=tp,
                hidden_size=args.hidden_size,
                lr=args.lr,
                num_epochs=args.num_epochs,
                device=device,
                use_scheduler=(args.scheduler != "none"),
                scheduler_type=args.scheduler if args.scheduler != "none" else "cosine",
                output_dir=args.output_dir,
                seed=args.seed,
                bidirectional=False,
                wandb_project=wandb_project,
                wandb_run_name=run_name_uni,
                save_model=False,
            )

            # BiGRU
            print("\n--- Training BiGRU ---")
            run_name_bi = f"stage3_biGRU_tp{tp}" if args.wandb_run_name is None else f"{args.wandb_run_name}_bi"
            classifier_bi, metrics_bi = train_classifiers(
                dataset=dataset,
                tp=tp,
                hidden_size=args.hidden_size,
                lr=args.lr,
                num_epochs=args.num_epochs,
                device=device,
                use_scheduler=(args.scheduler != "none"),
                scheduler_type=args.scheduler if args.scheduler != "none" else "cosine",
                output_dir=args.output_dir,
                seed=args.seed,
                bidirectional=True,
                wandb_project=wandb_project,
                wandb_run_name=run_name_bi,
                save_model=False,
            )

            print(f"\nDirectionality Results:")
            print(f"  UniGRU: F1={metrics_uni['final_val_f1']:.4f}")
            print(f"  BiGRU:  F1={metrics_bi['final_val_f1']:.4f}")

        else:  # args.stage == "train"
            print(f"\nTraining with LR={args.lr:.2e}, H={args.hidden_size}, {'Bi' if args.bidirectional else 'Uni'}GRU...")
            classifier, metrics = train_classifiers(
                dataset=dataset,
                tp=tp,
                hidden_size=args.hidden_size,
                lr=args.lr,
                num_epochs=args.num_epochs,
                device=device,
                use_scheduler=(args.scheduler != "none"),
                scheduler_type=args.scheduler if args.scheduler != "none" else "cosine",
                output_dir=args.output_dir,
                seed=args.seed,
                bidirectional=args.bidirectional,
                wandb_project=wandb_project,
                wandb_run_name=args.wandb_run_name,
                save_model=args.save_model,
            )

            # Save model weights (backward compatibility)
            if args.save_model:
                classifier.to("cpu")
                torch.save(
                    classifier.state_dict(),
                    f"{args.weights_path}/{args.model}_{args.hardware_accelerator}_tp{tp}.pt",
                )
                print(f"\nSaved weights to: {args.weights_path}/{args.model}_{args.hardware_accelerator}_tp{tp}.pt")

    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Model weights saved to: {args.weights_path}")
    print(f"{'='*70}")
