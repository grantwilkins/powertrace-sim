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
        "--lr_sweep",
        action="store_true",
        help="Run LR sweep before training to find optimal learning rate",
    )
    parser.add_argument(
        "--lr_sweep_epochs",
        type=int,
        default=15,
        help="Number of epochs for each LR in sweep (default: 15)",
    )
    parser.add_argument(
        "--use_best_lr",
        action="store_true",
        help="Use the best LR from a previous sweep (must have run --lr_sweep first)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for training (default: 1e-3)",
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
        "--multi_seed",
        type=int,
        default=None,
        help="Run training with N seeds and report mean±std (e.g., --multi_seed 3)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./training_results",
        help="Directory to save training logs and plots",
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

    # Main training loop
    for tp in tps_to_train:
        print(f"\n{'='*70}")
        print(f"Processing TP={tp}")
        print(f"{'='*70}\n")

        current_lr = args.lr

        # Step 1: Optional LR sweep
        if args.lr_sweep:
            print("Running LR sweep...")
            best_lr, sweep_results = lr_sweep(
                dataset=dataset,
                tp=tp,
                num_epochs=args.lr_sweep_epochs,
                device=device,
                output_dir=os.path.join(args.output_dir, f"lr_sweep_tp{tp}"),
            )
            print(f"\nBest LR from sweep: {best_lr:.2e}")
            current_lr = best_lr

        # Step 2: Multi-seed training or single training
        if args.multi_seed is not None:
            print(f"\nRunning multi-seed training with {args.multi_seed} seeds...")
            results = multi_seed_training(
                dataset=dataset,
                tp=tp,
                lr=current_lr,
                num_seeds=args.multi_seed,
                num_epochs=args.num_epochs,
                device=device,
                use_scheduler=(args.scheduler != "none"),
                scheduler_type=args.scheduler if args.scheduler != "none" else "cosine",
                output_dir=os.path.join(args.output_dir, f"tp{tp}"),
            )

            # Save the best classifier from multi-seed runs
            best_idx = np.argmax(results['all_f1s'])
            classifier = results['classifiers'][best_idx]
            print(f"\nUsing classifier from seed {best_idx} (F1={results['all_f1s'][best_idx]:.4f})")

        else:
            print(f"\nTraining with LR={current_lr:.2e}...")
            classifier, metrics = train_classifiers(
                dataset=dataset,
                tp=tp,
                lr=current_lr,
                num_epochs=args.num_epochs,
                device=device,
                use_scheduler=(args.scheduler != "none"),
                scheduler_type=args.scheduler if args.scheduler != "none" else "cosine",
                output_dir=os.path.join(args.output_dir, f"tp{tp}"),
            )

            # Save losses (backward compatibility)
            os.makedirs("./training_data/losses", exist_ok=True)
            np.save(
                f"./training_data/losses/training_losses_{args.model}_{args.hardware_accelerator}_tp{tp}.npy",
                np.array(metrics['train_losses']),
            )

        # Step 3: Save model weights
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
